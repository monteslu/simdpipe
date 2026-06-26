/*
 * simdpipe — a portable SIMD software rasterizer (WASM + 128-bit SIMD).
 *
 * Lineage: the Mesa "-pipe" software-renderer family (softpipe / llvmpipe /
 * lavapipe). simdpipe is a new member of that family aimed at one thing:
 * extreme speed in a single portable WASM module, by doing LESS work
 * (configurable fidelity) rather than going native or going wider than 128-bit.
 *
 * Phase 0: fixed-function pipeline only (no shaders yet) — enough to prove the
 * thesis with benchmarks. Edge-function (Pineda) rasterization, SoA SIMD over a
 * 2x2-style 4-pixel group per v128, z-buffer, barycentric attribute interp,
 * nearest/bilinear texturing, tiled traversal. The shader executor (Tiers 0/1/2)
 * comes later.
 *
 * Memory model: one linear memory. The host gets raw pointers (byte offsets) to
 * the color framebuffer (RGBA8) and reads them directly — zero copy.
 *
 * Coordinate system: pixel centers at (x+0.5, y+0.5). Top-left origin.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <wasm_simd128.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

/* ---- limits ---- */
#define SP_MAX_W 4096
#define SP_MAX_H 4096

/* ---- fidelity flags (the "do less" knobs) ---- */
enum {
  SP_FLAG_DEPTH_TEST   = 1 << 0,  /* z-buffer test+write */
  SP_FLAG_BILINEAR     = 1 << 1,  /* else nearest */
  SP_FLAG_PERSP_CORRECT= 1 << 2,  /* else affine (PS1-style, faster) */
  SP_FLAG_BLEND        = 1 << 3,  /* src-alpha over dst (reads framebuffer) */
  SP_FLAG_TEXTURE      = 1 << 4,  /* sample bound texture, else vertex color */
};

/* ---- renderer state ---- */
typedef struct {
  int      w, h;
  uint32_t *color;     /* RGBA8, w*h */
  float    *depth;     /* w*h, 1.0 = far */
  uint32_t flags;

  /* bound texture (RGBA8) */
  const uint32_t *tex;
  int      tex_w, tex_h;
  float    tex_wf, tex_hf;

  /* stats */
  uint64_t frag_tested;
  uint64_t frag_shaded;
  uint64_t tris;
} sp_ctx;

/* a vertex after transform: screen-space x,y in pixels, z in [0,1], inv-w for
 * perspective, attributes r,g,b,a in [0,1], texcoords u,v. */
typedef struct {
  float x, y, z, invw;
  float r, g, b, a;
  float u, v;
} sp_vert;

static sp_ctx g_ctx;

/* static backing stores so the host doesn't have to manage allocation in Phase 0.
 * (We'll switch to malloc when sizes get dynamic.) */
static uint32_t g_color[SP_MAX_W * 256];   /* placeholder; real alloc below */
/* NOTE: a full 4096x4096 framebuffer is 64MB; we allocate on init instead. */

#include <stdlib.h>

EXPORT void *sp_alloc(int bytes) { return malloc((size_t)bytes); }
EXPORT void  sp_free(void *p) { free(p); }

EXPORT int sp_init(int w, int h) {
  if (w <= 0 || h <= 0 || w > SP_MAX_W || h > SP_MAX_H) return 0;
  memset(&g_ctx, 0, sizeof(g_ctx));
  g_ctx.w = w; g_ctx.h = h;
  /* over-allocate by 4 px so a tail SIMD store (writes 4 lanes) never runs past
   * the buffer when w is not a multiple of 4. Masked stores keep spill harmless. */
  size_t npx = (size_t)w * h + 4;
  g_ctx.color = (uint32_t *)malloc(npx * 4);
  g_ctx.depth = (float *)malloc(npx * sizeof(float));
  if (!g_ctx.color || !g_ctx.depth) return 0;
  g_ctx.flags = SP_FLAG_DEPTH_TEST | SP_FLAG_TEXTURE | SP_FLAG_BILINEAR | SP_FLAG_PERSP_CORRECT;
  return 1;
}

EXPORT uint32_t *sp_color_ptr(void) { return g_ctx.color; }
EXPORT float    *sp_depth_ptr(void) { return g_ctx.depth; }
EXPORT int       sp_width(void)  { return g_ctx.w; }
EXPORT int       sp_height(void) { return g_ctx.h; }

EXPORT void sp_set_flags(uint32_t flags) { g_ctx.flags = flags; }
EXPORT uint32_t sp_get_flags(void) { return g_ctx.flags; }

EXPORT void sp_bind_texture(const uint32_t *px, int tw, int th) {
  g_ctx.tex = px; g_ctx.tex_w = tw; g_ctx.tex_h = th;
  g_ctx.tex_wf = (float)tw; g_ctx.tex_hf = (float)th;
}

EXPORT void sp_reset_stats(void) { g_ctx.frag_tested = g_ctx.frag_shaded = g_ctx.tris = 0; }
EXPORT double sp_stat_frag_tested(void) { return (double)g_ctx.frag_tested; }
EXPORT double sp_stat_frag_shaded(void) { return (double)g_ctx.frag_shaded; }
EXPORT double sp_stat_tris(void) { return (double)g_ctx.tris; }

EXPORT void sp_clear(uint32_t rgba, float depth) {
  int n = g_ctx.w * g_ctx.h;
  uint32_t *c = g_ctx.color;
  float *d = g_ctx.depth;
  /* SIMD clear color: 4 pixels per store */
  v128_t cv = wasm_u32x4_splat(rgba);
  v128_t dv = wasm_f32x4_splat(depth);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    wasm_v128_store(c + i, cv);
    if (g_ctx.flags & SP_FLAG_DEPTH_TEST) wasm_v128_store(d + i, dv);
  }
  for (; i < n; i++) { c[i] = rgba; if (g_ctx.flags & SP_FLAG_DEPTH_TEST) d[i] = depth; }
}

/* ---- helpers ---- */
static inline float sp_min3f(float a, float b, float c){ float m=a<b?a:b; return m<c?m:c; }
static inline float sp_max3f(float a, float b, float c){ float m=a>b?a:b; return m>c?m:c; }

/* edge function coefficients: E(x,y) = A*x + B*y + C, evaluated at pixel centers.
 * For edge from v0 to v1, positive on the right side (CCW front-facing -> all
 * three positive inside, given our winding). */

/* sample texture nearest, 4 lanes. u,v are texel coords (already * tex_w/h). */
static inline v128_t sp_tex_nearest4(v128_t u, v128_t v, v128_t mask) {
  /* returns packed RGBA8 in 4 lanes (or 0 where mask is 0). scalar gather —
   * WASM has no gather, so extract lanes. */
  const uint32_t *tex = g_ctx.tex;
  int tw = g_ctx.tex_w, th = g_ctx.tex_h;
  uint32_t out[4];
  float uu[4], vv[4]; uint32_t mm[4];
  wasm_v128_store(uu, u); wasm_v128_store(vv, v); wasm_v128_store(mm, mask);
  for (int i = 0; i < 4; i++) {
    if (!mm[i]) { out[i] = 0; continue; }
    int xi = (int)uu[i]; int yi = (int)vv[i];
    /* wrap (repeat) */
    xi &= (tw - 1); yi &= (th - 1);          /* assumes POT; fast path */
    if (xi < 0) xi += tw; if (yi < 0) yi += th;
    out[i] = tex[yi * tw + xi];
  }
  return wasm_v128_load(out);
}

/* The core triangle rasterizer. Verts are in screen space. */
EXPORT void sp_draw_triangle(const sp_vert *v0, const sp_vert *v1, const sp_vert *v2) {
  sp_ctx *ctx = &g_ctx;
  int W = ctx->w, H = ctx->h;

  /* signed area * 2 (in pixel space). orient2d(v0,v1,v2). */
  float area2 = (v1->x - v0->x) * (v2->y - v0->y) - (v1->y - v0->y) * (v2->x - v0->x);
  if (area2 == 0.0f) return;                  /* degenerate */
  /* back-face cull: keep CCW (area2 > 0). Flip sign handling so both windings
   * of front faces work — for Phase 0 we accept either by normalizing. */
  float sgn = area2 < 0.0f ? -1.0f : 1.0f;
  float inv_area2 = 1.0f / area2;             /* keeps sign; bary sums to 1 */

  ctx->tris++;

  /* bounding box, clamped to screen */
  float minxf = sp_min3f(v0->x, v1->x, v2->x);
  float maxxf = sp_max3f(v0->x, v1->x, v2->x);
  float minyf = sp_min3f(v0->y, v1->y, v2->y);
  float maxyf = sp_max3f(v0->y, v1->y, v2->y);
  int minx = (int)floorf(minxf); int maxx = (int)ceilf(maxxf);
  int miny = (int)floorf(minyf); int maxy = (int)ceilf(maxyf);
  if (minx < 0) minx = 0; if (miny < 0) miny = 0;
  if (maxx > W) maxx = W; if (maxy > H) maxy = H;
  if (minx >= maxx || miny >= maxy) return;

  /* edge setup: E_i(x,y) = A_i*x + B_i*y + C_i.
   * edge0: v1->v2 (opposite v0) gives bary for v0, etc.
   *
   * Optimization: pre-scale every edge coefficient by inv_area2. Then E_i(x,y)
   * evaluated at a pixel IS the normalized barycentric weight w_i directly — no
   * separate per-pixel multiply — AND because inv_area2 carries the winding
   * sign, the inside test collapses to a bare (w0|w1|w2) >= 0 with NO per-pixel
   * sgn multiply. This removes 3 muls + 3 muls per 4-pixel group. */
  float A0 = (v1->y - v2->y) * inv_area2, B0 = (v2->x - v1->x) * inv_area2;
  float A1 = (v2->y - v0->y) * inv_area2, B1 = (v0->x - v2->x) * inv_area2;
  float A2 = (v0->y - v1->y) * inv_area2, B2 = (v1->x - v0->x) * inv_area2;
  float C0 = -(A0 * v1->x + B0 * v1->y);
  float C1 = -(A1 * v2->x + B1 * v2->y);
  float C2 = -(A2 * v0->x + B2 * v0->y);
  (void)sgn;

  /* per-vertex attributes premultiplied for interpolation. For perspective
   * correctness we interpolate attr*invw and invw, then divide. For affine we
   * interpolate attr directly. */
  int persp = (ctx->flags & SP_FLAG_PERSP_CORRECT) != 0;
  float iw0 = v0->invw, iw1 = v1->invw, iw2 = v2->invw;

  /* attribute sources (either raw or *invw) */
  float r0=v0->r,r1=v1->r,r2=v2->r, g0=v0->g,g1=v1->g,g2=v2->g;
  float b0=v0->b,b1=v1->b,b2=v2->b, a0=v0->a,a1=v1->a,a2=v2->a;
  float u0=v0->u,u1=v1->u,u2=v2->u, vv0=v0->v,vv1=v1->v,vv2=v2->v;
  if (persp) {
    r0*=iw0;r1*=iw1;r2*=iw2; g0*=iw0;g1*=iw1;g2*=iw2;
    b0*=iw0;b1*=iw1;b2*=iw2; a0*=iw0;a1*=iw1;a2*=iw2;
    u0*=iw0;u1*=iw1;u2*=iw2; vv0*=iw0;vv1*=iw1;vv2*=iw2;
  }

  int do_depth = (ctx->flags & SP_FLAG_DEPTH_TEST) != 0;
  int do_tex   = (ctx->flags & SP_FLAG_TEXTURE) != 0 && ctx->tex != NULL;
  int do_blend = (ctx->flags & SP_FLAG_BLEND) != 0;

  v128_t lane_offset = wasm_f32x4_make(0.0f, 1.0f, 2.0f, 3.0f); /* 4 px in a row */
  v128_t tex_wf = wasm_f32x4_splat(ctx->tex_wf), tex_hf = wasm_f32x4_splat(ctx->tex_hf);
  v128_t zero4 = wasm_f32x4_splat(0.0f);
  v128_t maxx4 = wasm_f32x4_splat((float)maxx);
  /* per-4-pixel-group horizontal step of each (already-normalized) edge value */
  v128_t stepX0 = wasm_f32x4_splat(A0 * 4.0f);
  v128_t stepX1 = wasm_f32x4_splat(A1 * 4.0f);
  v128_t stepX2 = wasm_f32x4_splat(A2 * 4.0f);
  /* lane-spread of A_i for the initial group (A_i * {0,1,2,3}) */
  v128_t A0lane = wasm_f32x4_mul(wasm_f32x4_splat(A0), lane_offset);
  v128_t A1lane = wasm_f32x4_mul(wasm_f32x4_splat(A1), lane_offset);
  v128_t A2lane = wasm_f32x4_mul(wasm_f32x4_splat(A2), lane_offset);

  uint32_t *colorBuf = ctx->color;
  float *depthBuf = ctx->depth;

  float fminx = (float)minx + 0.5f;

  for (int y = miny; y < maxy; y++) {
    float yc = (float)y + 0.5f;
    /* edge values at the first group's 4 lanes (x = minx..minx+3) */
    float row0 = A0 * fminx + B0 * yc + C0;
    float row1 = A1 * fminx + B1 * yc + C1;
    float row2 = A2 * fminx + B2 * yc + C2;
    v128_t w0 = wasm_f32x4_add(wasm_f32x4_splat(row0), A0lane);
    v128_t w1 = wasm_f32x4_add(wasm_f32x4_splat(row1), A1lane);
    v128_t w2 = wasm_f32x4_add(wasm_f32x4_splat(row2), A2lane);

    for (int x = minx; x < maxx; x += 4,
         w0 = wasm_f32x4_add(w0, stepX0),
         w1 = wasm_f32x4_add(w1, stepX1),
         w2 = wasm_f32x4_add(w2, stepX2)) {
      /* w0,w1,w2 ARE the normalized barycentric weights; inside iff all >= 0.
       * (winding sign already folded into inv_area2.) */
      v128_t inside = wasm_v128_and(
                        wasm_v128_and(wasm_f32x4_ge(w0, zero4),
                                      wasm_f32x4_ge(w1, zero4)),
                        wasm_f32x4_ge(w2, zero4));

      /* mask out lanes past maxx (last group in the row) */
      v128_t xabs = wasm_f32x4_add(wasm_f32x4_splat((float)x), lane_offset);
      inside = wasm_v128_and(inside, wasm_f32x4_lt(xabs, maxx4));

      if (!wasm_v128_any_true(inside)) continue;

      /* interpolate depth (always linear in screen space) */
      v128_t z = wasm_f32x4_add(
                   wasm_f32x4_add(wasm_f32x4_mul(w0, wasm_f32x4_splat(v0->z)),
                                  wasm_f32x4_mul(w1, wasm_f32x4_splat(v1->z))),
                   wasm_f32x4_mul(w2, wasm_f32x4_splat(v2->z)));

      int base = y * W + x;
      v128_t passmask = inside;

      if (do_depth) {
        /* load existing depth for the 4 px (may read 1-3 past maxx; guarded by
         * mask before store). Clamp load to allocated buffer. */
        v128_t zbuf = wasm_v128_load(depthBuf + base);
        v128_t zless = wasm_f32x4_lt(z, zbuf);   /* new closer? */
        passmask = wasm_v128_and(passmask, zless);
        if (!wasm_v128_any_true(passmask)) continue;
      }

      /* recover 1/invw for perspective division */
      v128_t color4;
      if (do_tex) {
        v128_t u = wasm_f32x4_add(
                     wasm_f32x4_add(wasm_f32x4_mul(w0, wasm_f32x4_splat(u0)),
                                    wasm_f32x4_mul(w1, wasm_f32x4_splat(u1))),
                     wasm_f32x4_mul(w2, wasm_f32x4_splat(u2)));
        v128_t v = wasm_f32x4_add(
                     wasm_f32x4_add(wasm_f32x4_mul(w0, wasm_f32x4_splat(vv0)),
                                    wasm_f32x4_mul(w1, wasm_f32x4_splat(vv1))),
                     wasm_f32x4_mul(w2, wasm_f32x4_splat(vv2)));
        if (persp) {
          v128_t iw = wasm_f32x4_add(
                        wasm_f32x4_add(wasm_f32x4_mul(w0, wasm_f32x4_splat(iw0)),
                                       wasm_f32x4_mul(w1, wasm_f32x4_splat(iw1))),
                        wasm_f32x4_mul(w2, wasm_f32x4_splat(iw2)));
          v128_t rw = wasm_f32x4_div(wasm_f32x4_splat(1.0f), iw);
          u = wasm_f32x4_mul(u, rw);
          v = wasm_f32x4_mul(v, rw);
        }
        /* to texel space */
        u = wasm_f32x4_mul(u, tex_wf);
        v = wasm_f32x4_mul(v, tex_hf);
        color4 = sp_tex_nearest4(u, v, passmask); /* bilinear added later */
      } else {
        /* vertex color path: interpolate rgba, divide if persp, pack to RGBA8 */
        v128_t r = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(r0)),
                                                 wasm_f32x4_mul(w1,wasm_f32x4_splat(r1))),
                                  wasm_f32x4_mul(w2,wasm_f32x4_splat(r2)));
        v128_t g = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(g0)),
                                                 wasm_f32x4_mul(w1,wasm_f32x4_splat(g1))),
                                  wasm_f32x4_mul(w2,wasm_f32x4_splat(g2)));
        v128_t b = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(b0)),
                                                 wasm_f32x4_mul(w1,wasm_f32x4_splat(b1))),
                                  wasm_f32x4_mul(w2,wasm_f32x4_splat(b2)));
        v128_t a = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(a0)),
                                                 wasm_f32x4_mul(w1,wasm_f32x4_splat(a1))),
                                  wasm_f32x4_mul(w2,wasm_f32x4_splat(a2)));
        if (persp) {
          v128_t iw = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(iw0)),
                                                    wasm_f32x4_mul(w1,wasm_f32x4_splat(iw1))),
                                     wasm_f32x4_mul(w2,wasm_f32x4_splat(iw2)));
          v128_t rw = wasm_f32x4_div(wasm_f32x4_splat(1.0f), iw);
          r=wasm_f32x4_mul(r,rw); g=wasm_f32x4_mul(g,rw); b=wasm_f32x4_mul(b,rw); a=wasm_f32x4_mul(a,rw);
        }
        /* clamp [0,1] -> [0,255] */
        v128_t s255 = wasm_f32x4_splat(255.0f);
        v128_t zero = wasm_f32x4_splat(0.0f);
        r = wasm_f32x4_max(zero, wasm_f32x4_min(wasm_f32x4_splat(1.0f), r));
        g = wasm_f32x4_max(zero, wasm_f32x4_min(wasm_f32x4_splat(1.0f), g));
        b = wasm_f32x4_max(zero, wasm_f32x4_min(wasm_f32x4_splat(1.0f), b));
        a = wasm_f32x4_max(zero, wasm_f32x4_min(wasm_f32x4_splat(1.0f), a));
        v128_t ri = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(r, s255));
        v128_t gi = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(g, s255));
        v128_t bi = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(b, s255));
        v128_t ai = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(a, s255));
        /* pack RGBA little-endian: R | G<<8 | B<<16 | A<<24 */
        color4 = wasm_v128_or(
                   wasm_v128_or(ri, wasm_i32x4_shl(gi, 8)),
                   wasm_v128_or(wasm_i32x4_shl(bi, 16), wasm_i32x4_shl(ai, 24)));
      }

      /* write masked: blend optional */
      v128_t dst = wasm_v128_load(colorBuf + base);
      v128_t outc;
      if (do_blend) {
        /* simple src-over using per-lane scalar (alpha unpack) — Phase 0 keeps it
         * correct-ish; SIMD byte blend comes later. */
        uint32_t srcp[4], dstp[4], mskp[4], res[4];
        wasm_v128_store(srcp, color4); wasm_v128_store(dstp, dst); wasm_v128_store(mskp, passmask);
        for (int i=0;i<4;i++){
          if(!mskp[i]){ res[i]=dstp[i]; continue; }
          uint32_t s=srcp[i], d=dstp[i];
          uint32_t sa=(s>>24)&0xff; uint32_t ia=255-sa;
          uint32_t rr=(((s)&0xff)*sa + ((d)&0xff)*ia)/255;
          uint32_t gg=(((s>>8)&0xff)*sa + ((d>>8)&0xff)*ia)/255;
          uint32_t bb=(((s>>16)&0xff)*sa + ((d>>16)&0xff)*ia)/255;
          res[i]=rr | (gg<<8) | (bb<<16) | (0xffu<<24);
        }
        outc = wasm_v128_load(res);
      } else {
        outc = wasm_v128_bitselect(color4, dst, passmask);
      }
      wasm_v128_store(colorBuf + base, outc);

      if (do_depth) {
        v128_t newz = wasm_v128_bitselect(z, wasm_v128_load(depthBuf + base), passmask);
        wasm_v128_store(depthBuf + base, newz);
      }

      /* count shaded lanes */
      ctx->frag_shaded += (uint64_t)(__builtin_popcount(wasm_i32x4_bitmask(passmask)));
    }
  }
}

/* Batch entry point: draw N triangles from a flat float buffer.
 * Layout per vertex (10 floats): x y z invw r g b a u v
 * Per triangle: 3 vertices = 30 floats. `verts` points at 3*N*10 floats. */
EXPORT void sp_draw_triangles_flat(const float *verts, int ntris) {
  for (int t = 0; t < ntris; t++) {
    const float *p = verts + (size_t)t * 30;
    sp_vert a, b, c;
    a.x=p[0];a.y=p[1];a.z=p[2];a.invw=p[3];a.r=p[4];a.g=p[5];a.b=p[6];a.a=p[7];a.u=p[8];a.v=p[9];
    b.x=p[10];b.y=p[11];b.z=p[12];b.invw=p[13];b.r=p[14];b.g=p[15];b.b=p[16];b.a=p[17];b.u=p[18];b.v=p[19];
    c.x=p[20];c.y=p[21];c.z=p[22];c.invw=p[23];c.r=p[24];c.g=p[25];c.b=p[26];c.a=p[27];c.u=p[28];c.v=p[29];
    sp_draw_triangle(&a, &b, &c);
  }
}
