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

/* sample texture bilinear, 4 lanes (4 taps/lane = the expensive path). */
static inline v128_t sp_tex_bilinear4(v128_t u, v128_t v, v128_t mask) {
  const uint32_t *tex = g_ctx.tex;
  int tw = g_ctx.tex_w, th = g_ctx.tex_h;
  uint32_t out[4];
  float uu[4], vv[4]; uint32_t mm[4];
  wasm_v128_store(uu, u); wasm_v128_store(vv, v); wasm_v128_store(mm, mask);
  for (int i = 0; i < 4; i++) {
    if (!mm[i]) { out[i] = 0; continue; }
    float fx = uu[i] - 0.5f, fy = vv[i] - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float dx = fx - (float)x0, dy = fy - (float)y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    int x0m = x0 & (tw - 1), x1m = x1 & (tw - 1), y0m = y0 & (th - 1), y1m = y1 & (th - 1);
    uint32_t c00 = tex[y0m * tw + x0m], c10 = tex[y0m * tw + x1m];
    uint32_t c01 = tex[y1m * tw + x0m], c11 = tex[y1m * tw + x1m];
    float w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy), w01 = (1 - dx) * dy, w11 = dx * dy;
    uint32_t res = 0;
    for (int s = 0; s < 32; s += 8) {
      float c = ((c00 >> s) & 0xff) * w00 + ((c10 >> s) & 0xff) * w10
              + ((c01 >> s) & 0xff) * w01 + ((c11 >> s) & 0xff) * w11;
      int ci = (int)(c + 0.5f); if (ci > 255) ci = 255;
      res |= ((uint32_t)ci) << s;
    }
    out[i] = res;
  }
  return wasm_v128_load(out);
}

/* Thread-local Y-band clip. Each worker rasterizes the WHOLE triangle list but
 * only writes rows [clip_y0, clip_y1). Bands are disjoint → no two threads ever
 * touch the same framebuffer pixel → zero locking. Defaults cover full height. */
static _Thread_local int tl_clip_y0 = 0;
static _Thread_local int tl_clip_y1 = 1 << 30;

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
  /* clamp to this thread's band */
  if (miny < tl_clip_y0) miny = tl_clip_y0;
  if (maxy > tl_clip_y1) maxy = tl_clip_y1;
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
  int do_bilin = (ctx->flags & SP_FLAG_BILINEAR) != 0;
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
        color4 = do_bilin ? sp_tex_bilinear4(u, v, passmask)   /* 4 taps/lane */
                          : sp_tex_nearest4(u, v, passmask);   /* 1 tap/lane  */
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

/* ---------- programmable path: varying G-buffer ----------
 * Instead of writing final color, rasterize interpolated varyings into SoA
 * planes (u, v, r, g, b, a) + a coverage mask, performing the depth test/write.
 * A programmable fragment shader (JS callback in Tier-2, or a JIT'd WASM module
 * in Tier-1) then consumes these planes and writes color. This is what makes
 * simdpipe a real (programmable) renderer rather than fixed-function.
 *
 * G-buffer planes are full-framebuffer SoA (w*h each). `cover` is 0/1 per pixel
 * for "this pixel was written by some triangle this pass and passed depth". */
static float *g_gb_u, *g_gb_v, *g_gb_r, *g_gb_g, *g_gb_b, *g_gb_a;
static uint8_t *g_gb_cover;

EXPORT int sp_gbuffer_init(void) {
  size_t n = (size_t)g_ctx.w * g_ctx.h + 4;
  g_gb_u = malloc(n * 4); g_gb_v = malloc(n * 4);
  g_gb_r = malloc(n * 4); g_gb_g = malloc(n * 4);
  g_gb_b = malloc(n * 4); g_gb_a = malloc(n * 4);
  g_gb_cover = malloc(n);
  return g_gb_u && g_gb_v && g_gb_r && g_gb_g && g_gb_b && g_gb_a && g_gb_cover;
}
EXPORT float *sp_gb_u(void){return g_gb_u;} EXPORT float *sp_gb_v(void){return g_gb_v;}
EXPORT float *sp_gb_r(void){return g_gb_r;} EXPORT float *sp_gb_g(void){return g_gb_g;}
EXPORT float *sp_gb_b(void){return g_gb_b;} EXPORT float *sp_gb_a(void){return g_gb_a;}
EXPORT uint8_t *sp_gb_cover(void){return g_gb_cover;}

EXPORT void sp_gbuffer_clear(void) {
  memset(g_gb_cover, 0, (size_t)g_ctx.w * g_ctx.h);
}

/* Rasterize one triangle into the varying G-buffer (depth-tested). */
static void sp_raster_gbuffer(const sp_vert *v0, const sp_vert *v1, const sp_vert *v2) {
  sp_ctx *ctx = &g_ctx;
  int W = ctx->w, H = ctx->h;
  float area2 = (v1->x - v0->x) * (v2->y - v0->y) - (v1->y - v0->y) * (v2->x - v0->x);
  if (area2 == 0.0f) return;
  float inv_area2 = 1.0f / area2;

  float minxf = sp_min3f(v0->x, v1->x, v2->x), maxxf = sp_max3f(v0->x, v1->x, v2->x);
  float minyf = sp_min3f(v0->y, v1->y, v2->y), maxyf = sp_max3f(v0->y, v1->y, v2->y);
  int minx = (int)floorf(minxf), maxx = (int)ceilf(maxxf);
  int miny = (int)floorf(minyf), maxy = (int)ceilf(maxyf);
  if (minx < 0) minx = 0; if (miny < 0) miny = 0;
  if (maxx > W) maxx = W; if (maxy > H) maxy = H;
  if (minx >= maxx || miny >= maxy) return;

  float A0 = (v1->y - v2->y) * inv_area2, B0 = (v2->x - v1->x) * inv_area2;
  float A1 = (v2->y - v0->y) * inv_area2, B1 = (v0->x - v2->x) * inv_area2;
  float A2 = (v0->y - v1->y) * inv_area2, B2 = (v1->x - v0->x) * inv_area2;
  float C0 = -(A0 * v1->x + B0 * v1->y);
  float C1 = -(A1 * v2->x + B1 * v2->y);
  float C2 = -(A2 * v0->x + B2 * v0->y);

  int persp = (ctx->flags & SP_FLAG_PERSP_CORRECT) != 0;
  int do_depth = (ctx->flags & SP_FLAG_DEPTH_TEST) != 0;
  float iw0=v0->invw, iw1=v1->invw, iw2=v2->invw;
  float u0=v0->u,u1=v1->u,u2=v2->u, vv0=v0->v,vv1=v1->v,vv2=v2->v;
  float r0=v0->r,r1=v1->r,r2=v2->r, g0=v0->g,g1=v1->g,g2=v2->g;
  float b0=v0->b,b1=v1->b,b2=v2->b, a0=v0->a,a1=v1->a,a2=v2->a;
  if (persp){u0*=iw0;u1*=iw1;u2*=iw2;vv0*=iw0;vv1*=iw1;vv2*=iw2;
             r0*=iw0;r1*=iw1;r2*=iw2;g0*=iw0;g1*=iw1;g2*=iw2;
             b0*=iw0;b1*=iw1;b2*=iw2;a0*=iw0;a1*=iw1;a2*=iw2;}

  float *gu=g_gb_u,*gv=g_gb_v,*gr=g_gb_r,*gg=g_gb_g,*gb=g_gb_b,*ga=g_gb_a;
  uint8_t *gc=g_gb_cover; float *depthBuf=ctx->depth;

  for (int y = miny; y < maxy; y++) {
    float yc = (float)y + 0.5f;
    for (int x = minx; x < maxx; x++) {
      float xc = (float)x + 0.5f;
      float w0 = A0*xc + B0*yc + C0;
      float w1 = A1*xc + B1*yc + C1;
      float w2 = A2*xc + B2*yc + C2;
      if (w0 < 0 || w1 < 0 || w2 < 0) continue;
      int idx = y*W + x;
      float z = w0*v0->z + w1*v1->z + w2*v2->z;
      if (do_depth) { if (z >= depthBuf[idx]) continue; }
      float recip = 1.0f;
      if (persp) { float iw = w0*iw0 + w1*iw1 + w2*iw2; recip = 1.0f/iw; }
      gu[idx] = (w0*u0+w1*u1+w2*u2)*recip;
      gv[idx] = (w0*vv0+w1*vv1+w2*vv2)*recip;
      gr[idx] = (w0*r0+w1*r1+w2*r2)*recip;
      gg[idx] = (w0*g0+w1*g1+w2*g2)*recip;
      gb[idx] = (w0*b0+w1*b1+w2*b2)*recip;
      ga[idx] = (w0*a0+w1*a1+w2*a2)*recip;
      gc[idx] = 1;
      if (do_depth) depthBuf[idx] = z;
    }
  }
}

EXPORT void sp_draw_gbuffer_flat(const float *verts, int ntris) {
  for (int t = 0; t < ntris; t++) {
    const float *p = verts + (size_t)t * 30;
    sp_vert a, b, c;
    a.x=p[0];a.y=p[1];a.z=p[2];a.invw=p[3];a.r=p[4];a.g=p[5];a.b=p[6];a.a=p[7];a.u=p[8];a.v=p[9];
    b.x=p[10];b.y=p[11];b.z=p[12];b.invw=p[13];b.r=p[14];b.g=p[15];b.b=p[16];b.a=p[17];b.u=p[18];b.v=p[19];
    c.x=p[20];c.y=p[21];c.z=p[22];c.invw=p[23];c.r=p[24];c.g=p[25];c.b=p[26];c.a=p[27];c.u=p[28];c.v=p[29];
    sp_raster_gbuffer(&a, &b, &c);
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

/* ---------- threaded rasterization (optional; pthreads build) ---------- */
#ifdef SP_THREADS
#include <pthread.h>
#include <stdatomic.h>

/* Persistent worker pool with atomic tile-row dispatch.
 *
 * Created ONCE (sp_pool_start). The screen is split into TILE_ROWS-high bands;
 * a shared atomic counter hands the next band to whichever worker is free
 * (work-stealing → load-balanced, unlike static bands). Sync is two barriers per
 * draw: all threads rendezvous at `bar_start` (job is published), drain the
 * counter, then rendezvous at `bar_done`. No per-frame thread spawn.
 *
 * Bands are disjoint pixel rows, so workers never write the same pixel — zero
 * locking on the framebuffer/G-buffer. */
#define SP_TILE_ROWS 16
#define SP_POOL_MAX 32

static struct {
  int nthreads;
  pthread_t th[SP_POOL_MAX];
  pthread_barrier_t bar_start, bar_done;
  /* job */
  const float *verts;
  int ntris;
  int nbands;
  int gbuffer;             /* 1 = G-buffer path, 0 = fixed-function color path */
  _Atomic int next_band;   /* atomic work counter */
  _Atomic int shutdown;
  int started;
} g_pool;

static void sp_run_band(int band) {
  int y0 = band * SP_TILE_ROWS;
  int y1 = y0 + SP_TILE_ROWS;
  if (y1 > g_ctx.h) y1 = g_ctx.h;
  tl_clip_y0 = y0; tl_clip_y1 = y1;
  if (g_pool.gbuffer) sp_draw_gbuffer_flat(g_pool.verts, g_pool.ntris);
  else                sp_draw_triangles_flat(g_pool.verts, g_pool.ntris);
}

static void sp_drain_bands(void) {
  for (;;) {
    int band = atomic_fetch_add(&g_pool.next_band, 1);
    if (band >= g_pool.nbands) break;
    sp_run_band(band);
  }
}

static void *sp_pool_worker(void *arg) {
  (void)arg;
  for (;;) {
    pthread_barrier_wait(&g_pool.bar_start);
    if (atomic_load(&g_pool.shutdown)) return NULL;
    sp_drain_bands();
    pthread_barrier_wait(&g_pool.bar_done);
  }
}

/* Start the pool with n worker threads (call once after sp_init). */
EXPORT int sp_pool_start(int n) {
  if (g_pool.started) return 1;
  if (n < 1) n = 1; if (n > SP_POOL_MAX) n = SP_POOL_MAX;
  g_pool.nthreads = n;
  atomic_store(&g_pool.shutdown, 0);
  /* barriers include the main thread (+1) */
  pthread_barrier_init(&g_pool.bar_start, NULL, n + 1);
  pthread_barrier_init(&g_pool.bar_done, NULL, n + 1);
  for (int i = 0; i < n; i++) {
    if (pthread_create(&g_pool.th[i], NULL, sp_pool_worker, NULL) != 0) return 0;
  }
  g_pool.started = 1;
  return 1;
}

EXPORT void sp_pool_stop(void) {
  if (!g_pool.started) return;
  atomic_store(&g_pool.shutdown, 1);
  pthread_barrier_wait(&g_pool.bar_start);   /* release workers to see shutdown */
  for (int i = 0; i < g_pool.nthreads; i++) pthread_join(g_pool.th[i], NULL);
  pthread_barrier_destroy(&g_pool.bar_start);
  pthread_barrier_destroy(&g_pool.bar_done);
  g_pool.started = 0;
}

/* internal: dispatch the current job across the pool + main thread */
static void sp_pool_dispatch(const float *verts, int ntris, int gbuffer) {
  g_pool.verts = verts; g_pool.ntris = ntris; g_pool.gbuffer = gbuffer;
  g_pool.nbands = (g_ctx.h + SP_TILE_ROWS - 1) / SP_TILE_ROWS;
  atomic_store(&g_pool.next_band, 0);
  pthread_barrier_wait(&g_pool.bar_start);   /* go */
  sp_drain_bands();                          /* main thread participates */
  pthread_barrier_wait(&g_pool.bar_done);    /* join */
}

/* Below this triangle count, barrier/dispatch overhead exceeds the parallel win,
 * so run serially. (WASM futex-based barrier latency makes fine-grained sync
 * costly; only amortizes on substantial frames.) Tunable. */
#define SP_POOL_MIN_TRIS 256

/* Pooled fixed-function draw (color path). Falls back to serial if no pool or if
 * the frame is too small to amortize thread sync. */
EXPORT void sp_draw_triangles_pooled(const float *verts, int ntris) {
  if (!g_pool.started || ntris < SP_POOL_MIN_TRIS) { sp_draw_triangles_flat(verts, ntris); return; }
  sp_pool_dispatch(verts, ntris, 0);
}

/* Pooled programmable raster (G-buffer path). Caller clears the G-buffer first. */
EXPORT void sp_draw_gbuffer_pooled(const float *verts, int ntris) {
  if (!g_pool.started || ntris < SP_POOL_MIN_TRIS) { sp_draw_gbuffer_flat(verts, ntris); return; }
  sp_pool_dispatch(verts, ntris, 1);
}

/* ---- legacy per-frame band spawn (kept for the scaling benchmark) ---- */
typedef struct { const float *verts; int ntris; int y0, y1; } sp_band_job;
static void *sp_band_worker(void *arg) {
  sp_band_job *j = (sp_band_job *)arg;
  tl_clip_y0 = j->y0; tl_clip_y1 = j->y1;
  sp_draw_triangles_flat(j->verts, j->ntris);
  return NULL;
}
EXPORT void sp_draw_triangles_threaded(const float *verts, int ntris, int nthreads) {
  if (nthreads <= 1) { sp_draw_triangles_flat(verts, ntris); return; }
  if (nthreads > 32) nthreads = 32;
  int H = g_ctx.h;
  pthread_t th[32]; sp_band_job jobs[32];
  int rows_per = (H + nthreads - 1) / nthreads;
  int spawned = 0;
  for (int i = 0; i < nthreads; i++) {
    int y0 = i * rows_per, y1 = y0 + rows_per;
    if (y0 >= H) break; if (y1 > H) y1 = H;
    jobs[i].verts = verts; jobs[i].ntris = ntris; jobs[i].y0 = y0; jobs[i].y1 = y1;
    pthread_create(&th[i], NULL, sp_band_worker, &jobs[i]); spawned++;
  }
  for (int i = 0; i < spawned; i++) pthread_join(th[i], NULL);
}
#endif
