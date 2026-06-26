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

/* Rasterizer tile size (pixels). Used for hierarchical reject/accept AND the
 * coarse per-tile Zmax depth-rejection grid, so it's a file-level constant. */
#ifndef SP_RASTER_TILE
#define SP_RASTER_TILE 16   /* swept w/ correct coarse-depth: 16 best overall */
#endif

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

  /* coarse depth pyramid: per-tile MAX depth (conservative). A triangle whose
   * minimum depth over a tile exceeds the tile's Zmax is fully occluded there
   * and the whole tile is skipped — one compare instead of 256 per-pixel tests.
   * Only ever raised toward 1.0 on clear and lowered when a tile is fully
   * covered by closer geometry, so rejection is always safe. */
  float   *ztile;
  int      tiles_w, tiles_h;

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
  /* coarse Zmax tile grid */
  g_ctx.tiles_w = (w + SP_RASTER_TILE - 1) / SP_RASTER_TILE;
  g_ctx.tiles_h = (h + SP_RASTER_TILE - 1) / SP_RASTER_TILE;
  g_ctx.ztile = (float *)malloc((size_t)g_ctx.tiles_w * g_ctx.tiles_h * sizeof(float));
  if (!g_ctx.ztile) return 0;
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
  /* reset the coarse Zmax grid to the clear depth (the tile's farthest possible) */
  if (g_ctx.ztile) {
    int nt = g_ctx.tiles_w * g_ctx.tiles_h;
    for (int t = 0; t < nt; t++) g_ctx.ztile[t] = depth;
  }
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

  /* Factored barycentric interpolation: since w0+w1+w2==1, any attribute is
   * a == a2 + w0*(a0-a2) + w1*(a1-a2) — 2 muls/2 adds per channel instead of 3/2,
   * and w2 isn't needed for interpolation. Splat the base (v2) and the two deltas
   * once per triangle. (Saves ~1 mul per channel per shaded group.) */
  v128_t cR2=wasm_f32x4_splat(r2), Rd0=wasm_f32x4_splat(r0-r2), Rd1=wasm_f32x4_splat(r1-r2);
  v128_t cG2=wasm_f32x4_splat(g2), Gd0=wasm_f32x4_splat(g0-g2), Gd1=wasm_f32x4_splat(g1-g2);
  v128_t cB2=wasm_f32x4_splat(b2), Bd0=wasm_f32x4_splat(b0-b2), Bd1=wasm_f32x4_splat(b1-b2);
  v128_t cA2=wasm_f32x4_splat(a2), Ad0=wasm_f32x4_splat(a0-a2), Ad1=wasm_f32x4_splat(a1-a2);
  v128_t IW2=wasm_f32x4_splat(iw2), IWd0=wasm_f32x4_splat(iw0-iw2), IWd1=wasm_f32x4_splat(iw1-iw2);
  v128_t U2=wasm_f32x4_splat(u2), Ud0=wasm_f32x4_splat(u0-u2), Ud1=wasm_f32x4_splat(u1-u2);
  v128_t V2=wasm_f32x4_splat(vv2), Vd0=wasm_f32x4_splat(vv0-vv2), Vd1=wasm_f32x4_splat(vv1-vv2);
  v128_t Z2v=wasm_f32x4_splat(v2->z), Zd0=wasm_f32x4_splat(v0->z - v2->z), Zd1=wasm_f32x4_splat(v1->z - v2->z);
  /* a == base + w0*d0 + w1*d1 */
  #define SP_LERP(base,d0,d1) wasm_f32x4_add(base, wasm_f32x4_add(wasm_f32x4_mul(w0,d0), wasm_f32x4_mul(w1,d1)))

  uint32_t *colorBuf = ctx->color;
  float *depthBuf = ctx->depth;

  /* ---- hierarchical tiled traversal ----
   * Brute-forcing every pixel in the bbox is the real bottleneck on overdraw
   * scenes (most pixels get inside-/depth-rejected one at a time). Instead we
   * walk SP_RASTER_TILE-sized tiles and per tile do a trivial reject / accept:
   *   - reject: if any edge is < 0 at the tile corner where that edge is LARGEST,
   *     the whole tile is outside the triangle → skip it, zero per-pixel tests.
   *   - accept: if every edge is >= 0 at the tile corner where it is SMALLEST,
   *     the whole tile is inside → skip the per-pixel inside test entirely.
   * Corner selection per edge is fixed by the signs of A_i, B_i.
   *
   * On top of geometric reject/accept, a coarse Zmax test skips tiles where the
   * triangle is fully occluded (its min depth in the tile > the tile's Zmax). */
  const int TILE = SP_RASTER_TILE;
  /* Coarse-depth is safe only when THIS call owns the whole framebuffer height
   * (serial / single-thread fallback). Banded pool workers each see a y-slice and
   * share one g_ctx.ztile; gating on full-screen ownership keeps the grid
   * single-writer without a compile-time switch — the serial path (incl. the
   * pool's small-frame fallback) gets the win, banded workers skip it. */
  int do_ztile = do_depth && ctx->ztile != NULL
                 && tl_clip_y0 <= 0 && tl_clip_y1 >= ctx->h;
  /* Small triangles (bbox within one tile) don't benefit from the coarse grid and
   * would only pay its alignment + per-cell overhead. Skip ztile for them: not
   * updating/rejecting is always conservative-safe, and it lets the tile loop use
   * tight bbox-origin tiles (no grid snap) — a big win on tiny-triangle scenes. */
  if ((maxx - minx) <= SP_RASTER_TILE && (maxy - miny) <= SP_RASTER_TILE) do_ztile = 0;
#ifdef SP_FORCE_NO_ZTILE
  do_ztile = 0;
#endif
  float * ztileBuf = ctx->ztile;
  int tilesW = ctx->tiles_w;
  /* z is an affine plane in screen space. Since the normalized edges satisfy
   * w_i = A_i*x + B_i*y + C_i and z = vz0*w0+vz1*w1+vz2*w2, the z-plane is
   * z(x,y) = Az*x + Bz*y + Cz with: */
  float vz0 = v0->z, vz1 = v1->z, vz2 = v2->z;
  float Az = vz0 * A0 + vz1 * A1 + vz2 * A2;
  float Bz = vz0 * B0 + vz1 * B1 + vz2 * B2;
  float Cz = vz0 * C0 + vz1 * C1 + vz2 * C2;
  const int zxpos = (Az >= 0.0f), zypos = (Bz >= 0.0f); /* corner pick for z min/max */
  /* For edge i, the "min corner" (E smallest) uses xlo if A_i>=0 else xhi, and
   * ylo if B_i>=0 else yhi; the "max corner" is the opposite. Precompute which. */
  const int x0pos0 = (A0 >= 0.0f), x0pos1 = (A1 >= 0.0f), x0pos2 = (A2 >= 0.0f);
  const int y0pos0 = (B0 >= 0.0f), y0pos1 = (B1 >= 0.0f), y0pos2 = (B2 >= 0.0f);

  /* Tiles are ALWAYS snapped to the grid (start at floor(min/TILE)*TILE) and the
   * processed range clamped to the bbox. This keeps the per-pixel edge stepping
   * origin identical whether or not ztile is active, so coarse-depth is provably
   * output-invariant (ztile-on == ztile-off, byte for byte). Grid snap is also
   * required for ztile-cell correctness when it IS active. */
  int tyStart = (miny / TILE) * TILE;
  int txStart = (minx / TILE) * TILE;
  for (int tyB = tyStart; tyB < maxy; tyB += TILE) {
    int ty = tyB < miny ? miny : tyB;            /* clamp processed range to bbox */
    int tyend = tyB + TILE; if (tyend > maxy) tyend = maxy;
    if (ty >= tyend) continue;
    float ylo = (float)ty + 0.5f, yhi = (float)(tyend - 1) + 0.5f;
    for (int txB = txStart; txB < maxx; txB += TILE) {
      int tx = txB < minx ? minx : txB;          /* clamp processed range to bbox */
      int txend = txB + TILE; if (txend > maxx) txend = maxx;
      if (tx >= txend) continue;
      float xlo = (float)tx + 0.5f, xhi = (float)(txend - 1) + 0.5f;

      /* reject: edge at its MAX corner < 0  → tile fully outside */
      float mx0 = A0 * (x0pos0 ? xhi : xlo) + B0 * (y0pos0 ? yhi : ylo) + C0;
      if (mx0 < 0.0f) continue;
      float mx1 = A1 * (x0pos1 ? xhi : xlo) + B1 * (y0pos1 ? yhi : ylo) + C1;
      if (mx1 < 0.0f) continue;
      float mx2 = A2 * (x0pos2 ? xhi : xlo) + B2 * (y0pos2 ? yhi : ylo) + C2;
      if (mx2 < 0.0f) continue;

      /* accept: edge at its MIN corner >= 0 for all → tile fully inside */
      float mn0 = A0 * (x0pos0 ? xlo : xhi) + B0 * (y0pos0 ? ylo : yhi) + C0;
      float mn1 = A1 * (x0pos1 ? xlo : xhi) + B1 * (y0pos1 ? ylo : yhi) + C1;
      float mn2 = A2 * (x0pos2 ? xlo : xhi) + B2 * (y0pos2 ? ylo : yhi) + C2;
      int tile_full = (mn0 >= 0.0f && mn1 >= 0.0f && mn2 >= 0.0f);

      /* coarse depth reject: tightest triangle z over this tile is at the z-min
       * corner. If even that is farther than everything already in the tile, the
       * whole triangle is occluded here — skip without touching a single pixel. */
      int tileIdx = (tyB / TILE) * tilesW + (txB / TILE);  /* aligned grid cell */
      float zminTile = Az * (zxpos ? xlo : xhi) + Bz * (zypos ? ylo : yhi) + Cz;
#ifndef SP_ZTILE_NO_REJECT
      if (do_ztile && zminTile > ztileBuf[tileIdx]) continue;
#endif
      /* Track the ACTUAL max depth written in this tile (not the tile_full corner
       * estimate — that's an FP false-positive at edges: a triangle marked "full"
       * can leave uncovered pixels at clear depth while the corner Zmax dips below
       * them, causing later visible triangles to be wrongly rejected). Start at
       * -inf; max in each passing lane's z; commit to ztile at tile end. */
      v128_t tileZmaxAcc = wasm_f32x4_splat(-1.0e30f);
      int tileWrote = 0;

      float fminx = (float)tx + 0.5f;
      v128_t txmax4 = wasm_f32x4_splat((float)txend);

  for (int y = ty; y < tyend; y++) {
    float yc = (float)y + 0.5f;
    /* edge values at the first group's 4 lanes (x = tx..tx+3) */
    float row0 = A0 * fminx + B0 * yc + C0;
    float row1 = A1 * fminx + B1 * yc + C1;
    float row2 = A2 * fminx + B2 * yc + C2;
    v128_t w0 = wasm_f32x4_add(wasm_f32x4_splat(row0), A0lane);
    v128_t w1 = wasm_f32x4_add(wasm_f32x4_splat(row1), A1lane);
    v128_t w2 = wasm_f32x4_add(wasm_f32x4_splat(row2), A2lane);

    for (int x = tx; x < txend; x += 4,
         w0 = wasm_f32x4_add(w0, stepX0),
         w1 = wasm_f32x4_add(w1, stepX1),
         w2 = wasm_f32x4_add(w2, stepX2)) {
      v128_t inside;
      if (tile_full) {
        /* whole tile inside the triangle — only the right-edge x clamp matters */
        v128_t xabs = wasm_f32x4_add(wasm_f32x4_splat((float)x), lane_offset);
        inside = wasm_f32x4_lt(xabs, txmax4);
      } else {
        /* w0,w1,w2 ARE the normalized barycentric weights; inside iff all >= 0.
         * (winding sign already folded into inv_area2.) */
        inside = wasm_v128_and(
                   wasm_v128_and(wasm_f32x4_ge(w0, zero4),
                                 wasm_f32x4_ge(w1, zero4)),
                   wasm_f32x4_ge(w2, zero4));
        /* mask out lanes past the tile's right edge (last group in the row) */
        v128_t xabs = wasm_f32x4_add(wasm_f32x4_splat((float)x), lane_offset);
        inside = wasm_v128_and(inside, wasm_f32x4_lt(xabs, txmax4));
        if (!wasm_v128_any_true(inside)) continue;
      }

      /* interpolate depth (always linear in screen space; factored — runs for
       * every group incl. depth-rejected ones, so the saved mul matters most here) */
      v128_t z = SP_LERP(Z2v, Zd0, Zd1);

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
        v128_t u = SP_LERP(U2, Ud0, Ud1);
        v128_t v = SP_LERP(V2, Vd0, Vd1);
        if (persp) {
          v128_t iw = SP_LERP(IW2, IWd0, IWd1);
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
        /* vertex color path: interpolate rgba (factored), divide if persp, pack */
        v128_t r = SP_LERP(cR2, Rd0, Rd1);
        v128_t g = SP_LERP(cG2, Gd0, Gd1);
        v128_t b = SP_LERP(cB2, Bd0, Bd1);
        v128_t a = SP_LERP(cA2, Ad0, Ad1);
        if (persp) {
          v128_t iw = SP_LERP(IW2, IWd0, IWd1);
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
        /* fold this group's passing z into the tile's running max (only when the
         * coarse pyramid is active; non-passing lanes forced to -inf) */
        if (do_ztile) tileZmaxAcc = wasm_f32x4_max(tileZmaxAcc, wasm_v128_bitselect(z, wasm_f32x4_splat(-1.0e30f), passmask));
      }

      /* count shaded lanes */
      int wl = __builtin_popcount(wasm_i32x4_bitmask(passmask));
      ctx->frag_shaded += (uint64_t)wl;
      if (do_ztile) tileWrote += wl;
    }
  }
      /* coarse Zmax update — CORRECT version: commit the tile's farthest written
       * depth to the pyramid only when THIS triangle wrote EVERY pixel of the
       * tile (verified by a written-lane count == tile area, not the FP-fragile
       * tile_full corner test). When fully written, every pixel is this triangle's
       * z, so the tracked max IS the true tile max and there are no stale/uncovered
       * pixels to wrongly occlude later geometry. Blend keeps dst, so skip then. */
#ifndef SP_ZTILE_NO_UPDATE
      /* Only when this triangle covered the WHOLE, UNCLIPPED grid cell (the full
       * TILE×TILE region — not clipped by bbox/screen) AND wrote every one of its
       * pixels: then every pixel of the cell is this triangle's z, so the tracked
       * max is the true cell Zmax with no stale/uncovered pixels. A bbox-clipped
       * cell has out-of-bbox pixels at old depth, so we must NOT lower its Zmax. */
      if (do_ztile && do_depth && !do_blend
          && tx == txB && (txend - txB) == TILE
          && ty == tyB && (tyend - tyB) == TILE
          && tileWrote == TILE * TILE) {
        float m0 = wasm_f32x4_extract_lane(tileZmaxAcc, 0);
        float m1 = wasm_f32x4_extract_lane(tileZmaxAcc, 1);
        float m2 = wasm_f32x4_extract_lane(tileZmaxAcc, 2);
        float m3 = wasm_f32x4_extract_lane(tileZmaxAcc, 3);
        float mx = m0 > m1 ? m0 : m1; float mx2 = m2 > m3 ? m2 : m3; mx = mx > mx2 ? mx : mx2;
        if (mx < ztileBuf[tileIdx]) ztileBuf[tileIdx] = mx;
      }
#endif
    } /* tile x */
  }   /* tile y */
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

  /* SIMD + tiled traversal — same hierarchical reject/accept + coarse-depth as
   * sp_draw_triangle (this path was the shade-bound bottleneck: 94% of frame in
   * a scalar one-pixel loop). Writes 6 varying planes + a cover byte per lane. */
  const int TILE = SP_RASTER_TILE;
  /* coarse-depth only when this call owns the whole height (serial path); banded
   * pool workers share one ztile and would race — see sp_draw_triangle. */
  int do_ztile = do_depth && ctx->ztile != NULL
                 && tl_clip_y0 <= 0 && tl_clip_y1 >= ctx->h;
  if ((maxx - minx) <= SP_RASTER_TILE && (maxy - miny) <= SP_RASTER_TILE) do_ztile = 0;
#ifdef SP_FORCE_NO_ZTILE
  do_ztile = 0;
#endif
  float *ztileBuf = ctx->ztile; int tilesW = ctx->tiles_w;
  float Az = v0->z*A0 + v1->z*A1 + v2->z*A2;
  float Bz = v0->z*B0 + v1->z*B1 + v2->z*B2;
  float Cz = v0->z*C0 + v1->z*C1 + v2->z*C2;
  const int zxpos = (Az >= 0.0f), zypos = (Bz >= 0.0f);
  const int x0pos0=(A0>=0.0f),x0pos1=(A1>=0.0f),x0pos2=(A2>=0.0f);
  const int y0pos0=(B0>=0.0f),y0pos1=(B1>=0.0f),y0pos2=(B2>=0.0f);

  v128_t lane_off = wasm_f32x4_make(0.0f,1.0f,2.0f,3.0f);
  v128_t zero4 = wasm_f32x4_splat(0.0f);
  v128_t one4  = wasm_f32x4_splat(1.0f);
  v128_t sX0=wasm_f32x4_splat(A0*4.0f),sX1=wasm_f32x4_splat(A1*4.0f),sX2=wasm_f32x4_splat(A2*4.0f);
  v128_t A0l=wasm_f32x4_mul(wasm_f32x4_splat(A0),lane_off);
  v128_t A1l=wasm_f32x4_mul(wasm_f32x4_splat(A1),lane_off);
  v128_t A2l=wasm_f32x4_mul(wasm_f32x4_splat(A2),lane_off);
  v128_t Z0s=wasm_f32x4_splat(v0->z),Z1s=wasm_f32x4_splat(v1->z),Z2s=wasm_f32x4_splat(v2->z);

  /* always grid-snapped (see sp_draw_triangle) — output-invariant + ztile-safe */
  int tyStart = (miny/TILE)*TILE;
  int txStart = (minx/TILE)*TILE;
  for (int tyB=tyStart; tyB<maxy; tyB+=TILE) {
    int ty = tyB<miny?miny:tyB;
    int tyend=tyB+TILE; if(tyend>maxy)tyend=maxy;
    if(ty>=tyend) continue;
    float ylo=(float)ty+0.5f, yhi=(float)(tyend-1)+0.5f;
    for (int txB=txStart; txB<maxx; txB+=TILE) {
      int tx = txB<minx?minx:txB;
      int txend=txB+TILE; if(txend>maxx)txend=maxx;
      if(tx>=txend) continue;
      float xlo=(float)tx+0.5f, xhi=(float)(txend-1)+0.5f;
      float mx0=A0*(x0pos0?xhi:xlo)+B0*(y0pos0?yhi:ylo)+C0; if(mx0<0.0f)continue;
      float mx1=A1*(x0pos1?xhi:xlo)+B1*(y0pos1?yhi:ylo)+C1; if(mx1<0.0f)continue;
      float mx2=A2*(x0pos2?xhi:xlo)+B2*(y0pos2?yhi:ylo)+C2; if(mx2<0.0f)continue;
      float mn0=A0*(x0pos0?xlo:xhi)+B0*(y0pos0?ylo:yhi)+C0;
      float mn1=A1*(x0pos1?xlo:xhi)+B1*(y0pos1?ylo:yhi)+C1;
      float mn2=A2*(x0pos2?xlo:xhi)+B2*(y0pos2?ylo:yhi)+C2;
      int tile_full=(mn0>=0.0f&&mn1>=0.0f&&mn2>=0.0f);
      int tileIdx=(tyB/TILE)*tilesW+(txB/TILE);
      float zmnT=Az*(zxpos?xlo:xhi)+Bz*(zypos?ylo:yhi)+Cz;
      if(do_ztile && zmnT>ztileBuf[tileIdx]) continue;
      int tileWrote=0; v128_t tileZmaxAcc=wasm_f32x4_splat(-1.0e30f);
      float fminx=(float)tx+0.5f;
      v128_t txmax4=wasm_f32x4_splat((float)txend);

      for (int y=ty; y<tyend; y++) {
        float yc=(float)y+0.5f;
        float row0=A0*fminx+B0*yc+C0, row1=A1*fminx+B1*yc+C1, row2=A2*fminx+B2*yc+C2;
        v128_t w0=wasm_f32x4_add(wasm_f32x4_splat(row0),A0l);
        v128_t w1=wasm_f32x4_add(wasm_f32x4_splat(row1),A1l);
        v128_t w2=wasm_f32x4_add(wasm_f32x4_splat(row2),A2l);
        for (int x=tx; x<txend; x+=4,
             w0=wasm_f32x4_add(w0,sX0),w1=wasm_f32x4_add(w1,sX1),w2=wasm_f32x4_add(w2,sX2)) {
          v128_t inside;
          if (tile_full) {
            v128_t xabs=wasm_f32x4_add(wasm_f32x4_splat((float)x),lane_off);
            inside=wasm_f32x4_lt(xabs,txmax4);
          } else {
            inside=wasm_v128_and(wasm_v128_and(wasm_f32x4_ge(w0,zero4),wasm_f32x4_ge(w1,zero4)),wasm_f32x4_ge(w2,zero4));
            v128_t xabs=wasm_f32x4_add(wasm_f32x4_splat((float)x),lane_off);
            inside=wasm_v128_and(inside,wasm_f32x4_lt(xabs,txmax4));
            if(!wasm_v128_any_true(inside)) continue;
          }
          v128_t z=wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,Z0s),wasm_f32x4_mul(w1,Z1s)),wasm_f32x4_mul(w2,Z2s));
          int base=y*W+x;
          v128_t pass=inside;
          if (do_depth) {
            v128_t zbuf=wasm_v128_load(depthBuf+base);
            pass=wasm_v128_and(pass,wasm_f32x4_lt(z,zbuf));
            if(!wasm_v128_any_true(pass)) continue;
          }
          /* interpolate the 6 varyings (persp-divide if needed) */
          v128_t recip=one4;
          if (persp) {
            v128_t iw=wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(iw0)),wasm_f32x4_mul(w1,wasm_f32x4_splat(iw1))),wasm_f32x4_mul(w2,wasm_f32x4_splat(iw2)));
            recip=wasm_f32x4_div(one4,iw);
          }
          #define GB_INTERP(A_,B_,C_) wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(w0,wasm_f32x4_splat(A_)),wasm_f32x4_mul(w1,wasm_f32x4_splat(B_))),wasm_f32x4_mul(w2,wasm_f32x4_splat(C_))),recip)
          v128_t U=GB_INTERP(u0,u1,u2), V=GB_INTERP(vv0,vv1,vv2);
          v128_t R=GB_INTERP(r0,r1,r2), G=GB_INTERP(g0,g1,g2);
          v128_t B=GB_INTERP(b0,b1,b2), A=GB_INTERP(a0,a1,a2);
          #undef GB_INTERP
          /* masked plane writes: keep existing where !pass (read-modify-write) */
          #define GB_STORE(PLANE,VAL) wasm_v128_store(PLANE+base, wasm_v128_bitselect(VAL, wasm_v128_load(PLANE+base), pass))
          GB_STORE(gu,U); GB_STORE(gv,V); GB_STORE(gr,R); GB_STORE(gg,G); GB_STORE(gb,B); GB_STORE(ga,A);
          #undef GB_STORE
          if (do_depth) {
            wasm_v128_store(depthBuf+base, wasm_v128_bitselect(z, wasm_v128_load(depthBuf+base), pass));
            tileZmaxAcc = wasm_f32x4_max(tileZmaxAcc, wasm_v128_bitselect(z, wasm_f32x4_splat(-1.0e30f), pass));
          }
          /* cover byte per passing lane */
          uint32_t pm=(uint32_t)wasm_i32x4_bitmask(pass);
          int rem = txend - x; if (rem > 4) rem = 4;
          int wl=0;
          for (int l=0;l<rem;l++) if (pm&(1u<<l)) { gc[base+l]=1; wl++; }
          tileWrote += wl;
        }
      }
      /* safe Zmax update: full UNCLIPPED cell, every pixel written (see sp_draw_triangle) */
      if (do_ztile && do_depth
          && tx==txB && (txend-txB)==TILE && ty==tyB && (tyend-tyB)==TILE
          && tileWrote==TILE*TILE) {
        float m0=wasm_f32x4_extract_lane(tileZmaxAcc,0),m1=wasm_f32x4_extract_lane(tileZmaxAcc,1);
        float m2=wasm_f32x4_extract_lane(tileZmaxAcc,2),m3=wasm_f32x4_extract_lane(tileZmaxAcc,3);
        float zmxT=m0>m1?m0:m1; float t2=m2>m3?m2:m3; zmxT=zmxT>t2?zmxT:t2;
        if(zmxT<ztileBuf[tileIdx]) ztileBuf[tileIdx]=zmxT;
      }
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
