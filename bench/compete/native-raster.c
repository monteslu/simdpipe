/*
 * Native scalar-C edge-function rasterizer — the "no-SIMD floor" baseline.
 * Same algorithm as simdpipe's core (Pineda edge functions, z-buffer, Gouraud
 * vertex color), compiled with gcc -O3 -march=native, BUT scalar (one pixel at a
 * time, no intrinsics). This isolates what 128-bit SIMD + the WASM/native gap buy.
 *
 * Reads the same scene the JS harnesses use from a binary file:
 *   int32 W, int32 H, int32 ntris, then ntris*3 * 8 floats [x y z r g b u v].
 * Renders FRAMES times, prints median ms + a framebuffer fingerprint, and writes
 * a raw RGBA file for screenshot verification.
 *
 * Build: gcc -O3 -march=native -o native-raster native-raster.c -lm
 * Run:   ./native-raster <scene.bin> <frames> <warmup> [out.rgba]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static int W, H, NTRIS;
static float *VERTS;        // ntris*3*8
static uint32_t *FB;        // W*H RGBA8
static float *ZB;           // W*H depth

static inline float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static void render_frame(void) {
  // clear
  for (int i = 0; i < W * H; i++) { FB[i] = 0xff180f10u; ZB[i] = 1.0f; }

  for (int t = 0; t < NTRIS; t++) {
    const float *v = VERTS + t * 3 * 8;
    float x0 = v[0], y0 = v[1], z0 = v[2];
    float x1 = v[8], y1 = v[9], z1 = v[10];
    float x2 = v[16], y2 = v[17], z2 = v[18];
    float r0 = v[3], g0 = v[4], b0 = v[5];
    float r1 = v[11], g1 = v[12], b1 = v[13];
    float r2 = v[19], g2 = v[20], b2 = v[21];

    float area = edge(x0, y0, x1, y1, x2, y2);
    if (area == 0.0f) continue;
    float inv_area = 1.0f / area;

    int minx = (int)floorf(fminf(fminf(x0, x1), x2));
    int maxx = (int)ceilf(fmaxf(fmaxf(x0, x1), x2));
    int miny = (int)floorf(fminf(fminf(y0, y1), y2));
    int maxy = (int)ceilf(fmaxf(fmaxf(y0, y1), y2));
    if (minx < 0) minx = 0; if (miny < 0) miny = 0;
    if (maxx > W) maxx = W; if (maxy > H) maxy = H;

    for (int y = miny; y < maxy; y++) {
      for (int x = minx; x < maxx; x++) {
        float px = x + 0.5f, py = y + 0.5f;
        float w0 = edge(x1, y1, x2, y2, px, py) * inv_area;
        float w1 = edge(x2, y2, x0, y0, px, py) * inv_area;
        float w2 = edge(x0, y0, x1, y1, px, py) * inv_area;
        // inside test: all weights same sign as area (>=0 after inv_area normalize)
        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
        float z = w0 * z0 + w1 * z1 + w2 * z2;
        int idx = y * W + x;
        if (z >= ZB[idx]) continue; // LESS depth test
        ZB[idx] = z;
        float r = w0 * r0 + w1 * r1 + w2 * r2;
        float g = w0 * g0 + w1 * g1 + w2 * g2;
        float b = w0 * b0 + w1 * b1 + w2 * b2;
        uint32_t R = (uint32_t)(r * 255.0f + 0.5f); if (R > 255) R = 255;
        uint32_t G = (uint32_t)(g * 255.0f + 0.5f); if (G > 255) G = 255;
        uint32_t B = (uint32_t)(b * 255.0f + 0.5f); if (B > 255) B = 255;
        FB[idx] = 0xff000000u | (B << 16) | (G << 8) | R;
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s scene.bin frames warmup [out.rgba]\n", argv[0]); return 1; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("open scene"); return 1; }
  int hdr[3];
  if (fread(hdr, sizeof(int), 3, f) != 3) { fprintf(stderr, "bad header\n"); return 1; }
  W = hdr[0]; H = hdr[1]; NTRIS = hdr[2];
  size_t nfloats = (size_t)NTRIS * 3 * 8;
  VERTS = malloc(nfloats * sizeof(float));
  if (fread(VERTS, sizeof(float), nfloats, f) != nfloats) { fprintf(stderr, "bad verts\n"); return 1; }
  fclose(f);

  FB = malloc((size_t)W * H * 4);
  ZB = malloc((size_t)W * H * sizeof(float));

  int frames = atoi(argv[2]), warmup = atoi(argv[3]);
  for (int i = 0; i < warmup; i++) render_frame();
  double *times = malloc(frames * sizeof(double));
  for (int i = 0; i < frames; i++) { double a = now_ms(); render_frame(); times[i] = now_ms() - a; }
  qsort(times, frames, sizeof(double), cmp_double);
  double med = times[frames / 2];

  // fingerprint (match png.mjs fbStats: coverage vs bg, mean RGB)
  long covered = 0, sr = 0, sg = 0, sb = 0;
  int bgR = 16, bgG = 15, bgB = 24; // 0x10,0x0f,0x18
  for (int i = 0; i < W * H; i++) {
    uint32_t p = FB[i];
    int R = p & 0xff, G = (p >> 8) & 0xff, B = (p >> 16) & 0xff;
    if (abs(R - bgR) > 6 || abs(G - bgG) > 6 || abs(B - bgB) > 6) covered++;
    sr += R; sg += G; sb += B;
  }
  long n = (long)W * H;
  printf("{\"renderer\":\"native scalar C (gcc -O3)\",\"ms\":%.3f,\"coverage\":%.1f,\"meanRGB\":[%ld,%ld,%ld]}\n",
         med, 100.0 * covered / n, sr / n, sg / n, sb / n);

  if (argc >= 5) {
    FILE *o = fopen(argv[4], "wb");
    if (o) { fwrite(FB, 4, (size_t)W * H, o); fclose(o); }
  }
  return 0;
}
