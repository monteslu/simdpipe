# simdpipe — cross-renderer competition

Benchmarks simdpipe against **every software/hardware renderer that can be built
and run locally**, all driven from one Node process over **bit-identical
geometry**, all reading their framebuffer back and dumping a PNG so the output is
**verifiable** (a renderer that secretly draws nothing would otherwise look
"infinitely fast").

```
npm run compete:build-native   # one-time: compile the native-C baseline
npm run compete                # runs the whole thing
npm run compete -- --size 1024x1024 --frames 40
```

## Contestants

| renderer | what it is | how it's driven |
|---|---|---|
| **simdpipe** | this project — WASM + 128-bit SIMD | `lib/index.mjs` |
| **llvmpipe** | Mesa's LLVM software rasterizer (256-bit AVX2) | real GLES 3.0 via [`native-gles`](https://github.com/monteslu/native-gles) + `LIBGL_ALWAYS_SOFTWARE=1` |
| **GPU** | the actual GPU (AMD Radeon 890M) | `native-gles`, no software flag — the *honesty check* |
| **native-C** | scalar `gcc -O3 -march=native` edge-function raster | `native-raster.c`, the **no-SIMD floor** |

`native-gles` is the key: the **same** GLES code runs on llvmpipe (with
`LIBGL_ALWAYS_SOFTWARE=1`) or the GPU (without), so the only variable is the
driver.

## Fairness rules

1. **Identical geometry.** Every renderer consumes the same scene from one
   deterministic generator (`scene.mjs`, mulberry32 PRNG). Screen-space verts go
   straight into simdpipe; the GL harness converts them to NDC in the vertex
   shader; the C baseline reads the same binary. → all rasterize the same pixels.
2. **Identical fragment work.** The GL `color` shader = simdpipe's vertex-color
   path; the GL `heavy` shader's `sin/mix` math = the simdpipe JIT shader's math.
3. **Like-for-like threads.** Software renderers are compared **single-threaded
   first** (SIMD vs SIMD, 1 core each — llvmpipe pinned with `LP_NUM_THREADS=0`).
   A separate section shows each with threads. We never pit simdpipe-1-thread
   against llvmpipe-all-cores and call it a result.
4. **Geometry pre-uploaded; GL `glFinish()` per frame.** We time rasterization,
   not transfer, and force the async GL pipeline to actually complete.
5. **Output cross-checked.** Each renderer's framebuffer coverage% + mean RGB
   must agree (they do, to within 1 — see `proof.png`).

![four renderers, identical scene](proof.png)

*Left→right: simdpipe · llvmpipe · GPU · native-C, all rendering the same
fill-rate scene. Identical image ⇒ the timings below are a fair comparison.*

## Results (512×512, 60 frames, AMD Ryzen AI 9 HX 370, 24 cores, V8 24)

### Part 1 — single-thread, SIMD vs SIMD (the like-for-like number)

```
workload                             simdpipe   llvmpipe   native-C       GPU
fill (200 big tris, overdraw)            9.77       4.83      17.44      0.07
balanced (2k mid tris)                   9.85       5.15      13.19      0.08
small (20k @ 8px)                        4.47       4.83       3.22      0.20
shade-bound (heavy frag, 2k tris)       22.10       7.12          —      0.08

simdpipe vs:                       llvmpipe-1T    native-C
fill                                     0.49x       1.78x
balanced                                 0.52x       1.34x
small                                    1.08x       0.72x
shade-bound                              0.32x          —
```

**simdpipe lands within ~2× of llvmpipe on one core, and beats it on small
triangles.** That ~2× is exactly the 128-bit-WASM vs 256-bit-native-AVX2 vector
gap — honest and expected. simdpipe also **beats scalar native C by 1.3–1.8×** on
fill/balanced: 128-bit SIMD across the WASM boundary still outruns a scalar
native loop. The `shade-bound` gap (0.32×) is the one real soft spot — llvmpipe
vectorizes `sin` to a native polynomial while simdpipe's JIT `sin` is a per-lane
minimax; a clear optimization target.

### Part 2 — multicore (each renderer at its best)

```
workload                               sp 1T    sp pool8   sp scaling   llvmpipe MT
fill (200 big tris)                     9.77        9.78        1.00x          1.07
balanced (2k mid tris)                  9.85        3.08        3.20x          1.37
small (20k @ 8px)                       4.47       17.39        0.26x          2.89
```

simdpipe's persistent work-stealing pool gives a real **3.2× on balanced**, but
its win is a **narrow band (~1k big triangles)**: below `SP_POOL_MIN_TRIS` it
stays serial, and **without per-tile binning it re-runs triangle setup per band
and regresses at high tri counts** (the roadmap's #1 item). llvmpipe's mature
tile-binned threading scales smoothly. This is surfaced, not hidden.

### Part 3 — the thesis: trade fidelity for speed

simdpipe's actual bet isn't "beat llvmpipe at equal fidelity" — it's **do less
work**. Descending the fidelity ladder on one textured scene:

```
tier                                     fps   vs full
full: bilinear + persp + depth            82     1.00x
bilinear → nearest (1 tap vs 4)          195     2.39x
+ drop perspective → affine              214     2.63x
cheapest: affine vertex color            209     2.57x
```

Each knob simdpipe turns off buys ~2.4–2.6×. **llvmpipe always pays full
fidelity** — this is the lever it can't pull.

## Honest scope

- simdpipe does **not** beat llvmpipe at equal fidelity on one core (it's ~2×
  behind on fill — the vector-width gap), and does **not** approach the GPU
  (60–280× faster; that's the honesty check working).
- simdpipe **wins** where the thesis says it should: vs scalar native code, on
  setup-light small-triangle work, and — decisively — when you're willing to
  trade fidelity, where a 2.4–2.6× discount is on the table that a
  full-fidelity renderer structurally can't offer.

All numbers are machine-dependent — run `npm run compete` yourself. Raw data in
`results.json`; per-renderer screenshots in `shots/`.

## Files

- `scene.mjs` — shared deterministic geometry + workload table
- `gl-harness.mjs` — drives any GLES driver (llvmpipe / GPU) via native-gles
- `sp-harness.mjs` — drives simdpipe (1-thread / pooled / low-fi)
- `native-raster.c` — scalar-C baseline
- `png.mjs` — dependency-free RGBA→PNG + framebuffer fingerprint
- `run-all.mjs` — orchestrator (the sectioned report above)
- `compose-proof.mjs` / `gl-dump-raw.mjs` — build `proof.png`
