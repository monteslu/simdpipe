#!/usr/bin/env bash
# Build simdpipe WASM.
#   dist/simdpipe.mjs           — single-thread, SIMD128 (portable everywhere)
#   dist/simdpipe-threads.mjs   — SIMD128 + pthreads (needs shared memory)
#
# -DSP_FAST enables the speed-over-fidelity vertex-color path (incremental z-stepping
# + *255-folded color pack): ~3-13% faster, output within ≤1 LSB of the byte-exact
# path (same coverage % and mean RGB; ~3-5 px of a 512² frame differ by ±1, invisible
# — the same tolerance already accepted for JIT textures). This IS the project thesis
# (trade graphics fidelity for speed). Drop -DSP_FAST for the byte-exact reference path.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../dist
mkdir -p "$OUT"

COMMON_EXPORTS='["_sp_init","_sp_alloc","_sp_free","_sp_color_ptr","_sp_depth_ptr","_sp_width","_sp_height","_sp_set_flags","_sp_get_flags","_sp_set_tile","_sp_get_tile","_sp_set_adapt_tile","_sp_bind_texture","_sp_reset_stats","_sp_stat_frag_tested","_sp_stat_frag_shaded","_sp_stat_tris","_sp_clear","_sp_draw_triangle","_sp_draw_triangles_flat","_sp_gbuffer_init","_sp_gbuffer_set_varymask","_sp_gbuffer_clear","_sp_draw_gbuffer_flat","_sp_gb_u","_sp_gb_v","_sp_gb_r","_sp_gb_g","_sp_gb_b","_sp_gb_a","_sp_gb_cover","_malloc","_free"]'

echo ">> single-thread build"
emcc simdpipe.c \
  -O3 -msimd128 -ffast-math -DSP_FAST \
  -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s ENVIRONMENT=node,web,worker \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32","wasmMemory","cwrap","ccall"]' \
  -s EXPORTED_FUNCTIONS="$COMMON_EXPORTS" \
  -o "$OUT/simdpipe.mjs"

echo ">> threaded build"
# Threaded: fixed memory (no growth with threads is simpler/faster), shared.
# NOTE: coarse-depth (ztile) is gated at RUNTIME on whole-screen ownership
# (tl_clip covers [0,h)), so the serial path and the pool's small-frame serial
# fallback get the win while banded pool workers (which share one g_ctx.ztile and
# would race) skip it. No compile switch needed; pooled-vs-serial stays
# bit-identical. Per-band private ztile (to give banded workers coarse-depth too)
# is the tracked follow-up.
emcc simdpipe.c \
  -O3 -msimd128 -ffast-math -DSP_FAST -pthread -DSP_THREADS=1 \
  -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s ENVIRONMENT=node,web,worker \
  -s INITIAL_MEMORY=268435456 -s MAXIMUM_MEMORY=268435456 \
  -s PTHREAD_POOL_SIZE=24 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32","cwrap","ccall"]' \
  -s EXPORTED_FUNCTIONS='["_sp_init","_sp_alloc","_sp_free","_sp_color_ptr","_sp_depth_ptr","_sp_width","_sp_height","_sp_set_flags","_sp_get_flags","_sp_set_tile","_sp_get_tile","_sp_set_adapt_tile","_sp_bind_texture","_sp_reset_stats","_sp_stat_frag_tested","_sp_stat_frag_shaded","_sp_stat_tris","_sp_clear","_sp_draw_triangle","_sp_draw_triangles_flat","_sp_draw_triangles_threaded","_sp_pool_start","_sp_pool_stop","_sp_draw_triangles_pooled","_sp_draw_gbuffer_pooled","_sp_gbuffer_init","_sp_gbuffer_set_varymask","_sp_gbuffer_clear","_sp_draw_gbuffer_flat","_sp_gb_u","_sp_gb_v","_sp_gb_r","_sp_gb_g","_sp_gb_b","_sp_gb_a","_sp_gb_cover","_malloc","_free"]' \
  -o "$OUT/simdpipe-threads.mjs"

echo "built:"
ls -la "$OUT"
