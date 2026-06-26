#!/usr/bin/env bash
# Build simdpipe WASM.
#   dist/simdpipe.mjs           — single-thread, SIMD128 (portable everywhere)
#   dist/simdpipe-threads.mjs   — SIMD128 + pthreads (needs shared memory)
set -euo pipefail
cd "$(dirname "$0")"
OUT=../dist
mkdir -p "$OUT"

COMMON_EXPORTS='["_sp_init","_sp_alloc","_sp_free","_sp_color_ptr","_sp_depth_ptr","_sp_width","_sp_height","_sp_set_flags","_sp_get_flags","_sp_bind_texture","_sp_reset_stats","_sp_stat_frag_tested","_sp_stat_frag_shaded","_sp_stat_tris","_sp_clear","_sp_draw_triangle","_sp_draw_triangles_flat","_sp_gbuffer_init","_sp_gbuffer_clear","_sp_draw_gbuffer_flat","_sp_gb_u","_sp_gb_v","_sp_gb_r","_sp_gb_g","_sp_gb_b","_sp_gb_a","_sp_gb_cover","_malloc","_free"]'

echo ">> single-thread build"
emcc simdpipe.c \
  -O3 -msimd128 -ffast-math \
  -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s ENVIRONMENT=node,web,worker \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32","wasmMemory","cwrap","ccall"]' \
  -s EXPORTED_FUNCTIONS="$COMMON_EXPORTS" \
  -o "$OUT/simdpipe.mjs"

echo ">> threaded build"
# Threaded: fixed memory (no growth with threads is simpler/faster), shared.
# NOTE: -DSP_NO_ZTILE disables the coarse per-tile Zmax depth-rejection grid in
# the threaded build. ztile is a single shared g_ctx buffer; with band-parallel
# workers a triangle spanning bands has its tile slices updated by different
# threads, and the cross-thread read/update ordering diverges from the serial
# reference (pooled-vs-serial was bit-identical before ztile, breaks with it).
# The optimization is single-thread-only for now; making it thread-safe (per-band
# private ztile rows, no cross-band reads) is tracked as a follow-up. The pooled
# path's whole point is multicore throughput, where coarse-depth matters less.
emcc simdpipe.c \
  -O3 -msimd128 -ffast-math -pthread -DSP_THREADS=1 -DSP_NO_ZTILE \
  -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s ENVIRONMENT=node,web,worker \
  -s INITIAL_MEMORY=268435456 -s MAXIMUM_MEMORY=268435456 \
  -s PTHREAD_POOL_SIZE=24 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32","cwrap","ccall"]' \
  -s EXPORTED_FUNCTIONS='["_sp_init","_sp_alloc","_sp_free","_sp_color_ptr","_sp_depth_ptr","_sp_width","_sp_height","_sp_set_flags","_sp_get_flags","_sp_bind_texture","_sp_reset_stats","_sp_stat_frag_tested","_sp_stat_frag_shaded","_sp_stat_tris","_sp_clear","_sp_draw_triangle","_sp_draw_triangles_flat","_sp_draw_triangles_threaded","_sp_pool_start","_sp_pool_stop","_sp_draw_triangles_pooled","_sp_draw_gbuffer_pooled","_sp_gbuffer_init","_sp_gbuffer_clear","_sp_draw_gbuffer_flat","_sp_gb_u","_sp_gb_v","_sp_gb_r","_sp_gb_g","_sp_gb_b","_sp_gb_a","_sp_gb_cover","_malloc","_free"]' \
  -o "$OUT/simdpipe-threads.mjs"

echo "built:"
ls -la "$OUT"
