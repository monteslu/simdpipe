#!/usr/bin/env bash
# Build simdpipe WASM. Phase 0: SIMD128, single-thread. Threads added later.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../dist
mkdir -p "$OUT"

# Standalone-ish: export functions, no full Emscripten runtime bloat. We use
# MODULARIZE + EXPORT_ES6 so the .mjs is importable in Node and the browser.
emcc simdpipe.c \
  -O3 -msimd128 \
  -ffast-math \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s ENVIRONMENT=node,web,worker \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=67108864 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPU32","HEAPF32","cwrap","ccall"]' \
  -s EXPORTED_FUNCTIONS='["_sp_init","_sp_alloc","_sp_free","_sp_color_ptr","_sp_depth_ptr","_sp_width","_sp_height","_sp_set_flags","_sp_get_flags","_sp_bind_texture","_sp_reset_stats","_sp_stat_frag_tested","_sp_stat_frag_shaded","_sp_stat_tris","_sp_clear","_sp_draw_triangle","_sp_draw_triangles_flat","_malloc","_free"]' \
  -o "$OUT/simdpipe.mjs"

echo "built -> $OUT/simdpipe.mjs (+ .wasm)"
ls -la "$OUT"
