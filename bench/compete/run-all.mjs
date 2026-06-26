/**
 * Cross-renderer competition orchestrator — honest, sectioned report.
 *
 * Contestants (all read bit-identical geometry, all read back + dump a PNG):
 *   simdpipe   — WASM + 128-bit SIMD (this project)
 *   llvmpipe   — Mesa LLVM software rasterizer (256-bit AVX2) via native-gles
 *   GPU        — AMD Radeon 890M via native-gles            [honesty check]
 *   native-C   — scalar gcc -O3 -march=native edge raster   [the no-SIMD floor]
 *
 * FAIRNESS: software rasterizers are compared single-threaded first (SIMD vs
 * SIMD, 1 core each — llvmpipe pinned via LP_NUM_THREADS=0), THEN a separate
 * multicore section shows each renderer's threading at its best. We never compare
 * simdpipe-1-thread against llvmpipe-all-cores and call it a result.
 *
 * Usage: node bench/compete/run-all.mjs [--size WxH] [--frames N] [--warmup N]
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import os from 'node:os';

const __dir = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SIZE = getArg('--size', '512x512');
const FRAMES = getArg('--frames', '60');
const WARMUP = getArg('--warmup', '20');
const [W, H] = SIZE.split('x').map(Number);
const NCORES = os.cpus().length;
const POOL = Math.min(8, NCORES); // simdpipe pool size (matches its bench default)

const SHOTS = join(__dir, 'shots');
if (!existsSync(SHOTS)) mkdirSync(SHOTS, { recursive: true });
const PFX = join(SHOTS, 'cmp');
const node = process.execPath;

function runJSON(args, env) {
  const out = execFileSync(node, args, { env: { ...process.env, ...env }, encoding: 'utf8', maxBuffer: 64 << 20 });
  return JSON.parse(out.trim().split('\n').filter(Boolean).pop());
}
function runC(kind, ntris, px) {
  const bin = join(__dir, 'native-raster');
  if (!existsSync(bin)) return null;
  const sceneBin = join(os.tmpdir(), `cmp-${kind}-${W}x${H}.bin`);
  execFileSync(node, [join(__dir, 'export-scene.mjs'), kind, String(ntris), String(W), String(H), sceneBin, ...(px ? [String(px)] : [])]);
  const rgba = join(os.tmpdir(), `cmp-${kind}-native.rgba`);
  const out = execFileSync(bin, [sceneBin, FRAMES, WARMUP, rgba], { encoding: 'utf8' });
  return JSON.parse(out.trim().split('\n').pop());
}

const pad = (s, n) => String(s).padEnd(n);
const padr = (s, n) => String(s).padStart(n);
const ms = (x) => x == null ? '   —' : x.toFixed(2);

console.log(`\n${'='.repeat(82)}`);
console.log(`  simdpipe cross-renderer competition`);
console.log(`  ${SIZE}, ${FRAMES} frames (warmup ${WARMUP}) · node ${process.version} · ${NCORES} cores`);
console.log(`${'='.repeat(82)}`);

// ============================================================================
// PART 1 — single-thread, SIMD vs SIMD (the like-for-like core comparison)
// ============================================================================
process.stdout.write('\n[1/4] simdpipe (1 thread)...     ');
const sp1 = runJSON([join(__dir, 'sp-harness.mjs'), '--size', SIZE, '--frames', FRAMES, '--warmup', WARMUP, '--dump', PFX, '--tag', 'simdpipe']);
console.log('done');
process.stdout.write('[2/4] llvmpipe (1 thread)...     ');
const llvm1 = runJSON([join(__dir, 'gl-harness.mjs'), '--size', SIZE, '--frames', FRAMES, '--warmup', WARMUP, '--dump', PFX, '--tag', 'llvmpipe'],
  { LIBGL_ALWAYS_SOFTWARE: '1', GALLIUM_DRIVER: 'llvmpipe', LP_NUM_THREADS: '0' });
console.log('done →', llvm1.renderer);
process.stdout.write('[3/4] native scalar C...         ');
const nat = { fill: runC('fill', 200), balanced: runC('balanced', 2000), small: runC('small', 20000, 8) };
console.log(nat.fill ? 'done' : 'skipped (binary not built)');
process.stdout.write('[4/4] GPU (honesty check)...     ');
let gpu = null;
try { gpu = runJSON([join(__dir, 'gl-harness.mjs'), '--size', SIZE, '--frames', FRAMES, '--warmup', WARMUP, '--dump', PFX, '--tag', 'gpu'], {}); console.log('done →', gpu.renderer); }
catch (e) { console.log('skipped'); }

const byName = (a) => Object.fromEntries((a || []).map(r => [r.name, r]));
const sp1R = byName(sp1.results), llvm1R = byName(llvm1.results), gpuR = gpu ? byName(gpu.results) : {};
const WL = sp1.results.map(r => r.name);
const natKind = (n) => n.startsWith('fill') ? 'fill' : n.startsWith('balanced') ? 'balanced' : n.startsWith('small') ? 'small' : null;

console.log(`\n${'─'.repeat(82)}`);
console.log('PART 1 · SINGLE-THREAD — both software renderers on 1 core (SIMD vs SIMD)');
console.log('─'.repeat(82));
console.log('\nframe time (ms, median; lower=better):\n');
console.log(pad('workload', 34), padr('simdpipe', 10), padr('llvmpipe', 10), padr('native-C', 10), padr('GPU', 9));
console.log('-'.repeat(34 + 10 * 3 + 9 + 3));
for (const n of WL) {
  const nk = natKind(n);
  console.log(pad(n, 34), padr(ms(sp1R[n]?.ms), 10), padr(ms(llvm1R[n]?.ms), 10), padr(ms(nk ? nat[nk]?.ms : null), 10), padr(ms(gpuR[n]?.ms), 9));
}
console.log('\nsimdpipe vs llvmpipe (1T) and vs native-C (×, >1 = simdpipe faster):\n');
console.log(pad('workload', 34), padr('vs llvmpipe-1T', 16), padr('vs native-C', 14));
console.log('-'.repeat(34 + 16 + 14));
for (const n of WL) {
  const s = sp1R[n]?.ms, l = llvm1R[n]?.ms, nk = natKind(n), nm = nk ? nat[nk]?.ms : null;
  console.log(pad(n, 34), padr(l ? (l / s).toFixed(2) + 'x' : '—', 16), padr(nm ? (nm / s).toFixed(2) + 'x' : '—', 14));
}

// ============================================================================
// PART 2 — multicore: each renderer with threads (NOT apples-to-apples in count,
// but shows each at its best). simdpipe pool has a sweet spot (~1k big tris);
// below MIN_TRIS it falls back to serial, and without tile-binning it regresses
// on very high tri counts — we surface that honestly with a tri-count sweep.
// ============================================================================
console.log(`\n${'─'.repeat(82)}`);
console.log(`PART 2 · MULTICORE — simdpipe persistent pool (${POOL} threads) vs llvmpipe (all ${NCORES} cores)`);
console.log('─'.repeat(82));
process.stdout.write('\nsimdpipe pool + llvmpipe MT...   ');
const spP = runJSON([join(__dir, 'sp-harness.mjs'), '--size', SIZE, '--frames', FRAMES, '--warmup', WARMUP, '--threads', String(POOL)]);
const llvmMT = runJSON([join(__dir, 'gl-harness.mjs'), '--size', SIZE, '--frames', FRAMES, '--warmup', WARMUP],
  { LIBGL_ALWAYS_SOFTWARE: '1', GALLIUM_DRIVER: 'llvmpipe' });
console.log('done');
const spPR = byName(spP.results), llvmMTR = byName(llvmMT.results);
console.log('\nframe time (ms, median) — vertex-color workloads:\n');
console.log(pad('workload', 34), padr('sp 1T', 9), padr(`sp pool${POOL}`, 11), padr('sp scaling', 12), padr('llvmpipe MT', 13));
console.log('-'.repeat(34 + 9 + 11 + 12 + 13));
for (const n of WL) {
  if (!spPR[n]) continue; // shade-bound isn't pooled
  const s1 = sp1R[n]?.ms, sp = spPR[n]?.ms, lm = llvmMTR[n]?.ms;
  console.log(pad(n, 34), padr(ms(s1), 9), padr(ms(sp), 11), padr((s1 / sp).toFixed(2) + 'x', 12), padr(ms(lm), 13));
}
console.log('\n(simdpipe pool wins in a band ≈1k big tris; below MIN_TRIS it stays serial,');
console.log(' and without per-tile binning it regresses at very high tri counts — roadmap.)');

// ============================================================================
// PART 3 — the thesis: trade fidelity for speed (simdpipe's actual lever).
// Reuses bench/fidelity.mjs which descends the fidelity ladder on one scene.
// ============================================================================
console.log(`\n${'─'.repeat(82)}`);
console.log('PART 3 · THE THESIS — simdpipe trades fidelity for speed (llvmpipe always full-fi)');
console.log('─'.repeat(82));
try {
  const fOut = execFileSync(node, [join(__dir, '..', 'fidelity.mjs')], { encoding: 'utf8' });
  // print the ladder table lines
  const lines = fOut.split('\n').filter(l => /tier|full|nearest|texture|affine|→|vs full/.test(l));
  console.log('\n' + lines.join('\n'));
} catch (e) { console.log('(fidelity bench unavailable:', String(e.message).split('\n')[0], ')'); }

// ============================================================================
// Correctness — fingerprints must agree (proves identical scene drawn)
// ============================================================================
console.log(`\n${'─'.repeat(82)}`);
console.log('OUTPUT VERIFICATION — coverage% / meanRGB must match (proves same scene; PNGs saved)');
console.log('─'.repeat(82) + '\n');
console.log(pad('workload', 34), padr('simdpipe', 21), padr('llvmpipe', 21), padr('native-C', 21));
console.log('-'.repeat(34 + 21 * 3));
const fp = (r) => r ? `${r.coverage}% [${r.meanRGB.join(',')}]` : '—';
for (const n of WL) {
  const nk = natKind(n);
  console.log(pad(n, 34), padr(fp(sp1R[n]), 21), padr(fp(llvm1R[n]), 21), padr(nk ? fp(nat[nk]) : '(n/a)', 21));
}
console.log('\nPNG screenshots (every renderer × workload):', SHOTS);

// JSON
const outJSON = join(__dir, 'results.json');
writeFileSync(outJSON, JSON.stringify({
  size: { W, H }, frames: +FRAMES, node: process.version, cores: NCORES,
  renderers: { simdpipe: sp1.renderer, llvmpipe_1T: llvm1.renderer, llvmpipe_MT: llvmMT.renderer, gpu: gpu?.renderer, native: 'native scalar C (gcc -O3 -march=native)' },
  single_thread: WL.map(n => ({ name: n, simdpipe: sp1R[n]?.ms, llvmpipe: llvm1R[n]?.ms, native: nat[natKind(n)]?.ms, gpu: gpuR[n]?.ms })),
  multicore: WL.filter(n => spPR[n]).map(n => ({ name: n, sp_1t: sp1R[n]?.ms, sp_pool: spPR[n]?.ms, llvmpipe_mt: llvmMTR[n]?.ms })),
  fingerprints: Object.fromEntries(WL.map(n => [n, { simdpipe: sp1R[n], llvmpipe: llvm1R[n], native: nat[natKind(n)] }])),
}, null, 2));
console.log('wrote', outJSON);
