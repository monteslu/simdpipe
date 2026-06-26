/**
 * Minimal WebAssembly binary encoder — just enough to emit fragment-shader
 * functions at runtime, hand them to the host engine (V8 in Node, or a browser
 * engine), and let IT compile them to native machine code.
 *
 * This is the core of simdpipe's Tier-1 "JIT": we don't make memory executable
 * ourselves (the sandbox forbids that) — we generate WASM bytes and the engine
 * compiles them. Proven viable (Mesa ACO compiles shaders in-browser this way).
 *
 * Scope: f32 math, locals, linear-memory load/store, a tiny opcode set. SIMD
 * emission is a later step; this proves the generate→compile→call→whole-tile
 * pipeline first with scalar f32 over a tile.
 *
 * Spec refs: WebAssembly core binary format. We emit a single module with one
 * imported memory and one exported function.
 */

// ---- LEB128 ----
function uleb(n) {
  const out = [];
  do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; out.push(b); } while (n);
  return out;
}
function sleb(n) {
  const out = []; let more = true;
  while (more) {
    let b = n & 0x7f; n >>= 7;
    if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) more = false;
    else b |= 0x80;
    out.push(b);
  }
  return out;
}
function f32bytes(x) {
  const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, x, true);
  return [...b];
}

// ---- value types & opcodes ----
export const T = { i32: 0x7f, f32: 0x7d };
export const OP = {
  block: 0x02, loop: 0x03, br: 0x0c, br_if: 0x0d, end: 0x0b, ret: 0x0f,
  call: 0x10, local_get: 0x20, local_set: 0x21, local_tee: 0x22,
  i32_load: 0x28, f32_load: 0x2a, i32_store: 0x36, f32_store: 0x38,
  i32_const: 0x41, f32_const: 0x43,
  i32_eqz: 0x45, i32_eq: 0x46, i32_ne: 0x47, i32_lt_s: 0x48, i32_ge_s: 0x4e,
  i32_add: 0x6a, i32_sub: 0x6b, i32_mul: 0x6c, i32_and: 0x71, i32_or: 0x72, i32_shl: 0x74,
  f32_eq: 0x5b, f32_lt: 0x5d, f32_gt: 0x5e, f32_le: 0x5f, f32_ge: 0x60,
  f32_add: 0x92, f32_sub: 0x93, f32_mul: 0x94, f32_div: 0x95,
  f32_min: 0x96, f32_max: 0x97, f32_sqrt: 0x91, f32_abs: 0x8b, f32_neg: 0x8c,
  i32_trunc_f32_s: 0xa8, f32_convert_i32_s: 0xb2,
};

/* SIMD (v128) ops use the 0xFD prefix + a uleb opcode. We store them as
 * [0xfd, ...uleb(code)] and a helper expands them. */
export const SIMD = {
  v128_load: 0x00, v128_store: 0x0b,
  v128_const: 0x0c, v128_and: 0x4e, v128_or: 0x50, v128_bitselect: 0x52,
  i32x4_splat: 0x11, f32x4_splat: 0x13,
  f32x4_add: 0xe4, f32x4_sub: 0xe5, f32x4_mul: 0xe6, f32x4_div: 0xe7,
  f32x4_min: 0xe8, f32x4_max: 0xe9, f32x4_sqrt: 0xe3,
  f32x4_ge: 0x46, f32x4_lt: 0x43, f32x4_gt: 0x44, f32x4_le: 0x45,
  i32x4_add: 0xae, i32x4_shl: 0xab, i32x4_trunc_sat_f32x4_s: 0xf8,
  i32x4_mul: 0xb5,
};
/** Emit a SIMD instruction: prefix 0xFD + uleb(code) + optional immediates. */
export function simd(code, ...imm) { return [0xfd, ...uleb(code), ...imm]; }

function section(id, payload) { return [id, ...uleb(payload.length), ...payload]; }
function vec(items) { return [...uleb(items.length), ...items.flat()]; }

/**
 * Build a module exporting one function `run(args...)` with the given param/result
 * types and body bytecode, importing memory "env"."mem".
 * @param {{params:number[], results:number[], locals:number[], body:number[], exportName?:string}} fn
 * @returns {Uint8Array}
 */
export function buildModule(fn) {
  const exportName = fn.exportName || 'run';
  const magic = [0x00, 0x61, 0x73, 0x6d];
  const version = [0x01, 0x00, 0x00, 0x00];

  // Type section: one functype
  const functype = [0x60, ...vec(fn.params.map(t => [t])), ...vec(fn.results.map(t => [t]))];
  const typeSec = section(1, vec([functype]));

  // Import section: memory env.mem  (limits: min pages, flag 0 = no max)
  const enc = (s) => [...uleb(s.length), ...[...s].map(c => c.charCodeAt(0))];
  const memImport = [...enc('env'), ...enc('mem'), 0x02 /*mem*/, 0x00 /*flags*/, ...uleb(1)];
  const importSec = section(2, vec([memImport]));

  // Function section: function 0 has type 0
  const funcSec = section(3, vec([[0]]));

  // Export section: export func 0 as exportName
  const exportSec = section(7, vec([[...enc(exportName), 0x00 /*func*/, ...uleb(0)]]));

  // Code section: locals + body
  // group locals by type (we just emit them individually for simplicity)
  const localDecls = vec(fn.locals.map(t => [...uleb(1), t]));
  const code = [...localDecls, ...fn.body, OP.end];
  const codeEntry = [...uleb(code.length), ...code];
  const codeSec = section(10, vec([codeEntry]));

  return new Uint8Array([
    ...magic, ...version,
    ...typeSec, ...importSec, ...funcSec, ...exportSec, ...codeSec,
  ]);
}

// expose encoders for shader codegen
export const E = { uleb, sleb, f32bytes };
