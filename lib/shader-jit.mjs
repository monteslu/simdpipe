/**
 * Tier-1 JIT Program — compiles GLSL→WASM and runs the generated SIMD kernel
 * over the G-buffer planes (in the renderer's own linear memory) instead of a JS
 * callback. texture() sampling JITs too (in-kernel gather). Falls back to the JS
 * backend only for source the parser/codegen can't handle yet (never for a
 * shader that's pure calculation).
 *
 * A JIT program is created lazily against a specific renderer Module (so the
 * generated kernel imports the SAME WebAssembly.Memory — zero copy). Use
 * renderer.createJITProgram(src).
 */
import { compileWASM } from './shader-wasm.mjs';
import { compileJS } from './shader-js.mjs';

/**
 * @param {object} Module emscripten module (provides .wasmMemory / HEAP + ptrs)
 * @param {string} src GLSL-ish source
 * @param {{bilinear?:boolean}} [opts] texture filter for texture() taps
 * @returns {{fragShade:Function, jit:boolean, reason?:string}}
 */
export function makeJITProgram(Module, src, opts = {}) {
  const res = compileWASM(src, opts);
  if (!res.supported) {
    const js = compileJS(src);
    return { ...js, jit: false, reason: res.reason };
  }

  // Instantiate the generated module against the renderer's memory.
  const mod = new WebAssembly.Module(res.bytes);
  const memory = Module.wasmMemory || (Module.HEAPU8 && { buffer: Module.HEAPU8.buffer });
  const inst = new WebAssembly.Instance(mod, { env: { mem: memory } });
  const run = inst.exports.run;
  const uniformOrder = res.uniformOrder;
  const uniforms = res.uniforms;
  const needsTexture = !!res.needsTexture;
  const compsOf = (t) => ({ float: 1, vec2: 2, vec3: 3, vec4: 4 }[t]);

  // fragShade ignores the JS plane views; it calls the native kernel directly
  // over the plane POINTERS, which the renderer passes via planes._ptrs.
  function fragShade(planes, color, n, uni = {}) {
    const p = planes._ptrs;
    if (!p) throw new Error('JIT program requires renderer.drawProgram (needs plane pointers)');
    const n4 = (n / 4) | 0;
    // flatten uniform args in declared order
    const args = [p.u, p.v, p.r, p.g, p.b, p.a, p.cover, p.color, n4];
    for (const name of uniformOrder) {
      const c = compsOf(uniforms[name]);
      const val = uni[name];
      if (c === 1) args.push(typeof val === 'number' ? val : (Array.isArray(val) ? val[0] : 0));
      else { const arr = Array.isArray(val) ? val : [val]; for (let i = 0; i < c; i++) args.push(arr[i] ?? 0); }
    }
    // texture params (must match the kernel's 5 trailing i32s): base ptr, w, h,
    // wmask, hmask. The kernel gathers texels straight from linear memory — no JS.
    if (needsTexture) {
      const base = p.tex >>> 0, tw = p.texW >>> 0, th = p.texH >>> 0;
      if (!base || !tw || !th) {
        throw new Error('JIT texture shader: no texture bound (call renderer.bindTexture before drawProgram)');
      }
      args.push(base, tw, th, tw - 1, th - 1);
    }
    run(...args);
    // tail (n not multiple of 4) — kernel processes floor(n/4)*4; cover-masked,
    // remaining <4 px are rare (framebuffer width is usually a multiple of 4).
  }

  return { fragShade, jit: true, needsTexture, needsUV: !!res.needsUV, needsColor: !!res.needsColor };
}
