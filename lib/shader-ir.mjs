/**
 * simdpipe shader IR — a tiny expression language for fragment shaders.
 *
 * Real shaders, not hand-built kernels: a GLSL-ish source is parsed into this IR,
 * which two backends consume —
 *   - lib/shader-js.mjs   : evaluate the IR per-pixel in JS (Tier-2)
 *   - lib/shader-wasm.mjs : emit a SIMD WASM kernel from the IR (Tier-1 JIT)
 *
 * Scope (deliberately small, covers the bounded common shader set):
 *   types: float, vec2, vec3, vec4   (component-wise; SoA-friendly)
 *   inputs:  varyings  uv (vec2), color (vec4 from r,g,b,a planes)
 *   uniforms: declared `uniform float NAME;` / `uniform vec3 NAME;` ...
 *   builtins: texture(uv) -> vec4 (bound texture), sin cos abs floor fract
 *             min max clamp mix step  (+ * - / on matching/scalar types)
 *   output:  gl_FragColor = <vec4 expr>;
 *
 * IR node = { op, ... }. Everything is component arrays of length 1..4 at eval.
 */

// ---- node constructors ----
export const N = {
  num: (v) => ({ op: 'num', v }),
  vec: (parts) => ({ op: 'vec', parts }),        // parts: IR nodes, flattened to components
  varying: (name) => ({ op: 'varying', name }),  // 'uv' | 'color'
  uniform: (name) => ({ op: 'uniform', name }),
  tex: (uv) => ({ op: 'tex', uv }),
  swizzle: (e, sw) => ({ op: 'swizzle', e, sw }), // sw: string like 'xyz','rgb','x'
  bin: (o, a, b) => ({ op: 'bin', o, a, b }),     // o: + - * /
  call: (fn, args) => ({ op: 'call', fn, args }),
  neg: (e) => ({ op: 'neg', e }),
};

// swizzle channel → index
const CH = { x: 0, y: 1, z: 2, w: 3, r: 0, g: 1, b: 2, a: 3, s: 0, t: 1, p: 2, q: 3 };
export function swizzleIndices(sw) { return [...sw].map((c) => CH[c]); }

// builtins with arity (for validation); component-wise unless noted
export const BUILTINS = {
  sin: 1, cos: 1, abs: 1, floor: 1, fract: 1, sqrt: 1,
  min: 2, max: 2, step: 2, mod: 2,
  clamp: 3, mix: 3,
  texture: 1, // special: vec2 -> vec4
};

/* ---------------- a small GLSL-ish parser ----------------
 * Supports: `precision ...;` (ignored), `uniform <type> name;`,
 * `varying`/`in` decls (ignored — uv & color are implicit), and a final
 * `void main(){ ... gl_FragColor = EXPR; }` (we only read the gl_FragColor RHS;
 * simple `Type name = EXPR;` locals are inlined). Expression grammar handles
 * + - * /, unary -, calls, swizzles, vecN(...) constructors, numbers, parens.
 */
export function parseShader(src) {
  // strip comments
  src = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');

  const uniforms = {}; // name -> type ('float'|'vec2'|'vec3'|'vec4')
  // collect uniform declarations
  for (const m of src.matchAll(/uniform\s+(float|vec2|vec3|vec4)\s+([A-Za-z_]\w*)\s*;/g)) {
    uniforms[m[2]] = m[1];
  }

  // grab main body
  const mainM = src.match(/void\s+main\s*\(\s*\)\s*\{([\s\S]*)\}/);
  if (!mainM) throw new Error('shader: no void main(){...}');
  let body = mainM[1];

  // collect simple locals: `Type name = EXPR;` (no control flow in this subset)
  const locals = {}; // name -> source expr string
  const stmts = splitStatements(body);
  let fragExpr = null;
  for (const s of stmts) {
    const decl = s.match(/^\s*(float|vec2|vec3|vec4)\s+([A-Za-z_]\w*)\s*=\s*([\s\S]+)$/);
    const assignFrag = s.match(/^\s*gl_FragColor\s*=\s*([\s\S]+)$/);
    if (assignFrag) { fragExpr = assignFrag[1]; continue; }
    if (decl) { locals[decl[2]] = decl[3]; continue; }
    // ignore unknown statements (e.g. precision in body)
  }
  if (fragExpr == null) throw new Error('shader: no gl_FragColor assignment');

  // parse the fragColor expression, inlining locals
  const parser = new ExprParser(uniforms, locals);
  const ast = parser.parseExprString(fragExpr);
  return { uniforms, ast };
}

function splitStatements(body) {
  // split on ';' at paren depth 0
  const out = []; let depth = 0, cur = '';
  for (const ch of body) {
    if (ch === '(') depth++; else if (ch === ')') depth--;
    if (ch === ';' && depth === 0) { if (cur.trim()) out.push(cur); cur = ''; }
    else cur += ch;
  }
  if (cur.trim()) out.push(cur);
  return out;
}

class ExprParser {
  constructor(uniforms, locals) { this.uniforms = uniforms; this.locals = locals; this.inlining = new Set(); }

  parseExprString(str) {
    this.toks = tokenize(str); this.i = 0;
    const e = this.parseAdd();
    if (this.i !== this.toks.length) throw new Error('shader: trailing tokens near ' + JSON.stringify(this.toks[this.i]));
    return e;
  }
  peek() { return this.toks[this.i]; }
  next() { return this.toks[this.i++]; }
  expect(t) { const x = this.next(); if (x !== t) throw new Error(`shader: expected '${t}' got '${x}'`); }

  parseAdd() {
    let a = this.parseMul();
    while (this.peek() === '+' || this.peek() === '-') { const o = this.next(); a = N.bin(o, a, this.parseMul()); }
    return a;
  }
  parseMul() {
    let a = this.parseUnary();
    while (this.peek() === '*' || this.peek() === '/') { const o = this.next(); a = N.bin(o, a, this.parseUnary()); }
    return a;
  }
  parseUnary() {
    if (this.peek() === '-') { this.next(); return N.neg(this.parseUnary()); }
    if (this.peek() === '+') { this.next(); return this.parseUnary(); }
    return this.parsePostfix();
  }
  parsePostfix() {
    let e = this.parsePrimary();
    while (this.peek() === '.') { this.next(); const sw = this.next(); if (!/^[xyzwrgbastpq]+$/.test(sw)) throw new Error('shader: bad swizzle .' + sw); e = N.swizzle(e, sw); }
    return e;
  }
  parsePrimary() {
    const t = this.peek();
    if (t === '(') { this.next(); const e = this.parseAdd(); this.expect(')'); return e; }
    if (/^[0-9]/.test(t) || (t === '.' )) { this.next(); return N.num(parseFloat(t)); }
    if (/^[A-Za-z_]\w*$/.test(t)) {
      this.next();
      if (this.peek() === '(') {
        // function call or vecN constructor
        this.next();
        const args = [];
        if (this.peek() !== ')') { args.push(this.parseAdd()); while (this.peek() === ',') { this.next(); args.push(this.parseAdd()); } }
        this.expect(')');
        if (/^vec[234]$/.test(t)) return N.vec(args);
        if (t === 'texture' || t === 'texture2D') return N.tex(args[0]);
        if (BUILTINS[t]) return N.call(t, args);
        throw new Error('shader: unknown function ' + t);
      }
      // identifier: uniform, varying (uv/color), local, or builtin const
      if (t === 'uv') return N.varying('uv');
      if (t === 'color' || t === 'vColor' || t === 'vcolor') return N.varying('color');
      if (this.uniforms[t]) return N.uniform(t);
      if (this.locals[t]) {
        if (this.inlining.has(t)) throw new Error('shader: recursive local ' + t);
        this.inlining.add(t);
        const sub = new ExprParser(this.uniforms, this.locals);
        sub.inlining = this.inlining;
        const e = sub.parseExprString(this.locals[t]);
        this.inlining.delete(t);
        return e;
      }
      throw new Error('shader: unknown identifier ' + t);
    }
    throw new Error('shader: unexpected token ' + JSON.stringify(t));
  }
}

function tokenize(s) {
  const out = []; let i = 0;
  const num = /[0-9.]/, id = /[A-Za-z_0-9]/;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\n' || c === '\t' || c === '\r') { i++; continue; }
    if ('()+-*/,.'.includes(c)) {
      // '.' could be decimal point if preceded by digit-start; handle numbers below
      if (c === '.' && i + 1 < s.length && /[0-9]/.test(s[i + 1]) && (out.length === 0 || !/^[A-Za-z_0-9)]/.test(out[out.length - 1].slice(-1)))) {
        let j = i + 1; while (j < s.length && num.test(s[j])) j++; out.push(s.slice(i, j)); i = j; continue;
      }
      out.push(c); i++; continue;
    }
    if (/[0-9]/.test(c)) { let j = i; while (j < s.length && num.test(s[j])) j++; out.push(s.slice(i, j)); i = j; continue; }
    if (id.test(c)) { let j = i; while (j < s.length && id.test(s[j])) j++; out.push(s.slice(i, j)); i = j; continue; }
    throw new Error('shader: bad char ' + JSON.stringify(c));
  }
  return out;
}
