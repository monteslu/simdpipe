/** Minimal column-major 4x4 matrix math (GL convention). */
export function identity() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); }

export function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

export function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  const o = new Float32Array(16);
  o[0] = f / aspect; o[5] = f; o[10] = (far + near) * nf; o[11] = -1; o[14] = 2 * far * near * nf;
  return o;
}

export function translate(m, x, y, z) {
  const t = identity(); t[12] = x; t[13] = y; t[14] = z; return multiply(m, t);
}

export function rotateY(m, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const r = identity(); r[0] = c; r[2] = -s; r[8] = s; r[10] = c; return multiply(m, r);
}
export function rotateX(m, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const r = identity(); r[5] = c; r[6] = s; r[9] = -s; r[10] = c; return multiply(m, r);
}

/** transform a vec4 (x,y,z,1) by column-major m → [x,y,z,w] */
export function transformPoint(m, x, y, z) {
  return [
    m[0] * x + m[4] * y + m[8]  * z + m[12],
    m[1] * x + m[5] * y + m[9]  * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}
