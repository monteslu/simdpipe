/**
 * Minimal RGBA8 → PNG encoder (zlib deflate, no deps). Used to dump every
 * renderer's framebuffer so output is VISUALLY verifiable — a benchmark where
 * one renderer secretly draws nothing is meaningless, so we screenshot them all.
 */
import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';

function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xEDB88320 & -(c & 1));
  }
  return ~c >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const t = Buffer.from(type, 'ascii');
  const body = Buffer.concat([t, data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

/**
 * @param {string} path
 * @param {Uint8Array} rgba  length w*h*4, row-major top-to-bottom
 * @param {number} w @param {number} h
 * @param {boolean} [flipY] flip vertically (GL readback is bottom-to-top)
 */
export function writePNG(path, rgba, w, h, flipY = false) {
  const stride = w * 4;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const srcY = flipY ? (h - 1 - y) : y;
    raw[y * (stride + 1)] = 0; // filter: none
    Buffer.from(rgba.buffer, rgba.byteOffset + srcY * stride, stride).copy(raw, y * (stride + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0; // 8-bit RGBA
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  writeFileSync(path, png);
}

/** Cheap fingerprint of a framebuffer: coverage %, mean RGB, simple hash. Lets us
 *  assert two renderers produced *similar* images without pixel-exact match. */
export function fbStats(rgba, w, h, flipY = false) {
  const n = w * h;
  let covered = 0, sr = 0, sg = 0, sb = 0, hash = 0;
  const bg = [15, 15, 24]; // clear color 0x18,0x0f,0x10-ish (0.06*255≈15)
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    if (Math.abs(r - bg[0]) > 6 || Math.abs(g - bg[1]) > 6 || Math.abs(b - bg[2]) > 6) covered++;
    sr += r; sg += g; sb += b;
    hash = (hash * 31 + r + g * 7 + b * 13) >>> 0;
  }
  return {
    coverage: +(100 * covered / n).toFixed(1),
    meanRGB: [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)],
    hash,
  };
}
