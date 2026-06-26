/**
 * Export a workload scene to the binary format native-raster.c reads:
 *   int32 W, int32 H, int32 ntris, then ntris*3*8 float32 [x y z r g b u v].
 * Usage: node export-scene.mjs <kind> <ntris> <W> <H> <out.bin> [px]
 */
import { writeFileSync } from 'node:fs';
import { makeScene } from './scene.mjs';

const [kind, ntrisS, wS, hS, out, pxS] = process.argv.slice(2);
const ntris = +ntrisS, W = +wS, H = +hS;
const scene = makeScene(kind, ntris, W, H, { px: pxS ? +pxS : undefined });

const header = new Int32Array([W, H, ntris]);
const buf = Buffer.concat([Buffer.from(header.buffer), Buffer.from(scene.buffer)]);
writeFileSync(out, buf);
console.log(`wrote ${out}: ${kind} ${ntris} tris @ ${W}x${H} (${buf.length} bytes)`);
