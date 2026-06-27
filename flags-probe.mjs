import { createRenderer, FLAGS, VERTEX_STRIDE } from './lib/index.mjs';
import { makeScene, STRIDE } from './bench/compete/scene.mjs';
const W=512,H=512,FRAMES=120,WARMUP=150;
const now=()=>Number(process.hrtime.bigint())/1e6, median=a=>{const s=[...a].sort((x,y)=>x-y);return s[s.length>>1];};
function toSP(sc,n3){const out=new Float32Array(n3*VERTEX_STRIDE);for(let i=0;i<n3;i++){const s=i*STRIDE,o=i*VERTEX_STRIDE;out[o]=sc[s];out[o+1]=sc[s+1];out[o+2]=sc[s+2];out[o+3]=1;out[o+4]=sc[s+3];out[o+5]=sc[s+4];out[o+6]=sc[s+5];out[o+7]=1;out[o+8]=sc[s+6];out[o+9]=sc[s+7];}return out;}
const r=await createRenderer({width:W,height:H});const M=r._module;
function run(label,flags){r.setFlags(flags);const sc=makeScene('balanced',2000,W,H);const spbuf=toSP(sc,2000*3);const floats=2000*3*VERTEX_STRIDE;const ptr=r.alloc(floats*4);M.HEAPF32.set(spbuf.subarray(0,floats),ptr>>2);const draw=()=>{r.clear(0xff180f10>>>0,1.0);M._sp_draw_triangles_flat(ptr,2000);};for(let i=0;i<WARMUP;i++)draw();const t=[];for(let i=0;i<FRAMES;i++){const a=now();draw();t.push(now()-a);}r.free(ptr);console.log(`${label.padEnd(34)} ${median(t).toFixed(3)} ms`);}
run('depth+persp (ship)', 1|2);
run('depth only (affine, ship path)', 1);
run('NO depth (affine, all shade)', 0);
