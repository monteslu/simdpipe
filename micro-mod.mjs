// usage: node micro-mod.mjs <module-path>
import { makeScene, STRIDE } from '/home/monteslu/code/cliemu/simdpipe/bench/compete/scene.mjs';
import { fbStats } from '/home/monteslu/code/cliemu/simdpipe/bench/compete/png.mjs';
const VERTEX_STRIDE=10,W=512,H=512,FRAMES=100,WARMUP=120;
const now=()=>Number(process.hrtime.bigint())/1e6, median=a=>{const s=[...a].sort((x,y)=>x-y);return s[s.length>>1];};
function toSP(sc,n3){const out=new Float32Array(n3*VERTEX_STRIDE);for(let i=0;i<n3;i++){const s=i*STRIDE,o=i*VERTEX_STRIDE;out[o]=sc[s];out[o+1]=sc[s+1];out[o+2]=sc[s+2];out[o+3]=1;out[o+4]=sc[s+3];out[o+5]=sc[s+4];out[o+6]=sc[s+5];out[o+7]=1;out[o+8]=sc[s+6];out[o+9]=sc[s+7];}return out;}
const createModule = (await import(process.argv[2])).default;
const M = await createModule(); M._sp_init(W,H); M._sp_set_flags(1|2);
const fb=()=>new Uint8Array(M.HEAPU8.buffer,M._sp_color_ptr(),W*H*4);
const cases=[['fill','fill',200],['balanced',  'balanced',2000],['small','small',20000],['dense','balanced',16000]];
for (const [lbl,kind,nt] of cases){
  const sc=makeScene(kind,nt,W,H,{px:kind==='small'?8:undefined});const spbuf=toSP(sc,nt*3);const floats=nt*3*VERTEX_STRIDE;
  const ptr=M._sp_alloc(floats*4);M.HEAPF32.set(spbuf.subarray(0,floats),ptr>>2);
  const draw=()=>{M._sp_clear(0xff180f10>>>0,1.0);M._sp_draw_triangles_flat(ptr,nt);};
  for(let i=0;i<WARMUP;i++)draw();const t=[];for(let i=0;i<FRAMES;i++){const a=now();draw();t.push(now()-a);}
  draw();const s=fbStats(fb(),W,H);M._sp_free(ptr);
  console.log(`${lbl.padEnd(10)} ${median(t).toFixed(3).padStart(8)} ms  cov=${s.coverage.toFixed(2)}% hash=${s.hash}`);
}
