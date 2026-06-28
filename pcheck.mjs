import createThreads from './dist/simdpipe-threads.mjs';
import createSingle from './dist/simdpipe.mjs';
import { makeScene, STRIDE } from './bench/compete/scene.mjs';
import { fbStats } from './bench/compete/png.mjs';
const VS=10,W=512,H=512;
function toSP(sc,n3){const out=new Float32Array(n3*VS);for(let i=0;i<n3;i++){const s=i*STRIDE,o=i*VS;out[o]=sc[s];out[o+1]=sc[s+1];out[o+2]=sc[s+2];out[o+3]=1;out[o+4]=sc[s+3];out[o+5]=sc[s+4];out[o+6]=sc[s+5];out[o+7]=1;out[o+8]=sc[s+6];out[o+9]=sc[s+7];}return out;}
const S=await createSingle();S._sp_init(W,H);S._sp_set_flags(1|2);
const M=await createThreads();M._sp_init(W,H);M._sp_set_flags(1|2);M._sp_pool_start(8);
const sfb=()=>new Uint8Array(S.HEAPU8.buffer,S._sp_color_ptr(),W*H*4), mfb=()=>new Uint8Array(M.HEAPU8.buffer,M._sp_color_ptr(),W*H*4);
for(const [k,n] of [['fill',200],['balanced',2000],['dense',16000]]){
  const kk=k==='dense'?'balanced':k; const sc=makeScene(kk,n,W,H);const sp=toSP(sc,n*3);const f=n*3*VS;
  const sp_=S._sp_alloc(f*4);S.HEAPF32.set(sp.subarray(0,f),sp_>>2);S._sp_clear(0xff180f10>>>0,1.0);S._sp_draw_triangles_flat(sp_,n);const sh=fbStats(sfb(),W,H).hash;S._sp_free(sp_);
  const mp=M._sp_alloc(f*4);M.HEAPF32.set(sp.subarray(0,f),mp>>2);for(let i=0;i<10;i++){M._sp_clear(0xff180f10>>>0,1.0);M._sp_draw_triangles_pooled(mp,n);}const mh=fbStats(mfb(),W,H).hash;M._sp_free(mp);
  console.log(`${k}: serial=${sh} pooled=${mh} ${sh===mh?'MATCH':'DIFF'}`);
}
M._sp_pool_stop();
