// Precise shade-bound A/B + correctness harness, run inside the worktree.
import { createRenderer, VERTEX_STRIDE } from './lib/index.mjs';
import { makeScene, STRIDE } from './bench/compete/scene.mjs';
import { fbStats } from './bench/compete/png.mjs';
import { compileJS } from './lib/shader-js.mjs';
const W=512,H=512,FRAMES=120,WARMUP=150;
const now=()=>Number(process.hrtime.bigint())/1e6, median=a=>{const s=[...a].sort((x,y)=>x-y);return s[s.length>>1];};
function toSP(sc,n3){const out=new Float32Array(n3*VERTEX_STRIDE);for(let i=0;i<n3;i++){const s=i*STRIDE,o=i*VERTEX_STRIDE;out[o]=sc[s];out[o+1]=sc[s+1];out[o+2]=sc[s+2];out[o+3]=1;out[o+4]=sc[s+3];out[o+5]=sc[s+4];out[o+6]=sc[s+5];out[o+7]=1;out[o+8]=sc[s+6];out[o+9]=sc[s+7];}return out;}
const r=await createRenderer({width:W,height:H});
// The competition shade shader: color.xyz, 3 sins, heavy frag.
const SRC=`uniform float uT;
void main(){
  float wr = 0.5 + 0.5*sin(color.x*12.0 + uT);
  float wg = 0.5 + 0.5*sin(color.y*12.0 + uT*1.3);
  float wb = 0.5 + 0.5*sin(color.z*12.0 + uT*0.7);
  vec3 c = mix(color.rgb, vec3(wr, wg, wb), 0.5);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;
const jit=r.createJITProgram(SRC);
const js=compileJS(SRC);
console.log('jit:',jit.jit,'needsUV:',jit.needsUV,'needsColor:',jit.needsColor);
const scene=makeScene('balanced',2000,W,H);const spbuf=toSP(scene,2000*3);
// correctness: JIT vs JS oracle
r.clear(0xff180f10,1.0); r.drawProgram(spbuf,2000,jit,{uT:0.7}); const jitFB=r.getFramebuffer().slice();
r.clear(0xff180f10,1.0); r.drawProgram(spbuf,2000,js,{uT:0.7}); const jsFB=r.getFramebuffer().slice();
let maxd=0,nd=0; for(let i=0;i<jitFB.length;i++){const d=Math.abs(jitFB[i]-jsFB[i]); if(d){nd++; if(d>maxd)maxd=d;}}
console.log(`JIT vs JS oracle: ${nd} bytes differ (${(100*nd/jitFB.length).toFixed(2)}%), maxΔ=${maxd}  ${maxd<=1?'OK':'*** BROKEN ***'}`);
// timing (JIT)
const draw=()=>{r.clear(0xff180f10,1.0);r.drawProgram(spbuf,2000,jit,{uT:0.7});};
for(let i=0;i<WARMUP;i++)draw();const t=[];for(let i=0;i<FRAMES;i++){const a=now();draw();t.push(now()-a);}
draw();const s=fbStats(r.getFramebuffer(),W,H);
console.log(`shade-bound JIT: ${median(t).toFixed(3)} ms  cov=${s.coverage.toFixed(2)}% rgb=[${s.meanRGB.join(',')}] hash=${s.hash}`);
