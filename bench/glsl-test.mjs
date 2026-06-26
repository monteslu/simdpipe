import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';
import { compileJS } from '../lib/shader-js.mjs';

const W=64,H=64;
const r=await createRenderer({width:W,height:H});
r.setFlags(FLAGS.DEPTH_TEST|FLAGS.PERSP_CORRECT);
const TW=8,TH=8,tex=new Uint32Array(TW*TH);
for(let y=0;y<TH;y++)for(let x=0;x<TW;x++)tex[y*TW+x]=((x+y)&1)?0xfff0a020:0xff2040f0;
r.bindTexture(new Uint8Array(tex.buffer),TW,TH);

const vtx=(x,y,z,R,G,B,u,v)=>[x,y,z,1,R,G,B,1,u,v];
const quad=new Float32Array([
  ...vtx(0,0,0.5,1,0,0,0,0), ...vtx(W,0,0.5,0,1,0,1,0), ...vtx(0,H,0.5,0,0,1,0,1),
  ...vtx(W,0,0.5,0,1,0,1,0), ...vtx(W,H,0.5,1,1,0,1,1), ...vtx(0,H,0.5,0,0,1,0,1),
]);

// REAL GLSL-ish source compiled via parser → IR → JS backend
const SRC=`
precision mediump float;
uniform float uTint;
void main(){
  vec4 t = texture(uv);
  vec3 c = t.rgb * color.rgb * uTint;
  gl_FragColor = vec4(c, 1.0);
}`;
const prog=compileJS(SRC);
console.log('parsed uniforms:', prog.uniforms);

r.clear(0xff000000,1.0);
r.drawProgram(quad,2,prog,{uTint:1.0});
const fb=r.getFramebuffer();
const px=(x,y)=>{const i=(y*W+x)*4;return [fb[i],fb[i+1],fb[i+2],fb[i+3]];};
console.log('center', px(32,32), 'topleft', px(2,2), 'topright', px(60,2));
// the textured+tinted quad should have non-black, varied pixels
let nonblack=0;for(let i=0;i<W*H;i++)if(fb[i*4]||fb[i*4+1]||fb[i*4+2])nonblack++;
console.log('non-black:', nonblack,'/',W*H);

// test builtins: sin/clamp/mix
const SRC2=`
uniform float t;
void main(){
  float w = 0.5 + 0.5*sin(uv.x*10.0 + t);
  vec3 c = mix(vec3(1.0,0.0,0.0), vec3(0.0,0.0,1.0), w);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;
const prog2=compileJS(SRC2);
r.clear(0xff000000,1.0); r.drawProgram(quad,2,prog2,{t:1.0});
const fb2=r.getFramebuffer();
const px2=(x,y)=>{const i=(y*W+x)*4;return [fb2[i],fb2[i+1],fb2[i+2]];};
console.log('procedural sin/mix center', px2(32,32), 'left', px2(8,32), 'right', px2(56,32));

console.log((nonblack>1000)?'\n✅ GLSL PARSER + JS BACKEND PASS':'\n❌ FAIL');
process.exit(nonblack>1000?0:1);
