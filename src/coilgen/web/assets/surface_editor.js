/* Smooth CAD display and direct manipulation. Display triangles never enter the solver. */
(() => {
  'use strict';
  const defaults={cylinder:{radius:.08,height:.24},sphere:{radius:.06},box:{width:.16,depth:.16,height:.24},plane:{width:.16,height:.24}};
  const colors=[[.24,.55,.77],[.85,.50,.22],[.31,.65,.50],[.63,.46,.77],[.76,.59,.25],[.27,.64,.69]];
  const clone=x=>JSON.parse(JSON.stringify(x));
  const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,s)=>a.map(v=>v*s),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>mul(a,1/(Math.hypot(...a)||1));
  const roots=doc=>{const used=new Set(doc.nodes.filter(n=>n.kind==='subtract').flatMap(n=>[n.base,n.tool]));return doc.nodes.filter(n=>!used.has(n.id));};
  function rotation(v,r){let [x,y,z]=v;for(let a=0;a<3;a++){const c=Math.cos(r[a]),s=Math.sin(r[a]);if(a===0)[y,z]=[c*y-s*z,s*y+c*z];if(a===1)[x,z]=[c*x+s*z,-s*x+c*z];if(a===2)[x,y]=[c*x-s*y,s*x+c*y];}return [x,y,z];}
  function newPrimitive(kind,id){return{id,kind,name:kind[0].toUpperCase()+kind.slice(1)+' '+id.replace('object',''),parameters:{...defaults[kind]},center_m:[0,0,0],rotation_rad:[0,0,0]};}
  function primitivePreview(node){
    const vertices=[],triangles=[],face_indices=[],p=node.parameters,kind=node.kind;
    function triangle(a,b,c,f){const start=vertices.length;vertices.push(...[a,b,c].map(v=>add(rotation(v,node.rotation_rad),node.center_m)));triangles.push([start,start+1,start+2]);face_indices.push(f);}
    const n=48;
    if(kind==='sphere'){
      const point=(i,j)=>{const a=2*Math.PI*i/n,b=Math.PI*j/24;return[p.radius*Math.sin(b)*Math.cos(a),p.radius*Math.sin(b)*Math.sin(a),p.radius*Math.cos(b)];};
      for(let j=0;j<24;j++)for(let i=0;i<n;i++){if(j)triangle(point(i,j),point(i,j+1),point(i+1,j),0);if(j<23)triangle(point(i+1,j),point(i,j+1),point(i+1,j+1),0);}
    }else if(kind==='cylinder'){
      const point=(i,z)=>[p.radius*Math.cos(2*Math.PI*i/n),p.radius*Math.sin(2*Math.PI*i/n),z];
      const z=p.height/2;
      for(let i=0;i<n;i++){const a=point(i,-z),b=point(i+1,-z),c=point(i+1,z),d=point(i,z);triangle(a,b,c,0);triangle(a,c,d,0);triangle([0,0,z],d,c,1);triangle([0,0,-z],b,a,2);}
    }else if(kind==='plane'){
      const x=p.width/2,y=p.height/2;triangle([-x,-y,0],[x,-y,0],[x,y,0],0);triangle([-x,-y,0],[x,y,0],[-x,y,0],0);
    }else{
      const x=p.width/2,y=p.depth/2,z=p.height/2,pts=[[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]];
      [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]].forEach((f,i)=>{triangle(pts[f[0]],pts[f[1]],pts[f[2]],i);triangle(pts[f[0]],pts[f[2]],pts[f[3]],i);});
    }
    return{vertices,triangles,face_indices,faces:Array.from({length:6},(_,i)=>({key:'guide'+i,body:node.id,excluded:false}))};
  }
  function triangleHit(origin,direction,a,b,c){
    const e1=sub(b,a),e2=sub(c,a),h=cross(direction,e2),det=dot(e1,h);if(Math.abs(det)<1e-14)return null;
    const inv=1/det,s=sub(origin,a),u=dot(s,h)*inv;if(u<0||u>1)return null;
    const q=cross(s,e1),v=dot(direction,q)*inv;if(v<0||u+v>1)return null;
    const t=dot(e2,q)*inv;return t>0?t:null;
  }
  class SurfaceCanvas {
    constructor(canvas,change,select,feedback){
      this.canvas=canvas;this.change=change;this.select=select;this.feedback=feedback;
      this.yaw=-1.05;this.pitch=.48;this.scale=.2;this.center=[0,0,0];this.drag=null;this.mode='orbit';this.axis=0;this.selected='';this.blocked=false;
      this.gl=canvas.getContext('webgl',{antialias:true,alpha:false});
      if(!this.gl)throw Error('Geometry studio needs WebGL. Enable browser hardware acceleration.');
      const gl=this.gl;
      const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
      this.program=gl.createProgram();
      gl.attachShader(this.program,compile(gl.VERTEX_SHADER,'attribute vec3 position,normal;attribute vec4 color;uniform vec3 center,right,up,outward;uniform vec2 span;varying vec3 n;varying vec4 c;void main(){vec3 p=position-center;gl_Position=vec4(dot(p,right)/span.x,dot(p,up)/span.y,-dot(p,outward)/(span.y*20.0),1.0);n=normal;c=color;}'));
      gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 n;varying vec4 c;void main(){vec3 norm=normalize(n);if(!gl_FrontFacing)norm=-norm;float light=.48+.52*max(0.0,dot(norm,normalize(vec3(-.5,-.9,1.2))));gl_FragColor=vec4(c.rgb*light,c.a);}'));
      gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
      this.buffer=gl.createBuffer();gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
      canvas.addEventListener('pointerdown',e=>this.down(e));canvas.addEventListener('pointermove',e=>this.move(e));
      canvas.addEventListener('pointerup',()=>this.up());canvas.addEventListener('pointercancel',()=>this.cancel());
      canvas.addEventListener('wheel',e=>{e.preventDefault();this.scale=Math.max(1e-5,Math.min(100,this.scale*Math.exp(e.deltaY*.001)));this.draw();},{passive:false});
    }
    dispose(){this.resize.disconnect();this.gl.deleteBuffer(this.buffer);this.gl.deleteProgram(this.program);}
    camera(){const out=[Math.cos(this.yaw)*Math.cos(this.pitch),Math.sin(this.yaw)*Math.cos(this.pitch),Math.sin(this.pitch)],right=unit(cross([0,0,1],out));return{out,right,up:cross(out,right)};}
    fit(){const v=this.scene?.vertices;if(!v?.length)return;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of v)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}this.center=mul(add(lo,hi),.5);this.scale=Math.max(.001,Math.hypot(...sub(hi,lo))*.62);this.draw();}
    update(scene,doc,options){const first=!this.scene;this.scene=scene;this.doc=doc;Object.assign(this,options);if(first)this.fit();else this.draw();}
    geometry(){return this.showMesh&&this.scene?.mesh_preview?this.scene.mesh_preview:this.scene;}
    packets(data,alpha=1,guide=false){
      if(!data)return[];const normals=new Map(),packets=[];
      data.triangles.forEach((tri,i)=>{const fi=data.face_indices[i],f=data.faces[fi];if(f.excluded&&!this.showDeleted)return;const vs=tri.map(k=>data.vertices[k]),n=cross(sub(vs[1],vs[0]),sub(vs[2],vs[0]));for(const k of tri){const key=fi+':'+k;normals.set(key,add(normals.get(key)||[0,0,0],n));}});
      data.triangles.forEach((tri,i)=>{const fi=data.face_indices[i],face=data.faces[fi];if(face.excluded&&!this.showDeleted)return;const bodyIndex=Math.max(0,this.doc.nodes.findIndex(n=>n.id===face.body));let color=colors[bodyIndex%colors.length];if(face.key===this.face)color=[.95,.69,.18];let a=face.excluded?.16:alpha;
        const vertices=tri.map(k=>data.vertices[k]);
        tri.forEach((k,j)=>{const n=unit(normals.get(fi+':'+k)||[0,0,1]);packets.push(...vertices[j],...n,...color,a);});
        if(this.showMesh&&!guide){/* Wire overlay is drawn in a separate pass. */}
      });return packets;
    }
    draw(){
      if(!this.scene||!this.doc)return;
      const gl=this.gl,canvas=this.canvas,ratio=Math.min(window.devicePixelRatio||1,2),w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight);
      canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.953,.965,.977,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
      const c=this.camera();for(const [key,val] of Object.entries({center:this.center,right:c.right,up:c.up,outward:c.out}))gl.uniform3fv(gl.getUniformLocation(this.program,key),val);
      gl.uniform2f(gl.getUniformLocation(this.program,'span'),this.scale*w/h,this.scale);
      const data=this.geometry();let packets=this.packets(data,this.drag&&this.mode!=='orbit'?.28:1);
      const selected=this.doc.nodes.find(n=>n.id===this.selected);
      if(selected&&selected.kind!=='subtract'&&(this.drag&&this.mode!=='orbit'||!this.scene.roots.includes(selected.id)))packets.push(...this.packets(primitivePreview(selected),.45,true));
      gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(packets),gl.DYNAMIC_DRAW);
      for(const [name,size,offset] of [['position',3,0],['normal',3,12],['color',4,24]]){const a=gl.getAttribLocation(this.program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,40,offset);}
      gl.drawArrays(gl.TRIANGLES,0,packets.length/10);
      if(this.showMesh){const lines=[];data.triangles.forEach((tri,i)=>{if(data.faces[data.face_indices[i]].excluded)return;for(let j=0;j<3;j++)for(const k of [tri[j],tri[(j+1)%3]])lines.push(...data.vertices[k],0,0,1,.12,.18,.24,1);});gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(lines),gl.DYNAMIC_DRAW);gl.depthFunc(gl.LEQUAL);gl.drawArrays(gl.LINES,0,lines.length/10);gl.depthFunc(gl.LESS);}
    }
    pick(e){const rect=this.canvas.getBoundingClientRect(),c=this.camera(),u=((e.clientX-rect.left)/rect.width*2-1)*this.scale*rect.width/rect.height,v=(1-(e.clientY-rect.top)/rect.height*2)*this.scale,origin=add(this.center,add(mul(c.out,this.scale*20),add(mul(c.right,u),mul(c.up,v)))),direction=mul(c.out,-1),data=this.geometry();let nearest=Infinity,hit=null;
      data.triangles.forEach((tri,i)=>{const face=data.faces[data.face_indices[i]];if(face.excluded&&!this.showDeleted)return;const t=triangleHit(origin,direction,...tri.map(k=>data.vertices[k]));if(t!==null&&t<nearest){nearest=t;hit=face;}});return hit;
    }
    down(e){if(e.button!==0||this.blocked)return;
      if(this.mode==='face'){const hit=this.pick(e);this.select(hit?.key||'',true);return;}
      if(this.mode==='select'){const hit=this.pick(e);if(hit)this.select(hit.body,false);return;}
      const node=this.doc.nodes.find(n=>n.id===this.selected);if(this.mode!=='orbit'&&(!node||node.kind==='subtract')){this.feedback('Select a construction primitive in the object bar to move or rotate it.');return;}
      this.drag={x:e.clientX,y:e.clientY,yaw:this.yaw,pitch:this.pitch,node:node?clone(node):null,camera:this.camera(),changed:false};this.canvas.setPointerCapture(e.pointerId);
    }
    move(e){if(!this.drag)return;const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y,d=this.drag;
      if(Math.hypot(dx,dy)<2&&!d.changed)return;d.changed=true;
      if(this.mode==='orbit'){this.yaw=d.yaw-dx*.007;this.pitch=Math.max(-1.4,Math.min(1.4,d.pitch+dy*.007));this.draw();return;}
      const node=clone(d.node);
      if(this.mode==='rotate')node.rotation_rad[this.axis]=Math.max(-2*Math.PI,Math.min(2*Math.PI,d.node.rotation_rad[this.axis]+dx*.009));
      else{const a=d.camera.right[this.axis],b=-d.camera.up[this.axis],den=a*a+b*b,delta=(den>.04?(dx*a+dy*b)/den:dx)*(2*this.scale/this.canvas.clientHeight);node.center_m[this.axis]=Math.max(-10,Math.min(10,d.node.center_m[this.axis]+delta));}
      this.change(node,false);this.feedback('Release to rebuild the CAD result');
    }
    up(){if(!this.drag)return;const changed=this.drag.changed;this.drag=null;if(changed&&this.mode!=='orbit')this.change(null,true);else this.draw();}
    cancel(){if(!this.drag)return;const node=this.drag.node;this.drag=null;if(node&&this.mode!=='orbit')this.change(node,false);this.draw();}
  }
  const component={
    props:{scene:Object,busy:Boolean,error:String,status:String},emits:['action','close'],
    data(){return{doc:null,selected:'',face:'',mode:'select',axis:0,kind:'cylinder',base:'',tool:'',showDeleted:false,showMesh:false,pending:false,localError:'',hint:''};},
    computed:{blocked(){return this.busy||this.pending;},node(){return this.doc?.nodes.find(n=>n.id===this.selected);},outputs(){return this.doc?roots(this.doc):[];},faceInfo(){return this.scene?.faces.find(f=>f.key===this.face);},report(){return this.scene?.mesh_preview?.mesh_report;}},
    watch:{scene:{immediate:true,handler(value){if(value){this.doc=clone(value.model);this.pending=false;if(!this.doc.nodes.some(n=>n.id===this.selected))this.selected=this.doc.nodes[0].id;const out=roots(this.doc);if(!out.some(n=>n.id===this.base))this.base=out[0]?.id||'';if(!out.some(n=>n.id===this.tool))this.tool=out[1]?.id||'';if(!value.mesh_preview)this.showMesh=false;this.$nextTick(()=>this.redraw());}}},error(){this.pending=false;},selected(){this.face='';this.redraw();},mode(){this.redraw();},axis(){this.redraw();},showDeleted(){this.redraw();},showMesh(){this.redraw();}},
    mounted(){try{this.renderer=new SurfaceCanvas(this.$refs.canvas,(node,commit)=>{if(node){const i=this.doc.nodes.findIndex(n=>n.id===node.id);this.doc.nodes[i]=node;}if(commit)this.update();else this.redraw();},(key,isFace)=>{if(isFace)this.face=key;else this.selected=key;this.redraw();},text=>this.hint=text);this.redraw();}catch(e){this.localError=e.message;}},
    beforeUnmount(){this.renderer?.dispose();},
    methods:{
      redraw(){if(this.renderer&&this.scene&&this.doc){this.renderer.update(this.scene,this.doc,{selected:this.selected,face:this.face,mode:this.mode,axis:Number(this.axis),showDeleted:this.showDeleted,showMesh:this.showMesh,blocked:this.blocked});}},
      send(kind,extra={}){if(this.blocked)return;this.pending=true;this.localError='';this.hint='';this.redraw();this.$emit('action',{kind,identity:this.scene.identity,...extra});},
      update(){this.showMesh=false;this.face='';this.send('update',{model:clone(this.doc)});},
      nextId(prefix){let i=1;while(this.doc.nodes.some(n=>n.id===prefix+i))i++;return prefix+i;},
      add(){const node=newPrimitive(this.kind,this.nextId('object'));node.center_m=[.08,-.04,.04];this.doc.nodes.push(node);this.selected=node.id;this.update();},
      remove(){if(!this.node)return;if(this.doc.nodes.some(n=>n.base===this.selected||n.tool===this.selected)){this.localError='Remove the dependent subtraction first. Its original operands will reappear.';return;}if(this.doc.nodes.length===1){this.localError='Keep at least one object.';return;}this.doc.nodes=this.doc.nodes.filter(n=>n.id!==this.selected);this.update();},
      duplicate(){if(!this.node||this.node.kind==='subtract')return;const n=clone(this.node);n.id=this.nextId('object');n.name+=' copy';n.center_m[0]+=.02;this.doc.nodes.push(n);this.selected=n.id;this.update();},
      subtract(){if(!this.base||!this.tool||this.base===this.tool)return;const id=this.nextId('cut');this.doc.nodes.push({id,kind:'subtract',name:'Subtract '+id.replace('cut',''),base:this.base,tool:this.tool});this.selected=id;this.update();},
      deleteFace(){if(!this.face)return;if(!this.doc.excluded_faces.includes(this.face))this.doc.excluded_faces.push(this.face);this.update();},
      restoreFaces(){this.doc.excluded_faces=[];this.update();},
      number(event,field,key,factor){const input=event.target,value=input.valueAsNumber;if(!Number.isFinite(value)||!input.checkValidity()){this.localError='Enter a finite value within the displayed limits.';return;}this.node[field][key]=value/factor;this.update();},
      edge(event){const v=event.target.valueAsNumber;if(Number.isFinite(v)&&event.target.checkValidity()){this.doc.edge_length_m=v/1000;this.update();}},
      fmt(v){return Number(v.toFixed(3));},
      saveDraft(){const url=URL.createObjectURL(new Blob([JSON.stringify(this.doc,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='surface.cgsurface';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},
      async loadDraft(event){try{const file=event.target.files[0];if(!file)return;if(file.size>128000)throw Error('Surface recipe exceeds 128 kB.');const model=JSON.parse(await file.text());this.send('update',{model});}catch(e){this.localError=e.message;}event.target.value='';}
    },
    template:`<section class="cs-editor" aria-label="Geometry studio">
      <header><strong>Geometry studio</strong><span>Objects → Boolean operations → Retained faces → Mesh</span><button :disabled="blocked" @click="$emit('close')">Close</button></header>
      <div class="cs-toolbar"><label>Primitive<select v-model="kind" :disabled="blocked"><option v-for="k in ['cylinder','sphere','box','plane']" :value="k">{{k}}</option></select></label><button :disabled="blocked||!doc||doc.nodes.length>=24" @click="add">Add object</button><button v-for="m in ['select','move','rotate','orbit','face']" :aria-pressed="mode===m" :disabled="blocked" @click="mode=m">{{m==='face'?'Select face':m}}</button><label>Axis<select v-model="axis" :disabled="blocked"><option :value="0">X</option><option :value="1">Y</option><option :value="2">Z</option></select></label><button :disabled="blocked||!scene?.undo" @click="send('undo')">Undo</button><button :disabled="blocked||!scene?.redo" @click="send('redo')">Redo</button><button @click="renderer?.fit()">Fit view</button></div>
      <div class="cs-toolbar"><label>Body A<select v-model="base" :disabled="blocked"><option v-for="n in outputs" :value="n.id">{{n.name}}</option></select></label><label>Cutter B<select v-model="tool" :disabled="blocked"><option value="">Choose solid</option><option v-for="n in outputs.filter(n=>n.kind!=='plane')" :value="n.id">{{n.name}}</option></select></label><button :disabled="blocked||!base||!tool||base===tool" @click="subtract">Subtract A − B</button><button :disabled="blocked||!face||faceInfo?.excluded" @click="deleteFace">Delete face</button><button :disabled="blocked||!doc?.excluded_faces.length" @click="restoreFaces">Restore deleted faces</button><label><input type="checkbox" v-model="showDeleted">Show deleted faces</label><span>{{faceInfo?.label||''}}</span></div>
      <main class="cs-view"><canvas ref="canvas" aria-label="Smooth CAD surface. Select objects in the bar below; use numeric fields as an alternative to dragging."></canvas><div class="cs-view-note">{{showMesh?'Final surface mesh':'Smooth CAD preview'}} · {{mode==='face'?'Click a CAD face':mode==='orbit'?'Drag to orbit; scroll to zoom':mode==='select'?'Click an object or its bar entry':'Drag along the selected axis; values update below'}}</div></main>
      <div class="cs-objects" v-if="doc"><button v-for="(n,i) in doc.nodes" :key="n.id" :aria-pressed="selected===n.id" :disabled="blocked" @click="selected=n.id"><strong><i :style="{background:'rgb('+[...[[61,140,196],[217,128,56],[79,166,128],[161,117,196],[194,150,64],[69,163,176]][i%6]].join(',')+')'}"></i>{{n.name}}</strong><template v-if="n.kind!=='subtract'"><span>{{Object.entries(n.parameters).map(([k,v])=>k+' '+fmt(v*1000)).join(' · ')}} mm</span><span>XYZ {{n.center_m.map(v=>fmt(v*1000)).join(' / ')}} mm</span><span>Rotation {{n.rotation_rad.map(v=>fmt(v*180/Math.PI)).join(' / ')}}°</span></template><span v-else>{{n.base}} − {{n.tool}}</span><small>{{outputs.some(o=>o.id===n.id)?'Result object':'Construction operand'}}</small></button></div>
      <div class="cs-toolbar cs-parameters" v-if="node"><template v-if="node.kind!=='subtract'"><label v-for="(value,key) in node.parameters">{{key}} (mm)<input type="number" min="0.01" max="10000" step="any" :value="fmt(value*1000)" :disabled="blocked" @change="number($event,'parameters',key,1000)"></label><label v-for="(v,i) in node.center_m">{{['X','Y','Z'][i]}} (mm)<input type="number" min="-10000" max="10000" step="any" :value="fmt(v*1000)" :disabled="blocked" @change="number($event,'center_m',i,1000)"></label><label v-for="(v,i) in node.rotation_rad">R{{['x','y','z'][i]}} (°)<input type="number" min="-360" max="360" step="any" :value="fmt(v*180/Math.PI)" :disabled="blocked" @change="number($event,'rotation_rad',i,180/Math.PI)"></label><button :disabled="blocked||doc.nodes.length>=24" @click="duplicate">Duplicate</button></template><button :disabled="blocked" @click="remove">Remove object / operation</button><button :disabled="blocked" @click="saveDraft">Save recipe</button><label>Load recipe<input type="file" accept=".cgsurface,.json" :disabled="blocked" @change="loadDraft"></label></div>
      <div v-if="scene?.unresolved_faces.length" class="cs-error" role="alert">Geometry changed: {{scene.unresolved_faces.length}} deleted faces no longer match. Restore deleted faces, then reselect the faces to remove before meshing.</div>
      <footer><label>Mesh edge (mm)<input type="number" min="0.01" max="1000" step="any" :value="doc?fmt(doc.edge_length_m*1000):10" :disabled="blocked" @change="edge"></label><button :disabled="blocked||!scene||!!scene.unresolved_faces.length" @click="send('mesh')">Mesh retained surface</button><label><input type="checkbox" v-model="showMesh" :disabled="!report||blocked">Show mesh</label><span v-if="report">{{report.vertices}} vertices · {{report.triangles}} triangles · {{report.boundary_loops}} boundary loops</span><button class="cs-apply" :disabled="blocked||!scene?.can_apply" @click="send('apply')">Apply as winding surface</button><span class="cs-replaces">Replaces the current winding support. Project Undo restores it.</span></footer>
      <div class="cs-status" role="status" aria-live="polite">{{busy?'Working…':hint||status}}<span v-if="error||localError" class="cs-error" role="alert">{{error||localError}}</span></div>
    </section>`
  };
  if(typeof module!=='undefined'&&module.exports)module.exports={roots,rotation,newPrimitive,primitivePreview,triangleHit,SurfaceCanvas,component};
  if(typeof window!=='undefined')window.CoilGenSurfaceEditor={install(app){app.component('coilgen-surface-editor',component);}};
})();
