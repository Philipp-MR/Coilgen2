/* Local 3D interaction only; all winding mutations and certificates live in Python. */
(function () {
  "use strict";
  const add = (a,b) => a.map((v,i)=>v+b[i]);
  const sub = (a,b) => a.map((v,i)=>v-b[i]);
  const mul = (a,s) => a.map(v=>v*s);
  const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));

  function closestSegment(x,y,a,b) {
    const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy;
    const t=d ? clamp(((x-a[0])*dx+(y-a[1])*dy)/d,0,1) : 0;
    return {t,distance:Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)};
  }

  function exactCutValue(displayed, originalMetres) {
    return Number(displayed) === Number((originalMetres*1000).toFixed(5)) ? originalMetres*1000 : Number(displayed);
  }

  const flatPoint=p=>p ? [p[0],p[1],0] : null;
  function pathStation(points, fraction) {
    const lengths=points.slice(1).map((p,i)=>Math.hypot(...sub(p,points[i]))),total=lengths.reduce((a,b)=>a+b,0);
    let s=clamp(fraction,0,1)*total;
    for(let i=0;i<lengths.length;i++){if(s<=lengths[i]&&lengths[i]>0)return add(points[i],mul(sub(points[i+1],points[i]),s/lengths[i]));s-=lengths[i];}
    return points[points.length-1];
  }
  function depthTester(projected, faces, tolerance) {
    const bins=new Map(),size=24;
    for(const f of faces){const tri=f.map(i=>projected[i]);const xs=tri.map(p=>p[0]),ys=tri.map(p=>p[1]);
      for(let x=Math.floor(Math.min(...xs)/size);x<=Math.floor(Math.max(...xs)/size);x++)for(let y=Math.floor(Math.min(...ys)/size);y<=Math.floor(Math.max(...ys)/size);y++){
        const k=x+','+y;if(!bins.has(k))bins.set(k,[]);bins.get(k).push(tri);
      }
    }
    return p=>{for(const [a,b,c] of bins.get(Math.floor(p[0]/size)+','+Math.floor(p[1]/size))||[]){
      const v=sub(b,a),w=sub(c,a),q=sub(p,a),det=v[0]*w[1]-v[1]*w[0];if(Math.abs(det)<1e-12)continue;
      const u=(q[0]*w[1]-q[1]*w[0])/det,t=(v[0]*q[1]-v[1]*q[0])/det;
      if(u>=0&&t>=0&&u+t<=1&&a[2]+u*v[2]+t*w[2]>p[2]+tolerance)return false;
    }return true;};
  }
  function directionArrows(points, spacing=120) {
    const segments=[];let total=0;
    for(let i=1;i<points.length;i++)if(points[i-1]&&points[i]){
      const a=points[i-1],b=points[i],length=Math.hypot(b[0]-a[0],b[1]-a[1]);
      if(length>1e-8){segments.push({a,b,length,start:total});total+=length;}
    }
    if(total<spacing*.4)return [];
    const count=Math.min(12,Math.max(1,Math.floor(total/spacing))),result=[];
    for(let i=0;i<count;i++){
      const station=(i+.5)*total/count,s=segments.find(s=>s.start+s.length>=station),t=(station-s.start)/s.length;
      result.push({point:[s.a[0]+t*(s.b[0]-s.a[0]),s.a[1]+t*(s.b[1]-s.a[1])],angle:Math.atan2(s.b[1]-s.a[1],s.b[0]-s.a[0])});
    }
    return result;
  }
  function flatScene(scene) {
    if(!scene?.chart) return null;
    return {...scene,vertices:scene.chart.faces.flat().map(flatPoint),faces:scene.chart.faces.map((_,i)=>[3*i,3*i+1,3*i+2]),
      sections:scene.sections.map(s=>({...s,points:s.flat_points.map(flatPoint),guides:s.flat_guides.map(flatPoint)})),
      ports:scene.ports.map(p=>({...p,point:flatPoint(p.flat)})),cut_point:flatPoint(scene.flat_cut_point),
      drawing_ports:(scene.drawing_ports||[]).map(p=>({...p,point:flatPoint(p.flat)})),
      cut_curve:scene.flat_cut_curve?.map(flatPoint),baseline:scene.flat_baseline.map(flatPoint)};
  }
  function chartPick(scene, point) {
    for(let i=0;i<(scene?.chart?.faces.length||0);i++) {
      const [a,b,c]=scene.chart.faces[i],v=sub(b,a),w=sub(c,a),q=sub(point,a),det=v[0]*w[1]-v[1]*w[0];
      if(Math.abs(det)<1e-20) continue;
      const u=(q[0]*w[1]-q[1]*w[0])/det,t=(v[0]*q[1]-v[1]*q[0])/det;
      if(u>=-1e-9&&t>=-1e-9&&u+t<=1+1e-9) {
        const f=scene.faces[i],origin=scene.vertices[f[0]];
        return add(origin,add(mul(sub(scene.vertices[f[1]],origin),u),mul(sub(scene.vertices[f[2]],origin),t)));
      }
    }
    return null;
  }

  class Camera {
    constructor() { this.yaw=-Math.PI/4; this.pitch=1.05; this.zoom=1; this.pan=[0,0]; }
    fit(vertices) {
      const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
      vertices.forEach(p=>p.forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v);}));
      this.center=mul(add(lo,hi),.5); this.extents=sub(hi,lo);this.span=Math.hypot(...this.extents)||1;
      this.zoom=1; this.pan=[0,0];
    }
    basis() {
      const c=Math.cos(this.yaw),s=Math.sin(this.yaw),p=Math.cos(this.pitch),q=Math.sin(this.pitch);
      return [[c,-s,0],[p*s,p*c,-q],[q*s,q*c,p]];
    }
    scale() {
      const [u,v]=this.basis(),width=dot(u.map(Math.abs),this.extents),height=dot(v.map(Math.abs),this.extents);
      return .9*Math.min(this.width/Math.max(width,1e-12),this.height/Math.max(height,1e-12))*this.zoom;
    }
    project(point) {
      const p=sub(point,this.center),[u,v,n]=this.basis(),s=this.scale();
      return [this.width/2+this.pan[0]+dot(p,u)*s,this.height/2+this.pan[1]-dot(p,v)*s,dot(p,n)];
    }
    unproject(x,y,depth) {
      const [u,v,n]=this.basis(),s=this.scale();
      return add(this.center,add(add(mul(u,(x-this.width/2-this.pan[0])/s),mul(v,-(y-this.height/2-this.pan[1])/s)),mul(n,depth)));
    }
    surface(x,y,vertices,faces,referenceDepth=null) {
      const origin=this.unproject(x,y,this.span*2),direction=mul(this.basis()[2],-1);
      let distance=Infinity,hit=null;
      for(const face of faces) {
        const a=vertices[face[0]],e1=sub(vertices[face[1]],a),e2=sub(vertices[face[2]],a);
        const h=cross(direction,e2),det=dot(e1,h);
        if(Math.abs(det)<1e-15) continue;
        const s=sub(origin,a),u=dot(s,h)/det;
        if(u < -1e-9 || u > 1+1e-9) continue;
        const q=cross(s,e1),v=dot(direction,q)/det;
        if(v < -1e-9 || u+v > 1+1e-9) continue;
        const t=dot(e2,q)/det;
        const candidate=add(origin,mul(direction,t));
        const score=referenceDepth===null?t:Math.abs(this.project(candidate)[2]-referenceDepth);
        if(t>=0 && score<distance) {distance=score;hit=candidate;}
      }
      return hit;
    }
  }

  class WindingCanvas {
    constructor(canvas,emit,hint,flat=false) {
      this.canvas=canvas;this.emit=emit;this.hint=hint;this.camera=new Camera();this.drag=null;this.handles=[];
      this.flat=flat;this.mode='select';
      if(flat){this.camera.yaw=0;this.camera.pitch=0;}
      this.bindings={pointerdown:e=>this.down(e),pointermove:e=>this.move(e),pointerup:e=>this.up(e),
        pointercancel:()=>this.cancel(),wheel:e=>this.wheel(e),dblclick:()=>this.fit(),keydown:e=>this.key(e)};
      for(const [name,fn] of Object.entries(this.bindings)) canvas.addEventListener(name,fn,{passive:false});
      this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
    }
    destroy() { this.resize.disconnect(); for(const [name,fn] of Object.entries(this.bindings)) this.canvas.removeEventListener(name,fn); }
    update(scene,busy,ghost) {
      // Copy transport proxies once per snapshot, not once per animation frame.
      if(scene!==this.inputScene){this.inputScene=scene;this.plainScene=scene?JSON.parse(JSON.stringify(scene)):null;}
      scene=this.plainScene;
      this.physicalScene=scene;
      if(this.flat) scene=flatScene(scene);
      if(scene && scene.source!==this.scene?.source) this.camera.fit(scene.vertices);
      this.scene=scene;this.busy=busy;this.ghost=ghost;
      if(busy) this.cancel();
      this.draw();
    }
    fit() { if(this.scene) {this.camera.fit(this.scene.vertices);this.draw();} }
    view(name) {
      const views={Oblique:[-Math.PI/4,1.05],Front:[0,Math.PI/2],Back:[Math.PI,Math.PI/2],Top:[0,0],Side:[Math.PI/2,Math.PI/2]};
      [this.camera.yaw,this.camera.pitch]=views[name];this.draw();
    }
    faceSelection(local=false) {
      const frame=this.physicalScene?.local_frame;if(!frame)return;
      let n=frame.normal;if(dot(n,this.camera.basis()[2])<0)n=mul(n,-1);
      this.camera.pitch=Math.acos(clamp(n[2],-1,1));this.camera.yaw=Math.atan2(n[0],n[1]);
      this.camera.center=[...frame.center];this.camera.pan=[0,0];this.camera.zoom=local?2.5:1.25;this.localView=local;this.draw();
    }
    focusIssue(issue){if(!issue?.point)return;this.camera.center=[...issue.point];this.camera.pan=[0,0];this.camera.zoom=3;this.focusedIssue=issue;this.draw();}
    visible(p){return !!p&&(!this.hideRear||this.flat||!this.depthTest||this.depthTest(this.camera.project(p)));}
    shown(s){return this.isolateGroup===null||this.isolateGroup===undefined||s.group===this.isolateGroup;}
    displayPoints(points){return points.map(p=>this.visible(p)?p:null);}
    playbackFrame(key,fraction){
      const changed=key!==this.playKey;this.playKey=key;this.playFraction=fraction;
      if(changed||!this.playBackground){this.draw();return;}
      const ctx=this.canvas.getContext('2d'),ratio=window.devicePixelRatio||1;
      ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,this.canvas.width,this.canvas.height);ctx.drawImage(this.playBackground,0,0);
      ctx.setTransform(ratio,0,0,ratio,0,0);this.paintCurrent(ctx);
    }
    paintCurrent(ctx){const row=this.scene?.sections.find(s=>s.key===this.playKey),points=row?.points.filter(Boolean);if(!points?.length)return;
      const p=this.camera.project(pathStation(points,this.playFraction||0));ctx.fillStyle='#fff';ctx.strokeStyle='#007f83';ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill();ctx.stroke();
    }
    xy(e) {const r=this.canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
    line(ctx,points,color,width=1,dash=[]) {
      if(points.length<2) return;
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();
      let drawing=false;
      points.forEach(p=>{if(!p){drawing=false;return;}const s=this.camera.project(p);drawing?ctx.lineTo(s[0],s[1]):ctx.moveTo(s[0],s[1]);drawing=true;});ctx.stroke();ctx.setLineDash([]);
    }
    draw() {
      const c=this.canvas,w=c.clientWidth,h=c.clientHeight;
      if(!w||!h) return;
      const ratio=window.devicePixelRatio||1;
      if(c.width!==Math.round(w*ratio)||c.height!==Math.round(h*ratio)){c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);}
      const ctx=c.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
      this.camera.width=w;this.camera.height=h;
      if(!this.scene) return;
      const {vertices,faces,sections}=this.scene,projected=vertices.map(p=>this.camera.project(p));
      this.depthTest=this.hideRear&&!this.flat?depthTester(projected,faces,this.camera.span*1e-5):null;
      const sorted=faces.map(f=>({f,z:f.reduce((s,i)=>s+projected[i][2],0)})).sort((a,b)=>a.z-b.z);
      ctx.fillStyle="rgba(76,111,134,.045)";
      sorted.forEach(({f})=>{ctx.beginPath();f.forEach((i,k)=>k?ctx.lineTo(...projected[i].slice(0,2)):ctx.moveTo(...projected[i].slice(0,2)));ctx.closePath();ctx.fill();});
      if(this.ghost) this.line(ctx,this.scene.baseline,"#9aa9b7",1,[4,4]);
      const affected=new Set((this.scene.seam_preview?.cuts||[]).map(c=>c.key));
      sections.filter(s=>s.key!==this.scene.selected&&this.shown(s)).forEach(s=>{ctx.globalAlpha=(this.playKey&&s.key!==this.playKey)||(affected.size&&!affected.has(s.key))? .22 : 1;this.line(ctx,this.displayPoints(s.points),
        s.color||'#66798c',s.editable?2:1.5,s.disconnected?[3,5]:[]);});ctx.globalAlpha=1;
      for(const section of sections.filter(s=>s.contour&&!s.disconnected&&this.shown(s))) {
        ctx.globalAlpha=this.playKey&&section.key!==this.playKey? .22 : 1;
        ctx.strokeStyle=section.color;ctx.lineWidth=1.5;ctx.setLineDash([]);
        for(const {point:p,angle:a} of directionArrows(this.displayPoints(section.points).map(p=>p?this.camera.project(p):null))) {
          ctx.beginPath();ctx.moveTo(p[0]-5*Math.cos(a-.55),p[1]-5*Math.sin(a-.55));ctx.lineTo(...p);
          ctx.lineTo(p[0]-5*Math.cos(a+.55),p[1]-5*Math.sin(a+.55));ctx.stroke();
        }
      }
      ctx.globalAlpha=1;
      const selected=sections.find(s=>s.key===this.scene.selected);
      this.handles=[];
      if(selected) {
        ctx.globalAlpha=this.playKey&&selected.key!==this.playKey ? .18 : 1;
        this.line(ctx,this.displayPoints(selected.points),'#fff',6,selected.disconnected?[4,5]:[]);
        this.line(ctx,this.displayPoints(selected.points),selected.color||'#263f51',3,selected.disconnected?[4,5]:[]);
        if(!selected.locked && !selected.disconnected && !selected.bridge && this.mode==='select') {
          if(selected.cut) this.handles=[{kind:"cut",index:0,point:this.scene.cut_point}];
          else if(selected.editable) this.handles=(selected.guides.length?selected.guides:[selected.points[Math.floor(selected.points.length/2)]]).map((p,i)=>({kind:"guide",index:i,point:p}));
        }
        if(selected.bridge&&!selected.locked&&!this.flat&&['select','bridge'].includes(this.mode)&&this.scene.bridge_handle)this.handles=[{kind:'bridge_height',index:0,point:this.scene.bridge_handle.point}];
        const visible=selected.points.filter(Boolean),middle=Math.floor(visible.length/2),a=this.camera.project(visible[Math.max(0,middle-1)]),b=this.camera.project(visible[middle]);
        const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.fillStyle="#007f83";ctx.beginPath();ctx.moveTo(b[0]+7*Math.cos(angle),b[1]+7*Math.sin(angle));
        ctx.lineTo(b[0]-6*Math.cos(angle-.6),b[1]-6*Math.sin(angle-.6));ctx.lineTo(b[0]-6*Math.cos(angle+.6),b[1]-6*Math.sin(angle+.6));ctx.closePath();ctx.fill();
      }
      ctx.globalAlpha=1;if(this.playKey)this.handles=[];
      for(const handle of this.handles) {
        const p=this.camera.project(this.drag?.handle===handle?this.drag.point:handle.point);
        ctx.beginPath();ctx.arc(p[0],p[1],7,0,2*Math.PI);ctx.fillStyle="#fff";ctx.fill();ctx.strokeStyle="#007f83";ctx.lineWidth=3;ctx.stroke();
        ctx.font="600 11px system-ui";ctx.fillStyle="#005f63";ctx.fillText(handle.kind==="cut"?"CUT":handle.kind==='bridge_height'?'HEIGHT':`GUIDE ${handle.index+1}`,p[0]+12,p[1]-10);
      }
      if(this.drag?.point && this.drag?.handle) {
        this.line(ctx,[this.drag.handle.point,this.drag.point],"#007f83",2,[4,4]);
        const p=this.camera.project(this.drag.point);ctx.beginPath();ctx.arc(p[0],p[1],8,0,2*Math.PI);ctx.fillStyle="rgba(0,127,131,.3)";ctx.fill();
      }
      if(this.drag?.stroke) this.line(ctx,this.drag.stroke,'#008b8c',3,[5,3]);
      if(this.drag?.shapeHandle&&this.drag.point)this.line(ctx,[this.drag.shapeHandle.point,this.drag.point],'#a766c1',4);
      const preview=this.scene.seam_preview;
      if(!this.flat)for(const obj of this.scene.cut_objects||[])if(obj.id!==preview?.object_id){
        for(const line of obj.display.lines)this.line(ctx,line,'#bba4c6',1,[3,5]);
        const p=this.camera.project(obj.shape.center);ctx.fillStyle='#8346a0';ctx.font='11px system-ui';ctx.fillText(obj.label,p[0]+10,p[1]-10);
      }
      if(preview) {
        for(const route of preview.routes||[]){let points=route.points;if(this.flat)points=this.scene.sections.find(s=>s.key===route.key)?.preview_flat||[];this.line(ctx,points,'#0b9b91',3,[7,4]);}
        if(preview.stroke){const points=preview.stroke.map(p=>this.flat?flatPoint(p):chartPick(this.physicalScene,p)).filter(Boolean);this.line(ctx,points,'#008b8c',2,[5,3]);}
        if(preview.shape&&!this.flat){
          for(const points of preview.display.lines)this.line(ctx,points,'#a766c1',1.5,[5,3]);
          for(const cut of preview.cuts)this.line(ctx,cut.opening,'#e05b47',5);
          for(const cut of preview.alternatives||[])if(!cut.selected)this.line(ctx,cut.opening,'#e5a38f',4,[3,4]);
          if(this.mode==='shape')for(const handle of preview.display.handles){
            const p=this.camera.project(handle.point);ctx.fillStyle=handle.role==='center'?'#a766c1':handle.role==='normal'?'#307caf':'#fff';ctx.strokeStyle='#8346a0';ctx.lineWidth=2;
            ctx.beginPath();ctx.arc(p[0],p[1],7,0,2*Math.PI);ctx.fill();ctx.stroke();
            if(handle.role.startsWith('size')){ctx.fillStyle='#67457c';ctx.font='11px system-ui';ctx.fillText((preview.shape.size[Number(handle.role.slice(4))]*2000).toFixed(2)+' mm',p[0]+10,p[1]-9);}
          }
        }
        for(const cut of preview.cuts){const point=this.flat?this._flatOfCut(cut):cut.point;if(!point)continue;const p=this.camera.project(point);ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.strokeStyle='#008b8c';ctx.lineWidth=2;ctx.stroke();}
      }
      for(const issue of this.scene.issues||[])if(issue.point&&!this.flat&&this.visible(issue.point)){
        const p=this.camera.project(issue.point);ctx.strokeStyle=issue.category==='failed'?'#c84938':'#c48a2d';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.stroke();
      }
      if(this.focusedIssue?.point&&!this.flat){for(const [a,b] of this.focusedIssue.segments||[])this.line(ctx,[this.focusedIssue.points[a],this.focusedIssue.points[b]],'#c84938',6);const p=this.camera.project(this.focusedIssue.point);ctx.fillStyle='#fff';ctx.fillRect(p[0]+8,p[1]-25,260,25);ctx.fillStyle='#a35b2d';ctx.fillText(this.focusedIssue.label,p[0]+13,p[1]-9);}
      const playing=sections.find(s=>s.key===this.playKey);
      if(playing)this.line(ctx,this.displayPoints(playing.points),playing.color||'#007f83',5);
      if(this.drag?.snap){const p=this.camera.project(this.drag.snap.point);ctx.strokeStyle='#00a49b';ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],13,0,Math.PI*2);ctx.stroke();}
      if(['connect','draw'].includes(this.mode)) for(const port of this.ports()) {
        const p=this.camera.project(port.point);ctx.fillStyle=port.role==='out'?'#007f83':'#bd663d';ctx.strokeStyle='#fff';ctx.lineWidth=2;
        ctx.beginPath();ctx.arc(p[0],p[1],8,0,Math.PI*2);ctx.fill();ctx.stroke();
        ctx.fillStyle='#fff';ctx.font='bold 12px system-ui';ctx.textAlign='center';ctx.fillText(port.role==='out'?'→':'•',p[0],p[1]+4);ctx.textAlign='start';
      }
      // World-axis compass, independent of the winding scale.
      const origin=[w-62,h-57],basis=this.camera.basis();
      ["X","Y","Z"].forEach((label,i)=>{const x=origin[0]+30*basis[0][i],y=origin[1]-30*basis[1][i];ctx.strokeStyle=["#bd5b48","#44866e","#4f77b1"][i];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(label,x+3,y-3);});
      if(playing){if(!this.playBackground)this.playBackground=document.createElement('canvas');this.playBackground.width=c.width;this.playBackground.height=c.height;this.playBackground.getContext('2d').drawImage(c,0,0);this.paintCurrent(ctx);}else this.playBackground=null;
    }
    nearest(points,x,y) {
      let best={distance:Infinity};
      for(let i=1;i<points.length;i++) {
        if(!points[i-1]||!points[i])continue;
        const a=this.camera.project(points[i-1]),b=this.camera.project(points[i]);
        const hit=closestSegment(x,y,a,b);
        if(hit.distance<best.distance) best={...hit,point:add(points[i-1],mul(sub(points[i],points[i-1]),hit.t))};
      }
      return best;
    }
    _flatOfCut(cut) {
      // The stroke and the source contour share the same piecewise-isometric chart.
      const row=this.physicalScene.sections.find(s=>s.key===cut.key);
      if(!row)return null;
      let best=Infinity,point=null;
      for(let i=1;i<row.points.length;i++) {
        if(!row.flat_points[i-1]||!row.flat_points[i])continue;
        const v=sub(row.points[i],row.points[i-1]),t=clamp(dot(sub(cut.point,row.points[i-1]),v)/dot(v,v),0,1);
        const distance=Math.hypot(...sub(cut.point,add(row.points[i-1],mul(v,t))));
        if(distance<best){best=distance;point=flatPoint(add(row.flat_points[i-1],mul(sub(row.flat_points[i],row.flat_points[i-1]),t)));}
      }
      return point;
    }
    ports(){return (this.mode==='draw'?(this.scene.drawing_ports||[]):this.scene.ports).filter(p=>this.visible(p.point));}
    portAt(x,y,role=null){return this.ports().filter(p=>!role||p.role===role).map(p=>({p,d:Math.hypot(x-this.camera.project(p.point)[0],y-this.camera.project(p.point)[1])})).filter(hit=>hit.d<14).sort((a,b)=>a.d-b.d)[0]?.p;}
    surfaceAt(x,y){return this.camera.surface(x,y,this.scene.vertices,this.scene.faces);}
    emitPoint(message){
      if(this.flat && message.point) {
        message.point=chartPick(this.physicalScene,message.point.slice(0,2));
        if(!message.point){this.hint('Keep the handle inside the support.');return;}
      }
      this.emit(message);
    }
    down(e) {
      if(e.button!==0||!this.scene||this.busy) return;
      e.preventDefault();this.canvas.focus();this.canvas.setPointerCapture(e.pointerId);
      const [x,y]=this.xy(e);
      if(e.shiftKey){this.drag={x,y,orbit:true,pan:true,offset:[...this.camera.pan]};return;}
      if(this.mode==='shape'&&!this.flat){const handles=this.scene.seam_preview?.display?.handles||[];const handle=handles.map(h=>({h,p:this.camera.project(h.point)})).map(v=>({...v,d:Math.hypot(x-v.p[0],y-v.p[1])})).filter(v=>v.d<13).sort((a,b)=>a.d-b.d)[0]?.h;if(handle){this.drag={x,y,shapeHandle:handle,point:handle.point,identity:this.scene.identity};return;}}
      const handle=this.handles.find(h=>{const p=this.camera.project(h.point);return Math.hypot(x-p[0],y-p[1])<13;});
      // Cut objects must not steal endpoint/sketch gestures or a visible edit handle.
      if(!this.flat&&(this.mode==='shape'||(this.mode==='select'&&!handle))){
        const alternative=(this.scene.seam_preview?.alternatives||[]).find(c=>!c.selected&&this.nearest(c.opening,x,y).distance<10);
        if(alternative){this.emit({kind:'shape_choice',cut_key:alternative.key,index:alternative.index});return;}
        const obj=(this.scene.cut_objects||[]).find(o=>o.id!==this.scene.seam_preview?.object_id&&o.display.lines.some(line=>this.nearest(line,x,y).distance<9));
        if(obj){this.emit({kind:'select_cut_object',object_id:obj.id});return;}
        if(this.mode==='shape'&&this.placeOnSurface){const point=this.surfaceAt(x,y);if(point)this.emit({kind:'place_shape',point});return;}
      }
      if(this.mode==='seam') {const p=this.surfaceAt(x,y);if(p)this.drag={x,y,stroke:[p],identity:this.scene.identity};else this.hint('Start the seam on the support.');return;}
      if(this.mode==='connect') {const port=this.portAt(x,y);this.drag=port?{x,y,port}:null;return;}
      if(this.mode==='draw') {const port=this.portAt(x,y,'out');if(port){this.drag={x,y,port,stroke:[port.point],identity:this.scene.identity};return;}}
      if(handle) this.drag={x,y,handle,point:handle.point,identity:this.scene.identity};
      else {
        let hit=null,best=12,selectionPoint=null;
        this.scene.sections.forEach(s=>{if(!this.shown(s)||(this.mode==='bridge'&&s.key===this.scene.selected)||(this.mode==='draw'&&!s.editable))return;const n=this.nearest(this.displayPoints(s.points),x,y);if(n.distance<best){best=n.distance;hit=s;selectionPoint=n.point;}});
        this.drag={x,y,select:hit?.key,selectionPoint,orbit:!hit,pan:e.shiftKey||this.flat,yaw:this.camera.yaw,pitch:this.camera.pitch,offset:[...this.camera.pan]};
      }
    }
    move(e) {
      if(!this.drag||!this.scene) return;
      const [x,y]=this.xy(e),d=this.drag;d.moved=Math.hypot(x-d.x,y-d.y)>=3;
      if(d.shapeHandle){d.point=d.shapeHandle.role==='center'?this.camera.surface(x,y,this.scene.vertices,this.scene.faces,this.camera.project(d.shapeHandle.point)[2]):this.camera.unproject(x,y,this.camera.project(d.shapeHandle.point)[2]);this.hint('Release to preview the cut');}
      else if(d.stroke) {
        const p=d.port?this.camera.surface(x,y,this.scene.vertices,this.scene.faces,this.camera.project(d.port.point)[2]):this.surfaceAt(x,y);
        if(p&&d.stroke.length<512){const last=this.camera.project(d.stroke[d.stroke.length-1]);if(Math.hypot(x-last[0],y-last[1])>=3)d.stroke.push(p);}
        d.snap=d.port?this.portAt(x,y,'in'):null;
        this.hint(d.port?(d.snap?'Release to preview connection':'Draw to the orange endpoint'):'Release to preview cuts');
      } else if(d.port) {
        const target=this.portAt(x,y);this.hint(target?.role==='in'?'Release to reconnect to '+target.label:'Drag OUT to an IN endpoint, or click the two endpoints.');
      } else if(d.handle) {
        const section=this.scene.sections.find(s=>s.key===this.scene.selected);
        if(d.handle.kind==="cut") d.point=this.nearest(this.scene.cut_curve,x,y).point;
        else if(d.handle.kind==='bridge_height'||section.geometry.includes("free_3d")) d.point=this.camera.unproject(x,y,this.camera.project(d.handle.point)[2]);
        else d.point=this.camera.surface(x,y,this.scene.vertices,this.scene.faces,this.camera.project(d.handle.point)[2]);
        this.hint(d.point?"Release to commit one edit · Esc cancels":"Keep the guide over the support surface");
      } else if(d.orbit) {
        if(d.pan) this.camera.pan=[d.offset[0]+x-d.x,d.offset[1]+y-d.y];
        else {this.camera.yaw=d.yaw+(x-d.x)*.007;this.camera.pitch=d.pitch+(y-d.y)*.007;}
      }
      this.draw();
    }
    up(e) {
      const d=this.drag;if(!d) return;this.drag=null;
      if(this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
      if(d.shapeHandle&&d.moved&&d.point&&d.identity===this.scene.identity)this.emit({kind:'shape_handle',role:d.shapeHandle.role,point:d.point});
      else if(d.stroke&&d.port&&d.moved&&d.identity===this.scene.identity){
        const [x,y]=this.xy(e),target=this.portAt(x,y,'in');
        if(target?.role==='in')this.emit({kind:'preview_drawing',source:d.port.key,target:target.key,stroke:[...d.stroke,target.point].map(p=>this.flat?p.slice(0,2):p),flat:this.flat});
        else {this.hint('Finish on an orange endpoint');this.draw();return;}
      }
      else if(d.stroke && !d.port && d.stroke.length>=2 && d.identity===this.scene.identity){const [x,y]=this.xy(e),end=this.surfaceAt(x,y);if(end&&Math.hypot(...sub(end,d.stroke[d.stroke.length-1]))>1e-12){if(d.stroke.length>=512)d.stroke[d.stroke.length-1]=end;else d.stroke.push(end);}this.emit({kind:'preview_seam',stroke:d.stroke.map(p=>this.flat?p.slice(0,2):p),flat:this.flat});}
      else if(d.port){const [x,y]=this.xy(e),target=this.portAt(x,y);if(d.port.role==='out'&&target?.role==='in'&&d.moved)this.emit({kind:'reconnect_endpoints',source:d.port.key,target:target.key});else this.emit({kind:'port',port:d.port});}
      else if(d.handle && d.moved && d.point && d.identity===this.scene.identity) {
        this.emitPoint({kind:d.handle.kind==="cut"?"drag_cut":d.handle.kind==='bridge_height'?'bridge_height':"drag_guide",key:this.scene.selected,index:d.handle.index,point:d.point,flat:this.flat});
      } else if(d.select && !d.moved) this.emitPoint({kind:this.mode==='bridge'?'bridge_target':'select',key:d.select,point:d.selectionPoint});
      this.hint("");this.draw();
    }
    cancel() {this.drag=null;this.hint("");this.draw();}
    key(e) { if(e.key==="Escape") {e.preventDefault();e.stopPropagation();this.cancel();} }
    wheel(e) {if(!this.scene) return;e.preventDefault();const [x,y]=this.xy(e),anchor=this.camera.unproject(x,y,0);this.camera.zoom=clamp(this.camera.zoom*Math.exp(-e.deltaY*.001),.2,20);const p=this.camera.project(anchor);this.camera.pan=add(this.camera.pan,[x-p[0],y-p[1]]);this.draw();}
  }

  const component={
    props:{scene:Object,busy:Boolean,status:String,error:String,drafts:Array,download:String,reportUrl:String,receipt:Object},
    emits:["action","close"],
    data(){return {details:false,ghost:false,query:"",position:0,span:0,xyz:[0,0,0],guideIndex:0,draft:"",localHint:"",localError:"",pending:"",viewName:"Oblique",mode:'select',shapeKind:'plane',layout:'linked',seamGroup:'',portSource:'',portTarget:'',bridgeOther:'',separation:3,bridgeSpan:20,bridgePoint:null,hideRear:false,isolateGroup:null,placeOnSurface:false,showIssues:false,playOpen:false,playing:false,playIndex:0,playFraction:0,localView:false};},
    computed:{
      selected(){return this.scene?.sections.find(s=>s.key===this.scene.selected);},
      rows(){const q=this.query.toLowerCase();return (this.scene?.sections||[]).filter(s=>s.label.toLowerCase().includes(q));},
      locked(){return !this.selected||this.selected.locked||this.selected.disconnected;},
      blocked(){return this.busy||!!this.pending;},
      fieldChange(){return this.receipt?.diagnostics?.physical?.longitudinal_change;},
      current(){return this.receipt?.diagnostics?.circuits?.[this.scene?.circuit]?.current_A;},
      bend(){return this.receipt?.diagnostics?.circuits?.[this.scene?.circuit]?.manufacturing?.sweep_final_min_bend_radius_m;},
      groups(){return [...new Set((this.scene?.sections||[]).filter(s=>s.cut).map(s=>s.group))];},
      legendGroups(){return [...new Map((this.scene?.sections||[]).filter(s=>s.group>=0).map(s=>[s.group,{index:s.group,color:s.color}])).values()].sort((a,b)=>a.index-b.index);},
      outputs(){return (this.scene?.ports||[]).filter(p=>p.role==='out');},
      inputs(){return (this.scene?.ports||[]).filter(p=>p.role==='in');},
      step(){return this.scene?.sequence?.[this.playIndex];},
      canGenerate(){const p=this.scene?.seam_preview;return !this.blocked&&!!this.scene&&(!p?this.mode!=='shape':!p.error&&!!(p.cuts.length||p.drawing));},
      feedback(){if(this.blocked)return 'Working…';const s=this.status||'';if(s.startsWith('Draft saved:'))return 'Draft saved';if(s.startsWith('Draft reopened.'))return 'Draft reopened';return {'Routing unlocked connections completed.':'Connections generated','Draw_connection completed.':'Connection drawn','Moving seam completed.':'Connections generated','Undo completed.':'Undone','Redo completed.':'Redone'}[s]||'';},
      toolHint(){return {select:'Click a line · drag its handles',draw:'Select a connection · draw from → to •',seam:'Draw across turns',shape:'Drag center · white handles resize · blue handle tilts',connect:'Drag → to •',bridge:'Click the crossing partner'}[this.mode];},
    },
    watch:{
      scene(next,previous){this.pending="";if(previous?.seam_preview?.drawing&&next?.revision!==previous.revision)this.mode='select';if(this.lastIdentity!==this.scene?.identity){this.playing=false;this.playIndex=0;this.playFraction=0;this.lastIdentity=this.scene?.identity;}this.syncSurface();this.syncFields();this.$nextTick(()=>this.redraw());},
      busy(){if(!this.busy)this.pending="";this.redraw();},
      status(){this.pending="";}, error(){this.pending="";},
      ghost(){this.redraw();},
      hideRear(){this.redraw();}, isolateGroup(){this.redraw();},
      mode(){this.localHint='';if(this.mode==='shape'){this.layout='3d';if(!this.scene?.seam_preview?.shape)this.previewShape();}this.redraw();},
      layout(){this.$nextTick(()=>this.redraw());},
      guideIndex(){this.guideFields();},
    },
    mounted(){this.renderer=new WindingCanvas(this.$refs.canvas,m=>this.send(m),s=>this.localHint=s);this.flatRenderer=new WindingCanvas(this.$refs.flatCanvas,m=>this.send(m),s=>this.localHint=s,true);this.syncSurface();this.redraw();this.syncFields();this.playTimer=setInterval(()=>this.tick(),50);},
    beforeUnmount(){clearInterval(this.playTimer);this.renderer?.destroy();this.flatRenderer?.destroy();},
    methods:{
      previewShape(shape=null,fresh=false){this.send({kind:'preview_shape',shape,shape_kind:this.shapeKind,object_id:shape&&!fresh?this.scene?.seam_preview?.object_id:null});},
      selectObject(obj){this.shapeKind=obj.shape.kind;this.mode='shape';this.send({kind:'select_cut_object',object_id:obj.id});},
      selectGroup(g){this.seamGroup=g;this.isolateGroup=this.isolateGroup===g?null:g;if(this.mode==='shape')this.previewShape();},
      placeShape(){this.placeOnSurface=!this.placeOnSurface;this.localHint=this.placeOnSurface?'Click the support to place a new cut shape':'';this.redraw();},
      faceSelected(local=false){this.layout='3d';this.localView=local;this.renderer.faceSelection(local);},
      issueFocus(issue){if(issue.key)this.send({kind:'select',key:issue.key});this.layout='3d';this.renderer.focusIssue(issue);this.localHint=issue.detail;},
      tick(){if(!this.playing||this.blocked||!this.step)return;this.playFraction+=.025;if(this.playFraction>=1){this.playFraction=0;if(this.playIndex+1>=this.scene.sequence.length){this.playing=false;this.playFraction=1;}else this.playIndex++;}for(const r of [this.renderer,this.flatRenderer])if(r?.canvas.clientWidth&&r?.scene)r.playbackFrame(this.step?.key,this.playFraction);},
      stepPlayback(delta){this.playing=false;this.playFraction=0;this.playIndex=clamp(this.playIndex+delta,0,Math.max(0,(this.scene?.sequence?.length||1)-1));this.redraw();},
      togglePlayback(){this.playOpen=!this.playOpen;if(!this.playOpen)this.playing=false;this.redraw();},
      shapeSize(index,value){const shape=JSON.parse(JSON.stringify(this.scene.seam_preview.shape));shape.size[index]=Number(value)/2000;this.previewShape(shape);},
      shapeAxis(axis){if(!axis)return;const shape=JSON.parse(JSON.stringify(this.scene.seam_preview.shape));shape.normal=[0,0,0];shape.normal[Number(axis)-1]=1;delete shape.tangent;this.previewShape(shape);},
      syncSurface(){if(this.scene&&!this.scene.chart){this.layout='3d';if(this.mode==='seam')this.mode='select';}},
      send(message){
        if((this.blocked&&message.kind!=="cancel")||!this.scene)return;
        this.localError="";
        if(message.kind==='lock'&&this.mode==='shape')message.object_id=this.scene.seam_preview?.object_id||this.scene.cut_objects.find(o=>o.keys.includes(this.scene.selected))?.id;
        if(message.kind==='port'){if(message.port.role==='out')this.portSource=message.port.key;else this.portTarget=message.port.key;if(this.portSource&&this.portTarget)this.connectPorts();return;}
        if(message.kind==='bridge_target'){this.bridgeOther=message.key;this.bridgePoint=message.point;return;}
        if(message.kind==='select_cut_object'){this.mode='shape';const obj=this.scene.cut_objects.find(o=>o.id===message.object_id);if(obj)this.shapeKind=obj.shape.kind;}
        if(message.kind==='place_shape'){message.shape_kind=this.shapeKind;this.placeOnSurface=false;this.localHint='';}
        if(['preview_seam','preview_shape','shape_handle','place_shape'].includes(message.kind)&&this.seamGroup!==''){
          if(this.seamGroup==='selected')message.scope='selected';else message.group=Number(this.seamGroup);
        }
        this.$emit("action",{identity:this.scene.identity,circuit:this.scene.circuit,...message});
        if(!["select","circuit","cancel"].includes(message.kind)) this.pending=message.kind;
      },
      redraw(){for(const r of [this.renderer,this.flatRenderer])if(r){r.mode=this.mode;r.hideRear=this.hideRear;r.isolateGroup=this.isolateGroup;r.placeOnSurface=this.placeOnSurface;r.playKey=this.playOpen?this.step?.key:null;r.playFraction=this.playFraction;r.update(this.scene,this.blocked,this.ghost);}},
      fit(){this.localView=false;this.renderer?.fit();this.flatRenderer?.fit();},
      connectPorts(){const source=this.portSource,target=this.portTarget;this.portSource='';this.portTarget='';this.send({kind:'reconnect_endpoints',source,target});},
      bridge(relation){this.send({kind:'bridge',other:this.bridgeOther,relation,separation_mm:this.separation,span_mm:this.bridgeSpan,point:this.bridgePoint||null});},
      syncFields(){if(this.selected?.cut){this.position=Number((this.selected.cut[0]*1000).toFixed(5));this.span=Number((this.selected.cut[1]*1000).toFixed(5));}if(this.selected?.bridge){this.bridgeOther=this.selected.bridge.other;this.bridgePoint=null;this.separation=this.selected.bridge.separation_m*1000;this.bridgeSpan=this.selected.bridge.span_m*1000;}else if(this.bridgeOther===this.scene?.selected)this.bridgeOther='';if(this.lastSelected!==this.scene?.selected){this.bridgePoint=null;if(this.selected&&!this.selected.label.toLowerCase().includes(this.query.toLowerCase()))this.query='';this.lastSelected=this.scene?.selected;}this.guideIndex=0;this.guideFields();},
      guideFields(){const s=this.selected;if(!s)return;const p=s.guides[this.guideIndex]||s.points[Math.floor(s.points.length/2)];this.xyz=p.map(v=>Number((v*1000).toFixed(5)));},
      numericGuide(addNew){const p=this.xyz.map(v=>Number(v)/1000);if(!p.every(Number.isFinite)){this.localError="Enter three finite coordinates.";return;}this.send({kind:"guide",index:addNew?this.selected.guides.length:Number(this.guideIndex),point:p});},
      numericCut(){this.send({kind:"cut",position_mm:exactCutValue(this.position,this.selected.cut[0]),span_mm:exactCutValue(this.span,this.selected.cut[1])});},
      openDraft(){if(this.scene?.dirty&&!window.confirm("Replace the current unsaved winding edits with this draft?"))return;this.send({kind:"open",name:this.draft});},
      async importDraft(event){const file=event.target.files[0];event.target.value="";if(!file)return;if(file.size>50*1024**2){this.localError="Winding draft exceeds 50 MiB.";return;}if(this.scene?.dirty&&!window.confirm("Replace the current unsaved winding edits with this draft?"))return;this.send({kind:"import",content:await file.text()});},
      keys(e){if((e.ctrlKey||e.metaKey)&&!e.target.matches("input,select,textarea")){if(e.key.toLowerCase()==="z"){e.preventDefault();this.send({kind:e.shiftKey?"redo":"undo"});}if(e.key.toLowerCase()==="y"){e.preventDefault();this.send({kind:"redo"});}}},
      changeView(){this.renderer.view(this.viewName);},
      fmt(v){return Number.isFinite(v)?Number(v).toPrecision(5):"—";},
      color(g){return this.legendGroups.find(s=>s.index===g)?.color||'#66798c';},
    },
    template:`
    <section class="cg-winding" aria-label="Interconnection editor" @keydown="keys">
      <header class="cw-header"><button class="cw-icon" aria-label="Back to workbench" title="Back to workbench" :disabled="blocked" @click="$emit('close')">←</button><h1>Interconnections</h1><span class="cw-spacer"></span><span class="cw-draft-state">{{scene?.dirty ? 'Unsaved' : 'Saved'}}</span><button class="cw-icon" aria-label="Save draft" title="Save draft" :disabled="blocked" @click="send({kind:'save'})">↓</button><button class="cw-primary cw-generate" :disabled="!canGenerate" @click="send({kind:scene?.seam_preview?'apply_seam':'route'})">Generate connections</button><button :aria-pressed="details" aria-label="Details" title="Details" @click="details=!details">☷</button></header>
      <div class="cw-body" :class="{'cw-with-details':details}">
        <main class="cw-main">
          <nav class="cw-toolbar" aria-label="Winding editing tools">
            <button v-for="tool in ['select','draw','shape','seam','connect','bridge']" :aria-label="{select:'Select / drag',draw:'Draw connection',seam:'Draw seam',shape:'Cut shapes and planes',connect:'Connect endpoints',bridge:'Over / under'}[tool]" :title="{select:'Select / drag',draw:'Draw connection',seam:'Draw seam',shape:'Cut shapes and planes',connect:'Connect endpoints',bridge:'Over / under'}[tool]" :aria-pressed="mode===tool" :class="{'cw-active':mode===tool}" :disabled="blocked || !scene || (tool==='seam'&&!scene.chart)" @click="mode=tool"><svg viewBox="0 0 24 24" aria-hidden="true"><path :d="{select:'M5 3 L19 13 L12 14 L9 21 Z',draw:'M4 18 C7 2 14 23 20 5 M17 5 L20 5 L20 8',shape:'M3 7 L15 3 L21 17 L9 21 Z M12 2 L12 22',seam:'M5 5 L19 19 M5 19 L19 5 M3 6 A3 3 0 1 0 3 5 M3 19 A3 3 0 1 0 3 18',connect:'M5 12 L19 12 M15 8 L19 12 L15 16 M3 9 L6 9 L6 15 L3 15 Z',bridge:'M3 16 L8 16 C8 3 16 3 16 16 L21 16 M12 3 L12 7 M12 17 L12 21'}[tool]"/></svg></button>
            <span class="cw-divider"></span><button class="cw-icon" aria-label="Undo" :title="scene?.undo||'Undo'" :disabled="blocked||!scene?.undo" @click="send({kind:'undo'})">↶</button><button class="cw-icon" aria-label="Redo" title="Redo" :disabled="blocked||!scene?.redo" @click="send({kind:'redo'})">↷</button>
            <button class="cw-icon" :aria-label="selected?.locked?'Unlock section':'Lock section'" :title="selected?.locked?'Unlock':'Lock'" :disabled="blocked||!selected" @click="send({kind:'lock'})"><svg viewBox="0 0 24 24" aria-hidden="true"><path :d="selected?.locked?'M7 10 L7 6 C7 0 17 0 17 6 L17 10 M5 10 L19 10 L19 21 L5 21 Z':'M7 10 L7 6 C7 0 17 0 17 6 M5 10 L19 10 L19 21 L5 21 Z'"/></svg></button><button class="cw-icon" :aria-label="selected?.disconnected?'Reconnect':'Cut connection'" :title="selected?.disconnected?'Reconnect':'Cut connection'" :disabled="blocked||!selected?.editable||selected?.locked" @click="send({kind:'disconnect'})">✂</button>
            <span class="cw-spacer"></span><select v-if="scene?.circuits>1" aria-label="Winding circuit" :value="scene?.circuit" :disabled="blocked" @change="send({kind:'circuit',index:Number($event.target.value)})"><option v-for="n in scene?.circuits" :value="n-1">Circuit {{n}}</option></select><select aria-label="Editor view layout" :title="scene?.chart_error" v-model="layout"><option value="linked" :disabled="!scene?.chart">3D + flat</option><option value="3d">3D</option><option value="flat" :disabled="!scene?.chart">Flat</option></select><button class="cw-icon" aria-label="Fit views" title="Fit views" @click="fit">⛶</button>
          </nav>
          <div class="cw-groupbar"><button class="cw-group" v-for="g in legendGroups" :aria-pressed="isolateGroup===g.index" :style="{'--group-color':g.color}" :title="'Isolate group '+(g.index+1)" @click="selectGroup(g.index)"><i></i>Group {{g.index+1}}</button><button :aria-pressed="hideRear" @click="hideRear=!hideRear">Hide rear wires</button><button @click="faceSelected(false)">Face selected area</button><button @click="faceSelected(true)">Local view</button><button :aria-pressed="playOpen" @click="togglePlayback">Winding preview</button><span class="cw-spacer"></span><span class="cw-hint">{{localHint||toolHint}}</span></div>
          <div class="cw-objectbar" v-if="scene?.cut_objects?.length"><button v-for="obj in scene.cut_objects" :aria-pressed="scene.seam_preview?.object_id===obj.id" :disabled="blocked" @click="selectObject(obj)">◇ {{obj.label}} · {{obj.keys.length}} cuts {{obj.locked?'🔒':''}}</button></div>
          <div class="cw-playback" v-if="playOpen"><button aria-label="Previous winding step" @click="stepPlayback(-1)">‹</button><button :disabled="!scene?.sequence?.length" @click="playing=!playing">{{playing?'Pause':'Play'}}</button><button aria-label="Next winding step" @click="stepPlayback(1)">›</button><span>{{step ? (playIndex+1)+' / '+scene.sequence.length+' · '+step.label : 'Complete the circuit to preview winding'}}</span><span v-if="step?.bridge">Over {{step.bridge.other_label}} · {{step.bridge.order_note}}</span></div>
          <div class="cw-sketchbar" v-if="mode==='seam'"><select aria-label="Seam contour group" v-model="seamGroup"><option value="">All groups</option><option v-for="g in groups" :value="g">Group {{g+1}}</option></select><span v-if="scene?.seam_preview">{{scene.seam_preview.cuts.length}} cuts · Generate to apply</span></div>
          <div class="cw-sketchbar" v-if="mode==='shape'">
            <select aria-label="Cut shape" v-model="shapeKind" :disabled="blocked" @change="previewShape()"><option value="plane">Cut plane</option><option value="sphere">Sphere</option><option value="box">Box</option></select>
            <select aria-label="Cut shape contour group" v-model="seamGroup" :disabled="blocked" @change="previewShape(scene?.seam_preview?.shape,true)"><option value="">All groups</option><option value="selected">Selected turn</option><option v-for="g in groups" :value="g">Group {{g+1}}</option></select>
            <select v-if="shapeKind!=='sphere'&&scene?.seam_preview?.shape" aria-label="Cut plane normal" :disabled="blocked" @change="shapeAxis($event.target.value);$event.target.value='' "><option value="">Tilt handle</option><option value="1">Normal X</option><option value="2">Normal Y</option><option value="3">Normal Z</option></select>
            <template v-if="scene?.seam_preview?.shape"><label v-for="i in (shapeKind==='box'?[0,1,2]:[0])">{{shapeKind==='sphere'?'Diameter':shapeKind==='plane'?'Gap':['Width','Height','Depth'][i]}} <input :aria-label="'Cut shape dimension '+(i+1)+' in millimetres'" type="number" min="0.01" step="1" :disabled="blocked" :value="Number((scene.seam_preview.shape.size[i]*2000).toFixed(4))" @change="shapeSize(i,$event.target.value)"> mm</label><span>{{scene.seam_preview.cuts.length}} cuts<span v-if="scene.seam_preview.multiple"> · click a pale opening to switch</span></span></template>
            <button :aria-pressed="placeOnSurface" :disabled="blocked" @click="placeShape">Place on coil</button><button :disabled="blocked" @click="previewShape()">New at selection</button>
            <button aria-label="Discard cut preview" title="Discard preview" :disabled="blocked||!scene?.seam_preview" @click="send({kind:'clear_preview'})">×</button>
          </div>
          <div class="cw-sketchbar" v-if="mode==='bridge'"><span>{{bridgeOther ? 'Crossing selected' : 'Pick crossing'}}</span><label>Gap <input aria-label="Bridge separation in millimetres" type="number" min="0.001" step="0.1" v-model.number="separation"> mm</label><label>Span <input aria-label="Bridge span in millimetres" type="number" min="0.001" step="1" v-model.number="bridgeSpan"> mm</label><button :disabled="blocked||!bridgeOther||!selected?.editable||locked" @click="bridge('over')">Over</button><button :disabled="blocked||!bridgeOther||!selected" @click="bridge('under')">Under</button><button v-if="selected?.bridge" :disabled="blocked||locked" @click="send({kind:'remove_bridge'})">Remove</button></div>
          <div class="cw-preview-state" v-if="scene?.seam_preview"><span v-if="scene.seam_preview.error" role="alert">{{scene.seam_preview.error}}</span><span v-else>Dashed teal: proposed connections · Generate to keep</span><button v-if="mode!=='shape'" aria-label="Discard routing preview" @click="send({kind:'clear_preview'})">×</button></div>
          <div class="cw-issues" v-if="showIssues"><button v-for="issue in scene?.issues" :class="'cw-issue-'+issue.category" :title="issue.detail" @click="issueFocus(issue)">{{issue.category==='unverified'?'Not certified':issue.category==='advisory'?'Estimate':'Failed'}} · {{issue.label}}{{issue.point?' ↗':''}}</button><span v-if="!scene?.issues?.length">{{receipt?.accepted?'Checks passed':'Run Check winding for full validation'}}</span></div>
          <div class="cw-views" :class="{'cw-linked':layout==='linked'&&scene?.chart}"><div class="cw-canvas-wrap" v-show="layout!=='flat'"><canvas ref="canvas" tabindex="0" aria-label="3D winding editor. Drag a cut or guide handle to edit; drag empty space to orbit." :aria-busy="blocked"></canvas><div class="cw-view-title">{{localView ? 'Local projection · distances distorted' : '3D'}}</div><div v-if="!scene" class="cw-loading">{{status||'Open a routed design'}}</div></div><div class="cw-canvas-wrap" v-show="layout!=='3d'&&scene?.chart"><canvas ref="flatCanvas" tabindex="0" aria-label="Flat winding editor. Draw seams, reconnect endpoints, or drag linked cut and guide handles." :aria-busy="blocked"></canvas><div class="cw-view-title" :title="scene?.chart?.label">Flat</div></div></div>
        </main>
        <aside class="cw-inspector" aria-label="Winding inspector" v-show="details">
          <section v-if="mode==='connect'"><label>From<select aria-label="Outgoing endpoint" v-model="portSource"><option value="">Choose endpoint</option><option v-for="p in outputs" :value="p.key">{{p.label}}</option></select></label><label>To<select aria-label="Incoming endpoint" v-model="portTarget"><option value="">Choose endpoint</option><option v-for="p in inputs" :value="p.key">{{p.label}}</option></select></label><button :disabled="blocked||!portSource||!portTarget" @click="connectPorts">Connect endpoints</button></section>
          <section v-if="mode==='bridge'"><label>Crossing partner<select aria-label="Bridge crossing partner" v-model="bridgeOther" @change="bridgePoint=null"><option value="">Choose section</option><option v-for="s in scene?.sections.filter(s=>s.key!==selected?.key&&!s.disconnected)" :value="s.key">{{s.label}}</option></select></label></section>
          <section class="cw-sections"><h2>Sections <span>{{scene?.sections.length || 0}}</span></h2><input type="search" v-model="query" placeholder="Find turn, return, bridge…" aria-label="Find winding section"><div class="cw-section-list" role="listbox" aria-label="Winding sections"><button v-for="row in rows" :key="row.key" role="option" :aria-selected="row.key===scene.selected" :class="{'cw-selected':row.key===scene.selected}" :disabled="blocked" @click="send({kind:'select',key:row.key})"><i class="cw-row-dot" :style="{background:row.color}"></i>{{row.label}}<small>{{row.locked ? 'Locked' : row.disconnected ? 'Open' : row.editable ? 'Connection' : row.cut ? 'Movable cut' : 'Retained'}}</small></button></div></section>
          <section class="cw-properties" v-if="selected"><h2>{{selected.label}}</h2>
            <template v-if="selected.cut"><label>Position along contour [mm]<input aria-label="Cut position in millimetres" type="number" step="0.1" v-model.number="position" :disabled="blocked || locked"></label><label>Opening length [mm]<input aria-label="Cut opening length in millimetres" type="number" step="0.1" v-model.number="span" :disabled="blocked || locked"></label><button class="cw-wide" :disabled="blocked || locked" @click="numericCut">Move cut</button></template>
            <p v-else-if="selected.bridge" class="cw-muted">Explicit overpass. Remove the bridge before moving its guides or attachments.</p>
            <template v-else-if="selected.editable"><label>Guide point<select aria-label="Connection guide" v-model.number="guideIndex" :disabled="blocked || locked"><option v-for="n in Math.max(1,selected.guides.length)" :value="n-1">{{selected.guides.length ? 'Guide '+n : 'New midpoint guide'}}</option></select></label><div class="cw-xyz"><label v-for="(axis,i) in ['X','Y','Z']">{{axis}} [mm]<input :aria-label="'Guide '+axis+' in millimetres'" type="number" step="0.1" v-model.number="xyz[i]" :disabled="blocked || locked"></label></div><div class="cw-controls"><button :disabled="blocked || locked" @click="numericGuide(false)">Set guide</button><button :disabled="blocked || locked" @click="numericGuide(true)">Add</button><button :disabled="blocked || locked || !selected.guides.length" @click="send({kind:'remove_guide',index:guideIndex})">Remove</button></div></template>
            <p v-else class="cw-muted">This retained arc has no unambiguous complete cut. Its source geometry is protected.</p>
          </section>
          <section class="cw-validation" :class="{'cw-passed':receipt?.accepted,'cw-failed':receipt && !receipt.accepted}"><h2>{{receipt ? (receipt.accepted ? 'Engineering checks passed' : 'Engineering checks need attention') : 'Validation pending'}}</h2><template v-if="receipt"><div class="cw-metrics"><span>Current, fixed<strong>{{fmt(current)}} A</strong></span><span>Min. bend radius<strong>{{fmt(bend*1000)}} mm</strong></span><span>RMS ΔB₀<strong>{{fmt(fieldChange?.rms_T*1e6)}} µT</strong></span><span>Peak ΔB₀<strong>{{fmt(fieldChange?.peak_T*1e6)}} µT</strong></span></div><ul><li v-for="reason in receipt.reasons">{{reason}}</li></ul><p class="cw-muted">Fixed current · full engineering checks</p><a v-if="reportUrl" :href="reportUrl" download>Download validation report</a></template><p v-else>Check the edited winding before applying.</p></section>
          <label class="cw-baseline"><input type="checkbox" v-model="ghost"> Show automatic route</label><a v-if="download" :href="download" download>Download draft</a><details class="cw-drafts"><summary>Saved drafts & import</summary><label>Workspace draft<select aria-label="Saved winding draft" v-model="draft"><option value="">Choose a saved draft</option><option v-for="name in drafts" :value="name">{{name}}</option></select></label><button :disabled="blocked || !draft" @click="openDraft">Reopen draft</button><label class="cw-import">Import .cgwinding<input aria-label="Import winding draft" type="file" accept=".cgwinding,application/json" :disabled="blocked" @change="importDraft"></label></details>
        </aside>
      </div>
      <footer class="cw-footer"><span class="cw-circuit" v-if="scene" :class="{'cw-open':!scene.connectivity.complete}"><i></i>{{scene.connectivity.complete?'Continuous':scene.connectivity.free_ends+' open ends'}}</span><button class="cw-check-state" :class="{'cw-warning':receipt&&!receipt.accepted}" @click="showIssues=!showIssues">{{receipt?(receipt.accepted?'Checks passed':receipt.reasons.length+' checks failed'):((scene?.issues?.length||0)+' quick warnings')}}</button><div role="status" aria-live="polite"><span>{{feedback}}</span><span v-if="error||localError" class="cw-error" role="alert">{{error||localError}}</span><span v-if="scene&&!scene.model_matches" class="cw-error">Model changed</span></div><button v-if="blocked" @click="send({kind:'cancel'})">Cancel</button><button aria-label="Validate engineering" :disabled="blocked||!scene?.can_validate" @click="send({kind:'validate'})">Check winding</button><button :disabled="blocked||!scene?.can_apply" @click="send({kind:'apply'})">Apply</button></footer>
    </section>`,
  };
  if(typeof module!=="undefined"&&module.exports) module.exports={Camera,WindingCanvas,closestSegment,exactCutValue,flatScene,chartPick,directionArrows,pathStation,depthTester};
  if(typeof window!=="undefined") window.CoilGenWindingEditor={install(app){app.component("coilgen-winding-editor",component);}};
})();
