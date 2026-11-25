/* PCB previews display server-owned copper geometry. Canvas edits never change a winding. */
(() => {
  'use strict';
  const colors={top:'#168c85',bottom:'#cf7e37'};
  const distance=(p,a,b)=>{const dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy||1)));return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);};
  class PcbCanvas {
    constructor(canvas,flat,select){
      Object.assign(this,{canvas,flat,select,scene:null,selection:'',layer:'both',zoom:1,pan:[0,0],yaw:.6,pitch:.45,hits:[]});
      this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
      canvas.addEventListener('contextmenu',e=>e.preventDefault());
      canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);this.drag={start:[e.offsetX,e.offsetY],pan:[...this.pan],yaw:this.yaw,pitch:this.pitch,button:e.button};});
      canvas.addEventListener('pointermove',e=>{if(!this.drag){this.hover([e.offsetX,e.offsetY]);return;}const d=this.drag,dx=e.offsetX-d.start[0],dy=e.offsetY-d.start[1];if(flat||d.button===2)this.pan=[d.pan[0]+dx,d.pan[1]+dy];else{this.yaw=d.yaw+dx*.01;this.pitch=Math.max(-1.5,Math.min(1.5,d.pitch+dy*.01));}this.draw();});
      canvas.addEventListener('pointerup',e=>{if(!this.drag)return;if(Math.hypot(e.offsetX-this.drag.start[0],e.offsetY-this.drag.start[1])<4)this.pick([e.offsetX,e.offsetY]);this.drag=null;});
      canvas.addEventListener('pointercancel',()=>this.drag=null);
      canvas.addEventListener('wheel',e=>{if(window.matchMedia('(max-width:700px)').matches&&!e.ctrlKey)return;e.preventDefault();this.zoom=Math.max(.2,Math.min(50,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();},{passive:false});
    }
    dispose(){this.resize.disconnect();}
    fit(){this.zoom=1;this.pan=[0,0];this.focusPoint=null;this.draw();}
    focus(point){this.focusPoint=point;this.zoom=2.5;this.pan=[0,0];this.draw();}
    update(scene,selection,layer,options={}){Object.assign(this,{scene,selection,layer,options});this.draw();}
    project(p){if(this.flat)return[p[0],-p[1]];const c=Math.cos(this.yaw),s=Math.sin(this.yaw),a=Math.sin(this.pitch),b=Math.cos(this.pitch);return[p[0]*c-p[1]*s,-(p[0]*s*a+p[1]*c*a+p[2]*b)];}
    draw(){
      const canvas=this.canvas,ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2);
      if(!w||!h)return;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);this.hits=[];
      const scene=this.scene;if(!scene?.board)return;
      const dark=window.matchMedia('(prefers-color-scheme:dark)').matches,neutral=dark?'#658191':'#bbcbd5',text=dark?'#e1edf4':'#263e4e';
      const reference=this.flat?[[0,0],[scene.size_m[0],-scene.size_m[1]]]:scene.support.flat().map(p=>this.project(p));
      const lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];for(const p of reference)for(let i=0;i<2;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
      const center=this.focusPoint?this.project(this.focusPoint):lo.map((v,i)=>(v+hi[i])/2),scale=Math.min((w-45)/Math.max(hi[0]-lo[0],1e-9),(h-45)/Math.max(hi[1]-lo[1],1e-9))*this.zoom;
      const map=p=>this.project(p).map((v,i)=>(v-center[i])*scale+[w/2,h/2][i]+this.pan[i]);
      const line=(points,color,width=1,dashed=false,key='')=>{const ps=points.map(map);if(ps.length<2)return;ctx.beginPath();ctx.moveTo(...ps[0]);ps.slice(1).forEach(p=>ctx.lineTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[5,3]:[]);ctx.stroke();if(key)this.hits.push({key,points:ps});};
      const polygon=(rings,fill,stroke)=>{ctx.beginPath();for(const ring of rings){const ps=ring.map(map);if(!ps.length)continue;ctx.moveTo(...ps[0]);ps.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();}ctx.fillStyle=fill;ctx.fill('evenodd');if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.setLineDash([]);ctx.stroke();}};
      if(this.flat)for(const group of scene.board)polygon(group,dark?'#192731':'#fff',neutral);else for(const points of scene.support)line(points,neutral);
      const chosen=this.selection.split(':')[0];
      for(const t of scene.traces){if(this.layer!=='both'&&this.layer!==t.layer)continue;ctx.globalAlpha=this.selection&&chosen!==t.key?.35:1;const selected=chosen===t.key;
        if(this.flat)for(const group of t.polygons)polygon(group,colors[t.layer],selected?text:null);
        for(const p of t[this.flat?'flat':'wrapped'])line(p,colors[t.layer],selected?3:1.5,t.layer==='bottom',t.key);
        if(!this.flat&&this.options?.compare&&t.original?.length&&(!chosen||selected)){
          line(t.original,'#8055b4',2,true);
          if(selected&&t.comparison){const c=t.comparison;line([c.saved,c.proposed],'#8055b4',3);const p=map(c.saved);ctx.globalAlpha=1;ctx.fillStyle=text;ctx.font='12px system-ui';ctx.fillText(c.distance_mm.toFixed(3)+' mm',p[0]+8,p[1]-8);}
        }
      }ctx.globalAlpha=1;ctx.setLineDash([]);
      for(const e of scene.endpoints){if(this.layer!=='both'&&this.layer!==e.layer)continue;const p=map(e[this.flat?'uv':'xyz']),r=this.selection===e.key?6:4;ctx.beginPath();ctx.arc(...p,r,0,2*Math.PI);ctx.fillStyle=dark?'#192731':'#fff';ctx.fill();ctx.strokeStyle=colors[e.layer];ctx.lineWidth=2;ctx.stroke();this.hits.push({key:e.key,points:[p]});const index=scene.terminals.indexOf(e.key);if(index>=0){ctx.fillStyle=text;ctx.font='12px system-ui';ctx.fillText('AB'[index],p[0]+8,p[1]-7);}}
      for(const via of scene.vias){const p=map(via[this.flat?'uv':'xyz']);ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.strokeStyle=text;ctx.lineWidth=2;ctx.stroke();}
      for(const issue of scene.issues){const p=issue[this.flat?'uv':'xyz'];if(!p?.length)continue;ctx.beginPath();ctx.arc(...map(p),6,0,2*Math.PI);ctx.strokeStyle='#d44e3c';ctx.lineWidth=2;ctx.stroke();}
      if(scene.seam_line?.length){if(this.flat){for(const x of [0,scene.size_m[0]])line([[x,0],[x,scene.size_m[1]]],'#c44e38',1.5,true);}else line(scene.seam_line,'#c44e38',1.5,true);}
      const proposal=this.options?.proposal;
      if(proposal&&!proposal.error){line(proposal[this.flat?'flat':'wrapped'],'#8055b4',3,true);if(!this.flat)for(const via of proposal.via_lines||[])line(via,'#8055b4',3);}
      for(const [i,end] of (this.options?.pins||[]).entries()){
        if(!end)continue;const p=map(end[this.flat?'uv':'xyz']),color=i===0?'#2467ae':'#8055b4';
        ctx.globalAlpha=1;ctx.setLineDash([]);ctx.beginPath();ctx.arc(...p,8,0,2*Math.PI);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke();
        ctx.font='bold 14px system-ui';const dx=i===0?-25:12;ctx.fillStyle=dark?'#192731':'#fff';ctx.fillRect(p[0]+dx-2,p[1]-25,19,20);ctx.fillStyle=color;ctx.fillText('AB'[i],p[0]+dx,p[1]-10);
      }
    }
    nearest(p){let best={distance:11,key:''},endpoint={distance:11,key:''};for(const hit of this.hits){let d=Infinity;if(hit.points.length===1){d=Math.hypot(p[0]-hit.points[0][0],p[1]-hit.points[0][1]);if(d<endpoint.distance)endpoint={distance:d,key:hit.key};}else for(let i=1;i<hit.points.length;i++)d=Math.min(d,distance(p,hit.points[i-1],hit.points[i]));if(d<best.distance)best={distance:d,key:hit.key};}return endpoint.key||best.key;}
    hover(p){const key=this.nearest(p),end=this.scene?.endpoints?.find(e=>e.key===key),trace=this.scene?.traces?.find(t=>t.key===key);this.canvas.title=end?end.label+' · '+end.layer:trace?trace.label+(trace.comparison?' · '+trace.comparison.distance_mm.toFixed(3)+' mm maximum displacement':''):'';}
    pick(p){const key=this.nearest(p);if(key)this.select(key);}
  }
  const component = {
    props:{scene:Object,busy:Boolean,status:String,error:String,download:String,downloadLabel:String,preview:Object},
    emits:['action','close'],
    data:()=>({tab:'Board',selected:'',layer:'both',a:'',b:'',pickEndpoint:'a',connectionLayer:'top',assignLayer:'top',
      seam:0,svg:true,dxf:true,localError:'',showOriginal:false,controlsOpen:true,groupKey:'',locationIndex:0}),
    computed:{
      settings(){return this.scene?.settings||{};},
      endpoints(){return this.scene?.endpoints||[];},
      traces(){return (this.scene?.traces||[]).filter(t=>!t.key.startsWith('link'));},
      groups(){return this.scene?.issue_groups||[];},
      activeGroup(){return this.groups.find(g=>g.key===this.groupKey)||this.groups[0];},
      activeIssue(){const indices=this.activeGroup?.indices;return this.scene?.issues?.[indices?.[this.locationIndex%indices.length]];},
      proposal(){const p=this.preview;return p&&p.identity===this.scene?.identity&&p.a===this.a&&p.b===this.b&&p.layer===this.connectionLayer?p:null;},
      selectedLabel(){
        const end=this.endpoints.find(e=>e.key===this.selected),t=this.scene?.traces?.find(t=>t.key===this.selected.split(':')[0]);
        return (end?.label||t?.label||'Click a trace or endpoint in either view.')+
          (this.showOriginal&&t?.comparison?' · '+t.comparison.distance_mm.toFixed(3)+' mm maximum saved-to-PCB displacement':'');
      },
      performance(){
        const v=this.scene?.evaluation;
        return v?'Target RMS: '+v.target_rms_uT.toPrecision(4)+' µT\nChange from saved coil: '+v.delta_rms_uT.toPrecision(4)+
          ' µT RMS\n'+v.field_budget_summary+'\nTrace R: '+v.trace_resistance_ohm.toPrecision(4)+' Ω · '+
          v.trace_power_W.toPrecision(4)+' W at '+v.current_A.toPrecision(4)+' A\n'+v.scope:
          'Field and electrical review have not been run for this layout.';
      }
    },
    mounted(){
      this.flatView=new PcbCanvas(this.$refs.flat,true,key=>this.pick(key));
      this.wrapView=new PcbCanvas(this.$refs.wrapped,false,key=>this.pick(key));
      this.smallScreen=window.matchMedia('(max-width:700px)');
      this.resizeControls=()=>{this.controlsOpen=!this.smallScreen.matches;};
      this.smallScreen.addEventListener('change',this.resizeControls);this.resizeControls();this.sync();
    },
    beforeUnmount(){clearTimeout(this.previewTimer);this.smallScreen?.removeEventListener('change',this.resizeControls);this.flatView?.dispose();this.wrapView?.dispose();},
    watch:{
      scene(){this.sync();},selected(){this.draw();},layer(){this.draw();},showOriginal(){this.draw();},
      preview(){this.draw();},a(){this.schedulePreview();},b(){this.schedulePreview();},
      connectionLayer(){this.schedulePreview();},tab(){this.schedulePreview();this.draw();},
      busy(value){if(!value)this.schedulePreview();}
    },
    methods:{
      send(kind,values={}){if(this.busy||!this.scene)return;this.localError='';this.$emit('action',{kind,identity:this.scene.identity,values});},
      setting(key,value){if(typeof value==='number'&&!Number.isFinite(value)){this.localError='Enter a valid number.';return;}this.send('settings',{[key]:value});},
      openTab(name){this.tab=name;this.controlsOpen=true;},
      sync(){
        if(!this.scene)return;this.seam=this.settings.seam_deg;
        if(!this.endpoints.some(e=>e.key===this.a))this.a=this.endpoints[0]?.key||'';
        if(!this.endpoints.some(e=>e.key===this.b))this.b=this.endpoints[1]?.key||'';
        if(!this.groups.some(g=>g.key===this.groupKey))this.groupKey=this.groups[0]?.key||'';
        this.locationIndex=0;this.$nextTick(()=>{this.draw();this.schedulePreview();});
      },
      draw(){
        const options={compare:this.showOriginal,proposal:this.tab==='Connect'?this.proposal:null,
          pins:this.tab==='Connect'?[this.endpoints.find(e=>e.key===this.a),this.endpoints.find(e=>e.key===this.b)]:[]};
        this.flatView?.update(this.scene,this.selected,this.layer,options);this.wrapView?.update(this.scene,this.selected,this.layer,options);
      },
      select(key){this.selected=key;},
      pick(key){if(this.endpoints.some(e=>e.key===key)){this[this.pickEndpoint]=key;this.pickEndpoint='b';}this.select(key);},
      fit(){this.flatView?.fit();this.wrapView?.fit();},
      schedulePreview(){
        this.draw();clearTimeout(this.previewTimer);
        if(this.busy||this.tab!=='Connect'||!this.a||!this.b||this.a===this.b)return;
        this.previewTimer=setTimeout(()=>this.send('preview',{a:this.a,b:this.b,layer:this.connectionLayer}),100);
      },
      focusIssue(item){
        if(!item)return;this.select(item.trace);this.layer='both';
        if(item.code==='off_surface')this.showOriginal=true;
        if(item.uv?.length){this.flatView?.focus(item.uv);this.wrapView?.focus(item.xyz);}
      },
      focusGroup(key){this.groupKey=key;this.locationIndex=0;this.focusIssue(this.activeIssue);},
      nextLocation(){this.locationIndex++;this.focusIssue(this.activeIssue);},
      repair(){
        const action=this.activeGroup?.action;
        if(action==='seam')this.send('seam');
        else if(action==='compare'){this.showOriginal=true;this.focusIssue(this.activeIssue);}
        else if(action)this.openTab({connect:'Connect',copper:'Copper',board:'Board',export:'Export'}[action]);
      },
      async load(event){
        const file=event.target.files[0];if(!file)return;
        try{if(file.size>10*1024*1024)throw Error('PCB draft exceeds 10 MiB.');
          const document=JSON.parse(await file.text());this.$emit('action',{kind:'load',identity:this.scene.identity,document});
        }catch(e){this.localError=e.message;}event.target.value='';
      },
      exportPackage(){this.$emit('action',{kind:'export',identity:this.scene.identity,formats:[...(this.svg?['svg']:[]),...(this.dxf?['dxf']:[])]});}
    },
    template:`<section class="cg-pcb" aria-label="PCB layout workspace">
      <header>
        <strong>PCB layout</strong><span class="pcb-source">{{scene?.source||'Opening saved winding…'}}</span>
        <select aria-label="PCB part" :value="scene?.part" :disabled="busy" @change="send('part',{index:+$event.target.value})"><option v-for="(p,i) in scene?.parts||[]" :value="i">{{p}}</option></select>
        <button :disabled="busy||!scene?.undo" @click="send('undo')">Undo</button><button :disabled="busy||!scene?.redo" @click="send('redo')">Redo</button>
        <span class="pcb-grow"></span><button :disabled="busy" @click="$emit('close')">Close</button>
      </header>
      <nav aria-label="PCB steps">
        <button v-for="name in ['Board','Copper','Connect','Review','Export']" :aria-label="name" :aria-pressed="tab===name" @click="openTab(name)">{{name}}<span v-if="name==='Review'&&scene?.review?.errors"> · {{scene.review.errors}}</span></button>
        <span class="pcb-grow"></span><button :aria-expanded="controlsOpen" @click="controlsOpen=!controlsOpen">{{controlsOpen?'Hide controls':'Show controls'}}</button>
      </nav>
      <div class="pcb-summary" :class="{attention:scene?.review?.inspection}" role="status">
        <span>{{scene?.review?.summary||'Loading PCB checks…'}}</span><button @click="openTab('Review')">Review issues</button>
        <span class="pcb-grow"></span><span class="pcb-persistence">{{scene?.persistence}}</span>
      </div>
      <div class="pcb-body" :class="{'controls-hidden':!controlsOpen}"><aside>
        <div v-if="tab==='Board'">
          <h3>Fit the board</h3>
          <label>Layout source<select :value="settings.mode" :disabled="busy" @change="setting('mode',$event.target.value)"><option value="wire">Saved winding</option><option value="spirals">Two-layer group spirals</option></select></label>
          <label>Opening seam · {{seam}}°<input aria-label="Opening seam" type="range" min="0" max="359" step="1" v-model.number="seam" :disabled="busy||scene?.kind!=='cylinder'" @change="setting('seam_deg',seam)"></label>
          <button :disabled="busy||scene?.kind!=='cylinder'" @click="send('seam')">Find seam between windings</button>
          <label>Edge clearance [mm]<input type="number" min="0" max="100" step="0.1" :value="settings.edge_m*1000" :disabled="busy" @change="setting('edge_m',+$event.target.value/1000)"></label>
          <p v-if="scene?.size_m" class="pcb-help">{{scene.size_m.map(v=>(v*1000).toFixed(2)).join(' × ')}} mm · common origin · no mirroring</p>
          <p v-else-if="scene?.problem" class="pcb-help">No flat PCB development for this support.</p>
          <p class="pcb-help">The support defines the board outline. Board extensions and local rerouting are finished in PCB CAD.</p>
        </div>
        <div v-if="tab==='Copper'">
          <h3>Shape the copper</h3>
          <label>Trace width [mm]<input type="number" min="0.001" max="100" step="0.1" :value="settings.width_m*1000" :disabled="busy" @change="setting('width_m',+$event.target.value/1000)"></label>
          <label>Minimum copper gap [mm]<input type="number" min="0" max="100" step="0.1" :value="settings.clearance_m*1000" :disabled="busy" @change="setting('clearance_m',+$event.target.value/1000)"></label>
          <details><summary>Copper &amp; layers</summary>
            <label>Copper thickness [µm]<input type="number" min="1" max="10000" step="1" :value="settings.copper_m*1e6" :disabled="busy" @change="setting('copper_m',+$event.target.value/1e6)"></label>
            <label>Layer separation [mm]<input type="number" min="0" max="100" step="0.01" :value="settings.layer_gap_m*1000" :disabled="busy" @change="setting('layer_gap_m',+$event.target.value/1000)"></label>
          </details>
          <label>Selected trace<select :value="selected.split(':')[0]" @change="select($event.target.value)"><option value="">Select a trace</option><option v-for="t in traces" :value="t.key">{{t.label}}</option></select></label>
          <label>Move to layer<select v-model="assignLayer"><option>top</option><option>bottom</option></select></label>
          <button :disabled="busy||!traces.some(t=>t.key===selected.split(':')[0])" @click="send('layer',{key:selected.split(':')[0],layer:assignLayer})">Apply layer</button>
        </div>
        <div v-if="tab==='Connect'">
          <h3>Complete the circuit</h3>
          <label>Endpoint A<select v-model="a" @change="select(a)"><option v-for="e in endpoints" :value="e.key">{{e.label}}</option></select></label>
          <label>Endpoint B<select v-model="b" @change="select(b)"><option v-for="e in endpoints" :value="e.key">{{e.label}}</option></select></label>
          <label>Next click picks<select v-model="pickEndpoint"><option value="a">Endpoint A</option><option value="b">Endpoint B</option></select></label>
          <label>Connection layer<select v-model="connectionLayer"><option>top</option><option>bottom</option></select></label>
          <div class="pcb-proposal" role="status">
            <strong>Proposed connection</strong><p>{{a===b?'Choose two different endpoints.':proposal?.error||proposal?.summary||'Checking proposal…'}}</p>
            <ul v-if="proposal?.warnings?.length"><li v-for="warning in proposal.warnings">{{warning}}</li></ul>
            <p class="pcb-help">{{proposal?.scope||'The purple line and A/B markers show the pending link.'}}</p>
          </div>
          <button :disabled="busy||!proposal||proposal.error||proposal.duplicate||a===b" @click="send('connect',{a,b,layer:connectionLayer})">Connect A → B</button>
          <button :disabled="busy||!a||!b||a===b" @click="send('terminals',{a,b})">Use A / B as terminals</button>
          <p class="pcb-help">Hover an endpoint for its name. The proposed straight link includes required layer transitions.</p>
          <div class="pcb-connections" v-for="(c,i) in scene?.connections||[]">{{scene.connection_labels[i]}} <button :aria-label="'Remove '+scene.connection_labels[i]" :disabled="busy" @click="send('remove_connection',{index:i})">Remove</button></div>
        </div>
        <div v-if="tab==='Review'">
          <h3>{{scene?.problem?'PCB layout unavailable':(scene?.review?.errors||0)+' PCB issues · '+groups.length+' groups'}}</h3>
          <div class="pcb-issues"><button v-for="group in groups" :aria-pressed="activeGroup?.key===group.key" @click="focusGroup(group.key)">{{group.label}}</button></div>
          <div v-if="activeGroup" class="pcb-guidance">
            <strong>{{activeGroup.title}}</strong><p>{{activeIssue?.text}}</p>
            <p class="pcb-help">Finding {{locationIndex%activeGroup.indices.length+1}} / {{activeGroup.indices.length}}<span v-if="activeIssue?.uv?.length"> · Board location: {{activeIssue.uv.map(v=>(v*1000).toFixed(2)).join(', ')}} mm</span></p>
            <button v-if="activeGroup.indices.length>1" @click="nextLocation">Next finding</button>
            <p class="pcb-help">{{activeGroup.hint}}</p><button v-if="activeGroup.action" :disabled="busy" @click="repair">{{activeGroup.action_label}}</button>
          </div>
          <p v-if="scene?.board&&!groups.length">Geometric PCB checks passed.</p>
          <button :disabled="busy||!scene?.circuit_complete" @click="send('evaluate')">Evaluate PCB coil</button>
          <div class="pcb-performance">{{performance}}</div>
        </div>
        <div v-if="tab==='Export'">
          <h3>{{scene?.review?.inspection?'Inspection artwork':'PCB CAD handoff'}}</h3>
          <p><strong>{{scene?.review?.summary}}</strong></p><button @click="openTab('Review')">Review before export</button>
          <p>Top copper, bottom copper and board outline share a 1:1 millimetre coordinate system.</p>
          <p class="pcb-help">Both layers viewed from above. Drill, plating, pads and fabrication stackup are completed in PCB CAD.</p>
          <p class="pcb-persistence">{{scene?.persistence}}</p>
          <button :disabled="busy||!scene" @click="send('save')">Prepare draft download</button>
          <label>Open matching draft<input type="file" accept=".cgpcb,.json" :disabled="busy||!scene" @change="load"></label>
          <p class="pcb-help">Workspace drafts are separate from the saved coil. Download a copy to move the draft to another workstation.</p>
        </div>
      </aside><main>
        <div class="pcb-toolbar">
          <button v-for="pair in [['both','Both layers'],['top','Top'],['bottom','Bottom']]" :aria-pressed="layer===pair[0]" @click="layer=pair[0]">{{pair[1]}}</button>
          <label><input type="checkbox" v-model="showOriginal">Compare saved path</label><span class="pcb-grow"></span><button @click="fit">Fit views</button>
        </div>
        <div v-if="scene?.problem" class="pcb-error" role="alert">{{scene.problem}}</div>
        <div class="pcb-views">
          <div class="pcb-view"><h3>Flat PCB · mm</h3><canvas ref="flat" aria-label="Flat copper layout; click traces or endpoints to select"></canvas></div>
          <div class="pcb-view"><h3>Installed coil · drag to rotate</h3><canvas ref="wrapped" aria-label="Wrapped copper paths; click to select, drag to rotate, wheel to zoom"></canvas></div>
        </div>
        <div class="pcb-legend"><span><i class="pcb-key"></i>Top copper</span><span><i class="pcb-key bottom"></i>Bottom copper</span><span><i class="pcb-key problem"></i>PCB issue</span><span v-if="showOriginal"><i class="pcb-key saved"></i>Saved path (3D)</span><span v-if="tab==='Connect'"><i class="pcb-key saved"></i>Proposed connection · A / B</span></div>
        <div class="pcb-selection" role="status">{{selectedLabel}}</div>
      </main></div>
      <div class="pcb-status" role="status" aria-live="polite">{{busy?'Updating PCB layout and checks…':status}}<span class="pcb-error" role="alert">{{error||localError}}</span></div>
      <footer>
        <label><input type="checkbox" v-model="svg">SVG copper</label><label><input type="checkbox" v-model="dxf">DXF outlines</label><span class="pcb-grow"></span>
        <a v-if="download" class="pcb-download" :href="download" download>{{downloadLabel}}</a>
        <button class="pcb-primary" :disabled="busy||!scene?.board||(!svg&&!dxf)" @click="exportPackage">{{scene?.review?.export_label||'Export inspection artwork'}}</button>
      </footer>
    </section>`
  };
  if(typeof module!=='undefined'&&module.exports)module.exports={PcbCanvas,component,distance};
  if(typeof window!=='undefined')window.CoilGenPcbLayout={install(app){app.component('coilgen-pcb-layout',component);}};
})();
