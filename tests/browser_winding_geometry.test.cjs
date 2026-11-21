const test=require('node:test');
const assert=require('node:assert/strict');
const {Camera,closestSegment,exactCutValue,flatScene,chartPick,directionArrows}=require('../src/coilgen/web/assets/interconnection_editor.js');
const {pathStation,depthTester,WindingCanvas}=require('../src/coilgen/web/assets/interconnection_editor.js');

test('cut outlines cannot intercept drawing, reconnection, or guide handles',()=>{
  for(const mode of ['draw','connect','select']){
    const view=Object.create(WindingCanvas.prototype),events=[];
    const port={key:'s0',role:'out',point:[0,0,0]},handle={kind:'guide',index:0,point:[0,0,0]};
    Object.assign(view,{mode,busy:false,flat:false,
      scene:{identity:'revision',cut_objects:[{id:'plane',display:{lines:[[[0,0,0],[1,0,0]]]}}]},
      handles:mode==='select'?[handle]:[],camera:{project:p=>p},
      canvas:{focus(){},setPointerCapture(){}},xy:()=>[0,0],nearest:()=>({distance:0}),
      portAt:()=>port,emit:m=>events.push(m)});
    view.down({button:0,pointerId:1,preventDefault(){}});
    assert.deepEqual(events,[]);
    if(mode==='select')assert.equal(view.drag.handle,handle);
    else assert.equal(view.drag.port,port);
  }
});

test('cut outlines remain selectable away from edit handles',()=>{
  const view=Object.create(WindingCanvas.prototype),events=[];
  Object.assign(view,{mode:'select',busy:false,flat:false,
    scene:{cut_objects:[{id:'plane',display:{lines:[[[0,0,0],[1,0,0]]]}}]},
    handles:[],canvas:{focus(){},setPointerCapture(){}},xy:()=>[0,0],
    nearest:()=>({distance:0}),emit:m=>events.push(m)});
  view.down({button:0,pointerId:1,preventDefault(){}});
  assert.deepEqual(events,[{kind:'select_cut_object',object_id:'plane'}]);
});

test('playback reuses static geometry until the directed section changes',()=>{
  global.window={devicePixelRatio:1};
  try{
    let draws=0,dots=0,copies=0;
    const view=Object.create(WindingCanvas.prototype);
    Object.assign(view,{playKey:'s0',playBackground:{},draw(){draws++;},paintCurrent(){dots++;},
      canvas:{width:100,height:100,getContext(){return {setTransform(){},clearRect(){},drawImage(){copies++;}};}}});
    view.playbackFrame('s0',.25);view.playbackFrame('s0',.5);
    assert.equal(draws,0);assert.equal(dots,2);assert.equal(copies,2);
    view.playbackFrame('s1',0);assert.equal(draws,1);
  }finally{delete global.window;}
});

test('winding playback moves at arc length across unequal segments',()=>{
  assert.deepEqual(pathStation([[0,0,0],[1,0,0],[1,3,0]],.5),[1,1,0]);
  assert.deepEqual(pathStation([[0,0,0],[1,0,0]],1),[1,0,0]);
});
test('rear-wire visibility uses mesh depth and retains the near surface',()=>{
  const visible=depthTester([[0,0,1],[100,0,1],[0,100,1]],[[0,1,2]],1e-6);
  assert.equal(visible([20,20,0]),false);
  assert.equal(visible([20,20,1]),true);
  assert.equal(visible([20,20,2]),true);
  assert.equal(visible([110,110,0]),true);
});

test('current arrows reverse with the electrical path and ignore chart seam gaps',()=>{
  const forward=directionArrows([[0,0],[300,0]],100),reverse=directionArrows([[300,0],[0,0]],100);
  assert.equal(forward.length,3);
  assert.ok(forward.every(a=>Math.abs(a.angle)<1e-12));
  assert.ok(reverse.every(a=>Math.abs(Math.abs(a.angle)-Math.PI)<1e-12));
  assert.deepEqual(directionArrows([[0,0],[10,0],null,[1000,0],[1010,0]],100),[]);
});
test('3D camera projection and surface picking preserve SI coordinates',()=>{
  const camera=new Camera();
  const vertices=[[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]];
  camera.fit(vertices);camera.width=800;camera.height=600;
  const point=[.023,-.012,0],screen=camera.project(point);
  const inverse=camera.unproject(...screen);
  inverse.forEach((v,i)=>assert.ok(Math.abs(v-point[i])<1e-12));
  const hit=camera.surface(screen[0],screen[1],vertices,[[0,1,2],[0,2,3]]);
  hit.forEach((v,i)=>assert.ok(Math.abs(v-point[i])<1e-12));
  assert.equal(camera.surface(-500,-500,vertices,[[0,1,2],[0,2,3]]),null);
});
test('screen picking interpolates segment position and clamps endpoints',()=>{
  assert.deepEqual(closestSegment(5,2,[0,0],[10,0]),{t:.5,distance:2});
  assert.deepEqual(closestSegment(12,0,[0,0],[10,0]),{t:1,distance:2});
});
test('dragging a rear surface guide retains its side of a transparent support',()=>{
  const camera=new Camera();camera.yaw=0;camera.pitch=0;
  const vertices=[[-1,-1,-.2],[1,-1,-.2],[0,1,-.2],[-1,-1,.2],[1,-1,.2],[0,1,.2]];
  camera.fit(vertices);camera.width=800;camera.height=600;
  const [x,y]=camera.project([0,0,-.2]);
  assert.ok(Math.abs(camera.surface(x,y,vertices,[[0,1,2],[3,4,5]],-.2)[2]+.2)<1e-12);
  assert.ok(Math.abs(camera.surface(x,y,vertices,[[0,1,2],[3,4,5]])[2]-.2)<1e-12);
});
test('fit keeps the support inside portrait and landscape viewports',()=>{
  const vertices=[];
  for(const x of [-.08,.08])for(const y of [-.08,.08])for(const z of [-.1,.1])vertices.push([x,y,z]);
  for(const [width,height] of [[880,890],[600,300]]){
    const camera=new Camera();camera.fit(vertices);camera.width=width;camera.height=height;
    for(const point of vertices){const [x,y]=camera.project(point);assert.ok(x>=0&&x<=width&&y>=0&&y<=height);}
  }
});
test('numeric cut editing preserves untouched full precision values',()=>{
  assert.equal(exactCutValue(8.00187,.008001873767615741),.008001873767615741*1000);
  assert.equal(exactCutValue(9,.008001873767615741),9);
});
test('flat picking maps barycentric coordinates to the physical facet',()=>{
  const scene={vertices:[[1,2,3],[2,2,3],[1,2,4]],faces:[[0,1,2]],chart:{faces:[[[0,0],[1,0],[0,1]]]},
    sections:[{key:'s0',points:[[1,2,3],[2,2,3]],flat_points:[[0,0],[1,0]],flat_guides:[[.5,0]]}],
    ports:[{key:'s0',role:'out',flat:[0,0]}],flat_baseline:[[0,0],[1,0]],flat_cut_point:[.5,0]};
  assert.deepEqual(chartPick(scene,[.25,.5]),[1.25,2,3.5]);
  assert.equal(chartPick(scene,[1,1]),null);
  const flat=flatScene(scene);
  assert.deepEqual(flat.vertices,[[0,0,0],[1,0,0],[0,1,0]]);
  assert.deepEqual(flat.sections[0].points,[[0,0,0],[1,0,0]]);
  assert.deepEqual(scene.sections[0].points,[[1,2,3],[2,2,3]]);
});
