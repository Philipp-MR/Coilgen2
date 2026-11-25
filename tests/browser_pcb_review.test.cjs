const {test}=require('node:test');
const assert=require('node:assert/strict');
const {component,PcbCanvas}=require('../src/coilgen/web/assets/pcb_layout.js');

test('group navigation includes findings without spatial witnesses',()=>{
  const issues=[{code:'branch',uv:[0,0]},{code:'terminals',uv:[]}];
  const view={scene:{issues},activeGroup:{indices:[0,1],locations:[0]},locationIndex:0};
  assert.equal(component.computed.activeIssue.call(view),issues[0]);
  view.locationIndex=1;
  assert.equal(component.computed.activeIssue.call(view),issues[1]);
  view.locationIndex=2;
  assert.equal(component.computed.activeIssue.call(view),issues[0]);
});

test('endpoint selection takes priority over a nearby trace',()=>{
  const view={hits:[{key:'trace',points:[[0,0],[20,0]]},{key:'other:end',points:[[8,4]]}]};
  assert.equal(PcbCanvas.prototype.nearest.call(view,[8,0]),'other:end');
});

test('stale proposals cannot appear on a changed source or endpoints',()=>{
  const view={scene:{identity:'new'},a:'a',b:'b',connectionLayer:'top',
    preview:{identity:'old',a:'a',b:'b',layer:'top'}};
  assert.equal(component.computed.proposal.call(view),null);
  view.preview.identity='new';
  assert.equal(component.computed.proposal.call(view),view.preview);
  view.b='c';
  assert.equal(component.computed.proposal.call(view),null);
});
