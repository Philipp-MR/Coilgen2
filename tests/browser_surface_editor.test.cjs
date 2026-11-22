const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  rotation, newPrimitive, primitivePreview, triangleHit, roots, SurfaceCanvas, component,
} = require('../src/coilgen/web/assets/surface_editor.js');

function close(actual, expected, tolerance = 1e-12) {
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < tolerance));
}

test('display transforms use SI and Rz Ry Rx, including combined rotations', () => {
  close(rotation([1, 0, 0], [0, 0, Math.PI / 2]), [0, 1, 0]);
  close(rotation([0, 1, 0], [Math.PI / 2, Math.PI / 2, 0]), [1, 0, 0]);
  const plane = newPrimitive('plane', 'p');
  plane.center_m = [.1, -.2, .3];
  plane.rotation_rad = [Math.PI / 2, 0, 0];
  const preview = primitivePreview(plane);
  preview.vertices.forEach(vertex => assert.ok(Math.abs(vertex[1] + .2) < 1e-12));
  assert.equal(preview.triangles.length, 2);
});

test('picking is two-sided and ignores misses and hits behind the camera', () => {
  const tri = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];
  assert.equal(triangleHit([.2, .2, 1], [0, 0, -1], ...tri), 1);
  assert.equal(triangleHit([.2, .2, -1], [0, 0, 1], ...tri), 1);
  assert.equal(triangleHit([2, 2, 1], [0, 0, -1], ...tri), null);
  assert.equal(triangleHit([.2, .2, 1], [0, 0, 1], ...tri), null);
});

test('nested subtraction consumes operands without losing their editable recipes', () => {
  const doc = {nodes: [newPrimitive('cylinder', 'a'), newPrimitive('sphere', 'b'),
    {id: 'c', kind: 'subtract', base: 'a', tool: 'b'}, newPrimitive('box', 'd'),
    {id: 'e', kind: 'subtract', base: 'c', tool: 'd'}]};
  assert.deepEqual(roots(doc).map(node => node.id), ['e']);
  assert.equal(doc.nodes.length, 5);
});

function canvasHarness() {
  const renderer = Object.create(SurfaceCanvas.prototype);
  Object.assign(renderer, {
    canvas: {clientHeight: 400, setPointerCapture() {}},
    doc: {nodes: [newPrimitive('cylinder', 'a')]}, selected: 'a', mode: 'move', axis: 0,
    yaw: -1.05, pitch: .48, scale: .2, blocked: false, feedback() {}, draw() {}, commits: 0,
    change(node, commit) { if (node) this.doc.nodes[0] = node; if (commit) this.commits++; },
  });
  return renderer;
}

test('drag updates only the constrained coordinate live and commits once on release', () => {
  const renderer = canvasHarness();
  renderer.down({button: 0, clientX: 100, clientY: 100, pointerId: 1});
  renderer.move({clientX: 125, clientY: 110});
  assert.notEqual(renderer.doc.nodes[0].center_m[0], 0);
  close(renderer.doc.nodes[0].center_m.slice(1), [0, 0]);
  assert.equal(renderer.commits, 0);
  renderer.up();
  assert.equal(renderer.commits, 1);
  renderer.up();
  assert.equal(renderer.commits, 1);
});

test('rotation updates radians live; cancel restores the original pose', () => {
  const renderer = canvasHarness();
  renderer.mode = 'rotate';
  renderer.axis = 2;
  renderer.down({button: 0, clientX: 0, clientY: 0, pointerId: 1});
  renderer.move({clientX: 25, clientY: 0});
  close(renderer.doc.nodes[0].rotation_rad, [0, 0, .225]);
  renderer.cancel();
  close(renderer.doc.nodes[0].rotation_rad, [0, 0, 0]);
  assert.equal(renderer.commits, 0);
});

test('numeric edits preserve fractional mm and degrees in the SI recipe', () => {
  const context = {node: newPrimitive('cylinder', 'a'), update() { this.updated = true; }};
  component.methods.number.call(context, {target: {valueAsNumber: 81.125, checkValidity: () => true}}, 'parameters', 'radius', 1000);
  assert.equal(context.node.parameters.radius, .081125);
  component.methods.number.call(context, {target: {valueAsNumber: 12.375, checkValidity: () => true}}, 'rotation_rad', 1, 180 / Math.PI);
  assert.ok(Math.abs(context.node.rotation_rad[1] - 12.375 * Math.PI / 180) < 1e-12);
  assert.ok(context.updated);
});
