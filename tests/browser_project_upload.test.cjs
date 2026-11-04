const { test } = require("node:test");
const assert = require("node:assert/strict");

global.window = {};
global.document = { baseURI: "http://localhost/u/test-session/" };
require("../src/coilgen/web/assets/project_upload.js");

let definition;
window.CoilGenProjectUpload.install({ component(_name, component) { definition = component; } });

function selectFile(size) {
  const events = [];
  const component = { ...definition.data(), ...definition.methods, limit: 1024,
    $emit(name, value) { events.push([name, value]); } };
  let request;
  global.XMLHttpRequest = class {
    constructor() { this.upload = {}; request = this; }
    open(method, url) { this.url = url.href; this.method = method; }
    send(body) { this.body = body; }
  };
  const blob = new Blob([new Uint8Array(size)]);
  Object.defineProperty(blob, "name", { value: "design.coilgen" });
  component.upload({ target: { files: [blob], value: "selected" } });
  return { component, request, events };
}

test("file goes into multipart body, local progress counts bytes, trame receives only a token", () => {
  const { component, request, events } = selectFile(100);
  assert.equal(request.method, "POST");
  assert.equal(request.url, "http://localhost/u/test-session/upload/project");
  assert.equal(request.body.get("file").size, 100);
  request.upload.onprogress({ lengthComputable: true, loaded: 40, total: 100 });
  assert.equal(component.percent, 40);
  assert.deepEqual(events, []);
  request.status = 200;
  request.response = { token: "one-time", name: "design.coilgen" };
  request.onload();
  assert.deepEqual(events, [["uploaded", "one-time"]]);
  assert.equal(component.busy, true);
  definition.watch.result.call(component, { token: "one-time", ok: true, message: "Opened design.coilgen." });
  assert.equal(component.busy, false);
  assert.match(component.status, /Opened design/);
});

test("oversize and rejected uploads produce a visible error without an open event", () => {
  const oversized = selectFile(1025);
  assert.equal(oversized.request, undefined);
  assert.equal(oversized.component.failed, true);
  const { component, request, events } = selectFile(100);
  request.status = 403;
  request.response = { error: "Open your workspace before uploading." };
  request.onload();
  assert.equal(component.status, request.response.error);
  assert.equal(component.busy, false);
  assert.deepEqual(events, []);
});
