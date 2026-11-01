import * as THREE from 'three/webgpu';
import { float, mix, mrt, output, pass, screenUV, step, uniform } from 'three/tsl';
import { createPixelPipeline, markSharp } from './src/pixel-shader.js';

// URL flags: ?auto sweeps the divider and hides the controls. ?capture waits
// for demo.frame(t) calls (for a GIF). ?webgl forces the WebGL2 backend.
const flags = new URLSearchParams(location.search);
const AUTO = flags.has('auto') || flags.has('capture');
const CAPTURE = flags.has('capture');
if (AUTO) document.body.classList.add('auto');

const stage = document.getElementById('stage');
const divider = document.getElementById('divider');

const renderer = new THREE.WebGPURenderer({ antialias: false, forceWebGL: flags.has('webgl') });
renderer.setPixelRatio(CAPTURE ? 1 : Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14101c);

// ---------------------------------------------------------------- camera

// A fixed orthographic camera, 30 degrees above the ground.
const VIEW = 6.4;
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 100);
const ELEVATION = THREE.MathUtils.degToRad(30);
const BASE_AZIMUTH = THREE.MathUtils.degToRad(40);
function placeCamera(azimuth) {
  const d = 40;
  camera.position.set(
    d * Math.cos(ELEVATION) * Math.sin(azimuth), d * Math.sin(ELEVATION), d * Math.cos(ELEVATION) * Math.cos(azimuth));
  camera.lookAt(0, 0.6, 0);
}
function fitCamera() {
  const aspect = innerWidth / innerHeight;
  const h = aspect > 1.4 ? VIEW : VIEW * 1.4 / aspect;
  Object.assign(camera, { left: -h * aspect, right: h * aspect, top: h, bottom: -h });
  camera.updateProjectionMatrix();
}
fitCamera();
placeCamera(BASE_AZIMUTH);

// ---------------------------------------------------------------- light

// A warm, low sun and a violet bounce light. Hard shadows suit the pixel look.
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.BasicShadowMap;
scene.add(new THREE.HemisphereLight(0xd8dcff, 0x6a5a7a, 1.7));
const sun = new THREE.DirectionalLight(0xffe2b0, 2.7);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 80 });
sun.shadow.bias = -0.0006;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);

// Puts the sun to the upper left of the screen, so the shadows fall to the screen right.
function placeSun(azimuth) {
  const a = azimuth - THREE.MathUtils.degToRad(60);
  const e = THREE.MathUtils.degToRad(28);
  const d = 30;
  sun.position.set(d * Math.cos(e) * Math.sin(a), d * Math.sin(e), d * Math.cos(e) * Math.cos(a));
}
placeSun(BASE_AZIMUTH);

// ---------------------------------------------------------------- models

const materials = new Map();
function mat(color) {
  if (!materials.has(color)) {
    materials.set(color, new THREE.MeshStandardNodeMaterial({ color, flatShading: true, roughness: 1 }));
  }
  return materials.get(color);
}
function add(geometry, material, x, y, z, parent = scene) {
  const m = new THREE.Mesh(geometry, material);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  parent.add(m);
  return m;
}
const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);

// The ground: a grass slab with dirt sides.
const dirt = mat(0x9a6a3e), grass = mat(0x6f9a4e);
add(box(13, 1.2, 9), [dirt, dirt, grass, dirt, dirt, dirt], 0, -0.6, 0);

// A path of flat stones.
const stone = mat(0xc8bca4), stoneDark = mat(0x9a8e78);
for (let i = 0; i < 13; i++) {
  const t = i / 12;
  const x = -5.6 + t * 9.6;
  const z = 3.6 - t * 3.4 + Math.sin(t * 5) * 0.8;
  const s = add(box(0.75, 0.08, 0.6), i % 3 ? stone : stoneDark, x, 0.04, z);
  s.rotation.y = t * 1.3;
}

// A small house: cream walls, a red roof, a door, and a window.
const house = new THREE.Group();
house.position.set(2.8, 0, -1.6);
house.rotation.y = -0.15;
scene.add(house);
add(box(3, 2, 2.4), mat(0xeadcb8), 0, 1, 0, house);
// The gable roof: a triangle across the house depth, extruded along the width. Both have a 0.2 overhang.
const gable = new THREE.Shape([new THREE.Vector2(-1.4, 0), new THREE.Vector2(1.4, 0), new THREE.Vector2(0, 1.2)]);
const roofGeometry = new THREE.ExtrudeGeometry(gable, { depth: 3.4, bevelEnabled: false });
roofGeometry.translate(0, 0, -1.7);
roofGeometry.rotateY(Math.PI / 2); // the triangle now spans z, the extrusion spans x
add(roofGeometry, mat(0x9a3a2e), 0, 2, 0, house);
add(box(0.6, 1.1, 0.08), mat(0x5e3e24), -0.6, 0.55, 1.22, house);
add(box(0.6, 0.5, 0.08), mat(0x9ec9d6), 0.65, 1.15, 1.22, house);
add(box(0.45, 1.4, 0.45), mat(0x7a3a2c), 0.9, 3.05, -0.4, house); // chimney: the base is inside the roof

// Trees: a trunk and a clump of leaves.
function tree(x, z, s, leaves) {
  const g = new THREE.Group();
  g.position.set(x, 0, z);
  g.scale.setScalar(s);
  scene.add(g);
  add(new THREE.CylinderGeometry(0.14, 0.2, 1.4, 6), mat(0x7a5230), 0, 0.7, 0, g);
  const ball = new THREE.IcosahedronGeometry(1, 0);
  const parts = [[0, 1.9, 0, 0.85], [0.55, 1.55, 0.2, 0.6], [-0.5, 1.6, -0.15, 0.62], [0.1, 2.5, -0.1, 0.55]];
  parts.forEach(([px, py, pz, r], i) => {
    const m = add(ball, mat(leaves[i % leaves.length]), px, py, pz, g);
    m.scale.setScalar(r);
    m.rotation.set(i, i * 2, 0);
  });
}
tree(-4.4, -2.6, 1.15, [0x3a7d45, 0x4f9a50]);
tree(-1.6, -3.3, 0.95, [0xd9772b, 0xe8a03a]);
tree(5.4, 2.6, 0.9, [0x4f9a50, 0x3a7d45]);
tree(-5.4, 0.8, 0.8, [0xc4552a, 0xd9772b]);

// A bench by the path.
const bench = new THREE.Group();
bench.position.set(-1.2, 0, 2.6);
bench.rotation.y = 0.5;
scene.add(bench);
const wood = mat(0x9a6a3e), woodDark = mat(0x5e3e24);
add(box(1.6, 0.1, 0.45), wood, 0, 0.45, 0, bench);
add(box(1.6, 0.35, 0.08), wood, 0, 0.75, -0.2, bench);
for (const x of [-0.65, 0.65]) add(box(0.1, 0.45, 0.4), woodDark, x, 0.22, 0, bench);

// Rocks.
const rock = new THREE.DodecahedronGeometry(1, 0);
[[1.2, 3.1, 0.35], [4.8, 0.2, 0.5], [-3.2, -0.6, 0.3], [-5.6, 3.6, 0.45], [5.6, -3.4, 0.4]].forEach(([x, z, r], i) => {
  const m = add(rock, mat(i % 2 ? 0x9a8e78 : 0x8a8478), x, r * 0.4, z);
  m.scale.set(r * 1.3, r, r);
  m.rotation.set(i, i * 1.7, 0);
});

// Grass tufts: one instanced mesh.
const tuftCount = 70;
const tufts = new THREE.InstancedMesh(new THREE.ConeGeometry(0.1, 0.35, 4), mat(0x84ad5c), tuftCount);
const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
for (let i = 0; i < tuftCount; i++) {
  pos.set(-6.2 + rand() * 12.4, 0.15, -4.2 + rand() * 8.4);
  q.setFromEuler(new THREE.Euler((rand() - 0.5) * 0.4, rand() * 3, (rand() - 0.5) * 0.4));
  m4.compose(pos, q, sc.setScalar(0.7 + rand() * 0.6));
  tufts.setMatrixAt(i, m4);
}
tufts.castShadow = tufts.receiveShadow = true;
scene.add(tufts);

// Two figures. Sharp materials keep flat colors and get an inner outline.
const sharp = (color) => markSharp(new THREE.MeshStandardNodeMaterial({ color, flatShading: true, roughness: 1 }));
function figure(cloth, hair) {
  const g = new THREE.Group();
  g.scale.setScalar(1.4);
  scene.add(g);
  add(box(0.42, 0.55, 0.3), sharp(cloth), 0, 0.72, 0, g);
  add(box(0.16, 0.45, 0.2), sharp(0x2e2a30), -0.11, 0.22, 0, g);
  add(box(0.16, 0.45, 0.2), sharp(0x2e2a30), 0.11, 0.22, 0, g);
  add(box(0.32, 0.32, 0.32), sharp(0xd8a070), 0, 1.18, 0, g);
  add(box(0.36, 0.12, 0.36), sharp(hair), 0, 1.38, 0, g);
  return g;
}
const walker = figure(0x4a6fa5, 0x5a3a22);
const sitter = figure(0x6a4a8e, 0xc9a25a);
sitter.position.set(4.4, 0, 0.6);
sitter.rotation.y = -0.4;

// ---------------------------------------------------------------- pipeline

// The filter, and a plain pass of the same scene. The final image takes the
// plain pass left of the divider and the filter right of it.
// edgeDepth is a fraction of (far - near): 0.012 x 99 is about 1.2 units for this camera.
const pixel = createPixelPipeline(renderer, scene, camera, { edgeDepth: 0.012 });
const plain = pass(scene, camera);
plain.setMRT(mrt({ output, mask: float(1) })); // the same outputs as the sharp materials
const split = uniform(0.5);
const pipeline = new THREE.RenderPipeline(renderer, mix(plain, pixel.node, step(split, screenUV.x)));

// ---------------------------------------------------------------- controls

for (const key of ['pixelSize', 'dither', 'outlines', 'rim']) {
  const input = document.getElementById(key);
  input.value = pixel.options[key];
  input.addEventListener('input', () => pixel.setOptions({ [key]: Number(input.value) }));
}
const paletteSelect = document.getElementById('palette');
paletteSelect.addEventListener('change', () =>
  pixel.setOptions({ palette: paletteSelect.value === 'none' ? null : paletteSelect.value }));

let dragging = false;
const drag = (e) => { split.value = Math.min(1, Math.max(0, e.clientX / innerWidth)); };
stage.addEventListener('pointerdown', (e) => { dragging = true; stage.setPointerCapture(e.pointerId); drag(e); });
stage.addEventListener('pointermove', (e) => { if (dragging) drag(e); });
stage.addEventListener('pointerup', () => { dragging = false; });

addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight); fitCamera(); });

// ---------------------------------------------------------------- animation

/** One loop of the automatic sweep, in seconds. All motion in ?auto repeats with this period. */
const LOOP = 3.6;
const ease = (x) => x * x * (3 - 2 * x);

/** The divider position for the automatic sweep: plain, sweep to pixel, hold, fast sweep back. */
function sweep(t) {
  const s = ((t % LOOP) + LOOP) % LOOP;
  if (s < 0.6) return 1;
  if (s < 2.6) return 1 - ease((s - 0.6) / 2);
  if (s < 3.4) return 0;
  return ease((s - 3.4) / 0.2);
}

function update(t) {
  const w = (Math.PI * 2 * t) / LOOP;
  // The walker goes along the path and back, once for each loop.
  const k = 0.5 - 0.5 * Math.cos(w);
  walker.position.set(-4.2 + k * 3.2, Math.abs(Math.sin(w * 4)) * 0.06, 2.9 - k * 1.4 + Math.sin(k * 3) * 0.3);
  walker.rotation.y = Math.sin(w) > 0 ? 2.1 : -1.05;
  sitter.rotation.y = -0.4 + Math.sin(w) * 0.25;
  if (AUTO) {
    split.value = sweep(t);
  } else {
    // A slow sway of the camera. The sun follows, so the shadows stay to the screen right.
    const az = BASE_AZIMUTH + Math.sin(t * 0.15) * 0.35;
    placeCamera(az);
    placeSun(az);
  }
  divider.style.left = `${split.value * 100}%`;
  pixel.update();
  pipeline.render();
}

await renderer.init();
window.demo = {
  backend: renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl2',
  /** Renders the frame at time t (seconds). */
  frame: (t) => update(t),
};
if (!CAPTURE) {
  renderer.setAnimationLoop((now) => update(now / 1000));
}
window.demo.ready = true;
