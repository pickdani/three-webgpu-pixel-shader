import * as THREE from 'three/webgpu';
import { float, Fn, mix, mrt, output, pass, positionLocal, round, screenUV, sin, step, uniform } from 'three/tsl';
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

// A small seeded random generator, so the scene is the same at each load.
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const range = (a, b) => a + rand() * (b - a);

// Wind: a vertex offset along the screen right, in whole scene pixels, at about 8 steps a second.
// The offset grows with the height above `baseY`. The phase moves across the ground like a gust.
const windTime = uniform(0);
const windStrength = uniform(matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 1);
const pixelWorld = uniform(0.08); // the world size of one scene pixel
const cameraRight = uniform(new THREE.Vector3(1, 0, 0));
const TAU = Math.PI * 2;
function wind(baseY, perUnit, pixels) {
  return Fn(() => {
    const p = positionLocal;
    const h = p.y.sub(baseY).mul(perUnit).clamp(0, 1);
    const gust = sin(windTime.mul(TAU / 1.8).sub(p.x.mul(0.5)).sub(p.z.mul(0.3))).mul(0.7)
      .add(sin(windTime.mul(TAU / 1.2).add(p.x.mul(1.7)).add(p.z.mul(1.1))).mul(0.3));
    const steps = round(gust.add(0.35).mul(h).mul(pixels).mul(windStrength));
    return p.add(cameraRight.mul(steps.mul(pixelWorld)));
  })();
}

/** Joins geometries into one, with a vertex color that goes from `dark` at y = 0 to `light` at y = top. */
function merge(parts, dark, light, top) {
  const pos = [], col = [];
  const a = new THREE.Color(dark), b = new THREE.Color(light), c = new THREE.Color();
  for (const g of parts) {
    const p = g.toNonIndexed().attributes.position;
    for (let i = 0; i < p.count; i++) {
      pos.push(p.getX(i), p.getY(i), p.getZ(i));
      c.copy(a).lerp(b, Math.min(1, Math.max(0, p.getY(i) / top)));
      col.push(c.r, c.g, c.b);
    }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  out.computeVertexNormals();
  return out;
}

/** One InstancedMesh for a kind of object. `place(i, m4, color)` sets each instance. */
const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
const euler = new THREE.Euler(), tint = new THREE.Color();
function instances(geometry, material, count, place, parent = scene) {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  for (let i = 0; i < count; i++) {
    tint.set(0xffffff);
    place(i);
    m4.compose(pos, q.setFromEuler(euler), sc);
    mesh.setMatrixAt(i, m4);
    mesh.setColorAt(i, tint);
  }
  mesh.castShadow = mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}
const pick = (list) => list[Math.floor(rand() * list.length)];

// The ground: a grass slab, dirt sides, and a darker bottom layer.
const W = 13, D = 9;
const dirt = mat(0x9a6a3e), grass = mat(0x6a944a);
add(box(W, 0.8, D), [dirt, dirt, grass, dirt, dirt, dirt], 0, -0.4, 0);
add(box(W - 0.04, 0.45, D - 0.04), mat(0x6e4a30), 0, -1.02, 0);

// A ragged top edge on the dirt sides: short strips of sod that hang down.
const edge = [];
for (let x = -W / 2; x < W / 2; x += 0.3) edge.push([x + 0.15, D / 2, 0], [x + 0.15, -D / 2, 0]);
for (let z = -D / 2; z < D / 2; z += 0.3) edge.push([W / 2, z + 0.15, 1], [-W / 2, z + 0.15, 1]);
instances(box(1, 1, 1), mat(0x5d8a42), edge.length, (i) => {
  const [a, b, side] = edge[i];
  const drop = range(0.08, 0.32);
  if (side) pos.set(a + Math.sign(a) * 0.02, -drop / 2 + 0.01, b); else pos.set(a, -drop / 2 + 0.01, b + Math.sign(b) * 0.02);
  euler.set(0, 0, 0);
  sc.set(side ? 0.06 : range(0.22, 0.32), drop, side ? range(0.22, 0.32) : 0.06);
  tint.offsetHSL(0, 0, range(-0.04, 0.03));
});

// A path of flat stones, with pebbles beside it.
const stone = mat(0xc8bca4), stoneDark = mat(0x9a8e78);
const pathPoints = [];
for (let i = 0; i < 13; i++) {
  const t = i / 12;
  const x = -5.6 + t * 9.6;
  const z = 3.6 - t * 3.4 + Math.sin(t * 5) * 0.8;
  pathPoints.push([x, z]);
  const s = add(box(0.75, 0.08, 0.6), i % 3 ? stone : stoneDark, x, 0.04, z);
  s.rotation.y = t * 1.3;
}
instances(new THREE.DodecahedronGeometry(1, 0), mat(0xb4aa98), 70, () => {
  const [x, z] = pick(pathPoints);
  const a = range(0, TAU), r = range(0.25, 0.6);
  pos.set(x + Math.cos(a) * r, 0.02, z + Math.sin(a) * r);
  euler.set(range(0, 3), range(0, 3), 0);
  sc.setScalar(range(0.04, 0.08));
  tint.offsetHSL(0, 0, range(-0.1, 0.05));
});

// A small house: cream walls, a red roof, a door, a window, and a flower bed.
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
add(box(1.5, 0.1, 0.6), mat(0x5a4030), 0.65, 0.05, 1.6, house); // the flower bed
instances(new THREE.OctahedronGeometry(1, 0), mat(0xffffff), 26, (i) => {
  pos.set(0.65 + range(-0.65, 0.65), range(0.16, 0.26), 1.6 + range(-0.24, 0.24));
  euler.set(range(0, 3), range(0, 3), 0);
  sc.setScalar(range(0.09, 0.12));
  tint.set([0xe8c040, 0xf2ece0, 0xd86a8a, 0xe8c040][i % 4]);
}, house);
house.updateMatrixWorld(true);

// A fence with posts along the back and the left side.
const fenceLine = [];
for (let x = -6.1; x <= 0.5; x += 0.8) fenceLine.push([x, -4.1]);
for (let z = -3.3; z <= -0.9; z += 0.8) fenceLine.push([-6.1, z]);
const fenceWood = mat(0xb08a5a);
instances(box(0.12, 0.7, 0.12), fenceWood, fenceLine.length, (i) => {
  pos.set(fenceLine[i][0], 0.35, fenceLine[i][1]);
  euler.set(0, 0, range(-0.06, 0.06));
  sc.set(1, 1, 1);
});
for (const y of [0.25, 0.52]) {
  add(box(6.7, 0.07, 0.05), fenceWood, -2.8, y, -4.1);
  add(box(0.05, 0.07, 3.3), fenceWood, -6.1, y, -2.55);
}

// A lamp post by the path. The lantern glows a little.
add(new THREE.CylinderGeometry(0.05, 0.07, 1.9, 6), mat(0x2e2a30), 0.6, 0.95, 2.7);
add(box(0.24, 0.3, 0.24), new THREE.MeshStandardNodeMaterial({ color: 0xffd27a, emissive: 0xffb050, emissiveIntensity: 0.6, flatShading: true }), 0.6, 2.0, 2.7);
add(box(0.32, 0.06, 0.32), mat(0x2e2a30), 0.6, 2.18, 2.7);

// Trees: a trunk and a cluster of leaf clumps. All the clumps are one instanced mesh.
const trees = [
  [-4.4, -2.6, 1.15, [0x3a7d45, 0x44884c, 0x4f9a50]],
  [-1.6, -3.3, 0.95, [0xd9772b, 0xe08a32, 0xe8a03a]],
  [3.4, 3.4, 0.9, [0x44884c, 0x4f9a50, 0x5aa458]],
  [-5.4, 0.8, 0.8, [0xc4552a, 0xd0652c, 0xd9772b]],
];
const trunk = new THREE.CylinderGeometry(0.14, 0.2, 1.6, 6);
const clumps = [];
for (const [x, z, s, colors] of trees) {
  const t = add(trunk, mat(0x7a5230), x, 0.8 * s, z);
  t.scale.setScalar(s);
  for (let i = 0; i < 11; i++) {
    const a = (i / 11) * TAU * 2.3, r = i === 0 ? 0 : range(0.35, 0.75);
    clumps.push([x + Math.cos(a) * r * s, (1.75 + range(-0.3, 0.75) - r * 0.3) * s, z + Math.sin(a) * r * s,
      range(0.38, 0.55) * s, pick(colors)]);
  }
}
const leaves = instances(new THREE.IcosahedronGeometry(1, 0), mat(0xffffff), clumps.length, (i) => {
  const [x, y, z, r, color] = clumps[i];
  pos.set(x, y, z);
  euler.set(range(0, 3), range(0, 3), 0);
  sc.setScalar(r);
  tint.set(color);
});
leaves.material = leaves.material.clone();
leaves.material.positionNode = wind(1.2, 0.4, 1.2);

// Fallen leaves under the orange trees.
instances(box(1, 1, 1), mat(0xffffff), 34, (i) => {
  const [x, z] = i % 3 ? [-1.6, -3.3] : [-5.4, 0.8];
  const a = range(0, TAU), r = range(0.2, 1.4);
  pos.set(x + Math.cos(a) * r, 0.015, z + Math.sin(a) * r);
  euler.set(0, range(0, 3), 0);
  sc.set(0.12, 0.02, 0.08);
  tint.set(pick([0xd9772b, 0xe8a03a, 0xc4552a]));
});

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

// Grass: many small clumps of blades. The vertex color is dark at the base and light at the tips.
const blades = [];
for (let i = 0; i < 5; i++) {
  const blade = new THREE.ConeGeometry(0.05, range(0.13, 0.22), 3, 1, true);
  blade.translate(0, blade.parameters.height / 2, 0);
  blade.rotateZ(range(-0.4, 0.4));
  blade.rotateY((i / 5) * TAU);
  blade.translate(range(-0.07, 0.07), 0, range(-0.07, 0.07));
  blades.push(blade);
}
const grassMaterial = new THREE.MeshStandardNodeMaterial({ vertexColors: true, flatShading: true, roughness: 1 });
grassMaterial.positionNode = wind(0, 5, 1.4);
const inv = new THREE.Vector3();
const onGround = (x, z) => {
  if (pathPoints.some(([px, pz]) => (px - x) ** 2 + (pz - z) ** 2 < 0.2)) return false;
  house.worldToLocal(inv.set(x, 0, z));
  if (Math.abs(inv.x) < 1.6 && inv.z > -1.3 && inv.z < 1.95) return false; // the house and the flower bed
  return true;
};
const grassSpots = [];
while (grassSpots.length < 1300) {
  // More clumps near the border give a ragged edge; a few hang a little over it.
  const edgeBias = rand() < 0.25;
  const x = edgeBias ? pick([-1, 1]) * range(W / 2 - 0.3, W / 2 + 0.05) : range(-W / 2, W / 2);
  const z = edgeBias && rand() < 0.5 ? pick([-1, 1]) * range(D / 2 - 0.3, D / 2 + 0.05) : range(-D / 2, D / 2);
  if (onGround(x, z)) grassSpots.push([x, z]);
}
const grassClumps = instances(merge(blades, 0x638d46, 0x8fb85c, 0.2), grassMaterial, grassSpots.length, (i) => {
  pos.set(grassSpots[i][0], 0, grassSpots[i][1]);
  euler.set(0, 0, 0); // no rotation, so the wind offset stays along the screen right
  sc.set(1, range(0.7, 1.3), 1);
  tint.set(pick([0xffffff, 0xf0f8e8, 0xe8f0d8, 0xf8fff0]));
});
grassClumps.castShadow = false;

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
sitter.position.set(5.3, 0, 0.3);
sitter.rotation.y = -0.6;

// ---------------------------------------------------------------- pipeline

// The filter, and a plain pass of the same scene. The final image takes the
// plain pass left of the divider and the filter right of it.
// edgeDepth is a fraction of (far - near): 0.012 x 99 is about 1.2 units for this camera.
const pixel = createPixelPipeline(renderer, scene, camera, { pixelSize: 2, dither: 0.1, outlines: 0.5, edgeDepth: 0.012 });
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

const bufferSize = new THREE.Vector2();
function update(t) {
  const w = (Math.PI * 2 * t) / LOOP;
  // The walker goes along the path and back, once for each loop.
  const k = 0.5 - 0.5 * Math.cos(w);
  walker.position.set(-4.2 + k * 3.2, Math.abs(Math.sin(w * 4)) * 0.06, 2.9 - k * 1.4 + Math.sin(k * 3) * 0.3);
  walker.rotation.y = Math.sin(w) > 0 ? 2.1 : -1.05;
  sitter.rotation.y = -0.6 + Math.sin(w) * 0.25;
  if (AUTO) {
    split.value = sweep(t);
  } else {
    // A slow sway of the camera. The sun follows, so the shadows stay to the screen right.
    const az = BASE_AZIMUTH + Math.sin(t * 0.15) * 0.35;
    placeCamera(az);
    placeSun(az);
  }
  // The wind moves in steps of about 1/8 s (29 steps in each loop, so the loop is seamless).
  windTime.value = Math.floor(t / (LOOP / 29)) * (LOOP / 29);
  camera.updateMatrixWorld();
  renderer.getDrawingBufferSize(bufferSize);
  pixelWorld.value = (camera.top - camera.bottom) / (bufferSize.y / pixel.options.pixelSize);
  cameraRight.value.setFromMatrixColumn(camera.matrixWorld, 0);
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
