import * as THREE from 'three/webgpu';
import {
  float, Fn, If, int, Loop, max, min, mix, mod, mrt, output, screenCoordinate, sRGBTransferEOTF,
  sRGBTransferOETF, step, uniform, uniformArray, uv, vec2, vec3, vec4,
} from 'three/tsl';

/**
 * A pixel art post-process for the Three.js WebGPURenderer (TSL nodes).
 *
 * 1. The scene renders to a low-resolution target (1 / pixelSize of the
 *    screen). The filter runs at that resolution. A nearest upscale then fills
 *    the screen, so each scene pixel is a crisp square.
 * 2. Colors snap to the nearest color of a palette, with a 4 x 4 Bayer dither.
 * 3. A pixel just behind an object gets a dark outline. The upper-left edge
 *    of an object gets a warm rim light.
 * 4. Sharp materials (see `markSharp`) get no dither, and their own edge
 *    pixels get the outline. Use it for characters.
 */

// ---------------------------------------------------------------- palettes

/** The shadow tint of a hue-shifted ramp (violet). */
export const RAMP_SHADOW = 0x2a1f48;
/** The light tint of a hue-shifted ramp (warm yellow). */
export const RAMP_LIGHT = 0xfff0b0;

/**
 * Mixes two hex colors. `t` = 0 gives `a`, and `t` = 1 gives `b`.
 * @param {number} a
 * @param {number} b
 * @param {number} t
 * @returns {number}
 */
export function mixColor(a, b, t) {
  const r = ((a >> 16) & 255) + (((b >> 16) & 255) - ((a >> 16) & 255)) * t;
  const g = ((a >> 8) & 255) + (((b >> 8) & 255) - ((a >> 8) & 255)) * t;
  const bl = (a & 255) + ((b & 255) - (a & 255)) * t;
  return ((r & 255) << 16) | ((g & 255) << 8) | (bl & 255);
}

/**
 * Makes a hue-shifted ramp from dark to light. The dark end moves to violet,
 * and the light end moves to warm yellow. The steps are close, so a dither on
 * a large flat surface does not look like a checkerboard.
 * @param {number} base The middle color, as a hex number.
 * @param {number} [steps=4] The number of shades.
 * @param {number} [spread=0.3] How far the ends move to the tints (0 to 1).
 * @returns {number[]}
 */
export function hueRamp(base, steps = 4, spread = 0.3) {
  const out = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : (i / (steps - 1)) * 2 - 1; // -1 (dark) to 1 (light)
    out.push(t < 0 ? mixColor(base, RAMP_SHADOW, -t * spread * 1.4) : mixColor(base, RAMP_LIGHT, t * spread));
  }
  return out;
}

/** The base colors of the 'diorama' palette. Each one gets a ramp of 4 shades. */
const DIORAMA_BASES = [
  0x6f9a4e, // grass
  0x5a5d63, // dark stone
  0xb9b2a5, // light stone
  0xeadcb8, // cream
  0xa4503a, // brick
  0x9a3a2e, // red roof
  0x5b5f6b, // slate, steel
  0x9a6a3e, // wood
  0xd8a070, // skin
  0x4a6fa5, // blue
  0x6a4a8e, // purple
  0xe0b84a, // gold
  0x3a7d45, // leaf green
  0xd9772b, // orange
  0x2e2a30, // dark cloth
];

/**
 * Named palettes. A palette is a list of hex colors (64 or fewer).
 * 'diorama' has 15 ramps of 4 shades, plus a dark ink and a warm white.
 * @type {Record<string, number[]>}
 */
export const PALETTES = {
  diorama: [...DIORAMA_BASES.flatMap((c) => hueRamp(c)), 0x14101c, 0xfff8e8],
};

// ---------------------------------------------------------------- filter

/**
 * @typedef {object} PixelOptions
 * @property {boolean} enabled false gives a normal render with no filter.
 * @property {number} pixelSize Screen pixels for each scene pixel, an integer from 1 to 8.
 * @property {string | number[] | null} palette A key of PALETTES, a list of hex colors, or null for no color snap.
 * @property {number} dither The dither strength. 0 is off, 1 is strong.
 * @property {number} outlines The outline strength. 0 is off, 1 is black.
 * @property {number} rim The warm rim strength on upper-left edges. 0 is off.
 * @property {number} edgeDepth The depth difference (0 to 1) that makes an edge.
 */

/** @type {PixelOptions} */
export const DEFAULT_OPTIONS = {
  enabled: true,
  pixelSize: 3,
  palette: 'diorama',
  dither: 0.15,
  outlines: 0.7,
  rim: 0.35,
  edgeDepth: 0.0018,
};

const MAX_COLORS = 64;
/** A 4 x 4 Bayer matrix, values 0 to 15. */
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
const SHARP_MRT = mrt({ output, mask: float(0) });

/**
 * Marks a material as sharp. A sharp material gets no dither, and its own
 * edge pixels get the outline. Use it for characters.
 * The material must be a node material (for example MeshStandardNodeMaterial).
 * @template {THREE.Material} M
 * @param {M} material
 * @returns {M} The same material.
 */
export function markSharp(material) {
  material.userData.sharp = true;
  return material;
}

/** Gives each sharp material an MRT node that writes mask = 0, or removes it. */
function setSharpMRT(scene, on) {
  scene.traverse((o) => {
    const mat = o.material;
    if (!mat || Array.isArray(mat) || !mat.userData?.sharp) return;
    const want = on ? SHARP_MRT : null;
    if (mat.mrtNode !== want) {
      mat.mrtNode = want;
      mat.needsUpdate = true;
    }
  });
}

/**
 * @typedef {object} PixelPipeline
 * @property {Readonly<PixelOptions>} options The current options.
 * @property {THREE.Node} node The filtered, low-resolution image as a node.
 *   Use it in your own RenderPipeline to mix the filter with other passes.
 * @property {() => void} render Renders one frame. Call it instead of renderer.render.
 * @property {(o: Partial<PixelOptions>) => void} setOptions Changes the options.
 * @property {() => void} update Updates the size and the sharp materials.
 *   `render` calls it. Call it yourself only if you render `node` with your own pipeline.
 * @property {() => void} dispose Releases the GPU resources.
 */

/**
 * Makes the pixel art filter for a scene and a camera.
 * @param {THREE.WebGPURenderer} renderer
 * @param {THREE.Scene} scene
 * @param {THREE.Camera} camera
 * @param {Partial<PixelOptions>} [initial]
 * @returns {PixelPipeline}
 */
export function createPixelPipeline(renderer, scene, camera, initial = {}) {
  const options = { ...DEFAULT_OPTIONS, ...initial };

  // The scene pass at low resolution: color, depth, and the sharp mask.
  const scenePass = new THREE.PassNode(THREE.PassNode.COLOR, scene, camera, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  });
  scenePass.setMRT(mrt({ output, mask: float(1) }));
  const colorTex = scenePass.getTextureNode('output');
  const depthTex = scenePass.getTextureNode('depth');
  const maskTex = scenePass.getTextureNode('mask');

  const paletteColors = Array.from({ length: MAX_COLORS }, () => new THREE.Vector3());
  const paletteNode = uniformArray(paletteColors, 'vec3');
  const paletteCount = uniform(0, 'int');
  const ditherStrength = uniform(0);
  const outline = uniform(0);
  const rim = uniform(0);
  const edgeDepth = uniform(0);
  const texel = uniform(new THREE.Vector2(1, 1));
  const bayerNode = uniformArray(BAYER.map((v) => (v + 0.5) / 16 - 0.5), 'float');

  // The filter runs once for each low-resolution pixel.
  const filter = Fn(() => {
    const p = uv();
    const sample = (tex, dx, dy) => tex.sample(p.add(vec2(dx, dy).mul(texel))).r;
    const d = sample(depthTex, 0, 0);
    const dE = sample(depthTex, 1, 0), dW = sample(depthTex, -1, 0);
    const dN = sample(depthTex, 0, 1), dS = sample(depthTex, 0, -1);
    const dNW = sample(depthTex, -1, 1);
    const sharp = float(1).sub(sample(maskTex, 0, 0));

    // A neighbor much nearer: this pixel is just behind an object (an outer outline).
    const behind = step(edgeDepth, d.sub(min(min(dE, dW), min(dN, dS))));
    // A neighbor much farther: this pixel is on the edge of an object.
    const onEdge = step(edgeDepth, max(max(dE, dW), max(dN, dS)).sub(d));
    // The upper-left neighbor much farther: the upper-left edge gets a warm rim.
    const upperLeft = step(edgeDepth, dNW.sub(d));

    const c = vec4(colorTex.sample(p));
    const srgb = sRGBTransferOETF(c.rgb).toVar();
    // The rim first (not on the outline pixels of sharp objects), then the palette, then the outlines.
    srgb.assign(mix(srgb, vec3(1.0, 0.9, 0.62), upperLeft.mul(rim).mul(float(1).sub(sharp.mul(onEdge)))));

    If(paletteCount.greaterThan(0), () => {
      const cell = screenCoordinate.xy;
      const index = int(mod(cell.x, 4)).add(int(mod(cell.y, 4)).mul(4));
      const bayer = bayerNode.element(index);
      const target = srgb.add(bayer.mul(ditherStrength).mul(0.12).mul(float(1).sub(sharp)));
      const best = vec3(0).toVar();
      const bestDist = float(1e9).toVar();
      Loop({ start: int(0), end: paletteCount, type: 'int', condition: '<' }, ({ i }) => {
        const q = paletteNode.element(i);
        const e = target.sub(q);
        // A weighted distance: the eye sees green best and blue least.
        const dist = e.x.mul(e.x).mul(0.3).add(e.y.mul(e.y).mul(0.59)).add(e.z.mul(e.z).mul(0.11));
        If(dist.lessThan(bestDist), () => {
          bestDist.assign(dist);
          best.assign(q);
        });
      });
      srgb.assign(best);
    });

    const ink = vec3(0.07, 0.05, 0.1);
    const edge = max(behind.mul(float(1).sub(sharp)), onEdge.mul(sharp));
    srgb.assign(mix(srgb, ink, edge.mul(outline)));
    return vec4(sRGBTransferEOTF(srgb), 1);
  });

  const lowRes = new THREE.RTTNode(filter(), 1, 1, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, type: THREE.UnsignedByteType,
  });
  // A screen-size sample of the RTT texture with nearest filtering is the upscale.
  const pipeline = new THREE.RenderPipeline(renderer, lowRes);

  let lastW = 0, lastH = 0, lastSize = 0;
  const v = new THREE.Vector2();
  const resize = () => {
    renderer.getDrawingBufferSize(v);
    const size = Math.max(1, Math.min(8, Math.round(options.pixelSize)));
    if (v.x === lastW && v.y === lastH && size === lastSize) return;
    lastW = v.x; lastH = v.y; lastSize = size;
    const w = Math.max(1, Math.floor(v.x / size)), h = Math.max(1, Math.floor(v.y / size));
    scenePass.setResolutionScale(1 / size);
    lowRes.width = w;
    lowRes.height = h;
    lowRes.setSize(w, h);
    texel.value.set(1 / w, 1 / h);
  };

  const apply = () => {
    outline.value = options.outlines;
    rim.value = options.rim;
    ditherStrength.value = options.dither;
    edgeDepth.value = options.edgeDepth;
    const list = (typeof options.palette === 'string' ? PALETTES[options.palette] : options.palette) ?? [];
    const n = Math.min(list.length, MAX_COLORS);
    paletteCount.value = n;
    for (let i = 0; i < MAX_COLORS; i++) {
      const c = list[Math.min(i, n - 1)] ?? 0;
      paletteColors[i].set(((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255);
    }
    lastSize = 0;
  };
  apply();

  // New meshes can appear at any time, so check the sharp materials now and then.
  let frame = 0;
  let sharpOn = false;
  const update = () => {
    if (!sharpOn || frame++ % 30 === 0) setSharpMRT(scene, (sharpOn = true));
    resize();
  };

  return {
    options,
    node: lowRes,
    update,
    render() {
      if (!options.enabled) {
        // A material MRT breaks a normal render, so remove it first.
        if (sharpOn) setSharpMRT(scene, (sharpOn = false));
        renderer.render(scene, camera);
        return;
      }
      update();
      pipeline.render();
    },
    setOptions(o) {
      Object.assign(options, o);
      apply();
    },
    dispose() {
      setSharpMRT(scene, false);
      pipeline.dispose();
      scenePass.dispose();
      lowRes.dispose();
    },
  };
}
