import React, { useEffect, useRef } from 'react';
import type { JarvisFaceState } from '../lib/jarvisNeuralController';

export interface JarvisFaceRigProps {
  face: JarvisFaceState;
  isSpeaking: boolean;
  isBooting?: boolean;
  bootPercent?: number;
  morphDimmed?: boolean;
  width?: number;
  height?: number;
  onClick?: () => void;
}

// ============================================================================
// DETERMINISTIC 3D HUMAN NEURAL CONSTELLATION TOPOLOGY (ZERO Math.random)
// ============================================================================

type VertexGroup =
  | 'shell'
  | 'halo'
  | 'earL'
  | 'earR'
  | 'browL'
  | 'browR'
  | 'lidUpperL'
  | 'lidLowerL'
  | 'lidUpperR'
  | 'lidLowerR'
  | 'orbitalL'
  | 'orbitalR'
  | 'noseTip'
  | 'nostril'
  | 'lipUpper'
  | 'lipLower'
  | 'philtrum';

interface MeshVertex {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
  group: VertexGroup;
  starIntensity: number; // 0 = no dot, 0.25..1.0 = bioluminescent constellation star pinpoint
  paramT: number; // normalized parameter along feature loop [0..1]
  jawWeight: number; // 3D mandibular rotation/drop weight [0..1]
  lipWeight: number; // 3D articulatory lip deformation weight [0..1]
  cheekWeight: number; // Duchenne zygomatic warmth lift [0..1]
}

interface MeshEdge {
  a: number;
  b: number;
  baseTier: 0 | 1 | 2 | 3 | 4;
}

interface HumanConstellationTopology {
  vertices: MeshVertex[];
  edges: MeshEdge[];
  leftPalpebralLoop: number[];
  rightPalpebralLoop: number[];
  mouthInnerLoop: number[];
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function gauss(x: number, mu: number, sigma: number): number {
  const d = (x - mu) / sigma;
  return Math.exp(-0.5 * d * d);
}

function gauss2D(x: number, y: number, mx: number, my: number, sx: number, sy: number): number {
  const dx = (x - mx) / sx;
  const dy = (y - my) / sy;
  return Math.exp(-0.5 * (dx * dx + dy * dy));
}

function parseHexRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '').trim();
  if (clean.length === 6) {
    const n = parseInt(clean, 16);
    if (!Number.isNaN(n)) {
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
  }
  return [56, 189, 248];
}

/**
 * Classical Loomis-proportioned 3D human cranio-cervical surface at height `y` (`[-1.14 .. +1.04]`)
 * and azimuth `theta` (`0` = anterior facial midline `+Z`, `±pi/2` = lateral profile `±X`, `±pi` = occiput `-Z`).
 */
function evalCranioCervicalPoint(y: number, theta: number): {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
} {
  const absTh = Math.abs(theta);
  const sinTh = Math.sin(theta);
  const cosTh = Math.cos(theta);

  let rx = 0.335;
  let rzFront = 0.34;
  let rzBack = 0.35;

  if (y >= 0.30) {
    // Tall, noble neurocranium dome (`y = 0.30 .. 1.02`)
    const u = clamp((y - 0.30) / 0.72, 0.0, 1.0);
    const dome = Math.sqrt(Math.max(0.0004, 1.0 - Math.pow(u, 1.96)));
    rx = 0.632 * dome;
    rzFront = 0.672 * dome;
    rzBack = 0.730 * dome;
  } else {
    // Pure Cranio-Facial Ovoid (`y = 0.30 .. headBottomY(theta)`, ZERO neck, convex rounded chin!)
    const headBottomY = -0.475 - 0.092 * (0.5 + 0.5 * cosTh); // -0.567 at chin, -0.475 at occiput
    const tFace = clamp((0.30 - y) / (0.30 - headBottomY), 0.0, 1.0);

    const templePinch = -0.018 * gauss(y, 0.16, 0.095);
    const malarWidth = 0.024 * gauss(y, -0.035, 0.095);
    const gonionWidth = 0.022 * gauss(y, -0.33, 0.085) * smoothstep(0.15, 0.85, Math.abs(sinTh));

    // Convex superellipsoid cap: horizontal tangent at tFace -> 1 gives a smooth, rounded U-shaped human chin!
    const roundCap = Math.pow(Math.max(0.0004, 1.0 - Math.pow(tFace, 2.18)), 0.54);
    rx = 0.632 * roundCap + (templePinch + malarWidth + gonionWidth) * (1.0 - Math.pow(tFace, 3.0));
    rzFront = 0.535 * roundCap + 0.137 * (1.0 - Math.pow(tFace, 2.4));
    rzBack = 0.730 * Math.pow(Math.max(0.0004, 1.0 - Math.pow(tFace, 1.95)), 0.58);
  }

  // Smoothly shift lower cranial/mandibular axis slightly forward toward the submental chin pad
  const submentalShiftZ = y < 0.30 ? 0.155 * Math.pow(clamp((0.30 - y) / 0.867, 0.0, 1.0), 2.1) : 0.0;
  const rz = cosTh >= 0.0 ? rzFront : rzBack;
  let x = rx * sinTh;
  let z = submentalShiftZ + rz * cosTh;

  // Sculpted Anterior Human Facial Planes (`cosTh > 0`)
  if (cosTh > 0.02 && y > -0.58 && y < 0.60) {
    const frontMask = smoothstep(0.02, 0.30, cosTh) * smoothstep(-0.568, -0.49, y);

    // Superciliary brow ridge
    const browArch =
      0.030 *
      gauss(y, 0.192 - 0.020 * gauss(Math.abs(x), 0.38, 0.13), 0.044) *
      (1.0 - 0.25 * gauss(x, 0.0, 0.065)) *
      smoothstep(0.52, 0.34, Math.abs(x));

    // Concave orbital sockets
    const orbitL = -0.092 * gauss2D(x, y, -0.272, 0.055, 0.115, 0.068);
    const orbitR = -0.092 * gauss2D(x, y, 0.272, 0.055, 0.115, 0.068);

    // Zygomatic cheekbones & subtle buccal sculpting
    const cheekL = 0.044 * gauss2D(x, y, -0.40, -0.055, 0.125, 0.098);
    const cheekR = 0.044 * gauss2D(x, y, 0.40, -0.055, 0.125, 0.098);
    x += (-0.015 * gauss2D(x, y, -0.49, -0.04, 0.11, 0.10) + 0.015 * gauss2D(x, y, 0.49, -0.04, 0.11, 0.10)) * frontMask;
    const buccalHollow =
      -0.016 * (gauss2D(x, y, -0.34, -0.29, 0.10, 0.10) + gauss2D(x, y, 0.34, -0.29, 0.10, 0.10));

    // Slender nasal pyramid + rounded pronasale tip + compact alar wings
    const bridgeProgress = smoothstep(0.15, -0.150, y) * (1.0 - smoothstep(-0.155, -0.208, y));
    const noseBridge = 0.132 * bridgeProgress * gauss(x, 0.0, 0.048 + 0.014 * bridgeProgress);
    const noseBall = 0.078 * gauss2D(x, y, 0.0, -0.152, 0.046, 0.040);
    const noseAlaL = 0.048 * gauss2D(x, y, -0.080, -0.172, 0.034, 0.030);
    const noseAlaR = 0.048 * gauss2D(x, y, 0.080, -0.172, 0.034, 0.030);

    // Dental barrel, calm classical lips, mentolabial sulcus & rounded chin pad
    const oralBarrel = 0.040 * gauss2D(x, y, 0.0, -0.34, 0.175, 0.082);
    const philtrumGroove = -0.013 * gauss2D(x, y, 0.0, -0.258, 0.024, 0.040);
    const upperLip = 0.030 * gauss(y, -0.318, 0.018) * gauss(x, 0.0, 0.115);
    const oralCleft = -0.018 * gauss(y, -0.345, 0.010) * gauss(x, 0.0, 0.135);
    const lowerLip = 0.034 * gauss(y, -0.378, 0.022) * gauss(x, 0.0, 0.108);
    const mentolabialDip = -0.024 * gauss2D(x, y, 0.0, -0.435, 0.115, 0.032);
    const chinPad = 0.048 * gauss2D(x, y, 0.0, -0.505, 0.125, 0.048);

    z +=
      frontMask *
      (browArch +
        orbitL +
        orbitR +
        cheekL +
        cheekR +
        buccalHollow +
        noseBridge +
        noseBall +
        noseAlaL +
        noseAlaR +
        oralBarrel +
        philtrumGroove +
        upperLip +
        oralCleft +
        lowerLip +
        mentolabialDip +
        chinPad);
  }

  const invRx = 1.0 / Math.max(0.12, rx);
  const invRz = 1.0 / Math.max(0.12, rz);
  let nx = x * invRx * invRx;
  let ny = y > 0.30 ? (y - 0.30) * 1.45 : y < -0.38 ? (y + 0.36) * 2.1 : (y - 0.05) * 0.22;
  let nz = (z - submentalShiftZ) * invRz * invRz;
  const nLen = Math.hypot(nx, ny, nz) || 1.0;
  nx /= nLen;
  ny /= nLen;
  nz /= nLen;

  return { x, y, z, nx, ny, nz };
}

/**
 * Deterministic `1,024`-point 3D cranio-facial cloud (`[numPoints * 4]`: `x, y, z, salience`)
 * sampled directly on JARVIS's Loomis human facial surface so `PointScarFlow` OT manifolds
 * morph smoothly out of (and back into) JARVIS's exact 3D face in place.
 */
export function buildJarvisFacePointCloudF32(numPoints = 1024): Float32Array {
  const out = new Float32Array(numPoints * 4);
  const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
  for (let i = 0; i < numPoints; i++) {
    const u = (i + 0.5) / numPoints;
    const s = 1.0 - 2.0 * u; // +1 (crown) .. -1 (chin)
    const sWarped = 0.72 * s + 0.28 * s * s * s;
    const y = 0.225 + 0.785 * sWarped; // [-0.560, +1.010]

    // Azimuth topologically matched to PointScarFlow's golden-spiral initial sphere
    const th = i * goldenAngle;
    let thetaRaw = Math.PI * 0.5 - (th % (Math.PI * 2.0));
    while (thetaRaw > Math.PI) thetaRaw -= Math.PI * 2.0;
    while (thetaRaw < -Math.PI) thetaRaw += Math.PI * 2.0;
    const signTh = thetaRaw < 0.0 ? -1.0 : 1.0;
    const normTh = Math.abs(thetaRaw) / Math.PI;
    const theta = signTh * Math.pow(normTh, 1.38) * Math.PI;

    const pt = evalCranioCervicalPoint(y, theta);
    const front = Math.max(0.0, Math.cos(theta));
    const landmarkSal =
      0.42 * gauss2D(pt.x, pt.y, -0.272, 0.055, 0.12, 0.055) +
      0.42 * gauss2D(pt.x, pt.y, 0.272, 0.055, 0.12, 0.055) +
      0.38 * gauss2D(pt.x, pt.y, 0.0, -0.145, 0.075, 0.095) +
      0.40 * gauss2D(pt.x, pt.y, 0.0, -0.348, 0.14, 0.048) +
      0.32 * gauss(pt.y, 0.19, 0.045) * front +
      0.26 * (gauss2D(pt.x, pt.y, -0.38, -0.05, 0.12, 0.09) + gauss2D(pt.x, pt.y, 0.38, -0.05, 0.12, 0.09));

    out[i * 4 + 0] = pt.x;
    out[i * 4 + 1] = pt.y;
    out[i * 4 + 2] = pt.z;
    out[i * 4 + 3] = clamp(0.38 + 0.22 * front + landmarkSal, 0.32, 0.98);
  }
  return out;
}

function insideOrbitalOrOralCutout(x: number, y: number, z: number): boolean {
  if (z < 0.25) return false;
  // Tight almond cutouts strictly inside the palpebral fissure so surrounding mesh flows right up to the eyelids
  const exL = (x + 0.272) / 0.122;
  const eyL = (y - 0.055) / 0.036;
  if (exL * exL + eyL * eyL < 1.0) return true;

  const exR = (x - 0.272) / 0.122;
  const eyR = (y - 0.055) / 0.036;
  if (exR * exR + eyR * eyR < 1.0) return true;

  // Tight inner oral cleft cutout
  const mx = x / 0.155;
  const my = (y + 0.345) / 0.018;
  if (mx * mx + my * my < 1.0) return true;

  return false;
}

function buildHumanConstellationTopology(): HumanConstellationTopology {
  const vertices: MeshVertex[] = [];
  const edges: MeshEdge[] = [];
  const edgeSet = new Set<string>();

  const addVertex = (v: MeshVertex): number => {
    vertices.push(v);
    return vertices.length - 1;
  };

  const addEdge = (a: number, b: number, baseTier: 0 | 1 | 2 | 3 | 4, checkCutout = false) => {
    if (a === b || a < 0 || b < 0 || a >= vertices.length || b >= vertices.length) return;
    const lo = a < b ? a : b;
    const hi = a < b ? b : a;
    const key = `${lo}:${hi}`;
    if (edgeSet.has(key)) return;

    if (checkCutout) {
      const va = vertices[a];
      const vb = vertices[b];
      const mx = 0.5 * (va.x + vb.x);
      const my = 0.5 * (va.y + vb.y);
      const mz = 0.5 * (va.z + vb.z);
      if (
        insideOrbitalOrOralCutout(va.x, va.y, va.z) ||
        insideOrbitalOrOralCutout(vb.x, vb.y, vb.z) ||
        insideOrbitalOrOralCutout(mx, my, mz)
      ) {
        return;
      }
    }

    edgeSet.add(key);
    edges.push({ a, b, baseTier });
  };

  // --------------------------------------------------------------------------
  // 1. LUSH MULTI-SCALE GEODESIC CRANIO-CERVICAL SHELL (28 Staggered Rings)
  // --------------------------------------------------------------------------
  const crownIdx = addVertex({
    x: 0.0,
    y: 1.02,
    z: 0.01,
    nx: 0.0,
    ny: 1.0,
    nz: 0.04,
    group: 'shell',
    starIntensity: 0.96,
    paramT: 0.0,
    jawWeight: 0.0,
    lipWeight: 0.0,
    cheekWeight: 0.0,
  });

  // Pure Floating 3D Human Head Shell (Crown -> Face -> Rounded U-Shaped Jawline & Chin, ZERO Neck!)
  const ringSpecs: Array<{ v: number; cols: number }> = [
    { v: 0.042, cols: 11 },
    { v: 0.098, cols: 18 },
    { v: 0.164, cols: 24 },
    { v: 0.236, cols: 30 },
    { v: 0.308, cols: 34 },
    { v: 0.380, cols: 38 },
    { v: 0.448, cols: 40 },
    { v: 0.510, cols: 44 }, // brow ridge
    { v: 0.566, cols: 46 }, // upper orbital
    { v: 0.618, cols: 46 }, // mid-eye / temple
    { v: 0.668, cols: 46 }, // infraorbital / malar
    { v: 0.718, cols: 46 }, // mid-cheek / nose bridge
    { v: 0.766, cols: 46 }, // nasal tip / alar / cheek
    { v: 0.812, cols: 44 }, // subnasale / philtrum / buccal
    { v: 0.856, cols: 42 }, // lips / gonion
    { v: 0.900, cols: 38 }, // mentolabial / lower jaw
    { v: 0.942, cols: 32 }, // broad rounded chin curve
    { v: 0.976, cols: 22 }, // submental curve
    { v: 0.995, cols: 12 }, // submental cap convergence
  ];

  const shellRings: number[][] = [];
  const shellOnlyIndices: number[] = [crownIdx];

  for (let rIdx = 0; rIdx < ringSpecs.length; rIdx++) {
    const { v, cols } = ringSpecs[rIdx];
    const row: number[] = [];
    const stagger = (rIdx % 2) * 0.5;

    for (let cIdx = 0; cIdx < cols; cIdx++) {
      const baseAngle = ((cIdx + stagger) / cols) * Math.PI * 2.0 - Math.PI;
      const dTheta = 0.036 * Math.sin(rIdx * 2.39996 + cIdx * 1.61803);
      const dV = 0.007 * Math.cos(rIdx * 1.73205 - cIdx * 2.23606) * smoothstep(0.04, 0.20, v) * (1.0 - smoothstep(0.86, 1.0, v));
      const theta = baseAngle + dTheta;
      const vJit = clamp(v + dV, 0.012, 0.998);

      // Gentle anatomical mandibular plane: -0.567 at chin (theta=0) to -0.475 at occipital base (theta=±pi)
      const headBottomY = -0.475 - 0.092 * (0.5 + 0.5 * Math.cos(theta));
      const y = 1.02 * (1.0 - vJit) + headBottomY * vJit;

      const pt = evalCranioCervicalPoint(y, theta);

      const jawW =
        pt.z > -0.06 && pt.y < -0.25
          ? smoothstep(-0.25, -0.49, pt.y) * smoothstep(-0.06, 0.28, pt.z)
          : 0.0;
      const cheekW =
        pt.z > 0.16 && pt.y > -0.28 && pt.y < 0.08
          ? gauss(Math.abs(pt.x), 0.32, 0.15) * gauss(pt.y, -0.08, 0.10)
          : 0.0;

      const harmonicStar = 0.5 + 0.5 * Math.sin(rIdx * 2.291 + cIdx * 2.71828);
      const isCrownZone = rIdx <= 2;
      const isJawRim = rIdx >= ringSpecs.length - 3 && rIdx <= ringSpecs.length - 2 && Math.cos(theta) > -0.1;
      let starIntensity = 0.16 + 0.40 * harmonicStar;
      if (isCrownZone && harmonicStar > 0.52) starIntensity = 0.80 + 0.18 * harmonicStar;
      else if (isJawRim && harmonicStar > 0.60) starIntensity = 0.66 + 0.16 * harmonicStar;

      const idx = addVertex({
        x: pt.x,
        y: pt.y,
        z: pt.z,
        nx: pt.nx,
        ny: pt.ny,
        nz: pt.nz,
        group: 'shell',
        starIntensity,
        paramT: cIdx / cols,
        jawWeight: jawW,
        lipWeight: 0.0,
        cheekWeight: cheekW,
      });
      row.push(idx);
      shellOnlyIndices.push(idx);
    }
    shellRings.push(row);
  }

  const basePoleIdx = addVertex({
    x: 0.0,
    y: -0.567,
    z: 0.16,
    nx: 0.0,
    ny: -1.0,
    nz: 0.08,
    group: 'shell',
    starIntensity: 0.42,
    paramT: 0.0,
    jawWeight: 0.88,
    lipWeight: 0.0,
    cheekWeight: 0.0,
  });
  shellOnlyIndices.push(basePoleIdx);

  for (let i = 0; i < shellRings[0].length; i++) {
    addEdge(crownIdx, shellRings[0][i], 1);
    addEdge(shellRings[0][i], shellRings[0][(i + 1) % shellRings[0].length], 1);
  }

  const lastShellRow = shellRings[shellRings.length - 1];
  for (let i = 0; i < lastShellRow.length; i++) {
    addEdge(basePoleIdx, lastShellRow[i], 0);
  }

  for (let r = 0; r < shellRings.length - 1; r++) {
    const upper = shellRings[r];
    const lower = shellRings[r + 1];
    const nU = upper.length;
    const nL = lower.length;

    for (let i = 0; i < nL; i++) {
      addEdge(lower[i], lower[(i + 1) % nL], 1, true);
    }

    for (let i = 0; i < nL; i++) {
      const uFloat = (i / nL) * nU;
      const u0 = Math.floor(uFloat) % nU;
      const u1 = (u0 + 1) % nU;
      addEdge(lower[i], upper[u0], 1, true);
      addEdge(lower[i], upper[u1], 1, true);
    }

    if (r >= 1 && r <= shellRings.length - 3) {
      for (let i = 0; i < nU; i += 3) {
        const nextRing = shellRings[Math.min(shellRings.length - 1, r + 2)];
        const target = nextRing[Math.floor(((i + 1.6) / nU) * nextRing.length) % nextRing.length];
        addEdge(upper[i], target, 0, true);
      }
    }
  }

  // Helper to stitch anatomical boundary vertices seamlessly into the surrounding shell
  const stitchToShell = (vIdx: number, count = 2, tier: 0 | 1 | 2 = 1) => {
    const v = vertices[vIdx];
    let b1 = -1;
    let b2 = -1;
    let d1 = 1e9;
    let d2 = 1e9;
    for (let s = 0; s < shellOnlyIndices.length; s++) {
      const sIdx = shellOnlyIndices[s];
      const sv = vertices[sIdx];
      if (insideOrbitalOrOralCutout(sv.x, sv.y, sv.z)) continue;
      const d = (sv.x - v.x) * (sv.x - v.x) + (sv.y - v.y) * (sv.y - v.y) + (sv.z - v.z) * (sv.z - v.z);
      if (d < 0.00015) continue;
      if (d < d1) {
        d2 = d1;
        b2 = b1;
        d1 = d;
        b1 = sIdx;
      } else if (d < d2) {
        d2 = d;
        b2 = sIdx;
      }
    }
    if (b1 >= 0 && d1 < 0.025) addEdge(vIdx, b1, tier, true);
    if (count >= 2 && b2 >= 0 && d2 < 0.032) addEdge(vIdx, b2, 0, true);
  };

  // --------------------------------------------------------------------------
  // 2. REFINED SERENE ALMOND EYELIDS & SUPRATARSAL FOLDS (L & R)
  // --------------------------------------------------------------------------
  const leftPalpebralLoop: number[] = [];
  const rightPalpebralLoop: number[] = [];

  for (const side of ['L', 'R'] as const) {
    const sgn = side === 'L' ? -1 : 1;
    const cy = 0.055;

    const innerUpper: number[] = [];
    const innerLower: number[] = [];
    const outerUpper: number[] = [];
    const outerLower: number[] = [];
    const creaseRow: number[] = [];
    const infraRow: number[] = [];

    const N = 9;
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1); // 0 = medial tear duct, 1 = lateral corner
      const spanX = sgn * (0.150 + t * 0.244); // |x| from 0.150 to 0.394 (refined human almond proportion!)
      const cornerLineY = cy - 0.008 + t * 0.015;
      const archSin = Math.sin(t * Math.PI);
      const upperArch = 0.038 * Math.pow(archSin, 0.85) * (1.0 + 0.06 * Math.sin((t - 0.32) * Math.PI));
      const lowerArch = 0.026 * Math.pow(archSin, 0.92);
      const baseZ = 0.585 - 0.092 * Math.pow(t, 1.52) + 0.020 * archSin;

      const uIdx = addVertex({
        x: spanX,
        y: cornerLineY + upperArch,
        z: baseZ + 0.010 * archSin,
        nx: sgn * 0.26 * t,
        ny: 0.22,
        nz: 0.94,
        group: side === 'L' ? 'lidUpperL' : 'lidUpperR',
        starIntensity: i === 0 || i === N - 1 ? 0.72 : 0.34,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      innerUpper.push(uIdx);

      const lIdx =
        i === 0 || i === N - 1
          ? uIdx
          : addVertex({
              x: spanX,
              y: cornerLineY - lowerArch,
              z: baseZ + 0.005 * archSin,
              nx: sgn * 0.26 * t,
              ny: -0.24,
              nz: 0.94,
              group: side === 'L' ? 'lidLowerL' : 'lidLowerR',
              starIntensity: 0.32,
              paramT: t,
              jawWeight: 0.0,
              lipWeight: 0.0,
              cheekWeight: 0.52 * archSin,
            });
      innerLower.push(lIdx);

      const ouIdx = addVertex({
        x: sgn * (0.140 + t * 0.266),
        y: cornerLineY + upperArch * 1.36 + 0.009,
        z: baseZ - 0.006,
        nx: sgn * 0.28 * t,
        ny: 0.28,
        nz: 0.91,
        group: side === 'L' ? 'lidUpperL' : 'lidUpperR',
        starIntensity: 0.28,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      outerUpper.push(ouIdx);

      const olIdx = addVertex({
        x: sgn * (0.144 + t * 0.260),
        y: cornerLineY - lowerArch * 1.42 - 0.009,
        z: baseZ - 0.008,
        nx: sgn * 0.28 * t,
        ny: -0.28,
        nz: 0.91,
        group: side === 'L' ? 'lidLowerL' : 'lidLowerR',
        starIntensity: 0.28,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.6 * archSin,
      });
      outerLower.push(olIdx);

      const crIdx = addVertex({
        x: sgn * (0.136 + t * 0.280),
        y: cornerLineY + upperArch * 1.72 + 0.024,
        z: baseZ - 0.018,
        nx: sgn * 0.30 * t,
        ny: 0.16,
        nz: 0.92,
        group: side === 'L' ? 'orbitalL' : 'orbitalR',
        starIntensity: 0.24,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      creaseRow.push(crIdx);
      stitchToShell(crIdx, 2, 1);

      const infIdx = addVertex({
        x: sgn * (0.146 + t * 0.270),
        y: cornerLineY - lowerArch * 1.85 - 0.024,
        z: baseZ - 0.005,
        nx: sgn * 0.30 * t,
        ny: -0.2,
        nz: 0.92,
        group: side === 'L' ? 'orbitalL' : 'orbitalR',
        starIntensity: 0.26,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.42 * archSin,
      });
      infraRow.push(infIdx);
      stitchToShell(infIdx, 2, 1);
    }

    const targetLoop = side === 'L' ? leftPalpebralLoop : rightPalpebralLoop;
    for (let i = 0; i < N; i++) targetLoop.push(innerUpper[i]);
    for (let i = N - 2; i >= 1; i--) targetLoop.push(innerLower[i]);

    for (let i = 0; i < N - 1; i++) {
      addEdge(innerUpper[i], innerUpper[i + 1], 2);
      addEdge(innerLower[i], innerLower[i + 1], 2);
      addEdge(outerUpper[i], outerUpper[i + 1], 1);
      addEdge(outerLower[i], outerLower[i + 1], 1);
      addEdge(creaseRow[i], creaseRow[i + 1], 1);
      addEdge(infraRow[i], infraRow[i + 1], 0);

      addEdge(innerUpper[i], outerUpper[i], 1);
      addEdge(innerUpper[i + 1], outerUpper[i], 1);
      addEdge(outerUpper[i], creaseRow[i], 0);
      addEdge(outerUpper[i], creaseRow[i + 1], 0);

      addEdge(innerLower[i], outerLower[i], 1);
      addEdge(innerLower[i + 1], outerLower[i], 0);
      addEdge(outerLower[i], infraRow[i], 0);
      addEdge(outerLower[i + 1], infraRow[i], 0);
    }
    addEdge(innerUpper[N - 1], outerUpper[N - 1], 1);
    addEdge(outerUpper[N - 1], creaseRow[N - 1], 0);
  }

  // --------------------------------------------------------------------------
  // 3. NATURAL ARCHED EYEBROW CONSTELLATIONS (L & R)
  // --------------------------------------------------------------------------
  for (const side of ['L', 'R'] as const) {
    const sgn = side === 'L' ? -1 : 1;
    const browRows: number[][] = [];
    const NB = 9;

    for (let r = 0; r < 3; r++) {
      const row: number[] = [];
      const rowOffset = (r - 1) * 0.015;
      for (let i = 0; i < NB; i++) {
        const t = i / (NB - 1);
        const taper = 1.0 - 0.65 * smoothstep(0.55, 1.0, t);
        const x = sgn * (0.115 + t * 0.345 + (r === 1 ? sgn * 0.005 : 0.0));
        const archCurve = 0.036 * Math.sin(Math.pow(t, 0.86) * Math.PI) - 0.018 * t;
        const y = 0.186 + archCurve + rowOffset * taper;
        const z = 0.668 - 0.128 * Math.pow(t, 1.62) + (r === 1 ? 0.010 : 0.0);

        const idx = addVertex({
          x,
          y,
          z,
          nx: sgn * 0.32 * t,
          ny: 0.26,
          nz: 0.90,
          group: side === 'L' ? 'browL' : 'browR',
          starIntensity: r === 1 && (i === 0 || i === 5) ? 0.58 : 0.25,
          paramT: t,
          jawWeight: 0.0,
          lipWeight: 0.0,
          cheekWeight: 0.0,
        });
        row.push(idx);
        if (r === 0 || r === 2) stitchToShell(idx, 1, 0);
      }
      browRows.push(row);
    }

    for (let i = 0; i < NB - 1; i++) {
      addEdge(browRows[0][i], browRows[0][i + 1], 1);
      addEdge(browRows[1][i], browRows[1][i + 1], 2);
      addEdge(browRows[2][i], browRows[2][i + 1], 1);

      addEdge(browRows[0][i], browRows[1][i], 1);
      addEdge(browRows[0][i + 1], browRows[1][i], 1);
      addEdge(browRows[1][i], browRows[2][i], 1);
      addEdge(browRows[1][i], browRows[2][i + 1], 1);
    }
  }

  // --------------------------------------------------------------------------
  // 4. ORGANIC STAGGERED 3D NOSE BRIDGE, SOFT BALL & COMPACT C-CURVED NOSTRILS
  // --------------------------------------------------------------------------
  // Staggered organic nasal bridge facets (zero rectangular ladder!)
  const noseRows: number[][] = [];
  const noseRowDefs: Array<{ y: number; xs: number[]; zPeak: number }> = [
    { y: 0.105, xs: [-0.042, 0.0, 0.042], zPeak: 0.682 },
    { y: 0.048, xs: [-0.048, -0.016, 0.016, 0.048], zPeak: 0.728 },
    { y: -0.008, xs: [-0.046, 0.0, 0.046], zPeak: 0.772 },
    { y: -0.064, xs: [-0.052, -0.018, 0.018, 0.052], zPeak: 0.816 },
    { y: -0.114, xs: [-0.056, -0.022, 0.0, 0.022, 0.056], zPeak: 0.854 },
    { y: -0.148, xs: [-0.052, -0.026, -0.008, 0.008, 0.026, 0.052], zPeak: 0.876 },
    { y: -0.174, xs: [-0.044, -0.018, 0.0, 0.018, 0.044], zPeak: 0.858 },
    { y: -0.198, xs: [-0.032, 0.0, 0.032], zPeak: 0.785 }, // columella / subnasale
  ];

  for (let r = 0; r < noseRowDefs.length; r++) {
    const def = noseRowDefs[r];
    const row: number[] = [];
    for (let c = 0; c < def.xs.length; c++) {
      const nxPos = def.xs[c];
      const uNorm = nxPos / 0.056;
      const nzPos = def.zPeak - 0.038 * uNorm * uNorm;
      const isBall = r >= 4 && r <= 6 && Math.abs(nxPos) < 0.035;
      const idx = addVertex({
        x: nxPos,
        y: def.y - 0.004 * uNorm * uNorm,
        z: nzPos,
        nx: uNorm * 0.55,
        ny: r >= 6 ? -0.45 : 0.15,
        nz: 0.80,
        group: 'noseTip',
        starIntensity: isBall ? 0.56 : 0.28,
        paramT: r / (noseRowDefs.length - 1),
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      row.push(idx);
      if (c === 0 || c === def.xs.length - 1) stitchToShell(idx, 2, 1);
    }
    noseRows.push(row);
  }

  for (let r = 0; r < noseRows.length - 1; r++) {
    const up = noseRows[r];
    const dn = noseRows[r + 1];
    const tier: 0 | 1 | 2 = r >= 5 ? 1 : 0;
    if (r >= 5) {
      for (let i = 0; i < up.length - 1; i++) addEdge(up[i], up[i + 1], tier);
    }
    for (let i = 0; i < dn.length; i++) {
      const uFloat = (i / Math.max(1, dn.length - 1)) * (up.length - 1);
      const u0 = clamp(Math.floor(uFloat), 0, up.length - 1);
      const u1 = clamp(u0 + 1, 0, up.length - 1);
      addEdge(dn[i], up[u0], tier);
      if (u1 !== u0) addEdge(dn[i], up[u1], tier);
    }
  }

  // Compact, sculpted C-curved nostril wings (`alae nasi`) matching `media_1794691084373_d29d2db2.jpg`
  for (const side of ['L', 'R'] as const) {
    const sgn = side === 'L' ? -1 : 1;
    const outerAla: number[] = [];
    const innerAla: number[] = [];
    const NA = 8;

    for (let i = 0; i < NA; i++) {
      const t = i / (NA - 1);
      const ang = -0.32 * Math.PI + t * 1.34 * Math.PI;
      const cx = sgn * 0.072;
      const cy = -0.172;

      const ox = cx + sgn * Math.cos(ang) * 0.032;
      const oy = cy - Math.sin(ang) * 0.025;
      const oz = 0.775 + 0.036 * (1.0 - Math.sin(t * Math.PI)) - 0.022 * Math.max(0.0, Math.cos(ang));

      const oIdx = addVertex({
        x: ox,
        y: oy,
        z: oz,
        nx: sgn * Math.cos(ang) * 0.72,
        ny: -Math.sin(ang) * 0.52,
        nz: 0.58,
        group: 'nostril',
        starIntensity: i === 3 || i === 4 ? 0.64 : 0.34,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.06,
        cheekWeight: 0.16,
      });
      outerAla.push(oIdx);
      if (i >= 2 && i <= 5) stitchToShell(oIdx, 1, 1);

      const ix = cx + sgn * Math.cos(ang) * 0.019;
      const iy = cy - Math.sin(ang) * 0.015 - 0.003;
      const iz = oz + 0.010;
      const iIdx = addVertex({
        x: ix,
        y: iy,
        z: iz,
        nx: sgn * Math.cos(ang) * 0.58,
        ny: -0.45,
        nz: 0.68,
        group: 'nostril',
        starIntensity: 0.30,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.06,
        cheekWeight: 0.14,
      });
      innerAla.push(iIdx);
    }

    for (let i = 0; i < NA - 1; i++) {
      addEdge(outerAla[i], outerAla[i + 1], 1);
      addEdge(innerAla[i], innerAla[i + 1], 0);
      addEdge(outerAla[i], innerAla[i], 0);
      addEdge(outerAla[i + 1], innerAla[i], 0);
    }
    const sideBridgeIdx = side === 'L' ? noseRows[5][0] : noseRows[5][noseRows[5].length - 1];
    const sideColumellaIdx = side === 'L' ? noseRows[7][0] : noseRows[7][noseRows[7].length - 1];
    addEdge(outerAla[0], sideBridgeIdx, 0);
    addEdge(innerAla[NA - 1], sideColumellaIdx, 0);
  }

  // --------------------------------------------------------------------------
  // 5. CALM, CLASSICAL HUMAN LIPS, CUPID'S BOW & PHILTRUM CONSTELLATION
  // --------------------------------------------------------------------------
  const NL = 13;
  const upperVermilion: number[] = [];
  const upperMidRoll: number[] = [];
  const upperInnerOral: number[] = [];
  const lowerInnerOral: number[] = [];
  const lowerMidRoll: number[] = [];
  const lowerVermilion: number[] = [];

  for (let i = 0; i < NL; i++) {
    const u = (i / (NL - 1)) * 2.0 - 1.0;
    const absU = Math.abs(u);
    const x = u * 0.174; // Classical relaxed mouth width aligned with medial irises!
    const archZ = 0.726 - 0.095 * u * u;
    const cornerNeutral = 0.002 * Math.pow(absU, 2.0); // Serene horizontal repose

    const cupidsPeaks =
      0.030 * (1.0 - Math.pow(absU, 1.72)) +
      0.009 * gauss(absU, 0.22, 0.13) -
      0.008 * gauss(u, 0.0, 0.075);
    const commissureY = -0.345 + cornerNeutral;

    const uvIdx = addVertex({
      x,
      y: commissureY + cupidsPeaks,
      z: archZ + 0.008 * (1.0 - absU),
      nx: u * 0.38,
      ny: 0.44,
      nz: 0.80,
      group: 'lipUpper',
      starIntensity: i === 0 || i === NL - 1 || i === 5 || i === 7 ? 0.56 : 0.28,
      paramT: (u + 1.0) * 0.5,
      jawWeight: 0.08,
      lipWeight: 1.0,
      cheekWeight: 0.55 * absU,
    });
    upperVermilion.push(uvIdx);
    if (i > 0 && i < NL - 1 && i % 2 === 0) stitchToShell(uvIdx, 1, 0);

    const umIdx = addVertex({
      x: x * 0.98,
      y: commissureY + cupidsPeaks * 0.50,
      z: archZ + 0.016 * (1.0 - absU * absU),
      nx: u * 0.34,
      ny: 0.18,
      nz: 0.91,
      group: 'lipUpper',
      starIntensity: 0.24,
      paramT: (u + 1.0) * 0.5,
      jawWeight: 0.1,
      lipWeight: 1.0,
      cheekWeight: 0.52 * absU,
    });
    upperMidRoll.push(umIdx);

    const uiIdx = addVertex({
      x: x * 0.96,
      y: commissureY + 0.002 * (1.0 - absU),
      z: archZ + 0.006 * (1.0 - absU),
      nx: u * 0.32,
      ny: -0.14,
      nz: 0.93,
      group: 'lipUpper',
      starIntensity: i === 0 || i === NL - 1 ? 0.62 : 0.22,
      paramT: (u + 1.0) * 0.5,
      jawWeight: 0.12,
      lipWeight: 1.0,
      cheekWeight: 0.6 * absU,
    });
    upperInnerOral.push(uiIdx);

    const liIdx =
      i === 0 || i === NL - 1
        ? uiIdx
        : addVertex({
            x: x * 0.96,
            y: commissureY - 0.002 * (1.0 - absU),
            z: archZ + 0.005 * (1.0 - absU),
            nx: u * 0.32,
            ny: 0.14,
            nz: 0.93,
            group: 'lipLower',
            starIntensity: 0.22,
            paramT: (u + 1.0) * 0.5,
            jawWeight: 0.88,
            lipWeight: 1.0,
            cheekWeight: 0.55 * absU,
          });
    lowerInnerOral.push(liIdx);

    const lowerCurve = 0.038 * Math.pow(Math.max(0.0, 1.0 - u * u), 0.80);
    const lmIdx = addVertex({
      x: x * 0.97,
      y: commissureY - lowerCurve * 0.55,
      z: archZ + 0.018 * (1.0 - u * u),
      nx: u * 0.34,
      ny: -0.16,
      nz: 0.92,
      group: 'lipLower',
      starIntensity: 0.26,
      paramT: (u + 1.0) * 0.5,
      jawWeight: 0.92,
      lipWeight: 1.0,
      cheekWeight: 0.48 * absU,
    });
    lowerMidRoll.push(lmIdx);

    const lvIdx = addVertex({
      x,
      y: commissureY - lowerCurve,
      z: archZ + 0.004 * (1.0 - u * u),
      nx: u * 0.38,
      ny: -0.46,
      nz: 0.78,
      group: 'lipLower',
      starIntensity: i === Math.floor(NL / 2) ? 0.58 : 0.28,
      paramT: (u + 1.0) * 0.5,
      jawWeight: 0.95,
      lipWeight: 1.0,
      cheekWeight: 0.45 * absU,
    });
    lowerVermilion.push(lvIdx);
    if (i > 0 && i < NL - 1 && i % 2 === 0) stitchToShell(lvIdx, 1, 0);
  }

  for (let i = 0; i < NL - 1; i++) {
    addEdge(upperVermilion[i], upperVermilion[i + 1], 1);
    addEdge(upperMidRoll[i], upperMidRoll[i + 1], 1);
    addEdge(upperInnerOral[i], upperInnerOral[i + 1], 2);
    addEdge(lowerInnerOral[i], lowerInnerOral[i + 1], 2);
    addEdge(lowerMidRoll[i], lowerMidRoll[i + 1], 1);
    addEdge(lowerVermilion[i], lowerVermilion[i + 1], 1);

    addEdge(upperVermilion[i], upperMidRoll[i], 0);
    addEdge(upperVermilion[i + 1], upperMidRoll[i], 0);
    addEdge(upperMidRoll[i], upperInnerOral[i], 0);

    addEdge(lowerInnerOral[i], lowerMidRoll[i], 0);
    addEdge(lowerMidRoll[i], lowerVermilion[i], 1);
    addEdge(lowerMidRoll[i + 1], lowerVermilion[i], 0);
  }

  const philL = addVertex({
    x: -0.032,
    y: -0.252,
    z: 0.742,
    nx: -0.2,
    ny: 0.1,
    nz: 0.97,
    group: 'philtrum',
    starIntensity: 0.34,
    paramT: 0.3,
    jawWeight: 0.0,
    lipWeight: 0.35,
    cheekWeight: 0.0,
  });
  const philR = addVertex({
    x: 0.032,
    y: -0.252,
    z: 0.742,
    nx: 0.2,
    ny: 0.1,
    nz: 0.97,
    group: 'philtrum',
    starIntensity: 0.34,
    paramT: 0.7,
    jawWeight: 0.0,
    lipWeight: 0.35,
    cheekWeight: 0.0,
  });
  addEdge(noseRows[7][1], philL, 1);
  addEdge(noseRows[7][1], philR, 1);
  addEdge(philL, philR, 1);
  addEdge(philL, upperVermilion[5], 2);
  addEdge(philR, upperVermilion[7], 2);
  addEdge(philL, upperVermilion[6], 1);
  addEdge(philR, upperVermilion[6], 1);

  const mouthInnerLoop: number[] = [];
  for (let i = 0; i < NL; i++) mouthInnerLoop.push(upperInnerOral[i]);
  for (let i = NL - 2; i >= 1; i--) mouthInnerLoop.push(lowerInnerOral[i]);

  // --------------------------------------------------------------------------
  // 6. SMOOTH ANATOMICAL C-CURVED 3D HUMAN EARS (L & R)
  // --------------------------------------------------------------------------
  const earProfile: Array<[number, number, number]> = [
    [0.622, 0.068, 0.025], // 0: anterior helix root
    [0.666, 0.132, -0.010], // 1: ascending crus
    [0.712, 0.150, -0.046], // 2: superior helix dome
    [0.736, 0.112, -0.078], // 3: postero-superior helix
    [0.742, 0.048, -0.090], // 4: upper outer rim
    [0.732, -0.022, -0.086], // 5: mid outer rim
    [0.714, -0.090, -0.072], // 6: lower helix curve
    [0.692, -0.148, -0.054], // 7: antitragus / lobe transition
    [0.662, -0.192, -0.034], // 8: rounded earlobe tip
    [0.618, -0.190, -0.015], // 9: inner earlobe curve
    [0.580, -0.168, 0.005], // 10: otobasion inferior jaw attachment
  ];

  for (const side of ['L', 'R'] as const) {
    const sgn = side === 'L' ? -1 : 1;
    const helixOuter: number[] = [];
    const antihelixInner: number[] = [];
    const conchaRoot: number[] = [];

    for (let i = 0; i < earProfile.length; i++) {
      const [latX, ey, ez] = earProfile[i];
      const t = i / (earProfile.length - 1);
      const attachX = 0.612 - 0.048 * smoothstep(0.35, 1.0, t);

      const hIdx = addVertex({
        x: sgn * latX,
        y: ey,
        z: ez,
        nx: sgn * 0.92,
        ny: (0.35 - t) * 0.35,
        nz: -0.22,
        group: side === 'L' ? 'earL' : 'earR',
        starIntensity: i === 2 || i === 8 ? 0.72 : 0.32,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      helixOuter.push(hIdx);

      const aIdx = addVertex({
        x: sgn * (0.36 * attachX + 0.64 * latX),
        y: ey * 0.88 - 0.008,
        z: ez + 0.022,
        nx: sgn * 0.75,
        ny: 0.08,
        nz: 0.52,
        group: side === 'L' ? 'earL' : 'earR',
        starIntensity: 0.24,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      antihelixInner.push(aIdx);

      const cIdx = addVertex({
        x: sgn * attachX,
        y: ey * 0.82 - 0.010,
        z: ez + 0.042,
        nx: sgn * 0.62,
        ny: 0.0,
        nz: 0.48,
        group: side === 'L' ? 'earL' : 'earR',
        starIntensity: 0.20,
        paramT: t,
        jawWeight: 0.0,
        lipWeight: 0.0,
        cheekWeight: 0.0,
      });
      conchaRoot.push(cIdx);
      if (i % 2 === 0) stitchToShell(cIdx, 1, 1);
    }

    for (let i = 0; i < earProfile.length - 1; i++) {
      addEdge(helixOuter[i], helixOuter[i + 1], 2);
      addEdge(antihelixInner[i], antihelixInner[i + 1], 1);
      addEdge(conchaRoot[i], conchaRoot[i + 1], 0);

      if (i % 2 === 0) {
        addEdge(helixOuter[i], antihelixInner[i], 1);
        addEdge(antihelixInner[i], conchaRoot[i], 0);
      }
      addEdge(helixOuter[i + 1], antihelixInner[i], 0);
    }
  }

  // --------------------------------------------------------------------------
  // 7. OUTER FLOATING SATELLITE CONSTELLATION HALO (Head & Jawline Only — Zero Neck)
  // --------------------------------------------------------------------------
  const NHALO = 64;
  const haloIndices: number[] = [];
  for (let k = 0; k < NHALO; k++) {
    const u = (k + 0.5) / NHALO;
    const goldenAz = k * 2.39996323;
    const sideSign = k % 2 === 0 ? 1 : -1;
    const theta =
      u < 0.28 || u > 0.86
        ? ((goldenAz % (Math.PI * 2.0)) - Math.PI)
        : sideSign * (1.02 + 0.46 * Math.sin(goldenAz));

    const headBottomY = -0.475 - 0.092 * (0.5 + 0.5 * Math.cos(theta));
    const y = 1.04 * (1.0 - Math.pow(u, 0.92)) + headBottomY * Math.pow(u, 0.92);

    // Keep lateral ear silhouettes clean & unobstructed by spiky halo triangles
    if (y > -0.24 && y < 0.18 && Math.abs(Math.abs(theta) - 1.38) < 0.48) {
      continue;
    }

    const surf = evalCranioCervicalPoint(clamp(y, headBottomY + 0.03, 1.01), theta);
    const lift = 0.040 + 0.062 * (0.5 + 0.5 * Math.sin(k * 1.913 + 0.7));
    const hx = surf.x + surf.nx * lift;
    const hy = surf.y + surf.ny * lift * 0.85;
    const hz = surf.z + surf.nz * lift;

    const hIdx = addVertex({
      x: hx,
      y: hy,
      z: hz,
      nx: surf.nx,
      ny: surf.ny,
      nz: surf.nz,
      group: 'halo',
      starIntensity: k % 3 === 0 ? 0.92 : 0.62,
      paramT: u,
      jawWeight: 0.0,
      lipWeight: 0.0,
      cheekWeight: 0.0,
    });
    haloIndices.push(hIdx);

    let best1 = -1;
    let best2 = -1;
    let d1 = 1e9;
    let d2 = 1e9;
    for (let s = 0; s < shellOnlyIndices.length; s++) {
      const svIdx = shellOnlyIndices[s];
      const sv = vertices[svIdx];
      const dist2 = (sv.x - hx) * (sv.x - hx) + (sv.y - hy) * (sv.y - hy) + (sv.z - hz) * (sv.z - hz);
      if (dist2 < d1) {
        d2 = d1;
        best2 = best1;
        d1 = dist2;
        best1 = svIdx;
      } else if (dist2 < d2) {
        d2 = dist2;
        best2 = svIdx;
      }
    }
    if (best1 >= 0 && d1 < 0.055) addEdge(hIdx, best1, 0);
    if (best2 >= 0 && d2 < 0.048 && k % 2 === 0) addEdge(hIdx, best2, 0);
  }

  for (let i = 0; i < haloIndices.length; i++) {
    const aIdx = haloIndices[i];
    const va = vertices[aIdx];
    for (let j = i + 1; j < haloIndices.length; j++) {
      const bIdx = haloIndices[j];
      const vb = vertices[bIdx];
      const d2 = (va.x - vb.x) * (va.x - vb.x) + (va.y - vb.y) * (va.y - vb.y) + (va.z - vb.z) * (va.z - vb.z);
      if (d2 < 0.032) {
        addEdge(aIdx, bIdx, 0);
      }
    }
  }

  return {
    vertices,
    edges,
    leftPalpebralLoop,
    rightPalpebralLoop,
    mouthInnerLoop,
  };
}

let CACHED_TOPOLOGY: HumanConstellationTopology | null = null;
function getHumanTopology(): HumanConstellationTopology {
  if (!CACHED_TOPOLOGY) {
    CACHED_TOPOLOGY = buildHumanConstellationTopology();
  }
  return CACHED_TOPOLOGY;
}

// ============================================================================
// MINIMIZED CORNER ORB — HOLOGRAPHIC SAPPHIRE-CYAN MINI HEAD (ZERO Math.random)
// ============================================================================

export const JarvisMiniFaceOrb: React.FC<{
  face: JarvisFaceState;
  size?: number;
}> = ({ face, size = 48 }) => {
  const gx = face.gazeX * 1.8;
  const gy = face.gazeY * 1.4;
  const pupilR = 1.2 + face.pupilDilation * 1.0;
  const jawY = face.viseme.jawOpen * 2.0;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      className="shrink-0 overflow-visible"
      style={{
        filter: `drop-shadow(0 6px 14px rgba(0,0,0,0.88)) drop-shadow(0 0 12px ${face.primaryHex}66)`,
      }}
      aria-hidden="true"
    >
      <path
        d="M 24 5 C 33.5 5 38.5 12 38.5 20.5 C 38.5 30.5 32.5 40.5 24 43 C 15.5 40.5 9.5 30.5 9.5 20.5 C 9.5 12 14.5 5 24 5 Z"
        fill="#040814"
        fillOpacity="0.82"
        stroke={face.primaryHex}
        strokeWidth="0.85"
        strokeOpacity="0.52"
      />
      <polyline
        points="14,12 24,8 34,12 24,18 14,12"
        fill="none"
        stroke={face.secondaryHex}
        strokeOpacity="0.22"
        strokeWidth="0.5"
      />
      <path
        d={`M 14.0 ${17.0 - face.browLeftLift * 2.0} Q 17.8 ${15.8 - face.browLeftLift * 2.0} 21.0 ${
          17.2 + face.browFurrow * 1.3
        }`}
        fill="none"
        stroke={face.primaryHex}
        strokeOpacity="0.45"
        strokeWidth="0.7"
        strokeLinecap="round"
      />
      <path
        d={`M 34.0 ${17.0 - face.browRightLift * 2.0} Q 30.2 ${15.8 - face.browRightLift * 2.0} 27.0 ${
          17.2 + face.browFurrow * 1.3
        }`}
        fill="none"
        stroke={face.primaryHex}
        strokeOpacity="0.45"
        strokeWidth="0.7"
        strokeLinecap="round"
      />
      <ellipse cx={18.0} cy={20.5} rx={3.3} ry={1.75} fill="#061228" stroke={face.primaryHex} strokeOpacity="0.45" strokeWidth="0.65" />
      <ellipse cx={30.0} cy={20.5} rx={3.3} ry={1.75} fill="#061228" stroke={face.primaryHex} strokeOpacity="0.45" strokeWidth="0.65" />
      <circle cx={18.0 + gx} cy={20.5 + gy} r={pupilR} fill={face.primaryHex} fillOpacity="0.78" />
      <circle cx={30.0 + gx} cy={20.5 + gy} r={pupilR} fill={face.primaryHex} fillOpacity="0.78" />
      <path
        d="M 22.6 26.5 Q 24 27.8 25.4 26.5"
        fill="none"
        stroke={face.primaryHex}
        strokeOpacity="0.42"
        strokeWidth="0.65"
        strokeLinecap="round"
      />
      <path
        d={`M 19.8 31.8 Q 24 ${31.2 - face.viseme.upperLipRaise} 28.2 31.8 Q 24 ${32.6 + jawY} 19.8 31.8 Z`}
        fill="#050d1e"
        stroke={face.primaryHex}
        strokeOpacity="0.42"
        strokeWidth="0.6"
      />
      <circle cx="24" cy="4" r="1.15" fill="#ffffff" />
      <circle cx="23.2" cy="46" r="1.1" fill="#ffffff" />
    </svg>
  );
};

// ============================================================================
// FRAMELESS 3D HOLOGRAPHIC HUMAN NEURAL CONSTELLATION FACE RIG
// 2nd-Order Spring-Damper Biomechanics + Interactive Gaze + Syllabic Speech
// ============================================================================

const GOLDEN_PHI = 1.61803398875;

interface Spring1D {
  x: number;
  v: number;
}

function makeSpring(x0: number): Spring1D {
  return { x: x0, v: 0.0 };
}

function stepSpring(s: Spring1D, target: number, omega: number, zeta: number, dt: number): number {
  // Semi-implicit Euler 2nd-order spring-damper sub-stepped for unconditional stability
  const steps = dt > 0.022 ? 2 : 1;
  const h = dt / steps;
  const twoZetaOmega = 2.0 * zeta * omega;
  const omegaSq = omega * omega;
  for (let k = 0; k < steps; k++) {
    const accel = omegaSq * (target - s.x) - twoZetaOmega * s.v;
    s.v += accel * h;
    s.x += s.v * h;
  }
  return s.x;
}

interface BiomechanicalRigState {
  yaw: Spring1D;
  pitch: Spring1D;
  roll: Spring1D;
  swayX: Spring1D;
  swayY: Spring1D;
  swayZ: Spring1D;
  gazeX: Spring1D;
  gazeY: Spring1D;
  pupil: Spring1D;
  irisLock: Spring1D;
  apertureL: Spring1D;
  apertureR: Spring1D;
  browL: Spring1D;
  browR: Spring1D;
  furrow: Spring1D;
  cheek: Spring1D;
  jawOpen: Spring1D;
  lipWidth: Spring1D;
  lipRound: Spring1D;
  upperLip: Spring1D;
  nostrilFlare: Spring1D;
  primaryR: number;
  primaryG: number;
  primaryB: number;
  secondaryR: number;
  secondaryG: number;
  secondaryB: number;
  prevRingPhaseDeg: number;
  saccadeBlinkTimer: number;
  prevTargetGazeX: number;
  prevTargetGazeY: number;
  speechPhaseRad: number;
  pulsePhaseRad: number;
}

interface PointerTrackingState {
  normX: number;
  normY: number;
  proximity: number;
  lastMoveMs: number;
  active: boolean;
}

export const JarvisFaceRig: React.FC<JarvisFaceRigProps> = ({
  face,
  isSpeaking,
  isBooting = false,
  bootPercent = 0,
  morphDimmed = false,
  width = 214,
  height = 218,
  onClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestPropsRef = useRef({ face, isSpeaking, isBooting, bootPercent, width, height });
  latestPropsRef.current = { face, isSpeaking, isBooting, bootPercent, width, height };

  const rigRef = useRef<BiomechanicalRigState | null>(null);
  const pointerRef = useRef<PointerTrackingState>({
    normX: 0.0,
    normY: 0.0,
    proximity: 0.0,
    lastMoveMs: -1e9,
    active: false,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const handlePointerMove = (ev: PointerEvent) => {
      const el = canvasRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width * 0.5;
      const cy = rect.top + rect.height * 0.48;
      const dx = ev.clientX - cx;
      const dy = ev.clientY - cy;
      const dist = Math.hypot(dx, dy);
      const refSpan = Math.max(260, Math.min(window.innerWidth, window.innerHeight) * 0.48);
      pointerRef.current.normX = clamp(dx / refSpan, -1.0, 1.0);
      pointerRef.current.normY = clamp(dy / (refSpan * 0.85), -1.0, 1.0);
      pointerRef.current.proximity = clamp(1.0 - dist / (rect.width * 1.45), 0.0, 1.0);
      pointerRef.current.lastMoveMs = performance.now();
      pointerRef.current.active = true;
    };

    const handlePointerLeave = () => {
      pointerRef.current.active = false;
      pointerRef.current.proximity = 0.0;
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('pointermove', handlePointerMove, { passive: true });
      window.addEventListener('pointerleave', handlePointerLeave, { passive: true });
    }

    const topo = getHumanTopology();
    const numV = topo.vertices.length;

    const projX = new Float32Array(numV);
    const projY = new Float32Array(numV);
    const rotZ = new Float32Array(numV);
    const rotNz = new Float32Array(numV);
    const visWave = new Float32Array(numV);

    let rafId = 0;
    let startTimeMs: number | null = null;
    let lastTimeMs: number | null = null;

    const renderFrame = (nowMs: number) => {
      if (startTimeMs === null) startTimeMs = nowMs;
      const dt = lastTimeMs === null ? 0.016 : clamp((nowMs - lastTimeMs) * 0.001, 0.001, 0.05);
      lastTimeMs = nowMs;
      const elapsedSec = (nowMs - startTimeMs) * 0.001;

      const {
        face: curFace,
        isSpeaking: curSpeaking,
        isBooting: curBooting,
        bootPercent: curBootPct,
        width: cssW,
        height: cssH,
      } = latestPropsRef.current;

      const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2.5) : 2;
      const physW = Math.max(1, Math.round(cssW * dpr));
      const physH = Math.max(1, Math.round(cssH * dpr));
      if (canvas.width !== physW || canvas.height !== physH) {
        canvas.width = physW;
        canvas.height = physH;
      }

      const [pR, pG, pB] = parseHexRgb(curFace.primaryHex);
      const [sR, sG, sB] = parseHexRgb(curFace.secondaryHex);

      if (!rigRef.current) {
        rigRef.current = {
          yaw: makeSpring(curFace.headYaw * 0.34),
          pitch: makeSpring(curFace.headPitch * 0.26),
          roll: makeSpring(curFace.headRoll * 0.20),
          swayX: makeSpring(0.0),
          swayY: makeSpring(0.0),
          swayZ: makeSpring(0.0),
          gazeX: makeSpring(curFace.gazeX),
          gazeY: makeSpring(curFace.gazeY),
          pupil: makeSpring(curFace.pupilDilation),
          irisLock: makeSpring(curFace.irisRingContraction),
          apertureL: makeSpring(curFace.eyelidAperture),
          apertureR: makeSpring(curFace.eyelidAperture),
          browL: makeSpring(curFace.browLeftLift),
          browR: makeSpring(curFace.browRightLift),
          furrow: makeSpring(curFace.browFurrow),
          cheek: makeSpring(curFace.cheekWarmthLift),
          jawOpen: makeSpring(curFace.viseme.jawOpen),
          lipWidth: makeSpring(curFace.viseme.lipWidth),
          lipRound: makeSpring(curFace.viseme.lipRound),
          upperLip: makeSpring(curFace.viseme.upperLipRaise),
          nostrilFlare: makeSpring(0.0),
          primaryR: pR,
          primaryG: pG,
          primaryB: pB,
          secondaryR: sR,
          secondaryG: sG,
          secondaryB: sB,
          prevRingPhaseDeg: curFace.ringPhaseDeg,
          saccadeBlinkTimer: 0.0,
          prevTargetGazeX: curFace.gazeX,
          prevTargetGazeY: curFace.gazeY,
          speechPhaseRad: 0.0,
          pulsePhaseRad: 0.0,
        };
      }

      const rig = rigRef.current;

      // ----------------------------------------------------------------------
      // 1. INTERACTIVE POINTER PRESENCE + QUASI-PERIODIC AUTONOMOUS FIXATIONS
      // ----------------------------------------------------------------------
      const ptr = pointerRef.current;
      const pointerIdleSec = Math.max(0.0, (nowMs - ptr.lastMoveMs) * 0.001);
      const pointerPresence =
        ptr.active && curFace.posture !== 'ATTENTIVE_LISTENING'
          ? Math.exp(-Math.max(0.0, pointerIdleSec - 2.6) * 0.65)
          : 0.0;
      const hoverWarmth = ptr.active ? ptr.proximity * Math.exp(-Math.max(0.0, pointerIdleSec - 4.0) * 0.5) : 0.0;

      // Deterministic human staircase eye fixations (holds steady gaze, then crisp smooth shift — ZERO Math.random!)
      const fixArgX = Math.sin(elapsedSec * 0.67) + 0.62 * Math.sin(elapsedSec * 0.67 * GOLDEN_PHI + 1.3);
      const fixArgY = Math.cos(elapsedSec * 0.53 + 0.7) + 0.55 * Math.sin(elapsedSec * 0.53 * GOLDEN_PHI + 2.4);
      const autoFixGazeX = Math.tanh(fixArgX * 2.4) * (curSpeaking ? 0.16 : 0.28);
      const autoFixGazeY = Math.tanh(fixArgY * 2.1) * (curSpeaking ? 0.10 : 0.16);

      const isListening = curFace.posture === 'ATTENTIVE_LISTENING';
      const isPrefillThinking = curSpeaking && curFace.viseme.code === 'SIL' && Math.abs(curFace.viseme.jawOpen - 0.04) < 0.02;

      let targetGazeX = curFace.gazeX;
      let targetGazeY = curFace.gazeY;

      if (isListening) {
        // Track visitor's typing across the input box underneath JARVIS's chin
        targetGazeX = curFace.gazeX;
        targetGazeY = curFace.gazeY;
      } else if (isPrefillThinking) {
        // Contemplative upward-lateral gaze sweep while Qwen prefills attention sinks & ScarStore router
        targetGazeX = 0.28 * Math.sin(elapsedSec * 1.65) + 0.16;
        targetGazeY = -0.22 - 0.08 * Math.cos(elapsedSec * 1.3);
      } else {
        const autonomousBlend = 1.0 - pointerPresence * 0.85;
        targetGazeX = clamp(
          curFace.gazeX * 0.75 + autoFixGazeX * autonomousBlend + ptr.normX * 0.52 * pointerPresence,
          -0.88,
          0.88
        );
        targetGazeY = clamp(
          curFace.gazeY * 0.75 + autoFixGazeY * autonomousBlend + ptr.normY * 0.42 * pointerPresence,
          -0.72,
          0.72
        );
      }

      // Trigger a natural micro-blink whenever gaze shifts rapidly (`vestibulo-ocular saccadic blink`)
      const gazeJump = Math.hypot(targetGazeX - rig.prevTargetGazeX, targetGazeY - rig.prevTargetGazeY);
      rig.prevTargetGazeX = targetGazeX;
      rig.prevTargetGazeY = targetGazeY;
      if (gazeJump > 0.34 && rig.saccadeBlinkTimer <= 0.0) {
        rig.saccadeBlinkTimer = 0.14;
      }
      if (rig.saccadeBlinkTimer > 0.0) {
        rig.saccadeBlinkTimer = Math.max(0.0, rig.saccadeBlinkTimer - dt);
      }

      // ----------------------------------------------------------------------
      // 2. CONTINUOUS 3D FLOATING HEAD SWAY, VESTIBULO-OCULAR COUPLING & PROSODY
      // ----------------------------------------------------------------------
      // Noticeable, fluid multi-harmonic 3D levitation & cranial breathing (ZERO stiffness!)
      const breathYaw =
        (Math.sin(elapsedSec * 0.74) * 0.072 + Math.sin(elapsedSec * 0.74 * GOLDEN_PHI + 0.9) * 0.038) *
        (curSpeaking ? 0.75 : 1.0);
      const breathPitch =
        (Math.cos(elapsedSec * 0.58) * 0.048 + Math.sin(elapsedSec * 0.58 * GOLDEN_PHI + 2.1) * 0.026) *
        (curSpeaking ? 0.75 : 1.0);
      const breathRoll =
        Math.sin(elapsedSec * 0.46 + 1.1) * 0.042 + Math.cos(elapsedSec * 0.46 * GOLDEN_PHI + 0.4) * 0.022;

      const targetSwayX =
        Math.sin(elapsedSec * 0.62) * 0.024 + ptr.normX * 0.028 * pointerPresence;
      const targetSwayY =
        Math.sin(elapsedSec * 1.08) * 0.026 +
        Math.cos(elapsedSec * 0.67 * GOLDEN_PHI) * 0.012 -
        ptr.normY * 0.018 * pointerPresence;
      const targetSwayZ =
        (isListening ? 0.09 : 0.0) +
        (curSpeaking ? 0.06 : 0.0) +
        hoverWarmth * 0.08 +
        Math.sin(elapsedSec * 0.82) * 0.022;

      // Head naturally follows gaze (`vestibulo-ocular coordination`) + Qwen neural posture + speech cadence
      const targetYawRad =
        curFace.headYaw * 0.36 +
        breathYaw +
        targetGazeX * 0.24 +
        (isPrefillThinking ? 0.10 * Math.sin(elapsedSec * 1.4) : 0.0);

      const targetPitchRad =
        curFace.headPitch * 0.28 +
        breathPitch -
        targetGazeY * 0.20 +
        (isListening ? -0.09 : 0.0) +
        (isPrefillThinking ? 0.08 : 0.0);

      const targetRollRad =
        curFace.headRoll * 0.22 +
        breathRoll +
        targetGazeX * 0.075 +
        hoverWarmth * 0.045;

      // ----------------------------------------------------------------------
      // 3. TOKEN IMPULSES & FLUID SYLLABIC CO-ARTICULATION DURING SPEECH
      // ----------------------------------------------------------------------
      const tokenStepped = Math.abs(curFace.ringPhaseDeg - rig.prevRingPhaseDeg) > 0.5;
      rig.prevRingPhaseDeg = curFace.ringPhaseDeg;

      if (curSpeaking && tokenStepped) {
        // Inject natural physical speech impulses on every new Qwen BPE token!
        const syllabicDrive = 0.35 + 0.65 * curFace.viseme.jawOpen;
        rig.jawOpen.v += 2.6 * syllabicDrive;
        rig.pitch.v -= 0.42 * syllabicDrive; // subtle downward conversational nod on syllable onset
        rig.browL.v += 0.55 * syllabicDrive;
        rig.browR.v += 0.48 * syllabicDrive;
        rig.nostrilFlare.v += 0.9 * syllabicDrive;
      }

      const syllabicHz = curSpeaking ? clamp(4.6 + curFace.synapticPulseHz * 0.22, 4.6, 7.2) : 1.4;
      rig.speechPhaseRad = (rig.speechPhaseRad + dt * syllabicHz * Math.PI * 2.0) % (Math.PI * 200.0);
      rig.pulsePhaseRad =
        (rig.pulsePhaseRad + dt * (1.8 + curFace.synapticPulseHz * 0.42) * Math.PI * 2.0) % (Math.PI * 200.0);

      // Continuous syllabic modulation prevents the mouth from ever freezing open between tokens!
      const syllabicPulse = curSpeaking && !isPrefillThinking ? 0.68 + 0.32 * Math.sin(rig.speechPhaseRad) : 1.0;
      const targetJawOpen = curSpeaking
        ? isPrefillThinking
          ? 0.03 + 0.02 * Math.sin(elapsedSec * 3.2)
          : clamp(curFace.viseme.jawOpen * syllabicPulse, 0.02, 0.98)
        : curFace.viseme.jawOpen * 0.5;

      const targetLipWidth = curSpeaking
        ? clamp(curFace.viseme.lipWidth + 0.06 * Math.cos(rig.speechPhaseRad * 0.5), 0.12, 1.0)
        : 0.50 + hoverWarmth * 0.12;
      const targetLipRound = curSpeaking ? curFace.viseme.lipRound : 0.10;
      const targetUpperLip = curSpeaking ? curFace.viseme.upperLipRaise : 0.04 + hoverWarmth * 0.06;

      // Expressive Brows & Warm Duchenne Smile
      const breathBrow = Math.sin(elapsedSec * 0.88 + 0.4) * 0.045;
      const targetBrowL = clamp(
        curFace.browLeftLift + breathBrow + hoverWarmth * 0.12 + (isPrefillThinking ? 0.16 : 0.0),
        -0.55,
        0.62
      );
      const targetBrowR = clamp(
        curFace.browRightLift + breathBrow * 0.92 + hoverWarmth * 0.12 - (isPrefillThinking ? 0.08 : 0.0),
        -0.55,
        0.62
      );
      const targetFurrow = clamp(curFace.browFurrow + (isPrefillThinking ? 0.18 : 0.0), 0.0, 1.0);
      const targetCheek = clamp(curFace.cheekWarmthLift + hoverWarmth * 0.24, 0.08, 0.96);
      const targetNostril = clamp(
        (curSpeaking ? 0.28 * targetJawOpen : 0.06 * (0.5 + 0.5 * Math.sin(elapsedSec * 0.74))) + targetCheek * 0.18,
        0.0,
        0.85
      );

      // Natural Quasi-Periodic Single + Double Blink + Saccade Blink (ZERO Math.random!)
      const blinkPeriod = 4.35;
      const blinkIndex = Math.floor((elapsedSec + 1.6) / blinkPeriod);
      const blinkLocal = (elapsedSec + 1.6) - blinkIndex * blinkPeriod;
      let blinkDip = 0.0;
      if (blinkLocal < 0.145) {
        blinkDip = Math.sin((blinkLocal / 0.145) * Math.PI) * 0.92;
      } else if (blinkIndex % 3 === 1 && blinkLocal >= 0.22 && blinkLocal < 0.355) {
        // Every 3rd cycle: organic human double-blink!
        blinkDip = Math.sin(((blinkLocal - 0.22) / 0.135) * Math.PI) * 0.88;
      }
      if (rig.saccadeBlinkTimer > 0.0) {
        const sacDip = Math.sin((rig.saccadeBlinkTimer / 0.14) * Math.PI) * 0.62;
        blinkDip = Math.max(blinkDip, sacDip);
      }

      const duchenneSquint = targetCheek * 0.10;
      const targetAperture = clamp((curFace.eyelidAperture - duchenneSquint) * (1.0 - blinkDip), 0.05, 0.99);

      // ----------------------------------------------------------------------
      // 4. STEP ALL 2ND-ORDER BIOMECHANICAL SPRINGS
      // ----------------------------------------------------------------------
      const yawRad = stepSpring(rig.yaw, targetYawRad, 6.8, 0.74, dt);
      const pitchRad = stepSpring(rig.pitch, targetPitchRad, 7.2, 0.75, dt);
      const rollRad = stepSpring(rig.roll, targetRollRad, 6.4, 0.76, dt);
      const swayX = stepSpring(rig.swayX, targetSwayX, 5.2, 0.82, dt);
      const swayY = stepSpring(rig.swayY, targetSwayY, 5.2, 0.82, dt);
      const swayZ = stepSpring(rig.swayZ, targetSwayZ, 5.8, 0.84, dt);

      const smGazeX = stepSpring(rig.gazeX, targetGazeX, 24.0, 0.80, dt);
      const smGazeY = stepSpring(rig.gazeY, targetGazeY, 24.0, 0.80, dt);
      const smPupil = stepSpring(rig.pupil, curFace.pupilDilation + hoverWarmth * 0.06, 10.5, 0.85, dt);
      const smIrisLock = stepSpring(rig.irisLock, curFace.irisRingContraction, 11.0, 0.85, dt);
      // Slight 12ms physiological lead/lag between left and right eyelids adds uncanny human realism!
      const liveApertureL = clamp(stepSpring(rig.apertureL, targetAperture, 34.0, 0.84, dt), 0.05, 0.99);
      const liveApertureR = clamp(stepSpring(rig.apertureR, targetAperture, 31.5, 0.84, dt), 0.05, 0.99);

      const smBrowL = stepSpring(rig.browL, targetBrowL, 12.5, 0.76, dt);
      const smBrowR = stepSpring(rig.browR, targetBrowR, 12.0, 0.76, dt);
      const smFurrow = stepSpring(rig.furrow, targetFurrow, 11.0, 0.80, dt);
      const smCheek = stepSpring(rig.cheek, targetCheek, 9.5, 0.78, dt);

      const smJawOpen = clamp(stepSpring(rig.jawOpen, targetJawOpen, 29.0, 0.78, dt), 0.0, 1.05);
      const smLipWidth = clamp(stepSpring(rig.lipWidth, targetLipWidth, 25.0, 0.80, dt), 0.1, 1.05);
      const smLipRound = clamp(stepSpring(rig.lipRound, targetLipRound, 25.0, 0.80, dt), 0.0, 1.0);
      const smUpperLip = clamp(stepSpring(rig.upperLip, targetUpperLip, 26.0, 0.80, dt), 0.0, 1.0);
      const smNostril = clamp(stepSpring(rig.nostrilFlare, targetNostril, 16.0, 0.80, dt), 0.0, 1.0);

      const alphaColor = 1.0 - Math.exp(-dt * 8.0);
      rig.primaryR += (pR - rig.primaryR) * alphaColor;
      rig.primaryG += (pG - rig.primaryG) * alphaColor;
      rig.primaryB += (pB - rig.primaryB) * alphaColor;
      rig.secondaryR += (sR - rig.secondaryR) * alphaColor;
      rig.secondaryG += (sG - rig.secondaryG) * alphaColor;
      rig.secondaryB += (sB - rig.secondaryB) * alphaColor;

      const cy = Math.cos(yawRad);
      const sy = Math.sin(yawRad);
      const cp = Math.cos(pitchRad);
      const sp = Math.sin(pitchRad);
      const cr = Math.cos(rollRad);
      const sr = Math.sin(rollRad);

      // Pure floating 3D head framing (Crown +1.02 to Rounded Chin -0.57, ZERO neck)
      const camZ = 3.62 - swayZ;
      const focal = 2.84;
      const scalePx = Math.min(cssW * 0.56, cssH * 0.56);
      const strokeScale = Math.pow(Math.max(1.0, Math.min(cssW, cssH) / 214.0), 0.44);
      const centerX = cssW * 0.5 + swayX * scalePx;
      const centerY = cssH * 0.555 - swayY * scalePx;

      // Anatomical TMJ hinge rotation angle (`temporomandibular joint` at y = -0.02, z = 0.04)
      const jawAngleRad = smJawOpen * 0.205;
      const cosJaw = Math.cos(jawAngleRad);
      const sinJaw = Math.sin(jawAngleRad);
      const lipSpread = 1.0 + (smLipWidth - 0.42) * 0.28 - smLipRound * 0.24 + smCheek * 0.11;
      const lipPuckerZ = smLipRound * 0.036;
      const lidDeltaL = (liveApertureL - 0.80) * 0.058;
      const lidDeltaR = (liveApertureR - 0.80) * 0.058;

      // ----------------------------------------------------------------------
      // 5. PROJECT ALL 3D VERTICES WITH BIOMECHANICAL BLENDSHAPES & TMJ HINGE
      // ----------------------------------------------------------------------
      for (let i = 0; i < numV; i++) {
        const v = topo.vertices[i];
        let x = v.x;
        let y = v.y;
        let z = v.z;

        if (v.group === 'lidUpperL' || v.group === 'lidUpperR') {
          const lidDelta = v.group === 'lidUpperL' ? lidDeltaL : lidDeltaR;
          const archFactor = Math.sin(v.paramT * Math.PI);
          y += lidDelta * archFactor;
          z += lidDelta * 0.18 * archFactor;
        } else if (v.group === 'lidLowerL' || v.group === 'lidLowerR') {
          const lidDelta = v.group === 'lidLowerL' ? lidDeltaL : lidDeltaR;
          const archFactor = Math.sin(v.paramT * Math.PI);
          y -= lidDelta * 0.36 * archFactor;
          y += smCheek * 0.015 * archFactor;
          z += smCheek * 0.006 * archFactor;
        }

        if (v.group === 'browL' || v.group === 'browR') {
          const lift = v.group === 'browL' ? smBrowL : smBrowR;
          const sgn = v.group === 'browL' ? -1 : 1;
          const medialW = Math.pow(1.0 - v.paramT, 1.15);
          const archW = 0.32 + 0.68 * Math.sin(Math.pow(v.paramT, 0.85) * Math.PI);
          y += lift * 0.058 * archW - smFurrow * 0.028 * medialW;
          x -= sgn * smFurrow * 0.022 * medialW;
          z += lift * 0.014 * archW + smFurrow * 0.008 * medialW;
        }

        if (v.group === 'nostril') {
          const sgn = x < 0 ? -1 : 1;
          const flareFactor = Math.sin(v.paramT * Math.PI);
          x += sgn * smNostril * 0.011 * flareFactor;
          y += smNostril * 0.006 * flareFactor;
        }

        if (v.lipWeight > 0.0) {
          const uSigned = v.paramT * 2.0 - 1.0; // -1..+1 from left to right commissure
          const absU = Math.abs(uSigned);
          const centerBell = Math.max(0.0, 1.0 - absU * 1.45);
          const cornerMask = smoothstep(0.45, 1.0, absU);

          x *= 1.0 + (lipSpread - 1.0) * v.lipWeight;
          z += lipPuckerZ * v.lipWeight * (0.35 + 0.65 * centerBell);

          // Duchenne Zygomatic Smile: corners of the mouth curve gracefully upward, outward & slightly back!
          const smileLift = Math.max(0.0, smCheek - 0.16) * 0.030 * cornerMask * v.lipWeight;
          y += smileLift;
          z -= smileLift * 0.35;

          if (v.group === 'lipUpper') {
            y += smUpperLip * 0.024 * centerBell;
            z += smUpperLip * 0.008 * centerBell;
          } else if (v.group === 'lipLower') {
            // Subtle lower-lip roll outward on open vowels (`OPEN_A`, `ROUND_O_U`)
            z += smJawOpen * 0.014 * centerBell;
          }
        }

        if (v.cheekWeight > 0.0) {
          // Malar apple lift & buccal compression during warm smiles and speech
          const sgn = x < 0 ? -1 : 1;
          y += smCheek * 0.026 * v.cheekWeight;
          z += smCheek * 0.016 * v.cheekWeight;
          x += sgn * smCheek * 0.010 * v.cheekWeight;
        }

        if (v.jawWeight > 0.0 && jawAngleRad > 0.0005) {
          // True Temporomandibular Joint (TMJ) rotational arc around `(y = -0.02, z = 0.04)`
          const wAng = jawAngleRad * v.jawWeight;
          const cw = v.jawWeight > 0.98 ? cosJaw : Math.cos(wAng);
          const sw = v.jawWeight > 0.98 ? sinJaw : Math.sin(wAng);
          const dy = y - (-0.02);
          const dz = z - 0.04;
          y = -0.02 + dy * cw - dz * sw;
          z = 0.04 + dy * sw + dz * cw;
        }

        if (v.group === 'halo') {
          const harmonicSpeak = curSpeaking
            ? 0.014 * curFace.viseme.voiceHarmonics[i % 8] * Math.sin(rig.speechPhaseRad + i * 0.9)
            : 0.0;
          const drift = 0.012 * Math.sin(elapsedSec * 1.65 + i * GOLDEN_PHI) + harmonicSpeak;
          const orbitSpin = 0.016 * Math.cos(elapsedSec * 0.92 + i * 0.75);
          x += v.nx * drift - v.nz * orbitSpin * 0.35;
          y += v.ny * drift;
          z += v.nz * drift + v.nx * orbitSpin * 0.35;
        }

        const x1 = x * cy + z * sy;
        const z1 = -x * sy + z * cy;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const x3 = x1 * cr - y2 * sr;
        const y3 = x1 * sr + y2 * cr;

        const nz1 = -v.nx * sy + v.nz * cy;
        const nz2 = v.ny * sp + nz1 * cp;

        const persp = focal / Math.max(1.2, camZ - z2);
        projX[i] = centerX + x3 * persp * scalePx;
        projY[i] = centerY - y3 * persp * scalePx;
        rotZ[i] = z2;
        rotNz[i] = nz2;
        visWave[i] = 0.5 + 0.5 * Math.sin(rig.pulsePhaseRad - y * 4.2 + Math.abs(x) * 2.8);
      }

      const projectAux3D = (ax: number, ay: number, az: number): [number, number, number] => {
        const x1 = ax * cy + az * sy;
        const z1 = -ax * sy + az * cy;
        const y2 = ay * cp - z1 * sp;
        const z2 = ay * sp + z1 * cp;
        const x3 = x1 * cr - y2 * sr;
        const y3 = x1 * sr + y2 * cr;
        const p = focal / Math.max(1.2, camZ - z2);
        return [centerX + x3 * p * scalePx, centerY - y3 * p * scalePx, p * scalePx];
      };

      // ----------------------------------------------------------------------
      // 6. OPTICAL HOLOGRAPHIC CANVAS RENDERING
      // ----------------------------------------------------------------------
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const pr = Math.round(rig.primaryR);
      const pg = Math.round(rig.primaryG);
      const pb = Math.round(rig.primaryB);
      const srCol = Math.round(rig.secondaryR);
      const sgCol = Math.round(rig.secondaryG);
      const sbCol = Math.round(rig.secondaryB);

      const deepR = Math.round(pr * 0.22);
      const deepG = Math.round(pg * 0.36);
      const deepB = Math.round(Math.min(255, pb * 0.90 + 24));
      const midR = Math.round(pr * 0.56 + srCol * 0.24);
      const midG = Math.round(pg * 0.68 + sgCol * 0.22);
      const midB = Math.round(Math.min(255, pb * 0.96 + 12));
      const iceR = Math.round(Math.min(255, pr * 0.44 + 148));
      const iceG = Math.round(Math.min(255, pg * 0.44 + 150));
      const iceB = Math.round(Math.min(255, pb * 0.34 + 170));

      // PASS 1: Frameless Volumetric Aura + Breathing Core Depth
      const auraBreath = 1.0 + 0.045 * Math.sin(elapsedSec * 1.4) + (curSpeaking ? 0.06 * smJawOpen : 0.0);
      const auraR = Math.min(cssW, cssH) * 0.48 * auraBreath;
      const auraGrad = ctx.createRadialGradient(centerX, centerY - cssH * 0.03, auraR * 0.05, centerX, centerY, auraR);
      auraGrad.addColorStop(0.0, `rgba(${deepR}, ${deepG}, ${deepB}, 0.28)`);
      auraGrad.addColorStop(0.54, `rgba(${Math.round(deepR * 0.6)}, ${Math.round(deepG * 0.7)}, ${deepB}, 0.10)`);
      auraGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = auraGrad;
      ctx.fillRect(0, 0, cssW, cssH);

      ctx.save();
      ctx.beginPath();
      ctx.ellipse(centerX, centerY - cssH * 0.035, cssW * 0.245, cssH * 0.305, -rollRad, 0, Math.PI * 2);
      const coreGrad = ctx.createRadialGradient(
        centerX,
        centerY - cssH * 0.04,
        cssW * 0.04,
        centerX,
        centerY - cssH * 0.03,
        cssH * 0.315
      );
      coreGrad.addColorStop(0.0, 'rgba(2, 6, 18, 0.56)');
      coreGrad.addColorStop(0.68, 'rgba(3, 8, 22, 0.30)');
      coreGrad.addColorStop(1.0, 'rgba(3, 8, 22, 0)');
      ctx.fillStyle = coreGrad;
      ctx.fill();
      ctx.restore();

      // PASS 2: Dynamic Subsurface Bioluminescent Facial Planes (Reacts to Cheek Warmth & Speech!)
      const jawDropApprox = smJawOpen * 0.105;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const drawAnatomicalGlow = (gx3: number, gy3: number, gz3: number, radiusUnit: number, alpha: number) => {
        const [sx, sy, sc] = projectAux3D(gx3, gy3, gz3);
        const rPx = Math.max(2.0, sc * radiusUnit);
        const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, rPx);
        g.addColorStop(0.0, `rgba(${pr}, ${pg}, ${pb}, ${alpha.toFixed(3)})`);
        g.addColorStop(0.55, `rgba(${midR}, ${midG}, ${midB}, ${(alpha * 0.38).toFixed(3)})`);
        g.addColorStop(1.0, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(sx, sy, rPx, 0, Math.PI * 2);
        ctx.fill();
      };
      drawAnatomicalGlow(0.0, -0.150, 0.875, 0.095, 0.18);
      drawAnatomicalGlow(-0.075, -0.172, 0.785, 0.052, 0.12 + smNostril * 0.04);
      drawAnatomicalGlow(0.075, -0.172, 0.785, 0.052, 0.12 + smNostril * 0.04);
      drawAnatomicalGlow(-0.272, 0.055, 0.560, 0.148, 0.13);
      drawAnatomicalGlow(0.272, 0.055, 0.560, 0.148, 0.13);
      // Warm malar cheek blush luminescence when smiling / welcoming
      const cheekGlowAlpha = 0.05 + smCheek * 0.09;
      drawAnatomicalGlow(-0.34, -0.07 + smCheek * 0.02, 0.58, 0.12, cheekGlowAlpha);
      drawAnatomicalGlow(0.34, -0.07 + smCheek * 0.02, 0.58, 0.12, cheekGlowAlpha);
      drawAnatomicalGlow(0.0, -0.350 - jawDropApprox * 0.5, 0.730, 0.125, 0.10 + (curSpeaking ? 0.06 * smJawOpen : 0.0));
      drawAnatomicalGlow(0.0, -0.485 - jawDropApprox * 0.85, 0.440, 0.095, 0.08);
      ctx.restore();

      // PASS 3: Expressive Holographic Eyes (Responsive Gaze, Dual Corneal Catchlights & Lash Rim)
      const renderHolographicEye = (loopIndices: number[], side: 'L' | 'R') => {
        if (loopIndices.length < 6) return;
        const sgn = side === 'L' ? -1 : 1;
        const eyeBaseX = sgn * 0.272;
        const eyeBaseY = 0.055;
        const eyeBaseZ = 0.552;

        // Slight binocular convergence on close hover / listening!
        const vergence = -sgn * (0.008 + hoverWarmth * 0.010 + (isListening ? 0.008 : 0.0));
        const gx3 = eyeBaseX + smGazeX * 0.064 + vergence;
        const gy3 = eyeBaseY - smGazeY * 0.038;
        const gz3 = eyeBaseZ + 0.022;
        const [irisX, irisY, eyeScale] = projectAux3D(gx3, gy3, gz3);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(projX[loopIndices[0]], projY[loopIndices[0]]);
        for (let k = 1; k < loopIndices.length; k++) {
          const idx = loopIndices[k];
          ctx.lineTo(projX[idx], projY[idx]);
        }
        ctx.closePath();
        ctx.clip();

        // Subtle dark scleral depth inside palpebral fissure so the glowing iris pops cleanly
        ctx.fillStyle = 'rgba(2, 7, 18, 0.46)';
        ctx.fill();

        ctx.globalCompositeOperation = 'lighter';
        const irisR = eyeScale * 0.052;
        const lockR = irisR * (0.86 - smIrisLock * 0.16);
        const pupilR = irisR * (0.28 + smPupil * 0.34);

        const irisGlow = ctx.createRadialGradient(irisX, irisY, pupilR * 0.45, irisX, irisY, irisR * 1.52);
        irisGlow.addColorStop(0.0, `rgba(${iceR}, ${iceG}, ${iceB}, 0.52)`);
        irisGlow.addColorStop(0.48, `rgba(${pr}, ${pg}, ${pb}, 0.38)`);
        irisGlow.addColorStop(0.84, `rgba(${midR}, ${midG}, ${midB}, 0.16)`);
        irisGlow.addColorStop(1.0, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = irisGlow;
        ctx.beginPath();
        ctx.arc(irisX, irisY, irisR * 1.52, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.48)`;
        ctx.lineWidth = 0.72 * strokeScale;
        ctx.beginPath();
        ctx.arc(irisX, irisY, irisR, 0, Math.PI * 2);
        ctx.stroke();

        const fiberPhase = ((curFace.ringPhaseDeg * 0.35 * -sgn + elapsedSec * 16.0 * -sgn) * Math.PI) / 180.0;
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.30)`;
        ctx.lineWidth = 0.46 * strokeScale;
        ctx.beginPath();
        for (let f = 0; f < 16; f++) {
          const ang = fiberPhase + (f / 16) * Math.PI * 2.0;
          const ca = Math.cos(ang);
          const sa = Math.sin(ang);
          ctx.moveTo(irisX + ca * (pupilR * 1.06), irisY + sa * (pupilR * 1.06));
          ctx.lineTo(irisX + ca * (irisR * 0.95), irisY + sa * (irisR * 0.95));
        }
        ctx.stroke();

        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.32)`;
        ctx.lineWidth = 0.50 * strokeScale;
        ctx.beginPath();
        ctx.arc(irisX, irisY, lockR, 0, Math.PI * 2);
        ctx.stroke();

        // Deep velvet pupil core
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(2, 6, 16, 0.78)';
        ctx.beginPath();
        ctx.arc(irisX, irisY, pupilR, 0, Math.PI * 2);
        ctx.fill();

        // Primary & secondary corneal specular catchlights (brings instant soul & life to the eyes!)
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = 'rgba(242, 252, 255, 0.78)';
        ctx.beginPath();
        ctx.arc(irisX - irisR * 0.28, irisY - irisR * 0.26, Math.max(0.78, irisR * 0.15), 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.42)`;
        ctx.beginPath();
        ctx.arc(irisX + irisR * 0.24, irisY + irisR * 0.22, Math.max(0.48, irisR * 0.08), 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();

        // Crisp luminous upper palpebral lash crease so blinks & expressions read unmistakably
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.42)`;
        ctx.lineWidth = 0.78 * strokeScale;
        ctx.lineCap = 'round';
        ctx.beginPath();
        const upperCount = Math.min(9, loopIndices.length);
        ctx.moveTo(projX[loopIndices[0]], projY[loopIndices[0]]);
        for (let k = 1; k < upperCount; k++) {
          ctx.lineTo(projX[loopIndices[k]], projY[loopIndices[k]]);
        }
        ctx.stroke();
        ctx.restore();
      };

      renderHolographicEye(topo.leftPalpebralLoop, 'L');
      renderHolographicEye(topo.rightPalpebralLoop, 'R');

      // PASS 4: Expressive Oral Cavity Depth + Live 8-Band Acoustic Formant Harmonics
      if (topo.mouthInnerLoop.length >= 6) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(projX[topo.mouthInnerLoop[0]], projY[topo.mouthInnerLoop[0]]);
        for (let k = 1; k < topo.mouthInnerLoop.length; k++) {
          const idx = topo.mouthInnerLoop[k];
          ctx.lineTo(projX[idx], projY[idx]);
        }
        ctx.closePath();

        if (smJawOpen > 0.025) {
          ctx.save();
          ctx.clip();
          ctx.fillStyle = 'rgba(2, 6, 17, 0.62)';
          ctx.fill();

          ctx.globalCompositeOperation = 'lighter';
          const [mcX, mcY, mScale] = projectAux3D(0.0, -0.345 - jawDropApprox * 0.48, 0.72);
          const harmonics = curFace.viseme.voiceHarmonics;
          ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, ${curSpeaking ? 0.52 : 0.20})`;
          ctx.lineWidth = 0.68 * strokeScale;
          ctx.beginPath();
          for (let h = 0; h < harmonics.length; h++) {
            const u = (h - (harmonics.length - 1) * 0.5) * 0.030 * lipSpread;
            const hx = mcX + u * mScale;
            const liveOsc = 0.72 + 0.28 * Math.sin(rig.speechPhaseRad * 1.3 + h * 0.85);
            const hh = Math.max(0.8, harmonics[h] * liveOsc * smJawOpen * mScale * 0.088);
            ctx.moveTo(hx, mcY - hh);
            ctx.lineTo(hx, mcY + hh);
          }
          ctx.stroke();
          ctx.restore();
        }

        // Sculpted bioluminescent oral contour so smiles & vowel shapes read clearly
        ctx.globalCompositeOperation = 'lighter';
        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, ${(0.28 + smCheek * 0.14 + (curSpeaking ? 0.14 : 0.0)).toFixed(3)})`;
        ctx.lineWidth = 0.65 * strokeScale;
        ctx.stroke();
        ctx.restore();
      }

      // PASS 5: Gossamer 3D Constellation Threads + Traveling Synaptic Neural Waves
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.strokeStyle = `rgba(${deepR}, ${deepG}, ${deepB}, 0.08)`;
      ctx.lineWidth = 0.32 * strokeScale;
      ctx.beginPath();
      for (let e = 0; e < topo.edges.length; e++) {
        const edge = topo.edges[e];
        const a = edge.a;
        const b = edge.b;
        if (0.5 * (rotNz[a] + rotNz[b]) < -0.06) {
          ctx.moveTo(projX[a], projY[a]);
          ctx.lineTo(projX[b], projY[b]);
        }
      }
      ctx.stroke();

      ctx.globalCompositeOperation = 'lighter';

      const bucketPaths: Array<number[]> = [[], [], [], [], []];
      for (let e = 0; e < topo.edges.length; e++) {
        const edge = topo.edges[e];
        const a = edge.a;
        const b = edge.b;
        const avgNz = 0.5 * (rotNz[a] + rotNz[b]);
        if (avgNz < -0.06) continue;

        let tier: number = edge.baseTier;
        const ga = topo.vertices[a].group;
        const isExpressiveFeature =
          ga === 'browL' ||
          ga === 'browR' ||
          ga === 'lipUpper' ||
          ga === 'lipLower' ||
          ga === 'lidUpperL' ||
          ga === 'lidUpperR';
        if (isExpressiveFeature && tier >= 1) {
          tier = Math.min(4, tier + 1);
        }

        const fresnel = 1.0 - Math.abs(avgNz);
        if (tier <= 1 && fresnel > 0.74 && rotZ[a] > -0.24) {
          tier = Math.min(3, tier + 1);
        }
        const pulse = 0.5 * (visWave[a] + visWave[b]);
        const waveThresh = curSpeaking || isPrefillThinking || isListening ? 0.84 : 0.93;
        if (tier >= 1 && tier < 4 && pulse > waveThresh) {
          tier += 1;
        }
        bucketPaths[tier].push(a, b);
      }

      const bucketStyles: Array<{ stroke: string; width: number }> = [
        { stroke: `rgba(${deepR}, ${Math.min(255, deepG + 26)}, ${deepB}, 0.10)`, width: 0.35 * strokeScale },
        { stroke: `rgba(${midR}, ${midG}, ${midB}, 0.16)`, width: 0.42 * strokeScale },
        { stroke: `rgba(${pr}, ${pg}, ${pb}, 0.25)`, width: 0.50 * strokeScale },
        { stroke: `rgba(${pr}, ${pg}, ${pb}, 0.36)`, width: 0.58 * strokeScale },
        { stroke: `rgba(${iceR}, ${iceG}, ${iceB}, 0.48)`, width: 0.68 * strokeScale },
      ];

      for (let t = 0; t < 5; t++) {
        const pairs = bucketPaths[t];
        if (pairs.length === 0) continue;
        ctx.strokeStyle = bucketStyles[t].stroke;
        ctx.lineWidth = bucketStyles[t].width;
        ctx.beginPath();
        for (let k = 0; k < pairs.length; k += 2) {
          const a = pairs[k];
          const b = pairs[k + 1];
          ctx.moveTo(projX[a], projY[a]);
          ctx.lineTo(projX[b], projY[b]);
        }
        ctx.stroke();
      }

      // Glabella Concentration / Furrow Tension Threads
      if (smFurrow > 0.26) {
        const fAlpha = clamp((smFurrow - 0.26) * 0.48, 0.05, 0.32);
        const [glL1x, glL1y] = projectAux3D(-0.036 + smFurrow * 0.008, 0.205 - smFurrow * 0.012, 0.680);
        const [glL2x, glL2y] = projectAux3D(-0.026 + smFurrow * 0.006, 0.132, 0.676);
        const [glR1x, glR1y] = projectAux3D(0.036 - smFurrow * 0.008, 0.205 - smFurrow * 0.012, 0.680);
        const [glR2x, glR2y] = projectAux3D(0.026 - smFurrow * 0.006, 0.132, 0.676);
        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, ${fAlpha.toFixed(3)})`;
        ctx.lineWidth = 0.52 * strokeScale;
        ctx.beginPath();
        ctx.moveTo(glL1x, glL1y);
        ctx.lineTo(glL2x, glL2y);
        ctx.moveTo(glR1x, glR1y);
        ctx.lineTo(glR2x, glR2y);
        ctx.stroke();
      }

      // PASS 6: Twinkling Bioluminescent Constellation Stardust Pinpoints
      ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, 0.22)`;
      ctx.beginPath();
      for (let i = 0; i < numV; i++) {
        const v = topo.vertices[i];
        if (rotNz[i] < -0.08 && v.group !== 'halo') continue;
        if (v.starIntensity < 0.62) continue;
        const rMid = (0.95 + v.starIntensity * 1.18) * (0.82 + 0.34 * visWave[i]) * strokeScale;
        ctx.moveTo(projX[i] + rMid, projY[i]);
        ctx.arc(projX[i], projY[i], rMid, 0, Math.PI * 2);
      }
      ctx.fill();

      ctx.fillStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.70)`;
      ctx.beginPath();
      for (let i = 0; i < numV; i++) {
        const v = topo.vertices[i];
        if (rotNz[i] < -0.06 && v.group !== 'halo') continue;
        if (v.starIntensity < 0.58) continue;
        const rCore = (0.36 + (v.starIntensity - 0.48) * 0.72) * (0.88 + 0.24 * visWave[i]) * strokeScale;
        ctx.moveTo(projX[i] + rCore, projY[i]);
        ctx.arc(projX[i], projY[i], rCore, 0, Math.PI * 2);
      }
      ctx.fill();

      // PASS 7: Live Qwen 3B GPU Shader Boot Progress Ring
      if (curBooting) {
        const clampedPct = clamp(curBootPct, 0, 100) / 100.0;
        const bootR = Math.min(cssW, cssH) * 0.46;
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.22)`;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, bootR, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.92)`;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, bootR, -Math.PI * 0.5, -Math.PI * 0.5 + Math.PI * 2.0 * clampedPct);
        ctx.stroke();
      }

      ctx.restore();
      rafId = window.requestAnimationFrame(renderFrame);
    };

    rafId = window.requestAnimationFrame(renderFrame);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
      if (typeof window !== 'undefined') {
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerleave', handlePointerLeave);
      }
    };
  }, []);

  return (
    <div
      data-testid="jarvis-floating-face-rig"
      onClick={onClick}
      className={`relative select-none flex items-center justify-center ${
        onClick ? 'cursor-pointer hover:scale-[1.02]' : ''
      }`}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        opacity: morphDimmed ? 0.14 : 1.0,
        transition: 'opacity 580ms cubic-bezier(0.22, 1, 0.36, 1), transform 200ms ease',
        filter: `drop-shadow(0 16px 32px rgba(0,0,0,0.88)) drop-shadow(0 0 24px ${face.primaryHex}33)`,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: `${width}px`, height: `${height}px` }}
        className="block pointer-events-none"
        aria-hidden="true"
      />
    </div>
  );
};
