import React, { useEffect, useRef } from 'react';
import type { JarvisFaceState } from '../lib/jarvisNeuralController';
import { buildCanonicalFibonacciSphereF32 } from '../lib/pointScarFlowEngine';

export interface JarvisFaceRigProps {
  face: JarvisFaceState;
  isSpeaking: boolean;
  isBooting?: boolean;
  bootPercent?: number;
  morphDimmed?: boolean;
  neuralFaceMidF32?: Float32Array | null;
  neuralFaceFinalF32?: Float32Array | null;
  neuralFaceSeq?: number;
  width?: number;
  height?: number;
  onClick?: () => void;
}

const NUM_POINTS = 1024;
const K_NEIGHBORS = 4;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp((x - e0) / Math.max(1e-6, e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
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

interface DiscoveredNeuralLandmarks {
  nbrIdx: Int16Array; // [NUM_POINTS * K_NEIGHBORS]
  nbrDist: Float32Array; // [NUM_POINTS * K_NEIGHBORS]
  eyeLeftX: number;
  eyeLeftY: number;
  eyeRightX: number;
  eyeRightY: number;
  oralCleftY: number;
  oralZ: number;
}

/**
 * Discovers local k-NN constellation connectivity and anatomical feature centroids
 * (`leftEye`, `rightEye`, `oralCleftY`) purely from the GPU `PointScarFlow`-synthesized
 * 3D manifold `[x, y, z, salience]` — ZERO hardcoded facial meshes or lookup tables.
 */
function analyzeNeuralManifold(cloudF32: Float32Array, numPoints = NUM_POINTS): DiscoveredNeuralLandmarks {
  const nbrIdx = new Int16Array(numPoints * K_NEIGHBORS);
  const nbrDist = new Float32Array(numPoints * K_NEIGHBORS);

  let lSumX = 0;
  let lSumY = 0;
  let lW = 0;
  let rSumX = 0;
  let rSumY = 0;
  let rW = 0;
  let mSumY = 0;
  let mSumZ = 0;
  let mW = 0;

  const bestD = new Float32Array(K_NEIGHBORS);
  const bestJ = new Int16Array(K_NEIGHBORS);

  for (let i = 0; i < numPoints; i++) {
    const ix = cloudF32[i * 4 + 0];
    const iy = cloudF32[i * 4 + 1];
    const iz = cloudF32[i * 4 + 2];
    const sal = cloudF32[i * 4 + 3];

    if (sal >= 0.88 && iy > -0.22 && iy < 0.25 && iz > 0.12) {
      const w = (sal - 0.86) * (sal - 0.86);
      if (ix < -0.05) {
        lSumX += ix * w;
        lSumY += iy * w;
        lW += w;
      } else if (ix > 0.05) {
        rSumX += ix * w;
        rSumY += iy * w;
        rW += w;
      }
    }

    if (sal >= 0.80 && iy < -0.26 && iy > -0.68 && Math.abs(ix) < 0.24 && iz > 0.15) {
      const w = sal - 0.78;
      mSumY += iy * w;
      mSumZ += iz * w;
      mW += w;
    }

    for (let k = 0; k < K_NEIGHBORS; k++) {
      bestD[k] = 1e9;
      bestJ[k] = -1;
    }

    for (let j = 0; j < numPoints; j++) {
      if (i === j) continue;
      const dx = ix - cloudF32[j * 4 + 0];
      const dy = iy - cloudF32[j * 4 + 1];
      const dz = iz - cloudF32[j * 4 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= bestD[K_NEIGHBORS - 1]) continue;
      let pos = K_NEIGHBORS - 1;
      while (pos > 0 && d2 < bestD[pos - 1]) {
        bestD[pos] = bestD[pos - 1];
        bestJ[pos] = bestJ[pos - 1];
        pos--;
      }
      bestD[pos] = d2;
      bestJ[pos] = j;
    }

    for (let k = 0; k < K_NEIGHBORS; k++) {
      nbrIdx[i * K_NEIGHBORS + k] = bestJ[k];
      nbrDist[i * K_NEIGHBORS + k] = Math.sqrt(bestD[k]);
    }
  }

  return {
    nbrIdx,
    nbrDist,
    eyeLeftX: lW > 1e-5 ? lSumX / lW : -0.27,
    eyeLeftY: lW > 1e-5 ? lSumY / lW : -0.10,
    eyeRightX: rW > 1e-5 ? rSumX / rW : 0.27,
    eyeRightY: rW > 1e-5 ? rSumY / rW : -0.10,
    oralCleftY: mW > 1e-5 ? mSumY / mW : -0.52,
    oralZ: mW > 1e-5 ? mSumZ / mW : 0.58,
  };
}

/**
 * Compact Minimized Corner Orb rendered 100% from the GPU `PointScarFlow`-constructed
 * 3D manifold (or canonical Fibonacci sphere `X_0 in S^2` prior to GPU completion).
 */
export const JarvisMiniFaceOrb: React.FC<{
  face: JarvisFaceState;
  neuralFaceFinalF32?: Float32Array | null;
  size?: number;
}> = ({ face, neuralFaceFinalF32, size = 46 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cloud =
      neuralFaceFinalF32 && neuralFaceFinalF32.length >= NUM_POINTS * 4
        ? neuralFaceFinalF32
        : buildCanonicalFibonacciSphereF32(NUM_POINTS);

    const [pr, pg, pb] = parseHexRgb(face.primaryHex);
    const cx = size * 0.5;
    const cy = size * 0.5;
    const sc = size * 0.44;
    const camZ = 3.62;
    const focal = 2.84;

    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < NUM_POINTS; i += 2) {
      const x = cloud[i * 4 + 0];
      const y = cloud[i * 4 + 1];
      const z = cloud[i * 4 + 2];
      const sal = cloud[i * 4 + 3];
      if (z < -0.15) continue;
      const persp = focal / Math.max(1.2, camZ - z);
      const sx = cx + x * persp * sc;
      const sy = cy - y * persp * sc;
      const alpha = clamp((0.16 + 0.65 * sal) * smoothstep(-0.15, 0.45, z), 0.05, 0.92);
      const r = sal >= 0.90 ? 1.15 : 0.68;
      ctx.fillStyle = sal >= 0.92 ? `rgba(240, 250, 255, ${alpha.toFixed(2)})` : `rgba(${pr}, ${pg}, ${pb}, ${alpha.toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [face.primaryHex, neuralFaceFinalF32, size]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${size}px`, height: `${size}px` }}
      className="block pointer-events-none"
      aria-hidden="true"
    />
  );
};

/**
 * Pure Neural-Model-Constructed 3D JARVIS Face Rig (`Zero Standardized Hardcoded Mesh`):
 *
 * - Starts on the pure mathematical Canonical Fibonacci Sphere `X_0 in S^2` (`tau = 0`).
 * - Integrates along the GPU `PointScarFlow` (`12` WGSL 1.0 shaders, `VQ4-Scar + INT4-STE`)
 *   Optimal-Transport trajectory `X_0 -> X_0.5 -> X_1.0` conditioned on `Qwen2.5-3B` `Layer-23`
 *   `P_perp_lang` identity latents.
 * - Applies live `Qwen2.5-3B` biomechanical kinematics (gaze, epistemic pupillometry, blink,
 *   multilingual token visemes, acoustic harmonics) directly onto the model's generated 3D points.
 */
export const JarvisFaceRig: React.FC<JarvisFaceRigProps> = ({
  face,
  isSpeaking,
  isBooting = false,
  bootPercent = 0,
  morphDimmed = false,
  neuralFaceMidF32 = null,
  neuralFaceFinalF32 = null,
  neuralFaceSeq = 0,
  width = 214,
  height = 218,
  onClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const propsRef = useRef({
    face,
    isSpeaking,
    isBooting,
    bootPercent,
    neuralFaceMidF32,
    neuralFaceFinalF32,
    neuralFaceSeq,
    width,
    height,
  });
  propsRef.current = {
    face,
    isSpeaking,
    isBooting,
    bootPercent,
    neuralFaceMidF32,
    neuralFaceFinalF32,
    neuralFaceSeq,
    width,
    height,
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const x0Sphere = buildCanonicalFibonacciSphereF32(NUM_POINTS);
    const startCloud = new Float32Array(x0Sphere);
    const midCloud = new Float32Array(x0Sphere);
    const targetCloud = new Float32Array(x0Sphere);
    const liveCloud = new Float32Array(x0Sphere);

    let landmarks = analyzeNeuralManifold(x0Sphere, NUM_POINTS);
    let lastSeq = -1;
    let otStartMs = performance.now();
    let hasNeuralModelFace = false;

    const projX = new Float32Array(NUM_POINTS);
    const projY = new Float32Array(NUM_POINTS);
    const rotZ = new Float32Array(NUM_POINTS);
    const liveSal = new Float32Array(NUM_POINTS);
    const drawOrder = new Int16Array(NUM_POINTS);
    for (let i = 0; i < NUM_POINTS; i++) drawOrder[i] = i;

    let smYaw = 0;
    let smPitch = 0;
    let smRoll = 0;
    let smGazeX = 0;
    let smGazeY = 0;
    let smJaw = 0;
    let smLipRound = 0;
    let smLipSpread = 0;
    let smBlink = 0;
    let smPupil = 0.35;
    let smBrowRaise = 0;
    let smFurrow = 0;

    let ptrClientX = -9999;
    let ptrClientY = -9999;
    let ptrActive = false;

    const onPointerMove = (ev: PointerEvent) => {
      ptrClientX = ev.clientX;
      ptrClientY = ev.clientY;
      ptrActive = true;
    };
    const onPointerLeave = () => {
      ptrActive = false;
    };
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerleave', onPointerLeave);

    let prevMs = performance.now();
    let speechPhase = 0;
    let nextAutoBlinkMs = prevMs + 3400;
    let autoBlinkStartMs = -1;
    let rafId = 0;

    const renderFrame = (nowMs: number) => {
      const p = propsRef.current;
      const dt = clamp((nowMs - prevMs) * 0.001, 0.001, 0.05);
      prevMs = nowMs;

      const cssW = Math.max(64, p.width);
      const cssH = Math.max(64, p.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.round(cssW * dpr);
      const targetH = Math.round(cssH * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      // Ingest newly GPU-constructed PointScarFlow 3D JARVIS face (`X_0 -> X_0.5 -> X_1.0`)
      if (
        p.neuralFaceFinalF32 &&
        p.neuralFaceFinalF32.length >= NUM_POINTS * 4 &&
        p.neuralFaceSeq !== lastSeq
      ) {
        lastSeq = p.neuralFaceSeq;
        startCloud.set(hasNeuralModelFace ? liveCloud : x0Sphere);
        if (p.neuralFaceMidF32 && p.neuralFaceMidF32.length >= NUM_POINTS * 4) {
          midCloud.set(p.neuralFaceMidF32.subarray(0, NUM_POINTS * 4));
        } else {
          for (let k = 0; k < NUM_POINTS * 4; k++) {
            midCloud[k] = 0.5 * (startCloud[k] + p.neuralFaceFinalF32[k]);
          }
        }
        targetCloud.set(p.neuralFaceFinalF32.subarray(0, NUM_POINTS * 4));
        landmarks = analyzeNeuralManifold(targetCloud, NUM_POINTS);
        otStartMs = nowMs;
        hasNeuralModelFace = true;
      }

      // Deterministic physiological blink cadence layered with Qwen epistemic blink signal
      if (nowMs >= nextAutoBlinkMs && autoBlinkStartMs < 0) {
        autoBlinkStartMs = nowMs;
        nextAutoBlinkMs = nowMs + 3600 + 900 * Math.abs(Math.sin(nowMs * 0.001));
      }
      let physBlink = 0;
      if (autoBlinkStartMs > 0) {
        const bProg = (nowMs - autoBlinkStartMs) / 175.0;
        if (bProg >= 1.0) {
          autoBlinkStartMs = -1;
        } else {
          physBlink = Math.sin(bProg * Math.PI);
        }
      }

      // Subtle cursor tracking in local widget frame
      let cursorYaw = 0;
      let cursorPitch = 0;
      if (ptrActive) {
        const rect = canvas.getBoundingClientRect();
        const cxScreen = rect.left + rect.width * 0.5;
        const cyScreen = rect.top + rect.height * 0.5;
        cursorYaw = clamp((ptrClientX - cxScreen) / 520.0, -0.22, 0.22);
        cursorPitch = clamp((ptrClientY - cyScreen) / 520.0, -0.16, 0.16);
      }

      const springFast = 1.0 - Math.exp(-dt * 15.0);
      const springPose = 1.0 - Math.exp(-dt * 8.5);

      const targetYaw = p.face.headYaw * 0.36 + p.face.gazeX * 0.20 + cursorYaw * 0.45;
      const targetPitch = p.face.headPitch * 0.28 - p.face.gazeY * 0.16 - cursorPitch * 0.35;
      const targetRoll = p.face.headRoll * 0.22;

      smYaw += (targetYaw - smYaw) * springPose;
      smPitch += (targetPitch - smPitch) * springPose;
      smRoll += (targetRoll - smRoll) * springPose;
      smGazeX += ((p.face.gazeX + cursorYaw * 0.8) - smGazeX) * springFast;
      smGazeY += ((p.face.gazeY + cursorPitch * 0.8) - smGazeY) * springFast;
      smPupil += (p.face.pupilDilation - smPupil) * springPose;
      const meanBrowLift = 0.5 * (p.face.browLeftLift + p.face.browRightLift);
      const qwenBlink = clamp(1.0 - p.face.eyelidAperture, 0.0, 0.85);
      smBrowRaise += (meanBrowLift - smBrowRaise) * springPose;
      smFurrow += (p.face.browFurrow - smFurrow) * springPose;
      smBlink += (Math.max(qwenBlink, physBlink) - smBlink) * (1.0 - Math.exp(-dt * 24.0));

      if (p.isSpeaking) {
        const vh = p.face.viseme.voiceHarmonics;
        const harmEnergy = (vh[0] + vh[1] + vh[2] + vh[3]) * 0.25;
        speechPhase += dt * (9.5 + harmEnergy * 6.0);
      }
      const syllabicMod = p.isSpeaking ? 0.78 + 0.22 * Math.sin(speechPhase) : 0.0;
      const targetJaw = p.isSpeaking ? p.face.viseme.jawOpen * syllabicMod : 0.0;
      smJaw += (targetJaw - smJaw) * springFast;
      smLipRound += ((p.isSpeaking ? p.face.viseme.lipRound : 0.0) - smLipRound) * springFast;
      smLipSpread += ((p.isSpeaking ? p.face.viseme.lipWidth : 0.5) - smLipSpread) * springFast;

      // Before GPU model face arrives (~80 ms), gently spin the pure Fibonacci sphere X_0
      const preModelSpin = hasNeuralModelFace ? 0.0 : nowMs * 0.0008;
      const yaw = smYaw + preModelSpin;
      const pitch = smPitch;
      const roll = smRoll;

      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);
      const cosR = Math.cos(roll);
      const sinR = Math.sin(roll);

      // Exact camera shared 1:1 with `JarvisFloatingPointStage` (`morphW = 0`)
      const minWH = Math.min(cssW, cssH);
      const centerX = cssW * 0.5;
      const centerY = cssH * 0.5;
      const scalePx = minWH * 0.46;
      const camZ = 3.62;
      const focal = 2.84;
      const strokeScale = minWH / 214.0;

      const [pr, pg, pb] = parseHexRgb(p.face.primaryHex);
      const [sr, sg, sb] = parseHexRgb(p.face.secondaryHex);
      const iceR = Math.min(255, Math.round(pr * 0.35 + 166));
      const iceG = Math.min(255, Math.round(pg * 0.35 + 166));
      const iceB = Math.min(255, Math.round(pb * 0.35 + 166));

      // Soft ambient volumetric aura behind the neural point manifold
      const auraGrad = ctx.createRadialGradient(centerX, centerY, minWH * 0.04, centerX, centerY, minWH * 0.48);
      auraGrad.addColorStop(0.0, `rgba(${pr}, ${pg}, ${pb}, 0.16)`);
      auraGrad.addColorStop(0.55, `rgba(${sr}, ${sg}, ${sb}, 0.06)`);
      auraGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = auraGrad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, minWH * 0.48, 0, Math.PI * 2);
      ctx.fill();

      const otElapsed = Math.max(0.0, (nowMs - otStartMs) * 0.001);
      const breathY = Math.sin(nowMs * 0.0016) * 0.012;

      // Integrate OT trajectory & apply continuous Qwen biomechanical fields on GPU-generated points
      for (let i = 0; i < NUM_POINTS; i++) {
        const i4 = i * 4;
        let x: number;
        let y: number;
        let z: number;
        let sal: number;

        if (!hasNeuralModelFace) {
          x = x0Sphere[i4 + 0] * 0.76;
          y = x0Sphere[i4 + 1] * 0.76;
          z = x0Sphere[i4 + 2] * 0.76;
          sal = x0Sphere[i4 + 3];
        } else {
          const stagger = (i / NUM_POINTS) * 0.14;
          const uOt = clamp((otElapsed - stagger) / 0.74, 0.0, 1.0);
          if (uOt < 0.5) {
            const w = smoothstep(0.0, 0.5, uOt);
            x = startCloud[i4 + 0] + (midCloud[i4 + 0] - startCloud[i4 + 0]) * w;
            y = startCloud[i4 + 1] + (midCloud[i4 + 1] - startCloud[i4 + 1]) * w;
            z = startCloud[i4 + 2] + (midCloud[i4 + 2] - startCloud[i4 + 2]) * w;
            sal = startCloud[i4 + 3] + (midCloud[i4 + 3] - startCloud[i4 + 3]) * w;
          } else {
            const w = smoothstep(0.5, 1.0, uOt);
            x = midCloud[i4 + 0] + (targetCloud[i4 + 0] - midCloud[i4 + 0]) * w;
            y = midCloud[i4 + 1] + (targetCloud[i4 + 1] - midCloud[i4 + 1]) * w;
            z = midCloud[i4 + 2] + (targetCloud[i4 + 2] - midCloud[i4 + 2]) * w;
            sal = midCloud[i4 + 3] + (targetCloud[i4 + 3] - midCloud[i4 + 3]) * w;
          }

          liveCloud[i4 + 0] = x;
          liveCloud[i4 + 1] = y;
          liveCloud[i4 + 2] = z;
          liveCloud[i4 + 3] = sal;

          // Continuous spatial motor fields applied strictly to the model's generated 3D point cloud
          if (uOt > 0.35 && z > 0.10) {
            const kinMask = smoothstep(0.35, 0.85, uOt);

            // 1. Discovered Eye & Iris Receptor Field (`sal >= 0.87` near left/right orbital centroids)
            if (sal >= 0.87 && y > -0.24 && y < 0.24) {
              const isLeft = x < 0;
              const ecx = isLeft ? landmarks.eyeLeftX : landmarks.eyeRightX;
              const ecy = isLeft ? landmarks.eyeLeftY : landmarks.eyeRightY;
              const dxE = x - ecx;
              const dyE = y - ecy;
              if (dxE * dxE + dyE * dyE < 0.032) {
                // Eyelid blink compression toward discovered orbital horizontal axis
                y = ecy + dyE * (1.0 - 0.86 * smBlink * kinMask);
                // Iris & pupil gaze tracking (`sal >= 0.94` predicted by PointScarFlow on iris/pupil)
                if (sal >= 0.94) {
                  x += clamp(smGazeX, -1.0, 1.0) * 0.028 * kinMask;
                  y -= clamp(smGazeY, -1.0, 1.0) * 0.020 * kinMask;
                }
              }
            }

            // 2. Superciliary Brow Field (`0.76 <= sal < 0.87` above orbits)
            if (sal >= 0.76 && sal < 0.87 && y >= -0.04 && y < 0.26 && Math.abs(x) > 0.05) {
              y += (smBrowRaise * 0.026 - smFurrow * 0.016) * kinMask;
              x -= Math.sign(x) * smFurrow * 0.012 * kinMask;
            }

            // 3. Mandibular & Articulatory Lip Field (`y < oralCleftY + 0.06`)
            if (y < landmarks.oralCleftY + 0.055 && Math.abs(x) < 0.42) {
              const oralProximity = Math.exp(-((x * x) / 0.045));
              if (y < landmarks.oralCleftY) {
                // Lower lip & mandible drop during Qwen token articulation
                const jawFactor = smoothstep(landmarks.oralCleftY, landmarks.oralCleftY - 0.22, y);
                const lipLowerFactor = sal >= 0.82 ? 1.0 : 0.65 * jawFactor;
                y -= smJaw * (0.095 * lipLowerFactor + 0.045 * jawFactor) * oralProximity * kinMask;
                z -= smJaw * 0.024 * jawFactor * kinMask;
              } else if (sal >= 0.82) {
                // Upper lip subtle lift + rounding
                y += smJaw * 0.018 * oralProximity * kinMask;
              }
              if (sal >= 0.80 && Math.abs(y - landmarks.oralCleftY) < 0.085) {
                x *= 1.0 + ((smLipSpread - 0.5) * 0.16 - smLipRound * 0.14) * kinMask;
                z += smLipRound * 0.022 * oralProximity * kinMask;
              }
            }
          }
        }

        y += breathY;

        const rx1 = x * cosY + z * sinY;
        const rz1 = -x * sinY + z * cosY;
        const ry2 = y * cosP - rz1 * sinP;
        const rz2 = y * sinP + rz1 * cosP;
        const rx3 = rx1 * cosR - ry2 * sinR;
        const ry3 = rx1 * sinR + ry2 * cosR;

        const persp = focal / Math.max(1.2, camZ - rz2);
        projX[i] = centerX + rx3 * persp * scalePx;
        projY[i] = centerY - ry3 * persp * scalePx;
        rotZ[i] = rz2;
        liveSal[i] = sal;
      }

      // PASS 1: Neural Constellation Edges Discovered from PointScarFlow's 3D Coordinates
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';

      const maxEdge3D = hasNeuralModelFace ? 0.165 : 0.24;
      const synWavePhase = nowMs * 0.0032;

      // Faint structural cranial/facial lattice (`tier 0`) + luminous anatomical ridges (`tier 1`)
      ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.13)`;
      ctx.lineWidth = 0.45 * strokeScale;
      ctx.beginPath();
      for (let i = 0; i < NUM_POINTS; i++) {
        if (rotZ[i] < -0.25) continue;
        const baseK = i * K_NEIGHBORS;
        for (let k = 0; k < 2; k++) {
          const j = landmarks.nbrIdx[baseK + k];
          if (j <= i || landmarks.nbrDist[baseK + k] > maxEdge3D) continue;
          if (0.5 * (liveSal[i] + liveSal[j]) >= 0.76) continue;
          ctx.moveTo(projX[i], projY[i]);
          ctx.lineTo(projX[j], projY[j]);
        }
      }
      ctx.stroke();

      ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.34)`;
      ctx.lineWidth = 0.68 * strokeScale;
      ctx.beginPath();
      for (let i = 0; i < NUM_POINTS; i++) {
        if (rotZ[i] < -0.10) continue;
        const baseK = i * K_NEIGHBORS;
        for (let k = 0; k < 3; k++) {
          const j = landmarks.nbrIdx[baseK + k];
          if (j <= i || landmarks.nbrDist[baseK + k] > maxEdge3D * 0.88) continue;
          const meanSal = 0.5 * (liveSal[i] + liveSal[j]);
          if (meanSal < 0.76) continue;
          ctx.moveTo(projX[i], projY[i]);
          ctx.lineTo(projX[j], projY[j]);
        }
      }
      ctx.stroke();

      // Live acoustic harmonic wavebars inside the discovered oral aperture during speech
      if (hasNeuralModelFace && smJaw > 0.04) {
        const harmonics = p.face.viseme.voiceHarmonics;
        const oy = landmarks.oralCleftY - smJaw * 0.038 + breathY;
        const oz = landmarks.oralZ;
        const rx1 = oz * sinY;
        const rz1 = oz * cosY;
        const ry2 = oy * cosP - rz1 * sinP;
        const rz2 = oy * sinP + rz1 * cosP;
        const rx3 = rx1 * cosR - ry2 * sinR;
        const ry3 = rx1 * sinR + ry2 * cosR;
        const persp = focal / Math.max(1.2, camZ - rz2);
        const mx = centerX + rx3 * persp * scalePx;
        const my = centerY - ry3 * persp * scalePx;

        ctx.strokeStyle = `rgba(${iceR}, ${iceG}, ${iceB}, 0.56)`;
        ctx.lineWidth = 0.85 * strokeScale;
        ctx.beginPath();
        for (let h = 0; h < harmonics.length; h++) {
          const u = (h - (harmonics.length - 1) * 0.5) * 0.032;
          const hx = mx + u * persp * scalePx;
          const osc = 0.70 + 0.30 * Math.sin(speechPhase * 1.3 + h * 0.9);
          const hh = Math.max(0.8, harmonics[h] * osc * smJaw * persp * scalePx * 0.085);
          ctx.moveTo(hx, my - hh);
          ctx.lineTo(hx, my + hh);
        }
        ctx.stroke();
      }

      // PASS 2: Depth-Sorted Neural 3D Points (`[x, y, z, salience]` from `PointScarFlow`)
      drawOrder.sort((a, b) => rotZ[a] - rotZ[b]);
      const pupilBoost = 0.85 + 0.55 * smPupil;

      for (let idx = 0; idx < NUM_POINTS; idx++) {
        const i = drawOrder[idx];
        const z = rotZ[i];
        const sal = liveSal[i];
        const depthCue = clamp((z + 1.05) / 2.05, 0.14, 1.0);
        const pulse = 0.86 + 0.14 * Math.sin(synWavePhase + i * 0.13);
        const isIrisCore = hasNeuralModelFace && sal >= 0.94 && z > 0.15;
        const isFeature = sal >= 0.78;

        if (isIrisCore && smBlink < 0.82) {
          const rHalo = (2.8 + 2.2 * smPupil) * strokeScale * (1.0 - 0.65 * smBlink);
          ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${(0.26 * depthCue).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(projX[i], projY[i], rHalo, 0, Math.PI * 2);
          ctx.fill();

          const rCore = (1.15 * pupilBoost) * strokeScale * (1.0 - 0.55 * smBlink);
          ctx.fillStyle = `rgba(245, 252, 255, ${(0.92 * depthCue).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(projX[i], projY[i], rCore, 0, Math.PI * 2);
          ctx.fill();
        } else {
          const rPt = (0.55 + sal * (isFeature ? 1.15 : 0.75)) * (0.72 + 0.42 * depthCue) * strokeScale;
          const alpha = clamp((0.14 + 0.62 * sal) * depthCue * pulse, 0.04, 0.88);
          if (isFeature && z > 0.0) {
            ctx.fillStyle = `rgba(${iceR}, ${iceG}, ${iceB}, ${alpha.toFixed(3)})`;
          } else {
            ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${(alpha * 0.78).toFixed(3)})`;
          }
          ctx.beginPath();
          ctx.arc(projX[i], projY[i], rPt, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // PASS 3: Live Qwen 3B GPU Shader Boot Progress Ring (when loading 1.60 GB LLM shards)
      if (p.isBooting) {
        const clampedPct = clamp(p.bootPercent, 0, 100) / 100.0;
        const bootR = minWH * 0.46;
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
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerleave', onPointerLeave);
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
        opacity: morphDimmed ? 0.12 : 1.0,
        transition: 'opacity 520ms cubic-bezier(0.22, 1, 0.36, 1), transform 200ms ease',
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
