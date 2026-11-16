import React, { useEffect, useRef } from 'react';
import type { JarvisFaceState } from '../lib/jarvisNeuralController';
import { buildJarvisFacePointCloudF32 } from './JarvisFaceRig';

export interface JarvisFloatingPointStageProps {
  enabled: boolean;
  stageActive?: boolean;
  face: JarvisFaceState;
  isGenerating: boolean;
  isExpanded: boolean;
  activeCapsuleIds: string[];
  unlearnShockActive: boolean;
  pointScarMidF32: Float32Array | null;
  pointScarFinalF32: Float32Array | null;
  pointScarMs: number | null;
  pointScarSeq: number; // increments each time a new GPU PointScarFlow cloud arrives
  lang: 'en' | 'pt';
}

const NUM_POINTS = 1024;

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

/**
 * Computes 2 nearest 3D structural neighbors per point (`Int16Array [numPoints]`)
 * directly from the GPU-synthesized 3D coordinates so high-salience ridges form crisp filaments.
 */
function computeNearest2Edges(cloudF32: Float32Array, numPoints = NUM_POINTS): {
  nbrA: Int16Array;
  nbrB: Int16Array;
  distA: Float32Array;
  distB: Float32Array;
} {
  const nbrA = new Int16Array(numPoints);
  const nbrB = new Int16Array(numPoints);
  const distA = new Float32Array(numPoints);
  const distB = new Float32Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    const ix = cloudF32[i * 4 + 0];
    const iy = cloudF32[i * 4 + 1];
    const iz = cloudF32[i * 4 + 2];
    let best1 = 1e9;
    let best2 = 1e9;
    let idx1 = -1;
    let idx2 = -1;
    for (let j = 0; j < numPoints; j++) {
      if (i === j) continue;
      const dx = ix - cloudF32[j * 4 + 0];
      const dy = iy - cloudF32[j * 4 + 1];
      const dz = iz - cloudF32[j * 4 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best1) {
        best2 = best1;
        idx2 = idx1;
        best1 = d2;
        idx1 = j;
      } else if (d2 < best2) {
        best2 = d2;
        idx2 = j;
      }
    }
    nbrA[i] = idx1;
    nbrB[i] = idx2;
    distA[i] = Math.sqrt(best1);
    distB[i] = Math.sqrt(best2);
  }
  return { nbrA, nbrB, distA, distB };
}

function resolveRelevantDomSectionId(capsuleIds: string[]): string | null {
  for (const cid of capsuleIds) {
    const low = cid.toLowerCase();
    if (low.includes('contact') || low.includes('hire')) return 'contact';
    if (low.includes('scarnative') || low.includes('vq4') || low.includes('tabula')) return 'scar-native-lab';
    if (low.includes('codec') || low.includes('rfi') || low.includes('cfd') || low.includes('landauer')) {
      return 'physical-scar-lab';
    }
    if (low.includes('aura') || low.includes('rubik') || low.includes('system_one') || low.includes('scout')) {
      return 'aura-origin';
    }
    if (low.includes('mos') || low.includes('scache') || low.includes('scar') || low.includes('jarvis')) {
      return 'scar-ecosystem';
    }
  }
  return null;
}

export const JarvisFloatingPointStage: React.FC<JarvisFloatingPointStageProps> = ({
  enabled,
  stageActive = true,
  face,
  isGenerating,
  isExpanded,
  activeCapsuleIds,
  unlearnShockActive,
  pointScarMidF32,
  pointScarFinalF32,
  pointScarMs,
  pointScarSeq,
  lang,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const propsRef = useRef({
    enabled,
    stageActive,
    face,
    isGenerating,
    isExpanded,
    activeCapsuleIds,
    unlearnShockActive,
    pointScarMidF32,
    pointScarFinalF32,
    pointScarMs,
    pointScarSeq,
    lang,
  });
  propsRef.current = {
    enabled,
    stageActive,
    face,
    isGenerating,
    isExpanded,
    activeCapsuleIds,
    unlearnShockActive,
    pointScarMidF32,
    pointScarFinalF32,
    pointScarMs,
    pointScarSeq,
    lang,
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Exact 1,024-point 3D cranio-facial topology of JARVIS (`t = 0` morph origin & return target)
    const faceCloudF32 = buildJarvisFacePointCloudF32(NUM_POINTS);

    // Active GPU PointScarFlow 3D manifolds (`[1024 * 4]`: `x, y, z, salience`)
    const sculptStartLocal = new Float32Array(faceCloudF32);
    const sculptMidLocal = new Float32Array(faceCloudF32);
    const sculptTargetLocal = new Float32Array(faceCloudF32);
    const sculptLiveLocal = new Float32Array(faceCloudF32);

    const screenX = new Float32Array(NUM_POINTS);
    const screenY = new Float32Array(NUM_POINTS);
    const magOffsetX = new Float32Array(NUM_POINTS);
    const magOffsetY = new Float32Array(NUM_POINTS);
    const depthZ = new Float32Array(NUM_POINTS);
    const salArr = new Float32Array(NUM_POINTS);
    const drawOrder = new Int16Array(NUM_POINTS);
    for (let i = 0; i < NUM_POINTS; i++) drawOrder[i] = i;

    let edgeGraph = computeNearest2Edges(faceCloudF32, NUM_POINTS);
    let lastAppliedSeq = 0;
    let retargetStartMs = 0;
    let isSculptureToSculptureTransition = false;
    let hasActiveSculpture = false;

    let pointerClientX = -9999;
    let pointerClientY = -9999;
    let pointerInside = false;

    const onPointerMove = (ev: PointerEvent) => {
      pointerClientX = ev.clientX;
      pointerClientY = ev.clientY;
      pointerInside = true;
    };
    const onPointerLeave = () => {
      pointerInside = false;
    };
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerleave', onPointerLeave);

    let morphProgress = 0.0; // 0.0 = resting on JARVIS's 3D face, 1.0 = morphed into PointScarFlow 3D manifold
    let turntableOffset = 0.0;
    let prevTimeMs = performance.now();
    let rafId = 0;

    const render = (nowMs: number) => {
      const p = propsRef.current;
      const dt = clamp((nowMs - prevTimeMs) * 0.001, 0.001, 0.05);
      prevTimeMs = nowMs;

      // Measure exact CSS viewport box on the canvas itself so Retina `dpr = 2/3` never shifts coordinates
      const canvasRect = canvas.getBoundingClientRect();
      const vw = Math.max(1, canvasRect.width || window.innerWidth || 1280);
      const vh = Math.max(1, canvasRect.height || window.innerHeight || 800);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.max(1, Math.round(vw * dpr));
      const targetH = Math.max(1, Math.round(vh * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, vw, vh);

      if (p.pointScarSeq === 0 && lastAppliedSeq !== 0) {
        lastAppliedSeq = 0;
        hasActiveSculpture = false;
      }

      // Apply newly synthesized GPU PointScarFlow 3D manifold (`X_0.5` -> `X_1.0`)
      if (p.pointScarSeq !== lastAppliedSeq && p.pointScarFinalF32 && p.pointScarFinalF32.length >= NUM_POINTS * 4) {
        lastAppliedSeq = p.pointScarSeq;
        if (morphProgress > 0.35 && hasActiveSculpture) {
          sculptStartLocal.set(sculptLiveLocal);
          isSculptureToSculptureTransition = true;
        } else {
          sculptStartLocal.set(faceCloudF32);
          isSculptureToSculptureTransition = false;
        }

        if (p.pointScarMidF32 && p.pointScarMidF32.length >= NUM_POINTS * 4) {
          sculptMidLocal.set(p.pointScarMidF32.subarray(0, NUM_POINTS * 4));
        } else {
          for (let k = 0; k < NUM_POINTS * 4; k++) {
            sculptMidLocal[k] = 0.5 * (sculptStartLocal[k] + p.pointScarFinalF32[k]);
          }
        }
        sculptTargetLocal.set(p.pointScarFinalF32.subarray(0, NUM_POINTS * 4));
        edgeGraph = computeNearest2Edges(sculptTargetLocal, NUM_POINTS);
        retargetStartMs = nowMs;
        hasActiveSculpture = true;
      }

      const wantSculpture = p.enabled && p.stageActive && hasActiveSculpture;
      if (wantSculpture) {
        morphProgress = Math.min(1.0, morphProgress + dt / 0.88);
      } else {
        morphProgress = Math.max(0.0, morphProgress - dt / 0.64);
      }

      if (morphProgress < 0.004) {
        turntableOffset = 0.0;
        rafId = window.requestAnimationFrame(render);
        return;
      }

      // Lock 1:1 to JARVIS's live 3D face rig in the exact same CSS coordinate space as `canvasRect`
      const faceEl = document.querySelector('[data-testid="jarvis-floating-face-rig"]');
      if (!faceEl) {
        rafId = window.requestAnimationFrame(render);
        return;
      }
      const faceRect = faceEl.getBoundingClientRect();
      if (faceRect.width < 8 || faceRect.height < 8) {
        rafId = window.requestAnimationFrame(render);
        return;
      }

      const localFaceLeft = faceRect.left - canvasRect.left;
      const localFaceTop = faceRect.top - canvasRect.top;
      const faceW = faceRect.width;
      const faceH = faceRect.height;
      const minWH = Math.min(faceW, faceH);

      const morphW = smoothstep(0.0, 1.0, morphProgress);

      // Exact `JarvisFaceRig` camera at `morphW = 0` (`centerY = 0.555 * H`, `scale = 0.56 * minWH`)
      // smoothly centered to head-box midpoint at `morphW = 1` (`centerY = 0.485 * H`, `scale = 0.40 * minWH`)
      const originX = localFaceLeft + faceW * 0.5;
      const originY = localFaceTop + faceH * (0.555 - 0.070 * morphW);
      const scalePx = minWH * (0.56 - 0.16 * morphW);
      const camZ = 3.62;
      const focal = 2.84;

      const harmonics = p.face.viseme.voiceHarmonics;
      const voiceEnergy =
        p.isGenerating
          ? (harmonics[0] + harmonics[1] + harmonics[2] + harmonics[3]) * 0.25
          : 0.0;
      const epistemicCrisp = clamp(0.45 + 0.55 * p.face.epistemicMargin, 0.35, 1.0);
      const isShieldPosture = p.face.posture === 'OUT_OF_CONE_GUARD';

      // Continuous wrapped turntable rotation (`[-pi, +pi]`) so returning to JARVIS's face never unwinds multiple turns
      if (wantSculpture) {
        const rotSpeed = p.isGenerating ? 0.50 : 0.32;
        turntableOffset += dt * rotSpeed * morphW;
        turntableOffset = Math.atan2(Math.sin(turntableOffset), Math.cos(turntableOffset));
      }

      const ptrLocalX = pointerClientX - canvasRect.left;
      const ptrLocalY = pointerClientY - canvasRect.top;
      const ptrNormX = pointerInside ? clamp((ptrLocalX - originX) / Math.max(180.0, minWH * 1.4), -1.0, 1.0) : 0.0;
      const ptrNormY = pointerInside ? clamp((ptrLocalY - originY) / Math.max(180.0, minWH * 1.4), -1.0, 1.0) : 0.0;

      const faceYawRad = p.face.headYaw * 0.36 + p.face.gazeX * 0.20;
      const facePitchRad = p.face.headPitch * 0.28 - p.face.gazeY * 0.16;
      const faceRollRad = p.face.headRoll * 0.22;

      const yaw = faceYawRad * (1.0 - 0.70 * morphW) + turntableOffset * morphW + ptrNormX * 0.34 * morphW;
      const pitch = facePitchRad * (1.0 - 0.60 * morphW) - 0.12 * morphW + ptrNormY * 0.22 * morphW;
      const roll = faceRollRad * (1.0 - morphW);

      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);
      const cosR = Math.cos(roll);
      const sinR = Math.sin(roll);

      const [pr, pg, pb] = parseHexRgb(p.face.primaryHex);
      const [sr, sg, sb] = parseHexRgb(p.face.secondaryHex);

      const retargetElapsedS = Math.max(0.0, (nowMs - retargetStartMs) * 0.001);
      const springMag = 1.0 - Math.exp(-dt * 16.0);
      const magRadiusPx = Math.max(54.0, minWH * 0.32);
      const magRadiusSq = magRadiusPx * magRadiusPx;

      // Transform all 1,024 points between JARVIS's 3D cranio-facial topology and GPU OT manifold
      for (let i = 0; i < NUM_POINTS; i++) {
        const i4 = i * 4;
        const stagger = i / NUM_POINTS;

        // 1) Active sculpture coordinate (`sculptLiveLocal`) for smooth sculpture->sculpture updates
        let sx3: number;
        let sy3: number;
        let sz3: number;
        let sSal: number;
        if (isSculptureToSculptureTransition) {
          const uRet = clamp((retargetElapsedS - 0.14 * stagger) / 0.72, 0.0, 1.0);
          if (uRet < 0.5) {
            const w = smoothstep(0.0, 0.5, uRet);
            sx3 = sculptStartLocal[i4 + 0] + (sculptMidLocal[i4 + 0] - sculptStartLocal[i4 + 0]) * w;
            sy3 = sculptStartLocal[i4 + 1] + (sculptMidLocal[i4 + 1] - sculptStartLocal[i4 + 1]) * w;
            sz3 = sculptStartLocal[i4 + 2] + (sculptMidLocal[i4 + 2] - sculptStartLocal[i4 + 2]) * w;
            sSal = sculptStartLocal[i4 + 3] + (sculptMidLocal[i4 + 3] - sculptStartLocal[i4 + 3]) * w;
          } else {
            const w = smoothstep(0.5, 1.0, uRet);
            sx3 = sculptMidLocal[i4 + 0] + (sculptTargetLocal[i4 + 0] - sculptMidLocal[i4 + 0]) * w;
            sy3 = sculptMidLocal[i4 + 1] + (sculptTargetLocal[i4 + 1] - sculptMidLocal[i4 + 1]) * w;
            sz3 = sculptMidLocal[i4 + 2] + (sculptTargetLocal[i4 + 2] - sculptMidLocal[i4 + 2]) * w;
            sSal = sculptMidLocal[i4 + 3] + (sculptTargetLocal[i4 + 3] - sculptMidLocal[i4 + 3]) * w;
          }
        } else {
          sx3 = sculptTargetLocal[i4 + 0];
          sy3 = sculptTargetLocal[i4 + 1];
          sz3 = sculptTargetLocal[i4 + 2];
          sSal = sculptTargetLocal[i4 + 3];
        }
        sculptLiveLocal[i4 + 0] = sx3;
        sculptLiveLocal[i4 + 1] = sy3;
        sculptLiveLocal[i4 + 2] = sz3;
        sculptLiveLocal[i4 + 3] = sSal;

        // 2) Bidirectional 2-Stage OT Morph (`JARVIS 3D Face <-> X_0.5 <-> PointScarFlow X_1.0`)
        const uMorph = clamp((morphProgress - 0.14 * stagger) / 0.86, 0.0, 1.0);
        const fx = faceCloudF32[i4 + 0];
        const fy = faceCloudF32[i4 + 1];
        const fz = faceCloudF32[i4 + 2];
        const fSal = faceCloudF32[i4 + 3];

        let lx: number;
        let ly: number;
        let lz: number;
        let sal: number;
        if (uMorph < 0.5) {
          const w = smoothstep(0.0, 0.5, uMorph);
          lx = fx + (sculptMidLocal[i4 + 0] - fx) * w;
          ly = fy + (sculptMidLocal[i4 + 1] - fy) * w;
          lz = fz + (sculptMidLocal[i4 + 2] - fz) * w;
          sal = fSal + (sculptMidLocal[i4 + 3] - fSal) * w;
        } else {
          const w = smoothstep(0.5, 1.0, uMorph);
          lx = sculptMidLocal[i4 + 0] + (sx3 - sculptMidLocal[i4 + 0]) * w;
          ly = sculptMidLocal[i4 + 1] + (sy3 - sculptMidLocal[i4 + 1]) * w;
          lz = sculptMidLocal[i4 + 2] + (sz3 - sculptMidLocal[i4 + 2]) * w;
          sal = sculptMidLocal[i4 + 3] + (sSal - sculptMidLocal[i4 + 3]) * w;
        }

        // Subtle acoustic harmonic ripple + epistemic shimmer + 0x00 unlearn shockwave scaled by `morphW`
        const band = harmonics[i & 7];
        const acousticPulse =
          1.0 + morphW * voiceEnergy * (0.042 + 0.052 * band) * Math.sin(nowMs * 0.012 + i * 0.07);
        const unlearnShock = p.unlearnShockActive
          ? 1.0 + 0.25 * morphW * Math.sin(nowMs * 0.018 + i * 0.03)
          : 1.0;
        const jitterAmp = (1.0 - epistemicCrisp) * 0.018 * morphW;
        lx = lx * acousticPulse * unlearnShock + jitterAmp * Math.sin(nowMs * 0.005 + i * 1.7);
        ly = ly * acousticPulse * unlearnShock + jitterAmp * Math.cos(nowMs * 0.006 + i * 1.3);
        lz = lz * acousticPulse * unlearnShock;

        // 3) Exact 3D Yaw + Pitch + Roll + Perspective projection matching `JarvisFaceRig`
        const rx1 = lx * cosY + lz * sinY;
        const rz1 = -lx * sinY + lz * cosY;
        const ry2 = ly * cosP - rz1 * sinP;
        const rz2 = ly * sinP + rz1 * cpSafe(cosP);
        const rx3 = rx1 * cosR - ry2 * sinR;
        const ry3 = rx1 * sinR + ry2 * cosR;

        const persp = focal / Math.max(1.2, camZ - rz2);
        const baseScreenX = originX + rx3 * persp * scalePx;
        const baseScreenY = originY - ry3 * persp * scalePx;

        // Smooth local magnetic cursor swirl (zero lag when page scrolls or widget expands!)
        let targetMagX = 0.0;
        let targetMagY = 0.0;
        if (pointerInside && morphW > 0.15) {
          const dx = baseScreenX - ptrLocalX;
          const dy = baseScreenY - ptrLocalY;
          const d2 = dx * dx + dy * dy;
          if (d2 < magRadiusSq && d2 > 1e-3) {
            const d = Math.sqrt(d2);
            const push = (1.0 - d / magRadiusPx) * 15.0 * morphW;
            targetMagX = (dx / d) * push;
            targetMagY = (dy / d) * push;
          }
        }
        magOffsetX[i] += (targetMagX - magOffsetX[i]) * springMag;
        magOffsetY[i] += (targetMagY - magOffsetY[i]) * springMag;

        screenX[i] = baseScreenX + magOffsetX[i];
        screenY[i] = baseScreenY + magOffsetY[i];
        depthZ[i] = rz2;
        salArr[i] = sal;
      }

      // Smooth cross-dissolve on initial emergence & final merge onto JARVIS's 3D face
      const stageAlpha = wantSculpture
        ? smoothstep(0.0, 0.18, morphProgress)
        : smoothstep(0.0, 0.24, morphProgress);

      ctx.save();
      ctx.globalAlpha = clamp(stageAlpha, 0.0, 1.0);

      // Pass 1: Subtle DOM Section Guiding Arc (only in compact corner mode when page sections are unblurred)
      const secId = !p.isExpanded && morphW > 0.45 ? resolveRelevantDomSectionId(p.activeCapsuleIds) : null;
      if (secId) {
        const secEl = document.getElementById(secId);
        if (secEl) {
          const r = secEl.getBoundingClientRect();
          const localTopSec = r.top - canvasRect.top;
          const localBottomSec = r.bottom - canvasRect.top;
          if (localBottomSec > 40 && localTopSec < vh - 40) {
            const targetDomX = clamp(r.left - canvasRect.left + Math.min(220, r.width * 0.28), 40, vw - 40);
            const targetDomY = clamp(localTopSec + 36, 48, vh - 48);
            const domGrad = ctx.createLinearGradient(originX, originY, targetDomX, targetDomY);
            domGrad.addColorStop(0.0, `rgba(${pr}, ${pg}, ${pb}, ${(0.24 * morphW).toFixed(3)})`);
            domGrad.addColorStop(1.0, `rgba(${sr}, ${sg}, ${sb}, ${(0.06 * morphW).toFixed(3)})`);
            ctx.strokeStyle = domGrad;
            ctx.setLineDash([4, 6]);
            ctx.lineWidth = 1.1;
            ctx.beginPath();
            ctx.moveTo(originX, originY);
            ctx.quadraticCurveTo(
              0.5 * (originX + targetDomX),
              Math.min(originY, targetDomY) - 42,
              targetDomX,
              targetDomY
            );
            ctx.stroke();
            ctx.setLineDash([]);

            const uPulse = (nowMs * 0.00055) % 1.0;
            const cx = 0.5 * (originX + targetDomX);
            const cy = Math.min(originY, targetDomY) - 42;
            const px =
              (1 - uPulse) * (1 - uPulse) * originX +
              2 * (1 - uPulse) * uPulse * cx +
              uPulse * uPulse * targetDomX;
            const py =
              (1 - uPulse) * (1 - uPulse) * originY +
              2 * (1 - uPulse) * uPulse * cy +
              uPulse * uPulse * targetDomY;
            ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${(0.82 * morphW).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(px, py, 2.4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Pass 2: Co-centered Epistemic Shield Halo around JARVIS's morphed head when `OUT_OF_CONE_GUARD` is active
      if (isShieldPosture && morphW > 0.25) {
        const shieldR = minWH * 0.405;
        ctx.strokeStyle = `rgba(251, 191, 36, ${(0.25 * morphW).toFixed(3)})`;
        ctx.lineWidth = 1.15;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(originX, originY, shieldR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Pass 3: Structural `k=2` Nearest-Neighbor Ridge Filaments on GPU-synthesized 3D coordinates
      if (morphW > 0.32) {
        const edgeFade = smoothstep(0.32, 0.92, morphW);
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, ${(0.16 * edgeFade).toFixed(3)})`;
        ctx.lineWidth = 0.72;
        ctx.beginPath();
        const { nbrA, nbrB, distA, distB } = edgeGraph;
        for (let i = 0; i < NUM_POINTS; i += 2) {
          if (salArr[i] < 0.36) continue;
          const j1 = nbrA[i];
          if (j1 >= 0 && distA[i] < 0.19) {
            ctx.moveTo(screenX[i], screenY[i]);
            ctx.lineTo(screenX[j1], screenY[j1]);
          }
          const j2 = nbrB[i];
          if (j2 >= 0 && distB[i] < 0.15 && salArr[i] > 0.58) {
            ctx.moveTo(screenX[i], screenY[i]);
            ctx.lineTo(screenX[j2], screenY[j2]);
          }
        }
        ctx.stroke();
      }

      // Sort back-to-front by `depthZ` for true 3D occlusion & volumetric luminance
      drawOrder.sort((a, b) => depthZ[a] - depthZ[b]);

      // Pass 4: Volumetric Bioluminescent Halo + Crisp Core Pinpoints (`1,024` points)
      const ptScale = Math.pow(Math.max(0.9, minWH / 214.0), 0.35);
      for (let k = 0; k < NUM_POINTS; k++) {
        const i = drawOrder[k];
        const zNorm = clamp((depthZ[i] + 1.05) / 2.1, 0.0, 1.0);
        const sal = salArr[i];
        const alpha = clamp(0.90 * (0.26 + 0.74 * zNorm) * (0.52 + 0.48 * sal), 0.06, 0.97);
        const radius = (0.95 + 1.38 * zNorm * (0.52 + 0.65 * sal)) * ptScale;

        if (sal > 0.62 && zNorm > 0.50) {
          ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${(alpha * 0.23).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(screenX[i], screenY[i], radius * 2.25, 0, Math.PI * 2);
          ctx.fill();
        }

        const rMix = Math.round(pr + (245 - pr) * zNorm * 0.56);
        const gMix = Math.round(pg + (250 - pg) * zNorm * 0.56);
        const bMix = Math.round(pb + (255 - pb) * zNorm * 0.56);
        ctx.fillStyle = `rgba(${rMix}, ${gMix}, ${bMix}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(screenX[i], screenY[i], radius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
      rafId = window.requestAnimationFrame(render);
    };

    rafId = window.requestAnimationFrame(render);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerleave', onPointerLeave);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      data-testid="jarvis-floating-point-stage"
      style={{ width: '100vw', height: '100vh' }}
      className="fixed inset-0 w-screen h-screen z-[52] pointer-events-none select-none"
      aria-hidden="true"
    />
  );
};

function cpSafe(v: number): number {
  return v;
}
