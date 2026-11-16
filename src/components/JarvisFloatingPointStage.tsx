import React, { useEffect, useRef } from 'react';
import type { JarvisFaceState } from '../lib/jarvisNeuralController';

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
const GOLDEN_ANGLE = Math.PI * (3.0 - Math.sqrt(5.0));

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
 * Deterministic canonical Fibonacci sphere `X_0 [1024, 4]` matching `point_scar_flow.py` bit-for-bit.
 * ZERO `Math.random()`.
 */
function buildCanonicalSphereF32(numPoints = NUM_POINTS): Float32Array {
  const out = new Float32Array(numPoints * 4);
  for (let i = 0; i < numPoints; i++) {
    const y = 1.0 - (2.0 * (i + 0.5)) / numPoints;
    const rad = Math.sqrt(Math.max(0.0, 1.0 - y * y));
    const th = i * GOLDEN_ANGLE;
    out[i * 4 + 0] = rad * Math.cos(th);
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = rad * Math.sin(th);
    out[i * 4 + 3] = 0.45 + 0.35 * Math.abs(y);
  }
  return out;
}

/**
 * Computes 2 nearest 3D structural neighbors per point (`Int16Array [numPoints * 2]`)
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

    const canonicalSphere = buildCanonicalSphereF32(NUM_POINTS);

    // Current interpolated local 3D manifold coordinates + salience `[1024 * 4]`
    const curLocal = new Float32Array(canonicalSphere);
    const startLocal = new Float32Array(canonicalSphere);
    const midLocal = new Float32Array(canonicalSphere);
    const targetLocal = new Float32Array(canonicalSphere);

    // Per-particle screen-space spring positions for smooth detachment from JARVIS's cranial halo
    const screenX = new Float32Array(NUM_POINTS);
    const screenY = new Float32Array(NUM_POINTS);
    const depthZ = new Float32Array(NUM_POINTS);
    const salArr = new Float32Array(NUM_POINTS);
    const drawOrder = new Int16Array(NUM_POINTS);
    for (let i = 0; i < NUM_POINTS; i++) drawOrder[i] = i;

    let edgeGraph = computeNearest2Edges(canonicalSphere, NUM_POINTS);
    let lastAppliedSeq = 0;
    let morphStartMs = 0;
    let hasActiveSculpture = false;

    let pointerX = -9999;
    let pointerY = -9999;
    let pointerInside = false;

    const onPointerMove = (ev: PointerEvent) => {
      pointerX = ev.clientX;
      pointerY = ev.clientY;
      pointerInside = true;
    };
    const onPointerLeave = () => {
      pointerInside = false;
    };
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerleave', onPointerLeave);

    let initializedPositions = false;
    let stageAnchorX = 0;
    let stageAnchorY = 0;
    let stageVisibility = 0.0;
    let prevTimeMs = performance.now();
    let turntableAngle = 0.0;
    let rafId = 0;

    const render = (nowMs: number) => {
      const p = propsRef.current;
      const dt = clamp((nowMs - prevTimeMs) * 0.001, 0.001, 0.05);
      prevTimeMs = nowMs;

      const vw = window.innerWidth || 1280;
      const vh = window.innerHeight || 800;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.max(1, Math.round(vw * dpr));
      const targetH = Math.max(1, Math.round(vh * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, vw, vh);

      if (!p.enabled) {
        stageVisibility = 0.0;
        rafId = window.requestAnimationFrame(render);
        return;
      }

      if (p.pointScarSeq === 0 && lastAppliedSeq !== 0) {
        lastAppliedSeq = 0;
        hasActiveSculpture = false;
        curLocal.set(canonicalSphere);
      }

      // Check if a new GPU-synthesized PointScarFlow 3D manifold arrived
      if (p.pointScarSeq !== lastAppliedSeq && p.pointScarFinalF32 && p.pointScarFinalF32.length >= NUM_POINTS * 4) {
        lastAppliedSeq = p.pointScarSeq;
        startLocal.set(curLocal);
        if (p.pointScarMidF32 && p.pointScarMidF32.length >= NUM_POINTS * 4) {
          midLocal.set(p.pointScarMidF32.subarray(0, NUM_POINTS * 4));
        } else {
          for (let k = 0; k < NUM_POINTS * 4; k++) {
            midLocal[k] = 0.5 * (startLocal[k] + p.pointScarFinalF32[k]);
          }
        }
        targetLocal.set(p.pointScarFinalF32.subarray(0, NUM_POINTS * 4));
        edgeGraph = computeNearest2Edges(targetLocal, NUM_POINTS);
        morphStartMs = nowMs;
        hasActiveSculpture = true;
      }

      const targetVis = hasActiveSculpture && p.stageActive ? 1.0 : p.isExpanded ? 0.35 : 0.0;
      stageVisibility += (targetVis - stageVisibility) * (1.0 - Math.exp(-dt * 6.5));
      if (stageVisibility < 0.012) {
        initializedPositions = false;
        rafId = window.requestAnimationFrame(render);
        return;
      }

      // Locate JARVIS's live 3D face rig in viewport coordinates
      const faceEl = document.querySelector('[data-testid="jarvis-floating-face-rig"]');
      const faceRect = faceEl ? faceEl.getBoundingClientRect() : null;
      const faceCx = faceRect ? faceRect.left + faceRect.width * 0.5 : vw - 140;
      const faceCy = faceRect ? faceRect.top + faceRect.height * 0.48 : vh - 180;
      const faceRadius = faceRect ? Math.min(faceRect.width, faceRect.height) * 0.50 : 108;

      // Determine target 3D sculpture stage center in unobstructed open viewport space
      let desiredStageX = faceCx;
      let desiredStageY = faceCy;
      let desiredRadius = faceRadius * 1.08;

      if (hasActiveSculpture) {
        if (p.isExpanded) {
          // Place sculpture cleanly to the left of expanded centered JARVIS on desktop, or above on narrow viewports
          if (vw >= 960) {
            desiredStageX = clamp(faceCx - Math.min(340, vw * 0.28), 180, vw - 180);
            desiredStageY = clamp(faceCy - 10, 170, vh - 190);
            desiredRadius = Math.min(168, vw * 0.135);
          } else {
            desiredStageX = vw * 0.5;
            desiredStageY = clamp(faceCy - faceRadius - 115, 110, vh * 0.36);
            desiredRadius = Math.min(110, vw * 0.22);
          }
        } else {
          // Compact bottom-right corner mode: project floating sculpture into open space left-and-slightly-above JARVIS
          if (vw >= 760) {
            desiredStageX = clamp(faceCx - Math.min(350, vw * 0.30), 175, vw - 260);
            desiredStageY = clamp(faceCy - 72, 165, vh - 175);
            desiredRadius = Math.min(148, vw * 0.145, vh * 0.20);
          } else {
            desiredStageX = vw * 0.5;
            desiredStageY = clamp(faceCy - 250, 130, vh - 310);
            desiredRadius = Math.min(112, vw * 0.26);
          }
        }
      }

      if (!initializedPositions) {
        stageAnchorX = desiredStageX;
        stageAnchorY = desiredStageY;
      } else {
        const anchorAlpha = 1.0 - Math.exp(-dt * 5.2);
        stageAnchorX += (desiredStageX - stageAnchorX) * anchorAlpha;
        stageAnchorY += (desiredStageY - stageAnchorY) * anchorAlpha;
      }

      // Compute OT Rectified Flow morph progression `u in [0, 1]` (`0 -> X_0.5 -> X_1.0`)
      const morphElapsedS = hasActiveSculpture ? Math.max(0.0, (nowMs - morphStartMs) * 0.001) : 1.0;
      const harmonics = p.face.viseme.voiceHarmonics;
      const voiceEnergy =
        p.isGenerating
          ? (harmonics[0] + harmonics[1] + harmonics[2] + harmonics[3]) * 0.25
          : 0.0;
      const epistemicCrisp = clamp(0.45 + 0.55 * p.face.epistemicMargin, 0.35, 1.0);
      const isShieldPosture = p.face.posture === 'OUT_OF_CONE_GUARD';

      // Gentle 3D turntable rotation + pointer parallax
      const rotSpeed = hasActiveSculpture ? (p.isGenerating ? 0.52 : 0.34) : 0.24;
      turntableAngle += dt * rotSpeed;

      const ptrNormX = pointerInside ? clamp((pointerX - stageAnchorX) / 360.0, -1.0, 1.0) : 0.0;
      const ptrNormY = pointerInside ? clamp((pointerY - stageAnchorY) / 360.0, -1.0, 1.0) : 0.0;
      const yaw = turntableAngle + ptrNormX * 0.36;
      const pitch = -0.16 + ptrNormY * 0.26;
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);

      const [pr, pg, pb] = parseHexRgb(p.face.primaryHex);
      const [sr, sg, sb] = parseHexRgb(p.face.secondaryHex);

      // Update 1,024 particles along real GPU OT trajectory (`start -> mid -> target`)
      for (let i = 0; i < NUM_POINTS; i++) {
        const i4 = i * 4;
        let lx: number;
        let ly: number;
        let lz: number;
        let sal: number;

        if (!hasActiveSculpture) {
          // Idle ambient dual-ring synaptic halo orbiting JARVIS's 3D head (`0` Math.random!)
          const baseX = canonicalSphere[i4 + 0];
          const baseY = canonicalSphere[i4 + 1];
          const baseZ = canonicalSphere[i4 + 2];
          const ringFlatten = i % 3 === 0 ? 0.22 : 1.0;
          const breath = 1.0 + 0.035 * Math.sin(nowMs * 0.0018 + i * 0.015);
          lx = baseX * breath;
          ly = baseY * ringFlatten * breath;
          lz = baseZ * breath;
          sal = canonicalSphere[i4 + 3] * 0.55;
          curLocal[i4 + 0] = lx;
          curLocal[i4 + 1] = ly;
          curLocal[i4 + 2] = lz;
          curLocal[i4 + 3] = sal;
        } else {
          // Staggered detachment from JARVIS -> `X_0.5` -> `X_1.0` over `0.95s`
          const delayS = 0.22 * (i / NUM_POINTS);
          const u = clamp((morphElapsedS - delayS) / 0.78, 0.0, 1.0);
          if (u < 0.5) {
            const w = smoothstep(0.0, 0.5, u);
            lx = startLocal[i4 + 0] + (midLocal[i4 + 0] - startLocal[i4 + 0]) * w;
            ly = startLocal[i4 + 1] + (midLocal[i4 + 1] - startLocal[i4 + 1]) * w;
            lz = startLocal[i4 + 2] + (midLocal[i4 + 2] - startLocal[i4 + 2]) * w;
            sal = startLocal[i4 + 3] + (midLocal[i4 + 3] - startLocal[i4 + 3]) * w;
          } else {
            const w = smoothstep(0.5, 1.0, u);
            lx = midLocal[i4 + 0] + (targetLocal[i4 + 0] - midLocal[i4 + 0]) * w;
            ly = midLocal[i4 + 1] + (targetLocal[i4 + 1] - midLocal[i4 + 1]) * w;
            lz = midLocal[i4 + 2] + (targetLocal[i4 + 2] - midLocal[i4 + 2]) * w;
            sal = midLocal[i4 + 3] + (targetLocal[i4 + 3] - midLocal[i4 + 3]) * w;
          }
          curLocal[i4 + 0] = lx;
          curLocal[i4 + 1] = ly;
          curLocal[i4 + 2] = lz;
          curLocal[i4 + 3] = sal;

          // Subtle acoustic harmonic ripple + epistemic shimmer + 0x00 unlearn shockwave
          const band = harmonics[i & 7];
          const acousticPulse = 1.0 + voiceEnergy * (0.045 + 0.055 * band) * Math.sin(nowMs * 0.012 + i * 0.07);
          const unlearnShock = p.unlearnShockActive
            ? 1.0 + 0.28 * Math.sin(nowMs * 0.018 + i * 0.03)
            : 1.0;
          const jitterAmp = (1.0 - epistemicCrisp) * 0.022;
          lx = lx * acousticPulse * unlearnShock + jitterAmp * Math.sin(nowMs * 0.005 + i * 1.7);
          ly = ly * acousticPulse * unlearnShock + jitterAmp * Math.cos(nowMs * 0.006 + i * 1.3);
          lz = lz * acousticPulse * unlearnShock;
        }

        // 3D Yaw + Pitch + Perspective projection (`+Y` up -> `-Y` screen)
        const rx1 = lx * cosY + lz * sinY;
        const rz1 = -lx * sinY + lz * cosY;
        const ry2 = ly * cosP - rz1 * sinP;
        const rz2 = ly * sinP + rz1 * cosP;

        const persp = 2.65 / Math.max(1.2, 2.65 - rz2 * 0.68);

        // During detachment (`u < 1`), smoothly glide particle center from JARVIS's face to `stageAnchor`!
        const detachW = hasActiveSculpture
          ? smoothstep(0.0, 0.85, clamp((morphElapsedS - 0.18 * (i / NUM_POINTS)) / 0.68, 0.0, 1.0))
          : 0.0;
        const originX = faceCx + (stageAnchorX - faceCx) * detachW;
        const originY = faceCy + (stageAnchorY - faceCy) * detachW;
        const effRad = faceRadius * 1.06 + (desiredRadius - faceRadius * 1.06) * detachW;

        let tx = originX + rx1 * effRad * persp;
        let ty = originY - ry2 * effRad * persp;

        // Interactive cursor magnetic swirl when visitor hovers near floating points
        if (pointerInside) {
          const dx = tx - pointerX;
          const dy = ty - pointerY;
          const d2 = dx * dx + dy * dy;
          if (d2 < 8400 && d2 > 1e-3) {
            const d = Math.sqrt(d2);
            const push = (1.0 - d / 91.65) * 18.0;
            tx += (dx / d) * push;
            ty += (dy / d) * push;
          }
        }

        if (!initializedPositions) {
          screenX[i] = tx;
          screenY[i] = ty;
        } else {
          const spring = 1.0 - Math.exp(-dt * 16.5);
          screenX[i] += (tx - screenX[i]) * spring;
          screenY[i] += (ty - screenY[i]) * spring;
        }
        depthZ[i] = rz2;
        salArr[i] = sal;
      }
      initializedPositions = true;

      ctx.save();
      ctx.globalAlpha = clamp(stageVisibility, 0.0, 1.0);

      // Pass 1: Synaptic umbilical stream connecting JARVIS's cranial constellation to the 3D Point-Scar Stage
      if (hasActiveSculpture) {
        const grad = ctx.createLinearGradient(faceCx, faceCy, stageAnchorX, stageAnchorY);
        grad.addColorStop(0.0, `rgba(${pr}, ${pg}, ${pb}, 0.22)`);
        grad.addColorStop(0.5, `rgba(${sr}, ${sg}, ${sb}, 0.12)`);
        grad.addColorStop(1.0, `rgba(${pr}, ${pg}, ${pb}, 0.24)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.1;
        for (let b = -1; b <= 1; b++) {
          const midX = 0.5 * (faceCx + stageAnchorX);
          const midY = 0.5 * (faceCy + stageAnchorY) - 28 + b * 18 * Math.sin(nowMs * 0.002 + b);
          ctx.beginPath();
          ctx.moveTo(faceCx, faceCy);
          ctx.quadraticCurveTo(midX, midY, stageAnchorX, stageAnchorY);
          ctx.stroke();
        }
      }

      // Pass 2: Subtle DOM Section Guiding Arc (when active ScarStore cartridge maps to a visible page section)
      const secId = hasActiveSculpture ? resolveRelevantDomSectionId(p.activeCapsuleIds) : null;
      if (secId) {
        const secEl = document.getElementById(secId);
        if (secEl) {
          const r = secEl.getBoundingClientRect();
          if (r.bottom > 40 && r.top < vh - 40) {
            const targetDomX = clamp(r.left + Math.min(220, r.width * 0.28), 40, vw - 40);
            const targetDomY = clamp(r.top + 36, 48, vh - 48);
            const domGrad = ctx.createLinearGradient(stageAnchorX, stageAnchorY, targetDomX, targetDomY);
            domGrad.addColorStop(0.0, `rgba(${pr}, ${pg}, ${pb}, 0.26)`);
            domGrad.addColorStop(1.0, `rgba(${sr}, ${sg}, ${sb}, 0.06)`);
            ctx.strokeStyle = domGrad;
            ctx.setLineDash([4, 6]);
            ctx.lineWidth = 1.15;
            ctx.beginPath();
            ctx.moveTo(stageAnchorX, stageAnchorY);
            ctx.quadraticCurveTo(
              0.5 * (stageAnchorX + targetDomX),
              Math.min(stageAnchorY, targetDomY) - 42,
              targetDomX,
              targetDomY
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Traveling harmonic pulse dot along the DOM pointer arc
            const uPulse = (nowMs * 0.00055) % 1.0;
            const cx = 0.5 * (stageAnchorX + targetDomX);
            const cy = Math.min(stageAnchorY, targetDomY) - 42;
            const px =
              (1 - uPulse) * (1 - uPulse) * stageAnchorX +
              2 * (1 - uPulse) * uPulse * cx +
              uPulse * uPulse * targetDomX;
            const py =
              (1 - uPulse) * (1 - uPulse) * stageAnchorY +
              2 * (1 - uPulse) * uPulse * cy +
              uPulse * uPulse * targetDomY;
            ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, 0.85)`;
            ctx.beginPath();
            ctx.arc(px, py, 2.6, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Pass 3: Protective Epistemic Perimeter Ring when `OUT_OF_CONE_GUARD` is active
      if (hasActiveSculpture && isShieldPosture) {
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.28)';
        ctx.lineWidth = 1.25;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.arc(stageAnchorX, stageAnchorY, desiredRadius * 1.10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Pass 4: Structural `k=2` Nearest-Neighbor Ridge Filaments on GPU-synthesized 3D coordinates
      if (hasActiveSculpture && morphElapsedS > 0.28) {
        const edgeFade = smoothstep(0.28, 0.95, morphElapsedS);
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, ${(0.14 * edgeFade).toFixed(3)})`;
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

      // Pass 5: Volumetric Bioluminescent Halo + Crisp Core Pinpoints (`1,024` points)
      const baseAlpha = hasActiveSculpture ? 0.88 : 0.34;
      for (let k = 0; k < NUM_POINTS; k++) {
        const i = drawOrder[k];
        const zNorm = clamp((depthZ[i] + 1.15) / 2.3, 0.0, 1.0);
        const sal = salArr[i];
        const alpha = clamp(baseAlpha * (0.28 + 0.72 * zNorm) * (0.55 + 0.45 * sal), 0.06, 0.96);
        const radius = (hasActiveSculpture ? 1.15 : 0.85) + 1.45 * zNorm * (0.55 + 0.65 * sal);

        if (sal > 0.64 && zNorm > 0.52 && hasActiveSculpture) {
          ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${(alpha * 0.24).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(screenX[i], screenY[i], radius * 2.35, 0, Math.PI * 2);
          ctx.fill();
        }

        const rMix = Math.round(pr + (245 - pr) * zNorm * 0.55);
        const gMix = Math.round(pg + (250 - pg) * zNorm * 0.55);
        const bMix = Math.round(pb + (255 - pb) * zNorm * 0.55);
        ctx.fillStyle = `rgba(${rMix}, ${gMix}, ${bMix}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(screenX[i], screenY[i], radius, 0, Math.PI * 2);
        ctx.fill();
      }

      // Pass 6: Subtle Holographic Telemetry Label beneath the GPU-synthesized 3D Sculpture
      if (hasActiveSculpture) {
        const labelY = stageAnchorY + desiredRadius * 1.14 + 12;
        const msTxt = p.pointScarMs !== null ? ` · ${p.pointScarMs.toFixed(1)} ms` : '';
        const tag =
          p.lang === 'pt'
            ? `PointScarFlow VQ4 · 1.024 pts · Fluxo OT 4-Passos${msTxt}`
            : `PointScarFlow VQ4 · 1,024 pts · 4-Step OT Flow${msTxt}`;
        ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'center';
        const metrics = ctx.measureText(tag);
        const padX = 8;
        const boxW = metrics.width + padX * 2;
        const boxH = 18;
        ctx.fillStyle = 'rgba(2, 6, 23, 0.72)';
        ctx.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, 0.35)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(stageAnchorX - boxW * 0.5, labelY - boxH * 0.68, boxW, boxH, 9);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, 0.92)`;
        ctx.fillText(tag, stageAnchorX, labelY);
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
      className="fixed inset-0 z-[48] pointer-events-none select-none"
      aria-hidden="true"
    />
  );
};
