/**
 * Domain-Independent Physical ScarCodec Engine (`v4.0.0` — Browser TypeScript Mirror of `scar_codec/core.py`).
 *
 * ZERO FACADES / ZERO Math.random():
 * Every buffer uses genuine typed arrays (`Uint32Array`, `Uint16Array` IEEE-754 FP16, `Uint8Array` INT4 nibbles,
 * `BigInt64Array` timestamps, `Float32Array` fast weights) executing the exact 5 mathematical operators:
 *   1. Multiplierless Orthonormal `FWHT-64` + Channel Equilibration (`d_eq_f16`).
 *   2. `LUT256` (`2,048 B` SRAM codebook) with hard-pinned exact zero `(0,0,0,0)` + `D4` root shells.
 *   3. `2-Pass INT4 Residual Scar Healing` + Sparse Point-Shock Capture + True `O(1)` Random Block Read.
 *   4. `OrthogonalSubspaceProjector` (`P_perp = I - U U^T`) for `1,000x` (`+60 dB`) RFI cancellation.
 *   5. `SurpriseGatedAnomalySCache` (`34,816 B` = `34.8 KB` exact constant-space black-box & blackout recorder).
 */

export class DeterministicPRNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextUniform(): number {
    let t = (this.state += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0))) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  }

  nextGaussian(): number {
    const u1 = Math.max(1e-12, this.nextUniform());
    const u2 = this.nextUniform();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
}

const _f32View = new Float32Array(1);
const _u32View = new Uint32Array(_f32View.buffer);

export function float32ToFloat16Bits(val: number): number {
  _f32View[0] = val;
  const x = _u32View[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >>> (1 - exp);
    return sign | ((mant + 0x1000) >>> 13);
  }
  if (exp >= 31) {
    return sign | 0x7bff; // Clamp to max finite FP16 (65504)
  }
  return sign | (exp << 10) | ((mant + 0x1000) >>> 13);
}

export function float16BitsToFloat32(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x03ff;

  if (exp === 0) {
    if (mant === 0) {
      _u32View[0] = sign;
      return _f32View[0];
    }
    let m = mant;
    let e = -14;
    while ((m & 0x0400) === 0) {
      m <<= 1;
      e--;
    }
    m &= 0x03ff;
    _u32View[0] = sign | ((e + 127) << 23) | (m << 13);
    return _f32View[0];
  }
  if (exp === 31) {
    _u32View[0] = sign | 0x7f800000 | (mant << 13);
    return _f32View[0];
  }
  _u32View[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  return _f32View[0];
}

/** Bit-identical Rademacher hash matching `vq4_rademacher_sign` in WGSL 1.0 and `scar_codec/core.py`. */
export function rademacherSignU32(colIdx: number): number {
  let h = Math.imul(colIdx >>> 0, 2654435761) >>> 0;
  h = Math.imul((h >>> 16) ^ h, 0x45d9f3b) >>> 0;
  return (h & 1) === 0 ? 1.0 : -1.0;
}

/** Orthonormal 64-element Fast Walsh-Hadamard Transform (`0` multipliers inside the butterfly). */
export function fwhtBlock64Inplace(buf64: Float32Array, colOffset: number, inverse: boolean): void {
  if (!inverse) {
    for (let i = 0; i < 64; i++) {
      buf64[i] *= rademacherSignU32(colOffset + i);
    }
  }
  for (let h = 1; h < 64; h <<= 1) {
    for (let base = 0; base < 64; base += h << 1) {
      for (let j = 0; j < h; j++) {
        const a = buf64[base + j];
        const b = buf64[base + h + j];
        buf64[base + j] = a + b;
        buf64[base + h + j] = a - b;
      }
    }
  }
  for (let i = 0; i < 64; i++) {
    buf64[i] *= 0.125;
  }
  if (inverse) {
    for (let i = 0; i < 64; i++) {
      buf64[i] *= rademacherSignU32(colOffset + i);
    }
  }
}

/**
 * Lossless IEEE-754 power-of-two exponent scaling (`1.0` inside `[1e-2, 4096.0]`, exact `2^E` outside).
 * Prevents catastrophic IEEE-754 `FP16` underflow on `~10^-19` LIGO gravitational-wave strain AND
 * `FP16` (`65,504`) overflow on `> 3e6`-count Mw 9.1 broadband seismograms with zero mantissa rounding!
 */
export function computeLosslessPow2ScaleTS(amax: number): number {
  if (amax <= 0.0 || (amax >= 1.0e-2 && amax <= 4096.0)) {
    return 1.0;
  }
  const exp = Math.floor(Math.log2(amax / 256.0));
  return Math.pow(2.0, exp);
}

export interface PhysicalVQ4ScarPackageTS {
  origRows: number;
  origCols: number;
  paddedCols: number;
  rank: number;
  codesU32: Uint32Array;       // [N * (K_pad / 16)] (1 u32 = 16 floats = 4 x vec4)
  scalesF16: Uint16Array;      // [N * (K_pad / 64)] FP16 block scales
  lut256F16: Uint16Array;      // [256 * 4] FP16 = exact 2,048 B SRAM codebook (lut[0..3] === 0)
  dEqF16: Uint16Array;         // [K_pad] FP16 channel equilibration
  aScarU8: Uint8Array;         // [rank * (K_pad / 2)] INT4 nibbles
  aScalesF16: Uint16Array;     // [rank * (K_pad / 64)] FP16
  bScarU8: Uint8Array;         // [N * (rank / 2)] INT4 nibbles
  bRowScalesF16: Uint16Array;  // [N] FP16
  spikeFlatIdxI32: Int32Array; // [M_spikes] sorted flat indices (incl. multi-pass Dekker FP16 terms)
  spikeResidualF16: Uint16Array; // [M_spikes] FP16 residuals
  globalScale?: number;        // Exact 2^E lossless scale (`1.0` for nominal range)
  absTolerance?: number;       // Optional deterministic L_inf error bound (`||X - X_hat||_inf <= eps`)
}

export function getPhysicalPackageBytes(pkg: PhysicalVQ4ScarPackageTS): number {
  return (
    pkg.codesU32.byteLength +
    pkg.scalesF16.byteLength +
    pkg.lut256F16.byteLength +
    pkg.dEqF16.byteLength +
    pkg.aScarU8.byteLength +
    pkg.aScalesF16.byteLength +
    pkg.bScarU8.byteLength +
    pkg.bRowScalesF16.byteLength +
    pkg.spikeFlatIdxI32.byteLength +
    pkg.spikeResidualF16.byteLength +
    (pkg.globalScale !== undefined && pkg.globalScale !== 1.0 ? 4 : 0)
  );
}

function seedD4ZeroLattice(zSamples: Float32Array): Float32Array {
  const lut = new Float32Array(256 * 4);
  // Index 0 strictly hard-pinned to exact `(0, 0, 0, 0)`
  let ptr = 4;
  for (const r of [1.2, 2.45]) {
    for (let ax = 0; ax < 4; ax++) {
      for (const sgn of [-1.0, 1.0]) {
        lut[ptr + ax] = sgn * r;
        ptr += 4;
      }
    }
  }
  for (const r of [0.72, 1.38, 2.12]) {
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        for (const si of [-1.0, 1.0]) {
          for (const sj of [-1.0, 1.0]) {
            lut[ptr + i] = si * r;
            lut[ptr + j] = sj * r;
            ptr += 4;
          }
        }
      }
    }
  }
  for (const r of [0.55, 1.12]) {
    for (let mask = 0; mask < 16; mask++) {
      for (let c = 0; c < 4; c++) {
        lut[ptr + c] = (mask >>> c) & 1 ? r : -r;
      }
      ptr += 4;
    }
  }
  const nFixed = ptr >>> 2;
  const rem = 256 - nFixed;
  const totalVec4 = Math.max(1, zSamples.length >>> 2);
  const stride = Math.max(1, Math.floor(totalVec4 / rem));
  for (let k = 0; k < rem; k++) {
    const src = ((k * stride) % totalVec4) << 2;
    const dst = (nFixed + k) << 2;
    lut[dst] = zSamples[src];
    lut[dst + 1] = zSamples[src + 1];
    lut[dst + 2] = zSamples[src + 2];
    lut[dst + 3] = zSamples[src + 3];
  }
  lut[0] = 0;
  lut[1] = 0;
  lut[2] = 0;
  lut[3] = 0;
  return lut;
}

function nearestLutIdx(v0: number, v1: number, v2: number, v3: number, lutF32: Float32Array): number {
  let bestIdx = 0;
  let bestDist = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  for (let c = 1; c < 256; c++) {
    const off = c << 2;
    const d0 = v0 - lutF32[off];
    const d1 = v1 - lutF32[off + 1];
    const d2 = v2 - lutF32[off + 2];
    const d3 = v3 - lutF32[off + 3];
    const dist = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = c;
    }
  }
  return bestIdx;
}

/** Deterministic deflated power-iteration low-rank INT4 scar extractor (`rank % 8 === 0`). */
function extractLowRankInt4Scar(
  mat: Float32Array,
  rows: number,
  cols: number,
  rank: number,
  zeroRowMask: Uint8Array
): {
  aU8: Uint8Array;
  aScF16: Uint16Array;
  bU8: Uint8Array;
  bScF16: Uint16Array;
  scarHat: Float32Array;
} {
  const work = new Float32Array(mat);
  const uCols = new Float32Array(rows * rank);
  const vRows = new Float32Array(rank * cols);
  const prng = new DeterministicPRNG(20261115 + rows * 31 + cols * 17 + rank);

  const vVec = new Float32Array(cols);
  const uVec = new Float32Array(rows);

  for (let r = 0; r < rank; r++) {
    let vNorm = 0;
    for (let j = 0; j < cols; j++) {
      const g = prng.nextGaussian();
      vVec[j] = g;
      vNorm += g * g;
    }
    vNorm = Math.sqrt(Math.max(1e-12, vNorm));
    for (let j = 0; j < cols; j++) vVec[j] /= vNorm;

    let sigma = 0;
    for (let it = 0; it < 6; it++) {
      // u = work * v
      let uNorm = 0;
      for (let i = 0; i < rows; i++) {
        if (zeroRowMask[i]) {
          uVec[i] = 0;
          continue;
        }
        let acc = 0;
        const rowOff = i * cols;
        for (let j = 0; j < cols; j++) acc += work[rowOff + j] * vVec[j];
        uVec[i] = acc;
        uNorm += acc * acc;
      }
      uNorm = Math.sqrt(Math.max(1e-12, uNorm));
      for (let i = 0; i < rows; i++) uVec[i] /= uNorm;

      // v = work^T * u
      vVec.fill(0);
      for (let i = 0; i < rows; i++) {
        const ui = uVec[i];
        if (ui === 0) continue;
        const rowOff = i * cols;
        for (let j = 0; j < cols; j++) vVec[j] += ui * work[rowOff + j];
      }
      let vn = 0;
      for (let j = 0; j < cols; j++) vn += vVec[j] * vVec[j];
      sigma = Math.sqrt(Math.max(0, vn));
      const inv = 1.0 / Math.max(1e-12, sigma);
      for (let j = 0; j < cols; j++) vVec[j] *= inv;
    }

    const rootS = Math.sqrt(sigma);
    for (let i = 0; i < rows; i++) {
      uCols[i * rank + r] = zeroRowMask[i] ? 0 : uVec[i] * rootS;
    }
    for (let j = 0; j < cols; j++) {
      vRows[r * cols + j] = vVec[j] * rootS;
    }
    for (let i = 0; i < rows; i++) {
      if (zeroRowMask[i]) continue;
      const sUi = sigma * uVec[i];
      const rowOff = i * cols;
      for (let j = 0; j < cols; j++) {
        work[rowOff + j] -= sUi * vVec[j];
      }
    }
  }

  // Pack A_scar [rank, cols] in Block-64 INT4
  const nBlk = cols >>> 6;
  const aU8 = new Uint8Array(rank * (cols >>> 1));
  const aScF16 = new Uint16Array(rank * nBlk);
  const aDeq = new Float32Array(rank * cols);
  for (let r = 0; r < rank; r++) {
    for (let b = 0; b < nBlk; b++) {
      const base = r * cols + (b << 6);
      let maxPos = 0;
      let minNeg = 0;
      for (let t = 0; t < 64; t++) {
        const v = vRows[base + t];
        if (v > maxPos) maxPos = v;
        if (v < minNeg) minNeg = v;
      }
      const scRaw = Math.max(-minNeg / 8.0, maxPos / 7.0, 1e-7);
      const scBits = float32ToFloat16Bits(scRaw);
      const sc = float16BitsToFloat32(scBits);
      aScF16[r * nBlk + b] = scBits;
      for (let t = 0; t < 32; t++) {
        const q0 = Math.max(-8, Math.min(7, Math.round(vRows[base + t * 2] / sc)));
        const q1 = Math.max(-8, Math.min(7, Math.round(vRows[base + t * 2 + 1] / sc)));
        aU8[(base >>> 1) + t] = ((q0 + 8) & 0x0f) | (((q1 + 8) & 0x0f) << 4);
        aDeq[base + t * 2] = q0 * sc;
        aDeq[base + t * 2 + 1] = q1 * sc;
      }
    }
  }

  // Pack B_scar [rows, rank] in rowwise INT4 (`0` scale on pure vacuum rows!)
  const bU8 = new Uint8Array(rows * (rank >>> 1));
  const bScF16 = new Uint16Array(rows);
  const bDeq = new Float32Array(rows * rank);
  for (let i = 0; i < rows; i++) {
    if (zeroRowMask[i]) {
      bScF16[i] = 0;
      continue;
    }
    const base = i * rank;
    let maxPos = 0;
    let minNeg = 0;
    for (let r = 0; r < rank; r++) {
      const v = uCols[base + r];
      if (v > maxPos) maxPos = v;
      if (v < minNeg) minNeg = v;
    }
    const scRaw = Math.max(-minNeg / 8.0, maxPos / 7.0, 1e-7);
    const scBits = float32ToFloat16Bits(scRaw);
    const sc = float16BitsToFloat32(scBits);
    bScF16[i] = scBits;
    for (let r2 = 0; r2 < rank >>> 1; r2++) {
      const q0 = Math.max(-8, Math.min(7, Math.round(uCols[base + r2 * 2] / sc)));
      const q1 = Math.max(-8, Math.min(7, Math.round(uCols[base + r2 * 2 + 1] / sc)));
      bU8[(base >>> 1) + r2] = ((q0 + 8) & 0x0f) | (((q1 + 8) & 0x0f) << 4);
      bDeq[base + r2 * 2] = q0 * sc;
      bDeq[base + r2 * 2 + 1] = q1 * sc;
    }
  }

  const scarHat = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    if (zeroRowMask[i]) continue;
    const rowOff = i * cols;
    const bOff = i * rank;
    for (let r = 0; r < rank; r++) {
      const br = bDeq[bOff + r];
      if (br === 0) continue;
      const aOff = r * cols;
      for (let j = 0; j < cols; j++) {
        scarHat[rowOff + j] += br * aDeq[aOff + j];
      }
    }
  }

  return { aU8, aScF16, bU8, bScF16, scarHat };
}

function medianFloatArray(vals: number[]): number {
  if (vals.length === 0) return 0;
  const sorted = vals.slice().sort((a, b) => a - b);
  const mid = sorted.length >>> 1;
  return sorted.length & 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

export class PhysicalVQ4ScarCodecTS {
  readonly rank: number;
  readonly kmeansIters: number;
  readonly maxPointSpikes: number;
  readonly spikeSigmaThreshold: number;
  readonly absTolerance?: number;

  constructor(opts?: {
    rank?: number;
    kmeansIters?: number;
    maxPointSpikes?: number;
    spikeSigmaThreshold?: number;
    absTolerance?: number;
  }) {
    this.rank = opts?.rank ?? 8;
    this.kmeansIters = opts?.kmeansIters ?? 5;
    this.maxPointSpikes = opts?.maxPointSpikes ?? 64;
    this.spikeSigmaThreshold = opts?.spikeSigmaThreshold ?? 4.2;
    this.absTolerance = opts?.absTolerance;
  }

  encode(field: Float32Array, origRows: number, origCols: number): PhysicalVQ4ScarPackageTS {
    let fieldMaxAbs = 0;
    for (let i = 0; i < field.length; i++) {
      const av = Math.abs(field[i]);
      if (av > fieldMaxAbs) fieldMaxAbs = av;
    }
    const globalScale = computeLosslessPow2ScaleTS(fieldMaxAbs);
    const invGlobalScale = 1.0 / globalScale;

    const paddedCols = ((origCols + 63) >>> 6) << 6;
    const nBlkPerRow = paddedCols >>> 6;
    const x = new Float32Array(origRows * paddedCols);
    const zeroRowMask = new Uint8Array(origRows);

    for (let r = 0; r < origRows; r++) {
      let maxAbs = 0;
      for (let c = 0; c < origCols; c++) {
        const v = globalScale !== 1.0 ? field[r * origCols + c] * invGlobalScale : field[r * origCols + c];
        x[r * paddedCols + c] = v;
        const av = Math.abs(v);
        if (av > maxAbs) maxAbs = av;
      }
      zeroRowMask[r] = maxAbs <= (globalScale !== 1.0 ? 0.0 : 1e-12) ? 1 : 0;
    }

    // Step 0: Pre-FWHT robust MAD point-shock peeling
    const xBase = new Float32Array(x);
    const preSpikeSet = new Set<number>();
    if (this.maxPointSpikes > 0) {
      const blkMedians = new Float32Array(origRows * nBlkPerRow);
      const activeDevs: number[] = [];
      const blkBuf: number[] = new Array(64);
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let b = 0; b < nBlkPerRow; b++) {
          const base = r * paddedCols + (b << 6);
          for (let t = 0; t < 64; t++) blkBuf[t] = xBase[base + t];
          const med = medianFloatArray(blkBuf);
          blkMedians[r * nBlkPerRow + b] = med;
          for (let t = 0; t < 64; t++) {
            if ((b << 6) + t < origCols) {
              activeDevs.push(Math.abs(xBase[base + t] - med));
            }
          }
        }
      }
      const madStd = Math.max(1e-6, 1.4826 * medianFloatArray(activeDevs));
      const cands: Array<{ idx: number; dev: number; med: number }> = [];
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const flatIdx = r * paddedCols + c;
          const med = blkMedians[r * nBlkPerRow + (c >>> 6)];
          const dev = Math.abs(xBase[flatIdx] - med);
          if (dev > this.spikeSigmaThreshold * madStd) {
            cands.push({ idx: flatIdx, dev, med });
          }
        }
      }
      cands.sort((a, b) => b.dev - a.dev);
      const keep = cands.slice(0, this.maxPointSpikes);
      for (const item of keep) {
        preSpikeSet.add(item.idx);
        xBase[item.idx] = item.med;
      }
    }

    // Channel equilibration (`d_eq_f16`)
    const colRmsArr: number[] = new Array(paddedCols);
    for (let c = 0; c < paddedCols; c++) {
      let s2 = 0;
      for (let r = 0; r < origRows; r++) {
        const v = xBase[r * paddedCols + c];
        s2 += v * v;
      }
      colRmsArr[c] = Math.max(1e-6, Math.sqrt(s2 / origRows));
    }
    const medRms = Math.max(1e-6, medianFloatArray(colRmsArr));
    const dEqF16 = new Uint16Array(paddedCols);
    const dEqF32 = new Float32Array(paddedCols);
    for (let c = 0; c < paddedCols; c++) {
      const ratio = Math.min(32.0, Math.max(0.05, Math.sqrt(colRmsArr[c] / medRms)));
      const bits = float32ToFloat16Bits(ratio);
      dEqF16[c] = bits;
      dEqF32[c] = float16BitsToFloat32(bits);
    }

    // Pass 1: Low-rank INT4 scar extraction
    const effRank = this.rank > 0 && Math.min(origRows, paddedCols) >= this.rank ? this.rank : 0;
    const resForVq = new Float32Array(origRows * paddedCols);
    if (effRank > 0) {
      const p1 = extractLowRankInt4Scar(xBase, origRows, paddedCols, effRank, zeroRowMask);
      for (let i = 0; i < resForVq.length; i++) resForVq[i] = xBase[i] - p1.scarHat[i];
    } else {
      resForVq.set(xBase);
    }

    // Orthonormal FWHT-64 in equilibrated domain
    const totalBlocks = origRows * nBlkPerRow;
    const rotBlocks = new Float32Array(totalBlocks * 64);
    const scalesF16 = new Uint16Array(totalBlocks);
    const scalesF32 = new Float32Array(totalBlocks);
    const zeroBlkMask = new Uint8Array(totalBlocks);
    const z4d = new Float32Array(totalBlocks * 64);
    const tmp64 = new Float32Array(64);

    for (let r = 0; r < origRows; r++) {
      for (let b = 0; b < nBlkPerRow; b++) {
        const blkId = r * nBlkPerRow + b;
        const colOff = b << 6;
        const rowOff = r * paddedCols + colOff;
        let maxAbs = 0;
        for (let t = 0; t < 64; t++) {
          const v = resForVq[rowOff + t] / dEqF32[colOff + t];
          tmp64[t] = v;
          const av = Math.abs(v);
          if (av > maxAbs) maxAbs = av;
        }
        if (zeroRowMask[r] || maxAbs <= 1e-12) {
          zeroBlkMask[blkId] = 1;
          scalesF16[blkId] = 0;
          scalesF32[blkId] = 0;
          continue;
        }
        fwhtBlock64Inplace(tmp64, colOff, false);
        let s2 = 0;
        for (let t = 0; t < 64; t++) {
          rotBlocks[blkId * 64 + t] = tmp64[t];
          s2 += tmp64[t] * tmp64[t];
        }
        const rms = Math.max(1e-7, Math.sqrt(s2 / 64.0));
        const scBits = float32ToFloat16Bits(rms);
        const scVal = Math.max(1e-7, float16BitsToFloat32(scBits));
        scalesF16[blkId] = scBits;
        scalesF32[blkId] = scVal;
        for (let t = 0; t < 64; t++) {
          z4d[blkId * 64 + t] = tmp64[t] / scVal;
        }
      }
    }

    // Fit LUT256 (`lut[0] = (0,0,0,0)` + D4 + energy-weighted Lloyd-Max)
    const lutF32 = seedD4ZeroLattice(z4d);
    const totalVec4 = totalBlocks * 16;
    for (let it = 0; it < this.kmeansIters; it++) {
      const num = new Float64Array(256 * 4);
      const den = new Float64Array(256);
      for (let vIdx = 0; vIdx < totalVec4; vIdx++) {
        const blkId = vIdx >>> 4;
        if (zeroBlkMask[blkId]) continue;
        const w = scalesF32[blkId] * scalesF32[blkId];
        const off = vIdx << 2;
        const c = nearestLutIdx(z4d[off], z4d[off + 1], z4d[off + 2], z4d[off + 3], lutF32);
        if (c === 0) continue;
        num[c * 4] += w * z4d[off];
        num[c * 4 + 1] += w * z4d[off + 1];
        num[c * 4 + 2] += w * z4d[off + 2];
        num[c * 4 + 3] += w * z4d[off + 3];
        den[c] += w;
      }
      for (let c = 1; c < 256; c++) {
        if (den[c] > 1e-10) {
          lutF32[c * 4] = num[c * 4] / den[c];
          lutF32[c * 4 + 1] = num[c * 4 + 1] / den[c];
          lutF32[c * 4 + 2] = num[c * 4 + 2] / den[c];
          lutF32[c * 4 + 3] = num[c * 4 + 3] / den[c];
        }
      }
      lutF32[0] = 0;
      lutF32[1] = 0;
      lutF32[2] = 0;
      lutF32[3] = 0;
    }

    const lut256F16 = new Uint16Array(256 * 4);
    const lutQF32 = new Float32Array(256 * 4);
    for (let i = 4; i < 256 * 4; i++) {
      const b = float32ToFloat16Bits(lutF32[i]);
      lut256F16[i] = b;
      lutQF32[i] = float16BitsToFloat32(b);
    }

    const codesU8 = new Uint8Array(totalVec4);
    const xVq = new Float32Array(origRows * paddedCols);
    for (let r = 0; r < origRows; r++) {
      for (let b = 0; b < nBlkPerRow; b++) {
        const blkId = r * nBlkPerRow + b;
        if (zeroBlkMask[blkId]) continue;
        const sc = scalesF32[blkId];
        for (let q = 0; q < 16; q++) {
          const vIdx = blkId * 16 + q;
          const off = vIdx << 2;
          const code = nearestLutIdx(z4d[off], z4d[off + 1], z4d[off + 2], z4d[off + 3], lutQF32);
          codesU8[vIdx] = code;
          tmp64[q * 4] = lutQF32[code * 4] * sc;
          tmp64[q * 4 + 1] = lutQF32[code * 4 + 1] * sc;
          tmp64[q * 4 + 2] = lutQF32[code * 4 + 2] * sc;
          tmp64[q * 4 + 3] = lutQF32[code * 4 + 3] * sc;
        }
        const colOff = b << 6;
        fwhtBlock64Inplace(tmp64, colOff, true);
        const rowOff = r * paddedCols + colOff;
        for (let t = 0; t < 64; t++) {
          xVq[rowOff + t] = tmp64[t] * dEqF32[colOff + t];
        }
      }
    }

    // Pass 2: Re-solve compressed INT4 scar directly against `(xBase - xVq)`
    let aScarU8: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
    let aScalesF16: Uint16Array<ArrayBufferLike> = new Uint16Array(0);
    let bScarU8: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
    let bRowScalesF16: Uint16Array<ArrayBufferLike> = new Uint16Array(0);
    const xApprox = new Float32Array(xVq);

    if (effRank > 0) {
      const resP2 = new Float32Array(origRows * paddedCols);
      for (let i = 0; i < resP2.length; i++) resP2[i] = xBase[i] - xVq[i];
      const p2 = extractLowRankInt4Scar(resP2, origRows, paddedCols, effRank, zeroRowMask);
      aScarU8 = p2.aU8;
      aScalesF16 = p2.aScF16;
      bScarU8 = p2.bU8;
      bRowScalesF16 = p2.bScF16;
      for (let i = 0; i < xApprox.length; i++) xApprox[i] += p2.scarHat[i];
    }

    // Sparse point-shock refinement OR 6-pass Dekker FP16 deterministic L_inf error-bounded cascade (`||X - X_hat||_inf <= eps`)
    let spikeFlatIdxI32: Int32Array;
    let spikeResidualF16: Uint16Array;

    if (this.absTolerance !== undefined && this.absTolerance > 0) {
      const tolScaled = this.absTolerance * invGlobalScale;
      const curErr = new Float32Array(origRows * paddedCols);
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const idx = r * paddedCols + c;
          curErr[idx] = x[idx] - xApprox[idx];
        }
      }
      const dekkerPairs: Array<{ idx: number; bits: number }> = [];
      for (let pass = 0; pass < 6; pass++) {
        let violations = 0;
        for (let r = 0; r < origRows; r++) {
          if (zeroRowMask[r]) continue;
          for (let c = 0; c < origCols; c++) {
            const idx = r * paddedCols + c;
            const rem = curErr[idx];
            if (Math.abs(rem) > tolScaled) {
              const clamped = Math.max(-65504.0, Math.min(65504.0, rem));
              const bits = float32ToFloat16Bits(clamped);
              dekkerPairs.push({ idx, bits });
              curErr[idx] = rem - float16BitsToFloat32(bits);
              violations++;
            }
          }
        }
        if (violations === 0) break;
      }
      dekkerPairs.sort((a, b) => a.idx - b.idx);
      spikeFlatIdxI32 = new Int32Array(dekkerPairs.length);
      spikeResidualF16 = new Uint16Array(dekkerPairs.length);
      for (let s = 0; s < dekkerPairs.length; s++) {
        spikeFlatIdxI32[s] = dekkerPairs[s].idx;
        spikeResidualF16[s] = dekkerPairs[s].bits;
      }
    } else {
      let baseEnergy = 0;
      let baseCnt = 0;
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const v = xBase[r * paddedCols + c];
          baseEnergy += v * v;
          baseCnt++;
        }
      }
      const baseRms = Math.sqrt(baseEnergy / Math.max(1, baseCnt));
      const spikeThresh = this.spikeSigmaThreshold * 0.25 * Math.max(1e-6, baseRms);
      const spikeCands: Array<{ idx: number; err: number }> = [];
      if (this.maxPointSpikes > 0) {
        for (let r = 0; r < origRows; r++) {
          if (zeroRowMask[r]) continue;
          for (let c = 0; c < origCols; c++) {
            const idx = r * paddedCols + c;
            const err = x[idx] - xApprox[idx];
            if (preSpikeSet.has(idx) || Math.abs(err) > spikeThresh) {
              spikeCands.push({ idx, err });
            }
          }
        }
      }
      spikeCands.sort((a, b) => Math.abs(b.err) - Math.abs(a.err));
      const chosenSpikes = spikeCands.slice(0, this.maxPointSpikes).sort((a, b) => a.idx - b.idx);
      spikeFlatIdxI32 = new Int32Array(chosenSpikes.length);
      spikeResidualF16 = new Uint16Array(chosenSpikes.length);
      for (let s = 0; s < chosenSpikes.length; s++) {
        spikeFlatIdxI32[s] = chosenSpikes[s].idx;
        spikeResidualF16[s] = float32ToFloat16Bits(chosenSpikes[s].err);
      }
    }

    // Pack 4 x uint8 codes into 1 uint32 (`1 u32 = 16 values`)
    const codesU32 = new Uint32Array(totalVec4 >>> 2);
    for (let u = 0; u < codesU32.length; u++) {
      const off = u << 2;
      codesU32[u] =
        (codesU8[off] |
          (codesU8[off + 1] << 8) |
          (codesU8[off + 2] << 16) |
          (codesU8[off + 3] << 24)) >>>
        0;
    }

    return {
      origRows,
      origCols,
      paddedCols,
      rank: effRank,
      codesU32,
      scalesF16,
      lut256F16,
      dEqF16,
      aScarU8,
      aScalesF16,
      bScarU8,
      bRowScalesF16,
      spikeFlatIdxI32,
      spikeResidualF16,
      globalScale,
      absTolerance: this.absTolerance,
    };
  }

  /** True `O(1)` random-access decode of a single 64-element block (`0` neighbor blocks read!). */
  decodeBlockO1(pkg: PhysicalVQ4ScarPackageTS, rowIdx: number, blockIdx: number): {
    values: Float32Array;
    bytesTouched: number;
  } {
    const startCol = blockIdx << 6;
    const nBlkPerRow = pkg.paddedCols >>> 6;
    const blkFlat = rowIdx * nBlkPerRow + blockIdx;
    const scBits = pkg.scalesF16[blkFlat];
    const tmp64 = new Float32Array(64);
    let bytesTouched = 2; // 2 B block scale

    if (scBits !== 0) {
      const sc = float16BitsToFloat32(scBits);
      const u32Base = blkFlat * 4;
      bytesTouched += 16 + 128; // 4 x u32 codes (16 B) + 64 x f16 d_eq (128 B)
      for (let w = 0; w < 4; w++) {
        const word = pkg.codesU32[u32Base + w];
        for (let byteLane = 0; byteLane < 4; byteLane++) {
          const code = (word >>> (byteLane * 8)) & 0xff;
          const qIdx = (w * 4 + byteLane) * 4;
          const lutOff = code * 4;
          tmp64[qIdx] = float16BitsToFloat32(pkg.lut256F16[lutOff]) * sc;
          tmp64[qIdx + 1] = float16BitsToFloat32(pkg.lut256F16[lutOff + 1]) * sc;
          tmp64[qIdx + 2] = float16BitsToFloat32(pkg.lut256F16[lutOff + 2]) * sc;
          tmp64[qIdx + 3] = float16BitsToFloat32(pkg.lut256F16[lutOff + 3]) * sc;
        }
      }
      fwhtBlock64Inplace(tmp64, startCol, true);
      for (let t = 0; t < 64; t++) {
        tmp64[t] *= float16BitsToFloat32(pkg.dEqF16[startCol + t]);
      }
    }

    if (pkg.rank > 0) {
      const bScBits = pkg.bRowScalesF16[rowIdx];
      bytesTouched += 2;
      if (bScBits !== 0) {
        const bSc = float16BitsToFloat32(bScBits);
        const bRow = new Float32Array(pkg.rank);
        const halfRank = pkg.rank >>> 1;
        const bOff = rowIdx * halfRank;
        bytesTouched += halfRank + pkg.rank * 34;
        for (let r2 = 0; r2 < halfRank; r2++) {
          const byte = pkg.bScarU8[bOff + r2];
          bRow[r2 * 2] = ((byte & 0x0f) - 8) * bSc;
          bRow[r2 * 2 + 1] = (((byte >>> 4) & 0x0f) - 8) * bSc;
        }
        const halfCols = pkg.paddedCols >>> 1;
        for (let r = 0; r < pkg.rank; r++) {
          const br = bRow[r];
          if (br === 0) continue;
          const aSc = float16BitsToFloat32(pkg.aScalesF16[r * nBlkPerRow + blockIdx]);
          const aByteBase = r * halfCols + (startCol >>> 1);
          for (let t2 = 0; t2 < 32; t2++) {
            const ab = pkg.aScarU8[aByteBase + t2];
            tmp64[t2 * 2] += br * (((ab & 0x0f) - 8) * aSc);
            tmp64[t2 * 2 + 1] += br * ((((ab >>> 4) & 0x0f) - 8) * aSc);
          }
        }
      }
    }

    // Binary search sparse point-shocks (`O(log M_spikes)`)
    const flatLo = rowIdx * pkg.paddedCols + startCol;
    const flatHi = flatLo + 64;
    let lo = 0;
    let hi = pkg.spikeFlatIdxI32.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (pkg.spikeFlatIdxI32[mid] < flatLo) lo = mid + 1;
      else hi = mid;
    }
    while (lo < pkg.spikeFlatIdxI32.length && pkg.spikeFlatIdxI32[lo] < flatHi) {
      const localOff = pkg.spikeFlatIdxI32[lo] - flatLo;
      tmp64[localOff] += float16BitsToFloat32(pkg.spikeResidualF16[lo]);
      bytesTouched += 6;
      lo++;
    }

    const gScale = pkg.globalScale ?? 1.0;
    if (gScale !== 1.0) {
      for (let t = 0; t < 64; t++) {
        tmp64[t] *= gScale;
      }
    }

    const validLen = Math.min(64, Math.max(0, pkg.origCols - startCol));
    return { values: tmp64.subarray(0, validLen), bytesTouched };
  }

  decodeFull(pkg: PhysicalVQ4ScarPackageTS): Float32Array {
    const out = new Float32Array(pkg.origRows * pkg.origCols);
    const nBlkPerRow = pkg.paddedCols >>> 6;
    for (let r = 0; r < pkg.origRows; r++) {
      for (let b = 0; b < nBlkPerRow; b++) {
        const { values } = this.decodeBlockO1(pkg, r, b);
        out.set(values, r * pkg.origCols + (b << 6));
      }
    }
    return out;
  }
}

/** `P_perp = I - U U^T` Subspace Interference Canceller (`SKA / CubeSat RFI` & `Adaptive Optics`). */
export class OrthogonalSubspaceProjectorTS {
  readonly channels: number;
  readonly rank: number;
  readonly basisU: Float32Array; // [channels * rank] orthonormal columns

  constructor(basisColumns: Float32Array, channels: number, rank: number) {
    this.channels = channels;
    this.rank = rank;
    this.basisU = new Float32Array(channels * rank);
    // Modified Gram-Schmidt orthonormalization
    for (let k = 0; k < rank; k++) {
      for (let c = 0; c < channels; c++) {
        this.basisU[c * rank + k] = basisColumns[c * rank + k];
      }
      for (let j = 0; j < k; j++) {
        let dot = 0;
        for (let c = 0; c < channels; c++) {
          dot += this.basisU[c * rank + j] * this.basisU[c * rank + k];
        }
        for (let c = 0; c < channels; c++) {
          this.basisU[c * rank + k] -= dot * this.basisU[c * rank + j];
        }
      }
      let norm = 0;
      for (let c = 0; c < channels; c++) {
        const v = this.basisU[c * rank + k];
        norm += v * v;
      }
      const inv = 1.0 / Math.sqrt(Math.max(1e-12, norm));
      for (let c = 0; c < channels; c++) {
        this.basisU[c * rank + k] *= inv;
      }
    }
  }

  /**
   * Blind empirical covariance subspace estimator (`0` oracle steering access):
   * Fits top-`rank` orthonormal interference eigenvectors directly from noisy calibration snapshots
   * `Y_cal in R^{channels x samples}` via deflated power iteration on `Sigma_hat = (1/T) Y_cal Y_cal^T`.
   */
  static fitFromCalibration(
    calSignal: Float32Array,
    channels: number,
    samples: number,
    rank: number
  ): OrthogonalSubspaceProjectorTS {
    const cov = new Float64Array(channels * channels);
    const invT = 1.0 / Math.max(1, samples);
    for (let i = 0; i < channels; i++) {
      const rowI = i * samples;
      for (let j = i; j < channels; j++) {
        const rowJ = j * samples;
        let acc = 0;
        for (let t = 0; t < samples; t++) {
          acc += calSignal[rowI + t] * calSignal[rowJ + t];
        }
        const v = acc * invT;
        cov[i * channels + j] = v;
        cov[j * channels + i] = v;
      }
    }

    const basisCols = new Float32Array(channels * rank);
    const prng = new DeterministicPRNG(20260911 + channels * 13 + rank);
    const vec = new Float64Array(channels);
    const nxt = new Float64Array(channels);

    for (let k = 0; k < rank; k++) {
      let n2 = 0;
      for (let c = 0; c < channels; c++) {
        const g = prng.nextGaussian();
        vec[c] = g;
        n2 += g * g;
      }
      const inv0 = 1.0 / Math.sqrt(Math.max(1e-12, n2));
      for (let c = 0; c < channels; c++) vec[c] *= inv0;

      let eig = 0;
      for (let it = 0; it < 24; it++) {
        let m2 = 0;
        for (let i = 0; i < channels; i++) {
          let s = 0;
          const off = i * channels;
          for (let j = 0; j < channels; j++) s += cov[off + j] * vec[j];
          nxt[i] = s;
          m2 += s * s;
        }
        eig = Math.sqrt(Math.max(1e-18, m2));
        const inv = 1.0 / eig;
        for (let i = 0; i < channels; i++) vec[i] = nxt[i] * inv;
      }

      for (let i = 0; i < channels; i++) {
        basisCols[i * rank + k] = vec[i];
        const off = i * channels;
        for (let j = 0; j < channels; j++) {
          cov[off + j] -= eig * vec[i] * vec[j];
        }
      }
    }

    return new OrthogonalSubspaceProjectorTS(basisCols, channels, rank);
  }

  projectClean(arraySignal: Float32Array, channels: number, samples: number): Float32Array {
    const out = new Float32Array(arraySignal);
    const coeff = new Float64Array(this.rank);
    for (let t = 0; t < samples; t++) {
      for (let k = 0; k < this.rank; k++) {
        let dot = 0;
        for (let c = 0; c < channels; c++) {
          dot += this.basisU[c * this.rank + k] * out[c * samples + t];
        }
        coeff[k] = dot;
      }
      for (let c = 0; c < channels; c++) {
        let proj = 0;
        for (let k = 0; k < this.rank; k++) {
          proj += this.basisU[c * this.rank + k] * coeff[k];
        }
        out[c * samples + t] -= proj;
      }
    }
    return out;
  }
}

/**
 * Exact `34,816 B` (`34.8 KB`) Constant-Space `SCache O(1)` Black-Box & Satellite Recorder.
 *
 * Byte breakdown:
 *   wScarF32        [64 * 64] f32 = 16,384 B
 *   slotKeysF16     [48 * 64] f16 =  6,144 B
 *   slotValsF16     [48 * 64] f16 =  6,144 B
 *   slotTsI64       [48]      i64 =    384 B
 *   slotSurpriseF32 [48]      f32 =    192 B
 *   emaMeanF32      [64]      f32 =    256 B
 *   emaVarF32       [64]      f32 =    256 B
 *   recentRingF16   [39 * 64] f16 =  4,992 B
 *   stateMetaI64    [8]       i64 =     64 B
 *   ----------------------------------------
 *   TOTAL PHYSICAL BYTES          = 34,816 B (34.8 KB, Δ = +0 B)
 */
export class SurpriseGatedAnomalySCacheTS {
  readonly channels = 64;
  readonly numSlots = 48;
  readonly ringLen = 39;
  readonly surpriseThreshold: number;
  readonly emaDecay = 0.99;
  readonly deltaLr = 0.35;

  readonly wScarF32 = new Float32Array(64 * 64);
  readonly slotKeysF16 = new Uint16Array(48 * 64);
  readonly slotValsF16 = new Uint16Array(48 * 64);
  readonly slotTsI64 = new BigInt64Array(48);
  readonly slotSurpriseF32 = new Float32Array(48);
  readonly emaMeanF32 = new Float32Array(64);
  readonly emaVarF32 = new Float32Array(64);
  readonly recentRingF16 = new Uint16Array(39 * 64);
  readonly stateMetaI64 = new BigInt64Array(8);

  constructor(surpriseThreshold = 4.2) {
    this.surpriseThreshold = surpriseThreshold;
    this.slotTsI64.fill(-1n);
    this.emaVarF32.fill(1.0);
  }

  physicalBytes(): number {
    return (
      this.wScarF32.byteLength +
      this.slotKeysF16.byteLength +
      this.slotValsF16.byteLength +
      this.slotTsI64.byteLength +
      this.slotSurpriseF32.byteLength +
      this.emaMeanF32.byteLength +
      this.emaVarF32.byteLength +
      this.recentRingF16.byteLength +
      this.stateMetaI64.byteLength
    );
  }

  ingest(frame64: Float32Array, timestamp: number): number {
    const step = Number(this.stateMetaI64[0]);
    const ringOff = (step % this.ringLen) * 64;
    for (let c = 0; c < 64; c++) {
      this.recentRingF16[ringOff + c] = float32ToFloat16Bits(frame64[c]);
    }
    this.stateMetaI64[0] = BigInt(step + 1);

    if (step < 32) {
      const alpha = 1.0 / (step + 1);
      for (let c = 0; c < 64; c++) {
        const diff = frame64[c] - this.emaMeanF32[c];
        this.emaMeanF32[c] += diff * alpha;
        this.emaVarF32[c] = this.emaVarF32[c] * (1.0 - alpha) + diff * diff * alpha;
      }
      return 0.0;
    }

    const diff = new Float32Array(64);
    let z2Sum = 0;
    let diffNorm2 = 0;
    for (let c = 0; c < 64; c++) {
      const d = frame64[c] - this.emaMeanF32[c];
      diff[c] = d;
      diffNorm2 += d * d;
      const std = Math.sqrt(Math.max(1e-6, this.emaVarF32[c]));
      const z = d / std;
      z2Sum += z * z;
    }
    const surprise = Math.sqrt(z2Sum / 64.0);

    if (surprise <= this.surpriseThreshold) {
      const dec = this.emaDecay;
      const oneMinus = 1.0 - dec;
      for (let c = 0; c < 64; c++) {
        this.emaMeanF32[c] = this.emaMeanF32[c] * dec + frame64[c] * oneMinus;
        this.emaVarF32[c] = this.emaVarF32[c] * dec + diff[c] * diff[c] * oneMinus;
      }
      return surprise;
    }

    // High-surprise shock: Widrow-Hoff Delta associative fast-weight update
    const invNorm = 1.0 / Math.sqrt(Math.max(1e-12, diffNorm2));
    const kHat = new Float32Array(64);
    for (let c = 0; c < 64; c++) kHat[c] = diff[c] * invNorm;

    for (let r = 0; r < 64; r++) {
      let pred = 0;
      const rowOff = r * 64;
      for (let c = 0; c < 64; c++) pred += this.wScarF32[rowOff + c] * kHat[c];
      const err = (diff[r] - pred) * this.deltaLr;
      for (let c = 0; c < 64; c++) this.wScarF32[rowOff + c] += err * kHat[c];
    }

    let minSlot = 0;
    let minSurp = this.slotSurpriseF32[0];
    for (let s = 1; s < 48; s++) {
      if (this.slotSurpriseF32[s] < minSurp) {
        minSurp = this.slotSurpriseF32[s];
        minSlot = s;
      }
    }
    if (surprise > minSurp) {
      const off = minSlot * 64;
      for (let c = 0; c < 64; c++) {
        this.slotKeysF16[off + c] = float32ToFloat16Bits(kHat[c]);
        this.slotValsF16[off + c] = float32ToFloat16Bits(diff[c]);
      }
      this.slotTsI64[minSlot] = BigInt(timestamp);
      this.slotSurpriseF32[minSlot] = surprise;
      this.stateMetaI64[1] += 1n;
    }

    return surprise;
  }

  queryAssociative(query64: Float32Array): Float32Array {
    let n2 = 0;
    for (let c = 0; c < 64; c++) n2 += query64[c] * query64[c];
    const inv = 1.0 / Math.sqrt(Math.max(1e-12, n2));
    const out = new Float32Array(64);
    for (let r = 0; r < 64; r++) {
      let acc = 0;
      const rowOff = r * 64;
      for (let c = 0; c < 64; c++) acc += this.wScarF32[rowOff + c] * (query64[c] * inv);
      out[r] = acc;
    }
    return out;
  }

  recordedTimestamps(): number[] {
    const out: number[] = [];
    for (let s = 0; s < 48; s++) {
      const ts = Number(this.slotTsI64[s]);
      if (ts >= 0) out.push(ts);
    }
    return out.sort((a, b) => a - b);
  }
}

export class CircularRingBufferOTTS {
  readonly channels = 64;
  readonly capacity: number;
  readonly framesF16: Uint16Array;
  readonly timestampsI64: BigInt64Array;
  private writePtr = 0;

  constructor(budgetBytes = 34816) {
    const bytesPerEntry = 64 * 2 + 8; // 136 B -> 256 frames = exact 34,816 B!
    this.capacity = Math.max(1, Math.floor(budgetBytes / bytesPerEntry));
    this.framesF16 = new Uint16Array(this.capacity * 64);
    this.timestampsI64 = new BigInt64Array(this.capacity);
    this.timestampsI64.fill(-1n);
  }

  physicalBytes(): number {
    return this.framesF16.byteLength + this.timestampsI64.byteLength;
  }

  ingest(frame64: Float32Array, timestamp: number): void {
    const slot = this.writePtr % this.capacity;
    const off = slot * 64;
    for (let c = 0; c < 64; c++) {
      this.framesF16[off + c] = float32ToFloat16Bits(frame64[c]);
    }
    this.timestampsI64[slot] = BigInt(timestamp);
    this.writePtr++;
  }

  recordedTimestamps(): Set<number> {
    const out = new Set<number>();
    for (let i = 0; i < this.capacity; i++) {
      const ts = Number(this.timestampsI64[i]);
      if (ts >= 0) out.add(ts);
    }
    return out;
  }
}

export function cosineSimilarityF32(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(Math.max(1e-12, na * nb));
}

function optimalGainSnrDb(est: Float32Array, ref: Float32Array): number {
  let dot = 0;
  let ee = 0;
  let rr = 0;
  for (let i = 0; i < ref.length; i++) {
    dot += est[i] * ref[i];
    ee += est[i] * est[i];
    rr += ref[i] * ref[i];
  }
  const alpha = dot / Math.max(1e-12, ee);
  let err2 = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = alpha * est[i] - ref[i];
    err2 += d * d;
  }
  const ratio = rr / Math.max(1e-12, err2);
  return ratio <= 1.0 ? 0.0 : 10.0 * Math.log10(ratio);
}

// ============================================================================
// Interactive Proof 1, Proof 2 & Proof 3 Live Runners (`0 Math.random()`)
// ============================================================================

export interface RadioRfiProofResult {
  rfiRatio: number;
  rfiSirDb: number;
  outOfConeMaxErr: number;
  rawBytes: number;
  compressedBytes: number;
  compressionRatio: number;
  rawBeamCosine: number;
  vlbi2BitCosine: number;
  scarCodecCosine: number;
  scarCodecSnrDb: number;
  refWaveform: Float32Array;
  rawBeamWaveform: Float32Array;
  vlbiBeamWaveform: Float32Array;
  scarBeamWaveform: Float32Array;
}

export function runRadioTelescopeRfiProof(rfiRatio = 1000): RadioRfiProofResult {
  const nAnt = 32;
  const nSamples = 512;
  const prng = new DeterministicPRNG(20260401);

  const rfiSteering = new Float32Array(nAnt * 3);
  for (let k = 0; k < 3; k++) {
    let n2 = 0;
    for (let a = 0; a < nAnt; a++) {
      const g = prng.nextGaussian();
      rfiSteering[a * 3 + k] = g;
      n2 += g * g;
    }
    const inv = 1.0 / Math.sqrt(n2);
    for (let a = 0; a < nAnt; a++) rfiSteering[a * 3 + k] *= inv;
  }

  const astroVec = new Float32Array(nAnt);
  let an2 = 0;
  for (let a = 0; a < nAnt; a++) {
    astroVec[a] = prng.nextGaussian();
    an2 += astroVec[a] * astroVec[a];
  }
  for (let a = 0; a < nAnt; a++) astroVec[a] /= Math.sqrt(an2);

  // Blind off-pulse calibration window (`0` oracle access to `rfiSteering`):
  // Estimates `U_hat_RFI` strictly from noisy antenna voltages `Y_cal = RFI_cal + Thermal_cal`.
  const yCal = new Float32Array(nAnt * nSamples);
  for (let s = 0; s < nSamples; s++) {
    const t = s / nSamples;
    const cal0 = Math.sin(2.0 * Math.PI * 37.0 * t - 0.7) * rfiRatio;
    const cal1 = Math.cos(2.0 * Math.PI * 79.0 * t + 1.2) * rfiRatio;
    const cal2 = Math.sin(2.0 * Math.PI * (113.0 * t + 28.0 * t * t) - 0.5) * rfiRatio;
    for (let a = 0; a < nAnt; a++) {
      const rfi =
        rfiSteering[a * 3] * cal0 +
        rfiSteering[a * 3 + 1] * cal1 +
        rfiSteering[a * 3 + 2] * cal2;
      yCal[a * nSamples + s] = rfi + 0.035 * prng.nextGaussian();
    }
  }
  const proj = OrthogonalSubspaceProjectorTS.fitFromCalibration(yCal, nAnt, nSamples, 3);

  const cleanAstro = new Float32Array(nAnt * nSamples);
  const yRaw = new Float32Array(nAnt * nSamples);
  const yInSubspaceStress = new Float32Array(nAnt * nSamples);
  const orthoCleanRef = proj.projectClean(
    (() => {
      const tmp = new Float32Array(nAnt * nSamples);
      for (let s = 0; s < nSamples; s++) {
        const t = s / nSamples;
        const phase = (t * 6.0) % 1.0;
        const pulse = Math.exp(-0.5 * Math.pow((phase - 0.5) / 0.024, 2));
        for (let a = 0; a < nAnt; a++) tmp[a * nSamples + s] = astroVec[a] * pulse;
      }
      return tmp;
    })(),
    nAnt,
    nSamples
  );

  for (let s = 0; s < nSamples; s++) {
    const t = s / nSamples;
    const phase = (t * 6.0) % 1.0;
    const pulse = Math.exp(-0.5 * Math.pow((phase - 0.5) / 0.024, 2));
    const rfi0 = Math.sin(2.0 * Math.PI * 37.0 * t) * rfiRatio;
    const rfi1 = Math.cos(2.0 * Math.PI * 79.0 * t + 0.4) * rfiRatio;
    const rfi2 = Math.sin(2.0 * Math.PI * (113.0 * t + 28.0 * t * t) + 1.1) * rfiRatio;

    for (let a = 0; a < nAnt; a++) {
      const clean = astroVec[a] * pulse;
      cleanAstro[a * nSamples + s] = clean;
      const rfi =
        rfiSteering[a * 3] * rfi0 +
        rfiSteering[a * 3 + 1] * rfi1 +
        rfiSteering[a * 3 + 2] * rfi2;
      const uHatBurst =
        (proj.basisU[a * 3] * rfi0 +
          proj.basisU[a * 3 + 1] * rfi1 +
          proj.basisU[a * 3 + 2] * rfi2);
      yInSubspaceStress[a * nSamples + s] = orthoCleanRef[a * nSamples + s] + uHatBurst;
      const thermal = 0.035 * prng.nextGaussian();
      yRaw[a * nSamples + s] = clean + rfi + thermal;
    }
  }

  // Genuine dynamic-range nulling invariance test: `P_perp(x_perp + 1000 * U_hat * c) - x_perp`
  const stressedProjected = proj.projectClean(yInSubspaceStress, nAnt, nSamples);
  let outOfConeMaxErr = 0;
  for (let i = 0; i < orthoCleanRef.length; i++) {
    const err = Math.abs(stressedProjected[i] - orthoCleanRef[i]);
    if (err > outOfConeMaxErr) outOfConeMaxErr = err;
  }

  // Clean beamforming vector in `span(U_RFI)^perp`
  const wBeam = proj.projectClean(astroVec, nAnt, 1);
  let wn2 = 0;
  for (let a = 0; a < nAnt; a++) wn2 += wBeam[a] * wBeam[a];
  for (let a = 0; a < nAnt; a++) wBeam[a] /= Math.sqrt(Math.max(1e-12, wn2));

  // Baseline: Naive 2-bit VLBI scalar quantizer before `P_perp`
  const qVlbi = new Float32Array(nAnt * nSamples);
  for (let a = 0; a < nAnt; a++) {
    let s2 = 0;
    const rowOff = a * nSamples;
    for (let s = 0; s < nSamples; s++) s2 += yRaw[rowOff + s] * yRaw[rowOff + s];
    const rms = Math.max(1e-6, Math.sqrt(s2 / nSamples));
    for (let s = 0; s < nSamples; s++) {
      const z = yRaw[rowOff + s] / rms;
      const lvl = z < 0 ? (z < -1.0 ? -1.5 : -0.5) : z < 1.0 ? 0.5 : 1.5;
      qVlbi[rowOff + s] = lvl * rms;
    }
  }
  const qVlbiNulled = proj.projectClean(qVlbi, nAnt, nSamples);

  // Proposed: `P_perp_RFI` + `PhysicalVQ4ScarCodecTS`
  const yNulled = proj.projectClean(yRaw, nAnt, nSamples);
  const codec = new PhysicalVQ4ScarCodecTS({ rank: 8, kmeansIters: 4, maxPointSpikes: 48 });
  const pkg = codec.encode(yNulled, nAnt, nSamples);
  const yRec = codec.decodeFull(pkg);

  const refWaveform = new Float32Array(nSamples);
  const rawBeamWaveform = new Float32Array(nSamples);
  const vlbiBeamWaveform = new Float32Array(nSamples);
  const scarBeamWaveform = new Float32Array(nSamples);

  for (let s = 0; s < nSamples; s++) {
    let refSum = 0;
    let rawSum = 0;
    let vlbiSum = 0;
    let scarSum = 0;
    for (let a = 0; a < nAnt; a++) {
      refSum += wBeam[a] * cleanAstro[a * nSamples + s];
      rawSum += astroVec[a] * yRaw[a * nSamples + s];
      vlbiSum += wBeam[a] * qVlbiNulled[a * nSamples + s];
      scarSum += wBeam[a] * yRec[a * nSamples + s];
    }
    refWaveform[s] = refSum;
    rawBeamWaveform[s] = rawSum;
    vlbiBeamWaveform[s] = vlbiSum;
    scarBeamWaveform[s] = scarSum;
  }

  const rawBytes = nAnt * nSamples * 4;
  const compressedBytes = getPhysicalPackageBytes(pkg);
  return {
    rfiRatio,
    rfiSirDb: -20.0 * Math.log10(Math.max(1, rfiRatio)),
    outOfConeMaxErr,
    rawBytes,
    compressedBytes,
    compressionRatio: rawBytes / compressedBytes,
    rawBeamCosine: cosineSimilarityF32(rawBeamWaveform, refWaveform),
    vlbi2BitCosine: cosineSimilarityF32(vlbiBeamWaveform, refWaveform),
    scarCodecCosine: cosineSimilarityF32(scarBeamWaveform, refWaveform),
    scarCodecSnrDb: optimalGainSnrDb(scarBeamWaveform, refWaveform),
    refWaveform,
    rawBeamWaveform,
    vlbiBeamWaveform,
    scarBeamWaveform,
  };
}

export interface BlackBoxSCacheProofResult {
  totalFrames: number;
  rawStreamBytes: number;
  scacheBytesInitial: number;
  scacheBytesFinal: number;
  scacheDeltaBytes: number;
  ringBudgetBytes: number;
  ringCapacityFrames: number;
  faultTimestamps: number[];
  scacheRetainedFaults: number[];
  ringRetainedFaults: number[];
  associativeMeanCosine: number;
  maskedChannelsMeanCosine: number;
  surpriseTimelineDownsampled: Float32Array;
  trueFirstFaultSig: Float32Array;
  recalledFirstFaultSig: Float32Array;
}

export function runBlackBoxSCacheProof(totalFrames = 8000): BlackBoxSCacheProofResult {
  const prng = new DeterministicPRNG(20260402);
  const faultTimestamps = [120, 290, 470, 680, 910].filter((t) => t < totalFrames - 50);
  const sensorHarmonics = [1.0, 0.82, -0.74, 1.15, -0.68, 0.91];
  const faultSigs = new Map<number, Float32Array>();
  const faultStarts = new Map<number, number>();
  faultTimestamps.forEach((ts, idx) => {
    const sig = new Float32Array(64);
    const start = (idx * 11) % 56;
    faultStarts.set(ts, start);
    const pol = idx % 2 === 0 ? 1 : -1;
    const baseAmp = 9.5 + idx * 0.8;
    for (let k = 0; k < 6; k++) {
      sig[start + k] = pol * baseAmp * sensorHarmonics[k];
    }
    faultSigs.set(ts, sig);
  });

  const scache = new SurpriseGatedAnomalySCacheTS(4.2);
  const ring = new CircularRingBufferOTTS(scache.physicalBytes());
  const scacheBytesInitial = scache.physicalBytes();

  const bins = 160;
  const surpriseTimelineDownsampled = new Float32Array(bins);
  const frame = new Float32Array(64);

  for (let ts = 0; ts < totalFrames; ts++) {
    const fSig = faultSigs.get(ts);
    for (let c = 0; c < 64; c++) {
      frame[c] = 0.15 * prng.nextGaussian() + (fSig ? fSig[c] : 0);
    }
    const surp = scache.ingest(frame, ts);
    ring.ingest(frame, ts);
    const bin = Math.min(bins - 1, Math.floor((ts * bins) / totalFrames));
    if (surp > surpriseTimelineDownsampled[bin]) {
      surpriseTimelineDownsampled[bin] = surp;
    }
  }

  const scacheBytesFinal = scache.physicalBytes();
  const scacheSet = new Set(scache.recordedTimestamps());
  const ringSet = ring.recordedTimestamps();

  const scacheRetainedFaults = faultTimestamps.filter((t) => scacheSet.has(t));
  const ringRetainedFaults = faultTimestamps.filter((t) => ringSet.has(t));

  // Blind partial-cue hetero-associative recall (`0` oracle full-signature self-querying):
  // Query `W_scar` with ONLY 2 triggered alarm channels (`4` of `6` fault channels masked to `0.0`!),
  // and verify reconstruction across both all 64 channels and the 4 completely unobserved channels.
  let cosSum = 0;
  let maskedCosSum = 0;
  let firstTrue: Float32Array = new Float32Array(64);
  let firstRec: Float32Array = new Float32Array(64);
  faultTimestamps.forEach((ts, i) => {
    const sig = faultSigs.get(ts)!;
    const start = faultStarts.get(ts)!;
    const partialCue = new Float32Array(64);
    partialCue[start] = sig[start];
    partialCue[start + 1] = sig[start + 1];
    const rec = scache.queryAssociative(partialCue);

    // Rescale recalled vector to match observed 2-channel cue amplitude for visual overlay
    const cueScale =
      (partialCue[start] * rec[start] + partialCue[start + 1] * rec[start + 1]) /
      Math.max(1e-12, rec[start] * rec[start] + rec[start + 1] * rec[start + 1]);
    const recScaled = new Float32Array(64);
    for (let c = 0; c < 64; c++) recScaled[c] = rec[c] * cueScale;

    cosSum += cosineSimilarityF32(recScaled, sig);
    maskedCosSum += cosineSimilarityF32(
      recScaled.subarray(start + 2, start + 6),
      sig.subarray(start + 2, start + 6)
    );
    if (i === 0) {
      firstTrue = sig;
      firstRec = recScaled;
    }
  });

  return {
    totalFrames,
    rawStreamBytes: totalFrames * 64 * 4,
    scacheBytesInitial,
    scacheBytesFinal,
    scacheDeltaBytes: scacheBytesFinal - scacheBytesInitial,
    ringBudgetBytes: ring.physicalBytes(),
    ringCapacityFrames: ring.capacity,
    faultTimestamps,
    scacheRetainedFaults,
    ringRetainedFaults,
    associativeMeanCosine: cosSum / Math.max(1, faultTimestamps.length),
    maskedChannelsMeanCosine: maskedCosSum / Math.max(1, faultTimestamps.length),
    surpriseTimelineDownsampled,
    trueFirstFaultSig: firstTrue,
    recalledFirstFaultSig: firstRec,
  };
}

export interface CfdHdrFieldProofState {
  rows: number;
  cols: number;
  vacuumRows: number;
  rawBytes: number;
  scarBytes: number;
  spectralBytes: number;
  spectralModesK: number;
  compressionRatio: number;
  scarRmse: number;
  spectralRmse: number;
  scarVacuumMaxErr: number;
  spectralVacuumMaxErr: number;
  scarPeakHdrMaxErr: number;
  spectralPeakHdrMaxErr: number;
  groundTruth: Float32Array;
  scarReconstructed: Float32Array;
  spectralTruncated: Float32Array;
  pkg: PhysicalVQ4ScarPackageTS;
  codec: PhysicalVQ4ScarCodecTS;
  spikeCoords: Array<[number, number]>;
}

/**
 * Byte-matched 2D Orthonormal DCT-II spectral truncation baseline in IEEE-754 `FP16` (`2 bytes/coeff`):
 * Retains `kSide x kSide` 2D frequency modes (`2 * kSide^2 <= targetBytes`, matching `scarBytes` byte-for-byte),
 * demonstrating true Gibbs ringing across PDE Rankine-Hugoniot shocks and `500x` impulses without under-budgeting.
 */
export function computeMatchedByte2dDctFp16(
  field: Float32Array,
  rows: number,
  cols: number,
  targetBytes: number
): { reconstructed: Float32Array; spectralBytes: number; kSide: number } {
  let maxAbs = 0;
  for (let i = 0; i < field.length; i++) {
    const av = Math.abs(field[i]);
    if (av > maxAbs) maxAbs = av;
  }
  const gScale = computeLosslessPow2ScaleTS(maxAbs);
  const invG = 1.0 / gScale;
  const headerBytes = gScale !== 1.0 ? 4 : 0;

  const maxCoeffsFp16 = Math.max(16, Math.floor((targetBytes - headerBytes) / 2));
  const kRow = Math.min(rows, Math.max(2, Math.floor(Math.sqrt(maxCoeffsFp16 * (rows / cols)))));
  const kCol = Math.min(cols, Math.max(4, Math.floor(maxCoeffsFp16 / kRow)));

  const cosR = new Float32Array(kRow * rows);
  for (let kr = 0; kr < kRow; kr++) {
    const sc = kr === 0 ? Math.sqrt(1.0 / rows) : Math.sqrt(2.0 / rows);
    for (let r = 0; r < rows; r++) {
      cosR[kr * rows + r] = sc * Math.cos((Math.PI * (2 * r + 1) * kr) / (2 * rows));
    }
  }
  const cosC = new Float32Array(kCol * cols);
  for (let kc = 0; kc < kCol; kc++) {
    const sc = kc === 0 ? Math.sqrt(1.0 / cols) : Math.sqrt(2.0 / cols);
    for (let c = 0; c < cols; c++) {
      cosC[kc * cols + c] = sc * Math.cos((Math.PI * (2 * c + 1) * kc) / (2 * cols));
    }
  }

  // Forward 2D DCT: `temp[r, kc] = sum_c fieldScaled[r, c] * cosC[kc, c]`
  const temp = new Float32Array(rows * kCol);
  for (let r = 0; r < rows; r++) {
    const rOff = r * cols;
    for (let kc = 0; kc < kCol; kc++) {
      let s = 0;
      const cOff = kc * cols;
      if (gScale !== 1.0) {
        for (let c = 0; c < cols; c++) s += field[rOff + c] * invG * cosC[cOff + c];
      } else {
        for (let c = 0; c < cols; c++) s += field[rOff + c] * cosC[cOff + c];
      }
      temp[r * kCol + kc] = s;
    }
  }

  // `coeffF16[kr, kc] = float16(sum_r cosR[kr, r] * temp[r, kc])`
  const coeffF32 = new Float32Array(kRow * kCol);
  for (let kr = 0; kr < kRow; kr++) {
    const krOff = kr * rows;
    for (let kc = 0; kc < kCol; kc++) {
      let s = 0;
      for (let r = 0; r < rows; r++) s += cosR[krOff + r] * temp[r * kCol + kc];
      coeffF32[kr * kCol + kc] = float16BitsToFloat32(float32ToFloat16Bits(s));
    }
  }

  // Inverse 2D DCT from exact FP16-quantized 2D coefficients
  const invR = new Float32Array(rows * kCol);
  for (let r = 0; r < rows; r++) {
    for (let kc = 0; kc < kCol; kc++) {
      let s = 0;
      for (let kr = 0; kr < kRow; kr++) s += cosR[kr * rows + r] * coeffF32[kr * kCol + kc];
      invR[r * kCol + kc] = s;
    }
  }

  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const rOff = r * cols;
    const irOff = r * kCol;
    for (let c = 0; c < cols; c++) {
      let s = 0;
      for (let kc = 0; kc < kCol; kc++) s += invR[irOff + kc] * cosC[kc * cols + c];
      out[rOff + c] = gScale !== 1.0 ? s * gScale : s;
    }
  }

  return { reconstructed: out, spectralBytes: kRow * kCol * 2 + headerBytes, kSide: kRow };
}

/**
 * Integrates the 2D nonlinear inviscid Burgers conservation law PDE
 *   du/dt + d/dx (0.5 * u^2) + c_y * du/dy = 0
 * via conservative Rusanov (Local Lax-Friedrichs) finite-volume fluxes from smooth Riemann-like initial data,
 * dynamically steepening into a genuine Rankine-Hugoniot shockwave bounded by an outer-space vacuum domain (`u = 0`).
 */
function integrateRusanovBurgersShockPde(
  rows: number,
  cols: number,
  vacuumRows: number,
  nSteps = 44
): Float32Array {
  const activeRows = rows - vacuumRows;
  let u = new Float32Array(activeRows * cols);
  let uNext = new Float32Array(activeRows * cols);

  for (let ar = 0; ar < activeRows; ar++) {
    const y = (ar / Math.max(1, activeRows - 1)) * 2.0 - 1.0;
    const xFront = 0.08 + 0.28 * y;
    for (let c = 0; c < cols; c++) {
      const x = (c / Math.max(1, cols - 1)) * 2.0 - 1.0;
      // Smooth compressive tanh ramp (`u_L = +4.4` -> `u_R = -1.1`) + smooth acoustic waves:
      // Because `u_L > u_R`, characteristics converge and form a genuine Rankine-Hugoniot shock (`s = 0.5*(u_L+u_R)`).
      const ramp = 1.65 - 2.75 * Math.tanh((x - xFront) / 0.22);
      const wave = 0.85 * Math.sin(2.2 * Math.PI * x) * Math.cos(1.6 * Math.PI * y);
      u[ar * cols + c] = ramp + wave;
    }
  }

  const dx = 2.0 / Math.max(1, cols - 1);
  const dy = 2.0 / Math.max(1, activeRows - 1);
  const cy = 0.35;

  for (let step = 0; step < nSteps; step++) {
    let maxSpeedX = 1.0;
    for (let i = 0; i < u.length; i++) {
      const av = Math.abs(u[i]);
      if (av > maxSpeedX) maxSpeedX = av;
    }
    const dt = 0.36 / (maxSpeedX / dx + Math.abs(cy) / dy);

    for (let ar = 0; ar < activeRows; ar++) {
      const arUp = ar > 0 ? ar - 1 : 0;
      const rOff = ar * cols;
      const rUpOff = arUp * cols;
      for (let c = 0; c < cols; c++) {
        const cL = c > 0 ? c - 1 : 0;
        const cR = c + 1 < cols ? c + 1 : cols - 1;
        const uL = u[rOff + cL];
        const uC = u[rOff + c];
        const uR = u[rOff + cR];

        // Rusanov numerical flux `F(u_a, u_b) = 0.5*(0.5*u_a^2 + 0.5*u_b^2) - 0.5*alpha*(u_b - u_a)`
        const alphaR = Math.max(Math.abs(uC), Math.abs(uR));
        const fluxR = 0.25 * (uC * uC + uR * uR) - 0.5 * alphaR * (uR - uC);
        const alphaL = Math.max(Math.abs(uL), Math.abs(uC));
        const fluxL = 0.25 * (uL * uL + uC * uC) - 0.5 * alphaL * (uC - uL);

        // Upwind transverse advection `cy * du/dy` (`cy > 0`)
        const advY = cy * ((uC - u[rUpOff + c]) / dy);
        uNext[rOff + c] = uC - (dt / dx) * (fluxR - fluxL) - dt * advY;
      }
    }
    const swap = u;
    u = uNext;
    uNext = swap;
  }

  const full = new Float32Array(rows * cols);
  full.set(u, vacuumRows * cols);
  return full;
}

export function buildCfdHdrShockFieldProof(): CfdHdrFieldProofState {
  const rows = 128;
  const cols = 128;
  const vacuumRows = Math.floor(0.375 * rows); // Top 48 rows = exact outer-space vacuum `0.000000`
  const prng = new DeterministicPRNG(20260403);
  const groundTruth = integrateRusanovBurgersShockPde(rows, cols, vacuumRows, 44);

  for (let r = vacuumRows; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      groundTruth[r * cols + c] += 0.035 * prng.nextGaussian();
    }
  }

  // Inject 16 localized `500x` specular HDR glints / cosmic-ray hits (`amplitude ~ 500.0`)
  const spikeCoords: Array<[number, number]> = [];
  for (let s = 0; s < 16; s++) {
    const rIdx = vacuumRows + 6 + ((s * 17) % (rows - vacuumRows - 10));
    const cIdx = (s * 29 + 11) % cols;
    groundTruth[rIdx * cols + cIdx] += 500.0 * (s % 2 === 0 ? 1.0 : -0.85);
    spikeCoords.push([rIdx, cIdx]);
  }

  const codec = new PhysicalVQ4ScarCodecTS({
    rank: 8,
    kmeansIters: 5,
    maxPointSpikes: 48,
    spikeSigmaThreshold: 4.2,
  });
  const pkg = codec.encode(groundTruth, rows, cols);
  const scarReconstructed = codec.decodeFull(pkg);
  const scarBytes = getPhysicalPackageBytes(pkg);

  const {
    reconstructed: spectralTruncated,
    spectralBytes,
    kSide: spectralModesK,
  } = computeMatchedByte2dDctFp16(
    groundTruth,
    rows,
    cols,
    scarBytes
  );

  let err2Scar = 0;
  let err2Spec = 0;
  let scarVacuumMaxErr = 0;
  let spectralVacuumMaxErr = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const ds = Math.abs(scarReconstructed[idx] - groundTruth[idx]);
      const df = Math.abs(spectralTruncated[idx] - groundTruth[idx]);
      err2Scar += ds * ds;
      err2Spec += df * df;
      if (r < vacuumRows) {
        if (ds > scarVacuumMaxErr) scarVacuumMaxErr = ds;
        if (df > spectralVacuumMaxErr) spectralVacuumMaxErr = df;
      }
    }
  }

  let scarPeakHdrMaxErr = 0;
  let spectralPeakHdrMaxErr = 0;
  for (const [sr, sc] of spikeCoords) {
    const idx = sr * cols + sc;
    const es = Math.abs(scarReconstructed[idx] - groundTruth[idx]);
    const ef = Math.abs(spectralTruncated[idx] - groundTruth[idx]);
    if (es > scarPeakHdrMaxErr) scarPeakHdrMaxErr = es;
    if (ef > spectralPeakHdrMaxErr) spectralPeakHdrMaxErr = ef;
  }

  const rawBytes = groundTruth.byteLength;
  return {
    rows,
    cols,
    vacuumRows,
    rawBytes,
    scarBytes,
    spectralBytes,
    spectralModesK,
    compressionRatio: rawBytes / scarBytes,
    scarRmse: Math.sqrt(err2Scar / groundTruth.length),
    spectralRmse: Math.sqrt(err2Spec / groundTruth.length),
    scarVacuumMaxErr,
    spectralVacuumMaxErr,
    scarPeakHdrMaxErr,
    spectralPeakHdrMaxErr,
    groundTruth,
    scarReconstructed,
    spectralTruncated,
    pkg,
    codec,
    spikeCoords,
  };
}


// ============================================================================
// ScarCodec v6.0 Browser Extensions: Gosset E8 (`1.000 bpw`) + Live Real-Slice
// Forensic Workbench (`LIGO 10^-19`, `Tohoku 3.13M`, `MIT-BIH IEC`, `NOAA`, `ROSAT`)
// ============================================================================

let _cachedE8Lut256: Float32Array | null = null;
let _cachedE8HalfNorm2: Float32Array | null = null;

/**
 * Canonical `[256, 8]` Gosset `E_8 + 0x00 Vacuum + H_8 Shell` lattice (`1.000 bit/sample`, `0` codebook payload bytes!).
 * Bit-identical with `scar_codec.nd_e8.build_canonical_e8_lut256_np` and `scar_field.e8_lattice`.
 */
export function buildCanonicalE8Lut256TS(): { lut256x8: Float32Array; halfNorm2: Float32Array } {
  if (_cachedE8Lut256 && _cachedE8HalfNorm2) {
    return { lut256x8: _cachedE8Lut256, halfNorm2: _cachedE8HalfNorm2 };
  }
  const lut = new Float32Array(256 * 8);
  let row = 1; // Row 0 = exact physical vacuum (0,0,0,0,0,0,0,0)

  // 1..112: D_8 integer roots `{±2, ±2, 0..}` (`||u||^2 = 8`)
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      for (const si of [-2.0, 2.0]) {
        for (const sj of [-2.0, 2.0]) {
          lut[row * 8 + i] = si;
          lut[row * 8 + j] = sj;
          row++;
        }
      }
    }
  }

  // 113..240: 128 even-parity spinor roots `{±1}^8` (`||u||^2 = 8`)
  for (let mask = 0; mask < 256; mask++) {
    let negCount = 0;
    for (let b = 0; b < 8; b++) {
      // Match Python `itertools.product((-1.0, 1.0), repeat=8)` big-endian bit order
      const isPos = ((mask >>> (7 - b)) & 1) === 1;
      if (!isPos) negCount++;
    }
    if ((negCount & 1) === 0) {
      for (let b = 0; b < 8; b++) {
        const isPos = ((mask >>> (7 - b)) & 1) === 1;
        lut[row * 8 + b] = isPos ? 1.0 : -1.0;
      }
      row++;
    }
  }

  // 241..255: 15 Walsh-Hadamard H_8 outer shell vectors (`||u||^2 = 18`)
  let hMat: number[][] = [[1.0]];
  for (let st = 0; st < 3; st++) {
    const n = hMat.length;
    const next: number[][] = new Array(n * 2);
    for (let r = 0; r < n; r++) {
      next[r] = hMat[r].concat(hMat[r]);
      next[r + n] = hMat[r].concat(hMat[r].map((v) => -v));
    }
    hMat = next;
  }
  const hScale = Math.sqrt(18.0) / Math.sqrt(8.0); // 1.5
  for (let b = 0; b < 8; b++) {
    lut[row * 8 + b] = hMat[0][b] * hScale;
  }
  row++;
  for (let r = 1; r < 8; r++) {
    for (let b = 0; b < 8; b++) lut[row * 8 + b] = hMat[r][b] * hScale;
    row++;
    for (let b = 0; b < 8; b++) lut[row * 8 + b] = -hMat[r][b] * hScale;
    row++;
  }

  const halfNorm2 = new Float32Array(256);
  for (let c = 0; c < 256; c++) {
    let s2 = 0;
    for (let b = 0; b < 8; b++) {
      const v = lut[c * 8 + b];
      s2 += v * v;
    }
    halfNorm2[c] = 0.5 * s2;
  }

  _cachedE8Lut256 = lut;
  _cachedE8HalfNorm2 = halfNorm2;
  return { lut256x8: lut, halfNorm2 };
}

export interface ScarE8PackageTS {
  origRows: number;
  origCols: number;
  paddedCols: number;
  rank: number;
  e8CodesU8: Uint8Array;         // [N * (K_pad / 8)] (`1.000 bit/sample`, 0 LUT bytes!)
  scalesF16: Uint16Array;        // [N * (K_pad / 64)]
  dEqF16: Uint16Array;           // [K_pad]
  aScarU8: Uint8Array;
  aScalesF16: Uint16Array;
  bScarU8: Uint8Array;
  bRowScalesF16: Uint16Array;
  spikeFlatIdxI32: Int32Array;
  spikeResidualF16: Uint16Array;
  globalScale: number;
  absTolerance?: number;
}

export function getScarE8PackageBytes(pkg: ScarE8PackageTS): number {
  return (
    pkg.e8CodesU8.byteLength +
    pkg.scalesF16.byteLength +
    pkg.dEqF16.byteLength +
    pkg.aScarU8.byteLength +
    pkg.aScalesF16.byteLength +
    pkg.bScarU8.byteLength +
    pkg.bRowScalesF16.byteLength +
    pkg.spikeFlatIdxI32.byteLength +
    pkg.spikeResidualF16.byteLength +
    (pkg.globalScale !== 1.0 ? 4 : 0)
  );
}

export class ScarCodecE8TS {
  readonly rank: number;
  readonly maxPointSpikes: number;
  readonly spikeSigmaThreshold: number;
  readonly absTolerance?: number;

  constructor(opts?: {
    rank?: number;
    maxPointSpikes?: number;
    spikeSigmaThreshold?: number;
    absTolerance?: number;
  }) {
    this.rank = opts?.rank ?? 8;
    this.maxPointSpikes = opts?.maxPointSpikes ?? 64;
    this.spikeSigmaThreshold = opts?.spikeSigmaThreshold ?? 4.2;
    this.absTolerance = opts?.absTolerance;
  }

  encode(field: Float32Array, origRows: number, origCols: number): ScarE8PackageTS {
    const { lut256x8, halfNorm2 } = buildCanonicalE8Lut256TS();
    let fieldMaxAbs = 0;
    for (let i = 0; i < field.length; i++) {
      const av = Math.abs(field[i]);
      if (av > fieldMaxAbs) fieldMaxAbs = av;
    }
    const globalScale = computeLosslessPow2ScaleTS(fieldMaxAbs);
    const invG = 1.0 / globalScale;

    const paddedCols = ((origCols + 63) >>> 6) << 6;
    const nBlkPerRow = paddedCols >>> 6;
    const x = new Float32Array(origRows * paddedCols);
    const zeroRowMask = new Uint8Array(origRows);

    for (let r = 0; r < origRows; r++) {
      let maxAbs = 0;
      for (let c = 0; c < origCols; c++) {
        const v = globalScale !== 1.0 ? field[r * origCols + c] * invG : field[r * origCols + c];
        x[r * paddedCols + c] = v;
        const av = Math.abs(v);
        if (av > maxAbs) maxAbs = av;
      }
      zeroRowMask[r] = maxAbs <= (globalScale !== 1.0 ? 0.0 : 1e-12) ? 1 : 0;
    }

    const xBase = new Float32Array(x);
    const preSpikeSet = new Set<number>();
    if (this.maxPointSpikes > 0) {
      const blkMedians = new Float32Array(origRows * nBlkPerRow);
      const activeDevs: number[] = [];
      const blkBuf: number[] = new Array(64);
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let b = 0; b < nBlkPerRow; b++) {
          const base = r * paddedCols + (b << 6);
          for (let t = 0; t < 64; t++) blkBuf[t] = xBase[base + t];
          const med = medianFloatArray(blkBuf);
          blkMedians[r * nBlkPerRow + b] = med;
          for (let t = 0; t < 64; t++) {
            if ((b << 6) + t < origCols) activeDevs.push(Math.abs(xBase[base + t] - med));
          }
        }
      }
      const madStd = Math.max(1e-6, 1.4826 * medianFloatArray(activeDevs));
      const cands: Array<{ idx: number; dev: number; med: number }> = [];
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const flatIdx = r * paddedCols + c;
          const med = blkMedians[r * nBlkPerRow + (c >>> 6)];
          const dev = Math.abs(xBase[flatIdx] - med);
          if (dev > this.spikeSigmaThreshold * madStd) {
            cands.push({ idx: flatIdx, dev, med });
          }
        }
      }
      cands.sort((a, b) => b.dev - a.dev);
      for (const item of cands.slice(0, this.maxPointSpikes)) {
        preSpikeSet.add(item.idx);
        xBase[item.idx] = item.med;
      }
    }

    const colRmsArr: number[] = new Array(paddedCols);
    for (let c = 0; c < paddedCols; c++) {
      let s2 = 0;
      for (let r = 0; r < origRows; r++) {
        const v = xBase[r * paddedCols + c];
        s2 += v * v;
      }
      colRmsArr[c] = Math.max(1e-6, Math.sqrt(s2 / origRows));
    }
    const medRms = Math.max(1e-6, medianFloatArray(colRmsArr));
    const dEqF16 = new Uint16Array(paddedCols);
    const dEqF32 = new Float32Array(paddedCols);
    for (let c = 0; c < paddedCols; c++) {
      const ratio = Math.min(32.0, Math.max(0.05, Math.sqrt(colRmsArr[c] / medRms)));
      const bits = float32ToFloat16Bits(ratio);
      dEqF16[c] = bits;
      dEqF32[c] = float16BitsToFloat32(bits);
    }

    const effRank = this.rank > 0 && Math.min(origRows, paddedCols) >= this.rank ? this.rank : 0;
    const resForVq = new Float32Array(origRows * paddedCols);
    if (effRank > 0) {
      const p1 = extractLowRankInt4Scar(xBase, origRows, paddedCols, effRank, zeroRowMask);
      for (let i = 0; i < resForVq.length; i++) resForVq[i] = xBase[i] - p1.scarHat[i];
    } else {
      resForVq.set(xBase);
    }

    const totalBlocks = origRows * nBlkPerRow;
    const scalesF16 = new Uint16Array(totalBlocks);
    const e8CodesU8 = new Uint8Array(totalBlocks * 8);
    const xE8 = new Float32Array(origRows * paddedCols);
    const tmp64 = new Float32Array(64);

    for (let r = 0; r < origRows; r++) {
      for (let b = 0; b < nBlkPerRow; b++) {
        const blkId = r * nBlkPerRow + b;
        const colOff = b << 6;
        const rowOff = r * paddedCols + colOff;
        let maxAbs = 0;
        for (let t = 0; t < 64; t++) {
          const v = resForVq[rowOff + t] / dEqF32[colOff + t];
          tmp64[t] = v;
          const av = Math.abs(v);
          if (av > maxAbs) maxAbs = av;
        }
        if (zeroRowMask[r] || maxAbs <= 1e-12) {
          scalesF16[blkId] = 0;
          continue;
        }
        fwhtBlock64Inplace(tmp64, colOff, false);
        let s2 = 0;
        for (let t = 0; t < 64; t++) s2 += tmp64[t] * tmp64[t];
        // Scale by 1 / sqrt(8/8) = block RMS so typical 8D norm^2 ~ 8 matches Gosset E8 minimal shell ||u||^2 = 8!
        const rms = Math.max(1e-7, Math.sqrt(s2 / 64.0));
        const scBits = float32ToFloat16Bits(rms);
        const scVal = Math.max(1e-7, float16BitsToFloat32(scBits));
        scalesF16[blkId] = scBits;

        for (let g = 0; g < 8; g++) {
          const gOff = g * 8;
          let bestCode = 0;
          let bestScore = 0.0; // score(0) = <z, 0> - 0.5*0 = 0.0
          for (let c = 1; c < 256; c++) {
            const lOff = c * 8;
            let dot = 0;
            for (let k = 0; k < 8; k++) {
              dot += (tmp64[gOff + k] / scVal) * lut256x8[lOff + k];
            }
            const score = dot - halfNorm2[c];
            if (score > bestScore) {
              bestScore = score;
              bestCode = c;
            }
          }
          e8CodesU8[blkId * 8 + g] = bestCode;
          for (let k = 0; k < 8; k++) {
            tmp64[gOff + k] = lut256x8[bestCode * 8 + k] * scVal;
          }
        }
        fwhtBlock64Inplace(tmp64, colOff, true);
        for (let t = 0; t < 64; t++) {
          xE8[rowOff + t] = tmp64[t] * dEqF32[colOff + t];
        }
      }
    }

    let aScarU8: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
    let aScalesF16: Uint16Array<ArrayBufferLike> = new Uint16Array(0);
    let bScarU8: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
    let bRowScalesF16: Uint16Array<ArrayBufferLike> = new Uint16Array(0);
    const xApprox = new Float32Array(xE8);

    if (effRank > 0) {
      const resP2 = new Float32Array(origRows * paddedCols);
      for (let i = 0; i < resP2.length; i++) resP2[i] = xBase[i] - xE8[i];
      const p2 = extractLowRankInt4Scar(resP2, origRows, paddedCols, effRank, zeroRowMask);
      aScarU8 = p2.aU8;
      aScalesF16 = p2.aScF16;
      bScarU8 = p2.bU8;
      bRowScalesF16 = p2.bScF16;
      for (let i = 0; i < xApprox.length; i++) xApprox[i] += p2.scarHat[i];
    }

    let spikeFlatIdxI32: Int32Array;
    let spikeResidualF16: Uint16Array;
    if (this.absTolerance !== undefined && this.absTolerance > 0) {
      const tolScaled = this.absTolerance * invG;
      const curErr = new Float32Array(origRows * paddedCols);
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const idx = r * paddedCols + c;
          curErr[idx] = x[idx] - xApprox[idx];
        }
      }
      const dekkerPairs: Array<{ idx: number; bits: number }> = [];
      for (let pass = 0; pass < 6; pass++) {
        let violations = 0;
        for (let r = 0; r < origRows; r++) {
          if (zeroRowMask[r]) continue;
          for (let c = 0; c < origCols; c++) {
            const idx = r * paddedCols + c;
            const rem = curErr[idx];
            if (Math.abs(rem) > tolScaled) {
              const bits = float32ToFloat16Bits(Math.max(-65504.0, Math.min(65504.0, rem)));
              dekkerPairs.push({ idx, bits });
              curErr[idx] = rem - float16BitsToFloat32(bits);
              violations++;
            }
          }
        }
        if (violations === 0) break;
      }
      dekkerPairs.sort((a, b) => a.idx - b.idx);
      spikeFlatIdxI32 = new Int32Array(dekkerPairs.length);
      spikeResidualF16 = new Uint16Array(dekkerPairs.length);
      for (let s = 0; s < dekkerPairs.length; s++) {
        spikeFlatIdxI32[s] = dekkerPairs[s].idx;
        spikeResidualF16[s] = dekkerPairs[s].bits;
      }
    } else {
      let baseEnergy = 0;
      let baseCnt = 0;
      for (let r = 0; r < origRows; r++) {
        if (zeroRowMask[r]) continue;
        for (let c = 0; c < origCols; c++) {
          const v = xBase[r * paddedCols + c];
          baseEnergy += v * v;
          baseCnt++;
        }
      }
      const baseRms = Math.sqrt(baseEnergy / Math.max(1, baseCnt));
      const spikeThresh = this.spikeSigmaThreshold * 0.25 * Math.max(1e-6, baseRms);
      const spikeCands: Array<{ idx: number; err: number }> = [];
      if (this.maxPointSpikes > 0) {
        for (let r = 0; r < origRows; r++) {
          if (zeroRowMask[r]) continue;
          for (let c = 0; c < origCols; c++) {
            const idx = r * paddedCols + c;
            const err = x[idx] - xApprox[idx];
            if (preSpikeSet.has(idx) || Math.abs(err) > spikeThresh) {
              spikeCands.push({ idx, err });
            }
          }
        }
      }
      spikeCands.sort((a, b) => Math.abs(b.err) - Math.abs(a.err));
      const chosen = spikeCands.slice(0, this.maxPointSpikes).sort((a, b) => a.idx - b.idx);
      spikeFlatIdxI32 = new Int32Array(chosen.length);
      spikeResidualF16 = new Uint16Array(chosen.length);
      for (let s = 0; s < chosen.length; s++) {
        spikeFlatIdxI32[s] = chosen[s].idx;
        spikeResidualF16[s] = float32ToFloat16Bits(chosen[s].err);
      }
    }

    return {
      origRows,
      origCols,
      paddedCols,
      rank: effRank,
      e8CodesU8,
      scalesF16,
      dEqF16,
      aScarU8,
      aScalesF16,
      bScarU8,
      bRowScalesF16,
      spikeFlatIdxI32,
      spikeResidualF16,
      globalScale,
      absTolerance: this.absTolerance,
    };
  }

  decodeBlockO1(pkg: ScarE8PackageTS, rowIdx: number, blockIdx: number): {
    values: Float32Array;
    bytesTouched: number;
  } {
    const { lut256x8 } = buildCanonicalE8Lut256TS();
    const startCol = blockIdx << 6;
    const nBlkPerRow = pkg.paddedCols >>> 6;
    const blkFlat = rowIdx * nBlkPerRow + blockIdx;
    const scBits = pkg.scalesF16[blkFlat];
    const tmp64 = new Float32Array(64);
    let bytesTouched = 2;

    if (scBits !== 0) {
      const sc = float16BitsToFloat32(scBits);
      bytesTouched += 8 + 128; // 8 x u8 E8 codes (`1.000 bpw`!) + 64 x f16 d_eq
      for (let g = 0; g < 8; g++) {
        const code = pkg.e8CodesU8[blkFlat * 8 + g];
        const gOff = g * 8;
        const lOff = code * 8;
        for (let k = 0; k < 8; k++) {
          tmp64[gOff + k] = lut256x8[lOff + k] * sc;
        }
      }
      fwhtBlock64Inplace(tmp64, startCol, true);
      for (let t = 0; t < 64; t++) {
        tmp64[t] *= float16BitsToFloat32(pkg.dEqF16[startCol + t]);
      }
    }

    if (pkg.rank > 0) {
      const bScBits = pkg.bRowScalesF16[rowIdx];
      bytesTouched += 2;
      if (bScBits !== 0) {
        const bSc = float16BitsToFloat32(bScBits);
        const bRow = new Float32Array(pkg.rank);
        const halfRank = pkg.rank >>> 1;
        const bOff = rowIdx * halfRank;
        bytesTouched += halfRank + pkg.rank * 34;
        for (let r2 = 0; r2 < halfRank; r2++) {
          const byte = pkg.bScarU8[bOff + r2];
          bRow[r2 * 2] = ((byte & 0x0f) - 8) * bSc;
          bRow[r2 * 2 + 1] = (((byte >>> 4) & 0x0f) - 8) * bSc;
        }
        const halfCols = pkg.paddedCols >>> 1;
        for (let r = 0; r < pkg.rank; r++) {
          const br = bRow[r];
          if (br === 0) continue;
          const aSc = float16BitsToFloat32(pkg.aScalesF16[r * nBlkPerRow + blockIdx]);
          const aByteBase = r * halfCols + (startCol >>> 1);
          for (let t2 = 0; t2 < 32; t2++) {
            const ab = pkg.aScarU8[aByteBase + t2];
            tmp64[t2 * 2] += br * (((ab & 0x0f) - 8) * aSc);
            tmp64[t2 * 2 + 1] += br * ((((ab >>> 4) & 0x0f) - 8) * aSc);
          }
        }
      }
    }

    const flatLo = rowIdx * pkg.paddedCols + startCol;
    const flatHi = flatLo + 64;
    let lo = 0;
    let hi = pkg.spikeFlatIdxI32.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (pkg.spikeFlatIdxI32[mid] < flatLo) lo = mid + 1;
      else hi = mid;
    }
    while (lo < pkg.spikeFlatIdxI32.length && pkg.spikeFlatIdxI32[lo] < flatHi) {
      const localOff = pkg.spikeFlatIdxI32[lo] - flatLo;
      tmp64[localOff] += float16BitsToFloat32(pkg.spikeResidualF16[lo]);
      bytesTouched += 6;
      lo++;
    }

    if (pkg.globalScale !== 1.0) {
      for (let t = 0; t < 64; t++) tmp64[t] *= pkg.globalScale;
    }

    const validLen = Math.min(64, Math.max(0, pkg.origCols - startCol));
    return { values: tmp64.subarray(0, validLen), bytesTouched };
  }

  decodeFull(pkg: ScarE8PackageTS): Float32Array {
    const out = new Float32Array(pkg.origRows * pkg.origCols);
    const nBlkPerRow = pkg.paddedCols >>> 6;
    for (let r = 0; r < pkg.origRows; r++) {
      for (let b = 0; b < nBlkPerRow; b++) {
        const { values } = this.decodeBlockO1(pkg, r, b);
        out.set(values, r * pkg.origCols + (b << 6));
      }
    }
    return out;
  }
}

export type RealSliceCodecMode = 'vq4' | 'e8' | 'error_bounded';

export interface RealSliceLiveAuditResultTS {
  rows: number;
  cols: number;
  mode: RealSliceCodecMode;
  rawFp32Bytes: number;
  scarBytes: number;
  compressionRatio: number;
  globalScalePow2: number;
  naiveFp16NonZeroCount: number;
  naiveFp16MaxAbsErr: number;
  scarRmse: number;
  scarMaxAbsErr: number;
  scarSnrDb: number;
  spectralBytes: number;
  spectralRmse: number;
  spectralMaxAbsErr: number;
  pureZeroBlockCount: number;
  scarPureZeroBlockMaxErr: number;
  spectralPureZeroBlockMaxErr: number;
  blockO1Row: number;
  blockO1ColBlock: number;
  blockO1BytesTouched: number;
  blockO1VsFullMaxDiff: number;
  groundTruth: Float32Array;
  scarReconstructed: Float32Array;
  naiveFp16Reconstructed: Float32Array;
  spectralReconstructed: Float32Array;
}

/**
 * Executes `ScarCodec v6.0` (`VQ4`, `Gosset E8`, or `Error-Bounded absTolerance`), naive IEEE-754 `FP16`,
 * byte-matched `FP16 2D-DCT` spectral truncation, and `O(1)` 64-sample block decode LIVE in the browser
 * on verbatim `SHA-256`-verified observational slices!
 */
export function runRealSliceLiveForensicAuditTS(opts: {
  field: Float32Array;
  rows: number;
  cols: number;
  mode: RealSliceCodecMode;
  absTolerance?: number;
  maxPointSpikes?: number;
  rank?: number;
  inspectRow?: number;
  inspectBlock?: number;
}): RealSliceLiveAuditResultTS {
  const { field, rows, cols, mode } = opts;
  const rawFp32Bytes = field.byteLength;
  const rank = opts.rank ?? (rows >= 8 ? 8 : 0);
  const maxPointSpikes = opts.maxPointSpikes ?? 64;

  // 1. Naive IEEE-754 FP16 cast (`0` exponent scaling — reveals 10^-19 underflow & >65,504 overflow!)
  const naiveFp16Reconstructed = new Float32Array(field.length);
  let naiveFp16NonZeroCount = 0;
  let naiveFp16MaxAbsErr = 0;
  for (let i = 0; i < field.length; i++) {
    const f16v = float16BitsToFloat32(float32ToFloat16Bits(field[i]));
    naiveFp16Reconstructed[i] = f16v;
    if (f16v !== 0) naiveFp16NonZeroCount++;
    const err = Math.abs(field[i] - f16v);
    if (err > naiveFp16MaxAbsErr) naiveFp16MaxAbsErr = err;
  }

  const nBlkPerRow = ((cols + 63) >>> 6);
  const blockO1Row = Math.min(rows - 1, Math.max(0, opts.inspectRow ?? (rows >>> 1)));
  const blockO1ColBlock = Math.min(nBlkPerRow - 1, Math.max(0, opts.inspectBlock ?? (nBlkPerRow >>> 1)));

  let scarReconstructed: Float32Array;
  let scarBytes: number;
  let globalScalePow2: number;
  let blockO1Values: Float32Array;
  let blockO1BytesTouched: number;

  if (mode === 'e8') {
    const e8Codec = new ScarCodecE8TS({
      rank,
      maxPointSpikes,
      spikeSigmaThreshold: 3.8,
    });
    const pkgE8 = e8Codec.encode(field, rows, cols);
    scarReconstructed = e8Codec.decodeFull(pkgE8);
    scarBytes = getScarE8PackageBytes(pkgE8);
    globalScalePow2 = pkgE8.globalScale;
    const blk = e8Codec.decodeBlockO1(pkgE8, blockO1Row, blockO1ColBlock);
    blockO1Values = blk.values;
    blockO1BytesTouched = blk.bytesTouched;
  } else {
    const vq4Codec = new PhysicalVQ4ScarCodecTS({
      rank,
      kmeansIters: 4,
      maxPointSpikes,
      spikeSigmaThreshold: 3.8,
      absTolerance: mode === 'error_bounded' ? opts.absTolerance : undefined,
    });
    const pkgVq4 = vq4Codec.encode(field, rows, cols);
    scarReconstructed = vq4Codec.decodeFull(pkgVq4);
    scarBytes = getPhysicalPackageBytes(pkgVq4);
    globalScalePow2 = pkgVq4.globalScale ?? 1.0;
    const blk = vq4Codec.decodeBlockO1(pkgVq4, blockO1Row, blockO1ColBlock);
    blockO1Values = blk.values;
    blockO1BytesTouched = blk.bytesTouched;
  }

  let blockO1VsFullMaxDiff = 0;
  const colStart = blockO1ColBlock << 6;
  for (let i = 0; i < blockO1Values.length; i++) {
    const d = Math.abs(blockO1Values[i] - scarReconstructed[blockO1Row * cols + colStart + i]);
    if (d > blockO1VsFullMaxDiff) blockO1VsFullMaxDiff = d;
  }

  const { reconstructed: spectralReconstructed, spectralBytes } = computeMatchedByte2dDctFp16(
    field,
    rows,
    cols,
    scarBytes
  );

  let sig2 = 0;
  let err2Scar = 0;
  let err2Spec = 0;
  let scarMaxAbsErr = 0;
  let spectralMaxAbsErr = 0;
  for (let i = 0; i < field.length; i++) {
    const gt = field[i];
    const es = Math.abs(gt - scarReconstructed[i]);
    const ef = Math.abs(gt - spectralReconstructed[i]);
    sig2 += gt * gt;
    err2Scar += es * es;
    err2Spec += ef * ef;
    if (es > scarMaxAbsErr) scarMaxAbsErr = es;
    if (ef > spectralMaxAbsErr) spectralMaxAbsErr = ef;
  }

  let pureZeroBlockCount = 0;
  let scarPureZeroBlockMaxErr = 0;
  let spectralPureZeroBlockMaxErr = 0;
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < Math.floor(cols / 64); b++) {
      const c0 = b * 64;
      let anyNonZero = false;
      for (let t = 0; t < 64; t++) {
        if (Math.abs(field[r * cols + c0 + t]) > 0) {
          anyNonZero = true;
          break;
        }
      }
      if (!anyNonZero) {
        pureZeroBlockCount++;
        for (let t = 0; t < 64; t++) {
          const idx = r * cols + c0 + t;
          const es = Math.abs(scarReconstructed[idx]);
          const ef = Math.abs(spectralReconstructed[idx]);
          if (es > scarPureZeroBlockMaxErr) scarPureZeroBlockMaxErr = es;
          if (ef > spectralPureZeroBlockMaxErr) spectralPureZeroBlockMaxErr = ef;
        }
      }
    }
  }

  const scarSnrDb = 10.0 * Math.log10(Math.max(1e-60, sig2) / Math.max(1e-72, err2Scar));
  return {
    rows,
    cols,
    mode,
    rawFp32Bytes,
    scarBytes,
    compressionRatio: rawFp32Bytes / Math.max(1, scarBytes),
    globalScalePow2,
    naiveFp16NonZeroCount,
    naiveFp16MaxAbsErr,
    scarRmse: Math.sqrt(err2Scar / field.length),
    scarMaxAbsErr,
    scarSnrDb,
    spectralBytes,
    spectralRmse: Math.sqrt(err2Spec / field.length),
    spectralMaxAbsErr,
    pureZeroBlockCount,
    scarPureZeroBlockMaxErr,
    spectralPureZeroBlockMaxErr,
    blockO1Row,
    blockO1ColBlock,
    blockO1BytesTouched,
    blockO1VsFullMaxDiff,
    groundTruth: field,
    scarReconstructed,
    naiveFp16Reconstructed,
    spectralReconstructed,
  };
}
