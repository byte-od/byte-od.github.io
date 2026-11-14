/**
 * Browser `GoldenSliceScarStoreV4` + RFC 7233 `HTTP/1.1 206 Partial Content`
 *
 * Streams exact `62,720 B` (`61.25 KB` `Block-64 INT4`) universal conceptual cartridges on demand:
 * - Zero upfront cartridge payload download (`0 B` until conceptual gate activates)
 * - Language-Subspace Orthogonal Projector `P_perp_lang = I - U_lang U_lang^T` (`k_lang = 14`)
 *   at the `2L/3` Golden Slice (`Layer 15`), enabling 1 English-anchored cartridge to answer
 *   across `PT`, `EN`, `ES`, and `FR`
 * - Conjunctive Sleep-Bridge co-activation + `< 0.2 ms` `0x00` LGPD physical unlearning
 */

export const PORTFOLIO_DEFAULT_SYSTEM_PROMPT =
  "You are System One, the official AI assistant for Julio Reis's portfolio (byte-od.github.io), running locally via WebGPU WGSL, MoS, and SCache O(1). Never claim to be Julio; refer to Julio Reis in the third person. Always reply in the same language as the user's message in 2 to 4 complete sentences.";

export interface CapsuleLink {
  label: string;
  url: string;
  icon?: 'colab' | 'github' | 'ext';
}

export interface CapsuleMetadata {
  title?: string;
  title_pt?: string;
  category?: string;
  summary?: string;
  links?: CapsuleLink[];
  example_prompt?: string;
  prompts_polyglot?: Record<string, string>;
  supported_languages?: string[];
  [key: string]: any;
}

export interface CapsuleDirectoryEntry {
  index: number;
  capsule_id: string;
  slot_idx: number;
  coord: number[];
  byte_offset: number;
  record_bytes: number;
  record_kb: number;
  local_store_has_bytes: boolean;
  tau: number;
  ramp: number;
  parent_capsules: string[];
  metadata: CapsuleMetadata;
  cachedInBrowser?: boolean;
  unlearnedWiped?: boolean;
  lastCosine?: number | null;
  lastGate?: number;
}

export interface Http206Event {
  capsuleId: string;
  url: string;
  status: number;
  range: string;
  contentRange: string;
  bytes: number;
  kb: number;
  latencyMs: number;
  timestamp: string;
}

export interface ActiveCapsuleRuntime {
  capsuleId: string;
  gate: number;
  floats: Float32Array;
  midFloats: Float32Array | null;
  finalFloats: Float32Array | null;
  fromCache: boolean;
  meta: CapsuleDirectoryEntry;
}

interface UnpackedCapsule {
  midFloats: Float32Array | null;
  finalFloats: Float32Array | null;
  packedGpuFloats: Float32Array;
  rawBytesView: Uint8Array;
}

export function fp16BitsToF32(h: number): number {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x03ff;
  const buf = new ArrayBuffer(4);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);

  if (e === 0) {
    if (m === 0) {
      u32[0] = s;
      return f32[0];
    }
    let mant = m;
    let exp = -14;
    while ((mant & 0x0400) === 0) {
      mant <<= 1;
      exp -= 1;
    }
    mant &= 0x03ff;
    u32[0] = s | ((exp + 127) << 23) | (mant << 13);
    return f32[0];
  }
  if (e === 31) {
    u32[0] = s | 0x7f800000 | (m << 13);
    return f32[0];
  }
  u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
  return f32[0];
}

export function findSubsequenceLastEnd(seqIds: number[], pattern: number[] | null | undefined): number {
  if (!pattern || pattern.length === 0 || seqIds.length < pattern.length) {
    return 0;
  }
  const m = pattern.length;
  for (let idx = seqIds.length - m; idx >= 0; idx--) {
    let ok = true;
    for (let j = 0; j < m; j++) {
      if (seqIds[idx + j] !== pattern[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return Math.min(idx + m, Math.max(seqIds.length - 1, 0));
    }
  }
  return 0;
}

export function extractUserContentBounds(
  seqIds: number[],
  prefixIds?: number[] | null,
  suffixIds?: number[] | null
): { qStart: number; qEnd: number } {
  const n = seqIds.length;
  if (n === 0) return { qStart: 0, qEnd: 0 };
  let qStart = prefixIds ? findSubsequenceLastEnd(seqIds, prefixIds) : 0;
  qStart = Math.max(0, Math.min(qStart, n - 1));
  let qEnd = n;
  if (suffixIds && suffixIds.length > 0) {
    const ls = suffixIds.length;
    if (n >= qStart + ls + 1) {
      let matches = true;
      for (let i = 0; i < ls; i++) {
        if (seqIds[n - ls + i] !== suffixIds[i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        qEnd = n - ls;
      }
    }
  }
  if (qEnd <= qStart) qEnd = n;
  return { qStart, qEnd };
}

export function computeHorizonScale(stepIdx: number, seedHorizon = 50, fadeSteps = 6): number {
  if (seedHorizon <= 0) return 1.0;
  if (stepIdx <= seedHorizon) return 1.0;
  if (fadeSteps <= 0 || stepIdx >= seedHorizon + fadeSteps) return 0.0;
  const frac = (stepIdx - seedHorizon) / fadeSteps;
  return 0.5 * (1.0 + Math.cos(Math.PI * frac));
}

export class BrowserScarStoreHTTP206 {
  public readonly meta: any;
  public readonly baseUrl: string;
  public readonly isGoldenSliceV4: boolean;
  public readonly hiddenDim: number;
  public readonly numHeads: number;
  public readonly headDim: number;
  public readonly recordBytes: number;
  public readonly userTurnPrefixIds: number[];
  public readonly userTurnSuffixIds: number[];
  public readonly midLayerIdx: number;
  public readonly rankMid: number;
  public readonly rankFinal: number;
  public readonly rank: number;
  public readonly numPrototypes: number;
  public readonly attentionBeta: number;
  public readonly kLang: number;
  public readonly seedHorizon: number;
  public readonly seedFadeSteps: number;
  public readonly winnerMargin: number;
  public readonly spanWindow: number;
  public readonly spanStride: number;
  public readonly systemPrompt: string;

  public readonly muQueryLast: Float32Array;
  public readonly muHiddenMid: Float32Array;
  public readonly muHiddenFinal: Float32Array;
  public readonly muQHeads: Float32Array;
  public readonly uLangMid: Float32Array | null;
  public readonly zAnchorsFlat: Float32Array;
  private readonly zAnchorsBackup: Float32Array;

  public readonly hotCapsules = new Map<string, UnpackedCapsule>();
  public readonly unlearnedIds = new Set<string>();
  public readonly http206Events: Http206Event[] = [];
  public totalHttp206Bytes = 0;
  public readonly lastCosines = new Map<string, number>();
  public readonly lastActiveGates = new Map<string, number>();

  constructor(routerMeta: any, routerBinBuffer: ArrayBuffer, baseUrl = '.') {
    this.meta = routerMeta;
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.isGoldenSliceV4 = routerMeta.routing_architecture === 'golden_slice_v4';
    this.hiddenDim = routerMeta.hidden_dim;
    this.numHeads = routerMeta.num_heads;
    this.headDim = routerMeta.head_dim;
    this.recordBytes = routerMeta.record_bytes;
    this.userTurnPrefixIds = routerMeta.user_turn_prefix_ids || [151644, 872, 198];
    this.userTurnSuffixIds = routerMeta.user_turn_suffix_ids || [151645, 198, 151644, 77091, 198];
    this.systemPrompt = routerMeta.system_prompt || PORTFOLIO_DEFAULT_SYSTEM_PROMPT;

    const floats = new Float32Array(routerBinBuffer);
    const lay = routerMeta.router_bin_layout;

    if (this.isGoldenSliceV4) {
      this.midLayerIdx = routerMeta.mid_layer_idx;
      this.rankMid = routerMeta.rank_mid;
      this.rankFinal = routerMeta.rank_final;
      this.rank = this.rankMid + this.rankFinal;
      this.numPrototypes = 0;
      this.attentionBeta = 0.0;
      this.kLang = routerMeta.k_lang;
      this.seedHorizon = routerMeta.seed_horizon ?? 50;
      this.seedFadeSteps = routerMeta.seed_fade_steps ?? 6;
      this.winnerMargin = routerMeta.winner_margin ?? 0.0002;
      this.spanWindow = routerMeta.span_window ?? 10;
      this.spanStride = routerMeta.span_stride ?? 3;

      this.muQueryLast = floats.slice(
        lay.mu_query_mid.offset_floats,
        lay.mu_query_mid.offset_floats + lay.mu_query_mid.length_floats
      );
      this.muHiddenMid = floats.slice(
        lay.mu_hidden_mid.offset_floats,
        lay.mu_hidden_mid.offset_floats + lay.mu_hidden_mid.length_floats
      );
      this.muHiddenFinal = floats.slice(
        lay.mu_hidden_final.offset_floats,
        lay.mu_hidden_final.offset_floats + lay.mu_hidden_final.length_floats
      );
      this.uLangMid = floats.slice(
        lay.u_lang_mid.offset_floats,
        lay.u_lang_mid.offset_floats + lay.u_lang_mid.length_floats
      );
      this.muQHeads = new Float32Array(0);
    } else {
      this.midLayerIdx = Math.max(0, (routerMeta.num_layers ?? 24) - 2);
      this.rankMid = 0;
      this.rankFinal = routerMeta.rank;
      this.rank = routerMeta.rank;
      this.numPrototypes = routerMeta.num_key_prototypes;
      this.attentionBeta = routerMeta.attention_beta ?? 14.0;
      this.kLang = 0;
      this.seedHorizon = 0;
      this.seedFadeSteps = 0;
      this.winnerMargin = 0.0;
      this.spanWindow = routerMeta.span_window ?? 10;
      this.spanStride = routerMeta.span_stride ?? 3;

      this.muQueryLast = floats.slice(
        lay.mu_query_last.offset_floats,
        lay.mu_query_last.offset_floats + lay.mu_query_last.length_floats
      );
      this.muHiddenMid = new Float32Array(0);
      this.muHiddenFinal = floats.slice(
        lay.mu_hidden_final.offset_floats,
        lay.mu_hidden_final.offset_floats + lay.mu_hidden_final.length_floats
      );
      this.muQHeads = floats.slice(
        lay.mu_q_heads.offset_floats,
        lay.mu_q_heads.offset_floats + lay.mu_q_heads.length_floats
      );
      this.uLangMid = null;
    }

    this.zAnchorsFlat = floats.slice(
      lay.z_anchors.offset_floats,
      lay.z_anchors.offset_floats + lay.z_anchors.length_floats
    );
    this.zAnchorsBackup = Float32Array.from(this.zAnchorsFlat);
  }

  public static async load(routerJsonUrl = './scarstore_router.json'): Promise<BrowserScarStoreHTTP206> {
    const resp = await fetch(routerJsonUrl, { cache: 'no-cache' });
    if (!resp.ok) {
      throw new Error(`Failed to fetch ScarStore router ${routerJsonUrl}: HTTP ${resp.status}`);
    }
    const meta = await resp.json();
    const baseDir = routerJsonUrl.includes('/') ? routerJsonUrl.slice(0, routerJsonUrl.lastIndexOf('/')) : '.';
    const binResp = await fetch(`${baseDir}/${meta.router_bin_file}`, { cache: 'no-cache' });
    if (!binResp.ok) {
      throw new Error(`Failed to fetch ${meta.router_bin_file}: HTTP ${binResp.status}`);
    }
    const binBuf = await binResp.arrayBuffer();
    return new BrowserScarStoreHTTP206(meta, binBuf, baseDir);
  }

  public getCalibrationBufferF32(): Float32Array {
    if (this.isGoldenSliceV4) {
      const out = new Float32Array(this.muHiddenMid.length + this.muHiddenFinal.length);
      out.set(this.muHiddenMid, 0);
      out.set(this.muHiddenFinal, this.muHiddenMid.length);
      return out;
    }
    const out = new Float32Array(this.muHiddenFinal.length + this.muQHeads.length);
    out.set(this.muHiddenFinal, 0);
    out.set(this.muQHeads, this.muHiddenFinal.length);
    return out;
  }

  public getMidCapsuleStrideFloats(): number {
    if (this.isGoldenSliceV4) {
      return 1 + 2 * (this.rankMid * this.hiddenDim);
    }
    return 1;
  }

  public getCapsuleStrideFloats(): number {
    if (this.isGoldenSliceV4) {
      return 1 + 2 * (this.rankFinal * this.hiddenDim);
    }
    const qDim = this.numHeads * this.headDim;
    return 1 + 2 * (this.rank * this.hiddenDim) + this.numPrototypes * qDim;
  }

  public getCapsulesDirectory(): CapsuleDirectoryEntry[] {
    return ((this.meta.capsules || []) as CapsuleDirectoryEntry[]).map((c) => ({
      ...c,
      cachedInBrowser: this.hotCapsules.has(c.capsule_id),
      unlearnedWiped: this.unlearnedIds.has(c.capsule_id),
      lastCosine: this.lastCosines.get(c.capsule_id) ?? null,
      lastGate: this.lastActiveGates.get(c.capsule_id) ?? 0.0,
    }));
  }

  public unpackCapsuleInt4Bytes(arrayBuffer: ArrayBuffer): UnpackedCapsule {
    const bytes = new Uint8Array(arrayBuffer);
    const dv = new DataView(arrayBuffer);
    const d = this.hiddenDim;
    let off = 0;

    const dequantBlocks = (nBlk: number): Float32Array => {
      const out = new Float32Array(nBlk * 64);
      const pkOff = off;
      off += nBlk * 32;
      const scOff = off;
      off += nBlk * 2;

      for (let b = 0; b < nBlk; b++) {
        const scale = fp16BitsToF32(dv.getUint16(scOff + b * 2, true));
        const rowByteBase = pkOff + b * 32;
        const outBase = b * 64;
        for (let i = 0; i < 32; i++) {
          const pb = bytes[rowByteBase + i];
          out[outBase + i * 2] = ((pb & 0x0f) - 8) * scale;
          out[outBase + i * 2 + 1] = (((pb >> 4) & 0x0f) - 8) * scale;
        }
      }
      return out;
    };

    if (this.isGoldenSliceV4) {
      const nMid = (d * this.rankMid) >> 6;
      const nFin = (d * this.rankFinal) >> 6;
      const bMid = dequantBlocks(nMid);
      const aMid = dequantBlocks(nMid);
      const bFin = dequantBlocks(nFin);
      const aFin = dequantBlocks(nFin);

      const midFloats = new Float32Array(aMid.length + bMid.length);
      midFloats.set(aMid, 0);
      midFloats.set(bMid, aMid.length);

      const finalFloats = new Float32Array(aFin.length + bFin.length);
      finalFloats.set(aFin, 0);
      finalFloats.set(bFin, aFin.length);

      return {
        midFloats,
        finalFloats,
        packedGpuFloats: finalFloats,
        rawBytesView: bytes,
      };
    }

    const r = this.rank;
    const pCnt = this.numPrototypes;
    const qDim = this.numHeads * this.headDim;
    const nAb = (d * r) >> 6;
    const nK = (pCnt * qDim) >> 6;

    const bFlat = dequantBlocks(nAb);
    const aFlat = dequantBlocks(nAb);
    const kFlat = dequantBlocks(nK);

    const totalHeads = pCnt * this.numHeads;
    for (let ph = 0; ph < totalHeads; ph++) {
      const base = ph * this.headDim;
      let normSq = 0.0;
      for (let j = 0; j < this.headDim; j++) {
        const v = kFlat[base + j];
        normSq += v * v;
      }
      const invNorm = 1.0 / Math.max(Math.sqrt(normSq), 1e-8);
      for (let j = 0; j < this.headDim; j++) {
        kFlat[base + j] *= invNorm;
      }
    }

    const totalLen = aFlat.length + bFlat.length + kFlat.length;
    const combined = new Float32Array(totalLen);
    combined.set(aFlat, 0);
    combined.set(bFlat, aFlat.length);
    combined.set(kFlat, aFlat.length + bFlat.length);
    return { midFloats: null, finalFloats: null, packedGpuFloats: combined, rawBytesView: bytes };
  }

  public async fetchCapsuleSliceHTTP206(capEntry: CapsuleDirectoryEntry): Promise<{
    floats: Float32Array;
    midFloats: Float32Array | null;
    finalFloats: Float32Array | null;
    fromCache: boolean;
    bytesFetched: number;
    status: number;
    event?: Http206Event;
  }> {
    const cid = capEntry.capsule_id;
    if (this.hotCapsules.has(cid)) {
      const cached = this.hotCapsules.get(cid)!;
      return {
        floats: cached.packedGpuFloats,
        midFloats: cached.midFloats,
        finalFloats: cached.finalFloats,
        fromCache: true,
        bytesFetched: 0,
        status: 200,
      };
    }

    const startByte = capEntry.byte_offset;
    const endByte = startByte + capEntry.record_bytes - 1;
    const rangeHeader = `bytes=${startByte}-${endByte}`;

    const candidateUrls: string[] = [];
    if (this.meta.local_http206_store_url) {
      candidateUrls.push(`${this.baseUrl}/${this.meta.local_http206_store_url}`);
    }
    if (this.meta.cloud_http206_store_url) {
      candidateUrls.push(this.meta.cloud_http206_store_url);
    }

    let lastErr: any = null;
    for (const storeUrl of candidateUrls) {
      const t0 = performance.now();
      try {
        const resp = await fetch(storeUrl, {
          method: 'GET',
          headers: { Range: rangeHeader },
        });
        if (resp.status !== 206 && resp.status !== 200) {
          throw new Error(`Unexpected HTTP ${resp.status} on Range ${rangeHeader}`);
        }
        let buf = await resp.arrayBuffer();
        if (resp.status === 200 && buf.byteLength > capEntry.record_bytes) {
          buf = buf.slice(startByte, startByte + capEntry.record_bytes);
        }
        const elapsedMs = Math.round((performance.now() - t0) * 100) / 100;
        const unpacked = this.unpackCapsuleInt4Bytes(buf);
        this.hotCapsules.set(cid, unpacked);
        this.totalHttp206Bytes += buf.byteLength;

        const evt: Http206Event = {
          capsuleId: cid,
          url: storeUrl,
          status: resp.status,
          range: rangeHeader,
          contentRange: resp.headers.get('Content-Range') || `${rangeHeader}/*`,
          bytes: buf.byteLength,
          kb: Math.round((buf.byteLength / 1024) * 100) / 100,
          latencyMs: elapsedMs,
          timestamp: new Date().toISOString(),
        };
        this.http206Events.unshift(evt);
        return {
          floats: unpacked.packedGpuFloats,
          midFloats: unpacked.midFloats,
          finalFloats: unpacked.finalFloats,
          fromCache: false,
          bytesFetched: buf.byteLength,
          status: resp.status,
          event: evt,
        };
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error(`Could not stream capsule ${cid} via HTTP 206`);
  }

  public unlearnCapsule(capsuleId: string): { capsuleId: string; zeroedBytes: number; unlearnMs: number } {
    const t0 = performance.now();
    let zeroedBytes = 0;
    const entry = this.hotCapsules.get(capsuleId);
    if (entry) {
      if (entry.midFloats) {
        entry.midFloats.fill(0.0);
        zeroedBytes += entry.midFloats.byteLength;
      }
      if (entry.finalFloats) {
        entry.finalFloats.fill(0.0);
        zeroedBytes += entry.finalFloats.byteLength;
      } else if (entry.packedGpuFloats) {
        entry.packedGpuFloats.fill(0.0);
        zeroedBytes += entry.packedGpuFloats.byteLength;
      }
      entry.rawBytesView.fill(0);
      zeroedBytes += entry.rawBytesView.byteLength;
      this.hotCapsules.delete(capsuleId);
    }
    const caps: CapsuleDirectoryEntry[] = this.meta.capsules || [];
    for (let i = 0; i < caps.length; i++) {
      if (caps[i].capsule_id === capsuleId) {
        const zOff = i * this.hiddenDim;
        this.zAnchorsFlat.fill(0.0, zOff, zOff + this.hiddenDim);
        zeroedBytes += this.hiddenDim * 4;
      }
    }
    this.unlearnedIds.add(capsuleId);
    this.lastActiveGates.delete(capsuleId);
    const ms = Math.round((performance.now() - t0) * 1000) / 1000;
    return { capsuleId, zeroedBytes, unlearnMs: ms };
  }

  public restoreCapsule(capsuleId: string): boolean {
    if (!this.unlearnedIds.has(capsuleId)) return false;
    const caps: CapsuleDirectoryEntry[] = this.meta.capsules || [];
    for (let i = 0; i < caps.length; i++) {
      if (caps[i].capsule_id === capsuleId) {
        const zOff = i * this.hiddenDim;
        this.zAnchorsFlat.set(this.zAnchorsBackup.subarray(zOff, zOff + this.hiddenDim), zOff);
      }
    }
    this.unlearnedIds.delete(capsuleId);
    return true;
  }

  private _buildSpanProbes(xInFlat: Float32Array, numUserTokens: number): Float32Array[] {
    const d = this.hiddenDim;
    const mu = this.muQueryLast;
    const probes: Float32Array[] = [];

    const addWindowProbe = (sIdx: number, eIdx: number) => {
      const count = eIdx - sIdx;
      if (count <= 0) return;
      const invCount = 1.0 / count;
      const vec = new Float32Array(d);
      let normSq = 0.0;
      for (let c = 0; c < d; c++) {
        let sum = 0.0;
        for (let t = sIdx; t < eIdx; t++) {
          sum += xInFlat[t * d + c];
        }
        const v = sum * invCount - mu[c];
        vec[c] = v;
        normSq += v * v;
      }
      const invNorm = 1.0 / Math.max(Math.sqrt(normSq), 1e-8);
      for (let c = 0; c < d; c++) vec[c] *= invNorm;

      // Apply Language-Subspace Orthogonal Projector `P_perp_lang(v) = v - U_lang (U_lang^T v)`
      if (this.isGoldenSliceV4 && this.uLangMid && this.kLang > 0) {
        const kLang = this.kLang;
        const uLang = this.uLangMid;
        const proj = new Float32Array(kLang);
        for (let k = 0; k < kLang; k++) {
          let acc = 0.0;
          for (let c = 0; c < d; c++) {
            acc += vec[c] * uLang[c * kLang + k];
          }
          proj[k] = acc;
        }
        let projNormSq = 0.0;
        for (let c = 0; c < d; c++) {
          const row = c * kLang;
          let sub = 0.0;
          for (let k = 0; k < kLang; k++) {
            sub += uLang[row + k] * proj[k];
          }
          const vp = vec[c] - sub;
          vec[c] = vp;
          projNormSq += vp * vp;
        }
        const invProjNorm = 1.0 / Math.max(Math.sqrt(projNormSq), 1e-8);
        for (let c = 0; c < d; c++) vec[c] *= invProjNorm;
      }
      probes.push(vec);
    };

    addWindowProbe(0, numUserTokens);

    if (numUserTokens >= this.spanWindow + 4) {
      const win = Math.min(this.spanWindow, numUserTokens);
      const startsSet = new Set<number>();
      for (let s = 0; s <= Math.max(0, numUserTokens - win); s += Math.max(1, this.spanStride)) {
        startsSet.add(s);
      }
      startsSet.add(numUserTokens - win);
      if (numUserTokens > win + 5 && numUserTokens - 5 - win > 0) {
        startsSet.add(numUserTokens - 5 - win);
      }
      const starts = Array.from(startsSet).sort((a, b) => a - b);
      for (const s of starts) {
        const e = Math.min(numUserTokens, s + win);
        if (e - s >= 4) {
          addWindowProbe(s, e);
        }
      }
    }
    return probes;
  }

  public async routeAndFetchActiveCapsules(
    xInFlat: Float32Array,
    numUserTokens: number
  ): Promise<{ active: ActiveCapsuleRuntime[]; turnBytesFetched: number; httpEvents: Http206Event[] }> {
    const caps: CapsuleDirectoryEntry[] = this.meta.capsules || [];
    this.lastCosines.clear();
    this.lastActiveGates.clear();
    if (caps.length === 0 || numUserTokens <= 0) {
      return { active: [], turnBytesFetched: 0, httpEvents: [] };
    }

    const d = this.hiddenDim;
    const probes = this._buildSpanProbes(xInFlat, numUserTokens);
    const gVals = new Float32Array(caps.length);
    const cosArr = new Float32Array(caps.length);
    const idToIdx = new Map<string, number>();
    let maxGate = 0.0;

    for (let i = 0; i < caps.length; i++) {
      const cap = caps[i];
      idToIdx.set(cap.capsule_id, i);
      if (this.unlearnedIds.has(cap.capsule_id)) {
        this.lastCosines.set(cap.capsule_id, 0.0);
        cosArr[i] = 0.0;
        gVals[i] = 0.0;
        continue;
      }
      const zBase = i * d;
      let maxCos = -2.0;
      for (const probe of probes) {
        let dot = 0.0;
        for (let c = 0; c < d; c++) {
          dot += probe[c] * this.zAnchorsFlat[zBase + c];
        }
        if (dot > maxCos) maxCos = dot;
      }
      cosArr[i] = maxCos;
      this.lastCosines.set(cap.capsule_id, maxCos);
      const ramp = Math.max(cap.ramp, 1e-6);
      const g = Math.min(1.0, Math.max(0.0, (maxCos - cap.tau) / ramp));
      gVals[i] = g;
      if (g > maxGate) maxGate = g;
    }

    if (maxGate <= 0.0) {
      return { active: [], turnBytesFetched: 0, httpEvents: [] };
    }

    const preGates = Float32Array.from(gVals);
    const activeBridges: number[] = [];
    const parentDiv = this.isGoldenSliceV4 ? 0.3 : 0.5;
    for (let i = 0; i < caps.length; i++) {
      const parents = caps[i].parent_capsules || [];
      if (parents.length === 0) continue;
      let parentMin = Infinity;
      let missing = false;
      for (const pid of parents) {
        const pIdx = idToIdx.get(pid);
        if (pIdx === undefined || preGates[pIdx] <= 0.0) {
          missing = true;
          break;
        }
        if (preGates[pIdx] < parentMin) parentMin = preGates[pIdx];
      }
      if (missing) {
        gVals[i] = 0.0;
        continue;
      }
      const effBg = preGates[i] * Math.min(1.0, parentMin / parentDiv);
      gVals[i] = effBg;
      if (effBg > 0.0) activeBridges.push(i);
    }

    if (this.isGoldenSliceV4 && this.winnerMargin > 0.0) {
      const primIndices: number[] = [];
      let bestPCos = -2.0;
      for (let i = 0; i < caps.length; i++) {
        const parents = caps[i].parent_capsules || [];
        if (gVals[i] > 0.0 && parents.length === 0) {
          primIndices.push(i);
          if (cosArr[i] > bestPCos) bestPCos = cosArr[i];
        }
      }
      if (primIndices.length > 1) {
        for (const idx of primIndices) {
          if (bestPCos - cosArr[idx] > this.winnerMargin) {
            gVals[idx] = 0.0;
          }
        }
      }
    }

    for (const bIdx of activeBridges) {
      const atten = Math.max(0.0, 1.0 - gVals[bIdx]);
      for (const pid of caps[bIdx].parent_capsules || []) {
        const pIdx = idToIdx.get(pid);
        if (pIdx !== undefined) {
          gVals[pIdx] = preGates[pIdx] * atten;
        }
      }
    }

    const active: ActiveCapsuleRuntime[] = [];
    let turnBytesFetched = 0;
    const httpEvents: Http206Event[] = [];
    for (let i = 0; i < caps.length; i++) {
      if (gVals[i] > 1e-6) {
        const fetched = await this.fetchCapsuleSliceHTTP206(caps[i]);
        turnBytesFetched += fetched.bytesFetched;
        if (fetched.event) httpEvents.push(fetched.event);
        this.lastActiveGates.set(caps[i].capsule_id, gVals[i]);
        active.push({
          capsuleId: caps[i].capsule_id,
          gate: gVals[i],
          floats: fetched.floats,
          midFloats: fetched.midFloats,
          finalFloats: fetched.finalFloats,
          fromCache: fetched.fromCache,
          meta: caps[i],
        });
      }
    }

    return { active, turnBytesFetched, httpEvents };
  }
}
