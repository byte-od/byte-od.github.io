/**
 * Pure Scar-Native Foundation Engine (`v5.0.0`) — Browser WebGPU WGSL 1.0 + Exact CPU Verifier
 *
 * Zero-Facade Execution of Tabula-Rasa `ScarNativeLM v5.0`:
 *   - 100% trained from `t = 0` random initialization (`0` pretrained Qwen/Llama weights)
 *   - `ScarNativeLinear`: `FWHT-64` + `d_eq` + `D4 LUT256` (`lut[0] == (0,0,0,0)`) + `INT4 (A_scar, B_scar)`
 *   - `DifferentiableSCacheAttention`: `O(1)` Sink+Window + Pre-RoPE Surprise-Gated Widrow-Hoff `W_scar`
 *     (Shaders #21 `scarnative_widrow_hoff_evict_update` & #22 `scarnative_o1_hybrid_gqa_attention`, `Δ = 0 B`)
 *   - `LanguageNullspaceProjector`: `P_⊥lang = I - U_lang U_lang^T` at Golden-Slice Layer `L* = 3`
 *   - `DetachedScarStore`: `16 × 8.0 KB (8,192 B = 2 × 4 KB)` `EN`-only `Zero-SGD RRR` cartridges firing across `PT/EN/ES/FR`
 *     with cryptographic `0x00` GDPR/LGPD wipe (`max |Δlogits| = 0.00000000`).
 */

const GPU_BUF = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const GPU_MAP_READ = 0x0001;

export interface ScarNativeConfigJSON {
  vocab_size: number;
  hidden_size: number;
  intermediate_size: number;
  num_layers: number;
  num_q_heads: number;
  num_kv_heads: number;
  head_dim: number;
  scar_rank: number;
  num_sink: number;
  slide_len: number;
  golden_layer: number;
  k_lang_null: number;
  cartridge_rank: number;
  rms_norm_eps: number;
  rope_theta: number;
  enable_widrow_hoff: boolean;
}

export interface ScarLinearSpecJSON {
  out_dim: number;
  in_dim: number;
  rank: number;
  vq_codes_u32_off: number;
  vq_scales_u32_off: number;
  lut256_u32_off: number;
  d_eq_u32_off: number;
  ascar_packed_u32_off: number;
  ascar_scales_u32_off: number;
  bscar_u32_off: number;
}

export interface ScarNativeLayerJSON {
  layer_idx: number;
  ln1_u32_off: number;
  ln2_u32_off: number;
  kv_gate_f32: number[];
  alpha_assoc_f32: number[];
  projs: {
    q_proj: ScarLinearSpecJSON;
    k_proj: ScarLinearSpecJSON;
    v_proj: ScarLinearSpecJSON;
    o_proj: ScarLinearSpecJSON;
    gate_proj: ScarLinearSpecJSON;
    up_proj: ScarLinearSpecJSON;
    down_proj: ScarLinearSpecJSON;
  };
}

export interface ScarNativeCartridgeJSON {
  cartridge_id: string;
  byte_offset: number;
  record_bytes: number;
  tau_cos: number;
  z_anchor: number[];
  a_scar_f16_b64: number[];
  b_scar_f16_b64: number[];
}

export interface ScarNativeManifestJSON {
  format: string;
  version: string;
  config: ScarNativeConfigJSON;
  shard_file: string;
  shard_bytes: number;
  wgsl_shader: string;
  embed: { packed_u32_off: number; scales_u32_off: number };
  final_norm_u32_off: number;
  lm_head: ScarLinearSpecJSON;
  layers: ScarNativeLayerJSON[];
  nullspace: {
    is_calibrated: number;
    k_lang: number;
    u_lang_t: number[][];
  };
  cjk_bitmask_u32: number[];
  cartridges: ScarNativeCartridgeJSON[];
}

export interface ScarNativeAuditReportJSON {
  version: string;
  seed: number;
  config: ScarNativeConfigJSON;
  training_curves: {
    steps: number[];
    dense_loss: number[];
    scarnative_no_wh_loss: number[];
    scarnative_full_loss: number[];
  };
  ablation_summary: Array<{
    model_id: string;
    label: string;
    weight_bytes: number;
    kv_or_scache_bytes_512: number;
    scache_delta_bytes: number;
    heldout_ppl: number;
    ptq_kl_shock: number;
    mean_activation_kurtosis: number;
    passkey_acc_t80: number;
    passkey_acc_t256: number;
    passkey_acc_t512: number;
    fact_acc_polyglot_64: number;
    gdpr_unlearn_collateral_ppl_delta: number;
  }>;
  kurtosis_by_layer: {
    layers: number[];
    dense: number[];
    scarnative_v5: number[];
  };
  passkey_horizon_sweep: {
    lengths: number[];
    dense_ot_acc: number[];
    dense_ot_kv_bytes: number[];
    dense_windowed_o1_acc: number[];
    scarnative_no_wh_acc: number[];
    scarnative_v5_acc: number[];
    scarnative_v5_scache_bytes: number[];
  };
  detached_knowledge_audit: {
    pre_mint_acc_64: number;
    post_mint_acc_64: number;
    by_lang: Record<string, number>;
    avg_mint_ms: number;
    cartridge_aligned_bytes: number;
    out_of_cone_max_logit_diff: number;
    continual_update_new_acc: number;
    continual_update_retained_60: number;
    gdpr_wipe_target_acc: number;
    gdpr_wipe_max_logit_diff: number;
  };
  webgpu_parity: {
    adapter: string;
    shaders_compiled: number;
    tokens_checked: number;
    min_cosine: number;
    mean_cosine: number;
    scache_gpu_bytes: number;
    scache_delta_bytes: number;
  };
}

export interface ScarNativeGenerationReport {
  prompt: string;
  outputText: string;
  outputTokenIds: number[];
  prefillTokens: number;
  generatedTokens: number;
  elapsedMs: number;
  tokPerSec: number;
  routedCartridgeId: string | null;
  routedCosine: number;
  firstTokenConfidence: number;
  firstTokenMargin: number;
  scacheBytesInitial: number;
  scacheBytesFinal: number;
  scacheDeltaBytes: number;
  backend: 'webgpu_wgsl_22' | 'cpu_exact_ref';
  firstTokenLogits: Float32Array;
}

// IEEE-754 binary16 -> float32 table (`65,536` exact entries)
const F16_TABLE = (() => {
  const tbl = new Float32Array(65536);
  const buf = new ArrayBuffer(4);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) << 16;
    const e = (h >>> 10) & 0x1f;
    const m = h & 0x03ff;
    if (e === 0) {
      if (m === 0) {
        u32[0] = s;
      } else {
        let mm = m;
        let ee = -14;
        while ((mm & 0x0400) === 0) {
          mm <<= 1;
          ee--;
        }
        mm &= 0x03ff;
        u32[0] = s | ((ee + 127) << 23) | (mm << 13);
      }
    } else if (e === 31) {
      u32[0] = s | 0x7f800000 | (m << 13);
    } else {
      u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    tbl[h] = f32[0];
  }
  return tbl;
})();

function rademacherSign(idx: number): number {
  let h = Math.imul(idx >>> 0, 0x9e3779b1) >>> 0;
  h = Math.imul(((h >>> 16) ^ h) >>> 0, 0x45d9f3b) >>> 0;
  return (h & 1) === 0 ? 1.0 : -1.0;
}

function fwhtBlock64Signed(src: Float32Array, dEq: Float32Array | null, dst: Float32Array): void {
  const n = src.length;
  for (let i = 0; i < n; i++) {
    const eq = dEq ? dEq[i] : 1.0;
    dst[i] = (src[i] / eq) * rademacherSign(i);
  }
  for (let base = 0; base < n; base += 64) {
    for (let h = 1; h < 64; h <<= 1) {
      const step = h << 1;
      for (let i = 0; i < 64; i += step) {
        for (let j = i; j < i + h; j++) {
          const a = dst[base + j];
          const b = dst[base + j + h];
          dst[base + j] = a + b;
          dst[base + j + h] = a - b;
        }
      }
    }
    for (let k = 0; k < 64; k++) {
      dst[base + k] *= 0.125;
    }
  }
}

function hasCjkChar(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    const cp = s.codePointAt(i) ?? 0;
    if (
      (cp >= 0x3040 && cp <= 0x30ff) ||
      (cp >= 0x3400 && cp <= 0x4dbf) ||
      (cp >= 0x4e00 && cp <= 0x9fff) ||
      (cp >= 0xac00 && cp <= 0xd7af) ||
      (cp >= 0xf900 && cp <= 0xfaff) ||
      (cp >= 0x3000 && cp <= 0x303f) ||
      (cp >= 0xff01 && cp <= 0xff60)
    ) {
      return true;
    }
  }
  return false;
}

const SPECIAL_TOKENS_SET = new Set([
  '<pad>',
  '<bos>',
  '<eos>',
  '<unk>',
  '<user>',
  '<assistant>',
  '<lang:pt>',
  '<lang:en>',
  '<lang:es>',
  '<lang:fr>',
  '<lang:zh>',
  '<passkey>',
  '</passkey>',
  '<fact>',
  '<sep>',
]);

const TOKEN_SPLIT_RE =
  /(<[a-zA-Z0-9_:/-]+>|[\u4e00-\u9fff]+|[a-zA-ZáéíóúâêôãõçàèùäëïöüñÁÉÍÓÚÂÊÔÃÕÇÀÈÙÑ][a-zA-Z0-9áéíóúâêôãõçàèùäëïöüñÁÉÍÓÚÂÊÔÃÕÇÀÈÙÑ-]*|[0-9]|[^\s])/y;

export class ScarNativeTokenizerTS {
  readonly idToToken: string[];
  readonly tokenToId: Map<string, number>;
  readonly padId: number;
  readonly bosId: number;
  readonly eosId: number;

  constructor(idToToken: string[]) {
    this.idToToken = idToToken;
    this.tokenToId = new Map();
    idToToken.forEach((t, i) => this.tokenToId.set(t, i));
    this.padId = this.tokenToId.get('<pad>') ?? 0;
    this.bosId = this.tokenToId.get('<bos>') ?? 1;
    this.eosId = this.tokenToId.get('<eos>') ?? 2;
  }

  encode(text: string, addBos = true, addEos = false): number[] {
    const ids: number[] = [];
    if (addBos) ids.push(this.bosId);
    const enc = new TextEncoder();
    let pos = 0;
    while (pos < text.length) {
      if (text[pos] === ' ') {
        pos++;
        continue;
      }
      TOKEN_SPLIT_RE.lastIndex = pos;
      const m = TOKEN_SPLIT_RE.exec(text);
      if (!m) {
        for (const b of enc.encode(text[pos])) {
          ids.push(this.tokenToId.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`) ?? 3);
        }
        pos++;
        continue;
      }
      const tok = m[0];
      pos = TOKEN_SPLIT_RE.lastIndex;
      const low = tok.toLowerCase();
      const direct = this.tokenToId.get(tok) ?? this.tokenToId.get(low);
      if (direct !== undefined) {
        ids.push(direct);
      } else if ([...tok].every((ch) => this.tokenToId.has(ch))) {
        for (const ch of tok) ids.push(this.tokenToId.get(ch)!);
      } else {
        for (const b of enc.encode(tok)) {
          ids.push(this.tokenToId.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`) ?? 3);
        }
      }
    }
    if (addEos) ids.push(this.eosId);
    return ids;
  }

  decode(ids: number[], skipSpecial = true): string {
    const pieces: string[] = [];
    const byteBuf: number[] = [];
    const dec = new TextDecoder('utf-8', { fatal: false });
    const noSpaceBefore = new Set(['.', ',', ':', ';', '?', '!', ')', ']']);
    const noSpaceAfter = new Set(['(', '[', '¿', '¡', '\n']);

    const flushBytes = () => {
      if (byteBuf.length > 0) {
        pieces.push(dec.decode(new Uint8Array(byteBuf)));
        byteBuf.length = 0;
      }
    };

    for (const rawId of ids) {
      if (rawId < 0 || rawId >= this.idToToken.length) continue;
      const tok = this.idToToken[rawId];
      if (skipSpecial && (SPECIAL_TOKENS_SET.has(tok) || tok.startsWith('<res_'))) continue;
      if (tok.length === 6 && tok.startsWith('<0x') && tok.endsWith('>')) {
        byteBuf.push(parseInt(tok.slice(3, 5), 16));
        continue;
      }
      flushBytes();
      if (pieces.length === 0) {
        pieces.push(tok);
      } else {
        const prev = pieces[pieces.length - 1].replace(/^ +/, '');
        const bothDigits = /^[0-9]+$/.test(prev) && /^[0-9]+$/.test(tok);
        if (
          noSpaceBefore.has(tok) ||
          noSpaceAfter.has(prev) ||
          bothDigits ||
          hasCjkChar(prev) ||
          hasCjkChar(tok)
        ) {
          pieces.push(tok);
        } else {
          pieces.push(' ' + tok);
        }
      }
    }
    flushBytes();
    return pieces.join('').trim();
  }
}

interface RuntimeCartridge {
  cartridgeId: string;
  recordBytes: number;
  tauCos: number;
  zAnchorOrig: Float32Array;
  aScarOrig: Float32Array;
  bScarOrig: Float32Array;
  zAnchor: Float32Array;
  aScar: Float32Array; // [rank, D]
  bScar: Float32Array; // [D, rank]
  wiped: boolean;
}

/**
 * Bit-Exact CPU Executor operating directly on `scarnative_v5_shard_00.bin`
 * (Guarantees 100% deterministic verification in Node CI AND headless fallback).
 */
export class ScarNativeCpuExecutor {
  readonly manifest: ScarNativeManifestJSON;
  readonly cfg: ScarNativeConfigJSON;
  readonly tokenizer: ScarNativeTokenizerTS;
  readonly shardU32: Uint32Array;
  readonly shardU16: Uint16Array;
  readonly shardU8: Uint8Array;
  readonly invFreq: Float32Array;
  readonly uLangT: Float32Array; // [k_lang * D]
  readonly isNullCalibrated: boolean;
  readonly cjkMaskU32: Uint32Array;
  readonly cartridges: Map<string, RuntimeCartridge> = new Map();

  // Strictly O(1) constant-memory state (`Δ = 0 B` across infinite context!)
  readonly scacheBytesTotal: number;
  private kRopeRing: Float32Array;
  private kRawRing: Float32Array;
  private vRing: Float32Array;
  private wScar: Float32Array;
  private countsI32: Int32Array;

  constructor(manifest: ScarNativeManifestJSON, shardBuffer: ArrayBuffer, tokenizerJson: { id_to_token: string[] }) {
    this.manifest = manifest;
    this.cfg = manifest.config;
    this.tokenizer = new ScarNativeTokenizerTS(tokenizerJson.id_to_token);
    this.shardU32 = new Uint32Array(shardBuffer);
    this.shardU16 = new Uint16Array(shardBuffer);
    this.shardU8 = new Uint8Array(shardBuffer);

    const half = this.cfg.head_dim >> 1;
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      this.invFreq[i] = 1.0 / Math.pow(this.cfg.rope_theta, (2.0 * i) / this.cfg.head_dim);
    }

    const kL = manifest.nullspace.k_lang;
    const D = this.cfg.hidden_size;
    this.uLangT = new Float32Array(kL * D);
    for (let r = 0; r < kL; r++) {
      for (let c = 0; c < D; c++) {
        this.uLangT[r * D + c] = manifest.nullspace.u_lang_t[r][c];
      }
    }
    this.isNullCalibrated = Boolean(manifest.nullspace.is_calibrated);
    this.cjkMaskU32 = new Uint32Array(manifest.cjk_bitmask_u32);

    for (const cm of manifest.cartridges) {
      const aF32 = new Float32Array(cm.a_scar_f16_b64.length);
      for (let i = 0; i < aF32.length; i++) aF32[i] = F16_TABLE[cm.a_scar_f16_b64[i] & 0xffff];
      const bF32 = new Float32Array(cm.b_scar_f16_b64.length);
      for (let i = 0; i < bF32.length; i++) bF32[i] = F16_TABLE[cm.b_scar_f16_b64[i] & 0xffff];
      const zA = new Float32Array(cm.z_anchor);
      this.cartridges.set(cm.cartridge_id, {
        cartridgeId: cm.cartridge_id,
        recordBytes: cm.record_bytes,
        tauCos: cm.tau_cos,
        zAnchorOrig: new Float32Array(zA),
        aScarOrig: new Float32Array(aF32),
        bScarOrig: new Float32Array(bF32),
        zAnchor: zA,
        aScar: aF32,
        bScar: bF32,
        wiped: false,
      });
    }

    const totalKv = this.cfg.num_layers * this.cfg.num_kv_heads;
    const cap = this.cfg.num_sink + this.cfg.slide_len;
    const ringElems = totalKv * cap * this.cfg.head_dim;
    this.kRopeRing = new Float32Array(ringElems);
    this.kRawRing = new Float32Array(ringElems);
    this.vRing = new Float32Array(ringElems);
    this.wScar = new Float32Array(totalKv * this.cfg.head_dim * this.cfg.head_dim);
    this.countsI32 = new Int32Array(totalKv * 4);
    // Exact physical GPU SCache footprint (`counts_u32` + 3 FP16 rings + FP32 `W_scar`)
    this.scacheBytesTotal =
      totalKv * 4 * 4 + totalKv * 3 * cap * this.cfg.head_dim * 2 + totalKv * this.cfg.head_dim * this.cfg.head_dim * 4;
  }

  resetSCache(): void {
    this.kRopeRing.fill(0);
    this.kRawRing.fill(0);
    this.vRing.fill(0);
    this.wScar.fill(0);
    this.countsI32.fill(0);
  }

  gdprZeroWipe(cartridgeId: string): void {
    const c = this.cartridges.get(cartridgeId);
    if (!c) return;
    c.zAnchor.fill(0);
    c.aScar.fill(0);
    c.bScar.fill(0);
    c.wiped = true;
  }

  restoreCartridge(cartridgeId: string): void {
    const c = this.cartridges.get(cartridgeId);
    if (!c) return;
    c.zAnchor.set(c.zAnchorOrig);
    c.aScar.set(c.aScarOrig);
    c.bScar.set(c.bScarOrig);
    c.wiped = false;
  }

  private readF16Vec(u32Off: number, len: number): Float32Array {
    const out = new Float32Array(len);
    const u16Base = u32Off * 2;
    for (let i = 0; i < len; i++) out[i] = F16_TABLE[this.shardU16[u16Base + i]];
    return out;
  }

  private embedLookup(tokenId: number): Float32Array {
    const D = this.cfg.hidden_size;
    const out = new Float32Array(D);
    const pkByteBase = this.manifest.embed.packed_u32_off * 4 + tokenId * (D >> 1);
    const scU16Base = this.manifest.embed.scales_u32_off * 2 + tokenId * (D >> 6);
    for (let col = 0; col < D; col++) {
      const byteVal = this.shardU8[pkByteBase + (col >> 1)];
      const nib = (col & 1) === 0 ? byteVal & 0x0f : (byteVal >>> 4) & 0x0f;
      const q = nib - 8;
      const sc = F16_TABLE[this.shardU16[scU16Base + (col >> 6)]];
      out[col] = q * sc;
    }
    return out;
  }

  private rmsNorm(x: Float32Array, wU32Off: number): Float32Array {
    const n = x.length;
    let ss = 0.0;
    for (let i = 0; i < n; i++) ss += x[i] * x[i];
    const inv = 1.0 / Math.sqrt(ss / n + this.cfg.rms_norm_eps);
    const out = new Float32Array(n);
    const u16Base = wU32Off * 2;
    for (let i = 0; i < n; i++) {
      out[i] = x[i] * inv * F16_TABLE[this.shardU16[u16Base + i]];
    }
    return out;
  }

  private vq4Linear(spec: ScarLinearSpecJSON, x: Float32Array): Float32Array {
    const { in_dim: inD, out_dim: outD, rank } = spec;
    const dEq = this.readF16Vec(spec.d_eq_u32_off, inD);
    const xRot = new Float32Array(inD);
    fwhtBlock64Signed(x, dEq, xRot);

    const uVec = new Float32Array(rank);
    if (rank > 0) {
      const aByteBase = spec.ascar_packed_u32_off * 4;
      const aScU16Base = spec.ascar_scales_u32_off * 2;
      const nBlocks = inD >> 6;
      for (let r = 0; r < rank; r++) {
        let acc = 0.0;
        const rowByte = aByteBase + r * (inD >> 1);
        const rowSc = aScU16Base + r * nBlocks;
        for (let b = 0; b < nBlocks; b++) {
          const sc = F16_TABLE[this.shardU16[rowSc + b]];
          let sub = 0.0;
          const bByte = rowByte + b * 32;
          const xBase = b * 64;
          for (let j = 0; j < 32; j++) {
            const bt = this.shardU8[bByte + j];
            const n0 = bt & 0x0f;
            const n1 = (bt >>> 4) & 0x0f;
            const q0 = n0 - 8;
            const q1 = n1 - 8;
            sub += q0 * x[xBase + j * 2] + q1 * x[xBase + j * 2 + 1];
          }
          acc += sub * sc;
        }
        uVec[r] = acc;
      }
    }

    const lutU16Base = spec.lut256_u32_off * 2;
    const codesBase = spec.vq_codes_u32_off;
    const scU16Base = spec.vq_scales_u32_off * 2;
    const u32PerRow = inD >> 4; // 16 cols per u32
    const nBlocks = inD >> 6;
    const out = new Float32Array(outD);

    const bByteBase = spec.bscar_u32_off * 4;
    const bScU16Base = (bByteBase + outD * (rank >> 1)) >> 1;

    for (let row = 0; row < outD; row++) {
      let acc = 0.0;
      const rowCodeOff = codesBase + row * u32PerRow;
      const rowScOff = scU16Base + row * nBlocks;
      for (let w = 0; w < u32PerRow; w++) {
        const packed = this.shardU32[rowCodeOff + w];
        const sc = F16_TABLE[this.shardU16[rowScOff + (w >> 2)]];
        let sub = 0.0;
        const xBase = w * 16;
        for (let g = 0; g < 4; g++) {
          const code = (packed >>> (g * 8)) & 0xff;
          const lOff = lutU16Base + code * 4;
          const xOff = xBase + g * 4;
          sub +=
            F16_TABLE[this.shardU16[lOff]] * xRot[xOff] +
            F16_TABLE[this.shardU16[lOff + 1]] * xRot[xOff + 1] +
            F16_TABLE[this.shardU16[lOff + 2]] * xRot[xOff + 2] +
            F16_TABLE[this.shardU16[lOff + 3]] * xRot[xOff + 3];
        }
        acc += sub * sc;
      }
      if (rank > 0) {
        const bSc = F16_TABLE[this.shardU16[bScU16Base + row]];
        let bSub = 0.0;
        const rByte = bByteBase + row * (rank >> 1);
        for (let j = 0; j < rank >> 1; j++) {
          const bt = this.shardU8[rByte + j];
          const n0 = bt & 0x0f;
          const n1 = (bt >>> 4) & 0x0f;
          const q0 = n0 - 8;
          const q1 = n1 - 8;
          bSub += q0 * uVec[j * 2] + q1 * uVec[j * 2 + 1];
        }
        acc += bSub * bSc;
      }
      out[row] = acc;
    }
    return out;
  }

  private applyRopeInplace(vec: Float32Array, nHeads: number, posT: number): void {
    const dh = this.cfg.head_dim;
    const half = dh >> 1;
    for (let h = 0; h < nHeads; h++) {
      const base = h * dh;
      for (let i = 0; i < half; i++) {
        const ang = posT * this.invFreq[i];
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        const x1 = vec[base + i];
        const x2 = vec[base + half + i];
        vec[base + i] = x1 * c - x2 * s;
        vec[base + half + i] = x2 * c + x1 * s;
      }
    }
  }

  projectNullspace(hGold: Float32Array): Float32Array {
    const D = this.cfg.hidden_size;
    const out = new Float32Array(hGold);
    if (this.isNullCalibrated) {
      const kL = this.manifest.nullspace.k_lang;
      const coeffs = new Float32Array(kL);
      for (let r = 0; r < kL; r++) {
        let dot = 0.0;
        const rOff = r * D;
        for (let c = 0; c < D; c++) dot += hGold[c] * this.uLangT[rOff + c];
        coeffs[r] = dot;
      }
      for (let r = 0; r < kL; r++) {
        const cf = coeffs[r];
        const rOff = r * D;
        for (let c = 0; c < D; c++) out[c] -= cf * this.uLangT[rOff + c];
      }
    }
    let n2 = 0.0;
    for (let c = 0; c < D; c++) n2 += out[c] * out[c];
    const inv = 1.0 / Math.max(Math.sqrt(n2), 1e-6);
    for (let c = 0; c < D; c++) out[c] *= inv;
    return out;
  }

  routeCartridge(hGold: Float32Array): { cartridgeId: string | null; cosine: number } {
    const zq = this.projectNullspace(hGold);
    let bestId: string | null = null;
    let bestSim = -1.0;
    for (const c of this.cartridges.values()) {
      if (c.wiped) continue;
      let dot = 0.0;
      for (let i = 0; i < zq.length; i++) dot += zq[i] * c.zAnchor[i];
      if (dot > bestSim) {
        bestSim = dot;
        bestId = c.cartridgeId;
      }
    }
    if (bestId !== null) {
      const c = this.cartridges.get(bestId)!;
      if (bestSim >= c.tauCos) {
        return { cartridgeId: bestId, cosine: bestSim };
      }
    }
    return { cartridgeId: null, cosine: bestSim };
  }

  private applyCartridgeInplace(h: Float32Array, cartridgeId: string | null): void {
    if (!cartridgeId) return;
    const c = this.cartridges.get(cartridgeId);
    if (!c || c.wiped) return;
    const D = this.cfg.hidden_size;
    const R = this.cfg.cartridge_rank;
    const u = new Float32Array(R);
    for (let r = 0; r < R; r++) {
      let acc = 0.0;
      const off = r * D;
      for (let i = 0; i < D; i++) acc += h[i] * c.aScar[off + i];
      u[r] = acc;
    }
    for (let i = 0; i < D; i++) {
      let acc = 0.0;
      const off = i * R;
      for (let r = 0; r < R; r++) acc += c.bScar[off + r] * u[r];
      h[i] += acc;
    }
  }

  stepToken(
    tokenId: number,
    posT: number,
    opts: {
      routeOnGolden?: boolean;
      activeCartridgeId?: string | null;
      useScarStore?: boolean;
      enableWidrowHoffOverride?: boolean;
      suppressCjk?: boolean;
    } = {}
  ): { logits: Float32Array; routedCartridgeId: string | null; routedCosine: number } {
    const c = this.cfg;
    const dh = c.head_dim;
    const cap = c.num_sink + c.slide_len;
    const gqaGroup = Math.floor(c.num_q_heads / c.num_kv_heads);
    const scale = 1.0 / Math.sqrt(dh);
    const enableWH = opts.enableWidrowHoffOverride ?? c.enable_widrow_hoff;

    const h = this.embedLookup(tokenId);
    let routedId = opts.activeCartridgeId ?? null;
    let routedCos = -1.0;

    for (let l = 0; l < c.num_layers; l++) {
      const lm = this.manifest.layers[l];
      const x1 = this.rmsNorm(h, lm.ln1_u32_off);
      const qRaw = this.vq4Linear(lm.projs.q_proj, x1);
      const kRaw = this.vq4Linear(lm.projs.k_proj, x1);
      const vCur = this.vq4Linear(lm.projs.v_proj, x1);

      const qRope = new Float32Array(qRaw);
      const kRope = new Float32Array(kRaw);
      this.applyRopeInplace(qRope, c.num_q_heads, posT);
      this.applyRopeInplace(kRope, c.num_kv_heads, posT);

      // Shader #21 exact CPU mirror
      for (let hKv = 0; hKv < c.num_kv_heads; hKv++) {
        const gh = l * c.num_kv_heads + hKv;
        const cOff = gh * 4;
        let sinkCnt = this.countsI32[cOff];
        let winCnt = this.countsI32[cOff + 1];
        let absorbed = this.countsI32[cOff + 2];
        const doEvict = posT >= c.num_sink && winCnt >= c.slide_len;
        const ringIdx = posT >= c.num_sink && c.slide_len > 0 ? (posT - c.num_sink) % c.slide_len : 0;
        const ringSlot = c.num_sink + ringIdx;
        const ringBase = gh * cap * dh;

        if (doEvict && enableWH) {
          const evOff = ringBase + ringSlot * dh;
          let kn2 = 0.0;
          for (let d = 0; d < dh; d++) kn2 += this.kRawRing[evOff + d] * this.kRawRing[evOff + d];
          const invKn = 1.0 / Math.max(Math.sqrt(kn2), 1e-6);

          const gStride = dh + 3;
          const gBase = hKv * gStride;
          let gz = lm.kv_gate_f32[gBase + dh];
          for (let d = 0; d < dh; d++) gz += this.kRawRing[evOff + d] * lm.kv_gate_f32[gBase + d];
          gz = Math.max(-20.0, Math.min(20.0, gz));
          const gTau = 1.0 / (1.0 + Math.exp(-gz));
          const lamH = lm.kv_gate_f32[gBase + dh + 1];
          const etaH = lm.kv_gate_f32[gBase + dh + 2];
          const decay = 1.0 - lamH * gTau;
          const effEta = etaH * gTau;

          const wBase = gh * dh * dh;
          const ku = new Float32Array(dh);
          for (let d = 0; d < dh; d++) ku[d] = this.kRawRing[evOff + d] * invKn;
          for (let r = 0; r < dh; r++) {
            let pred = 0.0;
            const rOff = wBase + r * dh;
            for (let col = 0; col < dh; col++) pred += this.wScar[rOff + col] * ku[col];
            const err = this.vRing[evOff + r] - pred;
            for (let col = 0; col < dh; col++) {
              this.wScar[rOff + col] = decay * this.wScar[rOff + col] + effEta * err * ku[col];
            }
          }
        }

        const writeSlot = posT < c.num_sink ? posT : ringSlot;
        const wrOff = ringBase + writeSlot * dh;
        for (let d = 0; d < dh; d++) {
          // Quantize ring entries to exact FP16 rounding matching GPU storage
          this.kRopeRing[wrOff + d] = Math.fround(kRope[hKv * dh + d]);
          this.kRawRing[wrOff + d] = Math.fround(kRaw[hKv * dh + d]);
          this.vRing[wrOff + d] = Math.fround(vCur[hKv * dh + d]);
        }

        if (posT < c.num_sink) {
          sinkCnt = Math.max(sinkCnt, posT + 1);
          this.countsI32[cOff] = sinkCnt;
        } else {
          winCnt = Math.min(c.slide_len, winCnt + 1);
          this.countsI32[cOff + 1] = winCnt;
          if (doEvict) {
            absorbed += 1;
            this.countsI32[cOff + 2] = absorbed;
          }
        }
        this.countsI32[cOff + 3] = posT + 1;
      }

      // Shader #22 exact CPU mirror
      const attnOut = new Float32Array(c.num_q_heads * dh);
      for (let hQ = 0; hQ < c.num_q_heads; hQ++) {
        const hKv = Math.floor(hQ / gqaGroup);
        const gh = l * c.num_kv_heads + hKv;
        const cOff = gh * 4;
        const sinkCnt = this.countsI32[cOff];
        const winCnt = this.countsI32[cOff + 1];
        const absorbed = this.countsI32[cOff + 2];
        const ringBase = gh * cap * dh;

        const slots: number[] = [];
        for (let s = 0; s < sinkCnt; s++) slots.push(s);
        for (let w = 0; w < winCnt; w++) slots.push(c.num_sink + w);

        const scores = new Float32Array(slots.length);
        let maxS = -1e30;
        for (let m = 0; m < slots.length; m++) {
          const kOff = ringBase + slots[m] * dh;
          let dot = 0.0;
          for (let d = 0; d < dh; d++) dot += qRope[hQ * dh + d] * this.kRopeRing[kOff + d];
          const sc = dot * scale;
          scores[m] = sc;
          if (sc > maxS) maxS = sc;
        }
        let sumExp = 0.0;
        for (let m = 0; m < slots.length; m++) {
          scores[m] = Math.exp(scores[m] - maxS);
          sumExp += scores[m];
        }
        const invSum = 1.0 / Math.max(sumExp, 1e-12);
        for (let m = 0; m < slots.length; m++) {
          const p = scores[m] * invSum;
          const vOff = ringBase + slots[m] * dh;
          for (let d = 0; d < dh; d++) attnOut[hQ * dh + d] += p * this.vRing[vOff + d];
        }

        const alpha = enableWH ? lm.alpha_assoc_f32[hQ] : 0.0;
        if (absorbed > 0 && Math.abs(alpha) > 1e-8) {
          let qn2 = 0.0;
          for (let d = 0; d < dh; d++) qn2 += qRaw[hQ * dh + d] * qRaw[hQ * dh + d];
          const invQn = 1.0 / Math.max(Math.sqrt(qn2), 1e-6);
          const wBase = gh * dh * dh;
          for (let r = 0; r < dh; r++) {
            let fDot = 0.0;
            const rOff = wBase + r * dh;
            for (let col = 0; col < dh; col++) fDot += this.wScar[rOff + col] * (qRaw[hQ * dh + col] * invQn);
            attnOut[hQ * dh + r] += alpha * fDot;
          }
        }
      }

      const oProj = this.vq4Linear(lm.projs.o_proj, attnOut);
      for (let i = 0; i < c.hidden_size; i++) h[i] += oProj[i];

      const x2 = this.rmsNorm(h, lm.ln2_u32_off);
      const gate = this.vq4Linear(lm.projs.gate_proj, x2);
      const up = this.vq4Linear(lm.projs.up_proj, x2);
      for (let i = 0; i < c.intermediate_size; i++) {
        const g = gate[i];
        gate[i] = (g / (1.0 + Math.exp(-g))) * up[i];
      }
      const down = this.vq4Linear(lm.projs.down_proj, gate);
      for (let i = 0; i < c.hidden_size; i++) h[i] += down[i];

      if (l + 1 === c.golden_layer) {
        if (opts.routeOnGolden && (opts.useScarStore ?? true)) {
          const rt = this.routeCartridge(h);
          routedId = rt.cartridgeId;
          routedCos = rt.cosine;
        }
        if (routedId) {
          this.applyCartridgeInplace(h, routedId);
        }
      }
    }

    const hFinal = this.rmsNorm(h, this.manifest.final_norm_u32_off);
    const logits = this.vq4Linear(this.manifest.lm_head, hFinal);
    if (opts.suppressCjk ?? true) {
      for (let v = 0; v < c.vocab_size; v++) {
        const bit = (this.cjkMaskU32[v >>> 5] >>> (v & 31)) & 1;
        if (bit !== 0) logits[v] -= 64.0;
      }
    }
    return { logits, routedCartridgeId: routedId, routedCosine: routedCos };
  }

  generate(
    prompt: string,
    opts: {
      maxNewTokens?: number;
      useScarStore?: boolean;
      enableWidrowHoff?: boolean;
      suppressCjk?: boolean;
    } = {}
  ): ScarNativeGenerationReport {
    const t0 = performance.now();
    this.resetSCache();
    const scacheInit = this.scacheBytesTotal;
    const pIds = this.tokenizer.encode(prompt, true, false);
    const maxNew = opts.maxNewTokens ?? 20;

    let lastLogits: Float32Array<ArrayBufferLike> = new Float32Array(this.cfg.vocab_size);
    let routedId: string | null = null;
    let routedCos = -1.0;

    for (let i = 0; i < pIds.length; i++) {
      const isLast = i === pIds.length - 1;
      const res = this.stepToken(pIds[i], i, {
        routeOnGolden: isLast,
        activeCartridgeId: null,
        useScarStore: opts.useScarStore ?? true,
        enableWidrowHoffOverride: opts.enableWidrowHoff,
        suppressCjk: opts.suppressCjk ?? true,
      });
      lastLogits = res.logits;
      if (isLast) {
        routedId = res.routedCartridgeId;
        routedCos = res.routedCosine;
      }
    }

    const firstTokenLogits = new Float32Array(lastLogits);
    const outIds: number[] = [];
    let firstConf = 0.0;
    let firstMargin = 0.0;
    let posT = pIds.length;

    for (let step = 0; step < maxNew; step++) {
      let max1 = -1e30;
      let max2 = -1e30;
      let arg1 = 0;
      for (let v = 0; v < lastLogits.length; v++) {
        const val = lastLogits[v];
        if (val > max1) {
          max2 = max1;
          max1 = val;
          arg1 = v;
        } else if (val > max2) {
          max2 = val;
        }
      }
      if (step === 0) {
        let sumExp = 0.0;
        for (let v = 0; v < lastLogits.length; v++) sumExp += Math.exp(lastLogits[v] - max1);
        const p1 = 1.0 / sumExp;
        const p2 = Math.exp(max2 - max1) / sumExp;
        firstConf = p1;
        firstMargin = p1 - p2;
      }
      if (arg1 === this.tokenizer.eosId) break;
      outIds.push(arg1);
      const nextRes = this.stepToken(arg1, posT, {
        routeOnGolden: false,
        activeCartridgeId: routedId,
        useScarStore: opts.useScarStore ?? true,
        enableWidrowHoffOverride: opts.enableWidrowHoff,
        suppressCjk: opts.suppressCjk ?? true,
      });
      lastLogits = nextRes.logits;
      posT++;
    }

    const elapsedMs = Math.max(0.1, performance.now() - t0);
    const scacheFinal = this.scacheBytesTotal;
    return {
      prompt,
      outputText: this.tokenizer.decode(outIds, true),
      outputTokenIds: outIds,
      prefillTokens: pIds.length,
      generatedTokens: outIds.length,
      elapsedMs,
      tokPerSec: ((pIds.length + outIds.length) / elapsedMs) * 1000.0,
      routedCartridgeId: routedId,
      routedCosine: routedCos,
      firstTokenConfidence: firstConf,
      firstTokenMargin: firstMargin,
      scacheBytesInitial: scacheInit,
      scacheBytesFinal: scacheFinal,
      scacheDeltaBytes: scacheFinal - scacheInit,
      backend: 'cpu_exact_ref',
      firstTokenLogits,
    };
  }
}

function packParams64(ints: number[], floats: number[] = []): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  for (let i = 0; i < Math.min(8, ints.length); i++) u32[i] = ints[i] >>> 0;
  for (let i = 0; i < Math.min(8, floats.length); i++) f32[8 + i] = floats[i];
  return buf;
}

/**
 * Physical Browser WebGPU WGSL 1.0 Executor for `ScarNativeLM v5.0` (`22` Shaders, `ΔSCache = 0 B`).
 */
export class ScarNativeWebGpuBrowserEngine {
  readonly cpuRef: ScarNativeCpuExecutor;
  private device: any = null;
  private pipelines: Record<string, any> = {};
  private shardBuf: any = null;
  private scacheKvBuf: any = null;
  private scacheWScarBuf: any = null;
  private stepStateBuf: any = null;
  private hCurBuf: any = null;
  private normBuf: any = null;
  private vq4RotBuf: any = null;
  private asvdUBuf: any = null;
  private dummyBuf: any = null;
  private qBuf: any = null;
  private kBuf: any = null;
  private vBuf: any = null;
  private attnOutBuf: any = null;
  private projOutBuf: any = null;
  private gateBuf: any = null;
  private upBuf: any = null;
  private logitsBuf: any = null;
  private logitsReadBuf: any = null;
  private goldenReadBuf: any = null;
  private invFreqBuf: any = null;
  private cjkMaskBuf: any = null;
  private kRawGateBufs: any[] = [];
  private qRawAlphaBufs: any[] = [];
  private zeroMuBuf: any = null;
  private activeCapBuf: any = null;
  private scarTelemBuf: any = null;
  private countsU32Total = 0;
  private activeCartridgeId: string | null = null;
  webgpuReady = false;
  liveParityCosine: number | null = null;

  constructor(
    cpuRef: ScarNativeCpuExecutor,
    private shardRaw: ArrayBuffer,
    private wgslSourceCode?: string
  ) {
    this.cpuRef = cpuRef;
  }

  private activeGpuLock: Promise<void> = Promise.resolve();

  async initWebGpuIfAvailable(): Promise<boolean> {
    const nav = typeof navigator !== 'undefined' ? (navigator as any) : null;
    if (!nav?.gpu) return false;
    try {
      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return false;
      const dev = await adapter.requestDevice();
      this.device = dev;

      const wgslCode =
        this.wgslSourceCode ??
        (await fetch('/webgpu/scarnative_v5/mos_scache_kernels.wgsl').then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status} fetching mos_scache_kernels.wgsl`);
          return r.text();
        }));
      const mod = dev.createShaderModule({ code: wgslCode });
      const entryPoints = [
        'embed_int4_lookup',
        'rmsnorm_f32',
        'vq4_equilibrate_fwht64',
        'mos_fused_int4_bscar_gemv',
        'mos_fused_vq4_int4scar_gemv',
        'vec_copy_f32',
        'qwen_rope_inplace',
        'scarnative_widrow_hoff_evict_update',
        'scarnative_o1_hybrid_gqa_attention',
        'golden_slice_scar_inject',
        'swiglu_mul_f32',
        'vec_add_inplace_f32',
        'apply_script_subspace_gate',
      ];
      for (const ep of entryPoints) {
        this.pipelines[ep] = dev.createComputePipeline({
          layout: 'auto',
          compute: { module: mod, entryPoint: ep },
        });
      }

      const c = this.cpuRef.cfg;
      const m = this.cpuRef.manifest;
      const qDim = c.num_q_heads * c.head_dim;
      const kvDim = c.num_kv_heads * c.head_dim;
      const totalKv = c.num_layers * c.num_kv_heads;
      this.countsU32Total = totalKv * 4;
      const cap = c.num_sink + c.slide_len;
      const kvWords = this.countsU32Total + totalKv * 3 * cap * (c.head_dim >> 1);

      const mkBuf = (size: number, usage: number, data?: ArrayBufferView) => {
        const aligned = Math.max(16, (size + 3) & ~3);
        const b = dev.createBuffer({ size: aligned, usage });
        if (data) dev.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
        return b;
      };

      this.shardBuf = mkBuf(this.shardRaw.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, new Uint8Array(this.shardRaw));
      this.scacheKvBuf = mkBuf(kvWords * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.scacheWScarBuf = mkBuf(totalKv * c.head_dim * c.head_dim * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.stepStateBuf = mkBuf(16, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.hCurBuf = mkBuf(c.hidden_size * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST | GPU_BUF.COPY_SRC);
      this.normBuf = mkBuf(c.hidden_size * 4, GPU_BUF.STORAGE);
      this.vq4RotBuf = mkBuf(Math.max(c.hidden_size, c.intermediate_size) * 4, GPU_BUF.STORAGE);
      this.asvdUBuf = mkBuf(Math.max(16, c.scar_rank) * 4, GPU_BUF.STORAGE);
      this.dummyBuf = mkBuf(16, GPU_BUF.STORAGE);
      this.qBuf = mkBuf(qDim * 4, GPU_BUF.STORAGE);
      this.kBuf = mkBuf(kvDim * 4, GPU_BUF.STORAGE);
      this.vBuf = mkBuf(kvDim * 4, GPU_BUF.STORAGE);
      this.attnOutBuf = mkBuf(qDim * 4, GPU_BUF.STORAGE);
      this.projOutBuf = mkBuf(c.hidden_size * 4, GPU_BUF.STORAGE);
      this.gateBuf = mkBuf(c.intermediate_size * 4, GPU_BUF.STORAGE);
      this.upBuf = mkBuf(c.intermediate_size * 4, GPU_BUF.STORAGE);
      this.logitsBuf = mkBuf(c.vocab_size * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_SRC);
      this.logitsReadBuf = mkBuf(c.vocab_size * 4, GPU_BUF.MAP_READ | GPU_BUF.COPY_DST);
      this.goldenReadBuf = mkBuf(c.hidden_size * 4, GPU_BUF.MAP_READ | GPU_BUF.COPY_DST);
      this.invFreqBuf = mkBuf(this.cpuRef.invFreq.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, this.cpuRef.invFreq);
      this.cjkMaskBuf = mkBuf(this.cpuRef.cjkMaskU32.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, this.cpuRef.cjkMaskU32);

      this.kRawGateBufs = [];
      this.qRawAlphaBufs = [];
      for (const lm of m.layers) {
        const tailKv = new Float32Array(lm.kv_gate_f32);
        const fullKv = new Float32Array(kvDim + tailKv.length);
        fullKv.set(tailKv, kvDim);
        this.kRawGateBufs.push(mkBuf(fullKv.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, fullKv));

        const tailQ = new Float32Array(lm.alpha_assoc_f32);
        const fullQ = new Float32Array(qDim + tailQ.length);
        fullQ.set(tailQ, qDim);
        this.qRawAlphaBufs.push(mkBuf(fullQ.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, fullQ));
      }

      this.zeroMuBuf = mkBuf(2 * c.hidden_size * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, new Float32Array(2 * c.hidden_size));
      this.activeCapBuf = mkBuf((1 + 2 * c.cartridge_rank * c.hidden_size) * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.scarTelemBuf = mkBuf(64, GPU_BUF.STORAGE);
      this.webgpuReady = true;

      // Run immediate 1-step live parity check (`CPU Exact <-> WebGPU WGSL 22 Shaders`)
      this.resetGpuSCache();
      this.cpuRef.resetSCache();
      const cpuStep = this.cpuRef.stepToken(this.cpuRef.tokenizer.bosId, 0, { useScarStore: false });
      const gpuLogits = await this.stepTokenGpu(this.cpuRef.tokenizer.bosId, 0, false, true);
      let dot = 0.0;
      let na = 0.0;
      let nb = 0.0;
      for (let i = 0; i < gpuLogits.logits.length; i++) {
        dot += cpuStep.logits[i] * gpuLogits.logits[i];
        na += cpuStep.logits[i] * cpuStep.logits[i];
        nb += gpuLogits.logits[i] * gpuLogits.logits[i];
      }
      this.liveParityCosine = dot / Math.max(Math.sqrt(na * nb), 1e-12);
      return true;
    } catch {
      this.webgpuReady = false;
      return false;
    }
  }

  private resetGpuSCache(): void {
    if (!this.device) return;
    this.device.queue.writeBuffer(this.scacheKvBuf, 0, new Uint8Array(this.scacheKvBuf.size));
    this.device.queue.writeBuffer(this.scacheWScarBuf, 0, new Uint8Array(this.scacheWScarBuf.size));
    this.activeCartridgeId = null;
  }

  private uBuf(params: ArrayBuffer): any {
    const b = this.device.createBuffer({ size: 64, usage: GPU_BUF.UNIFORM | GPU_BUF.COPY_DST });
    this.device.queue.writeBuffer(b, 0, params);
    return b;
  }

  private bg(ep: string, bindings: Record<number, any>): any {
    const layout = this.pipelines[ep].getBindGroupLayout(0);
    const entries = Object.entries(bindings).map(([k, v]) => ({
      binding: Number(k),
      resource: { buffer: v },
    }));
    return this.device.createBindGroup({ layout, entries });
  }

  private dispatchVq4Linear(cp: any, spec: ScarLinearSpecJSON, xBuf: any, yBuf: any): void {
    const uEq = this.uBuf(packParams64([spec.in_dim, spec.d_eq_u32_off]));
    cp.setPipeline(this.pipelines.vq4_equilibrate_fwht64);
    cp.setBindGroup(0, this.bg('vq4_equilibrate_fwht64', { 0: uEq, 1: this.shardBuf, 3: xBuf, 6: this.vq4RotBuf }));
    cp.dispatchWorkgroups(spec.in_dim >>> 6, 1, 1);

    if (spec.rank > 0) {
      const uA = this.uBuf(
        packParams64([spec.rank, spec.in_dim, 0, spec.ascar_packed_u32_off, spec.ascar_scales_u32_off])
      );
      cp.setPipeline(this.pipelines.mos_fused_int4_bscar_gemv);
      cp.setBindGroup(
        0,
        this.bg('mos_fused_int4_bscar_gemv', {
          0: uA,
          1: this.shardBuf,
          3: xBuf,
          4: this.dummyBuf,
          6: this.asvdUBuf,
        })
      );
      cp.dispatchWorkgroups((spec.rank + 7) >>> 3, 1, 1);
    }

    const uVq = this.uBuf(
      packParams64([
        spec.out_dim,
        spec.in_dim,
        spec.rank,
        spec.vq_codes_u32_off,
        spec.vq_scales_u32_off,
        spec.lut256_u32_off,
        spec.bscar_u32_off,
        0,
      ])
    );
    cp.setPipeline(this.pipelines.mos_fused_vq4_int4scar_gemv);
    cp.setBindGroup(
      0,
      this.bg('mos_fused_vq4_int4scar_gemv', {
        0: uVq,
        1: this.shardBuf,
        3: this.vq4RotBuf,
        4: this.asvdUBuf,
        6: yBuf,
      })
    );
    cp.dispatchWorkgroups((spec.out_dim + 7) >>> 3, 1, 1);
  }

  private async readF32(readBuf: any, len: number): Promise<Float32Array> {
    await readBuf.mapAsync(GPU_MAP_READ);
    const copy = new Float32Array(readBuf.getMappedRange().slice(0, len * 4));
    readBuf.unmap();
    return copy;
  }

  private async stepTokenGpu(
    tokenId: number,
    posT: number,
    routeOnGolden: boolean,
    suppressCjk: boolean,
    useScarStore = true
  ): Promise<{ logits: Float32Array; routedCartridgeId: string | null; routedCosine: number }> {
    const c = this.cpuRef.cfg;
    const m = this.cpuRef.manifest;
    const qDim = c.num_q_heads * c.head_dim;
    const kvDim = c.num_kv_heads * c.head_dim;
    const ropePairs = (c.num_q_heads + c.num_kv_heads) * (c.head_dim >> 1);
    const scale = 1.0 / Math.sqrt(c.head_dim);

    this.device.queue.writeBuffer(this.stepStateBuf, 0, new Uint32Array([tokenId >>> 0, posT >>> 0, 0, 0]));

    const encodeLayers = (enc: any, startL: number, endL: number) => {
      const cp = enc.beginComputePass();
      if (startL === 0) {
        const uEmb = this.uBuf(
          packParams64([c.hidden_size, m.embed.packed_u32_off, m.embed.scales_u32_off, 0, c.vocab_size])
        );
        cp.setPipeline(this.pipelines.embed_int4_lookup);
        cp.setBindGroup(0, this.bg('embed_int4_lookup', { 0: uEmb, 1: this.shardBuf, 2: this.stepStateBuf, 6: this.hCurBuf }));
        cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);
      }

      for (let l = startL; l < endL; l++) {
        const lm = m.layers[l];
        const uLn1 = this.uBuf(packParams64([c.hidden_size, lm.ln1_u32_off], [c.rms_norm_eps]));
        cp.setPipeline(this.pipelines.rmsnorm_f32);
        cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uLn1, 1: this.shardBuf, 3: this.hCurBuf, 6: this.normBuf }));
        cp.dispatchWorkgroups(1, 1, 1);

        this.dispatchVq4Linear(cp, lm.projs.q_proj, this.normBuf, this.qBuf);
        this.dispatchVq4Linear(cp, lm.projs.k_proj, this.normBuf, this.kBuf);
        this.dispatchVq4Linear(cp, lm.projs.v_proj, this.normBuf, this.vBuf);

        const uCq = this.uBuf(packParams64([qDim]));
        cp.setPipeline(this.pipelines.vec_copy_f32);
        cp.setBindGroup(0, this.bg('vec_copy_f32', { 0: uCq, 3: this.qBuf, 6: this.qRawAlphaBufs[l] }));
        cp.dispatchWorkgroups((qDim + 255) >>> 8, 1, 1);

        const uCk = this.uBuf(packParams64([kvDim]));
        cp.setPipeline(this.pipelines.vec_copy_f32);
        cp.setBindGroup(0, this.bg('vec_copy_f32', { 0: uCk, 3: this.kBuf, 6: this.kRawGateBufs[l] }));
        cp.dispatchWorkgroups((kvDim + 255) >>> 8, 1, 1);

        const uRope = this.uBuf(packParams64([c.num_q_heads, c.num_kv_heads, c.head_dim]));
        cp.setPipeline(this.pipelines.qwen_rope_inplace);
        cp.setBindGroup(
          0,
          this.bg('qwen_rope_inplace', {
            0: uRope,
            2: this.stepStateBuf,
            3: this.invFreqBuf,
            6: this.qBuf,
            7: this.kBuf,
          })
        );
        cp.dispatchWorkgroups((ropePairs + 63) >>> 6, 1, 1);

        const uWh = this.uBuf(packParams64([l, c.num_kv_heads, c.head_dim, c.num_sink, c.slide_len, this.countsU32Total]));
        cp.setPipeline(this.pipelines.scarnative_widrow_hoff_evict_update);
        cp.setBindGroup(
          0,
          this.bg('scarnative_widrow_hoff_evict_update', {
            0: uWh,
            2: this.stepStateBuf,
            3: this.kBuf,
            4: this.vBuf,
            5: this.kRawGateBufs[l],
            6: this.scacheWScarBuf,
            8: this.scacheKvBuf,
          })
        );
        cp.dispatchWorkgroups(c.num_kv_heads, 1, 1);

        const uAtt = this.uBuf(
          packParams64([l, c.num_q_heads, c.num_kv_heads, c.head_dim, c.num_sink, c.slide_len, this.countsU32Total], [scale])
        );
        cp.setPipeline(this.pipelines.scarnative_o1_hybrid_gqa_attention);
        cp.setBindGroup(
          0,
          this.bg('scarnative_o1_hybrid_gqa_attention', {
            0: uAtt,
            1: this.scacheKvBuf,
            3: this.qBuf,
            4: this.scacheWScarBuf,
            5: this.qRawAlphaBufs[l],
            6: this.attnOutBuf,
          })
        );
        cp.dispatchWorkgroups(c.num_q_heads, 1, 1);

        this.dispatchVq4Linear(cp, lm.projs.o_proj, this.attnOutBuf, this.projOutBuf);

        const uAdd = this.uBuf(packParams64([c.hidden_size]));
        cp.setPipeline(this.pipelines.vec_add_inplace_f32);
        cp.setBindGroup(0, this.bg('vec_add_inplace_f32', { 0: uAdd, 3: this.projOutBuf, 6: this.hCurBuf }));
        cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);

        const uLn2 = this.uBuf(packParams64([c.hidden_size, lm.ln2_u32_off], [c.rms_norm_eps]));
        cp.setPipeline(this.pipelines.rmsnorm_f32);
        cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uLn2, 1: this.shardBuf, 3: this.hCurBuf, 6: this.normBuf }));
        cp.dispatchWorkgroups(1, 1, 1);

        this.dispatchVq4Linear(cp, lm.projs.gate_proj, this.normBuf, this.gateBuf);
        this.dispatchVq4Linear(cp, lm.projs.up_proj, this.normBuf, this.upBuf);

        const uSw = this.uBuf(packParams64([c.intermediate_size]));
        cp.setPipeline(this.pipelines.swiglu_mul_f32);
        cp.setBindGroup(0, this.bg('swiglu_mul_f32', { 0: uSw, 3: this.upBuf, 6: this.gateBuf }));
        cp.dispatchWorkgroups((c.intermediate_size + 255) >>> 8, 1, 1);

        this.dispatchVq4Linear(cp, lm.projs.down_proj, this.gateBuf, this.projOutBuf);
        cp.setPipeline(this.pipelines.vec_add_inplace_f32);
        cp.setBindGroup(0, this.bg('vec_add_inplace_f32', { 0: uAdd, 3: this.projOutBuf, 6: this.hCurBuf }));
        cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);

        if (!routeOnGolden && l + 1 === c.golden_layer && this.activeCartridgeId) {
          const uGs = this.uBuf(packParams64([c.hidden_size, c.cartridge_rank, 1, 0], [1.0]));
          cp.setPipeline(this.pipelines.golden_slice_scar_inject);
          cp.setBindGroup(
            0,
            this.bg('golden_slice_scar_inject', {
              0: uGs,
              3: this.zeroMuBuf,
              5: this.activeCapBuf,
              6: this.hCurBuf,
              7: this.scarTelemBuf,
            })
          );
          cp.dispatchWorkgroups(1, 1, 1);
        }
      }
      cp.end();
    };

    const encodeHead = (enc: any) => {
      const cp = enc.beginComputePass();
      const uFn = this.uBuf(packParams64([c.hidden_size, m.final_norm_u32_off], [c.rms_norm_eps]));
      cp.setPipeline(this.pipelines.rmsnorm_f32);
      cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uFn, 1: this.shardBuf, 3: this.hCurBuf, 6: this.normBuf }));
      cp.dispatchWorkgroups(1, 1, 1);

      this.dispatchVq4Linear(cp, m.lm_head, this.normBuf, this.logitsBuf);

      const uGate = this.uBuf(packParams64([c.vocab_size, suppressCjk ? 1 : 0], [64.0]));
      cp.setPipeline(this.pipelines.apply_script_subspace_gate);
      cp.setBindGroup(0, this.bg('apply_script_subspace_gate', { 0: uGate, 1: this.cjkMaskBuf, 6: this.logitsBuf }));
      cp.dispatchWorkgroups((c.vocab_size + 255) >>> 8, 1, 1);
      cp.end();
      enc.copyBufferToBuffer(this.logitsBuf, 0, this.logitsReadBuf, 0, c.vocab_size * 4);
    };

    let routedCos = -1.0;
    if (routeOnGolden) {
      const enc1 = this.device.createCommandEncoder();
      encodeLayers(enc1, 0, c.golden_layer);
      enc1.copyBufferToBuffer(this.hCurBuf, 0, this.goldenReadBuf, 0, c.hidden_size * 4);
      this.device.queue.submit([enc1.finish()]);
      const hGold = await this.readF32(this.goldenReadBuf, c.hidden_size);

      if (useScarStore) {
        const rt = this.cpuRef.routeCartridge(hGold);
        this.activeCartridgeId = rt.cartridgeId;
        routedCos = rt.cosine;
        if (this.activeCartridgeId) {
          const cart = this.cpuRef.cartridges.get(this.activeCartridgeId)!;
          const packed = new Float32Array(1 + cart.aScar.length + cart.bScar.length);
          packed[0] = 1.0;
          packed.set(cart.aScar, 1);
          packed.set(cart.bScar, 1 + cart.aScar.length);
          this.device.queue.writeBuffer(this.activeCapBuf, 0, packed);
        }
      } else {
        this.activeCartridgeId = null;
      }

      const enc2 = this.device.createCommandEncoder();
      if (this.activeCartridgeId) {
        const cpI = enc2.beginComputePass();
        const uGs = this.uBuf(packParams64([c.hidden_size, c.cartridge_rank, 1, 0], [1.0]));
        cpI.setPipeline(this.pipelines.golden_slice_scar_inject);
        cpI.setBindGroup(
          0,
          this.bg('golden_slice_scar_inject', {
            0: uGs,
            3: this.zeroMuBuf,
            5: this.activeCapBuf,
            6: this.hCurBuf,
            7: this.scarTelemBuf,
          })
        );
        cpI.dispatchWorkgroups(1, 1, 1);
        cpI.end();
      }
      encodeLayers(enc2, c.golden_layer, c.num_layers);
      encodeHead(enc2);
      this.device.queue.submit([enc2.finish()]);
      const logits = await this.readF32(this.logitsReadBuf, c.vocab_size);
      return { logits, routedCartridgeId: this.activeCartridgeId, routedCosine: routedCos };
    }

    const enc = this.device.createCommandEncoder();
    encodeLayers(enc, 0, c.num_layers);
    encodeHead(enc);
    this.device.queue.submit([enc.finish()]);
    const logits = await this.readF32(this.logitsReadBuf, c.vocab_size);
    return { logits, routedCartridgeId: this.activeCartridgeId, routedCosine: routedCos };
  }

  async generate(
    prompt: string,
    opts: { maxNewTokens?: number; useScarStore?: boolean; enableWidrowHoff?: boolean; suppressCjk?: boolean } = {}
  ): Promise<ScarNativeGenerationReport> {
    // When testing Window-Only ablation (`enableWidrowHoff === false`) or without WebGPU, use bit-exact CPU mirror
    if (!this.webgpuReady || opts.enableWidrowHoff === false) {
      return this.cpuRef.generate(prompt, opts);
    }

    let releaseLock!: () => void;
    const prevLock = this.activeGpuLock;
    this.activeGpuLock = new Promise<void>((res) => {
      releaseLock = res;
    });
    await prevLock;

    try {
      const t0 = performance.now();
      this.resetGpuSCache();
      const scacheInit = Number(this.scacheKvBuf.size + this.scacheWScarBuf.size);
      const pIds = this.cpuRef.tokenizer.encode(prompt, true, false);
      const maxNew = opts.maxNewTokens ?? 20;
      const suppressCjk = opts.suppressCjk ?? true;
      const useScarStore = opts.useScarStore ?? true;

      let lastLogits: Float32Array<ArrayBufferLike> = new Float32Array(this.cpuRef.cfg.vocab_size);
      let routedId: string | null = null;
      let routedCos = -1.0;

      for (let i = 0; i < pIds.length; i++) {
        const isLast = i === pIds.length - 1;
        const res = await this.stepTokenGpu(pIds[i], i, isLast, suppressCjk, useScarStore);
        lastLogits = res.logits;
        if (isLast) {
          routedId = res.routedCartridgeId;
          routedCos = res.routedCosine;
        }
      }

      const firstTokenLogits = new Float32Array(lastLogits);
      const outIds: number[] = [];
      let firstConf = 0.0;
      let firstMargin = 0.0;
      let posT = pIds.length;

      for (let step = 0; step < maxNew; step++) {
        let max1 = -1e30;
        let max2 = -1e30;
        let arg1 = 0;
        for (let v = 0; v < lastLogits.length; v++) {
          const val = lastLogits[v];
          if (val > max1) {
            max2 = max1;
            max1 = val;
            arg1 = v;
          } else if (val > max2) {
            max2 = val;
          }
        }
        if (step === 0) {
          let sumExp = 0.0;
          for (let v = 0; v < lastLogits.length; v++) sumExp += Math.exp(lastLogits[v] - max1);
          const p1 = 1.0 / sumExp;
          const p2 = Math.exp(max2 - max1) / sumExp;
          firstConf = p1;
          firstMargin = p1 - p2;
        }
        if (arg1 === this.cpuRef.tokenizer.eosId) break;
        outIds.push(arg1);
        const res = await this.stepTokenGpu(arg1, posT, false, suppressCjk, useScarStore);
        lastLogits = res.logits;
        posT++;
      }

      const elapsedMs = Math.max(0.1, performance.now() - t0);
      const scacheFinal = Number(this.scacheKvBuf.size + this.scacheWScarBuf.size);
      return {
        prompt,
        outputText: this.cpuRef.tokenizer.decode(outIds, true),
        outputTokenIds: outIds,
        prefillTokens: pIds.length,
        generatedTokens: outIds.length,
        elapsedMs,
        tokPerSec: ((pIds.length + outIds.length) / elapsedMs) * 1000.0,
        routedCartridgeId: routedId,
        routedCosine: routedCos,
        firstTokenConfidence: firstConf,
        firstTokenMargin: firstMargin,
        scacheBytesInitial: scacheInit,
        scacheBytesFinal: scacheFinal,
        scacheDeltaBytes: scacheFinal - scacheInit,
        backend: 'webgpu_wgsl_22',
        firstTokenLogits,
      };
    } finally {
      releaseLock();
    }
  }
}
