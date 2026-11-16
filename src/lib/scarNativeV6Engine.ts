/**
 * Pure Scar-Native Foundation Engine (`v6.0.0`) — Browser 26-Shader WebGPU WGSL 1.0 + Bit-Exact CPU Executor
 *
 * Zero-Facade Execution of Tabula-Rasa `ScarNativeLMV6`:
 *   - Trained 100% from `t = 0` random initialization (`0` pretrained external weights, `0` `nn.Linear`)
 *   - Native Polyglot Byte-Level BPE Tokenizer (`4,096` tokens, `0 <unk>`, individual-digit splitting, `<abstain>`)
 *   - Shared-Stage Universal `D4-LUT256 VQ4` Substrates + Sequential Stage-Streaming (`1` stage window in VRAM)
 *   - Native `Mixture of Scars` (`MoS`) Top-`k` Directional `ReLU` `INT4` Scar Experts (Shaders #23 & #24)
 *   - Dual-Tier Hierarchical `O(1)` `SCache` (`RoPE Sinks + Sliding Ring + Surprise Micro-Pages + Widrow-Hoff W_scar`,
 *     Shaders #25 & #26, `Δ = +0 B` across infinite context)
 *   - Orthogonal Procrustes `R_lang in SO(D)` + `P_⊥lang` Folded `HTTP 206` `.scarstore` Cartridge Routing (`mos-digest`)
 *     with Single-Entity and Document-Level Cryptographic `0x00` GDPR Wipe (`max |Δlogits| = 0.00000000`).
 */

const GPU_BUF = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const GPU_MAP_READ = 0x0001;

export type V6PolyglotLang = 'pt' | 'en' | 'es' | 'fr';
export const V6_LANGS: readonly V6PolyglotLang[] = ['pt', 'en', 'es', 'fr'] as const;

export interface ScarNativeV6ConfigJSON {
  vocab_size: number;
  embed_rank: number;
  hidden_size: number;
  intermediate_size: number;
  num_layers: number;
  num_stages: number;
  num_q_heads: number;
  num_kv_heads: number;
  head_dim: number;
  id_scar_rank: number;
  head_scar_rank: number;
  num_mos_experts: number;
  mos_top_k: number;
  mos_expert_rank: number;
  num_sink: number;
  slide_len: number;
  num_micro_pages: number;
  micro_page_theta: number;
  wy_chunk_size: number;
  golden_layer: number;
  k_lang_null: number;
  cartridge_rank: number;
  rms_norm_eps: number;
  rope_theta: number;
  enable_widrow_hoff: boolean;
  enable_micro_pages: boolean;
}

export interface ScarV6LinearSpecJSON {
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

export interface ScarV6MoSExpertSpecJSON {
  num_experts: number;
  top_k: number;
  rank: number;
  router_packed_u32_off: number;
  router_scales_u32_off: number;
  a_exp_packed_u32_off: number;
  a_exp_scales_u32_off: number;
  b_exp_packed_u32_off: number;
  b_exp_scales_u32_off: number;
}

export interface ScarV6StageMetaJSON {
  stage_idx: number;
  byte_offset: number;
  byte_length: number;
  start_u32: number;
  layer_start: number;
  layer_end: number;
}

export interface ScarV6LayerMetaJSON {
  layer_idx: number;
  stage_idx: number;
  ln1_u32_off: number;
  ln2_u32_off: number;
  kv_gate_f32: number[];
  alpha_assoc_f32: number[];
  projs: {
    q_proj: ScarV6LinearSpecJSON;
    k_proj: ScarV6LinearSpecJSON;
    v_proj: ScarV6LinearSpecJSON;
    o_proj: ScarV6LinearSpecJSON;
    gate_proj: ScarV6LinearSpecJSON;
    up_proj: ScarV6LinearSpecJSON;
    down_proj: ScarV6LinearSpecJSON;
  };
  mos_experts: ScarV6MoSExpertSpecJSON | null;
}

export interface ScarV6CartridgeMetaJSON {
  cartridge_id: string;
  doc_id: string;
  entity: string;
  attribute_en: string;
  value_str: string;
  attributes_polyglot: Record<string, string>;
  byte_offset: number;
  record_bytes: number;
  http_206_range: string;
  tau_cos: number;
}

export interface ScarNativeV6ManifestJSON {
  format: string;
  version: string;
  config: ScarNativeV6ConfigJSON;
  shard_file: string;
  shard_bytes: number;
  head_tail_bytes: number;
  max_stage_bytes: number;
  wgsl_shader: string;
  procrustes_file: string;
  procrustes_bytes: number;
  scarstore_file: string;
  scarstore_bytes: number;
  embed: {
    code_packed_u32_off: number;
    code_scales_u32_off: number;
    embed_rank: number;
    up_proj: ScarV6LinearSpecJSON;
  };
  final_norm_u32_off: number;
  lm_head: ScarV6LinearSpecJSON;
  stages: ScarV6StageMetaJSON[];
  layers: ScarV6LayerMetaJSON[];
  nullspace: {
    is_calibrated: number;
    langs: string[];
    k_lang: number;
  };
  cjk_bitmask_u32: number[];
  cartridges: ScarV6CartridgeMetaJSON[];
}

export interface ScarNativeBPETokenizerJSON {
  format: string;
  vocab_size: number;
  special_tokens: string[];
  token_bytes_hex: string[];
  merges: Array<[number, number, number]>;
}

export interface ScarV6ArchitecturalBudgetJSON {
  shared_stage_bases_bytes: number;
  one_stage_base_bytes: number;
  layer_identity_scars_bytes: number;
  mos_experts_all_bytes: number;
  mos_experts_lru_bytes: number;
  factorized_vocab_bytes: number;
  scache_o1_constant_bytes: number;
  scache_delta_bytes: number;
  active_cartridges_bytes: number;
  total_resident_bytes: number;
  total_resident_mib: number;
  total_resident_gib: number;
  total_lru_resident_gib: number;
  stage_streaming_peak_bytes: number;
  stage_streaming_peak_mib: number;
  stage_streaming_peak_gib: number;
  matched_dense_bf16_weights_bytes: number;
  matched_dense_bf16_total_with_kv_bytes: number;
  matched_dense_bf16_total_gib: number;
  weight_compression_vs_dense_bf16: number;
  total_vram_reduction_vs_dense_128k: number;
}

export interface ScarNativeV6AuditReportJSON {
  version: string;
  seed: number;
  training_hardware: string;
  elapsed_seconds: number;
  config: ScarNativeV6ConfigJSON;
  chunkwise_wy_benchmark: {
    device: string;
    gpu_name: string;
    seq_len: number;
    batch_size: number;
    chunk_size: number;
    chunkwise_wy_fwd_bwd_ms: number;
    sequential_fwd_bwd_ms: number;
    speedup_x: number;
    forward_read_cosine: number;
    backward_grad_cosine: number;
  };
  tokenizer_summary: {
    format: string;
    vocab_size: number;
    active_merges: number;
    unk_rate: number;
    abstain_token_id: number;
  };
  training_curves: {
    steps: number[];
    dense_loss: number[];
    scarnative_no_dualtier_loss: number[];
    scarnative_v6_loss: number[];
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
    passkey_acc_t85: number;
    passkey_acc_t290: number;
    passkey_acc_t560: number;
    fact_acc_polyglot_96: number;
    epistemic_abstain_acc_16: number;
    gdpr_unlearn_collateral_ppl_delta: number;
  }>;
  kurtosis_by_layer: {
    layers: number[];
    dense: number[];
    scarnative_v6: number[];
  };
  mos_expert_load_by_layer: number[][];
  stage_streaming_invariance: {
    peak_single_stage_base_bytes: number;
    max_logit_diff_vs_resident: number;
  };
  passkey_horizon_sweep: {
    lengths: number[];
    dense_ot_acc: number[];
    dense_ot_kv_bytes: number[];
    dense_windowed_o1_acc: number[];
    scarnative_no_dualtier_acc: number[];
    scarnative_v6_acc: number[];
    scarnative_v6_scache_bytes: number[];
  };
  mos_digest_audit: {
    markdown_documents_compiled: number;
    total_cartridges: number;
    pre_mint_fact_acc_96: number;
    pre_mint_abstain_rate_96: number;
    post_mint_fact_acc_96: number;
    by_lang_24: Record<string, number>;
    avg_mint_ms: number;
    cartridge_aligned_bytes: number;
    total_24_cartridges_bytes: number;
    out_of_cone_false_routes: number;
    out_of_cone_max_logit_diff: number;
    continual_update_new_acc_4: number;
    continual_update_retained_92: number;
    gdpr_wiped_doc_id: string;
    gdpr_wiped_cartridges_count: number;
    gdpr_wipe_leaks_24: number;
    gdpr_wipe_abstain_reversion_24: number;
    gdpr_wipe_max_logit_diff: number;
  };
  epistemic_abstention_audit: {
    heldout_queries: number;
    scarnative_v6_abstain_rate: number;
    dense_entangled_abstain_rate: number;
    sample_bleached_hair_scarnative_v6: string;
    sample_bleached_hair_dense_baseline: string;
  };
  vram_architectural_budgets: {
    v6_12l_d512: ScarV6ArchitecturalBudgetJSON;
    v6_frontier_96l_d8192: ScarV6ArchitecturalBudgetJSON;
  };
  webgpu_bundle: {
    shard_bytes: number;
    head_tail_bytes: number;
    max_stage_window_bytes: number;
    stage_streaming_weight_gpu_bytes: number;
    scache_total_gpu_bytes: number;
  };
  webgpu_parity?: {
    adapter: string;
    shaders_compiled: number;
    tokens_checked: number;
    min_cosine: number;
    mean_cosine: number;
    stage_streaming_max_diff: number;
    fr_cartridge_cosine: number;
    gdpr_gpu_max_diff: number;
    scache_gpu_bytes: number;
    scache_delta_bytes: number;
  };
}

export interface ScarNativeV6GenerationReport {
  prompt: string;
  outputText: string;
  outputTextWithTags: string;
  outputTokenIds: number[];
  abstained: boolean;
  prefillTokens: number;
  generatedTokens: number;
  elapsedMs: number;
  tokPerSec: number;
  routedCartridgeId: string | null;
  routedDocId: string | null;
  routedCosine: number;
  resolvedLang: V6PolyglotLang;
  firstTokenConfidence: number;
  firstTokenMargin: number;
  scacheBytesInitial: number;
  scacheBytesFinal: number;
  scacheDeltaBytes: number;
  activeWeightGpuBytes: number;
  streamStages: boolean;
  promotedMicroPagesTotal: number;
  selectedExpertsByLayer: number[][];
  backend: 'webgpu_wgsl_26' | 'webgpu_wgsl_26_stream' | 'cpu_exact_ref' | 'cpu_exact_ref_stream';
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

const F32_CONV_BUF = new ArrayBuffer(4);
const F32_CONV_VIEW = new Float32Array(F32_CONV_BUF);
const U32_CONV_VIEW = new Uint32Array(F32_CONV_BUF);

/**
 * Exact IEEE-754 float32 -> binary16 -> float32 rounding matching WGSL `pack2x16float` / `unpack2x16float`.
 */
function roundToF16(val: number): number {
  const clamped = Math.max(-65500.0, Math.min(65500.0, val));
  F32_CONV_VIEW[0] = clamped;
  const bits = U32_CONV_VIEW[0];
  const sign = (bits >>> 16) & 0x8000;
  let exp = ((bits >>> 23) & 0xff) - 127 + 15;
  let mant = bits & 0x007fffff;

  let h16: number;
  if (exp <= 0) {
    if (exp < -10) {
      h16 = sign;
    } else {
      mant |= 0x00800000;
      const shift = 14 - exp;
      const halfBit = 1 << (shift - 1);
      const lowMask = halfBit - 1;
      let m16 = mant >>> shift;
      if ((mant & halfBit) !== 0 && ((mant & lowMask) !== 0 || (m16 & 1) !== 0)) {
        m16++;
      }
      h16 = sign | m16;
    }
  } else if (exp >= 31) {
    h16 = sign | 0x7bff;
  } else {
    const roundBit = 0x00001000;
    const lowMask = 0x00000fff;
    let m16 = mant >>> 13;
    if ((mant & roundBit) !== 0 && ((mant & lowMask) !== 0 || (m16 & 1) !== 0)) {
      m16++;
      if (m16 === 0x0400) {
        m16 = 0;
        exp++;
      }
    }
    h16 = exp >= 31 ? sign | 0x7bff : sign | (exp << 10) | m16;
  }
  return F16_TABLE[h16 & 0xffff];
}

function rademacherSign(idx: number): number {
  let h = Math.imul(idx >>> 0, 0x9e3779b1) >>> 0;
  h = Math.imul(((h >>> 16) ^ h) >>> 0, 0x45d9f3b) >>> 0;
  return (h & 1) === 0 ? 1.0 : -1.0;
}

function fwhtBlock64Signed(src: Float32Array, dEq: Float32Array, dst: Float32Array): void {
  const n = src.length;
  for (let i = 0; i < n; i++) {
    dst[i] = (src[i] / dEq[i]) * rademacherSign(i);
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

function hexToBytes(hex: string): Uint8Array {
  if (!hex) return new Uint8Array(0);
  const out = new Uint8Array(hex.length >>> 1);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHexKey(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i++) {
    s += bytes[i].toString(16).padStart(2, '0');
  }
  return s;
}

// Exact mirror of `_V6_PRETOK_RE` in `scar_native/v6_bpe_tokenizer.py`
const V6_PRETOK_RE =
  /(<(?:pad|bos|eos|unk|user|assistant|lang:pt|lang:en|lang:es|lang:fr|lang:zh|passkey|\/passkey|fact|abstain|sep)>|[0-9]| ?[A-Za-zÀ-ÖØ-öø-ÿ_]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+| ?[^\sA-Za-zÀ-ÖØ-öø-ÿ0-9<]+|<|\s+)/g;

/**
 * Deterministic Byte-Level BPE Tokenizer (`vocab_size = 4096`, `0 <unk>`, lossless UTF-8 round-trip).
 */
export class ScarNativeBPETokenizerTS {
  readonly vocabSize: number;
  readonly specialTokens: string[];
  readonly specialToId: Map<string, number>;
  readonly tokenBytes: Uint8Array[];
  readonly byteOffset: number;
  readonly padId: number;
  readonly bosId: number;
  readonly eosId: number;
  readonly unkId: number;
  readonly abstainId: number;
  private readonly mergeRanks: Map<number, { rank: number; merged: number }>;
  private readonly bpeCache: Map<string, number[]> = new Map();

  constructor(json: ScarNativeBPETokenizerJSON) {
    this.vocabSize = json.vocab_size;
    this.specialTokens = [...json.special_tokens];
    this.byteOffset = this.specialTokens.length;
    this.specialToId = new Map();
    this.specialTokens.forEach((t, idx) => this.specialToId.set(t, idx));
    this.tokenBytes = json.token_bytes_hex.map(hexToBytes);
    while (this.tokenBytes.length < this.vocabSize) {
      this.tokenBytes.push(new Uint8Array(0));
    }

    this.mergeRanks = new Map();
    json.merges.forEach(([l, r, m], rank) => {
      this.mergeRanks.set(((l & 0xffff) << 16) | (r & 0xffff), { rank, merged: m });
    });

    this.padId = this.specialToId.get('<pad>') ?? 0;
    this.bosId = this.specialToId.get('<bos>') ?? 1;
    this.eosId = this.specialToId.get('<eos>') ?? 2;
    this.unkId = this.specialToId.get('<unk>') ?? 3;
    this.abstainId = this.specialToId.get('<abstain>') ?? 14;
  }

  private bpeEncodeBytes(rawBytes: Uint8Array): number[] {
    const key = bytesToHexKey(rawBytes);
    const cached = this.bpeCache.get(key);
    if (cached) return cached;

    let seq = Array.from(rawBytes, (b) => this.byteOffset + b);
    if (seq.length >= 2 && this.mergeRanks.size > 0) {
      while (seq.length >= 2) {
        let bestRank = 1_000_000_000;
        let bestPos = -1;
        let bestMerged = -1;
        for (let i = 0; i < seq.length - 1; i++) {
          const info = this.mergeRanks.get(((seq[i] & 0xffff) << 16) | (seq[i + 1] & 0xffff));
          if (info !== undefined && info.rank < bestRank) {
            bestRank = info.rank;
            bestPos = i;
            bestMerged = info.merged;
          }
        }
        if (bestPos < 0) break;

        const lWin = seq[bestPos];
        const rWin = seq[bestPos + 1];
        const nxt: number[] = [];
        let k = 0;
        while (k < seq.length) {
          if (k + 1 < seq.length && seq[k] === lWin && seq[k + 1] === rWin) {
            nxt.push(bestMerged);
            k += 2;
          } else {
            nxt.push(seq[k]);
            k += 1;
          }
        }
        seq = nxt;
      }
    }

    this.bpeCache.set(key, seq);
    return seq;
  }

  encode(text: string, addBos = true, addEos = false): number[] {
    const ids: number[] = [];
    if (addBos) ids.push(this.bosId);
    const enc = new TextEncoder();
    for (const m of text.matchAll(V6_PRETOK_RE)) {
      const piece = m[0];
      const spId = this.specialToId.get(piece);
      if (spId !== undefined) {
        ids.push(spId);
      } else {
        const bIds = this.bpeEncodeBytes(enc.encode(piece));
        for (let i = 0; i < bIds.length; i++) ids.push(bIds[i]);
      }
    }
    if (addEos) ids.push(this.eosId);
    return ids;
  }

  decode(ids: number[], skipSpecialTokens = true): string {
    const outParts: string[] = [];
    const byteAcc: number[] = [];
    const dec = new TextDecoder('utf-8', { fatal: false });

    const flush = () => {
      if (byteAcc.length > 0) {
        outParts.push(dec.decode(new Uint8Array(byteAcc)));
        byteAcc.length = 0;
      }
    };

    const nSpec = this.specialTokens.length;
    for (let idx = 0; idx < ids.length; idx++) {
      const tid = ids[idx] | 0;
      if (tid < 0 || tid >= this.vocabSize) continue;
      if (tid < nSpec) {
        if (!skipSpecialTokens) {
          flush();
          outParts.push(this.specialTokens[tid]);
        }
        continue;
      }
      const tb = this.tokenBytes[tid];
      for (let b = 0; b < tb.length; b++) byteAcc.push(tb[b]);
    }
    flush();
    return outParts.join('');
  }
}

export interface UnpackedScarstoreRecord {
  wiped: boolean;
  zAnchor: Float32Array;
  aScar: Float32Array; // [rank * hiddenSize]
  bScar: Float32Array; // [hiddenSize * rank]
  hasDual: boolean;
  tauDual: number;
  zDual: Float32Array;
}

/**
 * Unpacks a `4,096 B`-aligned `SNV5` `.scarstore` binary cartridge into exact float32 arrays.
 */
export function unpackScarstoreRecordBytes(
  rawBytes: Uint8Array,
  hiddenSize: number,
  rank: number
): UnpackedScarstoreRecord {
  if (
    rawBytes.byteLength < 4 ||
    rawBytes[0] !== 0x53 ||
    rawBytes[1] !== 0x4e ||
    rawBytes[2] !== 0x56 ||
    rawBytes[3] !== 0x35
  ) {
    return {
      wiped: true,
      zAnchor: new Float32Array(hiddenSize),
      aScar: new Float32Array(rank * hiddenSize),
      bScar: new Float32Array(hiddenSize * rank),
      hasDual: false,
      tauDual: 0.58,
      zDual: new Float32Array(hiddenSize),
    };
  }

  const dv = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
  let cur = 4;

  const zAnchor = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    zAnchor[i] = F16_TABLE[dv.getUint16(cur + i * 2, true)];
  }
  cur += hiddenSize * 2;

  const aU8Len = rank * (hiddenSize >> 1);
  const aU8Off = cur;
  cur += aU8Len;

  const nBlocks = hiddenSize >> 6;
  const aScLen = rank * nBlocks;
  const aScales = new Float32Array(aScLen);
  for (let i = 0; i < aScLen; i++) {
    aScales[i] = F16_TABLE[dv.getUint16(cur + i * 2, true)];
  }
  cur += aScLen * 2;

  const bU8Len = hiddenSize * (rank >> 1);
  const bU8Off = cur;
  cur += bU8Len;

  const bScales = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    bScales[i] = F16_TABLE[dv.getUint16(cur + i * 2, true)];
  }
  cur += hiddenSize * 2;

  let hasDual = false;
  let tauDual = 0.58;
  const zDual = new Float32Array(hiddenSize);
  if (
    rawBytes.byteLength >= cur + 8 + hiddenSize * 2 &&
    rawBytes[cur] === 0x44 &&
    rawBytes[cur + 1] === 0x55 &&
    rawBytes[cur + 2] === 0x41 &&
    rawBytes[cur + 3] === 0x4c
  ) {
    hasDual = true;
    tauDual = dv.getFloat32(cur + 4, true);
    const dBase = cur + 8;
    for (let i = 0; i < hiddenSize; i++) {
      zDual[i] = F16_TABLE[dv.getUint16(dBase + i * 2, true)];
    }
  }

  const aScar = new Float32Array(rank * hiddenSize);
  for (let r = 0; r < rank; r++) {
    const rowByte = aU8Off + r * (hiddenSize >> 1);
    const rowSc = r * nBlocks;
    const dstRow = r * hiddenSize;
    for (let col = 0; col < hiddenSize; col += 2) {
      const bt = rawBytes[rowByte + (col >> 1)];
      const sc = aScales[rowSc + (col >> 6)];
      aScar[dstRow + col] = ((bt & 0x0f) - 8) * sc;
      aScar[dstRow + col + 1] = (((bt >>> 4) & 0x0f) - 8) * sc;
    }
  }

  const bScar = new Float32Array(hiddenSize * rank);
  for (let d = 0; d < hiddenSize; d++) {
    const rowByte = bU8Off + d * (rank >> 1);
    const sc = bScales[d];
    const dstRow = d * rank;
    for (let r = 0; r < rank; r += 2) {
      const bt = rawBytes[rowByte + (r >> 1)];
      bScar[dstRow + r] = ((bt & 0x0f) - 8) * sc;
      bScar[dstRow + r + 1] = (((bt >>> 4) & 0x0f) - 8) * sc;
    }
  }

  return { wiped: false, zAnchor, aScar, bScar, hasDual, tauDual, zDual };
}

interface ShardMemoryView {
  u8: Uint8Array;
  u16: Uint16Array;
  u32: Uint32Array;
}

export interface RuntimeV6Cartridge {
  meta: ScarV6CartridgeMetaJSON;
  tauCos: number;
  hasDual: boolean;
  tauDual: number;
  zAnchorOrig: Float32Array;
  zDualOrig: Float32Array;
  aScarOrig: Float32Array;
  bScarOrig: Float32Array;
  zAnchor: Float32Array;
  zDual: Float32Array;
  aScar: Float32Array;
  bScar: Float32Array;
  wiped: boolean;
}

/**
 * Exact architectural byte budget calculator (`v6-12L` to `v6-Frontier 96L d=8192`).
 */
export function computeFrontierV6ByteBudget(
  cfg: ScarNativeV6ConfigJSON,
  activeCartridges = 4,
  mosLruHotPerLayer = 2,
  refContextTokens = 131072,
  tieLmHeadInFrontier = false
): ScarV6ArchitecturalBudgetJSON {
  const d = cfg.hidden_size;
  const dFf = cfg.intermediate_size;
  const qDim = cfg.num_q_heads * cfg.head_dim;
  const kvDim = cfg.num_kv_heads * cfg.head_dim;
  const lps = Math.floor(cfg.num_layers / cfg.num_stages);

  const denseLayerParams = d * (qDim + 2 * kvDim) + qDim * d + 3 * d * dFf + 2 * d;
  const denseTotalParams = cfg.vocab_size * d + cfg.num_layers * denseLayerParams + cfg.vocab_size * d;
  const denseFp16Bytes = denseTotalParams * 2;
  const denseKvRefBytes = cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * refContextTokens * 2;

  const vq4Base = (o: number, i: number) => o * (i >> 2) + o * (i >> 6) * 2;
  const idScar = (o: number, i: number, r: number) =>
    i * 2 + (r > 0 ? r * (i >> 1) + r * (i >> 6) * 2 + o * (r >> 1) + o * 2 : 0);

  const oneStageBase =
    2 * vq4Base(qDim, d) + 2 * vq4Base(kvDim, d) + 2 * vq4Base(dFf, d) + vq4Base(d, dFf);
  const allStageBases = cfg.num_stages * oneStageBase + 2048;

  const rk = cfg.id_scar_rank;
  const oneLayerIdScars =
    2 * idScar(qDim, d, rk) +
    2 * idScar(kvDim, d, rk) +
    2 * idScar(dFf, d, rk) +
    idScar(d, dFf, rk) +
    4 * d;
  const allIdScars = cfg.num_layers * oneLayerIdScars;

  const e = cfg.num_mos_experts;
  const er = cfg.mos_expert_rank;
  const singleExpert = er * (d >> 1) + er * (d >> 6) * 2 + d * (er >> 1) + d * 2;
  const routerPerLayer = e * (d >> 1) + e * (d >> 6) * 2;
  const mosAll = cfg.num_layers * (routerPerLayer + e * singleExpert);
  const mosHot = cfg.num_layers * (routerPerLayer + Math.min(e, mosLruHotPerLayer) * singleExpert);

  const embCodeBytes =
    cfg.vocab_size * (cfg.embed_rank >> 1) + cfg.vocab_size * (cfg.embed_rank >> 6) * 2;
  const embUpBytes =
    vq4Base(d, cfg.embed_rank) + idScar(d, cfg.embed_rank, Math.min(16, cfg.id_scar_rank));
  const lmHeadBytes = tieLmHeadInFrontier
    ? 0
    : vq4Base(cfg.vocab_size, d) + idScar(cfg.vocab_size, d, cfg.head_scar_rank);
  const vocabTotalBytes = embCodeBytes + embUpBytes + lmHeadBytes + 2 * d;

  const totalKvHeads = cfg.num_layers * cfg.num_kv_heads;
  const capLocal = cfg.num_sink + cfg.slide_len;
  const scacheBytes =
    totalKvHeads *
    (16 +
      (3 * capLocal + 2 * cfg.num_micro_pages) * cfg.head_dim * 2 +
      cfg.head_dim * cfg.head_dim * 4 +
      cfg.num_micro_pages * 4);

  const rawCart =
    4 +
    2 * d +
    cfg.cartridge_rank * (d >> 1) +
    cfg.cartridge_rank * (d >> 6) * 2 +
    d * (cfg.cartridge_rank >> 1) +
    2 * d;
  const cartAligned = ((rawCart + 4095) >> 12) << 12;
  const cartTotal = activeCartridges * cartAligned;

  const residentWeights = allStageBases + allIdScars + mosAll + vocabTotalBytes;
  const totalResidentVram = residentWeights + scacheBytes + cartTotal;
  const totalMosLruVram = allStageBases + allIdScars + mosHot + vocabTotalBytes + scacheBytes + cartTotal;
  const totalStreamVram =
    oneStageBase +
    lps * (oneLayerIdScars + routerPerLayer + Math.min(e, mosLruHotPerLayer) * singleExpert) +
    vocabTotalBytes +
    scacheBytes +
    cartTotal;
  const denseTotalAtRef = denseFp16Bytes + denseKvRefBytes;
  const mib = 1024 * 1024;
  const gib = 1024 * 1024 * 1024;

  return {
    shared_stage_bases_bytes: allStageBases,
    one_stage_base_bytes: oneStageBase,
    layer_identity_scars_bytes: allIdScars,
    mos_experts_all_bytes: mosAll,
    mos_experts_lru_bytes: mosHot,
    factorized_vocab_bytes: vocabTotalBytes,
    scache_o1_constant_bytes: scacheBytes,
    scache_delta_bytes: 0,
    active_cartridges_bytes: cartTotal,
    total_resident_bytes: totalResidentVram,
    total_resident_mib: Number((totalResidentVram / mib).toFixed(3)),
    total_resident_gib: Number((totalResidentVram / gib).toFixed(4)),
    total_lru_resident_gib: Number((totalMosLruVram / gib).toFixed(4)),
    stage_streaming_peak_bytes: totalStreamVram,
    stage_streaming_peak_mib: Number((totalStreamVram / mib).toFixed(3)),
    stage_streaming_peak_gib: Number((totalStreamVram / gib).toFixed(4)),
    matched_dense_bf16_weights_bytes: denseFp16Bytes,
    matched_dense_bf16_total_with_kv_bytes: denseTotalAtRef,
    matched_dense_bf16_total_gib: Number((denseTotalAtRef / gib).toFixed(3)),
    weight_compression_vs_dense_bf16: Number((denseFp16Bytes / Math.max(1, residentWeights)).toFixed(2)),
    total_vram_reduction_vs_dense_128k: Number((denseTotalAtRef / Math.max(1, totalMosLruVram)).toFixed(2)),
  };
}

/**
 * Bit-Exact CPU Executor operating directly on `scarnative_v6_shard_00.bin`, `procrustes_nullspace.bin`,
 * and `detached_cartridges.scarstore` in both Resident and Single-Stage Streaming (`streamStages: true`) modes.
 */
export class ScarNativeV6CpuExecutor {
  readonly manifest: ScarNativeV6ManifestJSON;
  readonly cfg: ScarNativeV6ConfigJSON;
  readonly tokenizer: ScarNativeBPETokenizerTS;
  private readonly fullShardView: ShardMemoryView;
  private readonly headTailView: ShardMemoryView;
  private readonly stageRawSlices: Uint8Array[];
  private readonly stageWindowBuf: ArrayBuffer;
  private readonly stageWindowView: ShardMemoryView;
  readonly stageStreamingWeightBytes: number;

  readonly invFreq: Float32Array;
  readonly rProcrustes: Float32Array; // [4 * D * D]
  readonly langCentroids: Float32Array; // [4 * D]
  readonly uLangT: Float32Array; // [k_lang * D]
  readonly isNullCalibrated: boolean;
  readonly cjkMaskU32: Uint32Array;
  readonly cartridges: Map<string, RuntimeV6Cartridge> = new Map();

  // Strictly O(1) Dual-Tier SCache state (`Δ = +0 B` across infinite context!)
  readonly scacheBytesTotal: number;
  private kRopeBuf: Float32Array; // [totalKv * capTotal * headDim]
  private vBuf: Float32Array; // [totalKv * capTotal * headDim]
  private kRawBuf: Float32Array; // [totalKv * capLocal * headDim]
  private wScar: Float32Array; // [totalKv * headDim * headDim]
  private pageScores: Float32Array; // [totalKv * max(1, numMicroPages)]
  private countsI32: Int32Array; // [totalKv * 4] -> [sinkCnt, winCnt, absorbedCnt, pageCnt]

  constructor(
    manifest: ScarNativeV6ManifestJSON,
    shardBuffer: ArrayBuffer,
    tokenizerJson: ScarNativeBPETokenizerJSON,
    procrustesBuffer: ArrayBuffer,
    scarstoreBuffer: ArrayBuffer
  ) {
    this.manifest = manifest;
    this.cfg = manifest.config;
    this.tokenizer = new ScarNativeBPETokenizerTS(tokenizerJson);

    this.fullShardView = {
      u8: new Uint8Array(shardBuffer),
      u16: new Uint16Array(shardBuffer),
      u32: new Uint32Array(shardBuffer),
    };

    const htBuf = shardBuffer.slice(0, manifest.head_tail_bytes);
    this.headTailView = {
      u8: new Uint8Array(htBuf),
      u16: new Uint16Array(htBuf),
      u32: new Uint32Array(htBuf),
    };

    const maxStageBytes = Math.max(16, (manifest.max_stage_bytes + 3) & ~3);
    this.stageWindowBuf = new ArrayBuffer(maxStageBytes);
    this.stageWindowView = {
      u8: new Uint8Array(this.stageWindowBuf),
      u16: new Uint16Array(this.stageWindowBuf),
      u32: new Uint32Array(this.stageWindowBuf),
    };
    this.stageRawSlices = manifest.stages.map(
      (sm) => new Uint8Array(shardBuffer, sm.byte_offset, sm.byte_length)
    );
    this.stageStreamingWeightBytes = htBuf.byteLength + maxStageBytes;

    const half = this.cfg.head_dim >> 1;
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      this.invFreq[i] = 1.0 / Math.pow(this.cfg.rope_theta, (2.0 * i) / this.cfg.head_dim);
    }

    const D = this.cfg.hidden_size;
    const kL = manifest.nullspace.k_lang;
    const rCnt = 4 * D * D;
    const cCnt = 4 * D;
    const uCnt = kL * D;
    const procF32 = new Float32Array(procrustesBuffer);
    this.rProcrustes = procF32.slice(0, rCnt);
    this.langCentroids = procF32.slice(rCnt, rCnt + cCnt);
    this.uLangT = procF32.slice(rCnt + cCnt, rCnt + cCnt + uCnt);
    this.isNullCalibrated = Boolean(manifest.nullspace.is_calibrated);
    this.cjkMaskU32 = new Uint32Array(manifest.cjk_bitmask_u32);

    const scarRaw = new Uint8Array(scarstoreBuffer);
    for (const cm of manifest.cartridges) {
      const slice = scarRaw.subarray(cm.byte_offset, cm.byte_offset + cm.record_bytes);
      const rec = unpackScarstoreRecordBytes(slice, D, this.cfg.cartridge_rank);
      this.cartridges.set(cm.cartridge_id, {
        meta: cm,
        tauCos: cm.tau_cos,
        hasDual: rec.hasDual,
        tauDual: rec.tauDual,
        zAnchorOrig: new Float32Array(rec.zAnchor),
        zDualOrig: new Float32Array(rec.zDual),
        aScarOrig: new Float32Array(rec.aScar),
        bScarOrig: new Float32Array(rec.bScar),
        zAnchor: rec.zAnchor,
        zDual: rec.zDual,
        aScar: rec.aScar,
        bScar: rec.bScar,
        wiped: rec.wiped,
      });
    }

    const totalKv = this.cfg.num_layers * this.cfg.num_kv_heads;
    const capLocal = this.cfg.num_sink + this.cfg.slide_len;
    const capTotal = capLocal + this.cfg.num_micro_pages;
    const dh = this.cfg.head_dim;
    this.kRopeBuf = new Float32Array(totalKv * capTotal * dh);
    this.vBuf = new Float32Array(totalKv * capTotal * dh);
    this.kRawBuf = new Float32Array(totalKv * capLocal * dh);
    this.wScar = new Float32Array(totalKv * dh * dh);
    this.pageScores = new Float32Array(totalKv * Math.max(1, this.cfg.num_micro_pages));
    this.countsI32 = new Int32Array(totalKv * 4);

    // Exact physical GPU Dual-Tier SCache byte footprint (`scache_kv_buf + scache_w_scar_buf + scache_page_scores_buf`)
    const kvBytes = totalKv * 16 + totalKv * (2 * capTotal + capLocal) * dh * 2;
    const wScarBytes = totalKv * dh * dh * 4;
    const pageScoreBytes = totalKv * Math.max(1, this.cfg.num_micro_pages) * 4;
    this.scacheBytesTotal = kvBytes + wScarBytes + pageScoreBytes;
  }

  resetSCache(): void {
    this.kRopeBuf.fill(0);
    this.vBuf.fill(0);
    this.kRawBuf.fill(0);
    this.wScar.fill(0);
    this.pageScores.fill(0);
    this.countsI32.fill(0);
  }

  totalPromotedMicroPages(): number {
    let total = 0;
    const totalKv = this.cfg.num_layers * this.cfg.num_kv_heads;
    for (let gh = 0; gh < totalKv; gh++) {
      total += this.countsI32[gh * 4 + 3];
    }
    return total;
  }

  gdprZeroWipe(cartridgeId: string): void {
    const c = this.cartridges.get(cartridgeId);
    if (!c) return;
    c.zAnchor.fill(0);
    c.zDual.fill(0);
    c.aScar.fill(0);
    c.bScar.fill(0);
    c.wiped = true;
  }

  restoreCartridge(cartridgeId: string): void {
    const c = this.cartridges.get(cartridgeId);
    if (!c) return;
    c.zAnchor.set(c.zAnchorOrig);
    c.zDual.set(c.zDualOrig);
    c.aScar.set(c.aScarOrig);
    c.bScar.set(c.bScarOrig);
    c.wiped = false;
  }

  gdprWipeDocument(docId: string): number {
    let count = 0;
    for (const [cid, c] of this.cartridges.entries()) {
      if (c.meta.doc_id === docId && !c.wiped) {
        this.gdprZeroWipe(cid);
        count++;
      }
    }
    return count;
  }

  restoreDocument(docId: string): number {
    let count = 0;
    for (const [cid, c] of this.cartridges.entries()) {
      if (c.meta.doc_id === docId && c.wiped) {
        this.restoreCartridge(cid);
        count++;
      }
    }
    return count;
  }

  resolveLangIdx(queryLang?: V6PolyglotLang | null, hGold?: Float32Array): number {
    if (queryLang && (V6_LANGS as readonly string[]).includes(queryLang)) {
      return V6_LANGS.indexOf(queryLang);
    }
    if (hGold && this.isNullCalibrated) {
      const D = this.cfg.hidden_size;
      let n2 = 0.0;
      for (let i = 0; i < D; i++) n2 += hGold[i] * hGold[i];
      const inv = 1.0 / Math.max(Math.sqrt(n2), 1e-6);
      let bestIdx = 1;
      let bestSim = -1e30;
      for (let l = 0; l < 4; l++) {
        let sim = 0.0;
        const cOff = l * D;
        for (let i = 0; i < D; i++) sim += this.langCentroids[cOff + i] * (hGold[i] * inv);
        if (sim > bestSim) {
          bestSim = sim;
          bestIdx = l;
        }
      }
      return bestIdx;
    }
    return 1;
  }

  projectNullspace(hGold: Float32Array, langIdx: number): Float32Array {
    const D = this.cfg.hidden_size;
    let hAl: Float32Array;
    if (this.isNullCalibrated) {
      hAl = new Float32Array(D);
      const rBase = langIdx * D * D;
      for (let r = 0; r < D; r++) {
        const hr = hGold[r];
        const rRow = rBase + r * D;
        for (let c = 0; c < D; c++) {
          hAl[c] += hr * this.rProcrustes[rRow + c];
        }
      }
      const kL = this.manifest.nullspace.k_lang;
      const coeffs = new Float32Array(kL);
      for (let k = 0; k < kL; k++) {
        let dot = 0.0;
        const uOff = k * D;
        for (let c = 0; c < D; c++) dot += hAl[c] * this.uLangT[uOff + c];
        coeffs[k] = dot;
      }
      for (let k = 0; k < kL; k++) {
        const cf = coeffs[k];
        const uOff = k * D;
        for (let c = 0; c < D; c++) hAl[c] -= cf * this.uLangT[uOff + c];
      }
    } else {
      hAl = new Float32Array(hGold);
    }
    let n2 = 0.0;
    for (let c = 0; c < D; c++) n2 += hAl[c] * hAl[c];
    const inv = 1.0 / Math.max(Math.sqrt(n2), 1e-6);
    for (let c = 0; c < D; c++) hAl[c] *= inv;
    return hAl;
  }

  buildFoldedCartridgeBuffer(cartridgeId: string, langIdx: number): Float32Array | null {
    const c = this.cartridges.get(cartridgeId);
    if (!c || c.wiped) return null;
    const D = this.cfg.hidden_size;
    const R = this.cfg.cartridge_rank;
    const packed = new Float32Array(1 + 2 * R * D);
    packed[0] = 1.0;

    if (this.isNullCalibrated) {
      // Exact Procrustes fold: `(h @ R_lang) @ A_scar^T == h @ (A_scar @ R_lang^T)^T`
      const rBase = langIdx * D * D;
      for (let r = 0; r < R; r++) {
        const aRow = r * D;
        const dstRow = 1 + r * D;
        for (let col = 0; col < D; col++) {
          let acc = 0.0;
          const rMatRow = rBase + col * D;
          for (let k = 0; k < D; k++) {
            acc += c.aScar[aRow + k] * this.rProcrustes[rMatRow + k];
          }
          packed[dstRow + col] = acc;
        }
      }
    } else {
      packed.set(c.aScar, 1);
    }
    packed.set(c.bScar, 1 + R * D);
    return packed;
  }

  routeCartridge(
    hGold: Float32Array,
    queryLang?: V6PolyglotLang | null
  ): { cartridgeId: string | null; docId: string | null; cosine: number; langIdx: number } {
    const langIdx = this.resolveLangIdx(queryLang, hGold);
    const zq = this.projectNullspace(hGold, langIdx);
    let useDual = false;
    for (const c of this.cartridges.values()) {
      if (!c.wiped && c.hasDual) {
        useDual = true;
        break;
      }
    }
    let bestId: string | null = null;
    let bestScore = -1e30;
    let bestCone = -1.0;
    for (const c of this.cartridges.values()) {
      if (c.wiped) continue;
      let dotCone = 0.0;
      for (let i = 0; i < zq.length; i++) dotCone += zq[i] * c.zAnchor[i];
      if (dotCone < c.tauCos) {
        if (!useDual && dotCone > bestCone) bestCone = dotCone;
        continue;
      }
      if (useDual && c.hasDual) {
        let dotDual = 0.0;
        for (let i = 0; i < zq.length; i++) dotDual += zq[i] * c.zDual[i];
        if (dotDual >= c.tauDual && dotDual > bestScore) {
          bestScore = dotDual;
          bestCone = dotCone;
          bestId = c.meta.cartridge_id;
        }
      } else if (dotCone > bestScore) {
        bestScore = dotCone;
        bestCone = dotCone;
        bestId = c.meta.cartridge_id;
      }
    }
    if (bestId !== null) {
      const c = this.cartridges.get(bestId)!;
      return { cartridgeId: bestId, docId: c.meta.doc_id, cosine: bestCone, langIdx };
    }
    return { cartridgeId: null, docId: null, cosine: bestCone, langIdx };
  }

  private readF16Vec(view: ShardMemoryView, u32Off: number, len: number): Float32Array {
    const out = new Float32Array(len);
    const u16Base = u32Off * 2;
    for (let i = 0; i < len; i++) out[i] = F16_TABLE[view.u16[u16Base + i]];
    return out;
  }

  private rmsNorm(view: ShardMemoryView, x: Float32Array, wU32Off: number): Float32Array {
    const n = x.length;
    let ss = 0.0;
    for (let i = 0; i < n; i++) ss += x[i] * x[i];
    const inv = 1.0 / Math.sqrt(ss / n + this.cfg.rms_norm_eps);
    const out = new Float32Array(n);
    const u16Base = wU32Off * 2;
    for (let i = 0; i < n; i++) {
      out[i] = x[i] * inv * F16_TABLE[view.u16[u16Base + i]];
    }
    return out;
  }

  private vq4Linear(view: ShardMemoryView, spec: ScarV6LinearSpecJSON, u32Sub: number, x: Float32Array): Float32Array {
    const { in_dim: inD, out_dim: outD, rank } = spec;
    const dEq = this.readF16Vec(view, spec.d_eq_u32_off - u32Sub, inD);
    const xRot = new Float32Array(inD);
    fwhtBlock64Signed(x, dEq, xRot);

    const uVec = new Float32Array(rank);
    if (rank > 0) {
      const aByteBase = (spec.ascar_packed_u32_off - u32Sub) * 4;
      const aScU16Base = (spec.ascar_scales_u32_off - u32Sub) * 2;
      const nBlocks = inD >> 6;
      for (let r = 0; r < rank; r++) {
        let acc = 0.0;
        const rowByte = aByteBase + r * (inD >> 1);
        const rowSc = aScU16Base + r * nBlocks;
        for (let b = 0; b < nBlocks; b++) {
          const sc = F16_TABLE[view.u16[rowSc + b]];
          let sub = 0.0;
          const bByte = rowByte + b * 32;
          const xBase = b * 64;
          for (let j = 0; j < 32; j++) {
            const bt = view.u8[bByte + j];
            const q0 = (bt & 0x0f) - 8;
            const q1 = ((bt >>> 4) & 0x0f) - 8;
            sub += q0 * x[xBase + j * 2] + q1 * x[xBase + j * 2 + 1];
          }
          acc += sub * sc;
        }
        uVec[r] = acc;
      }
    }

    const lutU16Base = (spec.lut256_u32_off - u32Sub) * 2;
    const codesBase = spec.vq_codes_u32_off - u32Sub;
    const scU16Base = (spec.vq_scales_u32_off - u32Sub) * 2;
    const u32PerRow = inD >> 4;
    const nBlocks = inD >> 6;
    const out = new Float32Array(outD);

    const bByteBase = rank > 0 ? (spec.bscar_u32_off - u32Sub) * 4 : 0;
    const bScU16Base = rank > 0 ? (bByteBase + outD * (rank >> 1)) >> 1 : 0;

    for (let row = 0; row < outD; row++) {
      let acc = 0.0;
      const rowCodeOff = codesBase + row * u32PerRow;
      const rowScOff = scU16Base + row * nBlocks;
      for (let w = 0; w < u32PerRow; w++) {
        const packed = view.u32[rowCodeOff + w];
        const sc = F16_TABLE[view.u16[rowScOff + (w >> 2)]];
        let sub = 0.0;
        const xBase = w * 16;
        for (let g = 0; g < 4; g++) {
          const code = (packed >>> (g * 8)) & 0xff;
          const lOff = lutU16Base + code * 4;
          const xOff = xBase + g * 4;
          sub +=
            F16_TABLE[view.u16[lOff]] * xRot[xOff] +
            F16_TABLE[view.u16[lOff + 1]] * xRot[xOff + 1] +
            F16_TABLE[view.u16[lOff + 2]] * xRot[xOff + 2] +
            F16_TABLE[view.u16[lOff + 3]] * xRot[xOff + 3];
        }
        acc += sub * sc;
      }
      if (rank > 0) {
        const bSc = F16_TABLE[view.u16[bScU16Base + row]];
        let bSub = 0.0;
        const rByte = bByteBase + row * (rank >> 1);
        for (let j = 0; j < rank >> 1; j++) {
          const bt = view.u8[rByte + j];
          const q0 = (bt & 0x0f) - 8;
          const q1 = ((bt >>> 4) & 0x0f) - 8;
          bSub += q0 * uVec[j * 2] + q1 * uVec[j * 2 + 1];
        }
        acc += bSub * bSc;
      }
      out[row] = acc;
    }
    return out;
  }

  private mosExpertsInplace(
    view: ShardMemoryView,
    mos: ScarV6MoSExpertSpecJSON,
    u32Sub: number,
    xNorm: Float32Array,
    hTarget: Float32Array
  ): number[] {
    const D = this.cfg.hidden_size;
    const nExp = mos.num_experts;
    const topK = mos.top_k;
    const rank = mos.rank;
    const nBlocks = D >> 6;

    // Shader #23 Phase 1: Router Block-64 INT4 dot products
    const rByteBase = (mos.router_packed_u32_off - u32Sub) * 4;
    const rScBase = (mos.router_scales_u32_off - u32Sub) * 2;
    const rLogits = new Float32Array(nExp);
    for (let e = 0; e < nExp; e++) {
      let acc = 0.0;
      const rowByte = rByteBase + e * (D >> 1);
      const rowSc = rScBase + e * nBlocks;
      for (let b = 0; b < nBlocks; b++) {
        const sc = F16_TABLE[view.u16[rowSc + b]];
        let sub = 0.0;
        const bByte = rowByte + b * 32;
        const xBase = b * 64;
        for (let j = 0; j < 32; j++) {
          const bt = view.u8[bByte + j];
          sub += ((bt & 0x0f) - 8) * xNorm[xBase + j * 2] + (((bt >>> 4) & 0x0f) - 8) * xNorm[xBase + j * 2 + 1];
        }
        acc += sub * sc;
      }
      rLogits[e] = acc;
    }

    // Shader #23 Phase 2: Exact Top-k selection (`-1e-7 * e` tie-break) + Softmax
    let usedMask = 0;
    let maxTop = -1e30;
    const selIdx: number[] = [];
    for (let j = 0; j < topK; j++) {
      let bestE = 0;
      let bestPert = -1e30;
      for (let e = 0; e < nExp; e++) {
        if (((usedMask >>> e) & 1) === 0) {
          const pert = rLogits[e] - 1e-7 * e;
          if (pert > bestPert) {
            bestPert = pert;
            bestE = e;
          }
        }
      }
      usedMask |= 1 << bestE;
      selIdx.push(bestE);
      if (rLogits[bestE] > maxTop) maxTop = rLogits[bestE];
    }

    const weights = new Float32Array(topK);
    let sumEx = 0.0;
    for (let j = 0; j < topK; j++) {
      const ex = Math.exp(rLogits[selIdx[j]] - maxTop);
      weights[j] = ex;
      sumEx += ex;
    }
    const invSum = 1.0 / Math.max(sumEx, 1e-12);
    for (let j = 0; j < topK; j++) weights[j] *= invSum;

    // Shader #23 Phase 3: Directional `ReLU(A_exp @ xNorm) * w_j`
    const aByteBase = (mos.a_exp_packed_u32_off - u32Sub) * 4;
    const aScBase = (mos.a_exp_scales_u32_off - u32Sub) * 2;
    const uAll = new Float32Array(topK * rank);
    for (let j = 0; j < topK; j++) {
      const eSel = selIdx[j];
      const wJ = weights[j];
      for (let r = 0; r < rank; r++) {
        const aRow = eSel * rank + r;
        const rowByte = aByteBase + aRow * (D >> 1);
        const rowSc = aScBase + aRow * nBlocks;
        let acc = 0.0;
        for (let b = 0; b < nBlocks; b++) {
          const sc = F16_TABLE[view.u16[rowSc + b]];
          let sub = 0.0;
          const bByte = rowByte + b * 32;
          const xBase = b * 64;
          for (let col = 0; col < 32; col++) {
            const bt = view.u8[bByte + col];
            sub +=
              ((bt & 0x0f) - 8) * xNorm[xBase + col * 2] +
              (((bt >>> 4) & 0x0f) - 8) * xNorm[xBase + col * 2 + 1];
          }
          acc += sub * sc;
        }
        uAll[j * rank + r] = wJ * Math.max(acc, 0.0);
      }
    }

    // Shader #24: Row-Scaled INT4 `B_exp` Residual Accumulator
    const bByteBase = (mos.b_exp_packed_u32_off - u32Sub) * 4;
    const bScBase = (mos.b_exp_scales_u32_off - u32Sub) * 2;
    const halfRank = rank >> 1;
    for (let d = 0; d < D; d++) {
      let addVal = 0.0;
      for (let j = 0; j < topK; j++) {
        const eSel = selIdx[j];
        const bRow = eSel * D + d;
        const sc = F16_TABLE[view.u16[bScBase + bRow]];
        const rowByte = bByteBase + bRow * halfRank;
        const uBase = j * rank;
        let sub = 0.0;
        for (let k = 0; k < halfRank; k++) {
          const bt = view.u8[rowByte + k];
          sub += ((bt & 0x0f) - 8) * uAll[uBase + k * 2] + (((bt >>> 4) & 0x0f) - 8) * uAll[uBase + k * 2 + 1];
        }
        addVal += sub * sc;
      }
      hTarget[d] += addVal;
    }

    return selIdx;
  }

  private embedLookup(tokenId: number, streamStages: boolean): Float32Array {
    const view = streamStages ? this.headTailView : this.fullShardView;
    const emb = this.manifest.embed;
    const rEmb = emb.embed_rank;
    const codeVec = new Float32Array(rEmb);
    const pkByteBase = emb.code_packed_u32_off * 4 + tokenId * (rEmb >> 1);
    const scU16Base = emb.code_scales_u32_off * 2 + tokenId * (rEmb >> 6);
    for (let col = 0; col < rEmb; col++) {
      const byteVal = view.u8[pkByteBase + (col >> 1)];
      const nib = (col & 1) === 0 ? byteVal & 0x0f : (byteVal >>> 4) & 0x0f;
      codeVec[col] = (nib - 8) * F16_TABLE[view.u16[scU16Base + (col >> 6)]];
    }
    return this.vq4Linear(view, emb.up_proj, 0, codeVec);
  }

  private applyRopeInplace(vec: Float32Array, nHeads: number, posT: number): void {
    const dh = this.cfg.head_dim;
    const half = dh >> 1;
    for (let h = 0; h < nHeads; h++) {
      const base = h * dh;
      for (let i = 0; i < half; i++) {
        const ang = posT * this.invFreq[i];
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        const x1 = vec[base + i];
        const x2 = vec[base + half + i];
        vec[base + i] = x1 * cs - x2 * sn;
        vec[base + half + i] = x2 * cs + x1 * sn;
      }
    }
  }

  private injectFoldedCartridgeInplace(h: Float32Array, foldedBuf: Float32Array | null): void {
    if (!foldedBuf) return;
    const D = this.cfg.hidden_size;
    const R = this.cfg.cartridge_rank;
    const wM = foldedBuf[0];
    const aBase = 1;
    const bBase = 1 + R * D;
    const u = new Float32Array(R);
    for (let r = 0; r < R; r++) {
      let acc = 0.0;
      const off = aBase + r * D;
      for (let i = 0; i < D; i++) acc += h[i] * foldedBuf[off + i];
      u[r] = acc;
    }
    for (let i = 0; i < D; i++) {
      let acc = 0.0;
      const off = bBase + i * R;
      for (let r = 0; r < R; r++) acc += foldedBuf[off + r] * u[r];
      h[i] += wM * acc;
    }
  }

  stepToken(
    tokenId: number,
    posT: number,
    opts: {
      routeOnGolden?: boolean;
      activeFoldedCartridge?: Float32Array | null;
      activeCartridgeId?: string | null;
      activeDocId?: string | null;
      queryLang?: V6PolyglotLang | null;
      useScarStore?: boolean;
      streamStages?: boolean;
      enableWidrowHoffOverride?: boolean;
      enableMicroPagesOverride?: boolean;
      suppressCjk?: boolean;
    } = {}
  ): {
    logits: Float32Array;
    routedCartridgeId: string | null;
    routedDocId: string | null;
    routedCosine: number;
    resolvedLangIdx: number;
    activeFoldedCartridge: Float32Array | null;
    selectedExpertsByLayer: number[][];
  } {
    const c = this.cfg;
    const dh = c.head_dim;
    const capLocal = c.num_sink + c.slide_len;
    const capTotal = capLocal + c.num_micro_pages;
    const nMp = Math.max(1, c.num_micro_pages);
    const gqaGroup = Math.floor(c.num_q_heads / c.num_kv_heads);
    const scale = 1.0 / Math.sqrt(dh);
    const streamStages = opts.streamStages ?? false;
    const enableWH = opts.enableWidrowHoffOverride ?? c.enable_widrow_hoff;
    const enableMP = opts.enableMicroPagesOverride ?? c.enable_micro_pages;

    const h = this.embedLookup(tokenId, streamStages);
    let routedId = opts.activeCartridgeId ?? null;
    let routedDoc = opts.activeDocId ?? null;
    let routedCos = -1.0;
    let resolvedLangIdx = this.resolveLangIdx(opts.queryLang);
    let foldedBuf = opts.activeFoldedCartridge ?? null;
    const selectedExpertsByLayer: number[][] = [];

    for (const sm of this.manifest.stages) {
      let view: ShardMemoryView;
      let u32Sub: number;
      if (streamStages) {
        this.stageWindowView.u8.set(this.stageRawSlices[sm.stage_idx], 0);
        view = this.stageWindowView;
        u32Sub = sm.start_u32;
      } else {
        view = this.fullShardView;
        u32Sub = 0;
      }

      for (let l = sm.layer_start; l < sm.layer_end; l++) {
        const lm = this.manifest.layers[l];
        const x1 = this.rmsNorm(view, h, lm.ln1_u32_off - u32Sub);
        const qRaw = this.vq4Linear(view, lm.projs.q_proj, u32Sub, x1);
        const kRaw = this.vq4Linear(view, lm.projs.k_proj, u32Sub, x1);
        const vCur = this.vq4Linear(view, lm.projs.v_proj, u32Sub, x1);

        const qRope = new Float32Array(qRaw);
        const kRope = new Float32Array(kRaw);
        this.applyRopeInplace(qRope, c.num_q_heads, posT);
        this.applyRopeInplace(kRope, c.num_kv_heads, posT);

        // Shader #25 exact CPU mirror (`scarnative_v6_hierarchical_evict_update`)
        for (let hKv = 0; hKv < c.num_kv_heads; hKv++) {
          const gh = l * c.num_kv_heads + hKv;
          const cOff = gh * 4;
          const sinkCnt = this.countsI32[cOff];
          const winCnt = this.countsI32[cOff + 1];
          const absorbed = this.countsI32[cOff + 2];
          const pageCnt = this.countsI32[cOff + 3];

          const doEvict = posT >= c.num_sink && winCnt >= c.slide_len;
          const activeEvict = doEvict && l > 0 && (enableWH || enableMP);
          const ringIdx = posT >= c.num_sink && c.slide_len > 0 ? (posT - c.num_sink) % c.slide_len : 0;
          const ringSlot = c.num_sink + ringIdx;
          const kvTotalBase = gh * capTotal * dh;
          const kvLocalBase = gh * capLocal * dh;

          if (activeEvict) {
            const evTotalOff = kvTotalBase + ringSlot * dh;
            const evLocalOff = kvLocalBase + ringSlot * dh;
            let kn2 = 0.0;
            for (let d = 0; d < dh; d++) {
              const kv = this.kRawBuf[evLocalOff + d];
              kn2 += kv * kv;
            }
            const invKn = 1.0 / Math.max(Math.sqrt(kn2), 1e-6);
            const ku = new Float32Array(dh);
            for (let d = 0; d < dh; d++) ku[d] = this.kRawBuf[evLocalOff + d] * invKn;

            const wBase = gh * dh * dh;
            const err = new Float32Array(dh);
            let errSq = 0.0;
            for (let r = 0; r < dh; r++) {
              let pred = 0.0;
              const rOff = wBase + r * dh;
              for (let col = 0; col < dh; col++) pred += this.wScar[rOff + col] * ku[col];
              const er = this.vBuf[evTotalOff + r] - pred;
              err[r] = er;
              errSq += er * er;
            }
            const errNorm = Math.sqrt(errSq);

            const gStride = dh + 3;
            const gBase = hKv * gStride;
            let gz = lm.kv_gate_f32[gBase + dh];
            for (let d = 0; d < dh; d++) gz += this.kRawBuf[evLocalOff + d] * lm.kv_gate_f32[gBase + d];
            gz = Math.max(-20.0, Math.min(20.0, gz));
            const gTau = 1.0 / (1.0 + Math.exp(-gz));
            const lamH = lm.kv_gate_f32[gBase + dh + 1];
            const etaH = lm.kv_gate_f32[gBase + dh + 2];
            const sTau = gTau * errNorm;

            // Step 3a: Surprise Micro-Page promotion BEFORE overwriting `ringSlot`
            if (enableMP && c.num_micro_pages > 0 && sTau >= c.micro_page_theta) {
              const psBase = gh * nMp;
              let targetPage = -1;
              if (pageCnt < c.num_micro_pages) {
                targetPage = pageCnt;
                this.pageScores[psBase + pageCnt] = sTau;
                this.countsI32[cOff + 3] = pageCnt + 1;
              } else {
                let minIdx = 0;
                let minSc = this.pageScores[psBase];
                for (let m = 1; m < c.num_micro_pages; m++) {
                  if (this.pageScores[psBase + m] < minSc) {
                    minSc = this.pageScores[psBase + m];
                    minIdx = m;
                  }
                }
                if (sTau > minSc + 1e-4) {
                  targetPage = minIdx;
                  this.pageScores[psBase + minIdx] = sTau;
                }
              }
              if (targetPage >= 0) {
                const dstOff = kvTotalBase + (capLocal + targetPage) * dh;
                for (let d = 0; d < dh; d++) {
                  this.kRopeBuf[dstOff + d] = this.kRopeBuf[evTotalOff + d];
                  this.vBuf[dstOff + d] = this.vBuf[evTotalOff + d];
                }
              }
            }

            // Step 3b: Widrow-Hoff Fast-Weight `W_scar` update
            if (enableWH) {
              const decay = 1.0 - lamH * gTau;
              const lrEff = etaH * gTau;
              for (let r = 0; r < dh; r++) {
                const rOff = wBase + r * dh;
                const er = err[r];
                for (let col = 0; col < dh; col++) {
                  this.wScar[rOff + col] = decay * this.wScar[rOff + col] + lrEff * er * ku[col];
                }
              }
            }
          }

          const writeSlot = posT < c.num_sink ? posT : ringSlot;
          const wrTotalOff = kvTotalBase + writeSlot * dh;
          const wrLocalOff = kvLocalBase + writeSlot * dh;
          for (let d = 0; d < dh; d++) {
            this.kRopeBuf[wrTotalOff + d] = roundToF16(kRope[hKv * dh + d]);
            this.kRawBuf[wrLocalOff + d] = roundToF16(kRaw[hKv * dh + d]);
            this.vBuf[wrTotalOff + d] = roundToF16(vCur[hKv * dh + d]);
          }

          if (posT < c.num_sink) {
            if (posT + 1 > sinkCnt) this.countsI32[cOff] = posT + 1;
          } else {
            this.countsI32[cOff + 1] = Math.min(c.slide_len, winCnt + 1);
            if (doEvict) this.countsI32[cOff + 2] = absorbed + 1;
          }
        }

        // Shader #26 exact CPU mirror (`scarnative_v6_hierarchical_o1_gqa_attention`)
        const attnOut = new Float32Array(c.num_q_heads * dh);
        for (let hQ = 0; hQ < c.num_q_heads; hQ++) {
          const hKv = Math.floor(hQ / gqaGroup);
          const gh = l * c.num_kv_heads + hKv;
          const cOff = gh * 4;
          const sinkCnt = this.countsI32[cOff];
          const winCnt = this.countsI32[cOff + 1];
          const absorbed = this.countsI32[cOff + 2];
          const pageCnt = l > 0 && enableMP ? this.countsI32[cOff + 3] : 0;
          const kvTotalBase = gh * capTotal * dh;

          const slots: number[] = [];
          for (let s = 0; s < sinkCnt; s++) slots.push(s);
          for (let w = 0; w < winCnt; w++) slots.push(c.num_sink + w);
          for (let p = 0; p < pageCnt; p++) slots.push(capLocal + p);

          const scores = new Float32Array(slots.length);
          let maxS = -1e30;
          for (let m = 0; m < slots.length; m++) {
            const kOff = kvTotalBase + slots[m] * dh;
            let dot = 0.0;
            for (let d = 0; d < dh; d++) dot += qRope[hQ * dh + d] * this.kRopeBuf[kOff + d];
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
            const pr = scores[m] * invSum;
            const vOff = kvTotalBase + slots[m] * dh;
            for (let d = 0; d < dh; d++) attnOut[hQ * dh + d] += pr * this.vBuf[vOff + d];
          }

          if (enableWH && l > 0 && absorbed > 0) {
            const alpha = lm.alpha_assoc_f32[hQ];
            if (Math.abs(alpha) > 1e-8) {
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
        }

        const oProj = this.vq4Linear(view, lm.projs.o_proj, u32Sub, attnOut);
        for (let i = 0; i < c.hidden_size; i++) h[i] += oProj[i];

        const x2 = this.rmsNorm(view, h, lm.ln2_u32_off - u32Sub);
        const gate = this.vq4Linear(view, lm.projs.gate_proj, u32Sub, x2);
        const up = this.vq4Linear(view, lm.projs.up_proj, u32Sub, x2);
        for (let i = 0; i < c.intermediate_size; i++) {
          const g = gate[i];
          gate[i] = (g / (1.0 + Math.exp(-g))) * up[i];
        }
        const down = this.vq4Linear(view, lm.projs.down_proj, u32Sub, gate);
        for (let i = 0; i < c.hidden_size; i++) h[i] += down[i];

        // Shaders #23 & #24 exact CPU mirror
        if (lm.mos_experts) {
          selectedExpertsByLayer.push(this.mosExpertsInplace(view, lm.mos_experts, u32Sub, x2, h));
        } else {
          selectedExpertsByLayer.push([]);
        }

        if (l + 1 === c.golden_layer) {
          if (opts.routeOnGolden && (opts.useScarStore ?? true)) {
            const rt = this.routeCartridge(h, opts.queryLang);
            routedId = rt.cartridgeId;
            routedDoc = rt.docId;
            routedCos = rt.cosine;
            resolvedLangIdx = rt.langIdx;
            foldedBuf = routedId ? this.buildFoldedCartridgeBuffer(routedId, rt.langIdx) : null;
          }
          if (foldedBuf) {
            this.injectFoldedCartridgeInplace(h, foldedBuf);
          }
        }
      }
    }

    const tailView = streamStages ? this.headTailView : this.fullShardView;
    const hFinal = this.rmsNorm(tailView, h, this.manifest.final_norm_u32_off);
    const logits = this.vq4Linear(tailView, this.manifest.lm_head, 0, hFinal);
    if (opts.suppressCjk ?? true) {
      for (let v = 0; v < c.vocab_size; v++) {
        const bit = (this.cjkMaskU32[v >>> 5] >>> (v & 31)) & 1;
        if (bit !== 0) logits[v] -= 64.0;
      }
    }

    return {
      logits,
      routedCartridgeId: routedId,
      routedDocId: routedDoc,
      routedCosine: routedCos,
      resolvedLangIdx,
      activeFoldedCartridge: foldedBuf,
      selectedExpertsByLayer,
    };
  }

  generate(
    prompt: string,
    opts: {
      maxNewTokens?: number;
      useScarStore?: boolean;
      queryLang?: V6PolyglotLang | null;
      streamStages?: boolean;
      enableWidrowHoff?: boolean;
      enableMicroPages?: boolean;
      suppressCjk?: boolean;
    } = {}
  ): ScarNativeV6GenerationReport {
    const t0 = performance.now();
    this.resetSCache();
    const scacheInit = this.scacheBytesTotal;
    const pIds = this.tokenizer.encode(prompt, true, false);
    const maxNew = opts.maxNewTokens ?? 20;
    const streamStages = opts.streamStages ?? false;

    let lastLogits: Float32Array<ArrayBufferLike> = new Float32Array(this.cfg.vocab_size);
    let routedId: string | null = null;
    let routedDoc: string | null = null;
    let routedCos = -1.0;
    let resolvedLangIdx = this.resolveLangIdx(opts.queryLang);
    let foldedBuf: Float32Array | null = null;
    let lastExperts: number[][] = [];

    for (let i = 0; i < pIds.length; i++) {
      const isLast = i === pIds.length - 1;
      const res = this.stepToken(pIds[i], i, {
        routeOnGolden: isLast,
        activeFoldedCartridge: null,
        activeCartridgeId: null,
        activeDocId: null,
        queryLang: opts.queryLang,
        useScarStore: opts.useScarStore ?? true,
        streamStages,
        enableWidrowHoffOverride: opts.enableWidrowHoff,
        enableMicroPagesOverride: opts.enableMicroPages,
        suppressCjk: opts.suppressCjk ?? true,
      });
      lastLogits = res.logits;
      if (isLast) {
        routedId = res.routedCartridgeId;
        routedDoc = res.routedDocId;
        routedCos = res.routedCosine;
        resolvedLangIdx = res.resolvedLangIdx;
        foldedBuf = res.activeFoldedCartridge;
        lastExperts = res.selectedExpertsByLayer;
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
        activeFoldedCartridge: foldedBuf,
        activeCartridgeId: routedId,
        activeDocId: routedDoc,
        queryLang: opts.queryLang,
        useScarStore: opts.useScarStore ?? true,
        streamStages,
        enableWidrowHoffOverride: opts.enableWidrowHoff,
        enableMicroPagesOverride: opts.enableMicroPages,
        suppressCjk: opts.suppressCjk ?? true,
      });
      lastLogits = nextRes.logits;
      posT++;
    }

    const elapsedMs = Math.max(0.1, performance.now() - t0);
    const scacheFinal = this.scacheBytesTotal;
    const activeWeights = streamStages ? this.stageStreamingWeightBytes : this.manifest.shard_bytes;

    return {
      prompt,
      outputText: this.tokenizer.decode(outIds, true).trim(),
      outputTextWithTags: this.tokenizer.decode(outIds, false).trim(),
      outputTokenIds: outIds,
      abstained: outIds.includes(this.tokenizer.abstainId),
      prefillTokens: pIds.length,
      generatedTokens: outIds.length,
      elapsedMs,
      tokPerSec: ((pIds.length + outIds.length) / elapsedMs) * 1000.0,
      routedCartridgeId: routedId,
      routedDocId: routedDoc,
      routedCosine: routedCos,
      resolvedLang: V6_LANGS[resolvedLangIdx] ?? 'en',
      firstTokenConfidence: firstConf,
      firstTokenMargin: firstMargin,
      scacheBytesInitial: scacheInit,
      scacheBytesFinal: scacheFinal,
      scacheDeltaBytes: scacheFinal - scacheInit,
      activeWeightGpuBytes: activeWeights,
      streamStages,
      promotedMicroPagesTotal: this.totalPromotedMicroPages(),
      selectedExpertsByLayer: lastExperts,
      backend: streamStages ? 'cpu_exact_ref_stream' : 'cpu_exact_ref',
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
 * Physical Browser WebGPU WGSL 1.0 Executor for `ScarNativeLMV6` (`26` Shaders, `ΔSCache = 0 B`,
 * supporting both GPU-Resident and Single-Stage Streaming `stageWindowBuf` execution!).
 */
export class ScarNativeV6WebGpuBrowserEngine {
  readonly cpuRef: ScarNativeV6CpuExecutor;
  private device: any = null;
  private pipelines: Record<string, any> = {};
  private shardBuf: any = null;
  private headTailBuf: any = null;
  private stageWindowBuf: any = null;
  private stageSlicesU8: Uint8Array[] = [];

  private scacheKvBuf: any = null;
  private scacheWScarBuf: any = null;
  private scachePageScoresBuf: any = null;
  private stepStateBuf: any = null;
  private embCodeBuf: any = null;
  private hCurBuf: any = null;
  private normBuf: any = null;
  private vq4RotBuf: any = null;
  private asvdUBuf: any = null;
  private mosUBuf: any = null;
  private mosSelBuf: any = null;
  private mosSelReadBuf: any = null;
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
  private countsReadBuf: any = null;
  private invFreqBuf: any = null;
  private cjkMaskBuf: any = null;
  private kRawGateBufs: any[] = [];
  private qRawAlphaBufs: any[] = [];
  private zeroMuBuf: any = null;
  private activeCapBuf: any = null;
  private scarTelemBuf: any = null;

  private countsU32Total = 0;
  private activeCartridgeId: string | null = null;
  private activeDocId: string | null = null;
  webgpuReady = false;
  liveParityCosine: number | null = null;
  stageStreamingMaxDiff: number | null = null;

  constructor(
    cpuRef: ScarNativeV6CpuExecutor,
    private shardRaw: ArrayBuffer,
    private wgslSourceCode?: string
  ) {
    this.cpuRef = cpuRef;
  }

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
        (await fetch('/webgpu/scarnative_v6/mos_scache_kernels.wgsl').then((r) => {
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
        'scarnative_v6_mos_router_topk_relu',
        'scarnative_v6_mos_bscar_accumulate',
        'scarnative_v6_hierarchical_evict_update',
        'scarnative_v6_hierarchical_o1_gqa_attention',
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
      const capLocal = c.num_sink + c.slide_len;
      const capTotal = capLocal + c.num_micro_pages;
      const kvWords = this.countsU32Total + totalKv * (2 * capTotal + capLocal) * (c.head_dim >> 1);

      const mkBuf = (size: number, usage: number, data?: ArrayBufferView) => {
        const aligned = Math.max(16, (size + 3) & ~3);
        const b = dev.createBuffer({ size: aligned, usage });
        if (data) dev.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
        return b;
      };

      const shardU8 = new Uint8Array(this.shardRaw);
      this.shardBuf = mkBuf(shardU8.byteLength, GPU_BUF.STORAGE | GPU_BUF.COPY_DST, shardU8);
      this.headTailBuf = mkBuf(
        m.head_tail_bytes,
        GPU_BUF.STORAGE | GPU_BUF.COPY_DST,
        shardU8.subarray(0, m.head_tail_bytes)
      );
      this.stageWindowBuf = mkBuf(Math.max(16, m.max_stage_bytes), GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.stageSlicesU8 = m.stages.map((sm) => shardU8.subarray(sm.byte_offset, sm.byte_offset + sm.byte_length));

      this.scacheKvBuf = mkBuf(kvWords * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST | GPU_BUF.COPY_SRC);
      this.scacheWScarBuf = mkBuf(totalKv * c.head_dim * c.head_dim * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.scachePageScoresBuf = mkBuf(
        totalKv * Math.max(1, c.num_micro_pages) * 4,
        GPU_BUF.STORAGE | GPU_BUF.COPY_DST
      );
      this.countsReadBuf = mkBuf(this.countsU32Total * 4, GPU_BUF.MAP_READ | GPU_BUF.COPY_DST);

      this.stepStateBuf = mkBuf(16, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.embCodeBuf = mkBuf(c.embed_rank * 4, GPU_BUF.STORAGE);
      this.hCurBuf = mkBuf(c.hidden_size * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST | GPU_BUF.COPY_SRC);
      this.normBuf = mkBuf(c.hidden_size * 4, GPU_BUF.STORAGE);
      this.vq4RotBuf = mkBuf(Math.max(c.hidden_size, c.intermediate_size, c.embed_rank) * 4, GPU_BUF.STORAGE);
      this.asvdUBuf = mkBuf(Math.max(16, c.id_scar_rank, c.head_scar_rank) * 4, GPU_BUF.STORAGE);
      this.mosUBuf = mkBuf(Math.max(16, c.mos_top_k * c.mos_expert_rank) * 4, GPU_BUF.STORAGE);
      this.mosSelBuf = mkBuf(16, GPU_BUF.STORAGE | GPU_BUF.COPY_SRC);
      this.mosSelReadBuf = mkBuf(c.num_layers * 16, GPU_BUF.MAP_READ | GPU_BUF.COPY_DST);
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
      this.cjkMaskBuf = mkBuf(
        this.cpuRef.cjkMaskU32.byteLength,
        GPU_BUF.STORAGE | GPU_BUF.COPY_DST,
        this.cpuRef.cjkMaskU32
      );

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

      this.zeroMuBuf = mkBuf(
        2 * c.hidden_size * 4,
        GPU_BUF.STORAGE | GPU_BUF.COPY_DST,
        new Float32Array(2 * c.hidden_size)
      );
      this.activeCapBuf = mkBuf((1 + 2 * c.cartridge_rank * c.hidden_size) * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_DST);
      this.scarTelemBuf = mkBuf(64, GPU_BUF.STORAGE);
      this.webgpuReady = true;

      // Immediate live 1-step parity check (`CPU Exact <-> WebGPU Resident <-> WebGPU Stage-Streaming`)
      this.resetGpuSCache();
      this.cpuRef.resetSCache();
      const cpuStep = this.cpuRef.stepToken(this.cpuRef.tokenizer.bosId, 0, { useScarStore: false });
      const gpuRes = await this.stepTokenGpu(this.cpuRef.tokenizer.bosId, 0, {
        routeOnGolden: false,
        suppressCjk: true,
        useScarStore: false,
        streamStages: false,
      });
      this.resetGpuSCache();
      const gpuStream = await this.stepTokenGpu(this.cpuRef.tokenizer.bosId, 0, {
        routeOnGolden: false,
        suppressCjk: true,
        useScarStore: false,
        streamStages: true,
      });

      let dot = 0.0;
      let na = 0.0;
      let nb = 0.0;
      let maxStreamDiff = 0.0;
      for (let i = 0; i < gpuRes.logits.length; i++) {
        dot += cpuStep.logits[i] * gpuRes.logits[i];
        na += cpuStep.logits[i] * cpuStep.logits[i];
        nb += gpuRes.logits[i] * gpuRes.logits[i];
        maxStreamDiff = Math.max(maxStreamDiff, Math.abs(gpuRes.logits[i] - gpuStream.logits[i]));
      }
      this.liveParityCosine = dot / Math.max(Math.sqrt(na * nb), 1e-12);
      this.stageStreamingMaxDiff = maxStreamDiff;
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
    this.device.queue.writeBuffer(this.scachePageScoresBuf, 0, new Uint8Array(this.scachePageScoresBuf.size));
    this.activeCartridgeId = null;
    this.activeDocId = null;
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

  private dispatchVq4Linear(
    cp: any,
    spec: ScarV6LinearSpecJSON,
    wBuf: any,
    u32Sub: number,
    xBuf: any,
    yBuf: any
  ): void {
    const uEq = this.uBuf(packParams64([spec.in_dim, spec.d_eq_u32_off - u32Sub]));
    cp.setPipeline(this.pipelines.vq4_equilibrate_fwht64);
    cp.setBindGroup(0, this.bg('vq4_equilibrate_fwht64', { 0: uEq, 1: wBuf, 3: xBuf, 6: this.vq4RotBuf }));
    cp.dispatchWorkgroups(spec.in_dim >>> 6, 1, 1);

    if (spec.rank > 0) {
      const uA = this.uBuf(
        packParams64([
          spec.rank,
          spec.in_dim,
          0,
          spec.ascar_packed_u32_off - u32Sub,
          spec.ascar_scales_u32_off - u32Sub,
        ])
      );
      cp.setPipeline(this.pipelines.mos_fused_int4_bscar_gemv);
      cp.setBindGroup(
        0,
        this.bg('mos_fused_int4_bscar_gemv', {
          0: uA,
          1: wBuf,
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
        spec.vq_codes_u32_off - u32Sub,
        spec.vq_scales_u32_off - u32Sub,
        spec.lut256_u32_off - u32Sub,
        spec.rank > 0 ? spec.bscar_u32_off - u32Sub : 0,
        0,
      ])
    );
    cp.setPipeline(this.pipelines.mos_fused_vq4_int4scar_gemv);
    cp.setBindGroup(
      0,
      this.bg('mos_fused_vq4_int4scar_gemv', {
        0: uVq,
        1: wBuf,
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

  private async readU32(readBuf: any, len: number): Promise<Uint32Array> {
    await readBuf.mapAsync(GPU_MAP_READ);
    const copy = new Uint32Array(readBuf.getMappedRange().slice(0, len * 4));
    readBuf.unmap();
    return copy;
  }

  private encodeEmbed(enc: any, streamStages: boolean): void {
    const c = this.cpuRef.cfg;
    const emb = this.cpuRef.manifest.embed;
    const wBuf = streamStages ? this.headTailBuf : this.shardBuf;
    const cp = enc.beginComputePass();
    const uEmb = this.uBuf(
      packParams64([c.embed_rank, emb.code_packed_u32_off, emb.code_scales_u32_off, 0, c.vocab_size])
    );
    cp.setPipeline(this.pipelines.embed_int4_lookup);
    cp.setBindGroup(
      0,
      this.bg('embed_int4_lookup', { 0: uEmb, 1: wBuf, 2: this.stepStateBuf, 6: this.embCodeBuf })
    );
    cp.dispatchWorkgroups((c.embed_rank + 255) >>> 8, 1, 1);
    this.dispatchVq4Linear(cp, emb.up_proj, wBuf, 0, this.embCodeBuf, this.hCurBuf);
    cp.end();
  }

  private encodeSingleLayer(
    enc: any,
    l: number,
    wBuf: any,
    u32Sub: number,
    routeOnGolden: boolean,
    flags: number,
    recordExperts: boolean
  ): void {
    const c = this.cpuRef.cfg;
    const lm = this.cpuRef.manifest.layers[l];
    const qDim = c.num_q_heads * c.head_dim;
    const kvDim = c.num_kv_heads * c.head_dim;
    const ropePairs = (c.num_q_heads + c.num_kv_heads) * (c.head_dim >> 1);
    const scale = 1.0 / Math.sqrt(c.head_dim);

    const cp = enc.beginComputePass();

    const uLn1 = this.uBuf(packParams64([c.hidden_size, lm.ln1_u32_off - u32Sub], [c.rms_norm_eps]));
    cp.setPipeline(this.pipelines.rmsnorm_f32);
    cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uLn1, 1: wBuf, 3: this.hCurBuf, 6: this.normBuf }));
    cp.dispatchWorkgroups(1, 1, 1);

    this.dispatchVq4Linear(cp, lm.projs.q_proj, wBuf, u32Sub, this.normBuf, this.qBuf);
    this.dispatchVq4Linear(cp, lm.projs.k_proj, wBuf, u32Sub, this.normBuf, this.kBuf);
    this.dispatchVq4Linear(cp, lm.projs.v_proj, wBuf, u32Sub, this.normBuf, this.vBuf);

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

    // Shader #25: Dual-Tier O(1) SCache Eviction + Surprise Micro-Pages + Widrow-Hoff W_scar
    const uEv = this.uBuf(
      packParams64(
        [
          l,
          c.num_kv_heads,
          c.head_dim,
          c.num_sink,
          c.slide_len,
          c.num_micro_pages,
          this.countsU32Total,
          flags,
        ],
        [c.micro_page_theta]
      )
    );
    cp.setPipeline(this.pipelines.scarnative_v6_hierarchical_evict_update);
    cp.setBindGroup(
      0,
      this.bg('scarnative_v6_hierarchical_evict_update', {
        0: uEv,
        2: this.stepStateBuf,
        3: this.kBuf,
        4: this.vBuf,
        5: this.kRawGateBufs[l],
        6: this.scacheWScarBuf,
        7: this.scachePageScoresBuf,
        8: this.scacheKvBuf,
      })
    );
    cp.dispatchWorkgroups(c.num_kv_heads, 1, 1);

    // Shader #26: Dual-Tier O(1) GQA Attention
    const uAtt = this.uBuf(
      packParams64(
        [
          l,
          c.num_q_heads,
          c.num_kv_heads,
          c.head_dim,
          c.num_sink,
          c.slide_len,
          c.num_micro_pages,
          ((flags << 28) | this.countsU32Total) >>> 0,
        ],
        [scale]
      )
    );
    cp.setPipeline(this.pipelines.scarnative_v6_hierarchical_o1_gqa_attention);
    cp.setBindGroup(
      0,
      this.bg('scarnative_v6_hierarchical_o1_gqa_attention', {
        0: uAtt,
        1: this.scacheKvBuf,
        3: this.qBuf,
        4: this.scacheWScarBuf,
        5: this.qRawAlphaBufs[l],
        6: this.attnOutBuf,
      })
    );
    cp.dispatchWorkgroups(c.num_q_heads, 1, 1);

    this.dispatchVq4Linear(cp, lm.projs.o_proj, wBuf, u32Sub, this.attnOutBuf, this.projOutBuf);

    const uAdd = this.uBuf(packParams64([c.hidden_size]));
    cp.setPipeline(this.pipelines.vec_add_inplace_f32);
    cp.setBindGroup(0, this.bg('vec_add_inplace_f32', { 0: uAdd, 3: this.projOutBuf, 6: this.hCurBuf }));
    cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);

    const uLn2 = this.uBuf(packParams64([c.hidden_size, lm.ln2_u32_off - u32Sub], [c.rms_norm_eps]));
    cp.setPipeline(this.pipelines.rmsnorm_f32);
    cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uLn2, 1: wBuf, 3: this.hCurBuf, 6: this.normBuf }));
    cp.dispatchWorkgroups(1, 1, 1);

    this.dispatchVq4Linear(cp, lm.projs.gate_proj, wBuf, u32Sub, this.normBuf, this.gateBuf);
    this.dispatchVq4Linear(cp, lm.projs.up_proj, wBuf, u32Sub, this.normBuf, this.upBuf);

    const uSw = this.uBuf(packParams64([c.intermediate_size]));
    cp.setPipeline(this.pipelines.swiglu_mul_f32);
    cp.setBindGroup(0, this.bg('swiglu_mul_f32', { 0: uSw, 3: this.upBuf, 6: this.gateBuf }));
    cp.dispatchWorkgroups((c.intermediate_size + 255) >>> 8, 1, 1);

    this.dispatchVq4Linear(cp, lm.projs.down_proj, wBuf, u32Sub, this.gateBuf, this.projOutBuf);
    cp.setPipeline(this.pipelines.vec_add_inplace_f32);
    cp.setBindGroup(0, this.bg('vec_add_inplace_f32', { 0: uAdd, 3: this.projOutBuf, 6: this.hCurBuf }));
    cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);

    // Shaders #23 & #24: Native Mixture of Scars Top-k Directional ReLU Scar Experts
    if (lm.mos_experts) {
      const mos = lm.mos_experts;
      const uM1 = this.uBuf(
        packParams64([
          c.hidden_size,
          mos.num_experts,
          mos.top_k,
          mos.rank,
          mos.router_packed_u32_off - u32Sub,
          mos.router_scales_u32_off - u32Sub,
          mos.a_exp_packed_u32_off - u32Sub,
          mos.a_exp_scales_u32_off - u32Sub,
        ])
      );
      cp.setPipeline(this.pipelines.scarnative_v6_mos_router_topk_relu);
      cp.setBindGroup(
        0,
        this.bg('scarnative_v6_mos_router_topk_relu', {
          0: uM1,
          1: wBuf,
          3: this.normBuf,
          6: this.mosUBuf,
          8: this.mosSelBuf,
        })
      );
      cp.dispatchWorkgroups(1, 1, 1);

      const uM2 = this.uBuf(
        packParams64([
          c.hidden_size,
          mos.top_k,
          mos.rank,
          mos.b_exp_packed_u32_off - u32Sub,
          mos.b_exp_scales_u32_off - u32Sub,
        ])
      );
      cp.setPipeline(this.pipelines.scarnative_v6_mos_bscar_accumulate);
      cp.setBindGroup(
        0,
        this.bg('scarnative_v6_mos_bscar_accumulate', {
          0: uM2,
          1: wBuf,
          2: this.mosSelBuf,
          3: this.mosUBuf,
          6: this.hCurBuf,
        })
      );
      cp.dispatchWorkgroups((c.hidden_size + 255) >>> 8, 1, 1);
    }

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

    cp.end();

    if (recordExperts && lm.mos_experts) {
      enc.copyBufferToBuffer(this.mosSelBuf, 0, this.mosSelReadBuf, l * 16, 16);
    }
  }

  private encodeHead(enc: any, streamStages: boolean, suppressCjk: boolean): void {
    const c = this.cpuRef.cfg;
    const m = this.cpuRef.manifest;
    const wBuf = streamStages ? this.headTailBuf : this.shardBuf;
    const cp = enc.beginComputePass();
    const uFn = this.uBuf(packParams64([c.hidden_size, m.final_norm_u32_off], [c.rms_norm_eps]));
    cp.setPipeline(this.pipelines.rmsnorm_f32);
    cp.setBindGroup(0, this.bg('rmsnorm_f32', { 0: uFn, 1: wBuf, 3: this.hCurBuf, 6: this.normBuf }));
    cp.dispatchWorkgroups(1, 1, 1);

    this.dispatchVq4Linear(cp, m.lm_head, wBuf, 0, this.normBuf, this.logitsBuf);

    const uGate = this.uBuf(packParams64([c.vocab_size, suppressCjk ? 1 : 0], [64.0]));
    cp.setPipeline(this.pipelines.apply_script_subspace_gate);
    cp.setBindGroup(0, this.bg('apply_script_subspace_gate', { 0: uGate, 1: this.cjkMaskBuf, 6: this.logitsBuf }));
    cp.dispatchWorkgroups((c.vocab_size + 255) >>> 8, 1, 1);
    cp.end();
    enc.copyBufferToBuffer(this.logitsBuf, 0, this.logitsReadBuf, 0, c.vocab_size * 4);
  }

  private submitLayerSpan(
    startL: number,
    endL: number,
    streamStages: boolean,
    routeOnGolden: boolean,
    flags: number,
    includeHead: boolean,
    suppressCjk: boolean,
    copyGolden: boolean,
    recordExperts: boolean
  ): void {
    const c = this.cpuRef.cfg;
    const m = this.cpuRef.manifest;

    if (!streamStages) {
      const enc = this.device.createCommandEncoder();
      if (startL === 0) this.encodeEmbed(enc, false);
      for (let l = startL; l < endL; l++) {
        this.encodeSingleLayer(enc, l, this.shardBuf, 0, routeOnGolden, flags, recordExperts);
      }
      if (copyGolden) {
        enc.copyBufferToBuffer(this.hCurBuf, 0, this.goldenReadBuf, 0, c.hidden_size * 4);
      }
      if (includeHead) {
        this.encodeHead(enc, false, suppressCjk);
      }
      this.device.queue.submit([enc.finish()]);
      return;
    }

    if (startL === 0) {
      const encE = this.device.createCommandEncoder();
      this.encodeEmbed(encE, true);
      this.device.queue.submit([encE.finish()]);
    }

    for (const sm of m.stages) {
      const sLo = Math.max(startL, sm.layer_start);
      const sHi = Math.min(endL, sm.layer_end);
      if (sLo >= sHi) continue;
      const slice = this.stageSlicesU8[sm.stage_idx];
      this.device.queue.writeBuffer(this.stageWindowBuf, 0, slice.buffer, slice.byteOffset, slice.byteLength);

      const encS = this.device.createCommandEncoder();
      for (let l = sLo; l < sHi; l++) {
        this.encodeSingleLayer(encS, l, this.stageWindowBuf, sm.start_u32, routeOnGolden, flags, recordExperts);
      }
      if (copyGolden && sHi === endL) {
        encS.copyBufferToBuffer(this.hCurBuf, 0, this.goldenReadBuf, 0, c.hidden_size * 4);
      }
      this.device.queue.submit([encS.finish()]);
    }

    if (includeHead) {
      const encH = this.device.createCommandEncoder();
      this.encodeHead(encH, true, suppressCjk);
      this.device.queue.submit([encH.finish()]);
    }
  }

  private async stepTokenGpu(
    tokenId: number,
    posT: number,
    opts: {
      routeOnGolden: boolean;
      suppressCjk: boolean;
      useScarStore: boolean;
      streamStages: boolean;
      queryLang?: V6PolyglotLang | null;
      enableWidrowHoff?: boolean;
      enableMicroPages?: boolean;
      recordExperts?: boolean;
    }
  ): Promise<{
    logits: Float32Array;
    routedCartridgeId: string | null;
    routedDocId: string | null;
    routedCosine: number;
    resolvedLangIdx: number;
    selectedExpertsByLayer: number[][];
  }> {
    const c = this.cpuRef.cfg;
    const enWh = opts.enableWidrowHoff ?? c.enable_widrow_hoff;
    const enMp = opts.enableMicroPages ?? c.enable_micro_pages;
    const flags = (enWh ? 1 : 0) | (enMp ? 2 : 0);
    const recExp = opts.recordExperts ?? false;

    this.device.queue.writeBuffer(this.stepStateBuf, 0, new Uint32Array([tokenId >>> 0, posT >>> 0, 0, 0]));

    let routedCos = -1.0;
    let resolvedLangIdx = this.cpuRef.resolveLangIdx(opts.queryLang);

    if (opts.routeOnGolden) {
      this.submitLayerSpan(
        0,
        c.golden_layer,
        opts.streamStages,
        true,
        flags,
        false,
        opts.suppressCjk,
        true,
        recExp
      );
      const hGold = await this.readF32(this.goldenReadBuf, c.hidden_size);

      if (opts.useScarStore) {
        const rt = this.cpuRef.routeCartridge(hGold, opts.queryLang);
        this.activeCartridgeId = rt.cartridgeId;
        this.activeDocId = rt.docId;
        routedCos = rt.cosine;
        resolvedLangIdx = rt.langIdx;
        if (this.activeCartridgeId) {
          const folded = this.cpuRef.buildFoldedCartridgeBuffer(this.activeCartridgeId, rt.langIdx);
          if (folded) {
            this.device.queue.writeBuffer(this.activeCapBuf, 0, folded);
          } else {
            this.activeCartridgeId = null;
            this.activeDocId = null;
          }
        }
      } else {
        resolvedLangIdx = this.cpuRef.resolveLangIdx(opts.queryLang, hGold);
        this.activeCartridgeId = null;
        this.activeDocId = null;
      }

      if (this.activeCartridgeId) {
        const encInj = this.device.createCommandEncoder();
        const cpI = encInj.beginComputePass();
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
        this.device.queue.submit([encInj.finish()]);
      }

      this.submitLayerSpan(
        c.golden_layer,
        c.num_layers,
        opts.streamStages,
        false,
        flags,
        true,
        opts.suppressCjk,
        false,
        recExp
      );
    } else {
      this.submitLayerSpan(
        0,
        c.num_layers,
        opts.streamStages,
        false,
        flags,
        true,
        opts.suppressCjk,
        false,
        recExp
      );
    }

    const logits = await this.readF32(this.logitsReadBuf, c.vocab_size);
    const selectedExpertsByLayer: number[][] = [];
    if (recExp) {
      const selRaw = await this.readU32(this.mosSelReadBuf, c.num_layers * 4);
      for (let l = 0; l < c.num_layers; l++) {
        const row: number[] = [];
        for (let k = 0; k < c.mos_top_k; k++) row.push(selRaw[l * 4 + k]);
        selectedExpertsByLayer.push(row);
      }
    }

    return {
      logits,
      routedCartridgeId: this.activeCartridgeId,
      routedDocId: this.activeDocId,
      routedCosine: routedCos,
      resolvedLangIdx,
      selectedExpertsByLayer,
    };
  }

  async generate(
    prompt: string,
    opts: {
      maxNewTokens?: number;
      useScarStore?: boolean;
      queryLang?: V6PolyglotLang | null;
      streamStages?: boolean;
      enableWidrowHoff?: boolean;
      enableMicroPages?: boolean;
      suppressCjk?: boolean;
    } = {}
  ): Promise<ScarNativeV6GenerationReport> {
    if (!this.webgpuReady) {
      return this.cpuRef.generate(prompt, opts);
    }

    const t0 = performance.now();
    this.resetGpuSCache();
    const scacheInit = Number(this.scacheKvBuf.size + this.scacheWScarBuf.size + this.scachePageScoresBuf.size);
    const pIds = this.cpuRef.tokenizer.encode(prompt, true, false);
    const maxNew = opts.maxNewTokens ?? 20;
    const suppressCjk = opts.suppressCjk ?? true;
    const useScarStore = opts.useScarStore ?? true;
    const streamStages = opts.streamStages ?? false;

    let lastLogits: Float32Array<ArrayBufferLike> = new Float32Array(this.cpuRef.cfg.vocab_size);
    let routedId: string | null = null;
    let routedDoc: string | null = null;
    let routedCos = -1.0;
    let resolvedLangIdx = this.cpuRef.resolveLangIdx(opts.queryLang);
    let lastExperts: number[][] = [];

    for (let i = 0; i < pIds.length; i++) {
      const isLast = i === pIds.length - 1;
      const res = await this.stepTokenGpu(pIds[i], i, {
        routeOnGolden: isLast,
        suppressCjk,
        useScarStore,
        streamStages,
        queryLang: opts.queryLang,
        enableWidrowHoff: opts.enableWidrowHoff,
        enableMicroPages: opts.enableMicroPages,
        recordExperts: isLast,
      });
      lastLogits = res.logits;
      if (isLast) {
        routedId = res.routedCartridgeId;
        routedDoc = res.routedDocId;
        routedCos = res.routedCosine;
        resolvedLangIdx = res.resolvedLangIdx;
        lastExperts = res.selectedExpertsByLayer;
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
      const res = await this.stepTokenGpu(arg1, posT, {
        routeOnGolden: false,
        suppressCjk,
        useScarStore,
        streamStages,
        queryLang: opts.queryLang,
        enableWidrowHoff: opts.enableWidrowHoff,
        enableMicroPages: opts.enableMicroPages,
        recordExperts: false,
      });
      lastLogits = res.logits;
      posT++;
    }

    // Read GPU micro-page promotion counts (`counts[gh * 4 + 3]`)
    const encC = this.device.createCommandEncoder();
    encC.copyBufferToBuffer(this.scacheKvBuf, 0, this.countsReadBuf, 0, this.countsU32Total * 4);
    this.device.queue.submit([encC.finish()]);
    const cntArr = await this.readU32(this.countsReadBuf, this.countsU32Total);
    let promotedPages = 0;
    for (let gh = 0; gh * 4 + 3 < cntArr.length; gh++) {
      promotedPages += cntArr[gh * 4 + 3];
    }

    const elapsedMs = Math.max(0.1, performance.now() - t0);
    const scacheFinal = Number(this.scacheKvBuf.size + this.scacheWScarBuf.size + this.scachePageScoresBuf.size);
    const activeWeights = streamStages
      ? Number(this.headTailBuf.size + this.stageWindowBuf.size)
      : Number(this.shardBuf.size);

    return {
      prompt,
      outputText: this.cpuRef.tokenizer.decode(outIds, true).trim(),
      outputTextWithTags: this.cpuRef.tokenizer.decode(outIds, false).trim(),
      outputTokenIds: outIds,
      abstained: outIds.includes(this.cpuRef.tokenizer.abstainId),
      prefillTokens: pIds.length,
      generatedTokens: outIds.length,
      elapsedMs,
      tokPerSec: ((pIds.length + outIds.length) / elapsedMs) * 1000.0,
      routedCartridgeId: routedId,
      routedDocId: routedDoc,
      routedCosine: routedCos,
      resolvedLang: V6_LANGS[resolvedLangIdx] ?? 'en',
      firstTokenConfidence: firstConf,
      firstTokenMargin: firstMargin,
      scacheBytesInitial: scacheInit,
      scacheBytesFinal: scacheFinal,
      scacheDeltaBytes: scacheFinal - scacheInit,
      activeWeightGpuBytes: activeWeights,
      streamStages,
      promotedMicroPagesTotal: promotedPages,
      selectedExpertsByLayer: lastExperts,
      backend: streamStages ? 'webgpu_wgsl_26_stream' : 'webgpu_wgsl_26',
      firstTokenLogits,
    };
  }
}
