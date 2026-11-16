/**
 * Native In-Browser WebGPU Engine (`MoS INT4+ASVD` + `VQ4-Scar` + `SCache O(1)` + `GoldenSliceScarStoreV4`)
 *
 * 100% Custom WGSL 1.0 Shaders (`mos_scache.wgsl`, 18 Compute Pipelines):
 * - Zero Python Backend, Zero ONNX Runtime, Zero `@huggingface/transformers`, Zero O(T) KV Cache
 * - Supports both `MoS Block-64 INT4 + ASVD` AND `VQ4-Scar` (`vq4_equilibrate_fwht64` + `mos_fused_vq4_int4scar_gemv`)
 * - Includes `P_perp_CJK` Script-Orthogonal Active Subspace Gate (`apply_script_subspace_gate`, `18.55 KB` bitmask)
 * - Persistent CacheStorage (`mos-webgpu-v4-assets-v1`) for static `< 44 MiB` MoS shards (`0 MB` repeat load)
 * - Live RFC 7233 `HTTP/1.1 206 Partial Content` streaming for `61.25 KB` (`62,720 B`) universal cartridges
 * - Strictly constant `SCache O(1)` (`4230.8 KB`, `Δ +0 B`) across multi-turn conversations
 */

import wgslBundledSource from './shaders/mos_scache.wgsl?raw';
import {
  MOS_WEBGPU_CACHE_NAME,
  QwenBPETokenizer,
  fetchWithPersistentCache,
  pruneLegacyMosWebgpuCaches,
} from './qwenBpeTokenizer';
import {
  ActiveCapsuleRuntime,
  BrowserScarStoreHTTP206,
  GoldenSliceRoutingDiagnostics,
  Http206Event,
  PORTFOLIO_DEFAULT_SYSTEM_PROMPT,
  computeHorizonScale,
  extractUserContentBounds,
  findSubsequenceLastEnd,
  selectOutOfConeEpistemicGate,
} from './scarstoreHttp206Browser';
import {
  BrowserPointScarFlowEngine,
  PointScarSampleResult,
} from './pointScarFlowEngine';

const MAX_ROUTING_USER_TOKENS = 192;
// 4 tokens per GPUCommandBuffer (~280 ms on 36-layer 3B) prevents macOS Metal / Dawn 5s GPU watchdog timeouts
// and allows <300 ms responsive AbortSignal cancellation + 60 FPS UI compositing during prefill.
const PREFILL_BATCH_TOKENS = 4;

function createAbortError(): Error {
  const err = new Error('Geração interrompida pelo usuário.');
  err.name = 'AbortError';
  return err;
}

// Standard WebGPU numeric bitflags (avoids requiring external `@webgpu/types` in `tsconfig.json`)
const GPU_BUF = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const GPU_MAP_READ = 0x0001;

type DispatchOp = [string, any, number];

export interface EngineLoadProgress {
  phase: 'adapter' | 'manifest' | 'tokenizer' | 'shards' | 'compile' | 'ready';
  pct: number;
  detail: string;
  loadedBytes?: number;
  totalBytes?: number;
  fromCache?: boolean;
}

export interface TokenStreamUpdate {
  text: string;
  deltaToken: string;
  generatedTokens: number;
  tokPerSec: number;
  scacheBytes: number;
  scacheDeltaBytes: number;
  confidence: number;
  margin: number;
  uncertainty: number;
  horizonScale: number;
  top1TokenId: number;
}

export interface EngineGenerationResult {
  answer: string;
  generatedTokenIds: number[];
  prefillTokens: number;
  sessionTokens: number;
  turnIndex: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  decodeTokPerSec: number;
  scacheBytesInitial: number;
  scacheBytesAfterPrefill: number;
  scacheBytesFinal: number;
  scacheDeltaBytes: number;
  vramWeightsMiB: number;
  activeCapsules: Array<{
    capsuleId: string;
    gate: number;
    fromCache: boolean;
    title?: string;
    titlePt?: string;
    category?: string;
  }>;
  routingDiagnostics?: GoldenSliceRoutingDiagnostics;
  turnHttp206Bytes: number;
  totalHttp206Bytes: number;
  httpEvents: Http206Event[];
  meanConfidenceP1: number;
  meanMarginP1P2: number;
  epistemicMode?: 'scar_grounded' | 'out_of_cone_guarded' | 'base_trunk';
  pointScarMs?: number;
  pointScarSteps?: number;
  pointScarShardBytes?: number;
}

import { buildCjkVocabBitmaskU32, hasCjkScript } from './scriptSubspaceGate';
export { buildCjkVocabBitmaskU32, hasCjkScript };

function packKernelParams(
  ints: Partial<Record<'p0' | 'p1' | 'p2' | 'p3' | 'p4' | 'p5' | 'p6' | 'p7', number>> = {},
  floats: Partial<Record<'f0' | 'f1' | 'f2' | 'f3', number>> = {}
): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = (ints.p0 ?? 0) >>> 0;
  u32[1] = (ints.p1 ?? 0) >>> 0;
  u32[2] = (ints.p2 ?? 0) >>> 0;
  u32[3] = (ints.p3 ?? 0) >>> 0;
  u32[4] = (ints.p4 ?? 0) >>> 0;
  u32[5] = (ints.p5 ?? 0) >>> 0;
  u32[6] = (ints.p6 ?? 0) >>> 0;
  u32[7] = (ints.p7 ?? 0) >>> 0;
  f32[8] = floats.f0 ?? 0.0;
  f32[9] = floats.f1 ?? 0.0;
  f32[10] = floats.f2 ?? 0.0;
  f32[11] = floats.f3 ?? 0.0;
  return buf;
}

export async function checkMosShardsCached(baseUrl = '/webgpu/qwen3b'): Promise<boolean> {
  if (typeof caches === 'undefined') return false;
  try {
    await pruneLegacyMosWebgpuCaches();
    const cleanBase = baseUrl.replace(/\/$/, '');
    const cache = await caches.open(MOS_WEBGPU_CACHE_NAME);
    const keys = await cache.keys();
    const shardHits = keys.filter((req) => req.url.includes(`${cleanBase}/mos_shard_`)).length;
    return cleanBase.includes('qwen3b') ? shardHits >= 40 : shardHits >= 8;
  } catch (_) {
    return false;
  }
}

function findRepeatedSixGramStart(ids: number[]): number {
  const n = ids.length;
  // Guard 0: Catch immediate 1-token x4 consecutive stutter (e.g. "gó gó gó gó")
  if (n >= 4 && ids[n - 1] === ids[n - 2] && ids[n - 2] === ids[n - 3] && ids[n - 3] === ids[n - 4]) {
    return Math.max(1, n - 3);
  }
  // Guard 0b: Catch immediate 2-token x4 consecutive oscillation (e.g. "ab ab ab ab")
  if (
    n >= 8 &&
    ids[n - 2] === ids[n - 4] &&
    ids[n - 4] === ids[n - 6] &&
    ids[n - 6] === ids[n - 8] &&
    ids[n - 1] === ids[n - 3] &&
    ids[n - 3] === ids[n - 5] &&
    ids[n - 5] === ids[n - 7]
  ) {
    return Math.max(2, n - 6);
  }
  const k = 8;
  if (ids.length >= k * 2) {
    const tailStart = ids.length - k;
    for (let i = 0; i <= tailStart - k; i++) {
      let match = true;
      for (let j = 0; j < k; j++) {
        if (ids[i + j] !== ids[tailStart + j]) {
          match = false;
          break;
        }
      }
      if (match) return tailStart;
    }
  }
  // Guard 2: Catch 3x repeated 4-token subword stutters (e.g. "-Web-GL-SCache-Web")
  const kShort = 4;
  if (ids.length >= kShort * 4) {
    const tailStart = ids.length - kShort;
    let hits = 0;
    let firstHit = -1;
    for (let i = Math.max(0, ids.length - 40); i <= tailStart - kShort; i++) {
      let match = true;
      for (let j = 0; j < kShort; j++) {
        if (ids[i + j] !== ids[tailStart + j]) {
          match = false;
          break;
        }
      }
      if (match) {
        if (firstHit < 0) firstHit = i + kShort;
        hits++;
      }
    }
    if (hits >= 2 && firstHit >= 0) {
      return tailStart;
    }
  }
  return -1;
}

function hasCompoundHyphenDegeneration(text: string): boolean {
  // Detects pathological 4+ hyphenated acronym chains like "WebGL-Web-GL-SCache-Web"
  return /(?:[A-Za-z0-9]+-){4,}[A-Za-z0-9]+/.test(text);
}

function endsWithCompleteSentence(text: string): boolean {
  const t = text.trimEnd();
  if (t.length < 24) return false;
  return /[.!?]["'”)]?$/.test(t) && !/\b(?:Dr|Sr|Sra|Mr|Ms|Prof|etc|ex|vs|v\d)\.$/.test(t);
}

function startsNewSentenceOrParagraph(piece: string): boolean {
  return /^(?:\s*\n|\s+["'“([¡¿]*[A-ZÀ-ÖØ-Þ])/.test(piece);
}

function shouldStopAtSentenceBoundary(
  epistemicMode: 'scar_grounded' | 'out_of_cone_guarded' | 'base_trunk',
  generatedCount: number,
  decStepIdx: number,
  seedHorizon: number,
  maxNewTokens: number
): boolean {
  if (epistemicMode === 'scar_grounded') {
    return decStepIdx >= Math.max(22, seedHorizon - 13);
  }
  if (epistemicMode === 'out_of_cone_guarded') {
    return generatedCount >= 28;
  }
  return generatedCount >= Math.max(48, maxNewTokens - 48);
}

export class BrowserMoSWebGPUEngine {
  public readonly device: any;
  public readonly adapterLabel: string;
  public readonly manifest: any;
  public readonly tokenizer: QwenBPETokenizer;
  public readonly scarStore: BrowserScarStoreHTTP206 | null;
  public readonly shardBuffers: any[];
  public readonly loadedFromCache: boolean;

  public readonly hiddenDim: number;
  public readonly numLayers: number;
  public readonly numQHeads: number;
  public readonly numKvHeads: number;
  public readonly headDim: number;
  public readonly interDim: number;
  public readonly vocabSize: number;
  public readonly rmsNormEps: number;
  public readonly mosRank: number;
  public readonly eosTokenIds: Set<number>;
  public readonly userTurnPrefixIds: number[];

  public readonly windowSize: number;
  public readonly numSink: number;
  public readonly slideLen: number;
  public readonly numSlots: number;
  public readonly deltaBlend: number;
  public readonly totalWeightGpuBytes: number;

  private readonly shaderModule: any;
  private readonly pipelines: Record<string, any> = {};

  private countsU32Total = 0;
  private scacheKvBuf: any;
  private scacheWScarBuf: any;
  private scacheSalienceBuf: any;
  private _zeroKvBytes!: Uint8Array;
  private _zeroWScarBytes!: Uint8Array;
  private _zeroSalBytes!: Uint8Array;
  private scacheKvSnapBuf: any;
  private scacheWScarSnapBuf: any;
  private scacheSalSnapBuf: any;
  private scacheKvSysSnapBuf: any;
  private scacheWScarSysSnapBuf: any;
  private scacheSalSysSnapBuf: any;
  private _sysSnapTokenIds: number[] = [];
  private _sysSnapReady = false;
  private _deviceLostError: Error | null = null;
  private _deviceLostPromise!: Promise<never>;
  private turnSuffixIds: number[] = [];
  public sessionPos = 0;
  public turnCount = 0;
  private _activeGpuLock: Promise<void> = Promise.resolve();

  private stepStateBufs: any[] = [];
  private stepStateBuf: any;
  private prefillStepsBuf: any;
  private prefillXInReadBuf: any;
  private invFreqBuf: any;
  private hCurBuf: any;
  private normBuf: any;
  private qBuf: any;
  private kBuf: any;
  private vBuf: any;
  private qUnropedLastBuf: any;
  private attnOutBuf: any;
  private projOutBuf: any;
  private gateBuf: any;
  private upBuf: any;
  private asvdUBuf: any;
  private vq4XRotBuf: any;
  private vq4DummyBuf: any;
  private logitsBuf: any;
  private top2S1F32Buf: any;
  private top2S1U32Buf: any;
  private top2OutBuf: any;
  private readbackTop2Buf: any;
  private cjkBitmaskBuf: any;
  private scriptGateUniform: any;
  private cjkBitmaskReady = false;

  public readonly isGoldenSliceV4: boolean;
  public readonly midLayerIdx: number;
  public readonly scarRankMid: number;
  public readonly scarRankFinal: number;
  public readonly seedHorizon: number;
  public readonly seedFadeSteps: number;
  public readonly pointScarFlow: BrowserPointScarFlowEngine | null;

  private scarCalibBuf: any;
  private scarRank = 60;
  private scarNumProto = 8;
  private scarBeta = 14.0;
  private scarMidStride = 1;
  private scarCapStride = 1;
  private activeMidCapsulesBuf: any;
  private activeCapsulesBuf: any;
  private scarTelemetryBuf: any;
  private readbackScarTelBuf: any;
  private scarMidUniform: any;
  private scarInjectUniform: any;
  private currentMActive = 0;

  private embedOpsSlots: DispatchOp[][] = [];
  private layerOpsSlots: DispatchOp[][][] = [];
  private embedOps: DispatchOp[] = [];
  private layerOps: DispatchOp[][] = [];
  private finalNormOp!: DispatchOp;
  private scarMidInjectOp: DispatchOp | null = null;
  private scarInjectOp!: DispatchOp;
  private lmHeadOps: DispatchOp[] = [];
  private reduceOps: DispatchOp[] = [];

  constructor(opts: {
    device: any;
    adapterLabel: string;
    manifest: any;
    wgslSource: string;
    tokenizer: QwenBPETokenizer;
    scarStore: BrowserScarStoreHTTP206 | null;
    pointScarFlow?: BrowserPointScarFlowEngine | null;
    shardBuffers: any[];
    loadedFromCache: boolean;
  }) {
    this.device = opts.device;
    this.adapterLabel = opts.adapterLabel;
    this._deviceLostPromise = new Promise<never>((_, reject) => {
      if (this.device?.lost && typeof this.device.lost.then === 'function') {
        this.device.lost.then((info: any) => {
          const reason = info?.reason || 'unknown';
          if (reason === 'destroyed') return;
          const err = new Error(
            `GPU WebGPU reiniciada pelo driver (${reason}): ${info?.message || 'timeout de comando GPU'}`
          );
          (err as any).code = 'WEBGPU_DEVICE_LOST';
          this._deviceLostError = err;
          reject(err);
        });
      }
    });
    // Prevent unhandled rejection when device is idle
    this._deviceLostPromise.catch(() => {});
    this.manifest = opts.manifest;
    this.tokenizer = opts.tokenizer;
    this.scarStore = opts.scarStore;
    this.pointScarFlow = opts.pointScarFlow ?? null;
    this.shardBuffers = opts.shardBuffers;
    this.loadedFromCache = opts.loadedFromCache;

    const arch = opts.manifest.architecture;
    const scDef = opts.manifest.scache_defaults;

    this.hiddenDim = arch.hidden_dim;
    this.numLayers = arch.num_layers;
    this.numQHeads = arch.num_q_heads;
    this.numKvHeads = arch.num_kv_heads;
    this.headDim = arch.head_dim;
    this.interDim = arch.intermediate_size;
    this.vocabSize = arch.vocab_size;
    this.rmsNormEps = arch.rms_norm_eps;
    this.mosRank = arch.mos_rank;
    this.eosTokenIds = new Set<number>(arch.eos_token_ids || [151645, 151643]);
    this.userTurnPrefixIds = arch.user_turn_prefix_ids || [151644, 872, 198];

    const defaultSysTxt = this.scarStore ? this.scarStore.systemPrompt : PORTFOLIO_DEFAULT_SYSTEM_PROMPT;
    const minSysSinkTokens =
      this.tokenizer.applyChatTemplate([{ role: 'system', content: defaultSysTxt }], false).length + 12;
    this.numSlots = scDef.num_slots;
    const rawWindow = Math.max(scDef.window_size, minSysSinkTokens * 2);
    this.windowSize = Math.min(rawWindow, 512 - this.numSlots);
    this.numSink = Math.min(Math.max(scDef.num_sink_tokens, minSysSinkTokens), Math.floor(this.windowSize / 2));
    this.slideLen = Math.max(1, this.windowSize - this.numSink);
    this.deltaBlend = scDef.delta_blend;

    this.totalWeightGpuBytes = opts.shardBuffers.reduce((acc, b) => acc + b.size, 0);

    this.isGoldenSliceV4 = Boolean(this.scarStore && this.scarStore.isGoldenSliceV4);
    this.midLayerIdx =
      this.isGoldenSliceV4 && this.scarStore ? this.scarStore.midLayerIdx : Math.max(0, this.numLayers - 2);
    this.scarRankMid = this.isGoldenSliceV4 && this.scarStore ? this.scarStore.rankMid : 0;
    this.scarRankFinal = this.isGoldenSliceV4 && this.scarStore ? this.scarStore.rankFinal : 0;
    this.seedHorizon = this.isGoldenSliceV4 && this.scarStore ? this.scarStore.seedHorizon : 0;
    this.seedFadeSteps = this.isGoldenSliceV4 && this.scarStore ? this.scarStore.seedFadeSteps : 0;

    this.shaderModule = this.device.createShaderModule({
      label: 'mos_scache_kernels_v4',
      code: opts.wgslSource,
    });

    const entryPoints = [
      'embed_int4_lookup',
      'rmsnorm_f32',
      'fp16_gemv_bias',
      'mos_fused_int4_bscar_gemv',
      'qwen_rope_inplace',
      'scache_o1_update_widrow_hoff',
      'scache_o1_gqa_attention',
      'swiglu_mul_f32',
      'vec_add_inplace_f32',
      'vec_copy_f32',
      'head_expanded_scar_inject',
      'golden_slice_scar_inject',
      'int4_vocab_logits_gemv',
      'logits_top2_reduce_stage1',
      'logits_top2_reduce_stage2',
      'vq4_equilibrate_fwht64',
      'mos_fused_vq4_int4scar_gemv',
      'apply_script_subspace_gate',
      'vec_equilibrate_f16',
      'mos_fused_prot_int4_int4scar_gemv',
    ];
    for (const ep of entryPoints) {
      this.pipelines[ep] = this.device.createComputePipeline({
        label: `pipe_${ep}`,
        layout: 'auto',
        compute: { module: this.shaderModule, entryPoint: ep },
      });
    }

    this._allocateFixedBuffers(arch);
    this._prebuildDispatchPlan();
    this.resetSCache();
  }

  public static async create(
    baseUrl = '/webgpu/qwen3b',
    onProgress?: (p: EngineLoadProgress) => void
  ): Promise<BrowserMoSWebGPUEngine> {
    const navGpu = typeof navigator !== 'undefined' ? (navigator as any).gpu : undefined;
    if (!navGpu) {
      throw new Error(
        'WebGPU não está ativo neste navegador. Use Chrome 113+, Edge 113+ ou Safari 18+ com WebGPU habilitado.'
      );
    }

    const report = (
      phase: EngineLoadProgress['phase'],
      pct: number,
      detail = '',
      loadedBytes?: number,
      totalBytes?: number,
      fromCache?: boolean
    ) => {
      if (onProgress) onProgress({ phase, pct, detail, loadedBytes, totalBytes, fromCache });
    };

    report('adapter', 2, 'Solicitando adaptador físico WebGPU (high-performance)...');
    const adapter = await navGpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('Nenhum adaptador físico WebGPU foi encontrado pelo navegador.');
    }

    let info: any = adapter.info || {};
    if (!info.vendor && typeof adapter.requestAdapterInfo === 'function') {
      try {
        info = await adapter.requestAdapterInfo();
      } catch (_) {
        // Ignore privacy restrictions
      }
    }
    const adapterLabel =
      [info.description || info.device, info.vendor, info.architecture].filter(Boolean).join(' • ') ||
      'WebGPU Hardware Adapter';

    const device = await adapter.requestDevice();

    const cleanBase = baseUrl.replace(/\/$/, '');
    report('manifest', 5, 'Carregando manifesto MoS + compilando 20 shaders WGSL 1.0...');
    const manifestResp = await fetch(`${cleanBase}/webgpu_manifest.json`, { cache: 'no-cache' });
    if (!manifestResp.ok) {
      throw new Error(`HTTP ${manifestResp.status} ao carregar webgpu_manifest.json`);
    }
    const manifest = await manifestResp.json();
    const wgslSource = wgslBundledSource;

    const testMod = device.createShaderModule({ code: wgslSource });
    if (typeof testMod.getCompilationInfo === 'function') {
      const comp = await testMod.getCompilationInfo();
      const errs = (comp.messages || []).filter((m: any) => m.type === 'error');
      if (errs.length > 0) {
        throw new Error(
          `Erro de compilação WGSL: ${errs.map((e: any) => `L${e.lineNum}: ${e.message}`).join(' | ')}`
        );
      }
    }

    report('tokenizer', 9, 'Inicializando tokenizador BPE + roteador Golden-Slice v4.0 (P_perp_lang)...');
    const [tokenizer, scarStore] = await Promise.all([
      QwenBPETokenizer.fromUrl(`${cleanBase}/${manifest.tokenizer_file}`),
      manifest.scarstore_router_file
        ? BrowserScarStoreHTTP206.load(`${cleanBase}/${manifest.scarstore_router_file}`)
        : Promise.resolve(null),
    ]);

    const shards: Array<{
      shard_index: number;
      filename: string;
      byte_length: number;
      mib: number;
      mirror_url?: string;
    }> = manifest.shards || [];
    const totalBytes: number = manifest.total_weights_bytes || 1;
    let loadedBytes = 0;
    let allCached = shards.length > 0;
    const shardBuffers: any[] = [];

    for (let i = 0; i < shards.length; i++) {
      const s = shards[i];
      report(
        'shards',
        Math.round(12 + (loadedBytes / totalBytes) * 82),
        `Fatia MoS VQ4-Scar ${i + 1}/${shards.length}: ${s.filename} (${s.mib.toFixed(1)} MiB)...`,
        loadedBytes,
        totalBytes
      );
      let sFetch = await fetchWithPersistentCache(`${cleanBase}/${s.filename}`);
      if (!sFetch.response.ok && s.mirror_url) {
        sFetch = await fetchWithPersistentCache(s.mirror_url);
      }
      const { response: sResp, fromCache } = sFetch;
      if (!sResp.ok) {
        throw new Error(`Falha ao carregar ${s.filename}: HTTP ${sResp.status}`);
      }
      if (!fromCache) allCached = false;
      const arrBuf = await sResp.arrayBuffer();
      const gpuBuf = device.createBuffer({
        label: s.filename,
        size: arrBuf.byteLength,
        usage: GPU_BUF.STORAGE | GPU_BUF.COPY_DST,
      });
      device.queue.writeBuffer(gpuBuf, 0, arrBuf, 0, arrBuf.byteLength);
      shardBuffers.push(gpuBuf);
      loadedBytes += arrBuf.byteLength;
      report(
        'shards',
        Math.round(12 + (loadedBytes / totalBytes) * 82),
        `Fatia MoS VQ4-Scar ${i + 1}/${shards.length} alocada na GPU`,
        loadedBytes,
        totalBytes,
        allCached
      );
    }

    const pointScarBaseUrl = cleanBase.endsWith('/qwen3b')
      ? `${cleanBase.slice(0, -'/qwen3b'.length)}/pointscar_v1`
      : '/webgpu/pointscar_v1';
    const pointScarFlow = await BrowserPointScarFlowEngine.createOnDevice(device, pointScarBaseUrl);

    report('compile', 95, 'Compilando 20+12 pipelines WGSL e alocando SCache O(1) constante...', loadedBytes, totalBytes, allCached);
    const engine = new BrowserMoSWebGPUEngine({
      device,
      adapterLabel,
      manifest,
      wgslSource,
      tokenizer,
      scarStore,
      pointScarFlow,
      shardBuffers,
      loadedFromCache: allCached,
    });
    await engine.warmSystemPromptSnapshot((done, total) => {
      const pct = Math.min(99, 96 + Math.floor((done / Math.max(1, total)) * 4));
      report(
        'compile',
        pct,
        `Aquecendo SCache O(1) do sistema (${done}/${total} tokens)...`,
        loadedBytes,
        totalBytes,
        allCached
      );
    });
    report('ready', 100, 'Motor Nativo WebGPU / WGSL v4.0 pronto!', loadedBytes, totalBytes, allCached);
    return engine;
  }

  public get isDeviceLost(): boolean {
    return this._deviceLostError !== null;
  }

  public get scacheGpuBytes(): number {
    return this.scacheKvBuf.size + this.scacheWScarBuf.size + this.scacheSalienceBuf.size;
  }

  public get scacheKbFormatted(): string {
    return `${(this.scacheGpuBytes / 1024).toFixed(1)} KB`;
  }

  public async waitForIdle(): Promise<void> {
    await this._activeGpuLock;
    if (this.device?.queue && typeof this.device.queue.onSubmittedWorkDone === 'function') {
      try {
        await Promise.race([this.device.queue.onSubmittedWorkDone(), this._deviceLostPromise]);
      } catch (_) {
        // Ignore device-lost errors while draining idle queue
      }
    }
  }

  public destroy(): void {
    try {
      this.pointScarFlow?.destroy();
      for (const b of this.shardBuffers) b?.destroy?.();
      for (const b of this.stepStateBufs) b?.destroy?.();
      this.scacheKvBuf?.destroy?.();
      this.scacheWScarBuf?.destroy?.();
      this.scacheSalienceBuf?.destroy?.();
      this.scacheKvSnapBuf?.destroy?.();
      this.scacheWScarSnapBuf?.destroy?.();
      this.scacheSalSnapBuf?.destroy?.();
      this.scacheKvSysSnapBuf?.destroy?.();
      this.scacheWScarSysSnapBuf?.destroy?.();
      this.scacheSalSysSnapBuf?.destroy?.();
      this.device?.destroy?.();
    } catch (_) {
      // Ignore cleanup errors
    }
  }

  private _allocateFixedBuffers(arch: any): void {
    const U = GPU_BUF;
    const totalKvHeads = this.numLayers * this.numKvHeads;
    this.countsU32Total = totalKvHeads * 4;
    const capPerHead = this.numSink + this.numSlots + this.slideLen;
    const vecU32 = this.headDim >> 1;
    const kvU32Words = this.countsU32Total + totalKvHeads * 2 * capPerHead * vecU32;

    this.scacheKvBuf = this.device.createBuffer({
      label: 'scache_kv_o1',
      size: kvU32Words * 4,
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this.scacheWScarBuf = this.device.createBuffer({
      label: 'scache_w_scar_o1',
      size: totalKvHeads * this.headDim * this.headDim * 4,
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this.scacheSalienceBuf = this.device.createBuffer({
      label: 'scache_salience_o1',
      size: Math.max(16, totalKvHeads * Math.max(1, this.numSlots) * 4),
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this._zeroKvBytes = new Uint8Array(this.scacheKvBuf.size);
    this._zeroWScarBytes = new Uint8Array(this.scacheWScarBuf.size);
    this._zeroSalBytes = new Uint8Array(this.scacheSalienceBuf.size);

    this.scacheKvSnapBuf = this.device.createBuffer({
      label: 'scache_kv_snap',
      size: this.scacheKvBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.scacheWScarSnapBuf = this.device.createBuffer({
      label: 'scache_w_scar_snap',
      size: this.scacheWScarBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.scacheSalSnapBuf = this.device.createBuffer({
      label: 'scache_sal_snap',
      size: this.scacheSalienceBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.scacheKvSysSnapBuf = this.device.createBuffer({
      label: 'scache_kv_sys_snap',
      size: this.scacheKvBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.scacheWScarSysSnapBuf = this.device.createBuffer({
      label: 'scache_w_scar_sys_snap',
      size: this.scacheWScarBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.scacheSalSysSnapBuf = this.device.createBuffer({
      label: 'scache_sal_sys_snap',
      size: this.scacheSalienceBuf.size,
      usage: U.COPY_DST | U.COPY_SRC,
    });
    this.turnSuffixIds = this.tokenizer.encode('<|im_end|>\n');
    this.sessionPos = 0;
    this.turnCount = 0;

    const qDim = this.numQHeads * this.headDim;
    const kvDim = this.numKvHeads * this.headDim;

    this.stepStateBufs = [];
    for (let s = 0; s < PREFILL_BATCH_TOKENS; s++) {
      this.stepStateBufs.push(this.device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST }));
    }
    this.stepStateBuf = this.stepStateBufs[0];
    this.prefillStepsBuf = this.device.createBuffer({
      size: PREFILL_BATCH_TOKENS * 16,
      usage: U.COPY_SRC | U.COPY_DST,
    });
    this.prefillXInReadBuf = this.device.createBuffer({
      size: MAX_ROUTING_USER_TOKENS * this.hiddenDim * 4,
      usage: U.COPY_DST | U.MAP_READ,
    });

    const invFreqF32 = new Float32Array(arch.rope_inv_freq);
    this.invFreqBuf = this.device.createBuffer({
      size: Math.max(16, invFreqF32.byteLength),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.device.queue.writeBuffer(this.invFreqBuf, 0, invFreqF32);

    this.hCurBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.normBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.qBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.kBuf = this.device.createBuffer({ size: kvDim * 4, usage: U.STORAGE });
    this.vBuf = this.device.createBuffer({ size: kvDim * 4, usage: U.STORAGE });
    this.qUnropedLastBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.attnOutBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE });
    this.projOutBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE });
    this.gateBuf = this.device.createBuffer({ size: this.interDim * 4, usage: U.STORAGE });
    this.upBuf = this.device.createBuffer({ size: this.interDim * 4, usage: U.STORAGE });
    this.asvdUBuf = this.device.createBuffer({ size: Math.max(256, this.mosRank) * 4, usage: U.STORAGE });
    const maxLinIn = Math.max(this.hiddenDim, qDim, this.interDim);
    this.vq4XRotBuf = this.device.createBuffer({ size: maxLinIn * 4, usage: U.STORAGE });
    this.vq4DummyBuf = this.device.createBuffer({ size: 16, usage: U.STORAGE });
    this.logitsBuf = this.device.createBuffer({ size: this.vocabSize * 4, usage: U.STORAGE | U.COPY_SRC });
    this.top2S1F32Buf = this.device.createBuffer({ size: 256 * 3 * 4, usage: U.STORAGE });
    this.top2S1U32Buf = this.device.createBuffer({ size: 256 * 4, usage: U.STORAGE });
    this.top2OutBuf = this.device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC });
    this.readbackTop2Buf = this.device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });

    const nMaskWords = Math.ceil(this.vocabSize / 32);
    this.cjkBitmaskBuf = this.device.createBuffer({
      size: Math.max(16, nMaskWords * 4),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.scriptGateUniform = this.device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
    this.cjkBitmaskReady = false;
    this._setScriptGateGpu(false);

    const calF32 = this.scarStore
      ? this.scarStore.getCalibrationBufferF32()
      : new Float32Array(this.hiddenDim + qDim);
    this.scarCalibBuf = this.device.createBuffer({
      size: calF32.byteLength,
      usage: U.STORAGE | U.COPY_DST,
    });
    this.device.queue.writeBuffer(this.scarCalibBuf, 0, calF32);

    this.scarRank = this.scarStore ? this.scarStore.rank : 60;
    this.scarNumProto = this.scarStore ? this.scarStore.numPrototypes : 8;
    this.scarBeta = this.scarStore ? this.scarStore.attentionBeta : 14.0;
    this.scarMidStride = this.scarStore ? this.scarStore.getMidCapsuleStrideFloats() : 1;
    this.scarCapStride = this.scarStore
      ? this.scarStore.getCapsuleStrideFloats()
      : 1 + 2 * (this.scarRank * this.hiddenDim) + this.scarNumProto * qDim;

    this.activeMidCapsulesBuf = this.device.createBuffer({
      size: Math.max(64, 8 * this.scarMidStride * 4),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.activeCapsulesBuf = this.device.createBuffer({
      size: Math.max(64, 8 * this.scarCapStride * 4),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.scarTelemetryBuf = this.device.createBuffer({
      size: 64,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    this.readbackScarTelBuf = this.device.createBuffer({
      size: 64,
      usage: U.COPY_DST | U.MAP_READ,
    });
    this.scarMidUniform = this.device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
    this.scarInjectUniform = this.device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
    this._setActiveCapsulesGpu([]);
  }

  private _makeUniform(buf64: ArrayBuffer): any {
    const u = this.device.createBuffer({
      size: 64,
      usage: GPU_BUF.UNIFORM | GPU_BUF.COPY_DST,
    });
    this.device.queue.writeBuffer(u, 0, buf64);
    return u;
  }

  private _bg(ep: string, bindingMap: Record<number, any>): any {
    const entries = Object.entries(bindingMap).map(([bIdx, buf]) => ({
      binding: Number(bIdx),
      resource: { buffer: buf, offset: 0, size: buf.size },
    }));
    return this.device.createBindGroup({
      layout: this.pipelines[ep].getBindGroupLayout(0),
      entries,
    });
  }

  private _setScriptGateGpu(suppressCjk: boolean, penalty = 64.0): void {
    const uBuf = packKernelParams(
      {
        p0: this.vocabSize,
        p1: suppressCjk ? 1 : 0,
      },
      { f0: penalty }
    );
    this.device.queue.writeBuffer(this.scriptGateUniform, 0, uBuf);
  }

  private _ensureCjkBitmaskGpu(): void {
    if (this.cjkBitmaskReady) return;
    const maskU32 = buildCjkVocabBitmaskU32(this.tokenizer, this.vocabSize);
    this.device.queue.writeBuffer(this.cjkBitmaskBuf, 0, maskU32);
    this.cjkBitmaskReady = true;
  }

  private _makeLinearOps(shardBuf: any, spec: any, xBuf: any, yBuf: any): DispatchOp[] {
    const ops: DispatchOp[] = [];
    const outDim = spec.out_dim;
    const inDim = spec.in_dim;
    const wgs = Math.ceil(outDim / 8);

    if (spec.mode === 'fp16') {
      const u = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: spec.w_u32_off,
          p3: spec.bias_u32_off,
          p4: spec.has_bias,
        })
      );
      ops.push(['fp16_gemv_bias', this._bg('fp16_gemv_bias', { 0: u, 1: shardBuf, 3: xBuf, 6: yBuf }), wgs]);
    } else if (spec.mode === 'vq4_scar') {
      const rank = spec.rank >>> 0;
      const uEq = this._makeUniform(
        packKernelParams({
          p0: inDim,
          p1: spec.d_eq_u32_off,
        })
      );
      ops.push([
        'vq4_equilibrate_fwht64',
        this._bg('vq4_equilibrate_fwht64', { 0: uEq, 1: shardBuf, 3: xBuf, 6: this.vq4XRotBuf }),
        inDim >>> 6,
      ]);

      if (rank > 0) {
        const uA = this._makeUniform(
          packKernelParams({
            p0: rank,
            p1: inDim,
            p2: 0,
            p3: spec.ascar_packed_u32_off,
            p4: spec.ascar_scales_u32_off,
          })
        );
        ops.push([
          'mos_fused_int4_bscar_gemv',
          this._bg('mos_fused_int4_bscar_gemv', {
            0: uA,
            1: shardBuf,
            3: xBuf,
            4: this.vq4DummyBuf,
            6: this.asvdUBuf,
          }),
          Math.ceil(rank / 8),
        ]);
      }

      const p7Bias = spec.has_bias ? (((1 << 31) | (spec.bias_u32_off >>> 0)) >>> 0) : 0;
      const uVq = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: rank,
          p3: spec.vq_codes_u32_off,
          p4: spec.vq_scales_u32_off,
          p5: spec.lut256_u32_off,
          p6: spec.bscar_u32_off,
          p7: p7Bias,
        })
      );
      ops.push([
        'mos_fused_vq4_int4scar_gemv',
        this._bg('mos_fused_vq4_int4scar_gemv', {
          0: uVq,
          1: shardBuf,
          3: this.vq4XRotBuf,
          4: this.asvdUBuf,
          6: yBuf,
        }),
        wgs,
      ]);
    } else if (spec.mode === 'prot_int4_scar') {
      const rank = spec.rank >>> 0;
      const hasDeq = Boolean(spec.has_d_eq);
      const xBaseBuf = hasDeq ? this.vq4XRotBuf : xBuf;
      if (hasDeq) {
        const uEq = this._makeUniform(
          packKernelParams({
            p0: inDim,
            p1: spec.d_eq_u32_off,
          })
        );
        ops.push([
          'vec_equilibrate_f16',
          this._bg('vec_equilibrate_f16', { 0: uEq, 1: shardBuf, 3: xBuf, 6: this.vq4XRotBuf }),
          (inDim + 255) >>> 8,
        ]);
      }

      if (rank > 0) {
        const uA = this._makeUniform(
          packKernelParams({
            p0: rank,
            p1: inDim,
            p2: 0,
            p3: spec.ascar_packed_u32_off,
            p4: spec.ascar_scales_u32_off,
          })
        );
        ops.push([
          'mos_fused_int4_bscar_gemv',
          this._bg('mos_fused_int4_bscar_gemv', {
            0: uA,
            1: shardBuf,
            3: xBuf,
            4: this.vq4DummyBuf,
            6: this.asvdUBuf,
          }),
          Math.ceil(rank / 8),
        ]);
      }

      const uProt = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: rank,
          p3: spec.packed_u32_off,
          p4: spec.scales_u32_off,
          p5: spec.bscar_u32_off,
          p6: spec.bias_u32_off,
          p7: spec.has_bias ? 1 : 0,
        })
      );
      ops.push([
        'mos_fused_prot_int4_int4scar_gemv',
        this._bg('mos_fused_prot_int4_int4scar_gemv', {
          0: uProt,
          1: shardBuf,
          3: xBaseBuf,
          4: this.asvdUBuf,
          6: yBuf,
        }),
        wgs,
      ]);
    } else {
      const rank = spec.rank;
      const uDown = this._makeUniform(
        packKernelParams({ p0: rank, p1: inDim, p2: spec.ascar_u32_off, p3: 0, p4: 0 })
      );
      ops.push([
        'fp16_gemv_bias',
        this._bg('fp16_gemv_bias', { 0: uDown, 1: shardBuf, 3: xBuf, 6: this.asvdUBuf }),
        Math.ceil(rank / 8),
      ]);

      const uUp = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: rank,
          p3: spec.packed_u32_off,
          p4: spec.scales_u32_off,
          p5: spec.bscar_u32_off,
          p6: spec.bias_u32_off,
          p7: spec.has_bias,
        })
      );
      ops.push([
        'mos_fused_int4_bscar_gemv',
        this._bg('mos_fused_int4_bscar_gemv', {
          0: uUp,
          1: shardBuf,
          3: xBuf,
          4: this.asvdUBuf,
          6: yBuf,
        }),
        wgs,
      ]);
    }
    return ops;
  }

  private _prebuildDispatchPlan(): void {
    this.embedOpsSlots = Array.from({ length: PREFILL_BATCH_TOKENS }, () => []);
    for (const ch of this.manifest.embed_chunks) {
      const sb = this.shardBuffers[ch.shard_index];
      const u = this._makeUniform(
        packKernelParams({
          p0: this.hiddenDim,
          p1: ch.packed_u32_off,
          p2: ch.scales_u32_off,
          p3: ch.vocab_start,
          p4: ch.vocab_rows,
        })
      );
      const wgs = Math.ceil(this.hiddenDim / 256);
      for (let s = 0; s < PREFILL_BATCH_TOKENS; s++) {
        this.embedOpsSlots[s].push([
          'embed_int4_lookup',
          this._bg('embed_int4_lookup', { 0: u, 1: sb, 2: this.stepStateBufs[s], 6: this.hCurBuf }),
          wgs,
        ]);
      }
    }
    this.embedOps = this.embedOpsSlots[0];

    const qDim = this.numQHeads * this.headDim;
    const totalRopePairs = (this.numQHeads + this.numKvHeads) * (this.headDim >> 1);
    const uRope = this._makeUniform(
      packKernelParams({ p0: this.numQHeads, p1: this.numKvHeads, p2: this.headDim })
    );
    const bgRopeSlots = this.stepStateBufs.map((ssBuf) =>
      this._bg('qwen_rope_inplace', {
        0: uRope,
        2: ssBuf,
        3: this.invFreqBuf,
        6: this.qBuf,
        7: this.kBuf,
      })
    );

    const uAddH = this._makeUniform(packKernelParams({ p0: this.hiddenDim }));
    const bgAddH = this._bg('vec_add_inplace_f32', { 0: uAddH, 3: this.projOutBuf, 6: this.hCurBuf });

    const uSwiglu = this._makeUniform(packKernelParams({ p0: this.interDim }));
    const bgSwiglu = this._bg('swiglu_mul_f32', { 0: uSwiglu, 3: this.upBuf, 6: this.gateBuf });

    const uCopyQ = this._makeUniform(packKernelParams({ p0: qDim }));
    const bgCopyQ = this._bg('vec_copy_f32', { 0: uCopyQ, 3: this.qBuf, 6: this.qUnropedLastBuf });

    const attnScale = 1.0 / Math.sqrt(this.headDim);

    this.layerOpsSlots = Array.from({ length: PREFILL_BATCH_TOKENS }, () => []);
    for (let lIdx = 0; lIdx < this.manifest.layers.length; lIdx++) {
      const lMeta = this.manifest.layers[lIdx];
      const sb = this.shardBuffers[lMeta.shard_index];
      const projs = lMeta.projections;

      const uLn1 = this._makeUniform(
        packKernelParams({ p0: this.hiddenDim, p1: lMeta.input_layernorm_u32_off }, { f0: this.rmsNormEps })
      );
      const ln1Op: DispatchOp = [
        'rmsnorm_f32',
        this._bg('rmsnorm_f32', { 0: uLn1, 1: sb, 3: this.hCurBuf, 6: this.normBuf }),
        1,
      ];
      const qOps = this._makeLinearOps(sb, projs['self_attn.q_proj'], this.normBuf, this.qBuf);
      const kOps = this._makeLinearOps(sb, projs['self_attn.k_proj'], this.normBuf, this.kBuf);
      const vOps = this._makeLinearOps(sb, projs['self_attn.v_proj'], this.normBuf, this.vBuf);

      const uScUpd = this._makeUniform(
        packKernelParams({
          p0: lIdx,
          p1: this.numKvHeads,
          p2: this.headDim,
          p3: this.numSink,
          p4: this.numSlots,
          p5: this.slideLen,
          p6: this.countsU32Total,
        })
      );

      const uScAtt = this._makeUniform(
        packKernelParams(
          {
            p0: lIdx,
            p1: this.numQHeads,
            p2: this.numKvHeads,
            p3: this.headDim,
            p4: this.numSink,
            p5: this.numSlots,
            p6: this.slideLen,
            p7: this.countsU32Total,
          },
          { f0: attnScale, f1: this.deltaBlend }
        )
      );
      const scAttOp: DispatchOp = [
        'scache_o1_gqa_attention',
        this._bg('scache_o1_gqa_attention', {
          0: uScAtt,
          1: this.scacheKvBuf,
          3: this.qBuf,
          4: this.scacheWScarBuf,
          6: this.attnOutBuf,
        }),
        this.numQHeads,
      ];
      const oOps = this._makeLinearOps(sb, projs['self_attn.o_proj'], this.attnOutBuf, this.projOutBuf);
      const addHOp: DispatchOp = ['vec_add_inplace_f32', bgAddH, Math.ceil(this.hiddenDim / 256)];

      const uLn2 = this._makeUniform(
        packKernelParams({ p0: this.hiddenDim, p1: lMeta.post_attention_layernorm_u32_off }, { f0: this.rmsNormEps })
      );
      const ln2Op: DispatchOp = [
        'rmsnorm_f32',
        this._bg('rmsnorm_f32', { 0: uLn2, 1: sb, 3: this.hCurBuf, 6: this.normBuf }),
        1,
      ];
      const gateOps = this._makeLinearOps(sb, projs['mlp.gate_proj'], this.normBuf, this.gateBuf);
      const upOps = this._makeLinearOps(sb, projs['mlp.up_proj'], this.normBuf, this.upBuf);
      const swigluOp: DispatchOp = ['swiglu_mul_f32', bgSwiglu, Math.ceil(this.interDim / 256)];
      const downOps = this._makeLinearOps(sb, projs['mlp.down_proj'], this.gateBuf, this.projOutBuf);

      for (let s = 0; s < PREFILL_BATCH_TOKENS; s++) {
        const ops: DispatchOp[] = [ln1Op, ...qOps, ...kOps, ...vOps];
        if (lIdx === this.numLayers - 1) {
          ops.push(['vec_copy_f32', bgCopyQ, Math.ceil(qDim / 256)]);
        }
        ops.push(['qwen_rope_inplace', bgRopeSlots[s], Math.ceil(totalRopePairs / 64)]);
        ops.push([
          'scache_o1_update_widrow_hoff',
          this._bg('scache_o1_update_widrow_hoff', {
            0: uScUpd,
            2: this.stepStateBufs[s],
            3: this.kBuf,
            4: this.vBuf,
            6: this.scacheWScarBuf,
            7: this.scacheSalienceBuf,
            8: this.scacheKvBuf,
          }),
          this.numKvHeads,
        ]);
        ops.push(scAttOp, ...oOps, addHOp, ln2Op, ...gateOps, ...upOps, swigluOp, ...downOps, addHOp);
        this.layerOpsSlots[s].push(ops);
      }
    }
    this.layerOps = this.layerOpsSlots[0];

    const fnMeta = this.manifest.final_norm;
    const fnSb = this.shardBuffers[fnMeta.shard_index];
    const uFn = this._makeUniform(
      packKernelParams({ p0: this.hiddenDim, p1: fnMeta.w_u32_off }, { f0: this.rmsNormEps })
    );
    this.finalNormOp = [
      'rmsnorm_f32',
      this._bg('rmsnorm_f32', { 0: uFn, 1: fnSb, 3: this.hCurBuf, 6: this.normBuf }),
      1,
    ];

    if (this.isGoldenSliceV4) {
      this.scarMidInjectOp = [
        'golden_slice_scar_inject',
        this._bg('golden_slice_scar_inject', {
          0: this.scarMidUniform,
          3: this.scarCalibBuf,
          5: this.activeMidCapsulesBuf,
          6: this.hCurBuf,
          7: this.scarTelemetryBuf,
        }),
        1,
      ];
      this.scarInjectOp = [
        'golden_slice_scar_inject',
        this._bg('golden_slice_scar_inject', {
          0: this.scarInjectUniform,
          3: this.scarCalibBuf,
          5: this.activeCapsulesBuf,
          6: this.normBuf,
          7: this.scarTelemetryBuf,
        }),
        1,
      ];
    } else {
      this.scarMidInjectOp = null;
      this.scarInjectOp = [
        'head_expanded_scar_inject',
        this._bg('head_expanded_scar_inject', {
          0: this.scarInjectUniform,
          3: this.scarCalibBuf,
          4: this.qUnropedLastBuf,
          5: this.activeCapsulesBuf,
          6: this.normBuf,
          7: this.scarTelemetryBuf,
        }),
        1,
      ];
    }

    this.lmHeadOps = [];
    for (const ch of this.manifest.lm_head_chunks) {
      const sb = this.shardBuffers[ch.shard_index];
      const u = this._makeUniform(
        packKernelParams({
          p0: ch.vocab_rows,
          p1: this.hiddenDim,
          p2: ch.packed_u32_off,
          p3: ch.scales_u32_off,
          p4: ch.vocab_start,
        })
      );
      this.lmHeadOps.push([
        'int4_vocab_logits_gemv',
        this._bg('int4_vocab_logits_gemv', { 0: u, 1: sb, 3: this.normBuf, 6: this.logitsBuf }),
        Math.ceil(ch.vocab_rows / 8),
      ]);
    }

    const uR1 = this._makeUniform(packKernelParams({ p0: this.vocabSize }));
    this.reduceOps = [
      [
        'apply_script_subspace_gate',
        this._bg('apply_script_subspace_gate', {
          0: this.scriptGateUniform,
          1: this.cjkBitmaskBuf,
          6: this.logitsBuf,
        }),
        Math.ceil(this.vocabSize / 256),
      ],
      [
        'logits_top2_reduce_stage1',
        this._bg('logits_top2_reduce_stage1', {
          0: uR1,
          3: this.logitsBuf,
          6: this.top2S1F32Buf,
          8: this.top2S1U32Buf,
        }),
        256,
      ],
      [
        'logits_top2_reduce_stage2',
        this._bg('logits_top2_reduce_stage2', {
          1: this.top2S1U32Buf,
          3: this.top2S1F32Buf,
          8: this.top2OutBuf,
        }),
        1,
      ],
    ];
  }

  private _updateHorizonScaleGpu(stepIdx: number): void {
    if (!this.isGoldenSliceV4) return;
    const hw = computeHorizonScale(stepIdx, this.seedHorizon, this.seedFadeSteps);
    const mAct = this.currentMActive;
    const uMid = packKernelParams(
      { p0: this.hiddenDim, p1: this.scarRankMid, p2: mAct, p3: 0 },
      { f0: hw }
    );
    const uFin = packKernelParams(
      { p0: this.hiddenDim, p1: this.scarRankFinal, p2: mAct, p3: this.hiddenDim },
      { f0: hw }
    );
    this.device.queue.writeBuffer(this.scarMidUniform, 0, uMid);
    this.device.queue.writeBuffer(this.scarInjectUniform, 0, uFin);
  }

  private _setActiveCapsulesGpu(activeCapsules: ActiveCapsuleRuntime[], horizonScale = 1.0): void {
    const mAct = Math.min(8, activeCapsules.length);
    this.currentMActive = mAct;

    if (this.isGoldenSliceV4) {
      if (mAct > 0) {
        let wSum = 0.0;
        for (let i = 0; i < mAct; i++) wSum += activeCapsules[i].gate;
        const normDiv = mAct > 1 && wSum > 1.0 ? wSum : 1.0;

        const pkMid = new Float32Array(mAct * this.scarMidStride);
        const pkFin = new Float32Array(mAct * this.scarCapStride);
        for (let i = 0; i < mAct; i++) {
          const effG = activeCapsules[i].gate / normDiv;
          const bMid = i * this.scarMidStride;
          const bFin = i * this.scarCapStride;
          pkMid[bMid] = effG;
          if (activeCapsules[i].midFloats) pkMid.set(activeCapsules[i].midFloats!, bMid + 1);
          pkFin[bFin] = effG;
          if (activeCapsules[i].finalFloats) pkFin.set(activeCapsules[i].finalFloats!, bFin + 1);
        }
        this.device.queue.writeBuffer(this.activeMidCapsulesBuf, 0, pkMid);
        this.device.queue.writeBuffer(this.activeCapsulesBuf, 0, pkFin);
      }
      const uMid = packKernelParams(
        { p0: this.hiddenDim, p1: this.scarRankMid, p2: mAct, p3: 0 },
        { f0: horizonScale }
      );
      const uFin = packKernelParams(
        { p0: this.hiddenDim, p1: this.scarRankFinal, p2: mAct, p3: this.hiddenDim },
        { f0: horizonScale }
      );
      this.device.queue.writeBuffer(this.scarMidUniform, 0, uMid);
      this.device.queue.writeBuffer(this.scarInjectUniform, 0, uFin);
      return;
    }

    if (mAct > 0) {
      const packed = new Float32Array(mAct * this.scarCapStride);
      for (let i = 0; i < mAct; i++) {
        const base = i * this.scarCapStride;
        packed[base] = activeCapsules[i].gate;
        packed.set(activeCapsules[i].floats, base + 1);
      }
      this.device.queue.writeBuffer(this.activeCapsulesBuf, 0, packed);
    }
    const uBuf = packKernelParams(
      {
        p0: this.hiddenDim,
        p1: this.numQHeads,
        p2: this.headDim,
        p3: this.scarRank,
        p4: mAct,
        p5: this.scarNumProto,
      },
      { f0: this.scarBeta }
    );
    this.device.queue.writeBuffer(this.scarInjectUniform, 0, uBuf);
  }

  public resetSCache(): void {
    this.device.queue.writeBuffer(this.scacheKvBuf, 0, this._zeroKvBytes);
    this.device.queue.writeBuffer(this.scacheWScarBuf, 0, this._zeroWScarBytes);
    this.device.queue.writeBuffer(this.scacheSalienceBuf, 0, this._zeroSalBytes);
    this._setActiveCapsulesGpu([]);
    this.sessionPos = 0;
    this.turnCount = 0;
  }

  private _recordOps(cpass: any, ops: DispatchOp[]): void {
    for (let i = 0; i < ops.length; i++) {
      const [ep, bg, wgs] = ops[i];
      cpass.setPipeline(this.pipelines[ep]);
      cpass.setBindGroup(0, bg);
      cpass.dispatchWorkgroups(wgs, 1, 1);
    }
  }

  private async _waitQueueSafe(signal: AbortSignal | null = null): Promise<void> {
    if (this._deviceLostError) throw this._deviceLostError;
    if (signal?.aborted) throw createAbortError();
    if (this.device?.queue && typeof this.device.queue.onSubmittedWorkDone === 'function') {
      await Promise.race([this.device.queue.onSubmittedWorkDone(), this._deviceLostPromise]);
    }
    if (this._deviceLostError) throw this._deviceLostError;
    if (signal?.aborted) throw createAbortError();
  }

  private async _mapReadSafe(buf: any, byteLength: number): Promise<ArrayBuffer> {
    if (this._deviceLostError) throw this._deviceLostError;
    await Promise.race([buf.mapAsync(GPU_MAP_READ, 0, byteLength), this._deviceLostPromise]);
    const raw = buf.getMappedRange(0, byteLength).slice(0);
    buf.unmap();
    if (this._deviceLostError) throw this._deviceLostError;
    return raw;
  }

  public async warmSystemPromptSnapshot(onProgress?: (done: number, total: number) => void): Promise<void> {
    if (this._sysSnapReady) return;
    const defaultSysTxt = this.scarStore ? this.scarStore.systemPrompt : PORTFOLIO_DEFAULT_SYSTEM_PROMPT;
    const dummyProbe = this.tokenizer.applyChatTemplate(
      [
        { role: 'system', content: defaultSysTxt },
        { role: 'user', content: 'oi' },
      ],
      true
    );
    const bounds = this.isGoldenSliceV4
      ? extractUserContentBounds(
          dummyProbe,
          this.userTurnPrefixIds,
          this.scarStore ? this.scarStore.userTurnSuffixIds : null
        )
      : {
          qStart: Math.min(
            findSubsequenceLastEnd(dummyProbe, this.userTurnPrefixIds),
            Math.max(0, dummyProbe.length - 1)
          ),
          qEnd: dummyProbe.length,
        };
    const prefixLen = bounds.qStart;
    if (prefixLen <= 0 || prefixLen >= this.numSink) {
      return;
    }
    const sysIds = dummyProbe.slice(0, prefixLen);
    this.resetSCache();

    const slotArr = new Uint32Array(4);
    for (let batchStart = 0; batchStart < prefixLen; batchStart += PREFILL_BATCH_TOKENS) {
      const batchEnd = Math.min(prefixLen, batchStart + PREFILL_BATCH_TOKENS);
      const batchLen = batchEnd - batchStart;
      for (let i = 0; i < batchLen; i++) {
        const pos = batchStart + i;
        slotArr[0] = sysIds[pos];
        slotArr[1] = pos;
        slotArr[2] = 0;
        slotArr[3] = 0;
        this.device.queue.writeBuffer(this.stepStateBufs[i], 0, slotArr);
      }
      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < batchLen; i++) {
        const cpass = enc.beginComputePass();
        this._recordOps(cpass, this.embedOpsSlots[i]);
        for (let l = 0; l < this.numLayers; l++) {
          this._recordOps(cpass, this.layerOpsSlots[i][l]);
        }
        cpass.end();
      }
      this.device.queue.submit([enc.finish()]);
      await this._waitQueueSafe(null);
      if (onProgress) onProgress(batchEnd, prefixLen);
    }

    const encSnap = this.device.createCommandEncoder();
    encSnap.copyBufferToBuffer(this.scacheKvBuf, 0, this.scacheKvSysSnapBuf, 0, this.scacheKvBuf.size);
    encSnap.copyBufferToBuffer(this.scacheWScarBuf, 0, this.scacheWScarSysSnapBuf, 0, this.scacheWScarBuf.size);
    encSnap.copyBufferToBuffer(this.scacheSalienceBuf, 0, this.scacheSalSysSnapBuf, 0, this.scacheSalienceBuf.size);
    this.device.queue.submit([encSnap.finish()]);
    await this._waitQueueSafe(null);

    this._sysSnapTokenIds = sysIds;
    this._sysSnapReady = true;
    this.resetSCache();
  }

  private _matchesSysSnapPrefix(ids: number[]): boolean {
    const snap = this._sysSnapTokenIds;
    if (!this._sysSnapReady || snap.length === 0 || ids.length <= snap.length) {
      return false;
    }
    for (let i = 0; i < snap.length; i++) {
      if (ids[i] !== snap[i]) return false;
    }
    return true;
  }

  private async _restoreSysSnapToActiveSCache(): Promise<void> {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.scacheKvSysSnapBuf, 0, this.scacheKvBuf, 0, this.scacheKvBuf.size);
    enc.copyBufferToBuffer(this.scacheWScarSysSnapBuf, 0, this.scacheWScarBuf, 0, this.scacheWScarBuf.size);
    enc.copyBufferToBuffer(this.scacheSalSysSnapBuf, 0, this.scacheSalienceBuf, 0, this.scacheSalienceBuf.size);
    this.device.queue.submit([enc.finish()]);
    await this._waitQueueSafe(null);
  }

  private async _submitPrefillSequence(
    tokenIds: number[],
    basePos: number,
    qStart: number,
    captureXIn: boolean,
    qEnd: number | null = null,
    signal: AbortSignal | null = null,
    onPrefillProgress?: (done: number, total: number) => void,
    runFullLayersOnLastToken = false
  ): Promise<void> {
    const lp = tokenIds.length;
    if (lp === 0) return;
    const effQEnd = qEnd === null ? lp : Math.min(lp, Math.max(qStart + 1, qEnd));
    const splitIdx = this.isGoldenSliceV4 ? this.midLayerIdx : this.numLayers - 2;
    const slotArr = new Uint32Array(4);

    for (let batchStart = 0; batchStart < lp; batchStart += PREFILL_BATCH_TOKENS) {
      if (signal?.aborted) throw createAbortError();
      const batchEnd = Math.min(lp, batchStart + PREFILL_BATCH_TOKENS);
      const batchLen = batchEnd - batchStart;
      for (let i = 0; i < batchLen; i++) {
        const relPos = batchStart + i;
        slotArr[0] = tokenIds[relPos];
        slotArr[1] = basePos + relPos;
        slotArr[2] = 0;
        slotArr[3] = 0;
        this.device.queue.writeBuffer(this.stepStateBufs[i], 0, slotArr);
      }

      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < batchLen; i++) {
        const relPos = batchStart + i;
        const slotEmbed = this.embedOpsSlots[i];
        const slotLayers = this.layerOpsSlots[i];

        if (relPos < lp - 1 || runFullLayersOnLastToken) {
          if (captureXIn && relPos >= qStart && relPos < effQEnd) {
            const cpassPre = enc.beginComputePass();
            this._recordOps(cpassPre, slotEmbed);
            for (let l = 0; l <= splitIdx; l++) {
              this._recordOps(cpassPre, slotLayers[l]);
            }
            cpassPre.end();

            const uSlot = relPos - qStart;
            enc.copyBufferToBuffer(
              this.hCurBuf,
              0,
              this.prefillXInReadBuf,
              uSlot * this.hiddenDim * 4,
              this.hiddenDim * 4
            );

            const cpassUpper = enc.beginComputePass();
            for (let l = splitIdx + 1; l < this.numLayers; l++) {
              this._recordOps(cpassUpper, slotLayers[l]);
            }
            cpassUpper.end();
          } else {
            const cpass = enc.beginComputePass();
            this._recordOps(cpass, slotEmbed);
            for (let l = 0; l < this.numLayers; l++) {
              this._recordOps(cpass, slotLayers[l]);
            }
            cpass.end();
          }
        } else {
          const cpassPre = enc.beginComputePass();
          this._recordOps(cpassPre, slotEmbed);
          for (let l = 0; l <= splitIdx; l++) {
            this._recordOps(cpassPre, slotLayers[l]);
          }
          cpassPre.end();

          if (captureXIn && relPos >= qStart && relPos < effQEnd) {
            const uSlot = relPos - qStart;
            enc.copyBufferToBuffer(
              this.hCurBuf,
              0,
              this.prefillXInReadBuf,
              uSlot * this.hiddenDim * 4,
              this.hiddenDim * 4
            );
          }
        }
      }
      this.device.queue.submit([enc.finish()]);
      await this._waitQueueSafe(signal);
      if (onPrefillProgress) onPrefillProgress(batchEnd, lp);
    }
  }

  private async _flushClosingTokensToSCache(closingIds: number[], startPos: number): Promise<number> {
    if (!closingIds || closingIds.length === 0) return startPos;
    const len = closingIds.length;
    const slotArr = new Uint32Array(4);
    for (let batchStart = 0; batchStart < len; batchStart += PREFILL_BATCH_TOKENS) {
      const batchEnd = Math.min(len, batchStart + PREFILL_BATCH_TOKENS);
      const batchLen = batchEnd - batchStart;
      for (let i = 0; i < batchLen; i++) {
        slotArr[0] = closingIds[batchStart + i];
        slotArr[1] = startPos + batchStart + i;
        slotArr[2] = 0;
        slotArr[3] = 0;
        this.device.queue.writeBuffer(this.stepStateBufs[i], 0, slotArr);
      }
      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < batchLen; i++) {
        const cpass = enc.beginComputePass();
        this._recordOps(cpass, this.embedOpsSlots[i]);
        for (let l = 0; l < this.numLayers; l++) {
          this._recordOps(cpass, this.layerOpsSlots[i][l]);
        }
        cpass.end();
      }
      this.device.queue.submit([enc.finish()]);
      await this._waitQueueSafe(null);
    }
    return startPos + len;
  }

  private async _readTop2AndTelemetry(): Promise<{ top1Id: number; p1: number; margin: number }> {
    const raw = await this._mapReadSafe(this.readbackTop2Buf, 16);

    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);
    const top1Id = u32[0];
    const v1 = f32[1];
    const v2 = f32[2];
    const sexp = Math.max(f32[3], 1e-12);
    const p1 = 1.0 / sexp;
    const p2 = Math.exp(Math.max(-80.0, v2 - v1)) / sexp;
    return { top1Id, p1, margin: p1 - p2 };
  }

  public async generate(
    userQuery: string,
    options: {
      maxNewTokens?: number;
      enableScarStore?: boolean;
      enableEpistemicGuard?: boolean;
      resetCache?: boolean;
      systemPrompt?: string | null;
      onPrefillProgress?: (done: number, total: number) => void;
      onToken?: (u: TokenStreamUpdate) => void;
      onRouting?: (r: {
        active: ActiveCapsuleRuntime[];
        turnBytesFetched: number;
        httpEvents: Http206Event[];
        diagnostics?: GoldenSliceRoutingDiagnostics;
        pointScarSample?: PointScarSampleResult | null;
        epistemicMode?: 'scar_grounded' | 'out_of_cone_guarded' | 'base_trunk';
      }) => void;
      signal?: AbortSignal | null;
    } = {}
  ): Promise<EngineGenerationResult> {
    const prev = this._activeGpuLock;
    let releaseLock!: () => void;
    this._activeGpuLock = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });
    await prev;
    try {
      return await this._runGenerateUnlocked(userQuery, options);
    } finally {
      releaseLock();
    }
  }

  private async _runGenerateUnlocked(
    userQuery: string,
    options: {
      maxNewTokens?: number;
      enableScarStore?: boolean;
      enableEpistemicGuard?: boolean;
      resetCache?: boolean;
      systemPrompt?: string | null;
      onPrefillProgress?: (done: number, total: number) => void;
      onToken?: (u: TokenStreamUpdate) => void;
      onRouting?: (r: {
        active: ActiveCapsuleRuntime[];
        turnBytesFetched: number;
        httpEvents: Http206Event[];
        diagnostics?: GoldenSliceRoutingDiagnostics;
        pointScarSample?: PointScarSampleResult | null;
        epistemicMode?: 'scar_grounded' | 'out_of_cone_guarded' | 'base_trunk';
      }) => void;
      signal?: AbortSignal | null;
    } = {}
  ): Promise<EngineGenerationResult> {
    const {
      maxNewTokens = 96,
      enableScarStore = true,
      enableEpistemicGuard,
      resetCache = false,
      systemPrompt = null,
      onPrefillProgress,
      onToken,
      onRouting,
      signal = null,
    } = options;

    if (this._deviceLostError) throw this._deviceLostError;
    if (signal?.aborted) throw createAbortError();

    if (resetCache) {
      this.resetSCache();
    }

    const suppressCjk = !hasCjkScript(userQuery);
    if (suppressCjk) {
      this._ensureCjkBitmaskGpu();
    }
    this._setScriptGateGpu(suppressCjk);

    const basePos = this.sessionPos;
    const sysTxt =
      systemPrompt ?? (this.scarStore ? this.scarStore.systemPrompt : PORTFOLIO_DEFAULT_SYSTEM_PROMPT);

    const cleanProbeIds = this.tokenizer.applyChatTemplate(
      [
        { role: 'system', content: sysTxt },
        { role: 'user', content: userQuery },
      ],
      true
    );
    const lpProbe = cleanProbeIds.length;
    let qStartProbe = 0;
    let qEndProbe = lpProbe;
    if (this.isGoldenSliceV4) {
      const bounds = extractUserContentBounds(
        cleanProbeIds,
        this.userTurnPrefixIds,
        this.scarStore ? this.scarStore.userTurnSuffixIds : null
      );
      qEndProbe = bounds.qEnd;
      qStartProbe = Math.max(bounds.qStart, qEndProbe - MAX_ROUTING_USER_TOKENS);
    } else {
      const rawQStartProbe = Math.min(
        findSubsequenceLastEnd(cleanProbeIds, this.userTurnPrefixIds),
        Math.max(0, lpProbe - 1)
      );
      qStartProbe = Math.max(rawQStartProbe, lpProbe - MAX_ROUTING_USER_TOKENS);
      qEndProbe = lpProbe;
    }
    const luProbe = Math.max(1, qEndProbe - qStartProbe);

    const promptIds =
      basePos === 0
        ? cleanProbeIds
        : this.tokenizer.encode(`<|im_start|>user\n${userQuery}<|im_end|>\n<|im_start|>assistant\n`);
    let effectiveLp = promptIds.length;
    const scacheBytesInitial = this.scacheGpuBytes;
    const useEpistemicGuard =
      enableEpistemicGuard ?? Boolean(enableScarStore && this.scarStore && systemPrompt === null);
    let epistemicMode: 'scar_grounded' | 'out_of_cone_guarded' | 'base_trunk' = 'base_trunk';

    // Save pre-turn SCache snapshot on Turn 2+ so mid-prefill AbortError restores exact conversational state
    if (basePos > 0) {
      const encSave = this.device.createCommandEncoder();
      encSave.copyBufferToBuffer(this.scacheKvBuf, 0, this.scacheKvSnapBuf, 0, this.scacheKvBuf.size);
      encSave.copyBufferToBuffer(this.scacheWScarBuf, 0, this.scacheWScarSnapBuf, 0, this.scacheWScarBuf.size);
      encSave.copyBufferToBuffer(this.scacheSalienceBuf, 0, this.scacheSalSnapBuf, 0, this.scacheSalienceBuf.size);
      this.device.queue.submit([encSave.finish()]);
      await this._waitQueueSafe(signal);
    }

    const restorePreTurnSCache = async (): Promise<void> => {
      if (basePos === 0) {
        this.resetSCache();
      } else {
        const encRest = this.device.createCommandEncoder();
        encRest.copyBufferToBuffer(this.scacheKvSnapBuf, 0, this.scacheKvBuf, 0, this.scacheKvBuf.size);
        encRest.copyBufferToBuffer(this.scacheWScarSnapBuf, 0, this.scacheWScarBuf, 0, this.scacheWScarBuf.size);
        encRest.copyBufferToBuffer(this.scacheSalSnapBuf, 0, this.scacheSalienceBuf, 0, this.scacheSalienceBuf.size);
        this.device.queue.submit([encRest.finish()]);
        this.sessionPos = basePos;
      }
      await this._waitQueueSafe(null);
    };

    const t0 = performance.now();
    let routeResult: {
      active: ActiveCapsuleRuntime[];
      turnBytesFetched: number;
      httpEvents: Http206Event[];
      diagnostics?: GoldenSliceRoutingDiagnostics;
      primaryProbeF32?: Float32Array | null;
    } = {
      active: [],
      turnBytesFetched: 0,
      httpEvents: [],
      primaryProbeF32: null,
    };

    const canUseSysSnap = this._matchesSysSnapPrefix(cleanProbeIds) && qStartProbe >= this._sysSnapTokenIds.length;
    const sysPrefixLen = canUseSysSnap ? this._sysSnapTokenIds.length : 0;
    let guardedCleanRollbackPos: number | null = null;
    let guardedCleanPrefixIds: number[] | null = null;

    try {
      if (basePos === 0) {
        const canSplitAtQueryEnd =
          Boolean(enableScarStore && this.scarStore && this.isGoldenSliceV4) &&
          qEndProbe > qStartProbe &&
          qEndProbe < cleanProbeIds.length;

        if (canSplitAtQueryEnd && this.scarStore) {
          if (canUseSysSnap) {
            await this._restoreSysSnapToActiveSCache();
            await this._submitPrefillSequence(
              cleanProbeIds.slice(sysPrefixLen, qEndProbe),
              sysPrefixLen,
              qStartProbe - sysPrefixLen,
              true,
              qEndProbe - sysPrefixLen,
              signal,
              onPrefillProgress
                ? (done) => onPrefillProgress(Math.min(lpProbe, sysPrefixLen + done), lpProbe)
                : undefined,
              true
            );
          } else {
            await this._submitPrefillSequence(
              cleanProbeIds.slice(0, qEndProbe),
              0,
              qStartProbe,
              true,
              qEndProbe,
              signal,
              onPrefillProgress ? (done) => onPrefillProgress(Math.min(lpProbe, done), lpProbe) : undefined,
              true
            );
          }

          const readBytes = luProbe * this.hiddenDim * 4;
          const xinRaw = await this._mapReadSafe(this.prefillXInReadBuf, readBytes);
          if (signal?.aborted) throw createAbortError();
          const xinF32 = new Float32Array(xinRaw);
          routeResult = await this.scarStore.routeAndFetchActiveCapsules(xinF32, luProbe);
          if (signal?.aborted) throw createAbortError();
          this._setActiveCapsulesGpu(routeResult.active, 1.0);

          if (routeResult.active.length > 0 || !useEpistemicGuard) {
            epistemicMode = routeResult.active.length > 0 ? 'scar_grounded' : 'base_trunk';
            const tailIds = cleanProbeIds.slice(qEndProbe);
            effectiveLp = qEndProbe + tailIds.length;
            await this._submitPrefillSequence(
              tailIds,
              qEndProbe,
              0,
              false,
              null,
              signal,
              onPrefillProgress ? (done) => onPrefillProgress(qEndProbe + done, effectiveLp) : undefined,
              false
            );
          } else {
            epistemicMode = 'out_of_cone_guarded';
            const encSnapQ = this.device.createCommandEncoder();
            encSnapQ.copyBufferToBuffer(this.scacheKvBuf, 0, this.scacheKvSnapBuf, 0, this.scacheKvBuf.size);
            encSnapQ.copyBufferToBuffer(this.scacheWScarBuf, 0, this.scacheWScarSnapBuf, 0, this.scacheWScarBuf.size);
            encSnapQ.copyBufferToBuffer(this.scacheSalienceBuf, 0, this.scacheSalSnapBuf, 0, this.scacheSalienceBuf.size);
            this.device.queue.submit([encSnapQ.finish()]);
            await this._waitQueueSafe(signal);
            guardedCleanRollbackPos = qEndProbe;
            guardedCleanPrefixIds = cleanProbeIds.slice(qEndProbe);

            const gateTxt = selectOutOfConeEpistemicGate(userQuery);
            const guardedFullIds = this.tokenizer.applyChatTemplate(
              [
                { role: 'system', content: sysTxt },
                { role: 'user', content: `${userQuery}\n\n${gateTxt}` },
              ],
              true
            );
            effectiveLp = guardedFullIds.length;

            // Check if `guardedFullIds[0..qEndProbe]` matches the already-prefilled `cleanProbeIds[0..qEndProbe]`
            // so we avoid re-prefilling ~105 tokens and never inflate user-visible prefill token counts!
            let sharesQueryPrefix = guardedFullIds.length > qEndProbe;
            if (sharesQueryPrefix) {
              for (let k = 0; k < qEndProbe; k++) {
                if (guardedFullIds[k] !== cleanProbeIds[k]) {
                  sharesQueryPrefix = false;
                  break;
                }
              }
            }

            if (sharesQueryPrefix) {
              await this._submitPrefillSequence(
                guardedFullIds.slice(qEndProbe),
                qEndProbe,
                0,
                false,
                null,
                signal,
                onPrefillProgress ? () => onPrefillProgress(lpProbe, lpProbe) : undefined,
                false
              );
            } else if (canUseSysSnap && this._matchesSysSnapPrefix(guardedFullIds)) {
              await this._restoreSysSnapToActiveSCache();
              await this._submitPrefillSequence(
                guardedFullIds.slice(sysPrefixLen),
                sysPrefixLen,
                0,
                false,
                null,
                signal,
                onPrefillProgress ? () => onPrefillProgress(lpProbe, lpProbe) : undefined,
                false
              );
            } else {
              this.device.queue.writeBuffer(this.scacheKvBuf, 0, this._zeroKvBytes);
              this.device.queue.writeBuffer(this.scacheWScarBuf, 0, this._zeroWScarBytes);
              this.device.queue.writeBuffer(this.scacheSalienceBuf, 0, this._zeroSalBytes);
              await this._submitPrefillSequence(
                guardedFullIds,
                0,
                0,
                false,
                null,
                signal,
                onPrefillProgress ? () => onPrefillProgress(lpProbe, lpProbe) : undefined,
                false
              );
            }
          }
        } else {
          if (canUseSysSnap) {
            await this._restoreSysSnapToActiveSCache();
            await this._submitPrefillSequence(
              promptIds.slice(sysPrefixLen),
              sysPrefixLen,
              qStartProbe - sysPrefixLen,
              Boolean(enableScarStore && this.scarStore),
              qEndProbe - sysPrefixLen,
              signal,
              onPrefillProgress
            );
          } else {
            await this._submitPrefillSequence(
              promptIds,
              0,
              qStartProbe,
              Boolean(enableScarStore && this.scarStore),
              qEndProbe,
              signal,
              onPrefillProgress
            );
          }
          if (enableScarStore && this.scarStore) {
            const readBytes = luProbe * this.hiddenDim * 4;
            const xinRaw = await this._mapReadSafe(this.prefillXInReadBuf, readBytes);
            if (signal?.aborted) throw createAbortError();
            const xinF32 = new Float32Array(xinRaw);
            routeResult = await this.scarStore.routeAndFetchActiveCapsules(xinF32, luProbe);
            if (signal?.aborted) throw createAbortError();
            this._setActiveCapsulesGpu(routeResult.active, 1.0);
            epistemicMode = routeResult.active.length > 0 ? 'scar_grounded' : 'base_trunk';
          } else {
            if (this.scarStore) this.scarStore.lastActiveGates.clear();
            this._setActiveCapsulesGpu([]);
            epistemicMode = 'base_trunk';
          }
        }
      } else {
        if (enableScarStore && this.scarStore) {
          const probeEnd =
            this.isGoldenSliceV4 && qEndProbe > qStartProbe && qEndProbe < cleanProbeIds.length
              ? qEndProbe
              : cleanProbeIds.length;
          if (canUseSysSnap) {
            await this._restoreSysSnapToActiveSCache();
            await this._submitPrefillSequence(
              cleanProbeIds.slice(sysPrefixLen, probeEnd),
              sysPrefixLen,
              qStartProbe - sysPrefixLen,
              true,
              qEndProbe - sysPrefixLen,
              signal
            );
          } else {
            this.device.queue.writeBuffer(this.scacheKvBuf, 0, this._zeroKvBytes);
            this.device.queue.writeBuffer(this.scacheWScarBuf, 0, this._zeroWScarBytes);
            this.device.queue.writeBuffer(this.scacheSalienceBuf, 0, this._zeroSalBytes);
            await this._submitPrefillSequence(cleanProbeIds.slice(0, probeEnd), 0, qStartProbe, true, qEndProbe, signal);
          }

          const encRestore = this.device.createCommandEncoder();
          encRestore.copyBufferToBuffer(this.scacheKvSnapBuf, 0, this.scacheKvBuf, 0, this.scacheKvBuf.size);
          encRestore.copyBufferToBuffer(this.scacheWScarSnapBuf, 0, this.scacheWScarBuf, 0, this.scacheWScarBuf.size);
          encRestore.copyBufferToBuffer(this.scacheSalSnapBuf, 0, this.scacheSalienceBuf, 0, this.scacheSalienceBuf.size);
          this.device.queue.submit([encRestore.finish()]);
          await this._waitQueueSafe(signal);

          const readBytes = luProbe * this.hiddenDim * 4;
          const xinRaw = await this._mapReadSafe(this.prefillXInReadBuf, readBytes);
          if (signal?.aborted) throw createAbortError();
          const xinF32 = new Float32Array(xinRaw);
          routeResult = await this.scarStore.routeAndFetchActiveCapsules(xinF32, luProbe);
          if (signal?.aborted) throw createAbortError();
          this._setActiveCapsulesGpu(routeResult.active, 1.0);
        } else {
          if (this.scarStore) this.scarStore.lastActiveGates.clear();
          this._setActiveCapsulesGpu([]);
        }

        let turnPromptIds = promptIds;
        if (enableScarStore && this.scarStore && routeResult.active.length === 0 && useEpistemicGuard) {
          epistemicMode = 'out_of_cone_guarded';
          guardedCleanRollbackPos = basePos;
          guardedCleanPrefixIds = promptIds;
          const gateTxt = selectOutOfConeEpistemicGate(userQuery);
          turnPromptIds = this.tokenizer.encode(
            `<|im_start|>user\n${userQuery}\n\n${gateTxt}<|im_end|>\n<|im_start|>assistant\n`
          );
        } else {
          epistemicMode =
            enableScarStore && this.scarStore && routeResult.active.length > 0 ? 'scar_grounded' : 'base_trunk';
        }
        effectiveLp = turnPromptIds.length;
        const userVisibleTurnTokens = promptIds.length;
        await this._submitPrefillSequence(
          turnPromptIds,
          basePos,
          0,
          false,
          null,
          signal,
          onPrefillProgress
            ? (done, tot) =>
                onPrefillProgress(
                  Math.min(
                    userVisibleTurnTokens,
                    Math.max(1, Math.round((done / Math.max(1, tot)) * userVisibleTurnTokens))
                  ),
                  userVisibleTurnTokens
                )
            : undefined
        );
      }
    } catch (prefillErr) {
      await restorePreTurnSCache();
      throw prefillErr;
    }

    let pointScarSample: PointScarSampleResult | null = null;
    if (this.pointScarFlow && routeResult.primaryProbeF32 && !signal?.aborted) {
      try {
        pointScarSample = await this.pointScarFlow.sampleEuler(routeResult.primaryProbeF32);
      } catch (psfErr) {
        console.warn('[PointScarFlow] GPU 3D sample warning:', psfErr);
      }
    }

    if (onRouting) {
      onRouting({
        active: routeResult.active,
        turnBytesFetched: routeResult.turnBytesFetched,
        httpEvents: routeResult.httpEvents,
        diagnostics: routeResult.diagnostics,
        pointScarSample,
        epistemicMode,
      });
    }

    const splitIdx = this.isGoldenSliceV4 ? this.midLayerIdx : this.numLayers - 2;
    const encTail = this.device.createCommandEncoder();
    const cpassTail = encTail.beginComputePass();
    if (this.isGoldenSliceV4 && this.scarMidInjectOp) {
      this._recordOps(cpassTail, [this.scarMidInjectOp]);
    }
    for (let l = splitIdx + 1; l < this.numLayers; l++) {
      this._recordOps(cpassTail, this.layerOps[l]);
    }
    this._recordOps(cpassTail, [this.finalNormOp, this.scarInjectOp]);
    this._recordOps(cpassTail, this.lmHeadOps);
    this._recordOps(cpassTail, this.reduceOps);
    cpassTail.end();
    encTail.copyBufferToBuffer(this.top2OutBuf, 0, this.readbackTop2Buf, 0, 16);
    this.device.queue.submit([encTail.finish()]);

    const firstStep = await this._readTop2AndTelemetry();
    if (signal?.aborted) {
      await restorePreTurnSCache();
      throw createAbortError();
    }
    const prefillMs = performance.now() - t0;
    const scacheBytesAfterPrefill = this.scacheGpuBytes;

    const generatedIds: number[] = [];
    const confs: number[] = [];
    const margins: number[] = [];
    let lastDecoded = '';
    let lastGenTokenFed = true;

    if (!this.eosTokenIds.has(firstStep.top1Id) && !(signal && signal.aborted)) {
      generatedIds.push(firstStep.top1Id);
      lastGenTokenFed = false;
      confs.push(firstStep.p1);
      margins.push(firstStep.margin);
      const curText = this.tokenizer.decode(generatedIds, true);
      const delta = curText.slice(lastDecoded.length);
      lastDecoded = curText;
      if (onToken) {
        onToken({
          text: curText,
          deltaToken: delta,
          generatedTokens: generatedIds.length,
          tokPerSec: 0,
          scacheBytes: this.scacheGpuBytes,
          scacheDeltaBytes: this.scacheGpuBytes - scacheBytesInitial,
          confidence: firstStep.p1,
          margin: firstStep.margin,
          uncertainty: Math.max(0.0, 1.0 - firstStep.p1),
          horizonScale: this.isGoldenSliceV4 && this.currentMActive > 0 ? 1.0 : 0.0,
          top1TokenId: firstStep.top1Id,
        });
      }
    }

    const tDec0 = performance.now();
    let curPos = basePos + effectiveLp;
    let decStepIdx = 0;
    const singleStepArr = new Uint32Array(4);

    while (
      generatedIds.length > 0 &&
      generatedIds.length < maxNewTokens &&
      !this.eosTokenIds.has(generatedIds[generatedIds.length - 1])
    ) {
      if (signal && signal.aborted) break;

      decStepIdx += 1;
      if (this.isGoldenSliceV4 && this.currentMActive > 0) {
        this._updateHorizonScaleGpu(decStepIdx);
      }

      singleStepArr[0] = generatedIds[generatedIds.length - 1];
      singleStepArr[1] = curPos;
      singleStepArr[2] = this.currentMActive;
      this.device.queue.writeBuffer(this.stepStateBuf, 0, singleStepArr);

      const enc = this.device.createCommandEncoder();
      const cpass = enc.beginComputePass();
      this._recordOps(cpass, this.embedOps);
      if (this.isGoldenSliceV4 && this.scarMidInjectOp) {
        for (let l = 0; l <= this.midLayerIdx; l++) {
          this._recordOps(cpass, this.layerOps[l]);
        }
        this._recordOps(cpass, [this.scarMidInjectOp]);
        for (let l = this.midLayerIdx + 1; l < this.numLayers; l++) {
          this._recordOps(cpass, this.layerOps[l]);
        }
      } else {
        for (let l = 0; l < this.numLayers; l++) {
          this._recordOps(cpass, this.layerOps[l]);
        }
      }
      this._recordOps(cpass, [this.finalNormOp, this.scarInjectOp]);
      this._recordOps(cpass, this.lmHeadOps);
      this._recordOps(cpass, this.reduceOps);
      cpass.end();
      enc.copyBufferToBuffer(this.top2OutBuf, 0, this.readbackTop2Buf, 0, 16);
      this.device.queue.submit([enc.finish()]);

      const stepOut = await this._readTop2AndTelemetry();
      curPos += 1;
      lastGenTokenFed = true;
      if (signal?.aborted) break;
      if (this.eosTokenIds.has(stepOut.top1Id)) break;
      if (
        shouldStopAtSentenceBoundary(epistemicMode, generatedIds.length, decStepIdx, this.seedHorizon, maxNewTokens) &&
        endsWithCompleteSentence(lastDecoded)
      ) {
        const nextPiece = this.tokenizer.decode([stepOut.top1Id], true);
        if (startsNewSentenceOrParagraph(nextPiece)) {
          break;
        }
      }

      generatedIds.push(stepOut.top1Id);
      lastGenTokenFed = false;
      confs.push(stepOut.p1);
      margins.push(stepOut.margin);

      const previewText = this.tokenizer.decode(generatedIds, true);
      const repStart = findRepeatedSixGramStart(generatedIds);
      const hyphenDeg = repStart < 0 && hasCompoundHyphenDegeneration(previewText);
      if (repStart >= 0 || hyphenDeg) {
        if (repStart >= 0) {
          generatedIds.length = repStart;
        } else {
          generatedIds.length = Math.max(1, generatedIds.length - 10);
        }
        lastGenTokenFed = true;
        const trimmedText = this.tokenizer.decode(generatedIds, true);
        const lastBoundary = Math.max(
          trimmedText.lastIndexOf('.'),
          trimmedText.lastIndexOf('!'),
          trimmedText.lastIndexOf('?')
        );
        const cleanOut = lastBoundary >= 24 ? trimmedText.slice(0, lastBoundary + 1) : trimmedText.replace(/[,;:\-\s]+$/, '.');
        lastDecoded = cleanOut;
        const decElapsedS = Math.max((performance.now() - tDec0) / 1000.0, 1e-4);
        const liveTokS = Math.max(0, generatedIds.length - 1) / decElapsedS;
        const curHw =
          this.isGoldenSliceV4 && this.currentMActive > 0
            ? computeHorizonScale(decStepIdx, this.seedHorizon, this.seedFadeSteps)
            : 0.0;
        if (onToken) {
          onToken({
            text: cleanOut,
            deltaToken: '',
            generatedTokens: generatedIds.length,
            tokPerSec: liveTokS,
            scacheBytes: this.scacheGpuBytes,
            scacheDeltaBytes: this.scacheGpuBytes - scacheBytesInitial,
            confidence: stepOut.p1,
            margin: stepOut.margin,
            uncertainty: Math.max(0.0, 1.0 - stepOut.p1),
            horizonScale: curHw,
            top1TokenId: stepOut.top1Id,
          });
        }
        break;
      }

      const hasTrailingSentenceNewline =
        shouldStopAtSentenceBoundary(epistemicMode, generatedIds.length, decStepIdx, this.seedHorizon, maxNewTokens) &&
        /\n\s*$/.test(previewText) &&
        endsWithCompleteSentence(previewText);
      const curText = hasTrailingSentenceNewline ? previewText.trimEnd() : previewText;
      const delta = curText.slice(lastDecoded.length);
      lastDecoded = curText;

      const decElapsedS = Math.max((performance.now() - tDec0) / 1000.0, 1e-4);
      const liveTokS = (generatedIds.length - 1) / decElapsedS;
      const curHw =
        this.isGoldenSliceV4 && this.currentMActive > 0
          ? computeHorizonScale(decStepIdx, this.seedHorizon, this.seedFadeSteps)
          : 0.0;
      if (onToken) {
        onToken({
          text: curText,
          deltaToken: delta,
          generatedTokens: generatedIds.length,
          tokPerSec: liveTokS,
          scacheBytes: this.scacheGpuBytes,
          scacheDeltaBytes: this.scacheGpuBytes - scacheBytesInitial,
          confidence: stepOut.p1,
          margin: stepOut.margin,
          uncertainty: Math.max(0.0, 1.0 - stepOut.p1),
          horizonScale: curHw,
          top1TokenId: stepOut.top1Id,
        });
      }

      if (hasTrailingSentenceNewline) {
        break;
      }
    }

    this._setActiveCapsulesGpu([]);
    if (signal?.aborted) {
      if (guardedCleanRollbackPos !== null) {
        await restorePreTurnSCache();
      } else {
        const closingAbortIds: number[] = [];
        if (generatedIds.length > 0 && !lastGenTokenFed) {
          closingAbortIds.push(generatedIds[generatedIds.length - 1]);
        }
        for (let i = 0; i < this.turnSuffixIds.length; i++) {
          closingAbortIds.push(this.turnSuffixIds[i]);
        }
        this.sessionPos = await this._flushClosingTokensToSCache(closingAbortIds, curPos);
        this.turnCount += 1;
      }
      throw createAbortError();
    }

    if (guardedCleanRollbackPos !== null && guardedCleanPrefixIds !== null) {
      const encCleanRest = this.device.createCommandEncoder();
      encCleanRest.copyBufferToBuffer(this.scacheKvSnapBuf, 0, this.scacheKvBuf, 0, this.scacheKvBuf.size);
      encCleanRest.copyBufferToBuffer(this.scacheWScarSnapBuf, 0, this.scacheWScarBuf, 0, this.scacheWScarBuf.size);
      encCleanRest.copyBufferToBuffer(this.scacheSalSnapBuf, 0, this.scacheSalienceBuf, 0, this.scacheSalienceBuf.size);
      this.device.queue.submit([encCleanRest.finish()]);
      await this._waitQueueSafe(null);

      const cleanFlushIds = [...guardedCleanPrefixIds, ...generatedIds, ...this.turnSuffixIds];
      await this._submitPrefillSequence(
        cleanFlushIds,
        guardedCleanRollbackPos,
        0,
        false,
        null,
        undefined,
        undefined,
        true
      );
      this.sessionPos = guardedCleanRollbackPos + cleanFlushIds.length;
    } else {
      const closingIds: number[] = [];
      if (generatedIds.length > 0 && !lastGenTokenFed) {
        closingIds.push(generatedIds[generatedIds.length - 1]);
      }
      for (let i = 0; i < this.turnSuffixIds.length; i++) {
        closingIds.push(this.turnSuffixIds[i]);
      }
      this.sessionPos = await this._flushClosingTokensToSCache(closingIds, curPos);
    }
    this.turnCount += 1;

    const decodeMs = performance.now() - tDec0;
    const scacheBytesFinal = this.scacheGpuBytes;
    const decSteps = Math.max(1, generatedIds.length - 1);
    const decodeTokPerSec = generatedIds.length > 1 && decodeMs > 0 ? decSteps / (decodeMs / 1000.0) : 0.0;
    const meanConf = confs.length > 0 ? confs.reduce((a, b) => a + b, 0) / confs.length : 0.0;
    const meanMargin = margins.length > 0 ? margins.reduce((a, b) => a + b, 0) / margins.length : 0.0;

    return {
      answer: (lastDecoded || this.tokenizer.decode(generatedIds, true)).trim(),
      generatedTokenIds: generatedIds,
      prefillTokens: effectiveLp,
      sessionTokens: this.sessionPos,
      turnIndex: this.turnCount,
      generatedTokens: generatedIds.length,
      prefillMs: Math.round(prefillMs * 100) / 100,
      decodeMs: Math.round(decodeMs * 100) / 100,
      decodeTokPerSec: Math.round(decodeTokPerSec * 100) / 100,
      scacheBytesInitial,
      scacheBytesAfterPrefill,
      scacheBytesFinal,
      scacheDeltaBytes: scacheBytesFinal - scacheBytesInitial,
      vramWeightsMiB: Math.round((this.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100,
      activeCapsules: routeResult.active.map((a) => ({
        capsuleId: a.capsuleId,
        gate: Math.round(a.gate * 10000) / 10000,
        fromCache: a.fromCache,
        title: a.meta?.metadata?.title,
        titlePt: a.meta?.metadata?.title_pt,
        category: a.meta?.metadata?.category,
      })),
      routingDiagnostics: routeResult.diagnostics,
      turnHttp206Bytes: routeResult.turnBytesFetched,
      totalHttp206Bytes: this.scarStore ? this.scarStore.totalHttp206Bytes : 0,
      httpEvents: routeResult.httpEvents,
      meanConfidenceP1: Math.round(meanConf * 10000) / 10000,
      meanMarginP1P2: Math.round(meanMargin * 10000) / 10000,
      epistemicMode,
      pointScarMs: pointScarSample?.latencyMs,
      pointScarSteps: pointScarSample?.numSteps,
      pointScarShardBytes: pointScarSample?.shardBytes,
    };
  }
}
