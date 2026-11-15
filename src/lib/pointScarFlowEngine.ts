/**
 * Pure In-Browser WebGPU Executor for `PointScarFlow` v1.0 (`ScarNative` Continuous 3D Point-Cloud Generator)
 *
 * STRICT ARCHITECTURAL INTEGRITY (`AGENTS.md` + `system-one-architecture`):
 * - 100% Tabula-Rasa `ScarNativeLinear` (`D4-LUT256 VQ4` + `FWHT-64` + `d_eq` + `INT4` structural scars)
 * - ZERO `torch.nn.Linear` dense weights, ZERO `Math.random()`, ZERO hardcoded shape templates/catalog
 * - Integrates Brenier Optimal-Transport Rectified Flow `dX/dtau = v_theta(X_tau, tau | z_23)` from the
 *   deterministic canonical Fibonacci sphere `X_0 in S^2` directly on Qwen2.5-3B's shared `GPUDevice`
 * - Captures both the intermediate OT flow state `X_0.5` and final converged 3D manifold `X_1.0` (`[1024, 4]`:
 *   `x, y, z, salience`) in a single `GPUCommandBuffer` submission (`0` extra LLM tokens, `Δ +0 B` `SCache O(1)`).
 */

import pointScarWgslSource from './shaders/point_scar_flow.wgsl?raw';
import { fetchWithPersistentCache } from './qwenBpeTokenizer';

const GPU_BUF = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const GPU_MAP_READ = 0x0001;

export interface PointScarFlowConfig {
  cond_in_dim: number;
  cond_dim: number;
  d_model: number;
  d_ff: number;
  num_layers: number;
  num_heads: number;
  head_dim: number;
  num_parts: number;
  num_points: number;
  scar_rank: number;
  cond_scar_rank: number;
  head_scar_rank: number;
  rms_norm_eps: number;
}

export interface ScarLinearManifestSpec {
  in_dim: number;
  out_dim: number;
  rank: number;
  d_eq_u32_off: number;
  lut256_u32_off: number;
  vq_codes_u32_off: number;
  vq_scales_u32_off: number;
  ascar_packed_u32_off: number;
  ascar_scales_u32_off: number;
  bscar_u32_off: number;
}

export interface PointScarBlockManifestSpec {
  ada_mod: ScarLinearManifestSpec;
  part_ln1_u32_off: number;
  pt_kv_ln_u32_off: number;
  part_q: ScarLinearManifestSpec;
  part_k_from_pt: ScarLinearManifestSpec;
  part_v_from_pt: ScarLinearManifestSpec;
  part_o: ScarLinearManifestSpec;
  part_ln2_u32_off: number;
  part_gate: ScarLinearManifestSpec;
  part_up: ScarLinearManifestSpec;
  part_down: ScarLinearManifestSpec;
  pt_ln1_u32_off: number;
  part_kv_ln_u32_off: number;
  pt_q: ScarLinearManifestSpec;
  pt_k_from_part: ScarLinearManifestSpec;
  pt_v_from_part: ScarLinearManifestSpec;
  pt_o: ScarLinearManifestSpec;
  pt_ln2_u32_off: number;
  pt_gate: ScarLinearManifestSpec;
  pt_up: ScarLinearManifestSpec;
  pt_down: ScarLinearManifestSpec;
}

export interface PointScarManifest {
  format: string;
  version: string;
  architecture: string;
  config: PointScarFlowConfig;
  shard_file: string;
  shard_bytes: number;
  wgsl_shader: string;
  x0_f32_u32_off: number;
  cond: {
    cond_z: ScarLinearManifestSpec;
    cond_t: ScarLinearManifestSpec;
    cond_norm_u32_off: number;
    cond_mlp: ScarLinearManifestSpec;
    part_seed: ScarLinearManifestSpec;
  };
  point_stem: ScarLinearManifestSpec;
  blocks: PointScarBlockManifestSpec[];
  final_norm_u32_off: number;
  vel_head: ScarLinearManifestSpec;
  training_metadata?: {
    recommended_inference?: { num_steps?: number; guidance_scale?: number };
    [key: string]: any;
  };
}

export interface PointScarSampleResult {
  midCloudF32: Float32Array; // [num_points * 4] at tau = 0.5
  finalCloudF32: Float32Array; // [num_points * 4] at tau = 1.0 (`x, y, z, salience`)
  numPoints: number;
  numSteps: number;
  guidanceScale: number;
  latencyMs: number;
  shardBytes: number;
}

type CompiledDispatchOp = [any, any, number, number, number];

interface CompiledFlowSchedule {
  firstHalfOps: CompiledDispatchOp[];
  secondHalfOps: CompiledDispatchOp[];
}

function packPsfParams(
  ints: Partial<Record<'p0' | 'p1' | 'p2' | 'p3' | 'p4' | 'p5' | 'p6' | 'p7', number>> = {},
  floats: Partial<Record<'f0' | 'f1' | 'f2' | 'f3', number>> = {}
): ArrayBuffer {
  const buf = new ArrayBuffer(48);
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

const PSF_ENTRY_POINTS = [
  'psf_build_point_features_f32',
  'psf_time_embed_64_f32',
  'psf_vq4_batched_equilibrate_fwht64',
  'psf_batched_int4_ascar_gemm',
  'psf_batched_vq4_int4scar_gemm',
  'psf_rmsnorm_adaln_f32',
  'psf_part_from_point_attn_hd32',
  'psf_point_from_part_attn_hd32',
  'psf_add_silu_f32',
  'psf_swiglu_mul_f32',
  'psf_gated_residual_add_f32',
  'psf_euler_step_cfg_f32',
] as const;

export class BrowserPointScarFlowEngine {
  public readonly device: any;
  public readonly manifest: PointScarManifest;
  public readonly cfg: PointScarFlowConfig;
  public readonly shardBytes: number;
  public readonly recommendedSteps: number;
  public readonly recommendedGuidance: number;

  private readonly shaderMod: any;
  private readonly pipelines: Record<string, any> = {};
  private readonly ownedBuffers: any[] = [];
  private readonly compiledSchedules = new Map<string, CompiledFlowSchedule>();

  private readonly shardBuf: any;
  private readonly x0Buf: any;
  private readonly initXBytes: Float32Array;
  private readonly dummyU32Buf: any;
  private readonly dummyF32Buf: any;

  private readonly zCondBuf: any;
  private readonly zNullBuf: any;
  private readonly tEmbBuf: any;
  private readonly condABuf: any;
  private readonly condBBuf: any;
  private readonly cGlobalBuf: any;
  private readonly cSiluBuf: any;
  private readonly modBuf: any;

  private readonly xCurBuf: any;
  private readonly feat64Buf: any;
  private readonly hPtsBuf: any;
  private readonly partsBuf: any;
  private readonly vq4RotBuf: any;
  private readonly ascarUBuf: any;

  private readonly partNBuf: any;
  private readonly ptNBuf: any;
  private readonly partQBuf: any;
  private readonly ptKBuf: any;
  private readonly ptVBuf: any;
  private readonly partAttnBuf: any;
  private readonly partProjBuf: any;
  private readonly partGateBuf: any;
  private readonly partUpBuf: any;
  private readonly partActBuf: any;

  private readonly ptQBuf: any;
  private readonly partKBuf: any;
  private readonly partVBuf: any;
  private readonly ptAttnBuf: any;
  private readonly ptProjBuf: any;
  private readonly ptGateBuf: any;
  private readonly ptUpBuf: any;
  private readonly ptActBuf: any;

  private readonly vCondBuf: any;
  private readonly vUncondBuf: any;
  private readonly readBuf: any;

  private activeLock: Promise<void> = Promise.resolve();
  private destroyed = false;

  private constructor(device: any, manifest: PointScarManifest, shardArrayBuf: ArrayBuffer) {
    this.device = device;
    this.manifest = manifest;
    this.cfg = manifest.config;
    this.shardBytes = shardArrayBuf.byteLength;
    this.recommendedSteps = manifest.training_metadata?.recommended_inference?.num_steps ?? 4;
    this.recommendedGuidance = manifest.training_metadata?.recommended_inference?.guidance_scale ?? 1.0;

    this.shaderMod = this.device.createShaderModule({
      label: 'point_scar_flow_wgsl_v1',
      code: pointScarWgslSource,
    });

    for (const ep of PSF_ENTRY_POINTS) {
      this.pipelines[ep] = this.device.createComputePipeline({
        label: `pipe_${ep}`,
        layout: 'auto',
        compute: { module: this.shaderMod, entryPoint: ep },
      });
    }

    const c = this.cfg;
    const n = c.num_points;
    const m = c.num_parts;
    const d = c.d_model;
    const ff = c.d_ff;

    this.shardBuf = this._allocWithData('psf_shard', shardArrayBuf, GPU_BUF.STORAGE);

    const x0ByteOff = manifest.x0_f32_u32_off * 4;
    const x0Slice = shardArrayBuf.slice(x0ByteOff, x0ByteOff + n * 3 * 4);
    this.x0Buf = this._allocWithData('psf_x0', x0Slice, GPU_BUF.STORAGE);

    const x0F32 = new Float32Array(x0Slice);
    this.initXBytes = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      this.initXBytes[i * 4 + 0] = x0F32[i * 3 + 0];
      this.initXBytes[i * 4 + 1] = x0F32[i * 3 + 1];
      this.initXBytes[i * 4 + 2] = x0F32[i * 3 + 2];
      this.initXBytes[i * 4 + 3] = 0.5;
    }

    this.dummyU32Buf = this._alloc('psf_dummy_u32', 16, GPU_BUF.STORAGE);
    this.dummyF32Buf = this._alloc('psf_dummy_f32', 16, GPU_BUF.STORAGE);

    this.zCondBuf = this._alloc('psf_z_cond', c.cond_in_dim * 4, GPU_BUF.STORAGE);
    this.zNullBuf = this._allocWithData(
      'psf_z_null',
      new Float32Array(c.cond_in_dim).buffer,
      GPU_BUF.STORAGE
    );
    this.tEmbBuf = this._alloc('psf_t_emb', 64 * 4, GPU_BUF.STORAGE);
    this.condABuf = this._alloc('psf_cond_a', c.cond_dim * 4, GPU_BUF.STORAGE);
    this.condBBuf = this._alloc('psf_cond_b', c.cond_dim * 4, GPU_BUF.STORAGE);
    this.cGlobalBuf = this._alloc('psf_c_global', c.cond_dim * 4, GPU_BUF.STORAGE);
    this.cSiluBuf = this._alloc('psf_c_silu', c.cond_dim * 4, GPU_BUF.STORAGE);
    this.modBuf = this._alloc('psf_mod', 6 * d * 4, GPU_BUF.STORAGE);

    this.xCurBuf = this._alloc('psf_x_cur', n * 4 * 4, GPU_BUF.STORAGE | GPU_BUF.COPY_SRC);
    this.feat64Buf = this._alloc('psf_feat64', n * 64 * 4, GPU_BUF.STORAGE);
    this.hPtsBuf = this._alloc('psf_h_pts', n * d * 4, GPU_BUF.STORAGE);
    this.partsBuf = this._alloc('psf_parts', m * d * 4, GPU_BUF.STORAGE);

    const maxBatchIn = Math.max(c.cond_in_dim, n * Math.max(d, ff, 64));
    const maxRank = Math.max(c.scar_rank, c.cond_scar_rank, c.head_scar_rank, 8);
    this.vq4RotBuf = this._alloc('psf_vq4_rot', maxBatchIn * 4, GPU_BUF.STORAGE);
    this.ascarUBuf = this._alloc('psf_ascar_u', n * maxRank * 4, GPU_BUF.STORAGE);

    this.partNBuf = this._alloc('psf_part_n', m * d * 4, GPU_BUF.STORAGE);
    this.ptNBuf = this._alloc('psf_pt_n', n * d * 4, GPU_BUF.STORAGE);
    this.partQBuf = this._alloc('psf_part_q', m * d * 4, GPU_BUF.STORAGE);
    this.ptKBuf = this._alloc('psf_pt_k', n * d * 4, GPU_BUF.STORAGE);
    this.ptVBuf = this._alloc('psf_pt_v', n * d * 4, GPU_BUF.STORAGE);
    this.partAttnBuf = this._alloc('psf_part_attn', m * d * 4, GPU_BUF.STORAGE);
    this.partProjBuf = this._alloc('psf_part_proj', m * d * 4, GPU_BUF.STORAGE);
    this.partGateBuf = this._alloc('psf_part_gate', m * ff * 4, GPU_BUF.STORAGE);
    this.partUpBuf = this._alloc('psf_part_up', m * ff * 4, GPU_BUF.STORAGE);
    this.partActBuf = this._alloc('psf_part_act', m * ff * 4, GPU_BUF.STORAGE);

    this.ptQBuf = this._alloc('psf_pt_q', n * d * 4, GPU_BUF.STORAGE);
    this.partKBuf = this._alloc('psf_part_k', m * d * 4, GPU_BUF.STORAGE);
    this.partVBuf = this._alloc('psf_part_v', m * d * 4, GPU_BUF.STORAGE);
    this.ptAttnBuf = this._alloc('psf_pt_attn', n * d * 4, GPU_BUF.STORAGE);
    this.ptProjBuf = this._alloc('psf_pt_proj', n * d * 4, GPU_BUF.STORAGE);
    this.ptGateBuf = this._alloc('psf_pt_gate', n * ff * 4, GPU_BUF.STORAGE);
    this.ptUpBuf = this._alloc('psf_pt_up', n * ff * 4, GPU_BUF.STORAGE);
    this.ptActBuf = this._alloc('psf_pt_act', n * ff * 4, GPU_BUF.STORAGE);

    this.vCondBuf = this._alloc('psf_v_cond', n * 4 * 4, GPU_BUF.STORAGE);
    this.vUncondBuf = this._alloc('psf_v_uncond', n * 4 * 4, GPU_BUF.STORAGE);

    // 2 x [num_points * 4] floats (`X_0.5` mid-flow + `X_1.0` converged manifold)
    this.readBuf = this.device.createBuffer({
      label: 'psf_read_2x',
      size: n * 4 * 4 * 2,
      usage: GPU_BUF.MAP_READ | GPU_BUF.COPY_DST,
    });
    this.ownedBuffers.push(this.readBuf);

    // Pre-compile default inference schedule immediately
    this._getSchedule(this.recommendedSteps, this.recommendedGuidance);
  }

  public static async createOnDevice(
    device: any,
    pointScarBaseUrl = '/webgpu/pointscar_v1'
  ): Promise<BrowserPointScarFlowEngine | null> {
    try {
      const cleanBase = pointScarBaseUrl.replace(/\/$/, '');
      const manifestResp = await fetch(`${cleanBase}/pointscar_manifest.json`, { cache: 'no-cache' });
      if (!manifestResp.ok) {
        return null;
      }
      const manifest: PointScarManifest = await manifestResp.json();
      const { response: shardResp } = await fetchWithPersistentCache(`${cleanBase}/${manifest.shard_file}`);
      if (!shardResp.ok) {
        return null;
      }
      const shardBuf = await shardResp.arrayBuffer();
      return new BrowserPointScarFlowEngine(device, manifest, shardBuf);
    } catch (err) {
      console.warn('[PointScarFlow] Optional 3D generator load skipped:', err);
      return null;
    }
  }

  private _alloc(label: string, byteSize: number, usageFlags: number): any {
    const aligned = Math.max(16, Math.ceil(byteSize / 4) * 4);
    const buf = this.device.createBuffer({
      label,
      size: aligned,
      usage: usageFlags | GPU_BUF.COPY_DST,
    });
    this.ownedBuffers.push(buf);
    return buf;
  }

  private _allocWithData(label: string, raw: ArrayBuffer, usageFlags: number): any {
    const aligned = Math.max(16, Math.ceil(raw.byteLength / 4) * 4);
    const buf = this.device.createBuffer({
      label,
      size: aligned,
      usage: usageFlags | GPU_BUF.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, raw, 0, raw.byteLength);
    this.ownedBuffers.push(buf);
    return buf;
  }

  private _makeParamsBuf(
    ints: Partial<Record<'p0' | 'p1' | 'p2' | 'p3' | 'p4' | 'p5' | 'p6' | 'p7', number>> = {},
    floats: Partial<Record<'f0' | 'f1' | 'f2' | 'f3', number>> = {}
  ): any {
    return this._allocWithData('psf_u', packPsfParams(ints, floats), GPU_BUF.UNIFORM);
  }

  private _compileOp(
    ops: CompiledDispatchOp[],
    ep: string,
    paramsBuf: any,
    bindings: Record<number, any>,
    gx: number,
    gy = 1,
    gz = 1
  ): void {
    const pipe = this.pipelines[ep];
    const bgl = pipe.getBindGroupLayout(0);
    const entries: Array<{ binding: number; resource: { buffer: any; offset: number; size: number } }> = [
      { binding: 0, resource: { buffer: paramsBuf, offset: 0, size: 48 } },
    ];
    for (const [k, buf] of Object.entries(bindings)) {
      entries.push({
        binding: Number(k),
        resource: { buffer: buf, offset: 0, size: buf.size },
      });
    }
    const bg = this.device.createBindGroup({ layout: bgl, entries });
    ops.push([pipe, bg, gx, gy, gz]);
  }

  private _compileScarLinear(
    ops: CompiledDispatchOp[],
    spec: ScarLinearManifestSpec,
    xInBuf: any,
    yOutBuf: any,
    batchSize: number
  ): void {
    const outDim = spec.out_dim;
    const inDim = spec.in_dim;
    const rank = spec.rank;

    this._compileOp(
      ops,
      'psf_vq4_batched_equilibrate_fwht64',
      this._makeParamsBuf({ p0: inDim, p1: spec.d_eq_u32_off, p2: batchSize }),
      { 1: this.shardBuf, 2: xInBuf, 5: this.vq4RotBuf },
      inDim >>> 6,
      batchSize
    );

    if (rank > 0) {
      this._compileOp(
        ops,
        'psf_batched_int4_ascar_gemm',
        this._makeParamsBuf({
          p0: rank,
          p1: inDim,
          p2: batchSize,
          p3: spec.ascar_packed_u32_off,
          p4: spec.ascar_scales_u32_off,
        }),
        { 1: this.shardBuf, 2: xInBuf, 5: this.ascarUBuf },
        (batchSize * rank + 63) >>> 6,
        1
      );
    }

    this._compileOp(
      ops,
      'psf_batched_vq4_int4scar_gemm',
      this._makeParamsBuf({
        p0: outDim,
        p1: inDim,
        p2: rank,
        p3: spec.vq_codes_u32_off,
        p4: spec.vq_scales_u32_off,
        p5: spec.lut256_u32_off,
        p6: spec.bscar_u32_off ?? 0,
        p7: batchSize,
      }),
      {
        1: this.shardBuf,
        2: this.vq4RotBuf,
        3: rank > 0 ? this.ascarUBuf : this.dummyF32Buf,
        5: yOutBuf,
      },
      (outDim + 63) >>> 6,
      (batchSize + 7) >>> 3
    );
  }

  private _compileVelocityPass(
    ops: CompiledDispatchOp[],
    zBuf: any,
    tauVal: number,
    vOutBuf: any
  ): void {
    const c = this.cfg;
    const n = c.num_points;
    const m = c.num_parts;
    const d = c.d_model;
    const ff = c.d_ff;
    const eps = c.rms_norm_eps;
    const cond = this.manifest.cond;

    this._compileOp(ops, 'psf_time_embed_64_f32', this._makeParamsBuf({}, { f0: tauVal }), { 5: this.tEmbBuf }, 1);
    this._compileScarLinear(ops, cond.cond_z, zBuf, this.condABuf, 1);
    this._compileScarLinear(ops, cond.cond_t, this.tEmbBuf, this.condBBuf, 1);

    this._compileOp(
      ops,
      'psf_add_silu_f32',
      this._makeParamsBuf({ p0: c.cond_dim, p1: 1 }),
      { 2: this.condABuf, 3: this.condBBuf, 5: this.cSiluBuf },
      (c.cond_dim + 255) >>> 8
    );
    this._compileOp(
      ops,
      'psf_rmsnorm_adaln_f32',
      this._makeParamsBuf({ p0: c.cond_dim, p1: cond.cond_norm_u32_off, p2: 1, p3: 0 }, { f0: eps }),
      { 1: this.shardBuf, 2: this.cSiluBuf, 3: this.dummyF32Buf, 5: this.condABuf },
      1
    );
    this._compileScarLinear(ops, cond.cond_mlp, this.condABuf, this.cGlobalBuf, 1);
    this._compileOp(
      ops,
      'psf_add_silu_f32',
      this._makeParamsBuf({ p0: c.cond_dim, p1: 0 }),
      { 2: this.cGlobalBuf, 3: this.dummyF32Buf, 5: this.cSiluBuf },
      (c.cond_dim + 255) >>> 8
    );
    this._compileScarLinear(ops, cond.part_seed, this.cSiluBuf, this.partsBuf, 1);

    this._compileOp(
      ops,
      'psf_build_point_features_f32',
      this._makeParamsBuf({ p0: n }, { f0: tauVal }),
      { 2: this.xCurBuf, 3: this.x0Buf, 5: this.feat64Buf },
      (n + 63) >>> 6
    );
    this._compileScarLinear(ops, this.manifest.point_stem, this.feat64Buf, this.hPtsBuf, n);

    const invSqrtHd = 1.0 / Math.sqrt(c.head_dim);

    for (const blk of this.manifest.blocks) {
      this._compileScarLinear(ops, blk.ada_mod, this.cSiluBuf, this.modBuf, 1);

      // Stage A: `Part <- Point`
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.part_ln1_u32_off, p2: m, p3: 0 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.partsBuf, 3: this.dummyF32Buf, 5: this.partNBuf },
        m
      );
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.pt_kv_ln_u32_off, p2: n, p3: 0 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.hPtsBuf, 3: this.dummyF32Buf, 5: this.ptNBuf },
        n
      );
      this._compileScarLinear(ops, blk.part_q, this.partNBuf, this.partQBuf, m);
      this._compileScarLinear(ops, blk.part_k_from_pt, this.ptNBuf, this.ptKBuf, n);
      this._compileScarLinear(ops, blk.part_v_from_pt, this.ptNBuf, this.ptVBuf, n);

      this._compileOp(
        ops,
        'psf_part_from_point_attn_hd32',
        this._makeParamsBuf({ p0: m, p1: n, p2: c.num_heads, p3: c.head_dim }, { f0: invSqrtHd }),
        { 2: this.partQBuf, 3: this.ptKBuf, 4: this.ptVBuf, 5: this.partAttnBuf },
        m,
        c.num_heads
      );
      this._compileScarLinear(ops, blk.part_o, this.partAttnBuf, this.partProjBuf, m);
      this._compileOp(
        ops,
        'psf_gated_residual_add_f32',
        this._makeParamsBuf({ p0: m * d, p1: d, p2: 0, p3: 0 }),
        { 2: this.partProjBuf, 3: this.dummyF32Buf, 5: this.partsBuf },
        (m * d + 255) >>> 8
      );

      // Part SwiGLU
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.part_ln2_u32_off, p2: m, p3: 0 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.partsBuf, 3: this.dummyF32Buf, 5: this.partNBuf },
        m
      );
      this._compileScarLinear(ops, blk.part_gate, this.partNBuf, this.partGateBuf, m);
      this._compileScarLinear(ops, blk.part_up, this.partNBuf, this.partUpBuf, m);
      this._compileOp(
        ops,
        'psf_swiglu_mul_f32',
        this._makeParamsBuf({ p0: m * ff }),
        { 2: this.partGateBuf, 3: this.partUpBuf, 5: this.partActBuf },
        (m * ff + 255) >>> 8
      );
      this._compileScarLinear(ops, blk.part_down, this.partActBuf, this.partProjBuf, m);
      this._compileOp(
        ops,
        'psf_gated_residual_add_f32',
        this._makeParamsBuf({ p0: m * d, p1: d, p2: 0, p3: 0 }),
        { 2: this.partProjBuf, 3: this.dummyF32Buf, 5: this.partsBuf },
        (m * d + 255) >>> 8
      );

      // Stage B: `Point <- Part`
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.pt_ln1_u32_off, p2: n, p3: 1, p4: 0 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.hPtsBuf, 3: this.modBuf, 5: this.ptNBuf },
        n
      );
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.part_kv_ln_u32_off, p2: m, p3: 0 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.partsBuf, 3: this.dummyF32Buf, 5: this.partNBuf },
        m
      );
      this._compileScarLinear(ops, blk.pt_q, this.ptNBuf, this.ptQBuf, n);
      this._compileScarLinear(ops, blk.pt_k_from_part, this.partNBuf, this.partKBuf, m);
      this._compileScarLinear(ops, blk.pt_v_from_part, this.partNBuf, this.partVBuf, m);

      this._compileOp(
        ops,
        'psf_point_from_part_attn_hd32',
        this._makeParamsBuf({ p0: n, p1: m, p2: c.num_heads, p3: c.head_dim }, { f0: invSqrtHd }),
        { 2: this.ptQBuf, 3: this.partKBuf, 4: this.partVBuf, 5: this.ptAttnBuf },
        (n + 63) >>> 6,
        c.num_heads
      );
      this._compileScarLinear(ops, blk.pt_o, this.ptAttnBuf, this.ptProjBuf, n);
      this._compileOp(
        ops,
        'psf_gated_residual_add_f32',
        this._makeParamsBuf({ p0: n * d, p1: d, p2: 1, p3: 2 }),
        { 2: this.ptProjBuf, 3: this.modBuf, 5: this.hPtsBuf },
        (n * d + 255) >>> 8
      );

      // Stage C: Point SwiGLU
      this._compileOp(
        ops,
        'psf_rmsnorm_adaln_f32',
        this._makeParamsBuf({ p0: d, p1: blk.pt_ln2_u32_off, p2: n, p3: 1, p4: 3 }, { f0: eps }),
        { 1: this.shardBuf, 2: this.hPtsBuf, 3: this.modBuf, 5: this.ptNBuf },
        n
      );
      this._compileScarLinear(ops, blk.pt_gate, this.ptNBuf, this.ptGateBuf, n);
      this._compileScarLinear(ops, blk.pt_up, this.ptNBuf, this.ptUpBuf, n);
      this._compileOp(
        ops,
        'psf_swiglu_mul_f32',
        this._makeParamsBuf({ p0: n * ff }),
        { 2: this.ptGateBuf, 3: this.ptUpBuf, 5: this.ptActBuf },
        (n * ff + 255) >>> 8
      );
      this._compileScarLinear(ops, blk.pt_down, this.ptActBuf, this.ptProjBuf, n);
      this._compileOp(
        ops,
        'psf_gated_residual_add_f32',
        this._makeParamsBuf({ p0: n * d, p1: d, p2: 1, p3: 5 }),
        { 2: this.ptProjBuf, 3: this.modBuf, 5: this.hPtsBuf },
        (n * d + 255) >>> 8
      );
    }

    this._compileOp(
      ops,
      'psf_rmsnorm_adaln_f32',
      this._makeParamsBuf({ p0: d, p1: this.manifest.final_norm_u32_off, p2: n, p3: 0 }, { f0: eps }),
      { 1: this.shardBuf, 2: this.hPtsBuf, 3: this.dummyF32Buf, 5: this.ptNBuf },
      n
    );
    this._compileScarLinear(ops, this.manifest.vel_head, this.ptNBuf, vOutBuf, n);
  }

  private _getSchedule(numSteps: number, guidanceScale: number): CompiledFlowSchedule {
    const safeSteps = Math.max(1, Math.round(numSteps));
    const key = `${safeSteps}_${Math.round(guidanceScale * 10000)}`;
    const existing = this.compiledSchedules.get(key);
    if (existing) return existing;

    const firstHalfOps: CompiledDispatchOp[] = [];
    const secondHalfOps: CompiledDispatchOp[] = [];
    const n4 = this.cfg.num_points * 4;
    const dt = 1.0 / safeSteps;
    const useCfg = Math.abs(guidanceScale - 1.0) > 1e-5;
    const midStep = Math.max(1, Math.floor(safeSteps / 2));

    for (let step = 0; step < safeSteps; step++) {
      const targetOps = step < midStep ? firstHalfOps : secondHalfOps;
      const tauVal = step * dt;
      this._compileVelocityPass(targetOps, this.zCondBuf, tauVal, this.vCondBuf);
      if (useCfg) {
        this._compileVelocityPass(targetOps, this.zNullBuf, tauVal, this.vUncondBuf);
      }
      const isFinal = step === safeSteps - 1 ? 1 : 0;
      this._compileOp(
        targetOps,
        'psf_euler_step_cfg_f32',
        this._makeParamsBuf({ p0: n4, p1: useCfg ? 1 : 0, p2: isFinal }, { f0: dt, f1: guidanceScale }),
        {
          2: this.vCondBuf,
          3: useCfg ? this.vUncondBuf : this.dummyF32Buf,
          5: this.xCurBuf,
        },
        (n4 + 255) >>> 8
      );
    }

    const sched: CompiledFlowSchedule = { firstHalfOps, secondHalfOps };
    this.compiledSchedules.set(key, sched);
    return sched;
  }

  public async sampleEuler(
    zCondF32: Float32Array,
    numSteps = this.recommendedSteps,
    guidanceScale = this.recommendedGuidance
  ): Promise<PointScarSampleResult> {
    if (this.destroyed) {
      throw new Error('PointScarFlow engine has been destroyed.');
    }

    let releaseLock!: () => void;
    const prev = this.activeLock;
    this.activeLock = new Promise<void>((res) => {
      releaseLock = res;
    });
    await prev;

    try {
      const t0 = performance.now();
      const c = this.cfg;
      const n4 = c.num_points * 4;
      const cloudByteSize = n4 * 4;

      const zClean =
        zCondF32.length === c.cond_in_dim ? zCondF32 : zCondF32.subarray(0, c.cond_in_dim);
      this.device.queue.writeBuffer(
        this.zCondBuf,
        0,
        zClean.buffer,
        zClean.byteOffset,
        c.cond_in_dim * 4
      );
      this.device.queue.writeBuffer(
        this.xCurBuf,
        0,
        this.initXBytes.buffer,
        this.initXBytes.byteOffset,
        cloudByteSize
      );

      const sched = this._getSchedule(numSteps, guidanceScale);
      const enc = this.device.createCommandEncoder();

      if (sched.firstHalfOps.length > 0) {
        const cpassA = enc.beginComputePass();
        for (const [pipe, bg, gx, gy, gz] of sched.firstHalfOps) {
          cpassA.setPipeline(pipe);
          cpassA.setBindGroup(0, bg);
          cpassA.dispatchWorkgroups(gx, gy, gz);
        }
        cpassA.end();
      }
      enc.copyBufferToBuffer(this.xCurBuf, 0, this.readBuf, 0, cloudByteSize);

      if (sched.secondHalfOps.length > 0) {
        const cpassB = enc.beginComputePass();
        for (const [pipe, bg, gx, gy, gz] of sched.secondHalfOps) {
          cpassB.setPipeline(pipe);
          cpassB.setBindGroup(0, bg);
          cpassB.dispatchWorkgroups(gx, gy, gz);
        }
        cpassB.end();
      }
      enc.copyBufferToBuffer(this.xCurBuf, 0, this.readBuf, cloudByteSize, cloudByteSize);

      this.device.queue.submit([enc.finish()]);

      await this.readBuf.mapAsync(GPU_MAP_READ, 0, cloudByteSize * 2);
      const mapped = this.readBuf.getMappedRange(0, cloudByteSize * 2);
      const fullCopy = new Float32Array(mapped.slice(0));
      this.readBuf.unmap();

      const midCloudF32 = fullCopy.slice(0, n4);
      const finalCloudF32 = fullCopy.slice(n4, n4 * 2);
      const latencyMs = Math.round((performance.now() - t0) * 100) / 100;

      return {
        midCloudF32,
        finalCloudF32,
        numPoints: c.num_points,
        numSteps: Math.max(1, Math.round(numSteps)),
        guidanceScale,
        latencyMs,
        shardBytes: this.shardBytes,
      };
    } finally {
      releaseLock();
    }
  }

  public destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buf of this.ownedBuffers) {
      try {
        buf.destroy();
      } catch {
        // Ignore already destroyed buffers
      }
    }
    this.ownedBuffers.length = 0;
    this.compiledSchedules.clear();
  }
}
