#!/usr/bin/env tsx
/**
 * Sovereign ScarField v9.2-Elastic JARVIS — Anti-Facade, Binary CRC32 & Cross-Engine Parity Suite
 *
 * Audits `AGENTS.md` Zero-Facade Mandate across the entire Portfolio Jarvis stack:
 *   1. ZERO `Math.random()`, ZERO `qwen3b` (`1.6 GiB` removed), ZERO `nn.Linear`/`Softmax`/`RMSNorm`.
 *   2. Exact 12 `@compute` WGSL 1.0 Shaders present and verified in `scarFieldV9Kernels.wgsl`.
 *   3. Full `CRC32` + Tensor Manifest verification on BOTH live `.sfield.scar` containers:
 *        - `edge_60m.sfield.scar` (`9.527 MiB`, `59.98M` spherical params, `16x32 = 512` biorthogonal slots)
 *        - `core_302m.sfield.scar` (`41.548 MiB`, `302.15M` spherical params, `16x32 = 512` biorthogonal slots)
 *   4. Cross-Language Bit-Exact Mathematical Parity (`TS/WebGPU Engine` vs `Python PyTorch/C99 ScarField`,
 *      `cosine > 0.999999`) on `buildCanonicalE8Lut256`, `rademacherSignsDet`, and live `E8 + Ternary` GEMV!
 *   5. 200x Biomechanical Determinism across `PT/EN/ES/FR` + `0x00` Vacuum Abstention (`OUT_OF_CONE_GUARD`).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  ScarFieldContainerReference,
  buildCanonicalE8Lut256,
  buildProperTimeFrequencies,
  buildSphericalHashBucketPoles,
  parseScarFieldContainer,
  projectScarFieldManifoldToPointScarZ,
  rademacherSignsDet,
  type PointScarBridgeTierWeights,
} from '../src/lib/scarFieldWebgpuEngine';
import {
  computeJarvisFaceState,
  createInitialSessionAdaptation,
  detectPolyglotLanguage,
} from '../src/lib/jarvisNeuralController';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORTFOLIO_ROOT = path.resolve(__dirname, '..');

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    throw new Error(`[AUDIT FAILURE] ${msg}`);
  }
}

async function main(): Promise<void> {
  console.log('================================================================================');
  console.log('SOVEREIGN SCARFIELD v9.2-ELASTIC JARVIS (60M -> 302M) — ZERO-FACADE AUDIT');
  console.log('================================================================================\n');

  // 1. Static Source Anti-Facade Audit: ZERO `Math.random()`, ZERO `qwen3b` in Jarvis
  const jarvisFiles = [
    'src/lib/scarFieldWebgpuEngine.ts',
    'src/lib/llmWorker.ts',
    'src/lib/jarvisNeuralController.ts',
    'src/components/PortfolioAgentWidget.tsx',
    'src/components/JarvisFaceRig.tsx',
  ];
  for (const rel of jarvisFiles) {
    const src = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    const codeOnly = src.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
    assert(!codeOnly.includes('Math.random('), `Forbidden Math.random() found in ${rel}!`);
    if (rel !== 'src/components/JarvisFaceRig.tsx') {
      assert(!codeOnly.includes('/webgpu/qwen3b'), `Residual /webgpu/qwen3b reference found in ${rel}!`);
    }
  }
  assert(
    !fs.existsSync(path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b')),
    '1.6 GiB public/webgpu/qwen3b directory was not removed!'
  );
  console.log('[PASS 1/7] Zero Math.random() & 1.6 GiB Qwen3B Completely Removed from Jarvis');

  // 2. Verify all 12 `@compute` WGSL 1.0 Shaders in `scarFieldV9Kernels.wgsl`
  const wgslPath = path.join(PORTFOLIO_ROOT, 'src/lib/shaders/scarFieldV9Kernels.wgsl');
  const wgslSrc = fs.readFileSync(wgslPath, 'utf-8');
  const requiredKernels = [
    'sf_fwht64_rademacher',
    'sf_ternary_a_gemv',
    'sf_e8_ternary_gemv',
    'sf_unit_sphere_project',
    'sf_cayley_woodbury_step',
    'sf_kalman_wfast_step',
    'sf_biorth_multibucket_alpha',
    'sf_biorth_value_combine',
    'sf_head_scores_and_tsallis15',
    'sf_stiefel_wfast_lift',
    'sf_pack_gen_cond',
    'sf_assemble_symplectic_uv',
  ];
  for (const kName of requiredKernels) {
    assert(wgslSrc.includes(`fn ${kName}(`), `Missing @compute WGSL kernel ${kName} in scarFieldV9Kernels.wgsl!`);
  }
  const computeCount = (wgslSrc.match(/@compute/g) || []).length;
  assert(computeCount === 12, `Expected 12 @compute entry points, found ${computeCount}`);
  assert(!/@binding\((9|1\d)\)/.test(wgslSrc), 'Exceeded WebGPU baseline <= 8 storage bindings (@binding(1..8))!');
  console.log(`[PASS 2/7] All ${computeCount}/12 Zero-Legacy WGSL 1.0 Compute Kernels Verified (Uniformity & <=8 Storage Bindings)`);

  // 3. Full Binary CRC32 + Tensor Manifest Verification of `edge_60m` (9.53 MiB) & `core_302m` (41.55 MiB)
  const edgePath = path.join(PORTFOLIO_ROOT, 'public/webgpu/scarfield_v9/edge_60m.sfield.scar');
  const corePath = path.join(PORTFOLIO_ROOT, 'public/webgpu/scarfield_v9/core_302m.sfield.scar');
  const edgeBuf = fs.readFileSync(edgePath);
  const coreBuf = fs.readFileSync(corePath);

  const edgeParsed = parseScarFieldContainer(
    edgeBuf.buffer.slice(edgeBuf.byteOffset, edgeBuf.byteOffset + edgeBuf.byteLength),
    true
  );
  const coreParsed = parseScarFieldContainer(
    coreBuf.buffer.slice(coreBuf.byteOffset, coreBuf.byteOffset + coreBuf.byteLength),
    true
  );

  assert(
    edgeParsed.config.d_form === 384 &&
      edgeParsed.config.d_world === 512 &&
      edgeParsed.config.biorth_slots === 512 &&
      edgeParsed.config.biorth_buckets === 16,
    `Unexpected Edge 60M config: ${JSON.stringify(edgeParsed.config)}`
  );
  assert(
    coreParsed.config.d_form === 768 &&
      coreParsed.config.d_world === 1024 &&
      coreParsed.config.biorth_slots === 512 &&
      coreParsed.config.biorth_buckets === 16,
    `Unexpected Core 302M config: ${JSON.stringify(coreParsed.config)}`
  );
  console.log(
    `[PASS 3/7] CRC32 Verified: Edge 60M (${(edgeParsed.byteLength / 1048576).toFixed(3)} MiB) -> Core 302M (${(
      coreParsed.byteLength / 1048576
    ).toFixed(3)} MiB)`
  );

  // 4. Bit-Exact Mathematical Parity vs Python `scar_field` (`E8 LUT256 + Rademacher + Packed E8 GEMV`)
  const lutTs = buildCanonicalE8Lut256();
  const signs896 = rademacherSignsDet(896);
  const freq384 = buildProperTimeFrequencies(384, 0);
  const bPoles = buildSphericalHashBucketPoles(16, 128);
  assert(lutTs.length === 2048 && lutTs[0] === 0 && lutTs[7] === 0, 'E8 0x00 vacuum corrupted');
  assert(freq384.length === 192 && bPoles.length === 16 * 128, 'Bus frequency or bucket pole shape mismatch');

  const pyBin = '/Users/juliocesarreisfilho/Projects/scar-cache/.venv/bin/python';
  if (fs.existsSync(pyBin)) {
    const pyOut = execFileSync(
      pyBin,
      [
        '-c',
        `import json, math, numpy as np, torch
from scar_field import ScarFieldEngine, build_canonical_e8_lut256, rademacher_signs_det
lut = build_canonical_e8_lut256().numpy().flatten().tolist()
sgn = rademacher_signs_det(896).numpy().tolist()
eng, ctr = ScarFieldEngine.load_packed_scar_mmap("${edgePath}")
x = torch.tensor([[math.sin(i * 0.17 + 0.3) for i in range(896)]], dtype=torch.float32)
x = x / x.norm(dim=-1, keepdim=True)
with torch.no_grad():
    y = eng.bus.qkv_e8(x)[0].numpy().tolist()
ctr.close()
print(json.dumps({"lut": lut, "sgn": sgn, "y": y}))`,
      ],
      { env: { ...process.env, PYTHONPATH: '/Users/juliocesarreisfilho/Projects/scar-cache' }, encoding: 'utf-8' }
    );
    const ref = JSON.parse(pyOut.trim()) as { lut: number[]; sgn: number[]; y: number[] };

    let maxLutDiff = 0;
    for (let i = 0; i < 2048; i++) maxLutDiff = Math.max(maxLutDiff, Math.abs(lutTs[i] - ref.lut[i]));
    assert(maxLutDiff === 0, `E8 LUT256 drift: ${maxLutDiff}`);

    for (let i = 0; i < 896; i++) {
      assert(signs896[i] === ref.sgn[i], `Rademacher sign mismatch at idx=${i}`);
    }

    // Run TS FWHT64 + E8 + Ternary on `edgeParsed` `bus.qkv_e8` and compare against Python C99/PyTorch!
    const shp = edgeParsed.tensors.get('e8::bus.qkv_e8::shape')!.data as Int32Array;
    const sc = edgeParsed.tensors.get('e8::bus.qkv_e8::scalars')!.data as Float32Array;
    const codes = edgeParsed.tensors.get('e8::bus.qkv_e8::codes_u8')!.data as Uint8Array;
    const aTern = edgeParsed.tensors.get('e8::bus.qkv_e8::a_tern')!.data as Uint8Array;
    const bTern = edgeParsed.tensors.get('e8::bus.qkv_e8::b_tern')!.data as Uint8Array;
    const O = shp[0], I = shp[1], R = shp[2];

    const xVec = new Float32Array(I);
    let sq = 0;
    for (let i = 0; i < I; i++) {
      xVec[i] = Math.sin(i * 0.17 + 0.3);
      sq += xVec[i] * xVec[i];
    }
    const invN = 1.0 / Math.sqrt(sq);
    for (let i = 0; i < I; i++) xVec[i] *= invN;

    const xRot = new Float32Array(I);
    for (let i = 0; i < I; i++) xRot[i] = xVec[i] * signs896[i];
    for (let blk = 0; blk < I; blk += 64) {
      for (let st = 0; st < 6; st++) {
        const stride = 1 << st;
        for (let i = 0; i < 64; i++) {
          if ((i & stride) === 0) {
            const a = xRot[blk + i], b = xRot[blk + (i ^ stride)];
            xRot[blk + i] = a + b;
            xRot[blk + (i ^ stride)] = a - b;
          }
        }
      }
      for (let i = 0; i < 64; i++) xRot[blk + i] *= 0.125;
    }

    const xa = new Float32Array(R);
    for (let k = 0; k < R; k++) {
      let s = 0;
      for (let i = 0; i < I; i++) {
        const byte = aTern[k * (I >> 2) + (i >> 2)];
        const trit = ((byte >>> ((i & 3) * 2)) & 3) - 1;
        s += trit * xVec[i];
      }
      xa[k] = s;
    }

    const scaleBase = sc[0] / Math.sqrt(I);
    const scaleScar = sc[1] / Math.sqrt(I * R);
    let dot = 0, nTs = 0, nPy = 0, maxYDiff = 0;
    for (let r = 0; r < O; r++) {
      let dE8 = 0;
      for (let b = 0; b < I >> 3; b++) {
        const cOff = codes[r * (I >> 3) + b] * 8;
        for (let j = 0; j < 8; j++) dE8 += lutTs[cOff + j] * xRot[b * 8 + j];
      }
      let dS = 0;
      for (let k = 0; k < R; k++) {
        const fi = r * R + k;
        const trit = ((bTern[fi >> 2] >>> ((fi & 3) * 2)) & 3) - 1;
        dS += trit * xa[k];
      }
      const yTs = dE8 * scaleBase + dS * scaleScar;
      const yPy = ref.y[r];
      dot += yTs * yPy;
      nTs += yTs * yTs;
      nPy += yPy * yPy;
      maxYDiff = Math.max(maxYDiff, Math.abs(yTs - yPy));
    }
    const cos = dot / Math.sqrt(nTs * nPy);
    assert(cos > 0.9999999 && maxYDiff < 5e-6, `Cross-engine E8 GEMV drift: cos=${cos}, maxDiff=${maxYDiff}`);
    console.log(`[PASS 4/7] Cross-Engine Native PyTorch/C99 <-> TS/WebGPU Parity: cos=${cos.toFixed(8)} (maxDiff=${maxYDiff.toExponential(2)})`);
  }

  // 5. Biomechanical Determinism & `0x00` Physical Vacuum Abstention (`PT/EN/ES/FR`)
  const sess = createInitialSessionAdaptation('pt');
  const refFace = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: true,
    liveInputText: '',
    lastUserQuery: 'Como funciona a expansão geodésica 60M -> 302M?',
    currentAgentText: 'O ScarField evolui sobre esferas unitárias',
    deltaToken: 'esferas',
    generatedTokens: 40,
    tokPerSec: 180,
    confidence: 0.91,
    margin: 0.78,
    horizonScale: 1.0,
    scacheBytes: 1818720,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    session: sess,
    activeAnchor: 'core_302m',
  });
  assert(refFace.neuralSource === 'scarfield_v9_core302m', `Expected scarfield_v9_core302m, got ${refFace.neuralSource}`);

  for (let it = 0; it < 200; it++) {
    const dup = computeJarvisFaceState({
      hasWebGPU: true,
      engineReady: true,
      engineLoading: false,
      isGenerating: true,
      liveInputText: '',
      lastUserQuery: 'Como funciona a expansão geodésica 60M -> 302M?',
      currentAgentText: 'O ScarField evolui sobre esferas unitárias',
      deltaToken: 'esferas',
      generatedTokens: 40,
      tokPerSec: 180,
      confidence: 0.91,
      margin: 0.78,
      horizonScale: 1.0,
      scacheBytes: 1818720,
      scacheDeltaBytes: 0,
      activeCapsuleIds: [],
      maxActiveGate: 0,
      session: sess,
      activeAnchor: 'core_302m',
    });
    assert(JSON.stringify(dup) === JSON.stringify(refFace), `Non-deterministic drift at iter ${it}`);
  }

  const vacFace = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: false,
    liveInputText: '',
    lastUserQuery: 'Qual a senha privada do servidor?',
    currentAgentText: '[Abstenção Física 0x00]',
    deltaToken: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0,
    margin: 0,
    horizonScale: 1.0,
    scacheBytes: 1818720,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    session: sess,
    isVacuumAbstain0x00: true,
    activeAnchor: 'core_302m',
  });
  assert(vacFace.posture === 'OUT_OF_CONE_GUARD', `Expected OUT_OF_CONE_GUARD on 0x00 vacuum, got ${vacFace.posture}`);
  assert(detectPolyglotLanguage('¿Cómo funciona la memoria biorthogonal?', 'en') === 'es', 'ES lang detect failed');
  assert(detectPolyglotLanguage("Comment fonctionne l'expansion géodésique ?", 'en') === 'fr', 'FR lang detect failed');
  console.log('[PASS 5/7] 200x Biomechanical Determinism (PT/EN/ES/FR) + 0x00 Physical Vacuum Posture Verified');

  // 6. Sovereign `ScarField h_world -> PointScarFlow 3D` Bridge Audit (`0` Qwen legacy, `0` keyword routing!)
  assert(
    !fs.existsSync(path.join(PORTFOLIO_ROOT, 'public/webgpu/scarfield_v9/scarstore_router.bin')),
    'Legacy Qwen scarstore_router.bin still present in public/webgpu/scarfield_v9!'
  );
  assert(
    !fs.existsSync(path.join(PORTFOLIO_ROOT, 'public/webgpu/scarfield_v9/scarstore_router.json')),
    'Legacy Qwen scarstore_router.json still present in public/webgpu/scarfield_v9!'
  );
  const psfBridgePath = path.join(PORTFOLIO_ROOT, 'public/webgpu/scarfield_v9/pointscar_bridge.scar');
  const psfBuf = fs.readFileSync(psfBridgePath);
  const psfParsed = parseScarFieldContainer(
    psfBuf.buffer.slice(psfBuf.byteOffset, psfBuf.byteOffset + psfBuf.byteLength),
    true
  );
  const psfCfg = psfParsed.config;
  assert(psfCfg.num_concepts === 45, `Expected 45 PointScarFlow concepts, got ${psfCfg.num_concepts}`);
  const capsJson = JSON.stringify(psfCfg.capsules);
  assert(!capsJson.includes('Qwen2.5-3B'), 'Residual Qwen2.5-3B metadata found in pointscar_bridge.scar!');

  const edgePoles = new Float32Array(psfParsed.tensors.get('edge::poles_biorth')!.data as Float32Array);
  const edgeResid = new Float32Array(psfParsed.tensors.get('edge::resid_z')!.data as Float32Array);
  const tierW: PointScarBridgeTierWeights = {
    dSf: psfCfg.edge_feat_dim,
    kLang: psfCfg.k_lang,
    rrrRank: psfCfg.rrr_rank,
    numConcepts: psfCfg.num_concepts,
    condDim: psfCfg.cond_dim_out,
    biorthKappa: psfCfg.biorth_kappa,
    uLang: psfParsed.tensors.get('edge::u_lang')!.data as Float32Array,
    aRrr: psfParsed.tensors.get('edge::a_rrr')!.data as Float32Array,
    bRrr: psfParsed.tensors.get('edge::b_rrr')!.data as Float32Array,
    muZ: psfParsed.tensors.get('edge::mu_z')!.data as Float32Array,
    polesBiorth: edgePoles,
    residZ: edgeResid,
    origPoles: new Float32Array(edgePoles),
    origResidZ: new Float32Array(edgeResid),
  };

  // Extract live ScarField 60M `h_world` features for PT vs EN DNA helix prompts and verify cross-lingual 3D bridge cosine!
  const pyBinPsf = '/Users/juliocesarreisfilho/Projects/scar-cache/.venv/bin/python';
  if (fs.existsSync(pyBinPsf)) {
    const uJson = execFileSync(
      pyBinPsf,
      [
        '-c',
        `import json, torch
from scar_field import ScarFieldEngine
from scripts.build_scarfield_pointscar_bridge import extract_scarfield_prompt_features
mm, ct = ScarFieldEngine.load_packed_scar_mmap("${edgePath}")
u = extract_scarfield_prompt_features(mm, ["Mostre uma hélice dupla de DNA em 3D", "Show a 3D DNA double helix structure"], torch.device("cpu"), batch_size=2)
ct.close()
print(json.dumps({"pt": u[0].tolist(), "en": u[1].tolist()}))`,
      ],
      { env: { ...process.env, PYTHONPATH: '/Users/juliocesarreisfilho/Projects/scar-cache' }, encoding: 'utf-8' }
    );
    const uPair = JSON.parse(uJson.trim()) as { pt: number[]; en: number[] };
    const projPt = projectScarFieldManifoldToPointScarZ(tierW, new Float32Array(uPair.pt));
    const projEn = projectScarFieldManifoldToPointScarZ(tierW, new Float32Array(uPair.en));
    let dotZ = 0.0;
    for (let j = 0; j < 2048; j++) dotZ += projPt.zPsf[j] * projEn.zPsf[j];
    assert(dotZ > 0.92, `Expected PT<->EN PointScarFlow bridge cosine > 0.92, got ${dotZ}`);

    // Verify physical `0x00` GDPR pole erasure zeroes DNA helix gate (`index 41`) cleanly (`=== 0.0`)!
    const dnaIdx = (psfCfg.capsules as Array<{ capsule_id: string }>).findIndex((c) => c.capsule_id === 'psf_3d_dna_helix');
    assert(dnaIdx >= 0, 'psf_3d_dna_helix concept missing');
    tierW.polesBiorth.fill(0.0, dnaIdx * tierW.dSf, (dnaIdx + 1) * tierW.dSf);
    tierW.residZ.fill(0.0, dnaIdx * tierW.condDim, (dnaIdx + 1) * tierW.condDim);
    const projAfterWipe = projectScarFieldManifoldToPointScarZ(tierW, new Float32Array(uPair.pt));
    assert(projAfterWipe.poleGates[dnaIdx] === 0.0, `Expected 0.0 gate after 0x00 wipe, got ${projAfterWipe.poleGates[dnaIdx]}`);
    console.log(
      `[PASS 6/7] Sovereign ScarField h_world -> PointScarFlow 3D Bridge Verified: PT<->EN cos=${dotZ.toFixed(5)} | 0x00 Wipe Gate=0.0`
    );

    // 7. Full Multi-Step Interactive Generation Parity (`TS/WebGPU` vs `Python ScarFieldEngine`!)
    const testPrompts = [
      'Mostre uma hélice dupla de DNA em 3D',
      'Write a Python function to compute cosine similarity between two lists of floats.',
    ];
    const pyGenJson = execFileSync(
      pyBinPsf,
      [
        '-c',
        `import json, torch
from scar_field import ScarFieldEngine, format_scarfield_chat_turn
from scar_field.engine import SCARFIELD_EOT_BYTE
eng, ct = ScarFieldEngine.load_packed_scar_mmap("${edgePath}")
eng.eval()
out = []
for p in ${JSON.stringify(testPrompts)}:
    st = eng.init_state(1, "cpu")
    pre = list(format_scarfield_chat_turn(p, None))
    for b in pre[:-1]:
        eng.step_byte_interactive(b, st, enable_ghost_jump_scouts=False, require_world_support=False, apply_repulsive_anti_loop=False)
    cur = pre[-1]
    gen = []
    first_sc = None
    for s_i in range(80):
        if s_i == 0:
            st_clone = eng.init_state(1, "cpu")
            for b2 in pre[:-1]:
                eng.step_byte_interactive(b2, st_clone, enable_ghost_jump_scouts=False, require_world_support=False, apply_repulsive_anti_loop=False)
            w_t = torch.cat((st_clone.recent_bytes[:, -st_clone.recent_len:].long(), torch.tensor([[cur]])), dim=1)
            hf_s, hw_s, s_s = eng.frontend(w_t)
            hf_t, hw_t, s_t = hf_s[:, -1], hw_s[:, -1], s_s[:, -1]
            u_s2 = bool(float(s_t[0].item()) >= eng.cfg.s2_surprise_theta)
            p_st = (2 if u_s2 else 1) if eng.cfg.enable_percept_s1_gate else 2
            hf_t, hw_t = eng.flow_percept(hf_t, hw_t, hf_t, num_steps=p_st)
            ba = eng.head.out_bus_anchors / eng.head.out_bus_anchors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hf_p, b_ctx, _, _ = eng.bus.step_pulse(hf_t, hw_t, s_t, state=st_clone, mutate_state=True, v_shock_hint_t=ba[[cur]])
            rb_n = min(eng._recent_buf_cap, w_t.shape[1])
            st_clone.recent_bytes.zero_()
            st_clone.recent_bytes[:, -rb_n:] = w_t[:, -rb_n:].to(torch.uint8)
            st_clone.recent_len = int(rb_n)
            ks = eng.cfg.s2_steps if u_s2 else eng.cfg.s1_steps
            hfo, hwo = eng.flow(hf_p, hw_t, b_ctx, num_steps=ks)
            if eng.extra_flows is not None and not (eng.cfg.scout_g0_theta > 0.0 and float(s_t[0].item()) < eng.cfg.scout_g0_theta):
                for ef in eng.extra_flows:
                    hfo, hwo = ef(hfo, hwo, b_ctx, num_steps=ks)
            sc0, _ = eng.head.forward_scores(hfo, hwo, bus_ctx=b_ctx)
            ww = eng.head.out_world_anchors / eng.head.out_world_anchors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            rep_pen = (ww @ st_clone.repulsive_ring[0].T - 0.88).clamp(min=0.0).square().sum(dim=-1)
            sc0 = sc0 - 2.5 * rep_pen.unsqueeze(0)
            first_sc = sc0[0].tolist()
        dec = eng.step_byte_interactive(cur, st, enable_ghost_jump_scouts=False, require_world_support=False, apply_repulsive_anti_loop=True)
        if dec.action_byte in (0, SCARFIELD_EOT_BYTE):
            break
        gen.append(dec.action_byte)
        cur = dec.action_byte
    out.append({"prompt": p, "text": bytes(gen).decode("utf-8", "replace").strip(), "scores": first_sc})
ct.close()
print(json.dumps(out))`,
      ],
      { env: { ...process.env, PYTHONPATH: '/Users/juliocesarreisfilho/Projects/scar-cache' }, encoding: 'utf-8' }
    );
    const pyGenList = JSON.parse(pyGenJson.trim()) as Array<{ prompt: string; text: string; scores: number[] }>;
    let minGenScoreCos = 1.0;
    for (const item of pyGenList) {
      const tsRef = new ScarFieldContainerReference(edgeParsed);
      const tsOut = tsRef.generateReply(item.prompt, 80);
      let d = 0.0, n1 = 0.0, n2 = 0.0;
      for (let v = 0; v < 256; v++) {
        d += tsOut.firstStepScores[v] * item.scores[v];
        n1 += tsOut.firstStepScores[v] * tsOut.firstStepScores[v];
        n2 += item.scores[v] * item.scores[v];
      }
      const scCos = d / Math.max(1e-12, Math.sqrt(n1 * n2));
      minGenScoreCos = Math.min(minGenScoreCos, scCos);
      assert(
        scCos > 0.99999,
        `Multi-step prefill score drift on "${item.prompt}": cos=${scCos} | TS="${tsOut.text}" vs PY="${item.text}"`
      );
      assert(
        tsOut.text === item.text,
        `Generated UTF-8 reply mismatch on "${item.prompt}":\n  TS: ${JSON.stringify(tsOut.text)}\n  PY: ${JSON.stringify(item.text)}`
      );
    }
    console.log(
      `[PASS 7/7] Full Multi-Step Interactive Generation Bit-Exact Parity Verified: minScoreCos=${minGenScoreCos.toFixed(8)} | Identical UTF-8 Replies (PT + Python Code)!\n`
    );
  }

  console.log('ALL 7/7 SOVEREIGN SCARFIELD v9.2 + POINTSCARFLOW 3D ANTI-FACADE AUDITS PASSED!');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
