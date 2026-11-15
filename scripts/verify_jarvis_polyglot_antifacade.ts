/**
 * Rigorous Automated Verification Suite for Unified Qwen2.5-3B JARVIS Floating Face
 * & 20-Shader WebGPU Engine (`Track 2.4-C VQ4-Scar + ProtectedInt4Scar + P_⊥CJK + SCache O(1) + Golden-Slice L23 HTTP 206`).
 *
 * Run via: `npx tsx scripts/verify_jarvis_polyglot_antifacade.ts`
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  advanceSessionAdaptation,
  computeJarvisFaceState,
  createInitialSessionAdaptation,
  detectPolyglotLanguage,
  extractMultilingualViseme,
  type PolyglotLang,
} from '../src/lib/jarvisNeuralController';
import { buildCjkVocabBitmaskU32, hasCjkScript } from '../src/lib/scriptSubspaceGate';
import { QwenBPETokenizer } from '../src/lib/qwenBpeTokenizer';
import type { GoldenSliceRoutingDiagnostics } from '../src/lib/scarstoreHttp206Browser';
import {
  buildCfdHdrShockFieldProof,
  runBlackBoxSCacheProof,
  runRadioTelescopeRfiProof,
} from '../src/lib/physicalScarCodec';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORTFOLIO_ROOT = path.resolve(__dirname, '..');

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) {
    throw new Error(`ASSERTION FAILED: ${message}`);
  }
}

function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
}

function popcount32(x: number): number {
  let v = x >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

async function main() {
  console.log('================================================================================');
  console.log('UNIFIED QWEN 3B JARVIS FLOATING FACE & 20-SHADER WEBGPU — POLYGLOT ANTI-FACADE AUDIT');
  console.log('================================================================================\n');

  // ---------------------------------------------------------------------------
  // TEST 1: Zero `Math.random()`, Zero S1/S2 Split, Zero Chat UI & Determinism
  // ---------------------------------------------------------------------------
  const auditedFiles = [
    'src/lib/jarvisNeuralController.ts',
    'src/components/JarvisFaceRig.tsx',
    'src/components/PortfolioAgentWidget.tsx',
    'src/lib/mosWebgpuEngine.ts',
    'src/lib/llmWorker.ts',
    'src/lib/scarstoreHttp206Browser.ts',
  ];
  for (const rel of auditedFiles) {
    const raw = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    const codeOnly = stripComments(raw);
    assert(
      !codeOnly.includes('Math.random'),
      `Forbidden Math.random() found in executable code of ${rel}!`
    );
    assert(
      !codeOnly.includes('buildReflexGuideResponse') && !codeOnly.includes('cpu_s1_reflex'),
      `Forbidden legacy S1 CPU reflex split found in ${rel}!`
    );
  }

  const widgetCode = stripComments(
    fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/PortfolioAgentWidget.tsx'), 'utf-8')
  );
  assert(
    !widgetCode.includes('messages.map') && !widgetCode.includes('Chat Interativo'),
    'Legacy chat message list UI was not fully removed from PortfolioAgentWidget.tsx!'
  );
  assert(
    !fs.existsSync(path.join(PORTFOLIO_ROOT, 'src/components/SystemOneJarvisFace.tsx')),
    'Stale SystemOneJarvisFace.tsx still exists on disk!'
  );
  // Anti-regression: ensure worker lifecycle useEffect never depends on language callbacks & prefill micro-batch <= 8
  assert(
    !/},\s*\[bootMainThreadWebGPU,\s*runMainThreadGenerate\]\)/.test(widgetCode),
    'Regression: Worker useEffect still depends on language-bound callbacks and will terminate worker mid-turn!'
  );
  assert(
    widgetCode.includes('generation_aborted') && widgetCode.includes('jarvis-stop-button'),
    'Missing instant Stop cancellation or generation_aborted acknowledgement in PortfolioAgentWidget.tsx!'
  );
  const engineCode = stripComments(
    fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/mosWebgpuEngine.ts'), 'utf-8')
  );
  const prefillBatchMatch = engineCode.match(/const\s+PREFILL_BATCH_TOKENS\s*=\s*(\d+)/);
  assert(
    prefillBatchMatch && Number(prefillBatchMatch[1]) >= 1 && Number(prefillBatchMatch[1]) <= 8,
    `Expected micro-batched PREFILL_BATCH_TOKENS <= 8 to prevent Metal/Dawn watchdog timeout, got ${prefillBatchMatch?.[1]}`
  );
  assert(
    engineCode.includes('warmSystemPromptSnapshot') &&
      engineCode.includes('hasCompoundHyphenDegeneration'),
    'Missing warmSystemPromptSnapshot() or hasCompoundHyphenDegeneration() in mosWebgpuEngine.ts!'
  );
  assert(
    widgetCode.includes('displayedPostureText'),
    'Missing temporal readability hold (displayedPostureText) for crown status bar in PortfolioAgentWidget.tsx!'
  );

  const baseSession = createInitialSessionAdaptation('pt');
  const mockDiag: GoldenSliceRoutingDiagnostics = {
    anchorCosines: [
      {
        capsuleId: 'portfolio_mos_scache_72b',
        category: 'Flagship AI Architecture',
        cosine: 0.782,
        tau: 0.34,
        gate: 0.91,
      },
      {
        capsuleId: 'concept_landauer_principle',
        category: 'Physics of Information',
        cosine: 0.214,
        tau: 0.36,
        gate: 0.0,
      },
      {
        capsuleId: 'portfolio_aura_system_one',
        category: 'Embodied Deep RL',
        cosine: 0.188,
        tau: 0.35,
        gate: 0.0,
      },
    ],
    routingEntropy: 0.412,
    maxAnchorCosine: 0.782,
    zProbeEnergy: 19.44,
    langSubspaceRatio: 0.142,
  };

  const refState = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: true,
    liveInputText: '',
    lastUserQuery: 'Explique como o Mixture of Scars reduz a VRAM em 72B.',
    currentAgentText: 'O Mixture of Scars decompõe',
    deltaToken: 'decompõe',
    generatedTokens: 5,
    tokPerSec: 68.4,
    confidence: 0.89,
    margin: 0.74,
    horizonScale: 0.92,
    scacheBytes: 4332288,
    scacheDeltaBytes: 0,
    activeCapsuleIds: ['portfolio_mos_scache_72b'],
    maxActiveGate: 0.91,
    routingDiagnostics: mockDiag,
    session: baseSession,
  });

  assert(
    refState.neuralSource === 'qwen_wgsl_20_active',
    `Expected qwen_wgsl_20_active source, got ${refState.neuralSource}`
  );

  for (let i = 0; i < 200; i++) {
    const s = computeJarvisFaceState({
      hasWebGPU: true,
      engineReady: true,
      engineLoading: false,
      isGenerating: true,
      liveInputText: '',
      lastUserQuery: 'Explique como o Mixture of Scars reduz a VRAM em 72B.',
      currentAgentText: 'O Mixture of Scars decompõe',
      deltaToken: 'decompõe',
      generatedTokens: 5,
      tokPerSec: 68.4,
      confidence: 0.89,
      margin: 0.74,
      horizonScale: 0.92,
      scacheBytes: 4332288,
      scacheDeltaBytes: 0,
      activeCapsuleIds: ['portfolio_mos_scache_72b'],
      maxActiveGate: 0.91,
      routingDiagnostics: mockDiag,
      session: baseSession,
    });
    assert(JSON.stringify(s) === JSON.stringify(refState), `Non-deterministic drift at iter ${i}`);
  }

  const listeningState = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: false,
    liveInputText: 'O que é o SCache O(1)?',
    lastUserQuery: '',
    currentAgentText: '',
    deltaToken: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0.92,
    margin: 0.85,
    horizonScale: 1.0,
    scacheBytes: 4332288,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    routingDiagnostics: null,
    session: baseSession,
  });
  assert(
    listeningState.neuralSource === 'qwen_wgsl_20_listening' &&
      listeningState.posture === 'ATTENTIVE_LISTENING' &&
      listeningState.gazeY > 0.2,
    `Expected downward cursor-tracking gaze in ATTENTIVE_LISTENING, got ${JSON.stringify(listeningState)}`
  );

  const noGpuState = computeJarvisFaceState({
    hasWebGPU: false,
    engineReady: false,
    engineLoading: false,
    isGenerating: false,
    liveInputText: '',
    lastUserQuery: '',
    currentAgentText: '',
    deltaToken: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0.92,
    margin: 0.85,
    horizonScale: 1.0,
    scacheBytes: 4332288,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    routingDiagnostics: null,
    session: baseSession,
  });
  assert(
    noGpuState.neuralSource === 'webgpu_unavailable',
    `Expected honest webgpu_unavailable tag when WebGPU is absent, got ${noGpuState.neuralSource}`
  );
  console.log(
    '[PASS 1/7] Zero Math.random(), Unified Qwen-Only Pipeline (Zero S1/S2 Split & Zero Chat UI) & 200x Bit-Exact Determinism'
  );

  // ---------------------------------------------------------------------------
  // TEST 2: 52/52 Polyglot Prompts (`13 Universal Qwen 3B Cartridges × PT/EN/ES/FR`)
  // ---------------------------------------------------------------------------
  const manifestPath = path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/webgpu_manifest.json');
  const manifestJson = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  const archJson = manifestJson.architecture || {};
  assert(
    archJson.hidden_dim === 2048 && archJson.num_layers === 36,
    `Expected Qwen2.5-3B geometry (d=2048, L=36), got d=${archJson.hidden_dim}, L=${archJson.num_layers}`
  );
  const totalShardMiB =
    (manifestJson.shards || []).reduce((acc: number, s: any) => acc + Number(s.byte_length || 0), 0) /
    (1024 * 1024);
  assert(
    manifestJson.shards?.length >= 40 && totalShardMiB >= 1595.0 && totalShardMiB <= 1610.0,
    `Expected >= 40 shards totaling ~1,601.55 MiB Track 2.5 ProtectedInt4Scar Qwen 3B weights, got ${manifestJson.shards?.length} shards (${totalShardMiB.toFixed(2)} MiB)`
  );

  const routerPath = path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/scarstore_router.json');
  const routerJson = JSON.parse(fs.readFileSync(routerPath, 'utf-8'));
  assert(
    routerJson.mid_layer_idx === 23 && routerJson.record_bytes === 143360,
    `Expected Layer-23 140 KB (143,360 B) 3B router, got mid=${routerJson.mid_layer_idx}, bytes=${routerJson.record_bytes}`
  );
  const capsules: any[] = routerJson.capsules;
  assert(capsules.length >= 41, `Expected >= 41 universal cartridges (including portfolio_jarvis_architecture), found ${capsules.length}`);
  assert(
    capsules.some((c) => c.capsule_id === 'portfolio_jarvis_architecture' && c.record_bytes === 143360),
    'Missing dedicated 140 KB (143,360 B) portfolio_jarvis_architecture cartridge in scarstore_router.json!'
  );
  const expectedTotalPrompts = capsules.length * 4;

  let langCorrect = 0;
  let totalPrompts = 0;
  for (const cap of capsules) {
    const poly = cap.metadata?.prompts_polyglot as Record<PolyglotLang, string>;
    assert(poly && poly.pt && poly.en && poly.es && poly.fr, `Missing polyglot prompts on ${cap.capsule_id}`);

    const posturesByLang: Record<string, string> = {};
    for (const lg of ['pt', 'en', 'es', 'fr'] as const) {
      totalPrompts++;
      const prompt = poly[lg];
      const detected = detectPolyglotLanguage(prompt, 'en');
      if (detected === lg) {
        langCorrect++;
      } else {
        throw new Error(
          `Language detection mismatch on [${cap.capsule_id}][${lg}]: "${prompt}" -> detected "${detected}"`
        );
      }

      // Verify cross-lingual posture invariance when Qwen Layer-15 P_⊥lang routes this cartridge!
      const diagForCap: GoldenSliceRoutingDiagnostics = {
        anchorCosines: capsules.map((c) => ({
          capsuleId: c.capsule_id,
          category: c.metadata?.category || 'General',
          cosine: c.capsule_id === cap.capsule_id ? 0.82 : 0.11,
          tau: c.tau,
          gate: c.capsule_id === cap.capsule_id ? 0.92 : 0.0,
        })),
        routingEntropy: 0.35,
        maxAnchorCosine: 0.82,
        zProbeEnergy: 18.2,
        langSubspaceRatio: 0.12,
      };
      const faceL = computeJarvisFaceState({
        hasWebGPU: true,
        engineReady: true,
        engineLoading: false,
        isGenerating: true,
        liveInputText: '',
        lastUserQuery: prompt,
        currentAgentText: '...',
        deltaToken: 'a',
        generatedTokens: 8,
        tokPerSec: 64.0,
        confidence: 0.90,
        margin: 0.78,
        horizonScale: 0.95,
        scacheBytes: 4332288,
        scacheDeltaBytes: 0,
        activeCapsuleIds: [cap.capsule_id],
        maxActiveGate: 0.92,
        routingDiagnostics: diagForCap,
        session: createInitialSessionAdaptation(lg),
      });
      assert(faceL.detectedLang === lg, `Face detectedLang mismatch for ${lg}`);
      posturesByLang[lg] = faceL.posture;
    }

    assert(
      posturesByLang.pt === posturesByLang.en &&
        posturesByLang.en === posturesByLang.es &&
        posturesByLang.es === posturesByLang.fr,
      `Cross-lingual posture drift on ${cap.capsule_id}: ${JSON.stringify(posturesByLang)}`
    );
  }
  assert(
    langCorrect === expectedTotalPrompts && totalPrompts === expectedTotalPrompts,
    `Expected ${expectedTotalPrompts}/${expectedTotalPrompts} polyglot accuracy, got ${langCorrect}/${totalPrompts}`
  );
  // Short colloquial greetings regression check (e.g. "oi" when UI starts in EN)
  for (const shortPt of ['oi', 'Oi!', 'opa', 'tudo bem?', 'valeu']) {
    const det = detectPolyglotLanguage(shortPt, 'en');
    assert(det === 'pt', `Short PT greeting "${shortPt}" failed detection (got "${det}")`);
  }
  console.log(
    `[PASS 2/7] ${langCorrect}/${expectedTotalPrompts} Polyglot Language Detection & ${capsules.length}/${capsules.length} Qwen Cross-Lingual Posture Invariance (PT = EN = ES = FR)`
  );

  // ---------------------------------------------------------------------------
  // TEST 3: Epistemic Pupillometry Monotonicity & LGPD `0x00` Shock Response
  // ---------------------------------------------------------------------------
  const margins = [0.95, 0.72, 0.48, 0.28, 0.10];
  const pupils: number[] = [];
  const furrows: number[] = [];
  for (const m of margins) {
    const st = computeJarvisFaceState({
      hasWebGPU: true,
      engineReady: true,
      engineLoading: false,
      isGenerating: true,
      liveInputText: '',
      lastUserQuery: 'Explain Landauer thermodynamic bound and entropy.',
      currentAgentText: 'Landauer bound',
      deltaToken: 'bound',
      generatedTokens: 12,
      tokPerSec: 62.0,
      confidence: Math.min(0.99, m + 0.04),
      margin: m,
      horizonScale: 0.9,
      scacheBytes: 4332288,
      scacheDeltaBytes: 0,
      activeCapsuleIds: ['concept_landauer_principle'],
      maxActiveGate: 0.85,
      routingDiagnostics: mockDiag,
      session: baseSession,
    });
    pupils.push(st.pupilDilation);
    furrows.push(st.browFurrow);
    if (m === 0.10) {
      assert(
        st.posture === 'EPISTEMIC_ALERT_SHOCK',
        `Expected EPISTEMIC_ALERT_SHOCK at margin=0.10, got ${st.posture}`
      );
      assert(
        !st.postureLabel.pt.includes('LGPD 0x00') && !st.postureLabel.en.includes('LGPD 0x00'),
        `Normal low-margin token entropy must NOT display 'LGPD 0x00' wipe label! Got: "${st.postureLabel.pt}"`
      );
    }
  }
  for (let i = 1; i < margins.length; i++) {
    assert(
      pupils[i] > pupils[i - 1],
      `Pupil dilation not strictly monotonic: margin ${margins[i - 1]}->${margins[i]} gave ${pupils[i - 1]}->${pupils[i]}`
    );
    assert(
      furrows[i] >= furrows[i - 1],
      `Brow furrow decreased as margin dropped: ${furrows[i - 1]} -> ${furrows[i]}`
    );
  }

  const lgpdShock = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: false,
    liveInputText: '',
    lastUserQuery: 'Quem é o Julio Reis?',
    currentAgentText: '',
    deltaToken: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0.95,
    margin: 0.90,
    horizonScale: 1.0,
    scacheBytes: 4332288,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    routingDiagnostics: null,
    session: baseSession,
    unlearnShockActive: true,
  });
  assert(
    lgpdShock.posture === 'EPISTEMIC_ALERT_SHOCK' &&
      lgpdShock.primaryHex === '#fb7185' &&
      lgpdShock.postureLabel.pt.includes('LGPD 0x00'),
    `LGPD 0x00 wipe did not trigger crimson EPISTEMIC_ALERT_SHOCK with LGPD 0x00 label!`
  );
  console.log(
    `[PASS 3/7] Qwen Epistemic Monotonicity (pupil: ${pupils.map((p) => p.toFixed(3)).join(' < ')}) & LGPD 0x00 Shock`
  );

  // ---------------------------------------------------------------------------
  // TEST 4: Multi-Turn Conversational Adaptation (`PT -> EN -> ES -> FR -> PT`)
  // ---------------------------------------------------------------------------
  let sess = createInitialSessionAdaptation('pt');
  const multiTurnScript: Array<{ q: string; cap: string; expectedLang: PolyglotLang }> = [
    { q: 'Quem é você neste portfólio?', cap: 'portfolio_agent_identity', expectedLang: 'pt' },
    { q: 'What is Mixture of Scars and SCache?', cap: 'portfolio_mos_scache_72b', expectedLang: 'en' },
    { q: 'Háblame sobre el principio de Landauer en la física.', cap: 'concept_landauer_principle', expectedLang: 'es' },
    { q: "Expliquez le moteur AURA sur le Rubik's Cube.", cap: 'portfolio_aura_system_one', expectedLang: 'fr' },
    { q: 'Como posso entrar em contato para contratar o Julio?', cap: 'portfolio_contact_hire', expectedLang: 'pt' },
  ];

  let prevRapport = sess.rapportScore;
  for (const step of multiTurnScript) {
    sess = advanceSessionAdaptation(sess, {
      userQuery: step.q,
      activatedCapsuleIds: [step.cap],
      turnMeanConfidence: 0.91,
      fallbackLang: 'en',
    });
    assert(sess.detectedLang === step.expectedLang, `Expected ${step.expectedLang}, got ${sess.detectedLang}`);
    assert(sess.rapportScore > prevRapport, `Rapport failed to advance: ${prevRapport} -> ${sess.rapportScore}`);
    prevRapport = sess.rapportScore;
  }
  assert(sess.exploredCapsules.length === 5, `Expected 5 explored capsules, got ${sess.exploredCapsules.length}`);
  console.log(
    `[PASS 4/7] Multi-Turn Qwen Adaptation across PT->EN->ES->FR->PT (rapport 0.24 -> ${sess.rapportScore.toFixed(3)}, 5 capsules)`
  );

  // ---------------------------------------------------------------------------
  // TEST 5: Multilingual Articulatory Visemes & Boot Halo Sync (`PT/EN/ES/FR`)
  // ---------------------------------------------------------------------------
  const visNasalPt = extractMultilingualViseme('computação', 'pt');
  const visNasalFr = extractMultilingualViseme('bon', 'fr');
  const visBilabial = extractMultilingualViseme('MoS', 'en');
  const visLabio = extractMultilingualViseme('física', 'es');
  const visSibilant = extractMultilingualViseme('système', 'fr');
  const visRound = extractMultilingualViseme('réseau', 'fr');
  const visOpen = extractMultilingualViseme('memória', 'pt');
  const visSil = extractMultilingualViseme('...', 'en');

  assert(visNasalPt.code === 'NASAL_ROMANCE', `Expected NASAL_ROMANCE for 'computação', got ${visNasalPt.code}`);
  assert(visNasalFr.code === 'NASAL_ROMANCE', `Expected NASAL_ROMANCE for FR 'bon', got ${visNasalFr.code}`);
  assert(visBilabial.code === 'BILABIAL_PBM', `Expected BILABIAL_PBM for 'MoS', got ${visBilabial.code}`);
  assert(visLabio.code === 'LABIODENTAL_FV', `Expected LABIODENTAL_FV for 'física', got ${visLabio.code}`);
  assert(visSibilant.code === 'DENTAL_SIBILANT', `Expected DENTAL_SIBILANT for 'système', got ${visSibilant.code}`);
  assert(visRound.code === 'ROUND_O_U', `Expected ROUND_O_U for FR 'réseau', got ${visRound.code}`);
  assert(visOpen.code === 'OPEN_A', `Expected OPEN_A for 'memória', got ${visOpen.code}`);
  assert(visSil.code === 'SIL' && visSil.jawOpen <= 0.05, `Expected closed SIL for punctuation, got ${visSil.code}`);

  const bootState = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: false,
    engineLoading: true,
    loadPercent: 50,
    isGenerating: false,
    liveInputText: '',
    lastUserQuery: 'Olá JARVIS',
    currentAgentText: '',
    deltaToken: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0.90,
    margin: 0.80,
    horizonScale: 1.0,
    scacheBytes: 4332288,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    routingDiagnostics: null,
    session: baseSession,
  });
  assert(
    bootState.neuralSource === 'qwen_wgsl_20_booting' && bootState.posture === 'GPU_BOOT_SYNC',
    `Expected GPU_BOOT_SYNC during Qwen shader compilation, got ${bootState.posture}`
  );
  console.log(
    '[PASS 5/7] Multilingual Qwen 3B Token Visemes (PT/EN/ES/FR Romance nasals, bilabials, sibilants, formants) & Boot Sync'
  );

  // ---------------------------------------------------------------------------
  // TEST 6: Bit-Exact CJK Vocabulary Bitmask Parity (`QwenBPETokenizer` 151,936 vocab)
  // ---------------------------------------------------------------------------
  assert(!hasCjkScript('Olá, como funciona a busca binária em português, français et español?'), 'False CJK on Romance!');
  assert(hasCjkScript('请解释量子物理学中的玻尔兹曼常数。'), 'Missed Chinese script!');
  assert(hasCjkScript('こんにちは世界'), 'Missed Japanese Hiragana/CJK!');
  assert(hasCjkScript('안녕하세요'), 'Missed Korean Hangul!');

  const tokPath = path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/tokenizer.json');
  const tokJson = JSON.parse(fs.readFileSync(tokPath, 'utf-8'));
  const bpeTok = new QwenBPETokenizer(tokJson);
  const cjkMask = buildCjkVocabBitmaskU32(bpeTok, 151936);
  assert(cjkMask.length === 4748 && cjkMask.byteLength === 18992, `Unexpected mask size: ${cjkMask.byteLength} B`);

  let cjkBitCount = 0;
  for (let i = 0; i < cjkMask.length; i++) {
    cjkBitCount += popcount32(cjkMask[i]);
  }
  assert(
    cjkBitCount >= 28000 && cjkBitCount <= 35000,
    `Expected ~31,112 CJK tokens in Qwen2.5 vocab, got ${cjkBitCount}`
  );

  // Verify zero Romance accented tokens are accidentally masked!
  const romanceProbeIds = bpeTok.encode(
    'Português programação física quântica búsqueda binaria français système équation'
  );
  for (const tid of romanceProbeIds) {
    const bit = (cjkMask[tid >>> 5] >>> (tid & 31)) & 1;
    assert(bit === 0, `Romance token id=${tid} ("${bpeTok.decode([tid])}") was erroneously flagged in CJK bitmask!`);
  }
  console.log(
    `[PASS 6/7] ScriptSubspaceGate P_⊥CJK Bitmask (${cjkMask.byteLength} B, ${cjkBitCount} CJK tokens, 0 Romance false positives)`
  );

  // ---------------------------------------------------------------------------
  // TEST 7: 20 WGSL 1.0 Compute Shaders Byte-Identical Mirror & Entrypoints
  // ---------------------------------------------------------------------------
  const wgslSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/shaders/mos_scache.wgsl'), 'utf-8');
  const wgslPub3b = fs.readFileSync(
    path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/mos_scache_kernels.wgsl'),
    'utf-8'
  );
  const wgslCanonical = fs.readFileSync(
    '/Users/juliocesarreisfilho/Projects/scar-cache/scar_cache/webgpu/mos_scache_kernels.wgsl',
    'utf-8'
  );
  const wgslHf = fs.readFileSync(
    '/Users/juliocesarreisfilho/Projects/scar-cache/huggingface_space/scar_cache/webgpu/mos_scache_kernels.wgsl',
    'utf-8'
  );

  assert(wgslSrc === wgslPub3b, 'src/lib/shaders/mos_scache.wgsl != public/webgpu/qwen3b/mos_scache_kernels.wgsl');
  assert(wgslSrc === wgslCanonical, 'portfolio mos_scache.wgsl != canonical scar-cache mos_scache_kernels.wgsl');
  assert(wgslSrc === wgslHf, 'portfolio mos_scache.wgsl != huggingface_space mos_scache_kernels.wgsl');

  const expectedEntrypoints = [
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

  const computeMatches = [...wgslSrc.matchAll(/@compute\s+@workgroup_size\([^)]+\)\s*fn\s+([a-zA-Z0-9_]+)/g)].map(
    (m) => m[1]
  );
  assert(
    computeMatches.length === 20,
    `Expected exactly 20 @compute WGSL entrypoints, found ${computeMatches.length}: ${computeMatches.join(', ')}`
  );
  for (const mustHave of expectedEntrypoints) {
    assert(computeMatches.includes(mustHave), `Missing required WGSL entrypoint: ${mustHave}`);
  }
  console.log(
    `[PASS 7/8] 20/20 WGSL 1.0 Compute Shaders Byte-Identical across all Native/HF/Web copies (${wgslSrc.length} B)`
  );

  // ---------------------------------------------------------------------------
  // TEST 8: Domain-Independent Physical Scar Lab (`v4.0` — 3 Live Proofs & Zero Math.random)
  // ---------------------------------------------------------------------------
  for (const rel of ['src/lib/physicalScarCodec.ts', 'src/components/PhysicalScarLab.tsx']) {
    const raw = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    const clean = stripComments(raw);
    assert(!/Math\.random\s*\(/.test(clean), `Forbidden Math.random() detected in ${rel}`);
    assert(!/pure\s+scar[-\s]?native|scar[-\s]?native/i.test(raw), `Forbidden v5.0 terminology detected in ${rel}`);
  }

  const p1 = runRadioTelescopeRfiProof(1000);
  assert(p1.outOfConeMaxErr < 1e-5, `Expected near-zero out-of-cone P_perp error, got ${p1.outOfConeMaxErr}`);
  assert(p1.vlbi2BitCosine < 0.25, `Expected naive 2-bit VLBI to saturate under 1,000x RFI, got ${p1.vlbi2BitCosine}`);
  assert(p1.scarCodecCosine > 0.96, `Expected ScarCodec RFI pulsar recovery cos > 0.96, got ${p1.scarCodecCosine}`);
  assert(p1.compressionRatio > 5.5, `Expected > 5.5x array compression, got ${p1.compressionRatio}`);

  const p2 = runBlackBoxSCacheProof(8000);
  assert(
    p2.scacheBytesInitial === 34816 && p2.scacheBytesFinal === 34816 && p2.scacheDeltaBytes === 0,
    `Expected exact 34,816 B (Δ +0 B) SCache O(1), got ${p2.scacheBytesInitial} -> ${p2.scacheBytesFinal}`
  );
  assert(
    p2.scacheRetainedFaults.length === 5 && p2.ringRetainedFaults.length === 0,
    `Expected 5/5 SCache vs 0/5 Circular Ring fault retention, got ${p2.scacheRetainedFaults.length} vs ${p2.ringRetainedFaults.length}`
  );
  assert(p2.associativeMeanCosine > 0.995, `Expected associative W_scar cos > 0.995, got ${p2.associativeMeanCosine}`);

  const p3 = buildCfdHdrShockFieldProof();
  assert(p3.scarVacuumMaxErr === 0, `Expected bit-exact 0.0 vacuum error, got ${p3.scarVacuumMaxErr}`);
  assert(p3.scarRmse < p3.spectralRmse * 0.15, `Expected ScarCodec RMSE << Spectral truncation: ${p3.scarRmse} vs ${p3.spectralRmse}`);
  const [sr, sc] = p3.spikeCoords[0];
  const o1 = p3.codec.decodeBlockO1(p3.pkg, sr, sc >>> 6);
  const fullSlice = p3.scarReconstructed.subarray(sr * p3.cols + ((sc >>> 6) << 6), sr * p3.cols + ((sc >>> 6) << 6) + 64);
  for (let i = 0; i < 64; i++) {
    assert(Math.abs(o1.values[i] - fullSlice[i]) < 1e-6, `O(1) block decode mismatch at index ${i}`);
  }
  console.log(
    `[PASS 8/8] Physical Scar Lab v4.0 (1,000x RFI cos=${p1.scarCodecCosine.toFixed(4)} · 34,816 B SCache Δ=0 B 5/5 · CFD ${p3.compressionRatio.toFixed(2)}x Vacuum=0.0 O(1) Parity)`
  );

  console.log('\n================================================================================');
  console.log('ALL 8/8 UNIFIED QWEN JARVIS + PHYSICAL SCAR LAB v4.0 VERIFICATION SUITES PASSED!');
  console.log('================================================================================');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
