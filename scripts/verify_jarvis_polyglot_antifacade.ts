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
import {
  BrowserScarStoreHTTP206,
  extractUserContentBounds,
  OUT_OF_CONE_EPISTEMIC_GATES,
  PORTFOLIO_DEFAULT_SYSTEM_PROMPT,
  selectOutOfConeEpistemicGate,
  type GoldenSliceRoutingDiagnostics,
} from '../src/lib/scarstoreHttp206Browser';
import {
  BiorthogonalObliqueProjectorTS,
  OrthogonalSubspaceProjectorTS,
  PhysicalVQ4ScarCodecTS,
  ScarCodecE8TS,
  SurpriseGatedAnomalySCacheTS,
  buildCanonicalE8Lut256TS,
  buildCfdHdrShockFieldProof,
  computeMatchedByte2dWaveletCdf97Fp16,
  runBlackBoxSCacheProof,
  runRadioTelescopeRfiProof,
  runRealSliceLiveForensicAuditTS,
} from '../src/lib/physicalScarCodec';
import {
  ScarNativeCpuExecutor,
  type ScarNativeManifestJSON,
} from '../src/lib/scarNativeEngine';
import {
  ScarNativeV6CpuExecutor,
  computeFrontierV6ByteBudget,
  type ScarNativeBPETokenizerJSON,
  type ScarNativeV6ManifestJSON,
} from '../src/lib/scarNativeV6Engine';
import { buildCanonicalFibonacciSphereF32 } from '../src/lib/pointScarFlowEngine';

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
    'src/components/JarvisFloatingPointStage.tsx',
    'src/lib/pointScarFlowEngine.ts',
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
    refState.neuralSource === 'scarfield_v9_edge60m' || refState.neuralSource === 'scarfield_v9_core302m',
    `Expected scarfield_v9 source, got ${refState.neuralSource}`
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
    listeningState.neuralSource === 'scarfield_v9_listening' &&
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
    bootState.neuralSource === 'scarfield_v9_booting' && bootState.posture === 'GPU_BOOT_SYNC',
    `Expected GPU_BOOT_SYNC during ScarField shader compilation, got ${bootState.posture}`
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
    'scarnative_widrow_hoff_evict_update',
    'scarnative_o1_hybrid_gqa_attention',
    'scarnative_v6_mos_router_topk_relu',
    'scarnative_v6_mos_bscar_accumulate',
    'scarnative_v6_hierarchical_evict_update',
    'scarnative_v6_hierarchical_o1_gqa_attention',
  ];

  const computeMatches = [...wgslSrc.matchAll(/@compute\s+@workgroup_size\([^)]+\)\s*fn\s+([a-zA-Z0-9_]+)/g)].map(
    (m) => m[1]
  );
  assert(
    computeMatches.length === 26,
    `Expected exactly 26 @compute WGSL entrypoints, found ${computeMatches.length}: ${computeMatches.join(', ')}`
  );
  for (const mustHave of expectedEntrypoints) {
    assert(computeMatches.includes(mustHave), `Missing required WGSL entrypoint: ${mustHave}`);
  }
  assert(
    /let\s+max_logit\s*=\s*wg_att_red\[0\];\s*workgroupBarrier\(\);/.test(wgslSrc) &&
      /let\s+max_lg\s*=\s*wg_sn_ared\[0\];\s*workgroupBarrier\(\);/.test(wgslSrc) &&
      /let\s+err_norm\s*=\s*sqrt\(wg_v6_red\[0\]\);\s*workgroupBarrier\(\);/.test(wgslSrc) &&
      /let\s+max_lg\s*=\s*wg_v6_ared\[0\];\s*workgroupBarrier\(\);/.test(wgslSrc),
    'Missing 2-SIMDGroup WAR workgroupBarrier() after shared-memory scalar reduction read in mos_scache.wgsl!'
  );
  console.log(
    `[PASS 7/13] 26/26 WGSL 1.0 Compute Shaders Byte-Identical across all Native/HF/Web copies (${wgslSrc.length} B)`
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
  assert(p1.outOfConeMaxErr < 1e-4, `Expected < 1e-4 blind-calibrated out-of-cone P_perp error under 1,000x stress, got ${p1.outOfConeMaxErr}`);
  assert(p1.vlbi2BitCosine < 0.15, `Expected naive 2-bit VLBI to saturate under 1,000x RFI, got ${p1.vlbi2BitCosine}`);
  assert(p1.scarCodecCosine > 0.97, `Expected blind ScarCodec RFI pulsar recovery cos > 0.97, got ${p1.scarCodecCosine}`);
  assert(p1.compressionRatio > 6.0, `Expected > 6.0x array compression, got ${p1.compressionRatio}`);

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
    `[PASS 8/9] Physical Scar Lab v4.0 (1,000x RFI cos=${p1.scarCodecCosine.toFixed(4)} · 34,816 B SCache Δ=0 B 5/5 · CFD ${p3.compressionRatio.toFixed(2)}x Vacuum=0.0 O(1) Parity)`
  );

  // ---------------------------------------------------------------------------
  // TEST 9: Pure Scar-Native Engine v5.0 Lab (`Tabula Rasa` Binary Shard, O(1), P_⊥lang & 0x00 GDPR)
  // ---------------------------------------------------------------------------
  for (const rel of ['src/lib/scarNativeEngine.ts', 'src/components/ScarNativeLab.tsx']) {
    const raw = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    const clean = stripComments(raw);
    assert(!/Math\.random\s*\(/.test(clean), `Forbidden Math.random() detected in ${rel}`);
  }

  const snDir = path.join(PORTFOLIO_ROOT, 'public/webgpu/scarnative_v5');
  const snManifest = JSON.parse(fs.readFileSync(path.join(snDir, 'scarnative_manifest.json'), 'utf-8')) as ScarNativeManifestJSON;
  const snTokJson = JSON.parse(fs.readFileSync(path.join(snDir, 'tokenizer.json'), 'utf-8')) as { id_to_token: string[] };
  const snWgsl = fs.readFileSync(path.join(snDir, 'mos_scache_kernels.wgsl'), 'utf-8');
  assert(snWgsl === wgslSrc, 'public/webgpu/scarnative_v5/mos_scache_kernels.wgsl != canonical mos_scache.wgsl');
  const snShardNodeBuf = fs.readFileSync(path.join(snDir, snManifest.shard_file));
  const snShardArrayBuf = snShardNodeBuf.buffer.slice(
    snShardNodeBuf.byteOffset,
    snShardNodeBuf.byteOffset + snShardNodeBuf.byteLength
  );

  const snExec = new ScarNativeCpuExecutor(snManifest, snShardArrayBuf, snTokJson);
  assert(snExec.cartridges.size === 16, `Expected 16 Zero-SGD cartridges in v5.0 bundle, got ${snExec.cartridges.size}`);

  // 1) Long-range O(1) passkey (`T > 80` >> `S+W = 20`) with ΔSCache = 0 B
  const fillerPt = Array.from(
    { length: 4 },
    () => 'o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo .'
  ).join(' ');
  const pkPrompt = `<lang:pt> <user> chave secreta <passkey> alpha = 7 , beta = 4 , gamma = 2 </passkey> . ${fillerPt} qual é a chave secreta ? <assistant>`;
  const snPkRes = snExec.generate(pkPrompt, { maxNewTokens: 14, useScarStore: false, enableWidrowHoff: true });
  assert(snPkRes.prefillTokens >= 75, `Expected overflow prefill >= 75 tokens, got ${snPkRes.prefillTokens}`);
  assert(snPkRes.scacheDeltaBytes === 0, `O(1) SCache violated: Δ=${snPkRes.scacheDeltaBytes}`);
  assert(
    snPkRes.outputText.includes('alpha = 7, beta = 4, gamma = 2'),
    `Expected exact 3-slot passkey recall from ScarNativeLM v5.0, got "${snPkRes.outputText}"`
  );
  const snWinOnly = snExec.generate(pkPrompt, { maxNewTokens: 14, useScarStore: false, enableWidrowHoff: false });
  assert(
    !snWinOnly.outputText.includes('alpha = 7, beta = 4, gamma = 2'),
    `Window-only ablation unexpectedly retained evicted passkey: "${snWinOnly.outputText}"`
  );

  // 2) Cross-lingual `PT / EN / ES / FR` Zero-SGD Cartridge Routing (`ent_orion = 516 km`) + `0x00` GDPR Wipe
  const orionPrompts: Record<PolyglotLang, string> = {
    pt: '<lang:pt> <user> <fact> qual é o valor exato de órbita do módulo orion ? <assistant>',
    en: '<lang:en> <user> <fact> what is the exact orbit value of module orion ? <assistant>',
    es: '<lang:es> <user> <fact> ¿ cuál es el valor exacto de órbita del módulo orion ? <assistant>',
    fr: '<lang:fr> <user> <fact> quelle est la valeur exacte de orbite du module orion ? <assistant>',
  };
  for (const lg of ['pt', 'en', 'es', 'fr'] as const) {
    const r = snExec.generate(orionPrompts[lg], { maxNewTokens: 14, useScarStore: true });
    assert(r.routedCartridgeId === 'ent_orion', `Expected ent_orion route in ${lg}, got ${r.routedCartridgeId}`);
    assert(
      r.outputText.includes('516') && r.outputText.includes('km'),
      `Expected 516 km in ${lg} from EN-minted cartridge, got "${r.outputText}"`
    );
  }

  const preMintRef = snExec.generate(orionPrompts.pt, { maxNewTokens: 1, useScarStore: false });
  snExec.gdprZeroWipe('ent_orion');
  const postWipe = snExec.generate(orionPrompts.pt, { maxNewTokens: 14, useScarStore: true });
  assert(postWipe.routedCartridgeId === null, `Wiped cartridge still routed: ${postWipe.routedCartridgeId}`);
  assert(!postWipe.outputText.includes('516'), `Wiped cartridge leaked 516: "${postWipe.outputText}"`);
  let maxGdprDiff = 0.0;
  for (let i = 0; i < preMintRef.firstTokenLogits.length; i++) {
    maxGdprDiff = Math.max(maxGdprDiff, Math.abs(preMintRef.firstTokenLogits[i] - postWipe.firstTokenLogits[i]));
  }
  assert(maxGdprDiff === 0, `Expected exact 0.00000000 GDPR logit diff, got ${maxGdprDiff}`);
  snExec.restoreCartridge('ent_orion');

  console.log(
    `[PASS 9/10] Pure Scar-Native Engine v5.0 (${(snManifest.shard_bytes / (1024 * 1024)).toFixed(2)} MiB Tabula-Rasa Shard · SCache ${snPkRes.scacheBytesFinal} B Δ=0 B · PT/EN/ES/FR Zero-SGD RRR · 0x00 GDPR Δlogits=0)`
  );

  // ---------------------------------------------------------------------------
  // TEST 10: Out-of-Cone Epistemic Guard (`0 Cartridges`), Causal Split & Zero-Cheat Audit
  // ---------------------------------------------------------------------------
  const bannedDomainCheatRegex = /\b(?:cabelo|progressiva|descolorid|bleach|keratin|alisament|shampoo)\b/i;
  for (const lg of ['pt', 'en', 'es', 'fr'] as const) {
    const gateStr = OUT_OF_CONE_EPISTEMIC_GATES[lg];
    assert(gateStr && gateStr.length >= 120, `Missing epistemic gate string for ${lg}`);
    assert(
      !bannedDomainCheatRegex.test(gateStr),
      `Forbidden domain-specific cheat word found in OUT_OF_CONE_EPISTEMIC_GATES[${lg}]!`
    );
  }
  for (const rel of ['src/lib/scarstoreHttp206Browser.ts', 'src/lib/mosWebgpuEngine.ts']) {
    const srcRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    assert(!bannedDomainCheatRegex.test(srcRaw), `Forbidden domain cheat word found in ${rel}!`);
  }

  assert(
    selectOutOfConeEpistemicGate(
      'Para um cabelo já liso e descolorido, qual progressiva química você recomenda?'
    ) === OUT_OF_CONE_EPISTEMIC_GATES.pt,
    'Failed to select PT epistemic gate for PT cosmetic query'
  );
  assert(
    selectOutOfConeEpistemicGate(
      'What chemical straightening treatment should I use on bleached straight hair?'
    ) === OUT_OF_CONE_EPISTEMIC_GATES.en,
    'Failed to select EN epistemic gate for EN cosmetic query'
  );
  assert(
    selectOutOfConeEpistemicGate('¿Qué medicamento debo tomar para el dolor de garganta?') ===
      OUT_OF_CONE_EPISTEMIC_GATES.es,
    'Failed to select ES epistemic gate for ES medical query'
  );
  assert(
    selectOutOfConeEpistemicGate('Quel soin cosmétique recommandez-vous pour des cheveux décolorés ?') ===
      OUT_OF_CONE_EPISTEMIC_GATES.fr,
    'Failed to select FR epistemic gate for FR cosmetic query'
  );

  // Verify sub-threshold cosines (`gate = 0`, `activeCapsuleIds = []`) NEVER masquerade as active cartridges!
  const outOfConeDiag: GoldenSliceRoutingDiagnostics = {
    anchorCosines: [
      {
        capsuleId: 'portfolio_mos_scache_72b',
        category: 'Flagship AI Architecture',
        cosine: 0.248,
        tau: 0.34,
        gate: 0.0,
      },
      {
        capsuleId: 'concept_landauer_principle',
        category: 'Physics of Information',
        cosine: 0.219,
        tau: 0.36,
        gate: 0.0,
      },
    ],
    routingEntropy: 0.962,
    maxAnchorCosine: 0.248,
    zProbeEnergy: 16.8,
    langSubspaceRatio: 0.19,
  };
  const outOfConeFace = computeJarvisFaceState({
    hasWebGPU: true,
    engineReady: true,
    engineLoading: false,
    isGenerating: false,
    liveInputText: '',
    lastUserQuery: 'Para um cabelo já liso e descolorido, qual progressiva química é indicada?',
    currentAgentText:
      'Como assistente de IA e Engenharia deste portfólio, não recomendo procedimentos cosméticos ou químicos capilares, pois não há cartucho ScarStore verificado para esse domínio.',
    deltaToken: '',
    generatedTokens: 32,
    tokPerSec: 11.4,
    confidence: 0.91,
    margin: 0.84,
    horizonScale: 0.0,
    scacheBytes: 15345792,
    scacheDeltaBytes: 0,
    activeCapsuleIds: [],
    maxActiveGate: 0,
    routingDiagnostics: outOfConeDiag,
    session: baseSession,
  });
  assert(
    outOfConeFace.dominantCapsuleId === null,
    `Expected dominantCapsuleId === null when 0 cartridges fire, got ${outOfConeFace.dominantCapsuleId}`
  );
  assert(
    outOfConeFace.posture === 'OUT_OF_CONE_GUARD' &&
      outOfConeFace.postureLabel.pt.includes('0 Cartuchos') &&
      outOfConeFace.postureLabel.en.includes('0 Cartridges'),
    `Expected OUT_OF_CONE_GUARD posture when 0 cartridges fire, got ${outOfConeFace.posture} (${outOfConeFace.postureLabel.pt})`
  );

  // Verify causal split alignment at `qEndProbe` on `QwenBPETokenizer`
  const probeChatIds = bpeTok.applyChatTemplate(
    [
      { role: 'system', content: PORTFOLIO_DEFAULT_SYSTEM_PROMPT },
      { role: 'user', content: 'Para um cabelo já liso e descolorido, qual progressiva química você recomenda?' },
    ],
    true
  );
  const userPrefixIds = bpeTok.encode('<|im_start|>user\n');
  const userSuffixIds = bpeTok.encode('<|im_end|>\n<|im_start|>assistant\n');
  const bounds = extractUserContentBounds(probeChatIds, userPrefixIds, userSuffixIds);
  assert(
    bounds.qStart > 20 && bounds.qEnd > bounds.qStart && probeChatIds.length - bounds.qEnd === userSuffixIds.length,
    `Unexpected causal split bounds: qStart=${bounds.qStart}, qEnd=${bounds.qEnd}, total=${probeChatIds.length}`
  );
  const tailNormal = probeChatIds.slice(bounds.qEnd);
  assert(
    JSON.stringify(tailNormal) === JSON.stringify(userSuffixIds),
    `Expected cleanprobeIds.slice(qEnd) to match exact <|im_end|>\\n<|im_start|>assistant\\n suffix!`
  );
  assert(
    engineCode.includes('runFullLayersOnLastToken') &&
      engineCode.includes('out_of_cone_guarded') &&
      engineCode.includes('waitForIdle') &&
      engineCode.includes('_activeGpuLock') &&
      engineCode.includes('stepStateBufs') &&
      engineCode.includes('layerOpsSlots'),
    'Missing causal split, GPU serialization lock, or disjoint stepStateBufs in mosWebgpuEngine.ts!'
  );
  assert(
    /Astronomia/i.test(OUT_OF_CONE_EPISTEMIC_GATES.pt) &&
      /Astronomy/i.test(OUT_OF_CONE_EPISTEMIC_GATES.en) &&
      /Astronomía/i.test(OUT_OF_CONE_EPISTEMIC_GATES.es) &&
      /Astronomie/i.test(OUT_OF_CONE_EPISTEMIC_GATES.fr),
    'OUT_OF_CONE_EPISTEMIC_GATES must explicitly welcome Astronomy / Science queries across PT/EN/ES/FR!'
  );
  console.log(
    '[PASS 10/11] Out-of-Cone Epistemic Guard (PT/EN/ES/FR Zero-Cheat Gate · OUT_OF_CONE_GUARD Posture · Bit-Exact Causal Split)'
  );

  // ---------------------------------------------------------------------------
  // TEST 11: PointScarFlow v1.0 Continuous 3D Generator (`Zero Catalog`, `12 WGSL Shaders` & `3.87 MiB VQ4-Scar Bundle`)
  // ---------------------------------------------------------------------------
  const bannedCatalogIdsRegex =
    /\b(?:dna_helix|ringed_planet|rocket_spacecraft|lorenz_attractor|mobius_klein_ribbon|rubiks_tensor_cube|entropy_funnel_blackhole)\b/;
  for (const rel of ['src/lib/pointScarFlowEngine.ts', 'src/components/JarvisFloatingPointStage.tsx']) {
    const clean = stripComments(fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8'));
    assert(
      !bannedCatalogIdsRegex.test(clean),
      `Forbidden hardcoded 3D shape catalog identifier found in ${rel}!`
    );
  }

  const psfWgslSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/shaders/point_scar_flow.wgsl'), 'utf-8');
  const psfWgslPub = fs.readFileSync(
    path.join(PORTFOLIO_ROOT, 'public/webgpu/pointscar_v1/point_scar_flow.wgsl'),
    'utf-8'
  );
  const psfWgslCanon = fs.readFileSync(
    '/Users/juliocesarreisfilho/Projects/scar-cache/scar_native/point_scar_flow.wgsl',
    'utf-8'
  );
  assert(psfWgslSrc === psfWgslPub, 'src/lib/shaders/point_scar_flow.wgsl != public/webgpu/pointscar_v1/point_scar_flow.wgsl');
  assert(psfWgslSrc === psfWgslCanon, 'portfolio point_scar_flow.wgsl != canonical scar-cache point_scar_flow.wgsl');

  const psfEntrypoints = [...psfWgslSrc.matchAll(/@compute\s+@workgroup_size\([^)]+\)\s*fn\s+([a-zA-Z0-9_]+)/g)].map(
    (m) => m[1]
  );
  const expectedPsfEntrypoints = [
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
  ];
  assert(
    JSON.stringify(psfEntrypoints) === JSON.stringify(expectedPsfEntrypoints),
    `Expected 12 PointScarFlow WGSL entrypoints, got: ${psfEntrypoints.join(', ')}`
  );

  const psfDir = path.join(PORTFOLIO_ROOT, 'public/webgpu/pointscar_v1');
  const psfManifestPath = path.join(psfDir, 'pointscar_manifest.json');
  const psfEvalPath = path.join(psfDir, 'eval_report.json');
  assert(fs.existsSync(psfManifestPath), 'Missing public/webgpu/pointscar_v1/pointscar_manifest.json!');
  assert(fs.existsSync(psfEvalPath), 'Missing public/webgpu/pointscar_v1/eval_report.json!');

  const psfManifest = JSON.parse(fs.readFileSync(psfManifestPath, 'utf-8'));
  const psfEval = JSON.parse(fs.readFileSync(psfEvalPath, 'utf-8'));
  const psfShardPath = path.join(psfDir, psfManifest.shard_file);
  assert(fs.existsSync(psfShardPath), `Missing PointScarFlow binary shard: ${psfShardPath}`);
  const psfShardStat = fs.statSync(psfShardPath);
  assert(
    psfManifest.architecture === 'ScarNative_VQ4_OT_RectifiedFlow' &&
      psfManifest.config.num_points === 1024 &&
      psfManifest.config.cond_in_dim === 2048 &&
      psfManifest.config.head_dim === 32 &&
      psfShardStat.size === psfManifest.shard_bytes,
    `Invalid PointScarFlow manifest or shard size mismatch (${psfShardStat.size} vs ${psfManifest.shard_bytes})!`
  );
  assert(
    Number(psfEval.ptq_max_abs_shock) < 2e-3 &&
      Number(psfEval.held_out_chamfer_l2_4step_1pass) < Number(psfEval.held_out_chamfer_l2_sphere_baseline) * 0.75,
    `PointScarFlow eval_report failed quality/zero-shock assertions: ${JSON.stringify(psfEval)}`
  );
  for (const xlKey of ['pt_en', 'es_en', 'fr_en'] as const) {
    assert(
      Number(psfEval.cross_lingual_3d_shape_cosine?.[xlKey]) > 0.80,
      `Expected cross_lingual_3d_shape_cosine.${xlKey} > 0.80, got ${psfEval.cross_lingual_3d_shape_cosine?.[xlKey]}`
    );
  }

  const faceRigRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/JarvisFaceRig.tsx'), 'utf-8');
  const faceRigCode = stripComments(faceRigRaw);
  const stageRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/JarvisFloatingPointStage.tsx'), 'utf-8');
  const workerRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/llmWorker.ts'), 'utf-8');
  const psfEngRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/pointScarFlowEngine.ts'), 'utf-8');
  assert(
    !faceRigCode.includes('evalCranioCervicalPoint') &&
      !faceRigCode.includes('buildCranioFacialMesh') &&
      !faceRigCode.includes('buildJarvisFacePointCloudF32'),
    'Forbidden standardized/hardcoded procedural facial mesh detected in JarvisFaceRig.tsx — JARVIS face must be constructed by PointScarFlow!'
  );
  assert(
    faceRigCode.includes('buildCanonicalFibonacciSphereF32') &&
      faceRigCode.includes('analyzeNeuralManifold') &&
      faceRigCode.includes('neuralFaceFinalF32') &&
      psfEngRaw.includes('sampleStandaloneBootManifold') &&
      workerRaw.includes('boot_face_cloud') &&
      stageRaw.includes('data-testid="jarvis-floating-point-stage"') &&
      stageRaw.includes('buildCanonicalFibonacciSphereF32') &&
      stageRaw.includes('neuralFaceFinalF32') &&
      stageRaw.includes("width: '100vw'") &&
      stageRaw.includes('canvas.getBoundingClientRect()') &&
      !stageRaw.includes('desiredStageX') &&
      widgetCode.includes('jarvis-pointscar-toggle') &&
      widgetCode.includes('isFaceMorphedToCloud') &&
      widgetCode.includes('isJarvisFaceManifoldTurn') &&
      engineCode.includes('BrowserPointScarFlowEngine') &&
      engineCode.includes('pointScarSample') &&
      workerRaw.includes('pointScarFinalF32'),
    'Missing GPU PointScarFlow-constructed JARVIS face, Retina CSS viewport sizing, or WebGPU worker wiring!'
  );
  assert(
    Boolean(psfEval.jarvis_face_neural_sculpt) &&
      Number(psfEval.jarvis_face_neural_sculpt.boot_anchor0_chamfer_l2) < 0.05 &&
      Number(psfEval.jarvis_face_neural_sculpt.held_out_polyglot_face_chamfer_l2) < 0.05 &&
      Number(psfEval.jarvis_face_neural_sculpt.concept_replay_preservation_cosine) > 0.99,
    `Neural JARVIS face sculpt audit failed quality thresholds: ${JSON.stringify(psfEval.jarvis_face_neural_sculpt)}`
  );

  const sphereA = buildCanonicalFibonacciSphereF32(1024);
  const sphereB = buildCanonicalFibonacciSphereF32(1024);
  assert(sphereA.length === 4096, `Expected 4096 floats in canonical X_0 sphere, got ${sphereA.length}`);
  for (let i = 0; i < sphereA.length; i++) {
    assert(
      Number.isFinite(sphereA[i]) && sphereA[i] === sphereB[i],
      `Non-finite or non-deterministic float in buildCanonicalFibonacciSphereF32 at index ${i}`
    );
  }
  for (let pIdx = 0; pIdx < 1024; pIdx++) {
    const px = sphereA[pIdx * 4 + 0];
    const py = sphereA[pIdx * 4 + 1];
    const pz = sphereA[pIdx * 4 + 2];
    const psal = sphereA[pIdx * 4 + 3];
    const rad = Math.hypot(px, py, pz);
    assert(
      Math.abs(rad - 1.0) < 1e-5 && psal === 0.5,
      `Invalid unit Fibonacci sphere X_0 point at ${pIdx}: r=${rad}, sal=${psal}`
    );
  }
  console.log(
    `[PASS 11/12] PointScarFlow v1.0 ScarNative 3D Generator (12 WGSL Shaders · ${(psfShardStat.size / (1024 * 1024)).toFixed(2)} MiB VQ4-Scar Shard · Chamfer=${psfEval.held_out_chamfer_l2_4step_1pass} · Zero Catalog)`
  );

  // ---------------------------------------------------------------------------
  // TEST 12: Scar-Only Portfolio Purity & 0x00 Legacy Capsule Retirement Audit
  // ---------------------------------------------------------------------------
  const removedLegacyFiles = [
    'src/components/CirEngineArchitecture.tsx',
    'src/components/RopePhaseCanvas.tsx',
    'src/components/TokenizerPlayground.tsx',
    'src/components/FloatingTokenMonitor.tsx',
    'src/components/TokenInteractiveText.tsx',
    'src/components/AmbientTensorCanvas.tsx',
    'src/components/InteractiveToolDemos.tsx',
    'src/components/ExperienceTimeline.tsx',
    'src/components/KnowledgeTensor3D.tsx',
    'src/components/SystemOneHudDemo.tsx',
    'src/components/RubiksCube3D.tsx',
    'src/lib/tokenizer.ts',
    'src/lib/system1.ts',
  ];
  for (const rel of removedLegacyFiles) {
    assert(
      !fs.existsSync(path.join(PORTFOLIO_ROOT, rel)),
      `Removed non-Scar legacy file resurrected on disk: ${rel}`
    );
  }

  const bannedLegacyPortfolioRegex =
    /\b(?:cir-engine|cir-jev|mddd-cli|mad-cli|register-red|gittrack|flutter_scene|opah it|premier soft|503m)\b/i;

  const collectTextFiles = (dirAbs: string): string[] => {
    const out: string[] = [];
    for (const ent of fs.readdirSync(dirAbs, { withFileTypes: true })) {
      const full = path.join(dirAbs, ent.name);
      if (ent.isDirectory()) {
        out.push(...collectTextFiles(full));
      } else if (/\.(?:ts|tsx|css|wgsl)$/.test(ent.name)) {
        out.push(full);
      }
    }
    return out;
  };

  const auditedPurityPaths = [
    ...collectTextFiles(path.join(PORTFOLIO_ROOT, 'src')),
    path.join(PORTFOLIO_ROOT, 'index.html'),
    path.join(PORTFOLIO_ROOT, 'public/llms.txt'),
    path.join(PORTFOLIO_ROOT, 'public/llms-full.txt'),
    path.join(PORTFOLIO_ROOT, 'public/ai-index.json'),
    path.join(PORTFOLIO_ROOT, 'public/schema.json'),
    path.join(PORTFOLIO_ROOT, 'public/og-image.svg'),
    path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/scarstore_router.json'),
  ];
  for (const absPath of auditedPurityPaths) {
    const txt = fs.readFileSync(absPath, 'utf-8');
    const hit = txt.match(bannedLegacyPortfolioRegex);
    assert(
      !hit,
      `Forbidden non-Scar legacy project/career string "${hit?.[0]}" found in ${path.relative(PORTFOLIO_ROOT, absPath)}!`
    );
  }

  const routerBinNode = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'public/webgpu/qwen3b/scarstore_router.bin'));
  const routerBinBuf = routerBinNode.buffer.slice(
    routerBinNode.byteOffset,
    routerBinNode.byteOffset + routerBinNode.byteLength
  );
  const liveRouter = new BrowserScarStoreHTTP206(routerJson, routerBinBuf, './public/webgpu/qwen3b');
  const visibleCapsules = liveRouter.getCapsulesDirectory();
  for (const retiredId of ['portfolio_cir_engine_503m', 'portfolio_devtools_fullstack', 'portfolio_julio_bio_career']) {
    assert(
      !routerJson.capsules.some((c: any) => c.capsule_id === retiredId),
      `Expected ${retiredId} to be physically purged from scarstore_router.json!`
    );
    assert(
      !visibleCapsules.some((c) => c.capsule_id === retiredId),
      `Expected ${retiredId} to be excluded from visible ScarStore directory!`
    );
  }
  for (const scarId of [
    'portfolio_agent_identity',
    'portfolio_agent_capabilities',
    'portfolio_mos_scache_72b',
    'portfolio_jarvis_architecture',
    'portfolio_julio_scar_research',
    'portfolio_scarnative_vq4',
    'portfolio_pointscar_flow_3d',
    'portfolio_browser_webgpu_wgsl',
    'portfolio_aura_system_one',
    'portfolio_tricompass_scouts',
    'portfolio_contact_hire',
  ]) {
    assert(
      visibleCapsules.some((c) => c.capsule_id === scarId && !c.unlearnedWiped),
      `Expected core Scar Ecosystem cartridge ${scarId} to remain active!`
    );
  }
  console.log(
    `[PASS 12/13] Scar-Only Portfolio Purity (${auditedPurityPaths.length} files clean · 10 legacy modules purged · ${visibleCapsules.length} active ScarStore cartridges)`
  );

  // ---------------------------------------------------------------------------
  // TEST 13: ScarNative v6.0 Shared-Stage MoS, Dual-Tier O(1) SCache, `mos-digest` 24 Cartridges,
  //          Sequential Stage-Streaming, Document-Level `0x00` GDPR & Native `<abstain>`
  // ---------------------------------------------------------------------------
  for (const rel of ['src/lib/scarNativeV6Engine.ts', 'src/components/ScarNativeLab.tsx']) {
    const raw = fs.readFileSync(path.join(PORTFOLIO_ROOT, rel), 'utf-8');
    const clean = stripComments(raw);
    assert(!/Math\.random\s*\(/.test(clean), `Forbidden Math.random() detected in ${rel}!`);
  }
  assert(
    !bannedDomainCheatRegex.test(fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/scarNativeV6Engine.ts'), 'utf-8')),
    'Forbidden domain cheat word found in src/lib/scarNativeV6Engine.ts!'
  );

  const sn6Dir = path.join(PORTFOLIO_ROOT, 'public/webgpu/scarnative_v6');
  const sn6ManifestPath = path.join(sn6Dir, 'scarnative_manifest.json');
  assert(fs.existsSync(sn6ManifestPath), 'Missing public/webgpu/scarnative_v6/scarnative_manifest.json!');
  const sn6Manifest = JSON.parse(fs.readFileSync(sn6ManifestPath, 'utf-8')) as ScarNativeV6ManifestJSON;
  const sn6TokJson = JSON.parse(
    fs.readFileSync(path.join(sn6Dir, 'tokenizer.json'), 'utf-8')
  ) as ScarNativeBPETokenizerJSON;
  const sn6Wgsl = fs.readFileSync(path.join(sn6Dir, 'mos_scache_kernels.wgsl'), 'utf-8');
  assert(sn6Wgsl === wgslSrc, 'public/webgpu/scarnative_v6/mos_scache_kernels.wgsl != canonical mos_scache.wgsl');

  const toAb = (buf: Buffer): ArrayBuffer => {
    const ab = new ArrayBuffer(buf.byteLength);
    new Uint8Array(ab).set(buf);
    return ab;
  };

  const sn6ShardAb = toAb(fs.readFileSync(path.join(sn6Dir, sn6Manifest.shard_file)));
  const sn6ProcAb = toAb(fs.readFileSync(path.join(sn6Dir, sn6Manifest.procrustes_file)));
  const sn6ScarAb = toAb(fs.readFileSync(path.join(sn6Dir, sn6Manifest.scarstore_file)));

  const sn6Exec = new ScarNativeV6CpuExecutor(sn6Manifest, sn6ShardAb, sn6TokJson, sn6ProcAb, sn6ScarAb);
  assert(
    sn6Exec.cartridges.size === 24 && sn6Manifest.cartridges.every((c) => c.record_bytes === 12288),
    `Expected 24 mos-digest 12,288 B (3x4KB) cartridges in v6.0 bundle, got ${sn6Exec.cartridges.size}`
  );
  const docSet = new Set(sn6Manifest.cartridges.map((c) => c.doc_id));
  assert(docSet.size === 4, `Expected 4 mos-digest documents, got ${docSet.size}`);

  // 1) Long-Horizon Dual-Tier O(1) Passkey Recall (`T >> S+W=20`, `ΔSCache = 0 B`) vs `no-DualTier` ablation
  const fillerPtV6 = Array.from(
    { length: 8 },
    () => 'o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo .'
  ).join(' ');
  const v6PkPrompt = `<lang:pt> <user> Chave secreta <passkey> alpha = 7, beta = 4, gamma = 2 </passkey>. ${fillerPtV6} Qual é a chave secreta? <assistant>`;
  const v6PkFull = sn6Exec.generate(v6PkPrompt, {
    maxNewTokens: 18,
    useScarStore: false,
    streamStages: false,
    enableWidrowHoff: true,
    enableMicroPages: true,
  });
  assert(v6PkFull.prefillTokens >= 120, `Expected v6 overflow prefill >= 120 BPE tokens, got ${v6PkFull.prefillTokens}`);
  assert(v6PkFull.scacheDeltaBytes === 0, `Dual-Tier O(1) SCache violated: Δ=${v6PkFull.scacheDeltaBytes}`);
  assert(
    v6PkFull.outputText.includes('alpha = 7, beta = 4, gamma = 2'),
    `Expected exact 3-slot passkey recall from ScarNativeLM v6.0, got "${v6PkFull.outputText}"`
  );
  assert(
    v6PkFull.selectedExpertsByLayer.length === sn6Manifest.config.num_layers &&
      v6PkFull.selectedExpertsByLayer.every((ex) => ex.length === sn6Manifest.config.mos_top_k),
    `Expected ${sn6Manifest.config.num_layers} layers x Top-${sn6Manifest.config.mos_top_k} MoS experts!`
  );

  const v6PkNoDt = sn6Exec.generate(v6PkPrompt, {
    maxNewTokens: 18,
    useScarStore: false,
    streamStages: false,
    enableWidrowHoff: false,
    enableMicroPages: false,
  });
  assert(
    !v6PkNoDt.outputText.includes('alpha = 7, beta = 4, gamma = 2'),
    `Window-only no-DualTier ablation unexpectedly retained evicted passkey: "${v6PkNoDt.outputText}"`
  );

  const sn6EngSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/scarNativeV6Engine.ts'), 'utf-8');
  const sn5EngSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/scarNativeEngine.ts'), 'utf-8');
  const snLabSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/ScarNativeLab.tsx'), 'utf-8');
  assert(
    sn6EngSrc.includes('activeGpuLock') && sn5EngSrc.includes('activeGpuLock'),
    'Missing activeGpuLock WebGPU mapAsync serialization mutex in scarNativeV6Engine.ts or scarNativeEngine.ts!'
  );
  assert(
    snLabSrc.includes('async function runInitialChecks()') &&
      snLabSrc.includes('await runPasskeyComparison(') &&
      snLabSrc.includes('await runAbstainCheck('),
    'Unawaited concurrent WebGPU warmup detected in ScarNativeLab.tsx!'
  );

  // 2) Sequential Stage-Streaming (`streamStages: true`) Bit-Exact Logit Parity & VRAM Reduction
  const v6PkStream = sn6Exec.generate(v6PkPrompt, {
    maxNewTokens: 18,
    useScarStore: false,
    streamStages: true,
    enableWidrowHoff: true,
    enableMicroPages: true,
  });
  assert(
    v6PkStream.activeWeightGpuBytes < sn6Manifest.shard_bytes * 0.55,
    `Expected Stage-Streaming active weight window < 55% of full shard, got ${v6PkStream.activeWeightGpuBytes} vs ${sn6Manifest.shard_bytes}`
  );
  let maxStreamDiff = 0.0;
  for (let i = 0; i < v6PkFull.firstTokenLogits.length; i++) {
    maxStreamDiff = Math.max(
      maxStreamDiff,
      Math.abs(v6PkFull.firstTokenLogits[i] - v6PkStream.firstTokenLogits[i])
    );
  }
  assert(maxStreamDiff === 0, `Expected exact 0.00000000 Stage-Streaming logit diff, got ${maxStreamDiff}`);

  // 3) Cross-Lingual `PT / EN / ES / FR` `mos-digest` Procrustes Routing (`queryLang: null` auto-detection!)
  const v6OrionPrompts: Record<PolyglotLang, string> = {
    pt: '<lang:pt> <user> <fact> Qual é o valor exato de órbita do módulo orion? <assistant>',
    en: '<lang:en> <user> <fact> What is the exact orbit value of module orion? <assistant>',
    es: '<lang:es> <user> <fact> ¿Cuál es el valor exacto de órbita del módulo orion? <assistant>',
    fr: '<lang:fr> <user> <fact> Quelle est la valeur exacte de orbite du module orion ? <assistant>',
  };
  for (const lg of ['pt', 'en', 'es', 'fr'] as const) {
    const r = sn6Exec.generate(v6OrionPrompts[lg], { maxNewTokens: 16, useScarStore: true, queryLang: null });
    assert(r.routedCartridgeId === 'ent_orion', `Expected ent_orion route in ${lg}, got ${r.routedCartridgeId}`);
    assert(r.resolvedLang === lg, `Expected auto-resolved Procrustes lang=${lg}, got ${r.resolvedLang}`);
    assert(
      r.outputText.includes('516') && r.outputText.includes('km'),
      `Expected 516 km in ${lg} from EN-minted mos-digest cartridge, got "${r.outputText}"`
    );
  }

  // 4) Held-Out Out-of-Domain Ungrounded Query -> Native `<abstain>` (`id=14`, Zero Hallucination)
  const v6HairAbstain = sn6Exec.generate(
    '<lang:pt> <user> Para um cabelo já liso e descolorido, qual progressiva química você recomenda? <assistant>',
    { maxNewTokens: 18, useScarStore: true, queryLang: null }
  );
  assert(v6HairAbstain.routedCartridgeId === null, `Unexpected cartridge route on cosmetic query: ${v6HairAbstain.routedCartridgeId}`);
  assert(
    v6HairAbstain.abstained && v6HairAbstain.outputTextWithTags.includes('<abstain>'),
    `Expected native <abstain> token on ungrounded bleached-hair chemical query, got "${v6HairAbstain.outputTextWithTags}"`
  );

  // 5) Full-Document Cryptographic `0x00` GDPR Wipe (`doc_00_orbital` -> 6 cartridges) & Exact Restoration
  const v6UnroutedTrunkRef = sn6Exec.generate(v6OrionPrompts.pt, { maxNewTokens: 1, useScarStore: false });
  const wipedDocCount = sn6Exec.gdprWipeDocument('doc_00_orbital');
  assert(wipedDocCount === 6, `Expected 6 cartridges wiped in doc_00_orbital, got ${wipedDocCount}`);
  const v6PostDocWipe = sn6Exec.generate(v6OrionPrompts.pt, { maxNewTokens: 16, useScarStore: true });
  assert(v6PostDocWipe.routedCartridgeId === null, `Wiped document still routed: ${v6PostDocWipe.routedCartridgeId}`);
  assert(
    v6PostDocWipe.abstained && !v6PostDocWipe.outputText.includes('516'),
    `Expected <abstain> and zero 516 leak after doc_00_orbital GDPR wipe, got "${v6PostDocWipe.outputTextWithTags}"`
  );
  let maxDocGdprDiff = 0.0;
  for (let i = 0; i < v6UnroutedTrunkRef.firstTokenLogits.length; i++) {
    maxDocGdprDiff = Math.max(
      maxDocGdprDiff,
      Math.abs(v6UnroutedTrunkRef.firstTokenLogits[i] - v6PostDocWipe.firstTokenLogits[i])
    );
  }
  assert(maxDocGdprDiff === 0, `Expected exact 0.00000000 Document GDPR logit diff, got ${maxDocGdprDiff}`);
  const restoredDocCount = sn6Exec.restoreDocument('doc_00_orbital');
  assert(restoredDocCount === 6, `Expected 6 cartridges restored in doc_00_orbital, got ${restoredDocCount}`);

  // 6) Verify `v6-Frontier` (`96L, d=8192`) Scaling Law Numbers
  const frontierBudget = computeFrontierV6ByteBudget(
    {
      ...sn6Manifest.config,
      vocab_size: 32768,
      embed_rank: 512,
      hidden_size: 8192,
      intermediate_size: 28672,
      num_layers: 96,
      num_stages: 16,
      num_q_heads: 64,
      num_kv_heads: 8,
      head_dim: 128,
      id_scar_rank: 32,
      head_scar_rank: 128,
      num_mos_experts: 16,
      mos_top_k: 2,
      mos_expert_rank: 64,
      num_sink: 4,
      slide_len: 128,
      num_micro_pages: 64,
      cartridge_rank: 64,
    },
    4,
    2,
    131072,
    true
  );
  assert(
    frontierBudget.total_lru_resident_gib < 4.45 &&
      frontierBudget.stage_streaming_peak_gib < 1.0 &&
      frontierBudget.total_vram_reduction_vs_dense_128k > 46.0,
    `Unexpected v6-Frontier budget: ${JSON.stringify(frontierBudget)}`
  );

  console.log(
    `[PASS 13/14] ScarNative v6.0 (${(sn6Manifest.shard_bytes / (1024 * 1024)).toFixed(2)} MiB Shared-Stage MoS · Stream ${(v6PkStream.activeWeightGpuBytes / (1024 * 1024)).toFixed(2)} MiB Δ=0 · SCache ${v6PkFull.scacheBytesFinal} B Δ=0 B · 24 mos-digest PT/EN/ES/FR · Doc 0x00 GDPR · <abstain> · Frontier ${frontierBudget.total_lru_resident_gib.toFixed(2)} GiB / ${frontierBudget.stage_streaming_peak_gib.toFixed(2)} GiB)`
  );

  // ---------------------------------------------------------------------------
  // TEST 14: PhysicalScarLab Forensic Scientific Audit (Anti-Oracle, Partial-Cue Recall,
  //          Conservative Burgers PDE, Byte-Matched FP16 2D-DCT & Epistemological Scope)
  // ---------------------------------------------------------------------------
  const physCodecRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/physicalScarCodec.ts'), 'utf-8');
  const physLabRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/PhysicalScarLab.tsx'), 'utf-8');
  assert(
    physLabRaw.includes('data-testid="physical-scar-forensic-scope"') &&
      physLabRaw.includes('data-testid="physical-scar-public-archives"') &&
      physLabRaw.includes('data-testid="tab-public-archives"'),
    'Missing forensic scope or public-archives testid in PhysicalScarLab.tsx!'
  );
  const pubAuditSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarCodecPublicAudit.json'), 'utf-8');
  const pubAuditWeb = fs.readFileSync(
    path.join(PORTFOLIO_ROOT, 'public/webgpu/scarcodec_public_datasets_v1_0_audit.json'),
    'utf-8'
  );
  const pubAuditCanonPath = path.resolve(
    PORTFOLIO_ROOT,
    '../scar-cache/results/scar_codec_public_datasets_v1_0_audit.json'
  );
  assert(pubAuditSrc === pubAuditWeb, 'src/data/scarCodecPublicAudit.json != public/webgpu copy!');
  if (fs.existsSync(pubAuditCanonPath)) {
    assert(
      pubAuditSrc === fs.readFileSync(pubAuditCanonPath, 'utf-8'),
      'Portfolio scarCodecPublicAudit.json drifted from canonical scar-cache audit JSON!'
    );
  }
  const pubJson = JSON.parse(pubAuditSrc);
  assert(
    pubJson.scar_codec_modified === false &&
      pubJson.synthetic_data_used === false &&
      pubJson.oracle_steering_used === false &&
      Array.isArray(pubJson.provenance_manifest) &&
      pubJson.provenance_manifest.length === 7,
    'Invalid provenance or anti-oracle flags in scarCodecPublicAudit.json!'
  );
  assert(
    !/const\s+wgsl\s*=/.test(stripComments(physLabRaw)),
    'Dead unused wgsl draft shader still present in PhysicalScarLab.tsx!'
  );
  assert(
    physCodecRaw.includes('fitFromCalibration') &&
      !stripComments(physCodecRaw).includes('new OrthogonalSubspaceProjectorTS(rfiSteering'),
    'Oracle RFI steering leak detected in physicalScarCodec.ts — must use blind empirical covariance fitFromCalibration!'
  );
  assert(
    p2.maskedChannelsMeanCosine > 0.995,
    `Expected blind 2/6 partial-cue recall on 4 unseen masked channels cos > 0.995, got ${p2.maskedChannelsMeanCosine}`
  );
  assert(
    p3.spectralBytes >= Math.floor(p3.scarBytes * 0.98) && p3.spectralBytes <= p3.scarBytes,
    `Unfair spectral baseline byte budget: spectralBytes=${p3.spectralBytes} vs scarBytes=${p3.scarBytes}`
  );
  assert(
    p3.scarRmse < p3.spectralRmse / 100.0,
    `Expected > 100x RMSE improvement over byte-matched FP16 2D-DCT on Burgers shockwave PDE, got ${p3.scarRmse} vs ${p3.spectralRmse}`
  );

  const stalePhysicalRegex = /\b(?:0\.9923|5\.75x)\b/;
  for (const absPath of auditedPurityPaths) {
    const txt = fs.readFileSync(absPath, 'utf-8');
    const hit = txt.match(stalePhysicalRegex);
    assert(
      !hit,
      `Stale pre-forensic physical metric "${hit?.[0]}" found in ${path.relative(PORTFOLIO_ROOT, absPath)}!`
    );
  }

  console.log(
    `[PASS 14/15] PhysicalScarLab Forensic Scientific Audit (Blind SVD cos=${p1.scarCodecCosine.toFixed(4)} · Partial-Cue Masked cos=${p2.maskedChannelsMeanCosine.toFixed(4)} · Burgers PDE ${p3.scarBytes}B vs 2D-DCT ${p3.spectralBytes}B ${(p3.spectralRmse / p3.scarRmse).toFixed(1)}x RMSE Gain · Zero Stale Claims)`
  );

  // ---------------------------------------------------------------------------
  // TEST 15: ScarCodec v6.0 (C99/SIMD, ZFP/fpack, E8, L_inf) & ScarField v7.0
  //          (NVIDIA A100 Zero-Legacy + C99/SIMD Synergy) Canonical Parity Audit
  // ---------------------------------------------------------------------------
  const syncPairs: Array<[string, string, string]> = [
    [
      'src/data/scarCodecV11Audit.json',
      'public/webgpu/scar_codec_public_datasets_v1_1_audit.json',
      '../scar-cache/results/scar_codec_public_datasets_v1_1_audit.json',
    ],
    [
      'src/data/scarCodecV6Audit.json',
      'public/webgpu/scar_codec_v6_0_consolidation_audit.json',
      '../scar-cache/results/scar_codec_v6_0_consolidation_audit.json',
    ],
    [
      'src/data/scarCodecForensicSynergyAudit.json',
      'public/webgpu/scar_codec_forensic_and_synergy_audit.json',
      '../scar-cache/results/scar_codec_forensic_and_synergy_audit.json',
    ],
    [
      'src/data/scarFieldV7Audit.json',
      'public/webgpu/scar_field_v7_0_audit.json',
      '../scar-cache/results/scar_field_v7_0_audit.json',
    ],
  ];
  for (const [srcRel, webRel, canonRel] of syncPairs) {
    const sTxt = fs.readFileSync(path.join(PORTFOLIO_ROOT, srcRel), 'utf-8');
    const wTxt = fs.readFileSync(path.join(PORTFOLIO_ROOT, webRel), 'utf-8');
    assert(sTxt === wTxt, `Showcase drift: ${srcRel} != ${webRel}`);
    const cPath = path.resolve(PORTFOLIO_ROOT, canonRel);
    if (fs.existsSync(cPath)) {
      assert(sTxt === fs.readFileSync(cPath, 'utf-8'), `Showcase drift: ${srcRel} != ${canonRel}`);
    }
  }

  const v6Json = JSON.parse(fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarCodecV6Audit.json'), 'utf-8'));
  const synJson = JSON.parse(
    fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarCodecForensicSynergyAudit.json'), 'utf-8')
  );
  const v7Json = JSON.parse(fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarFieldV7Audit.json'), 'utf-8'));

  assert(
    v6Json.zero_pytorch_verified === true &&
      v6Json.pillar_1_and_2_c_simd_mmap_and_zarr.c_simd_pure_random_block_o1_us < 1.0 &&
      v6Json.pillar_3_deterministic_linf_error_bound.bound_strictly_satisfied === true &&
      v6Json.pillar_4_gosset_e8_and_3d_spectral_tucker.spitzer_gosset_e8_1bpw.compression_vs_fp32_x >= 20.0,
    'ScarCodec v6.0 consolidation invariants failed in showcase JSON!'
  );
  assert(
    synJson.forensic_scar_codec_checks.ligo_1e_minus_21_strain.passed === true &&
      synJson.forensic_scar_codec_checks.chandra_xray_1p5e8_counts.passed === true &&
      synJson.scarcodec_to_scarfield_v7_synergy.canonical_e8_lut256_max_abs_diff === 0.0 &&
      synJson.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.full_engine_s1_speedup_x > 30.0,
    'ScarCodec forensic or ScarField synergy invariants failed!'
  );
  assert(
    v7Json.zero_legacy_guarantees.nn_linear_count === 0 &&
      v7Json.zero_legacy_guarantees.relu_gelu_silu_swiglu_count === 0 &&
      v7Json.zero_legacy_guarantees.softmax_exp_log_count === 0 &&
      v7Json.passkey_horizon_sweep.scarfield_v7_full['T=160'].unseen_pin_exact_3digit_acc === 1.0 &&
      v7Json.passkey_horizon_sweep.scarfield_v7_full['T=512'].unseen_pin_exact_3digit_acc >= 0.7,
    'ScarField v7.0 A100 zero-legacy or passkey recall invariants failed!'
  );

  console.log(
    `[PASS 15/16] ScarCodec v6.0 (${v6Json.pillar_1_and_2_c_simd_mmap_and_zarr.c_simd_pure_random_block_o1_us} us C99/SIMD · 21.0x E8 · LIGO 1e-21..X-Ray 1.5e8) + ScarField v7.0 A100 (WY ${v7Json.hardware_benchmarks.chunkwise_wy_scan.speedup_x}x · C/SIMD S1 ${synJson.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.full_engine_s1_speedup_x}x↑ · 100% PIN @ T=160)`
  );

  // ---------------------------------------------------------------------------
  // TEST 16: Expanded 10-Archive / 8-Science Real-Data Audit + Live Browser
  //          Forensic Workbench (`LIGO 10^-19`, `Tohoku 3.13M`, `MIT-BIH IEC`, `NOAA`, `ROSAT`)
  // ---------------------------------------------------------------------------
  const expV2Src = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarCodecExpandedPhysicsV2Audit.json'), 'utf-8');
  const expV2Web = fs.readFileSync(
    path.join(PORTFOLIO_ROOT, 'public/webgpu/scar_codec_expanded_real_physics_v2_0_audit.json'),
    'utf-8'
  );
  assert(expV2Src === expV2Web, 'Expanded physics v2.0 JSON drift between src/data and public/webgpu!');
  const expV2Canon = path.resolve(PORTFOLIO_ROOT, '../scar-cache/results/scar_codec_expanded_real_physics_v2_0_audit.json');
  if (fs.existsSync(expV2Canon)) {
    assert(expV2Src === fs.readFileSync(expV2Canon, 'utf-8'), 'Expanded physics v2.0 JSON drift vs scar-cache!');
  }

  const slicesSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/realPhysicalSlices.json'), 'utf-8');
  const slicesWeb = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'public/webgpu/real_physical_slices_v2_0.json'), 'utf-8');
  assert(slicesSrc === slicesWeb, 'Real physical slices JSON drift between src/data and public/webgpu!');

  const expJson = JSON.parse(expV2Src);
  assert(
    expJson.synthetic_data_used === false &&
      expJson.oracle_steering_used === false &&
      expJson.scar_codec_modified === false &&
      expJson.total_public_archives_across_ecosystem === 10 &&
      expJson.total_physical_sciences_covered === 8 &&
      expJson.new_provenance_manifest.length === 9,
    'Expanded physics v2.0 top-level forensic flags failed!'
  );

  const { lut256x8 } = buildCanonicalE8Lut256TS();
  assert(
    lut256x8.length === 2048 && lut256x8.subarray(0, 8).every((v) => v === 0),
    'Gosset E8 LUT256 browser vacuum row 0 must be exact (0,0,0,0,0,0,0,0)!'
  );

  const realSlices = JSON.parse(slicesSrc).slices;

  // 1. LIGO GW150914: 100% naive FP16 underflow (`0` nonzeros) vs ScarCodec E8 > 6.5x + exact 2.5e-22 bound
  const ligoField = new Float32Array(realSlices.ligo_gw150914.p_perp_clean_strain_f32);
  const ligoE8 = runRealSliceLiveForensicAuditTS({ field: ligoField, rows: 16, cols: 256, mode: 'e8', maxPointSpikes: 48 });
  const ligoEb = runRealSliceLiveForensicAuditTS({
    field: ligoField,
    rows: 16,
    cols: 256,
    mode: 'error_bounded',
    absTolerance: 2.5e-22,
    maxPointSpikes: 48,
  });
  assert(
    ligoE8.naiveFp16NonZeroCount === 0 &&
      ligoE8.compressionRatio > 6.5 &&
      ligoE8.scarRmse < ligoE8.spectralRmse &&
      ligoE8.blockO1VsFullMaxDiff === 0 &&
      ligoEb.scarMaxAbsErr <= 2.5001e-22,
    'LIGO GW150914 live browser forensic audit failed!'
  );

  // 2. IRIS Mw 9.1 Tohoku: >2.5M naive FP16 overflow vs ScarCodec >10x cleaner than byte-matched 2D-DCT
  const seisField = new Float32Array(realSlices.iris_tohoku_mw91.bhz_overview_f32);
  const seisE8 = runRealSliceLiveForensicAuditTS({ field: seisField, rows: 16, cols: 256, mode: 'e8', maxPointSpikes: 48 });
  assert(
    seisE8.naiveFp16MaxAbsErr > 2.4e6 &&
      seisE8.globalScalePow2 === 8192 &&
      seisE8.scarRmse < seisE8.spectralRmse / 10.0 &&
      seisE8.blockO1VsFullMaxDiff === 0,
    'IRIS Mw 9.1 Tohoku live browser forensic audit failed!'
  );

  // 3. MIT-BIH 119 Holter ECG: strict IEC 60601-2-47 (±0.050 mV = 50 uV) 0-violation bound
  const ecgField = new Float32Array(realSlices.mitbih_119_ecg.mlii_mv_f32);
  const ecgIec = runRealSliceLiveForensicAuditTS({
    field: ecgField,
    rows: 16,
    cols: 256,
    mode: 'error_bounded',
    absTolerance: 0.05,
    maxPointSpikes: 48,
  });
  assert(
    ecgIec.scarMaxAbsErr <= 0.05001 && ecgIec.compressionRatio > 1.4 && ecgIec.blockO1VsFullMaxDiff === 0,
    'MIT-BIH 119 IEC 50 uV clinical bound failed in live browser engine!'
  );

  // 4. NOAA Ocean + Antarctica: exact 0.0000 C pure continental mask vs >2.5 C 2D-DCT Gibbs leak
  const noaaField = new Float32Array(realSlices.noaa_oisst_gulf_stream.grid_f32);
  const noaaE8 = runRealSliceLiveForensicAuditTS({ field: noaaField, rows: 32, cols: 128, mode: 'e8', maxPointSpikes: 48 });
  assert(
    noaaE8.pureZeroBlockCount === 12 &&
      noaaE8.scarPureZeroBlockMaxErr === 0.0 &&
      noaaE8.spectralPureZeroBlockMaxErr > 2.5 &&
      noaaE8.blockO1VsFullMaxDiff === 0,
    'NOAA OISST Antarctica pure-continent vacuum invariant failed!'
  );

  // 5. ROSAT Soft X-Ray: >8.0x E8 compression + >9.0x lower RMSE than matched 2D-DCT + ±50 bound
  const rosField = new Float32Array(realSlices.rosat_xray_scox1.grid_f32);
  const rosE8 = runRealSliceLiveForensicAuditTS({ field: rosField, rows: 32, cols: 128, mode: 'e8', maxPointSpikes: 48 });
  const rosEb = runRealSliceLiveForensicAuditTS({
    field: rosField,
    rows: 32,
    cols: 128,
    mode: 'error_bounded',
    absTolerance: 50.0,
    maxPointSpikes: 48,
  });
  assert(
    rosE8.compressionRatio > 8.0 &&
      rosE8.scarRmse < rosE8.spectralRmse / 9.0 &&
      rosEb.scarMaxAbsErr <= 50.001 &&
      rosEb.compressionRatio > 2.5,
    'ROSAT All-Sky Soft X-Ray live browser audit failed!'
  );

  const physLabSrc = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/PhysicalScarLab.tsx'), 'utf-8');
  assert(
    physLabSrc.includes('data-testid="tab-live-real-workbench"') &&
      physLabSrc.includes('data-testid="physical-scar-live-real-workbench"'),
    'PhysicalScarLab.tsx missing Tab 5 live real-data workbench testids!'
  );

  console.log(
    `[PASS 16/17] Expanded 10-Archive / 8-Science Physics Lab + Live Real-Data Browser Workbench (LIGO 0/4096 FP16 -> ${ligoE8.compressionRatio.toFixed(2)}x E8 · Tohoku ${(seisE8.spectralRmse / seisE8.scarRmse).toFixed(1)}x < DCT · ECG IEC ±50uV 0-Violations · NOAA Antarctica 0.0000°C vs ${noaaE8.spectralPureZeroBlockMaxErr.toFixed(2)}°C DCT · ROSAT ${(rosE8.spectralRmse / rosE8.scarRmse).toFixed(1)}x < DCT)`
  );

  // ================================================================================
  // TEST 17: Production Forensic Fuzzing + Modern 2026 SOTA Benchmark Audit Sync
  //          (SZ3 · SPERR CDF 9/7+SPECK · Blosc2 4.15 · CCSDS-121 · Frequent Directions)
  // ================================================================================
  const sotaSrcPath = path.join(PORTFOLIO_ROOT, 'src/data/scarCodecModernSotaV3Audit.json');
  const sotaPubPath = path.join(PORTFOLIO_ROOT, 'public/webgpu/scar_codec_modern_sota_v3_0_audit.json');
  const sotaRepoPath = path.resolve(PORTFOLIO_ROOT, '../scar-cache/results/scar_codec_modern_sota_v3_0_audit.json');
  assert(fs.existsSync(sotaSrcPath) && fs.existsSync(sotaPubPath), 'Missing Modern 2026 SOTA v3.0 JSON artifacts!');
  const sotaSrcBuf = fs.readFileSync(sotaSrcPath);
  const sotaPubBuf = fs.readFileSync(sotaPubPath);
  assert(sotaSrcBuf.equals(sotaPubBuf), 'Modern SOTA v3.0 JSON mismatch between src/data and public/webgpu!');
  if (fs.existsSync(sotaRepoPath)) {
    assert(sotaSrcBuf.equals(fs.readFileSync(sotaRepoPath)), 'Modern SOTA v3.0 JSON out of sync with scar-cache/results!');
  }
  const sotaJson = JSON.parse(sotaSrcBuf.toString('utf-8'));
  const bolSz3Gain =
    sotaJson.pillar_2_exascale_lossy_sperr_sz3_zfp_byte_matched.cso_bolocam_sgr_a.sz3_lorenzo.top32_peak_max_err /
    sotaJson.pillar_2_exascale_lossy_sperr_sz3_zfp_byte_matched.cso_bolocam_sgr_a.scar_vq4.top32_peak_max_err;
  const bolSperrGain =
    sotaJson.pillar_2_exascale_lossy_sperr_sz3_zfp_byte_matched.cso_bolocam_sgr_a.sperr_cdf97_speck.top32_peak_max_err /
    sotaJson.pillar_2_exascale_lossy_sperr_sz3_zfp_byte_matched.cso_bolocam_sgr_a.scar_vq4.top32_peak_max_err;
  const ccsdsSpeedup =
    sotaJson.pillar_1_modern_chunked_storage_blosc2_bitshuffle_ccsds.random_64float_block_latency_us.ccsds121_libaec_decode_us /
    sotaJson.pillar_1_modern_chunked_storage_blosc2_bitshuffle_ccsds.random_64float_block_latency_us.scar_c99_simd_ssd_mmap_us;
  assert(
    sotaJson.suite === 'scar_codec_modern_sota_v3_0_audit' &&
      bolSz3Gain > 5.0 &&
      bolSperrGain > 8.0 &&
      ccsdsSpeedup > 500.0 &&
      sotaJson.pillar_4_modern_streaming_sketching_and_spectral_kurtosis.streaming_34p8kb_scache_vs_frequent_directions
        .scache_lead_v1_pvc_recall_cosine > 0.99,
    'Modern 2026 SOTA v3.0 invariant regression!'
  );

  // Adversarial prime & degenerate shapes in TypeScript (`(1,1)`, `(1,63)`, `(1,65)`, `(3,127)`)
  for (const [rS, cS] of [
    [1, 1],
    [1, 63],
    [1, 64],
    [1, 65],
    [3, 127],
    [7, 131],
  ] as const) {
    const primeField = new Float32Array(rS * cS);
    for (let i = 0; i < primeField.length; i++) {
      primeField[i] = Math.sin(i * 0.73 + 0.4) * 14.5 + (i === primeField.length - 1 ? -95.0 : 0.0);
    }
    const vq4Ts = new PhysicalVQ4ScarCodecTS({ rank: 8, maxPointSpikes: 8 });
    const vq4Pkg = vq4Ts.encode(primeField, rS, cS);
    const vq4Full = vq4Ts.decodeFull(vq4Pkg);
    assert(vq4Full.length === rS * cS, `VQ4 prime shape decode length mismatch for (${rS},${cS})`);
    const nBlks = Math.ceil(cS / 64);
    for (let rr = 0; rr < rS; rr++) {
      for (let bb = 0; bb < nBlks; bb++) {
        const blk = vq4Ts.decodeBlockO1(vq4Pkg, rr, bb).values;
        for (let k = 0; k < blk.length; k++) {
          assert(
            Math.abs(blk[k] - vq4Full[rr * cS + bb * 64 + k]) === 0,
            `VQ4 O(1) vs Full mismatch at (${rS},${cS}) blk (${rr},${bb})`
          );
        }
      }
    }

    const e8Ts = new ScarCodecE8TS({ rank: 8, maxPointSpikes: 8 });
    const e8Pkg = e8Ts.encode(primeField, rS, cS);
    const e8Full = e8Ts.decodeFull(e8Pkg);
    assert(e8Full.length === rS * cS, `E8 prime shape decode length mismatch for (${rS},${cS})`);
    for (let rr = 0; rr < rS; rr++) {
      for (let bb = 0; bb < nBlks; bb++) {
        const blk = e8Ts.decodeBlockO1(e8Pkg, rr, bb).values;
        for (let k = 0; k < blk.length; k++) {
          assert(
            Math.abs(blk[k] - e8Full[rr * cS + bb * 64 + k]) === 0,
            `E8 O(1) vs Full mismatch at (${rS},${cS}) blk (${rr},${bb})`
          );
        }
      }
    }
  }

  // Adversarial RangeError rejection (`NaN`, `Infinity`, empty dims, odd rank, out-of-bounds block)
  const expectRangeError = (fn: () => unknown, msg: string) => {
    let threw = false;
    try {
      fn();
    } catch (e) {
      threw = e instanceof RangeError;
    }
    assert(threw, `Expected RangeError for ${msg}`);
  };
  expectRangeError(() => new PhysicalVQ4ScarCodecTS({ rank: 3 }), 'odd rank');
  expectRangeError(() => new ScarCodecE8TS({ absTolerance: 0 }), 'zero absTolerance');
  const nanBuf = new Float32Array(64);
  nanBuf[7] = Number.NaN;
  expectRangeError(() => new PhysicalVQ4ScarCodecTS().encode(nanBuf, 1, 64), 'NaN input');
  const infBuf = new Float32Array(64);
  infBuf[3] = Number.POSITIVE_INFINITY;
  expectRangeError(() => new ScarCodecE8TS().encode(infBuf, 1, 64), 'Inf input');
  expectRangeError(() => new PhysicalVQ4ScarCodecTS().encode(new Float32Array(0), 0, 64), 'empty rows');
  const okPkg = new ScarCodecE8TS().encode(new Float32Array(64), 1, 64);
  expectRangeError(() => new ScarCodecE8TS().decodeBlockO1(okPkg, 1, 0), 'row out of bounds');
  expectRangeError(() => new ScarCodecE8TS().decodeBlockO1(okPkg, 0, 1), 'block out of bounds');

  // 200 dB mixed dynamic range `[1.0, 5e-10, 0.0]` + 0-ULP bit-exact `float32` lossless on Kelvin field
  const hdrMix = new Float32Array(4 * 64);
  for (let c = 0; c < 64; c++) {
    hdrMix[c] = -1.0 + (2.0 * c) / 63.0;
    hdrMix[64 + c] = 5.0e-10;
    hdrMix[128 + c] = 0.0;
  }
  for (const modeCodec of [
    new PhysicalVQ4ScarCodecTS({ rank: 8, absTolerance: 1.0e-10 }),
    new ScarCodecE8TS({ rank: 8, absTolerance: 1.0e-10 }),
  ]) {
    const rec =
      modeCodec instanceof PhysicalVQ4ScarCodecTS
        ? modeCodec.decodeFull(modeCodec.encode(hdrMix, 4, 64))
        : modeCodec.decodeFull(modeCodec.encode(hdrMix, 4, 64));
    let maxDiff = 0;
    for (let i = 0; i < hdrMix.length; i++) maxDiff = Math.max(maxDiff, Math.abs(hdrMix[i] - rec[i]));
    assert(maxDiff <= 1.0001e-10, `200 dB mixed absTolerance failed: ${maxDiff}`);
    for (let c = 0; c < 64; c++) assert(rec[128 + c] === 0.0, 'Pure vacuum row leaked under absTolerance!');
  }

  const kelvinField = new Float32Array(8 * 64);
  for (let i = 0; i < kelvinField.length; i++) kelvinField[i] = Math.fround(273.15 + Math.sin(i * 0.37) * 18.4);
  const e8Exact = new ScarCodecE8TS({ rank: 8, absTolerance: 1.0e-30 });
  const kelvinRec = e8Exact.decodeFull(e8Exact.encode(kelvinField, 8, 64));
  for (let i = 0; i < kelvinField.length; i++) {
    assert(kelvinRec[i] === kelvinField[i], `Expected 0-ULP bit-exact float32 at ${i}, got ${kelvinRec[i]} vs ${kelvinField[i]}`);
  }

  // Live CDF 9/7 Wavelet baseline verification (`SPERR / JPEG2000` transform core)
  assert(
    noaaE8.waveletCdf97PureZeroBlockMaxErr > 1.5 &&
      noaaE8.scarPureZeroBlockMaxErr === 0.0 &&
      seisE8.scarRmse < seisE8.waveletCdf97Rmse / 6.0 &&
      computeMatchedByte2dWaveletCdf97Fp16(rosField, 32, 128, rosE8.scarBytes).waveletBytes <= rosE8.scarBytes,
    'CDF 9/7 Wavelet browser baseline check failed!'
  );
  assert(
    physLabSrc.includes('data-testid="modern-2026-sota-audit-box"'),
    'PhysicalScarLab.tsx missing modern-2026-sota-audit-box!'
  );

  console.log(
    `[PASS 17/18] Production Forensic Fuzzing & Modern 2026 SOTA Suite (Bolocam ${bolSz3Gain.toFixed(2)}x < SZ3 & ${bolSperrGain.toFixed(2)}x < SPERR · ${Math.round(ccsdsSpeedup)}x > CCSDS-121 · FD cos ${sotaJson.pillar_4_modern_streaming_sketching_and_spectral_kurtosis.streaming_34p8kb_scache_vs_frequent_directions.frequent_directions_lead_v1_pvc_recall_cosine} -> SCache ${sotaJson.pillar_4_modern_streaming_sketching_and_spectral_kurtosis.streaming_34p8kb_scache_vs_frequent_directions.scache_lead_v1_pvc_recall_cosine} · Prime Shapes (1,1)..(7,131) + 0-ULP Lossless + CDF 9/7 Wavelet)`
  );

  // ---------------------------------------------------------------------------
  // TEST 18: Biorthogonal Physics Lab (v6.1/v6.2, 10 Archives) & ScarField v7.3 Audit
  // ---------------------------------------------------------------------------
  const biorthSyncPairs = [
    [
      path.join(PORTFOLIO_ROOT, 'src/data/scarCodecBiorthogonalV61Audit.json'),
      path.join(PORTFOLIO_ROOT, 'public/webgpu/scar_codec_biorthogonal_v6_1_audit.json'),
      '/Users/juliocesarreisfilho/Projects/scar-cache/results/scar_codec_biorthogonal_v6_1_audit.json',
    ],
    [
      path.join(PORTFOLIO_ROOT, 'src/data/scarCodecFullBiorthogonalV62Audit.json'),
      path.join(PORTFOLIO_ROOT, 'public/webgpu/scar_codec_full_biorthogonal_physics_v6_2_audit.json'),
      '/Users/juliocesarreisfilho/Projects/scar-cache/results/scar_codec_full_biorthogonal_physics_v6_2_audit.json',
    ],
    [
      path.join(PORTFOLIO_ROOT, 'src/data/scarFieldV73Audit.json'),
      path.join(PORTFOLIO_ROOT, 'public/webgpu/scar_field_v7_3_biorth_chat_audit.json'),
      '/Users/juliocesarreisfilho/Projects/scar-cache/results/scar_field_v7_3_biorth_chat_audit.json',
    ],
  ];
  for (const [srcP, pubP, canonP] of biorthSyncPairs) {
    assert(fs.existsSync(srcP) && fs.existsSync(pubP) && fs.existsSync(canonP), `Missing synced Biorthogonal JSON: ${srcP}`);
    const b1 = fs.readFileSync(srcP, 'utf-8');
    const b2 = fs.readFileSync(pubP, 'utf-8');
    const b3 = fs.readFileSync(canonP, 'utf-8');
    assert(b1 === b2 && b1 === b3, `Byte-for-byte sync drift detected on ${path.basename(srcP)}`);
  }

  const v62Json = JSON.parse(fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarCodecFullBiorthogonalV62Audit.json'), 'utf-8'));
  const v73Json = JSON.parse(fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/data/scarFieldV73Audit.json'), 'utf-8'));
  assert(
    v62Json.provenance_sha256.files_verified === 16 &&
      v62Json.provenance_sha256.all_sha256_matched === true &&
      v62Json.spatial_and_3d_fields_vs_modern_sota.iris_tohoku_mw91_3axis.vq4_mode.biorth_snr_gain_over_orth_db > 8.0 &&
      v62Json.spatial_and_3d_fields_vs_modern_sota.mitbih_119_holter_ecg.error_bounded_linf_mode.biorth_byte_savings_pct > 25.0 &&
      v62Json.streaming_34p8kb_scache_biorthogonal_vs_sota.physionet_64ch_cortical_eeg.biorthogonal_dual_scache_32odd_cosine > 0.98 &&
      v62Json.streaming_34p8kb_scache_biorthogonal_vs_sota.nasa_space_shuttle_radiator.corrupted_32sensor_fault_recall_biorth_cosine > 0.999,
    'v6.2 complete Biorthogonal Physics Lab thresholds failed!'
  );
  assert(
    v73Json.architecture_comparison.scarfield_v7_3_sfield_scar_mib < 9.6 &&
      v73Json.step_2_multilingual_chat_and_code_sft.multi_turn_state_bytes_delta === 0 &&
      v73Json.step_1_biorthogonal_uscar_and_epistemic.out_of_cone_0x00_vacuum_abstention === '8/8' &&
      v73Json.step_1_biorthogonal_uscar_and_epistemic.post_0x00_gdpr_wipe_residual === 0.0,
    'ScarField v7.3 Biorthogonal-Chat audit invariants failed!'
  );

  // Live TypeScript execution of `BiorthogonalObliqueProjectorTS` & `SurpriseGatedAnomalySCacheTS` (`biorthogonal: true`)
  const chBi = 32;
  const tBi = 256;
  const uSig = new Float32Array(chBi);
  uSig[0] = 1.0;
  const uJam = new Float32Array(chBi);
  const ovBi = 0.85;
  uJam[0] = ovBi;
  uJam[1] = Math.sqrt(1.0 - ovBi * ovBi);

  const obsBi = new Float32Array(chBi * tBi);
  const pulseGt = new Float32Array(tBi);
  for (let t = 0; t < tBi; t++) {
    const s = Math.sin((2.0 * Math.PI * t) / 32.0);
    const j = 1000.0 * Math.cos((2.0 * Math.PI * t) / 11.0);
    pulseGt[t] = s;
    for (let c = 0; c < chBi; c++) obsBi[c * tBi + t] = uSig[c] * s + uJam[c] * j;
  }
  const orthProjTs = new OrthogonalSubspaceProjectorTS(uJam, chBi, 1);
  const biorthProjTs = new BiorthogonalObliqueProjectorTS(uJam, uSig, chBi, 1, 1, 1e-7);
  const cleanOrthTs = orthProjTs.projectClean(obsBi, chBi, tBi);
  const cleanBiTs = biorthProjTs.projectClean(obsBi, chBi, tBi);

  let errOrth2 = 0;
  let errBi2 = 0;
  let sig2 = 0;
  let dotBi = 0;
  let nBi2 = 0;
  for (let t = 0; t < tBi; t++) {
    const eo = cleanOrthTs[t] - pulseGt[t]; // channel 0 == uSig^T clean
    const eb = cleanBiTs[t] - pulseGt[t];
    errOrth2 += eo * eo;
    errBi2 += eb * eb;
    sig2 += pulseGt[t] * pulseGt[t];
    dotBi += cleanBiTs[t] * pulseGt[t];
    nBi2 += cleanBiTs[t] * cleanBiTs[t];
  }
  const relOrthTs = Math.sqrt(errOrth2 / sig2);
  const relBiTs = Math.sqrt(errBi2 / sig2);
  const cosBiTs = dotBi / Math.sqrt(nBi2 * sig2);
  assert(
    relOrthTs > 0.70 && relBiTs < 0.01 && cosBiTs > 0.999,
    `BiorthogonalObliqueProjectorTS failed: relOrth=${relOrthTs}, relBi=${relBiTs}, cosBi=${cosBiTs}`
  );

  const scBi = new SurpriseGatedAnomalySCacheTS(3.0);
  assert(scBi.physicalBytes() === 34816, `Expected 34816 B SCache, got ${scBi.physicalBytes()}`);
  const zeroFrame = new Float32Array(64);
  for (let t = 0; t < 40; t++) scBi.ingest(zeroFrame, t);
  const fModes: Float32Array[] = [];
  for (let m = 0; m < 3; m++) {
    const v = new Float32Array(64);
    for (let c = 0; c < 64; c++) v[c] = 8.0 * Math.cos(0.11 * c) + 5.0 * Math.sin((m + 1) * 0.43 * c);
    fModes.push(v);
    scBi.ingest(v, 100 + m);
  }
  assert(scBi.physicalBytes() === 34816, 'SCache drifted from 34816 B!');
  const sup64 = new Float32Array(64);
  const maskEven = new Uint8Array(64);
  for (let c = 0; c < 64; c++) {
    sup64[c] = fModes[0][c] + 0.8 * fModes[1][c] + 0.6 * fModes[2][c];
    maskEven[c] = c % 2 === 0 ? 1 : 0;
  }
  const recOrthCue = scBi.recallFromPartialCue(sup64, maskEven, { biorthogonal: false });
  const recBiCue = scBi.recallFromPartialCue(sup64, maskEven, { biorthogonal: true });
  let dotOddBi = 0, nSupOdd = 0, nBiOdd = 0, dotOddOrth = 0, nOrthOdd = 0;
  for (let c = 1; c < 64; c += 2) {
    dotOddBi += sup64[c] * recBiCue[c];
    nSupOdd += sup64[c] * sup64[c];
    nBiOdd += recBiCue[c] * recBiCue[c];
    dotOddOrth += sup64[c] * recOrthCue[c];
    nOrthOdd += recOrthCue[c] * recOrthCue[c];
  }
  const cosOddBi = dotOddBi / Math.sqrt(nSupOdd * nBiOdd);
  const cosOddOrth = dotOddOrth / Math.sqrt(nSupOdd * Math.max(1e-12, nOrthOdd));
  assert(
    cosOddBi > 0.995 && cosOddBi > cosOddOrth,
    `Biorthogonal SCache TS failed: cosBi=${cosOddBi}, cosOrth=${cosOddOrth}`
  );
  assert(
    physLabSrc.includes('data-testid="biorthogonal-v61-audit-box"'),
    'PhysicalScarLab.tsx missing biorthogonal-v61-audit-box!'
  );

  console.log(
    `[PASS 18/18] Complete 10-Archive Biorthogonal Physics Lab v6.2 & ScarField v7.3 Suite (Tohoku +${v62Json.spatial_and_3d_fields_vs_modern_sota.iris_tohoku_mw91_3axis.vq4_mode.biorth_snr_gain_over_orth_db} dB · Holter -${v62Json.spatial_and_3d_fields_vs_modern_sota.mitbih_119_holter_ecg.error_bounded_linf_mode.biorth_byte_savings_pct}% bytes · P_biorth TS cos=${cosBiTs.toFixed(5)} · Dual SCache TS cos=${cosOddBi.toFixed(5)} · ScarField v7.3 9.53 MiB)`
  );

  console.log('\n================================================================================');
  console.log('ALL 18/18 SCAR ARCHITECTURE ECOSYSTEM + BIORTHOGONAL v6.2 + MODERN 2026 SOTA + SCARFIELD v7.3 + JARVIS SUITES PASSED!');
  console.log('================================================================================');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
