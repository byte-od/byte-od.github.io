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
  buildCfdHdrShockFieldProof,
  runBlackBoxSCacheProof,
  runRadioTelescopeRfiProof,
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
    engineCode.includes('runFullLayersOnLastToken') && engineCode.includes('out_of_cone_guarded'),
    'Missing causal split runFullLayersOnLastToken or out_of_cone_guarded in mosWebgpuEngine.ts!'
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

  const stageRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/components/JarvisFloatingPointStage.tsx'), 'utf-8');
  const workerRaw = fs.readFileSync(path.join(PORTFOLIO_ROOT, 'src/lib/llmWorker.ts'), 'utf-8');
  assert(
    stageRaw.includes('data-testid="jarvis-floating-point-stage"') &&
      widgetCode.includes('jarvis-pointscar-toggle') &&
      engineCode.includes('BrowserPointScarFlowEngine') &&
      engineCode.includes('pointScarSample') &&
      workerRaw.includes('pointScarFinalF32'),
    'Missing PointScarFlow full-viewport stage or WebGPU worker wiring!'
  );
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
    'src/lib/tokenizer.ts',
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
    physLabRaw.includes('data-testid="physical-scar-forensic-scope"'),
    'Missing data-testid="physical-scar-forensic-scope" epistemological scope banner in PhysicalScarLab.tsx!'
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
    `[PASS 14/14] PhysicalScarLab Forensic Scientific Audit (Blind SVD cos=${p1.scarCodecCosine.toFixed(4)} · Partial-Cue Masked cos=${p2.maskedChannelsMeanCosine.toFixed(4)} · Burgers PDE ${p3.scarBytes}B vs 2D-DCT ${p3.spectralBytes}B ${(p3.spectralRmse / p3.scarRmse).toFixed(1)}x RMSE Gain · Zero Stale Claims)`
  );

  console.log('\n================================================================================');
  console.log('ALL 14/14 SCAR ARCHITECTURE ECOSYSTEM + SCARNATIVE v6.0 + PHYSICAL FORENSIC + JARVIS + POINTSCARFLOW SUITES PASSED!');
  console.log('================================================================================');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
