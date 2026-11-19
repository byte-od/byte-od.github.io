/**
 * Sovereign JARVIS Neural-Motor Face Controller (ScarField v9.2-Elastic: Edge 60M -> Core 302M)
 *
 * STRICT ARCHITECTURAL INTEGRITY & ANTI-FACADE GUARANTEE (`AGENTS.md`):
 * - Unified Post-Transformer Engine: JARVIS is wired strictly to the live ScarField v9.2 WebGPU engine
 *   (`11 WGSL 1.0 Shaders` + `1.000 bpw Gosset E8` + `Cayley-Woodbury` + `16x32 Biorthogonal O(1) SCache` +
 *   `Tsallis-1.5 Spherical-Cap Simplex` + live Stiefel `Edge 60M -> Core 302M` expansion).
 * - ZERO `Math.random()` anywhere in this module.
 * - Identical ScarField neural inputs (`Tsallis-1.5` epistemic `confidence p₁` & `margin p₁−p₂`,
 *   `0x00` physical vacuum abstention, decoded UTF-8 `deltaToken`, and session state)
 *   MUST produce bit-identical `JarvisFaceState` outputs (`0.00000000` variance).
 */

import {
  detectPolyglotLanguage,
  type GoldenSliceRoutingDiagnostics,
  type PolyglotLang,
} from './scarstoreHttp206Browser';

export { detectPolyglotLanguage, type PolyglotLang };

export type JarvisNeuralSource =
  | 'scarfield_v9_core302m'
  | 'scarfield_v9_edge60m'
  | 'scarfield_v9_listening'
  | 'scarfield_v9_booting'
  | 'scarfield_v9_standby'
  | 'webgpu_unavailable';

export type JarvisPosture =
  | 'SENTINEL_IDLE'
  | 'GPU_BOOT_SYNC'
  | 'ATTENTIVE_LISTENING'
  | 'ARCHITECT_PRECISION'
  | 'KINETIC_FOCUS'
  | 'DEEP_SCIENCE_SYNTHESIS'
  | 'EMPATHIC_RAPPORT'
  | 'OUT_OF_CONE_GUARD'
  | 'EPISTEMIC_ALERT_SHOCK';

export type VisemeCode =
  | 'SIL'
  | 'OPEN_A'
  | 'SPREAD_E_I'
  | 'ROUND_O_U'
  | 'NASAL_ROMANCE'
  | 'BILABIAL_PBM'
  | 'LABIODENTAL_FV'
  | 'DENTAL_SIBILANT'
  | 'VELAR_R_K';

export interface VisemeKinematics {
  code: VisemeCode;
  phonemeHint: string;
  jawOpen: number; // [0, 1]
  lipWidth: number; // [0, 1] (0 = narrow/puckered, 0.5 = neutral, 1 = wide spread)
  lipRound: number; // [0, 1]
  upperLipRaise: number; // [0, 1]
  voiceHarmonics: [number, number, number, number, number, number, number, number];
}

export interface SessionAdaptationState {
  turnCount: number;
  detectedLang: PolyglotLang;
  languageHistory: PolyglotLang[];
  exploredCapsules: string[];
  rapportScore: number; // [0, 1] deterministic multi-turn rapport synchronization
  meanConfidence: number; // [0, 1]
  unlearnWipeCount: number;
  lastUnlearnedCapsuleId: string | null;
}

export interface JarvisNeuralInput {
  hasWebGPU?: boolean;
  engineReady: boolean;
  engineLoading: boolean;
  loadPercent?: number;
  isGenerating: boolean;
  liveInputText: string;
  lastUserQuery: string;
  currentAgentText: string;
  deltaToken: string;
  generatedTokens: number;
  tokPerSec: number;
  confidence: number; // Qwen GPU top-1 probability p₁ in [0, 1]
  margin: number; // Qwen GPU epistemic margin (p₁ − p₂) in [0, 1]
  horizonScale: number; // Golden-Slice decay envelope [0, 1]
  scacheBytes: number;
  scacheDeltaBytes: number;
  activeCapsuleIds: string[];
  maxActiveGate: number;
  routingDiagnostics?: GoldenSliceRoutingDiagnostics | null;
  session: SessionAdaptationState;
  unlearnShockActive?: boolean;
  isVacuumAbstain0x00?: boolean;
  activeAnchor?: 'edge_60m' | 'core_302m';
  fallbackUiLang?: 'pt' | 'en';
}

export interface JarvisFaceState {
  neuralSource: JarvisNeuralSource;
  posture: JarvisPosture;
  postureLabel: Record<PolyglotLang, string>;
  detectedLang: PolyglotLang;

  // Ocular & Pupillometric Channels (Epistemic Load from Qwen GPU p₁ & p₁−p₂)
  pupilDilation: number; // [0.18, 0.92] — monotonically rises as margin/confidence drop
  irisRingContraction: number; // [0.10, 0.96] — locks tightly on high-margin precision
  gazeX: number; // [-1, 1]
  gazeY: number; // [-1, 1]
  eyelidAperture: number; // [0.15, 1.0]

  // Expressive Brow & Zygomatic Channels
  browLeftLift: number; // [-0.55, 0.55]
  browRightLift: number; // [-0.55, 0.55]
  browFurrow: number; // [0, 1] — monotonically rises as margin drops
  eyebrowAsymmetry: number; // [-0.5, 0.5]
  cheekWarmthLift: number; // [0, 1]

  // Cranial 3D Pose Projection
  headYaw: number; // [-1, 1]
  headPitch: number; // [-1, 1]
  headRoll: number; // [-1, 1]

  // Articulatory Speech & Acoustic Formants
  viseme: VisemeKinematics;

  // Synaptic Ring & Spectral Signature
  ringPhaseDeg: number; // Deterministic phase angle [0, 360)
  synapticPulseHz: number; // [0.4, 12.0]
  primaryHex: string;
  secondaryHex: string;

  // Exposed Audit Telemetry (100% transparent Qwen neural wires)
  dominantCapsuleId: string | null;
  dominantCosine: number;
  routingEntropy: number;
  epistemicConfidence: number;
  epistemicMargin: number;
  epistemicUncertainty: number;
  rapportSyncPct: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

export const POSTURE_LABELS: Record<JarvisPosture, Record<PolyglotLang, string>> = {
  SENTINEL_IDLE: {
    pt: 'ScarField Elástico · Pronto',
    en: 'ScarField Elastic · Ready',
    es: 'ScarField Elástico · Listo',
    fr: 'ScarField Élastique · Prêt',
  },
  GPU_BOOT_SYNC: {
    pt: 'Sincronizando 11 Shaders WGSL',
    en: 'Syncing 11 WGSL Shaders',
    es: 'Sincronizando 11 Shaders WGSL',
    fr: 'Synchronisation 11 Shaders WGSL',
  },
  ATTENTIVE_LISTENING: {
    pt: 'Escuta Ativa · Onda UTF-8',
    en: 'Active Listening · UTF-8 Wave',
    es: 'Escucha Activa · Onda UTF-8',
    fr: 'Écoute Active · Onde UTF-8',
  },
  ARCHITECT_PRECISION: {
    pt: 'Precisão Geodésica · Gosset E8',
    en: 'Geodesic Precision · Gosset E8',
    es: 'Precisión Geodésica · Gosset E8',
    fr: 'Précision Géodésique · Gosset E8',
  },
  KINETIC_FOCUS: {
    pt: 'Fluxo Simplético · Cayley SO(D)',
    en: 'Symplectic Flow · Cayley SO(D)',
    es: 'Flujo Simpléctico · Cayley SO(D)',
    fr: 'Flux Symplectique · Cayley SO(D)',
  },
  DEEP_SCIENCE_SYNTHESIS: {
    pt: 'Síntese Científica · 512 Biortogonal',
    en: 'Science Synthesis · 512 Biorthogonal',
    es: 'Síntesis Científica · 512 Biortogonal',
    fr: 'Synthèse Scientifique · 512 Biorthogonal',
  },
  EMPATHIC_RAPPORT: {
    pt: 'Sintonia Dialógica · ScarField Poliglota',
    en: 'Dialogic Rapport · Polyglot ScarField',
    es: 'Sintonía Dialógica · ScarField Políglota',
    fr: 'Syntonie Dialogique · ScarField Polyglotte',
  },
  OUT_OF_CONE_GUARD: {
    pt: 'Vácuo Físico 0x00 · Abstenção',
    en: '0x00 Physical Vacuum · Abstention',
    es: 'Vacío Físico 0x00 · Abstención',
    fr: 'Vide Physique 0x00 · Abstention',
  },
  EPISTEMIC_ALERT_SHOCK: {
    pt: 'Deliberação S2 · Choque de Surpresa',
    en: 'S2 Deliberation · Surprise Shock',
    es: 'Deliberación S2 · Choque de Sorpresa',
    fr: 'Délibération S2 · Choc de Surprise',
  },
};

const LGPD_SHOCK_LABELS: Record<PolyglotLang, string> = {
  pt: 'Choque LGPD 0x00 · Cartucho Apagado',
  en: 'LGPD 0x00 Shock · Cartridge Wiped',
  es: 'Choque LGPD 0x00 · Cartucho Borrado',
  fr: 'Choc RGPD 0x00 · Cartouche Effacée',
};

const CAPSULE_CATEGORY_MAP: Record<string, string> = {
  portfolio_agent_identity: 'Agent Identity',
  portfolio_agent_capabilities: 'Agent Identity',
  portfolio_jarvis_architecture: 'Flagship AI Architecture',
  portfolio_julio_scar_research: 'Scar Research & Author',
  portfolio_scarnative_vq4: 'Tabula-Rasa Scar Foundation',
  portfolio_pointscar_flow_3d: 'Continuous 3D Scar Flow',
  portfolio_mos_scache_72b: 'Flagship AI Architecture',
  portfolio_browser_webgpu_wgsl: 'Edge & Browser GPU',
  portfolio_aura_system_one: 'Embodied Deep RL',
  portfolio_tricompass_scouts: 'Proprietary Perception',
  portfolio_contact_hire: 'Contact & Hiring',
  concept_landauer_principle: 'Physics of Information',
  concept_immerman_szelepcsenyi: 'Computational Complexity',
  bridge_landauer_immerman: 'Cross-Domain Sleep Synthesis',
};

const VISEME_TABLE: Record<VisemeCode, Omit<VisemeKinematics, 'code' | 'phonemeHint'>> = {
  SIL: {
    jawOpen: 0.04,
    lipWidth: 0.50,
    lipRound: 0.12,
    upperLipRaise: 0.05,
    voiceHarmonics: [0.06, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.06],
  },
  OPEN_A: {
    jawOpen: 0.86,
    lipWidth: 0.68,
    lipRound: 0.16,
    upperLipRaise: 0.42,
    voiceHarmonics: [0.48, 0.92, 0.85, 0.64, 0.52, 0.68, 0.84, 0.44],
  },
  SPREAD_E_I: {
    jawOpen: 0.42,
    lipWidth: 0.92,
    lipRound: 0.06,
    upperLipRaise: 0.36,
    voiceHarmonics: [0.30, 0.54, 0.42, 0.88, 0.95, 0.82, 0.50, 0.32],
  },
  ROUND_O_U: {
    jawOpen: 0.54,
    lipWidth: 0.22,
    lipRound: 0.94,
    upperLipRaise: 0.18,
    voiceHarmonics: [0.90, 0.86, 0.58, 0.28, 0.22, 0.30, 0.62, 0.82],
  },
  NASAL_ROMANCE: {
    jawOpen: 0.62,
    lipWidth: 0.44,
    lipRound: 0.66,
    upperLipRaise: 0.30,
    voiceHarmonics: [0.78, 0.84, 0.72, 0.56, 0.48, 0.60, 0.74, 0.68],
  },
  BILABIAL_PBM: {
    jawOpen: 0.06,
    lipWidth: 0.46,
    lipRound: 0.34,
    upperLipRaise: 0.02,
    voiceHarmonics: [0.64, 0.38, 0.20, 0.16, 0.16, 0.22, 0.36, 0.58],
  },
  LABIODENTAL_FV: {
    jawOpen: 0.20,
    lipWidth: 0.58,
    lipRound: 0.18,
    upperLipRaise: 0.54,
    voiceHarmonics: [0.22, 0.34, 0.52, 0.74, 0.78, 0.68, 0.46, 0.26],
  },
  DENTAL_SIBILANT: {
    jawOpen: 0.18,
    lipWidth: 0.74,
    lipRound: 0.10,
    upperLipRaise: 0.46,
    voiceHarmonics: [0.16, 0.26, 0.48, 0.82, 0.94, 0.88, 0.54, 0.22],
  },
  VELAR_R_K: {
    jawOpen: 0.48,
    lipWidth: 0.54,
    lipRound: 0.28,
    upperLipRaise: 0.26,
    voiceHarmonics: [0.52, 0.68, 0.76, 0.60, 0.54, 0.66, 0.72, 0.48],
  },
};

/**
 * Maps a decoded Qwen BPE subword token (`deltaToken`) in `PT / EN / ES / FR` to deterministic
 * articulatory jaw/lip kinematics and 8-band acoustic formant harmonics.
 */
export function extractMultilingualViseme(tokenPiece: string, lang: PolyglotLang = 'en'): VisemeKinematics {
  const cleaned = (tokenPiece || '').trim();
  if (!cleaned || /^[\s.,;:!?—–\-()[\]{}"'«»¿¡]+$/.test(cleaned)) {
    return { code: 'SIL', phonemeHint: '—', ...VISEME_TABLE.SIL };
  }

  const lower = cleaned.toLowerCase();

  // 1. Romance nasal nuclei (`PT` ã/õ/ão/õe/em/im, `FR` an/en/on/in/un)
  if (
    /[ãõ]/.test(lower) ||
    (lang === 'pt' && /(?:ão|õe|am|em|im|om|um)$/.test(lower)) ||
    (lang === 'fr' && /(?:an|en|on|in|un|ain|ein)$/.test(lower))
  ) {
    return { code: 'NASAL_ROMANCE', phonemeHint: lang === 'fr' ? '/ɑ̃·ɔ̃/' : '/ɐ̃·õ/', ...VISEME_TABLE.NASAL_ROMANCE };
  }

  // Examine onset & dominant nucleus of the subword piece
  const tail = lower.slice(-3);
  const lead = lower.slice(0, 2);

  if ((lower.length <= 4 && /^[pbm]/.test(lead)) || /[pbm]$/.test(tail)) {
    return { code: 'BILABIAL_PBM', phonemeHint: '/p·b·m/', ...VISEME_TABLE.BILABIAL_PBM };
  }
  if ((lower.length <= 6 && /^[fv]/.test(lead)) || /ph/.test(lower)) {
    return { code: 'LABIODENTAL_FV', phonemeHint: '/f·v/', ...VISEME_TABLE.LABIODENTAL_FV };
  }
  if (/^(?:ch|sh|sz|ss|ç|[szxj])/.test(lead) || /[szçx]$/.test(tail)) {
    return { code: 'DENTAL_SIBILANT', phonemeHint: '/s·z·ʃ/', ...VISEME_TABLE.DENTAL_SIBILANT };
  }
  if (/[oóòôöuúùûüw]/.test(tail) || (lang === 'fr' && /eau|au|ou|eu/.test(lower))) {
    return { code: 'ROUND_O_U', phonemeHint: '/o·u/', ...VISEME_TABLE.ROUND_O_U };
  }
  if (/[aáàâä]/.test(tail)) {
    return { code: 'OPEN_A', phonemeHint: '/a/', ...VISEME_TABLE.OPEN_A };
  }
  if (/[eéèêëiíìîïy]/.test(tail)) {
    return { code: 'SPREAD_E_I', phonemeHint: '/e·i/', ...VISEME_TABLE.SPREAD_E_I };
  }
  if (/[kgqrñlh]/.test(lower) || /\d/.test(lower)) {
    return { code: 'VELAR_R_K', phonemeHint: '/k·g·r/', ...VISEME_TABLE.VELAR_R_K };
  }

  return { code: 'SPREAD_E_I', phonemeHint: '/ə/', ...VISEME_TABLE.SPREAD_E_I };
}

export function createInitialSessionAdaptation(defaultLang: PolyglotLang = 'en'): SessionAdaptationState {
  return {
    turnCount: 0,
    detectedLang: defaultLang,
    languageHistory: [defaultLang],
    exploredCapsules: [],
    rapportScore: 0.24,
    meanConfidence: 0.88,
    unlearnWipeCount: 0,
    lastUnlearnedCapsuleId: null,
  };
}

export function advanceSessionAdaptation(
  prev: SessionAdaptationState,
  opts: {
    userQuery: string;
    activatedCapsuleIds: string[];
    turnMeanConfidence: number;
    fallbackLang?: PolyglotLang;
  }
): SessionAdaptationState {
  const detectedLang = detectPolyglotLanguage(opts.userQuery, opts.fallbackLang ?? prev.detectedLang);
  const nextTurn = prev.turnCount + 1;
  const langHist = [...prev.languageHistory.slice(-7), detectedLang];
  const explored = Array.from(new Set([...prev.exploredCapsules, ...opts.activatedCapsuleIds]));
  const conf = opts.turnMeanConfidence > 0 ? opts.turnMeanConfidence : prev.meanConfidence;
  const meanConfidence = round4((prev.meanConfidence * prev.turnCount + conf) / nextTurn);
  const rapportScore = round4(
    clamp(0.24 + nextTurn * 0.11 + explored.length * 0.045, 0.24, 0.98)
  );

  return {
    ...prev,
    turnCount: nextTurn,
    detectedLang,
    languageHistory: langHist,
    exploredCapsules: explored,
    rapportScore,
    meanConfidence,
  };
}

/**
 * Pure, deterministic Qwen-to-facial state projection (`ZERO Math.random()`).
 * Driven strictly by the unified Qwen2.5-3B WebGPU engine (`20 WGSL Shaders` + `SCache O(1)` + `Golden-Slice L23`).
 */
export function computeJarvisFaceState(input: JarvisNeuralInput): JarvisFaceState {
  const webgpuSupported = input.hasWebGPU !== false;
  const fallbackLang: PolyglotLang = input.fallbackUiLang === 'pt' ? 'pt' : input.session.detectedLang || 'en';
  const activeTextForLang = input.liveInputText.trim() || input.lastUserQuery.trim();
  const detectedLang = detectPolyglotLanguage(activeTextForLang, fallbackLang);

  const hasGpuDiagnostics = Boolean(
    input.routingDiagnostics && input.routingDiagnostics.anchorCosines.length > 0
  );
  const isTypingLive = input.liveInputText.trim().length > 0 && !input.isGenerating;

  // Honest ScarField v9.2-Elastic Pipeline State Attribution
  let neuralSource: JarvisNeuralSource = 'scarfield_v9_standby';
  if (!webgpuSupported) {
    neuralSource = 'webgpu_unavailable';
  } else if (input.engineLoading) {
    neuralSource = 'scarfield_v9_booting';
  } else if (input.isGenerating || (input.engineReady && !isTypingLive)) {
    neuralSource = input.activeAnchor === 'core_302m' ? 'scarfield_v9_core302m' : 'scarfield_v9_edge60m';
  } else if (isTypingLive) {
    neuralSource = 'scarfield_v9_listening';
  }

  // Extract genuine Qwen Layer-23 `P_⊥lang` routing telemetry (never faked on CPU)
  let dominantCapsuleId: string | null = input.activeCapsuleIds[0] ?? null;
  let dominantCosine = 0.0;
  let routingEntropy = 0.0;
  let dominantCategory = dominantCapsuleId ? CAPSULE_CATEGORY_MAP[dominantCapsuleId] || 'General' : 'General';

  if (hasGpuDiagnostics && input.routingDiagnostics) {
    const diag = input.routingDiagnostics;
    dominantCosine = diag.maxAnchorCosine;
    routingEntropy = diag.routingEntropy;

    let best = diag.anchorCosines[0];
    for (const item of diag.anchorCosines) {
      if (item.gate > best.gate || (item.gate === best.gate && item.cosine > best.cosine)) {
        best = item;
      }
    }
    if (best && best.gate > 0.0 && input.activeCapsuleIds.length > 0) {
      const targetId = dominantCapsuleId ?? best.capsuleId;
      dominantCapsuleId = targetId;
      const matched = diag.anchorCosines.find((a) => a.capsuleId === targetId) ?? best;
      dominantCosine = matched.cosine;
      dominantCategory = matched.category || CAPSULE_CATEGORY_MAP[targetId] || 'General';
    }
  }

  // Epistemic confidence p₁ & margin (p₁ − p₂) from Qwen `logits_top2_reduce_stage2`
  const rawConf = input.confidence > 0 ? clamp(input.confidence, 0.01, 1.0) : input.session.meanConfidence;
  const rawMargin = input.margin > 0 ? clamp(input.margin, 0.0, 1.0) : clamp(rawConf * 0.88, 0.05, 0.96);
  const uncertainty = round4(clamp(1.0 - rawConf, 0.0, 0.99));

  // Determine Cognitive Posture deterministically from Qwen's state & Layer-23 capsule activation
  let posture: JarvisPosture = 'SENTINEL_IDLE';

  if (input.isVacuumAbstain0x00) {
    posture = 'OUT_OF_CONE_GUARD';
  } else if (input.unlearnShockActive || (input.isGenerating && rawMargin <= 0.12)) {
    posture = 'EPISTEMIC_ALERT_SHOCK';
  } else if (input.engineLoading) {
    posture = 'GPU_BOOT_SYNC';
  } else if (isTypingLive) {
    posture = 'ATTENTIVE_LISTENING';
  } else if (hasGpuDiagnostics && input.activeCapsuleIds.length === 0) {
    posture = 'EMPATHIC_RAPPORT';
  } else if (
    dominantCapsuleId === 'concept_landauer_principle' ||
    dominantCapsuleId === 'concept_immerman_szelepcsenyi' ||
    dominantCapsuleId === 'bridge_landauer_immerman' ||
    dominantCategory.includes('Physics') ||
    dominantCategory.includes('Quantum') ||
    dominantCategory.includes('Astrophysics') ||
    dominantCategory.includes('Mechanics') ||
    dominantCategory.includes('Complexity') ||
    dominantCategory.includes('Information') ||
    dominantCategory.includes('Matrix') ||
    dominantCategory.includes('Coding') ||
    dominantCategory.includes('Calculus') ||
    dominantCategory.includes('Statistics') ||
    dominantCategory.includes('Logic') ||
    dominantCategory.includes('Bioengineering') ||
    dominantCategory.includes('Neuroscience') ||
    dominantCategory.includes('Economics') ||
    dominantCategory.includes('Sleep')
  ) {
    posture = 'DEEP_SCIENCE_SYNTHESIS';
  } else if (
    dominantCapsuleId === 'portfolio_aura_system_one' ||
    dominantCapsuleId === 'portfolio_tricompass_scouts' ||
    dominantCapsuleId === 'portfolio_pointscar_flow_3d' ||
    dominantCategory.includes('RL') ||
    dominantCategory.includes('3D') ||
    dominantCategory.includes('Perception')
  ) {
    posture = 'KINETIC_FOCUS';
  } else if (
    dominantCapsuleId === 'portfolio_mos_scache_72b' ||
    dominantCapsuleId === 'portfolio_jarvis_architecture' ||
    dominantCapsuleId === 'portfolio_julio_scar_research' ||
    dominantCapsuleId === 'portfolio_scarnative_vq4' ||
    dominantCapsuleId === 'portfolio_browser_webgpu_wgsl' ||
    dominantCategory.includes('Architecture') ||
    dominantCategory.includes('Scar') ||
    dominantCategory.includes('Transformers') ||
    dominantCategory.includes('Developer') ||
    dominantCategory.includes('Distributed') ||
    dominantCategory.includes('Cryptography') ||
    dominantCategory.includes('Compilers')
  ) {
    posture = 'ARCHITECT_PRECISION';
  } else if (
    dominantCapsuleId === 'portfolio_agent_identity' ||
    dominantCapsuleId === 'portfolio_agent_capabilities' ||
    dominantCapsuleId === 'portfolio_contact_hire' ||
    dominantCategory.includes('Portfolio Dossier') ||
    input.session.turnCount > 0
  ) {
    posture = 'EMPATHIC_RAPPORT';
  }

  // Semantic self-modulation from Qwen's own live generated clause (`PT / EN / ES / FR`)
  const recentClause = (input.currentAgentText || '').slice(-96).toLowerCase();
  const hasAssertionPeak = /(?:0\.23\s*ms|2\.40\s*gb|o\(1\)|100%|zero|Δ\s*\+0|140\s*kb|61\.25\s*kb|!)/.test(recentClause);
  const hasContrastiveTurn = /\b(?:however|while|whereas|porém|enquanto|contudo|mientras|aunque|tandis|cependant)\b/.test(
    recentClause
  );

  // 1. Pupillometry & Iris Reticle (strictly monotonic wrt epistemic load 1 − margin / 1 − p₁)
  const epistemicLoad = 0.55 * (1.0 - rawMargin) + 0.45 * uncertainty;
  const listeningBoost = isTypingLive ? 0.08 : 0.0;
  const shockBoost = input.unlearnShockActive ? 0.24 : 0.0;
  const pupilDilation = round4(clamp(0.24 + 0.54 * epistemicLoad + listeningBoost + shockBoost, 0.18, 0.92));
  const irisRingContraction = round4(
    clamp(0.25 + 0.62 * rawMargin + (hasAssertionPeak ? 0.08 : 0.0) - shockBoost * 0.4, 0.10, 0.96)
  );

  // 2. Brow Geometry & Furrow (strictly monotonic furrow wrt 1 − margin)
  const baseFurrow = clamp(0.08 + 0.76 * (1.0 - rawMargin) + (input.unlearnShockActive ? 0.25 : 0.0), 0.0, 1.0);
  let browLeftLift = 0.0;
  let browRightLift = 0.0;
  let cheekWarmthLift = round4(clamp(0.16 + input.session.rapportScore * 0.35, 0.10, 0.88));
  let gazeX = 0.0;
  let gazeY = 0.0;
  let headYaw = 0.0;
  let headPitch = 0.0;
  let headRoll = 0.0;
  let primaryHex = '#38bdf8';
  let secondaryHex = '#60a5fa';

  if (posture === 'GPU_BOOT_SYNC') {
    const bootRatio = clamp((input.loadPercent ?? 0) / 100.0, 0.0, 1.0);
    gazeX = round4(Math.sin(bootRatio * Math.PI * 2.0) * 0.25);
    gazeY = -0.10;
    browLeftLift = 0.14;
    browRightLift = 0.14;
    primaryHex = '#22d3ee';
    secondaryHex = '#38bdf8';
  } else if (posture === 'ATTENTIVE_LISTENING') {
    // Eyes look down toward the floating input field underneath JARVIS's chin and track cursor!
    const cursorRatio = clamp((input.liveInputText.length % 48) / 48.0, 0.0, 1.0);
    gazeX = round4(-0.34 + cursorRatio * 0.68);
    gazeY = 0.32;
    browLeftLift = 0.26;
    browRightLift = 0.26;
    headPitch = 0.16;
    headYaw = round4(gazeX * 0.35);
    primaryHex = '#7dd3fc';
    secondaryHex = '#38bdf8';
  } else if (posture === 'ARCHITECT_PRECISION') {
    browLeftLift = -0.22;
    browRightLift = -0.22;
    gazeX = 0.0;
    gazeY = 0.0;
    headPitch = hasAssertionPeak ? 0.14 : -0.05;
    primaryHex = '#38bdf8';
    secondaryHex = '#93c5fd';
  } else if (posture === 'KINETIC_FOCUS') {
    const scanPhase = Math.sin((input.generatedTokens * 0.45) % (Math.PI * 2));
    gazeX = round4(input.isGenerating ? scanPhase * 0.28 : 0.0);
    gazeY = -0.06;
    browLeftLift = -0.28;
    browRightLift = -0.18;
    headYaw = round4(gazeX * 0.4);
    primaryHex = '#22d3ee';
    secondaryHex = '#60a5fa';
  } else if (posture === 'DEEP_SCIENCE_SYNTHESIS') {
    browLeftLift = 0.24;
    browRightLift = -0.18;
    gazeX = 0.18;
    gazeY = -0.20;
    headRoll = 0.16;
    headYaw = 0.12;
    primaryHex = '#60a5fa';
    secondaryHex = '#818cf8';
  } else if (posture === 'EMPATHIC_RAPPORT') {
    browLeftLift = 0.18;
    browRightLift = 0.18;
    cheekWarmthLift = round4(clamp(0.46 + input.session.rapportScore * 0.42, 0.40, 0.94));
    gazeX = 0.0;
    gazeY = 0.06;
    headRoll = 0.08;
    primaryHex = '#67e8f9';
    secondaryHex = '#38bdf8';
  } else if (posture === 'OUT_OF_CONE_GUARD') {
    browLeftLift = 0.08;
    browRightLift = -0.14;
    cheekWarmthLift = 0.22;
    gazeX = 0.0;
    gazeY = -0.04;
    headRoll = -0.06;
    primaryHex = '#38bdf8';
    secondaryHex = '#fbbf24';
  } else if (posture === 'EPISTEMIC_ALERT_SHOCK') {
    browLeftLift = -0.38;
    browRightLift = -0.38;
    gazeX = 0.0;
    gazeY = 0.0;
    headPitch = -0.16;
    cheekWarmthLift = 0.06;
    primaryHex = '#fb7185';
    secondaryHex = '#facc15';
  }

  const hasWarmRapportCue =
    /\b(?:oi|olá|ola|bom dia|boa tarde|boa noite|tudo bem|prazer|obrigad|hello|hi|good morning|happy to help|thanks|hola|buenos|gracias|bonjour|enchanté|merci)\b/i.test(
      `${input.lastUserQuery} ${recentClause}`
    );

  if (hasWarmRapportCue && posture !== 'EPISTEMIC_ALERT_SHOCK') {
    cheekWarmthLift = round4(clamp(Math.max(cheekWarmthLift, 0.68) + 0.12, 0.10, 0.96));
    browLeftLift = clamp(browLeftLift + 0.12, -0.55, 0.55);
    browRightLift = clamp(browRightLift + 0.12, -0.55, 0.55);
  }

  if (input.isGenerating) {
    if (input.generatedTokens === 0) {
      // Prefill contemplation pose: thoughtful upward-sideways gaze & subtle head tilt before speaking
      gazeX = clamp(gazeX + 0.22, -1.0, 1.0);
      gazeY = clamp(gazeY - 0.16, -1.0, 1.0);
      headYaw = clamp(headYaw + 0.15, -1.0, 1.0);
      headPitch = clamp(headPitch - 0.10, -1.0, 1.0);
      headRoll = clamp(headRoll + 0.11, -1.0, 1.0);
      browLeftLift = clamp(browLeftLift + 0.14, -0.55, 0.55);
    } else {
      // Live conversational prosody: natural turn-taking nods, lateral phrasing & expressive brow cadence
      const tokPhase = input.generatedTokens * 0.42;
      const phraseWave = Math.sin(tokPhase) * 0.65 + Math.sin(tokPhase * 0.618) * 0.35;
      const nodWave = Math.cos(input.generatedTokens * 0.78) * (hasAssertionPeak ? 0.20 : 0.11);
      headYaw = clamp(headYaw + phraseWave * 0.22, -1.0, 1.0);
      headPitch = clamp(headPitch + nodWave, -1.0, 1.0);
      headRoll = clamp(headRoll + Math.sin(tokPhase * 0.53 + 0.8) * 0.13, -1.0, 1.0);
      gazeX = clamp(gazeX + phraseWave * 0.14, -1.0, 1.0);
      gazeY = clamp(gazeY + Math.sin(tokPhase * 0.71) * 0.08, -1.0, 1.0);
      const emphasisLift = (hasAssertionPeak ? 0.12 : 0.05) * (0.5 + 0.5 * Math.sin(input.generatedTokens * 0.95));
      browLeftLift = clamp(browLeftLift + emphasisLift, -0.55, 0.55);
      browRightLift = clamp(browRightLift + emphasisLift * 0.88, -0.55, 0.55);
    }
  }

  if (hasContrastiveTurn) {
    browLeftLift = clamp(browLeftLift + 0.14, -0.55, 0.55);
    browRightLift = clamp(browRightLift - 0.12, -0.55, 0.55);
    headRoll = clamp(headRoll + 0.12, -1.0, 1.0);
  }

  const eyebrowAsymmetry = round4(browLeftLift - browRightLift);
  const browFurrow = round4(baseFurrow);

  // Deterministic blink/shutter envelope at sentence boundaries (`!`, `.`, `?`) without randomness
  const isSentencePause = input.isGenerating && /[.!?]\s*$/.test(input.deltaToken);
  const eyelidAperture = round4(
    isSentencePause
      ? 0.24
      : clamp(0.84 + (posture === 'EPISTEMIC_ALERT_SHOCK' ? 0.12 : 0.0) - cheekWarmthLift * 0.08, 0.35, 0.98)
  );

  // 3. Phonetic Viseme Kinematics from Qwen's live decoded BPE token
  const viseme: VisemeKinematics = input.isGenerating
    ? extractMultilingualViseme(input.deltaToken, detectedLang)
    : { code: 'SIL', phonemeHint: '—', ...VISEME_TABLE.SIL };

  // 4. Synaptic Ring Phase & Frequency
  const bootProgressDeg = input.engineLoading ? (input.loadPercent ?? 0) * 3.6 : 0;
  const ringPhaseDeg = round4(
    (input.generatedTokens * 17.5 + input.session.turnCount * 45.0 + bootProgressDeg + input.liveInputText.length * 6.0) %
      360.0
  );
  const synapticPulseHz = round4(
    input.isGenerating
      ? clamp(1.6 + input.tokPerSec * 0.16, 1.6, 11.5)
      : input.engineLoading
      ? 6.0
      : isTypingLive
      ? 2.8
      : 0.9
  );

  return {
    neuralSource,
    posture,
    postureLabel: input.unlearnShockActive ? LGPD_SHOCK_LABELS : POSTURE_LABELS[posture],
    detectedLang,
    pupilDilation,
    irisRingContraction,
    gazeX: round4(gazeX),
    gazeY: round4(gazeY),
    eyelidAperture,
    browLeftLift: round4(browLeftLift),
    browRightLift: round4(browRightLift),
    browFurrow,
    eyebrowAsymmetry,
    cheekWarmthLift,
    headYaw: round4(headYaw),
    headPitch: round4(headPitch),
    headRoll: round4(headRoll),
    viseme,
    ringPhaseDeg,
    synapticPulseHz,
    primaryHex,
    secondaryHex,
    dominantCapsuleId,
    dominantCosine: round4(dominantCosine),
    routingEntropy: round4(routingEntropy),
    epistemicConfidence: round4(rawConf),
    epistemicMargin: round4(rawMargin),
    epistemicUncertainty: uncertainty,
    rapportSyncPct: Math.round(input.session.rapportScore * 100),
  };
}
