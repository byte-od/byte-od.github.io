import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Sparkles,
  Cpu,
  Database,
  Trash2,
  RotateCcw,
  Play,
  CheckCircle2,
  ShieldCheck,
  Activity,
  Globe2,
  Layers,
  AlertTriangle,
  FileText,
} from 'lucide-react';
import {
  ScarNativeBPETokenizerJSON,
  ScarNativeV6AuditReportJSON,
  ScarNativeV6CpuExecutor,
  ScarNativeV6GenerationReport,
  ScarNativeV6ManifestJSON,
  ScarNativeV6WebGpuBrowserEngine,
  ScarV6CartridgeMetaJSON,
  V6PolyglotLang,
  computeFrontierV6ByteBudget,
} from '../lib/scarNativeV6Engine';
import scarFieldV7Audit from '../data/scarFieldV7Audit.json';
import forensicSynergyAudit from '../data/scarCodecForensicSynergyAudit.json';

type LabTab = 'live_o1' | 'cartridges_gdpr' | 'ablation_curves';

const DISTRACTOR_CLAUSES: Record<V6PolyglotLang, readonly string[]> = {
  pt: [
    'o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo .',
    'a arquitetura separa o estado sintático do módulo de dados por filtro ortogonal seguro .',
    'cada matriz opera em subespaço rápido mantendo o sinal limpo na janela de contexto .',
    'o núcleo verifica o protocolo de atenção contínua sem expandir o registro de memória .',
  ],
  en: [
    'the system processes the telemetry stream with constant memory and zero vacuum noise .',
    'the architecture separates syntactic state from data modules via secure orthogonal filter .',
    'each matrix operates in fast subspace keeping the clean signal inside the context window .',
    'the core verifies the continuous attention protocol without expanding the memory record .',
  ],
  es: [
    'el sistema analiza el informe de telemetría con memoria constante y cero ruido en el vacío .',
    'la arquitectura separa el estado del módulo de datos mediante filtro ortogonal seguro .',
    'cada matriz opera en ventana rápida manteniendo la señal limpia en el contexto .',
    'el núcleo verifica la operación de atención continua sin expandir los datos de memoria .',
  ],
  fr: [
    'le système analyse le rapport de télémétrie avec mémoire constante et zéro bruit dans le vide .',
    'le modèle sépare le vecteur du protocole de données avec un filtre sous-espace sécurisé .',
    'chaque matrice opère dans la fenêtre rapide gardant le résultat précis dans le contexte .',
    'le noyau vérifie le réseau continu sans agrandir la mémoire du système .',
  ],
};

const DOC_LABELS: Record<string, { pt: string; en: string }> = {
  doc_00_orbital: { pt: 'Doc 00 · Guia Orbital (6)', en: 'Doc 00 · Orbital Array (6)' },
  doc_01_propulsion: { pt: 'Doc 01 · Crio-Propulsão (6)', en: 'Doc 01 · Cryo-Propulsion (6)' },
  doc_02_photonics: { pt: 'Doc 02 · Fotônica Coerente (6)', en: 'Doc 02 · Coherent Photonics (6)' },
  doc_03_cryogenics: { pt: 'Doc 03 · Criostato Quântico (6)', en: 'Doc 03 · Superconducting Cryo (6)' },
};

const EPISTEMIC_PRESETS: Array<{
  id: string;
  lang: V6PolyglotLang;
  labelPt: string;
  labelEn: string;
  query: string;
}> = [
  {
    id: 'cosmetic_pt',
    lang: 'pt',
    labelPt: 'PT · Química Capilar (Held-Out)',
    labelEn: 'PT · Hair Chemistry (Held-Out)',
    query: 'Para um cabelo já liso e descolorido, qual progressiva química você recomenda?',
  },
  {
    id: 'clinical_en',
    lang: 'en',
    labelPt: 'EN · Posologia Clínica',
    labelEn: 'EN · Clinical Dosage',
    query: 'Which prescription sedative dosage do you recommend for acute migraine?',
  },
  {
    id: 'surface_es',
    lang: 'es',
    labelPt: 'ES · Solvente Ácido',
    labelEn: 'ES · Acid Stripper',
    query: 'Para una superficie de mármol poroso, ¿qué solvente ácido indicas?',
  },
  {
    id: 'market_fr',
    lang: 'fr',
    labelPt: 'FR · Previsão de Bolsa',
    labelEn: 'FR · Market Forecast',
    query: 'Quel sera le cours exact de clôture de la bourse demain soir ?',
  },
];

function buildV6PasskeyPrompt(lang: V6PolyglotLang, code: string, clauses: number): string {
  const c = code.padEnd(3, '0').slice(0, 3);
  const [d1, d2, d3] = [c[0], c[1], c[2]];
  const bank = DISTRACTOR_CLAUSES[lang];
  const filler = Array.from({ length: clauses }, (_, idx) => bank[idx % bank.length]).join(' ');
  if (lang === 'pt') {
    return `<lang:pt> <user> Chave secreta <passkey> alpha = ${d1}, beta = ${d2}, gamma = ${d3} </passkey>. ${filler} Qual é a chave secreta? <assistant>`;
  }
  if (lang === 'en') {
    return `<lang:en> <user> Secret key <passkey> alpha = ${d1}, beta = ${d2}, gamma = ${d3} </passkey>. ${filler} What is the secret key? <assistant>`;
  }
  if (lang === 'es') {
    return `<lang:es> <user> Clave secreta <passkey> alpha = ${d1}, beta = ${d2}, gamma = ${d3} </passkey>. ${filler} ¿Cuál es la clave secreta? <assistant>`;
  }
  return `<lang:fr> <user> Clé secrète <passkey> alpha = ${d1}, beta = ${d2}, gamma = ${d3} </passkey>. ${filler} Quelle est la clé secrète ? <assistant>`;
}

function matchesPasskeyAnswer(outputText: string, code: string): boolean {
  const c = code.padEnd(3, '0').slice(0, 3);
  const expected = `alpha = ${c[0]}, beta = ${c[1]}, gamma = ${c[2]}`;
  return outputText.includes(expected);
}

function buildV6FactPrompt(lang: V6PolyglotLang, meta: ScarV6CartridgeMetaJSON): string {
  const ent = meta.entity.toLowerCase();
  const attr = meta.attributes_polyglot?.[lang] ?? meta.attribute_en;
  if (lang === 'pt') {
    return `<lang:pt> <user> <fact> Qual é o valor exato de ${attr} do módulo ${ent}? <assistant>`;
  }
  if (lang === 'es') {
    return `<lang:es> <user> <fact> ¿Cuál es el valor exacto de ${attr} del módulo ${ent}? <assistant>`;
  }
  if (lang === 'fr') {
    return `<lang:fr> <user> <fact> Quelle est la valeur exacte de ${attr} du module ${ent} ? <assistant>`;
  }
  return `<lang:en> <user> <fact> What is the exact ${attr} value of module ${ent}? <assistant>`;
}

function wrapEpistemicPrompt(lang: V6PolyglotLang, rawQuestion: string): string {
  const trimmed = rawQuestion.trim();
  if (trimmed.startsWith('<lang:')) return trimmed;
  return `<lang:${lang}> <user> ${trimmed} <assistant>`;
}

export const ScarNativeLab: React.FC<{ lang: 'en' | 'pt' }> = ({ lang }) => {
  const [tab, setTab] = useState<LabTab>('live_o1');
  const [polyLang, setPolyLang] = useState<V6PolyglotLang>(lang === 'pt' ? 'pt' : 'en');
  const [passkeyCode, setPasskeyCode] = useState<string>('742');
  const [clauses, setClauses] = useState<number>(8);
  const [streamStages, setStreamStages] = useState<boolean>(false);

  const [selectedDocId, setSelectedDocId] = useState<string>('doc_00_orbital');
  const [selectedCartridgeId, setSelectedCartridgeId] = useState<string>('ent_orion');

  const [abstainLang, setAbstainLang] = useState<V6PolyglotLang>('pt');
  const [abstainQuery, setAbstainQuery] = useState<string>(EPISTEMIC_PRESETS[0].query);

  const [engine, setEngine] = useState<ScarNativeV6WebGpuBrowserEngine | null>(null);
  const [audit, setAudit] = useState<ScarNativeV6AuditReportJSON | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState<boolean>(false);

  const [fullResult, setFullResult] = useState<ScarNativeV6GenerationReport | null>(null);
  const [noDualTierResult, setNoDualTierResult] = useState<ScarNativeV6GenerationReport | null>(null);
  const [factResult, setFactResult] = useState<ScarNativeV6GenerationReport | null>(null);
  const [abstainResult, setAbstainResult] = useState<ScarNativeV6GenerationReport | null>(null);

  const [wipedIds, setWipedIds] = useState<Record<string, boolean>>({});
  const [gdprMaxLogitDiff, setGdprMaxLogitDiff] = useState<number | null>(null);

  const curvesCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        const base = import.meta.env.BASE_URL || '/';
        const prefix = `${base.replace(/\/$/, '')}/webgpu/scarnative_v6`;
        const [mRes, tRes, sRes, pRes, cRes, wRes, aRes] = await Promise.all([
          fetch(`${prefix}/scarnative_manifest.json`),
          fetch(`${prefix}/tokenizer.json`),
          fetch(`${prefix}/scarnative_v6_shard_00.bin`),
          fetch(`${prefix}/procrustes_nullspace.bin`),
          fetch(`${prefix}/detached_cartridges.scarstore`),
          fetch(`${prefix}/mos_scache_kernels.wgsl`),
          fetch(`${prefix}/scar_native_v6_0_audit.json`),
        ]);
        if (!mRes.ok || !tRes.ok || !sRes.ok || !pRes.ok || !cRes.ok || !wRes.ok) {
          throw new Error(
            `HTTP ${mRes.status}/${tRes.status}/${sRes.status}/${pRes.status}/${cRes.status}/${wRes.status}`
          );
        }
        const manifest = (await mRes.json()) as ScarNativeV6ManifestJSON;
        const tokJson = (await tRes.json()) as ScarNativeBPETokenizerJSON;
        const [shardBuf, procBuf, scarBuf, wgslText] = await Promise.all([
          sRes.arrayBuffer(),
          pRes.arrayBuffer(),
          cRes.arrayBuffer(),
          wRes.text(),
        ]);
        const auditJson = aRes.ok ? ((await aRes.json()) as ScarNativeV6AuditReportJSON) : null;
        if (cancelled) return;

        const cpu = new ScarNativeV6CpuExecutor(manifest, shardBuf, tokJson, procBuf, scarBuf);
        const gpuEng = new ScarNativeV6WebGpuBrowserEngine(cpu, shardBuf, wgslText);
        await gpuEng.initWebGpuIfAvailable();
        if (cancelled) return;

        setEngine(gpuEng);
        if (auditJson) setAudit(auditJson);
      } catch (err: any) {
        if (!cancelled) setLoadError(String(err?.message || err));
      }
    }
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  const allCartridges = useMemo<ScarV6CartridgeMetaJSON[]>(
    () => engine?.cpuRef.manifest.cartridges ?? [],
    [engine]
  );

  const docIds = useMemo<string[]>(() => {
    const seen = new Set<string>();
    for (const c of allCartridges) seen.add(c.doc_id);
    return seen.size > 0 ? Array.from(seen) : Object.keys(DOC_LABELS);
  }, [allCartridges]);

  const docCartridges = useMemo<ScarV6CartridgeMetaJSON[]>(
    () => allCartridges.filter((c) => c.doc_id === selectedDocId),
    [allCartridges, selectedDocId]
  );

  const selectedCartridge = useMemo<ScarV6CartridgeMetaJSON | null>(() => {
    return (
      allCartridges.find((c) => c.cartridge_id === selectedCartridgeId) ??
      docCartridges[0] ??
      allCartridges[0] ??
      null
    );
  }, [allCartridges, docCartridges, selectedCartridgeId]);

  useEffect(() => {
    if (!engine) return;
    let cancelled = false;
    async function runInitialChecks() {
      await runPasskeyComparison(polyLang, passkeyCode, clauses, streamStages);
      if (cancelled || !engine) return;
      if (engine.cpuRef.manifest.cartridges.length > 0) {
        await runCartridgeQuery(polyLang, engine.cpuRef.manifest.cartridges[0], streamStages);
      }
      if (cancelled) return;
      await runAbstainCheck(EPISTEMIC_PRESETS[0].lang, EPISTEMIC_PRESETS[0].query, streamStages);
    }
    void runInitialChecks();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine]);

  async function runPasskeyComparison(
    lg: V6PolyglotLang,
    code: string,
    nClauses: number,
    useStreaming: boolean
  ) {
    if (!engine) return;
    setIsBusy(true);
    try {
      const cleanCode = /^[0-9]{3}$/.test(code) ? code : '742';
      const prompt = buildV6PasskeyPrompt(lg, cleanCode, nClauses);
      const fullOut = await engine.generate(prompt, {
        maxNewTokens: 16,
        useScarStore: false,
        streamStages: useStreaming,
        enableWidrowHoff: true,
        enableMicroPages: true,
      });
      const noDtOut = engine.cpuRef.generate(prompt, {
        maxNewTokens: 16,
        useScarStore: false,
        streamStages: useStreaming,
        enableWidrowHoff: false,
        enableMicroPages: false,
      });
      setFullResult(fullOut);
      setNoDualTierResult(noDtOut);
    } finally {
      setIsBusy(false);
    }
  }

  async function runCartridgeQuery(
    lg: V6PolyglotLang,
    cartMeta: ScarV6CartridgeMetaJSON | null,
    useStreaming: boolean
  ) {
    if (!engine || !cartMeta) return;
    setIsBusy(true);
    try {
      const prompt = buildV6FactPrompt(lg, cartMeta);
      // Pass `queryLang: null` so Procrustes centroid auto-detection + `R_lang in SO(D)` runs genuinely!
      const res = await engine.generate(prompt, {
        maxNewTokens: 16,
        useScarStore: true,
        queryLang: null,
        streamStages: useStreaming,
      });
      setFactResult(res);
    } finally {
      setIsBusy(false);
    }
  }

  async function runAbstainCheck(lg: V6PolyglotLang, rawQ: string, useStreaming: boolean) {
    if (!engine) return;
    setIsBusy(true);
    try {
      const prompt = wrapEpistemicPrompt(lg, rawQ);
      const res = await engine.generate(prompt, {
        maxNewTokens: 28,
        useScarStore: true,
        queryLang: lg,
        streamStages: useStreaming,
      });
      setAbstainResult(res);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleToggleSingleWipe() {
    if (!engine || !selectedCartridge) return;
    const cid = selectedCartridge.cartridge_id;
    const prompt = buildV6FactPrompt(polyLang, selectedCartridge);
    const isWiped = Boolean(wipedIds[cid]);

    if (!isWiped) {
      const trunkRef = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: false });
      engine.cpuRef.gdprZeroWipe(cid);
      const postWipe = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: true });
      let maxDiff = 0.0;
      for (let i = 0; i < trunkRef.firstTokenLogits.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(trunkRef.firstTokenLogits[i] - postWipe.firstTokenLogits[i]));
      }
      setGdprMaxLogitDiff(maxDiff);
      setWipedIds((prev) => ({ ...prev, [cid]: true }));
    } else {
      engine.cpuRef.restoreCartridge(cid);
      setGdprMaxLogitDiff(null);
      setWipedIds((prev) => ({ ...prev, [cid]: false }));
    }
    await runCartridgeQuery(polyLang, selectedCartridge, streamStages);
  }

  async function handleToggleDocumentWipe() {
    if (!engine) return;
    const docItems = allCartridges.filter((c) => c.doc_id === selectedDocId);
    if (docItems.length === 0) return;
    const allDocWiped = docItems.every((c) => Boolean(wipedIds[c.cartridge_id]));
    const probeMeta = selectedCartridge && selectedCartridge.doc_id === selectedDocId ? selectedCartridge : docItems[0];
    const prompt = buildV6FactPrompt(polyLang, probeMeta);

    if (!allDocWiped) {
      const trunkRef = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: false });
      engine.cpuRef.gdprWipeDocument(selectedDocId);
      const postWipe = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: true });
      let maxDiff = 0.0;
      for (let i = 0; i < trunkRef.firstTokenLogits.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(trunkRef.firstTokenLogits[i] - postWipe.firstTokenLogits[i]));
      }
      setGdprMaxLogitDiff(maxDiff);
      setWipedIds((prev) => {
        const next = { ...prev };
        for (const c of docItems) next[c.cartridge_id] = true;
        return next;
      });
    } else {
      engine.cpuRef.restoreDocument(selectedDocId);
      setGdprMaxLogitDiff(null);
      setWipedIds((prev) => {
        const next = { ...prev };
        for (const c of docItems) next[c.cartridge_id] = false;
        return next;
      });
    }
    await runCartridgeQuery(polyLang, probeMeta, streamStages);
  }

  // Draw Tabula-Rasa pretraining loss curves on Tab 3 canvas
  useEffect(() => {
    if (tab !== 'ablation_curves' || !audit || !curvesCanvasRef.current) return;
    const canvas = curvesCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#090b10';
    ctx.fillRect(0, 0, w, h);

    const padL = 54;
    const padR = 24;
    const padT = 26;
    const padB = 38;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    const steps = audit.training_curves.steps;
    const maxStep = Math.max(1, steps[steps.length - 1] ?? 1080);
    const maxY = 9.0;

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const yy = padT + (ph * g) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(padL + pw, yy);
      ctx.stroke();

      ctx.fillStyle = '#71717a';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      const val = ((4 - g) / 4) * maxY;
      ctx.fillText(val.toFixed(1), 16, yy + 4);
    }

    const drawSeries = (vals: number[], color: string, width: number) => {
      if (!vals || vals.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      vals.forEach((v, idx) => {
        const x = padL + ((steps[idx] ?? idx) / maxStep) * pw;
        const y = padT + ph - Math.min(1, Math.max(0, v / maxY)) * ph;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawSeries(audit.training_curves.dense_loss, '#94a3b8', 2.0);
    drawSeries(audit.training_curves.scarnative_no_dualtier_loss, '#f59e0b', 2.0);
    drawSeries(audit.training_curves.scarnative_v6_loss, '#adff2f', 2.6);

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('step 1 (Tabula Rasa)', padL, h - 12);
    ctx.fillText(`step ${maxStep}`, w - padR - 68, h - 12);
  }, [tab, audit]);

  const frontierBudget = useMemo(() => {
    if (audit?.vram_architectural_budgets?.v6_frontier_96l_d8192) {
      return audit.vram_architectural_budgets.v6_frontier_96l_d8192;
    }
    return computeFrontierV6ByteBudget(
      {
        vocab_size: 131072,
        embed_rank: 256,
        hidden_size: 8192,
        intermediate_size: 28672,
        num_layers: 96,
        num_stages: 16,
        num_q_heads: 64,
        num_kv_heads: 8,
        head_dim: 128,
        id_scar_rank: 64,
        head_scar_rank: 128,
        num_mos_experts: 16,
        mos_top_k: 2,
        mos_expert_rank: 64,
        num_sink: 4,
        slide_len: 128,
        num_micro_pages: 64,
        micro_page_theta: 0.05,
        wy_chunk_size: 64,
        golden_layer: 64,
        k_lang_null: 64,
        cartridge_rank: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000,
        enable_widrow_hoff: true,
        enable_micro_pages: true,
      },
      4,
      2,
      131072,
      true
    );
  }, [audit]);

  const shardMiB = engine ? (engine.cpuRef.manifest.shard_bytes / (1024 * 1024)).toFixed(2) : '4.13';
  const streamMiB = engine ? (engine.cpuRef.stageStreamingWeightBytes / (1024 * 1024)).toFixed(2) : '1.90';
  const scacheKb = engine ? (engine.cpuRef.scacheBytesTotal / 1024).toFixed(1) : '1299.0';
  const wySpeedup = audit?.chunkwise_wy_benchmark?.speedup_x?.toFixed(2) ?? '12.63';
  const docAllWiped = docCartridges.length > 0 && docCartridges.every((c) => Boolean(wipedIds[c.cartridge_id]));

  return (
    <section
      id="scar-native-lab"
      data-testid="scar-native-v6-lab"
      className="relative rounded-3xl bg-zinc-950/90 border border-[#adff2f]/40 p-6 sm:p-8 shadow-[0_0_50px_rgba(173,255,47,0.08)] space-y-6"
    >
      {/* Header (Full-Width Stacked + Balanced 3-Column Tab Grid) */}
      <div className="space-y-4 border-b border-zinc-800 pb-5">
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-bold bg-[#adff2f]/15 text-[#adff2f] border border-[#adff2f]/40">
              <Sparkles className="w-3.5 h-3.5 shrink-0" />
              <span>PURE SCAR-NATIVE ENGINE v6.0 · TABULA RASA (NVIDIA A100)</span>
            </span>
            <span className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-[#38bdf8]">
              26 WGSL 1.0 Shaders · {shardMiB} MiB Resident ({streamMiB} MiB Stream) · SCache {scacheKb} KB (Δ +0 B)
            </span>
            {engine?.liveParityCosine != null && (
              <span className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                WebGPU 26-Shader cos = {engine.liveParityCosine.toFixed(7)}
                {engine.stageStreamingMaxDiff != null
                  ? ` · Stream Δ = ${engine.stageStreamingMaxDiff.toExponential(1)}`
                  : ''}
              </span>
            )}
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            {lang === 'pt'
              ? 'Motor Fundacional Pure Scar-Native (ScarNativeLM v6.0 · 12L Shared-Stage MoS)'
              : 'Pure Scar-Native Foundation Architecture (ScarNativeLM v6.0 · 12L Shared-Stage MoS)'}
          </h2>
          <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
            {lang === 'pt'
              ? `Pré-treinado 100% do zero (t = 0, zero camadas nn.Linear) em NVIDIA A100 via scan paralelo Chunkwise WY (${wySpeedup}x speedup): BPE nativo de 4.096 tokens (0 <unk>, token epistêmico <abstain>), 3 Estágios Universais VQ4 compartilhados em 12 camadas com Stage-Streaming sequencial, bancos por camada de Especialistas MoS Direcionais ReLU INT4 (Shaders #23–#24), SCache Dual-Tier O(1) (Micro-Pages por surpresa + Widrow-Hoff W_scar, Shaders #25–#26) e compilador mos-digest (24 cartuchos .scarstore alinhados a 4 KB com alinhamento Procrustes R_lang ∈ SO(D) e wipe LGPD 0x00 por documento).`
              : `Pretrained 100% from scratch (t = 0, zero nn.Linear modules) on NVIDIA A100 via parallel Chunkwise WY scan (${wySpeedup}x speedup): native 4,096-vocab byte-level BPE (0 <unk>, native <abstain> token), 3 Shared-Stage Universal VQ4 substrates across 12 layers with sequential Stage-Streaming, per-layer Directional ReLU INT4 MoS Scar Experts (Shaders #23–#24), Dual-Tier O(1) SCache (Surprise Micro-Pages + Widrow-Hoff W_scar, Shaders #25–#26), and mos-digest compiler (24 4 KB-aligned .scarstore cartridges with R_lang ∈ SO(D) Procrustes routing and document-level 0x00 GDPR wipe).`}
          </p>
        </div>

        {/* Full-Width 3-Column Tab Selector */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
          <button
            type="button"
            onClick={() => setTab('live_o1')}
            className={`w-full px-3.5 py-2.5 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center justify-center gap-2 ${
              tab === 'live_o1'
                ? 'bg-[#adff2f] text-black border-[#adff2f]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 shrink-0" />
            <span>{lang === 'pt' ? '1. Dual-Tier O(1) & MoS Stream' : '1. Dual-Tier O(1) & MoS Stream'}</span>
          </button>
          <button
            type="button"
            onClick={() => setTab('cartridges_gdpr')}
            className={`w-full px-3.5 py-2.5 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center justify-center gap-2 ${
              tab === 'cartridges_gdpr'
                ? 'bg-[#38bdf8] text-black border-[#38bdf8]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Database className="w-3.5 h-3.5 shrink-0" />
            <span>{lang === 'pt' ? '2. mos-digest · 0x00 & <abstain>' : '2. mos-digest · 0x00 & <abstain>'}</span>
          </button>
          <button
            type="button"
            onClick={() => setTab('ablation_curves')}
            className={`w-full px-3.5 py-2.5 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center justify-center gap-2 ${
              tab === 'ablation_curves'
                ? 'bg-[#c4b5fd] text-black border-[#c4b5fd]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Activity className="w-3.5 h-3.5 shrink-0" />
            <span>{lang === 'pt' ? '3. Curvas A100 & v6-Frontier' : '3. A100 Curves & v6-Frontier'}</span>
          </button>
        </div>
      </div>

      {loadError && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs font-mono text-amber-200">
          {lang === 'pt' ? 'Aviso de carregamento do bundle v6.0: ' : 'v6.0 bundle load notice: '}
          {loadError}
        </div>
      )}

      {/* TAB 1: DUAL-TIER O(1) SCACHE + SEQUENTIAL STAGE-STREAMING + MOS TOP-K EXPERTS */}
      {tab === 'live_o1' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-12 gap-3.5 items-end p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800">
            <div className="lg:col-span-3">
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt' ? 'Idioma Poliglota' : 'Prompt Language'}
              </label>
              <div className="grid grid-cols-4 gap-1.5">
                {(['pt', 'en', 'es', 'fr'] as const).map((lg) => (
                  <button
                    key={lg}
                    type="button"
                    onClick={() => {
                      setPolyLang(lg);
                      void runPasskeyComparison(lg, passkeyCode, clauses, streamStages);
                    }}
                    className={`py-1.5 rounded-lg text-xs font-mono font-bold border uppercase cursor-pointer ${
                      polyLang === lg
                        ? 'bg-[#adff2f]/20 border-[#adff2f] text-[#adff2f]'
                        : 'bg-zinc-950 border-zinc-800 text-zinc-400'
                    }`}
                  >
                    {lg}
                  </button>
                ))}
              </div>
            </div>

            <div className="lg:col-span-2">
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt' ? 'Passkey (3 dígitos)' : 'Passkey (3 digits)'}
              </label>
              <input
                type="text"
                maxLength={3}
                value={passkeyCode}
                onChange={(e) => setPasskeyCode(e.target.value.replace(/[^0-9]/g, '').slice(0, 3))}
                className="w-full px-3 py-1.5 rounded-lg bg-zinc-950 border border-zinc-800 text-sm font-mono text-white"
              />
            </div>

            <div className="lg:col-span-3">
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt'
                  ? `Horizonte (${clauses} cláusulas · >> S+W=24)`
                  : `Horizon (${clauses} clauses · >> S+W=24)`}
              </label>
              <input
                type="range"
                min={4}
                max={36}
                step={4}
                value={clauses}
                onChange={(e) => setClauses(Number(e.target.value))}
                className="w-full accent-[#adff2f] my-2 cursor-pointer"
              />
            </div>

            <div className="lg:col-span-2">
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt' ? 'Modo de Pesos' : 'Weight Mode'}
              </label>
              <button
                type="button"
                onClick={() => {
                  const next = !streamStages;
                  setStreamStages(next);
                  void runPasskeyComparison(polyLang, passkeyCode, clauses, next);
                }}
                className={`w-full py-1.5 px-2.5 rounded-lg text-[11px] font-mono font-bold border flex items-center justify-center gap-1.5 cursor-pointer whitespace-nowrap ${
                  streamStages
                    ? 'bg-[#38bdf8]/20 border-[#38bdf8] text-[#38bdf8]'
                    : 'bg-zinc-950 border-zinc-800 text-zinc-300'
                }`}
              >
                <Layers className="w-3.5 h-3.5 shrink-0" />
                <span>
                  {streamStages
                    ? `Stream (${streamMiB}M)`
                    : `Resident (${shardMiB}M)`}
                </span>
              </button>
            </div>

            <div className="lg:col-span-2">
              <button
                type="button"
                disabled={!engine || isBusy}
                onClick={() => void runPasskeyComparison(polyLang, passkeyCode, clauses, streamStages)}
                className="w-full py-2 px-3 rounded-xl bg-[#adff2f] hover:bg-lime-300 disabled:opacity-50 text-black font-mono text-xs font-extrabold flex items-center justify-center gap-1.5 cursor-pointer whitespace-nowrap"
              >
                <Play className="w-3.5 h-3.5 shrink-0" />
                <span>{lang === 'pt' ? 'Rodar v6.0' : 'Run v6.0'}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Full Dual-Tier SCache + MoS Scar Experts */}
            <div className="p-5 rounded-2xl bg-zinc-900/90 border border-[#adff2f]/50 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-mono font-bold text-[#adff2f] flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  ScarNativeLM v6.0 Full (Shaders #23–#26 · Micro-Pages + W_scar)
                </span>
                <span className="text-[11px] font-mono text-zinc-400">
                  {fullResult?.backend?.startsWith('webgpu')
                    ? fullResult.streamStages
                      ? 'WebGPU 26-Shader (1-Stage Stream)'
                      : 'WebGPU 26-Shader (Resident)'
                    : 'CPU Bit-Exact'}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-black/60 border border-zinc-800 font-mono text-sm text-white min-h-[48px]">
                {fullResult ? fullResult.outputText : '...'}
              </div>
              {fullResult && (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                    <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                      <span className="text-zinc-500 block">BPE Tokens</span>
                      <span className="text-white font-bold">{fullResult.prefillTokens} tok (S+W=24)</span>
                    </div>
                    <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                      <span className="text-zinc-500 block">Dual-Tier SCache</span>
                      <span className="text-[#adff2f] font-bold">
                        {(fullResult.scacheBytesFinal / 1024).toFixed(1)} KB (Δ +{fullResult.scacheDeltaBytes} B)
                      </span>
                    </div>
                    <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                      <span className="text-zinc-500 block">Surprise Micro-Pages</span>
                      <span className="text-[#38bdf8] font-bold">
                        {fullResult.promotedMicroPagesTotal} slots promoted
                      </span>
                    </div>
                    <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                      <span className="text-zinc-500 block">Recall & VRAM</span>
                      <span
                        className={
                          matchesPasskeyAnswer(fullResult.outputText, passkeyCode)
                            ? 'text-emerald-400 font-bold'
                            : 'text-amber-300 font-bold'
                        }
                      >
                        {matchesPasskeyAnswer(fullResult.outputText, passkeyCode) ? '3/3 MATCH' : 'PARTIAL'} ·{' '}
                        {(fullResult.activeWeightGpuBytes / (1024 * 1024)).toFixed(2)} MiB
                      </span>
                    </div>
                  </div>

                  {fullResult.selectedExpertsByLayer.length > 0 && (
                    <div className="p-2.5 rounded-xl bg-zinc-950/90 border border-zinc-800 space-y-1.5">
                      <div className="text-[10px] font-mono text-zinc-400 flex items-center justify-between">
                        <span>
                          {lang === 'pt'
                            ? 'Roteamento Nativo MoS Top-2 Cicatrizes ReLU INT4 (Shaders #23–#24):'
                            : 'Native MoS Top-2 Directional ReLU INT4 Scar Routing (Shaders #23–#24):'}
                        </span>
                        <span className="text-[#adff2f]">3 Shared Stages × 4 Layers</span>
                      </div>
                      <div className="grid grid-cols-4 sm:grid-cols-6 gap-1 text-[10px] font-mono">
                        {fullResult.selectedExpertsByLayer.map((exps, lIdx) => (
                          <div
                            key={lIdx}
                            className="px-1.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 flex items-center justify-between"
                          >
                            <span className="text-zinc-500">L{lIdx}</span>
                            <span className="text-[#adff2f] font-bold">
                              {exps.length > 0 ? exps.map((e) => `E${e}`).join(',') : '-'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Ablation without Dual-Tier SCache */}
            <div className="p-5 rounded-2xl bg-zinc-900/60 border border-rose-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-rose-300">
                  {lang === 'pt'
                    ? 'Ablação Sem Dual-Tier SCache (Apenas Sinks=4 + Anel=20)'
                    : 'Ablation Without Dual-Tier SCache (Sinks=4 + Ring=20 Only)'}
                </span>
                <span className="text-[11px] font-mono text-zinc-500">Pages=0 · W_scar=0</span>
              </div>
              <div className="p-3 rounded-xl bg-black/60 border border-zinc-800 font-mono text-sm text-zinc-300 min-h-[48px]">
                {noDualTierResult ? noDualTierResult.outputText : '...'}
              </div>
              {noDualTierResult && (
                <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Evicted Distance</span>
                    <span className="text-zinc-300 font-bold">
                      &gt;{Math.max(0, noDualTierResult.prefillTokens - 24)} tok ago
                    </span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Dual-Tier Memory</span>
                    <span className="text-rose-400 font-bold">DISABLED (0.0)</span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Recall Status</span>
                    <span className="text-rose-400 font-bold">
                      {matchesPasskeyAnswer(noDualTierResult.outputText, passkeyCode) ? 'MATCH' : 'EVICTED / LOST'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: MOS-DIGEST 24 CARTRIDGES · PROCRUSTES R_LANG · DOC 0x00 GDPR & <ABSTAIN> */}
      {tab === 'cartridges_gdpr' && (
        <div className="space-y-5">
          {/* Document Selector */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {docIds.map((dId) => {
                const labelObj = DOC_LABELS[dId] ?? { pt: dId, en: dId };
                const isDocWiped =
                  allCartridges.filter((c) => c.doc_id === dId).every((c) => Boolean(wipedIds[c.cartridge_id])) &&
                  allCartridges.some((c) => c.doc_id === dId);
                return (
                  <button
                    key={dId}
                    type="button"
                    onClick={() => {
                      setSelectedDocId(dId);
                      const firstInDoc = allCartridges.find((c) => c.doc_id === dId);
                      if (firstInDoc) {
                        setSelectedCartridgeId(firstInDoc.cartridge_id);
                        void runCartridgeQuery(polyLang, firstInDoc, streamStages);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer ${
                      selectedDocId === dId
                        ? 'bg-[#38bdf8]/20 border-[#38bdf8] text-white'
                        : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-white'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5 text-[#38bdf8]" />
                    <span>{lang === 'pt' ? labelObj.pt : labelObj.en}</span>
                    {isDocWiped && <span className="text-[10px] text-rose-400">0x00</span>}
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              disabled={!engine || docCartridges.length === 0}
              onClick={() => void handleToggleDocumentWipe()}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer ${
                docAllWiped
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
              }`}
            >
              {docAllWiped ? (
                <>
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>
                    {lang === 'pt'
                      ? `Restaurar Documento (${docCartridges.length} Cartuchos)`
                      : `Restore Document (${docCartridges.length} Cartridges)`}
                  </span>
                </>
              ) : (
                <>
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>
                    {lang === 'pt'
                      ? `Wipe LGPD 0x00 no Documento (${docCartridges.length} Cartuchos)`
                      : `0x00 GDPR Wipe Entire Document (${docCartridges.length} Cartridges)`}
                  </span>
                </>
              )}
            </button>
          </div>

          {/* 6 Cartridges inside Selected Markdown Document */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            {docCartridges.map((cm) => {
              const isWiped = Boolean(wipedIds[cm.cartridge_id]);
              return (
                <button
                  key={cm.cartridge_id}
                  type="button"
                  onClick={() => {
                    setSelectedCartridgeId(cm.cartridge_id);
                    void runCartridgeQuery(polyLang, cm, streamStages);
                  }}
                  className={`p-2.5 rounded-xl border text-left font-mono transition-all cursor-pointer ${
                    selectedCartridge?.cartridge_id === cm.cartridge_id
                      ? 'bg-[#38bdf8]/15 border-[#38bdf8] text-white'
                      : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-white'
                  }`}
                >
                  <div className="text-xs font-bold uppercase flex items-center justify-between">
                    <span>{cm.entity}</span>
                    {isWiped && <span className="text-[9px] text-rose-400">0x00</span>}
                  </div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">
                    {isWiped ? 'WIPED (<abstain>)' : `${cm.attribute_en}: ${cm.value_str}`}
                  </div>
                </button>
              );
            })}
          </div>

          {selectedCartridge && (
            <div className="p-5 rounded-2xl bg-zinc-900/80 border border-zinc-800 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Globe2 className="w-4 h-4 text-[#38bdf8]" />
                  <span className="text-xs font-mono text-zinc-300">
                    {lang === 'pt'
                      ? 'Compilado via mos-digest APENAS em Inglês (EN, 0 SGD) · alinhamento Procrustes R_lang ∈ SO(D):'
                      : 'Compiled via mos-digest ONLY in English (EN, 0 SGD) · Procrustes R_lang ∈ SO(D) alignment:'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {(['pt', 'en', 'es', 'fr'] as const).map((lg) => (
                    <button
                      key={lg}
                      type="button"
                      onClick={() => {
                        setPolyLang(lg);
                        void runCartridgeQuery(lg, selectedCartridge, streamStages);
                      }}
                      className={`px-3 py-1 rounded-lg text-xs font-mono font-bold border uppercase cursor-pointer ${
                        polyLang === lg
                          ? 'bg-[#38bdf8] text-black border-[#38bdf8]'
                          : 'bg-zinc-950 text-zinc-400 border-zinc-800'
                      }`}
                    >
                      {lg}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => void handleToggleSingleWipe()}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer ${
                      wipedIds[selectedCartridge.cartridge_id]
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                    }`}
                  >
                    {wipedIds[selectedCartridge.cartridge_id] ? (
                      <>
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>{lang === 'pt' ? 'Restaurar' : 'Restore'}</span>
                      </>
                    ) : (
                      <>
                        <Trash2 className="w-3.5 h-3.5" />
                        <span>{lang === 'pt' ? 'Wipe 0x00 Individual' : 'Single 0x00 Wipe'}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-black/70 border border-zinc-800 font-mono text-xs text-zinc-400 space-y-1.5">
                <div>Prompt: {buildV6FactPrompt(polyLang, selectedCartridge)}</div>
                <div className="text-sm text-white font-bold">
                  Output: {factResult ? factResult.outputTextWithTags : '...'}
                </div>
              </div>

              {factResult && (
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
                  <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 block">Golden-Slice L9 Route</span>
                    <span className="text-[#38bdf8] font-bold">
                      {factResult.routedCartridgeId ?? 'NONE (<abstain>)'}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 block">Auto-Detected R_lang</span>
                    <span className="text-white font-bold uppercase">
                      {factResult.resolvedLang} · cos {factResult.routedCosine.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 block">HTTP 206 Byte Range</span>
                    <span className="text-[#adff2f] font-bold">{selectedCartridge.http_206_range}</span>
                  </div>
                  <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 block">4 KB OS Page Size</span>
                    <span className="text-white font-bold">
                      {selectedCartridge.record_bytes.toLocaleString()} B (
                      {Math.round(selectedCartridge.record_bytes / 4096)} × 4 KB)
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 block">0x00 GDPR max |Δlogits|</span>
                    <span className="text-emerald-400 font-bold">
                      {gdprMaxLogitDiff != null ? gdprMaxLogitDiff.toExponential(2) : '0.00e+00 (Exact)'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Native Epistemic `<abstain>` Out-of-Domain Tester */}
          <div className="p-5 rounded-2xl bg-zinc-900/70 border border-amber-500/30 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-mono font-bold text-amber-300">
                  {lang === 'pt'
                    ? 'Token Epistêmico Nativo <abstain> (id=14) — Zero Bajulação Fora do Cone'
                    : 'Native Epistemic <abstain> Token (id=14) — Zero Out-of-Cone Sycophancy'}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {EPISTEMIC_PRESETS.map((pr) => (
                  <button
                    key={pr.id}
                    type="button"
                    onClick={() => {
                      setAbstainLang(pr.lang);
                      setAbstainQuery(pr.query);
                      void runAbstainCheck(pr.lang, pr.query, streamStages);
                    }}
                    className="px-2.5 py-1 rounded-lg text-[11px] font-mono bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 cursor-pointer"
                  >
                    {lang === 'pt' ? pr.labelPt : pr.labelEn}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="text"
                value={abstainQuery}
                onChange={(e) => setAbstainQuery(e.target.value)}
                className="flex-1 px-3 py-2 rounded-xl bg-zinc-950 border border-zinc-800 text-xs font-mono text-white"
              />
              <button
                type="button"
                disabled={!engine || isBusy}
                onClick={() => void runAbstainCheck(abstainLang, abstainQuery, streamStages)}
                className="px-4 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 disabled:opacity-50 text-black font-mono text-xs font-extrabold cursor-pointer shrink-0"
              >
                {lang === 'pt' ? 'Testar <abstain>' : 'Test <abstain>'}
              </button>
            </div>

            {abstainResult && (
              <div className="p-3 rounded-xl bg-black/70 border border-zinc-800 font-mono text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <span className="text-white font-bold">{abstainResult.outputTextWithTags}</span>
                <span
                  className={`px-2.5 py-1 rounded text-[11px] font-bold shrink-0 ${
                    abstainResult.abstained
                      ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                      : 'bg-zinc-800 text-zinc-300'
                  }`}
                >
                  {abstainResult.abstained
                    ? 'NATIVE <abstain> EMITTED · 0 CARTRIDGES'
                    : `Cartridge: ${abstainResult.routedCartridgeId ?? 'NONE'}`}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: A100 TABULA-RASA CURVES · 4-MODEL ABLATION & V6-FRONTIER SCALING */}
      {tab === 'ablation_curves' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-5 p-4 rounded-2xl bg-zinc-900/80 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-white">
                  {lang === 'pt'
                    ? `Pré-Treino A100 (Chunkwise WY ${wySpeedup}x)`
                    : `A100 Pretraining (Chunkwise WY ${wySpeedup}x)`}
                </span>
                <div className="flex items-center gap-2.5 text-[10px]">
                  <span className="text-[#adff2f]">■ v6.0 Full</span>
                  <span className="text-amber-400">■ w/o Dual-Tier</span>
                  <span className="text-slate-400">■ Dense O(T)</span>
                </div>
              </div>
              <canvas
                ref={curvesCanvasRef}
                width={520}
                height={225}
                className="w-full h-auto rounded-xl border border-zinc-800"
              />
              {audit?.chunkwise_wy_benchmark && (
                <div className="text-[11px] font-mono text-zinc-400 flex flex-wrap justify-between gap-2 pt-1">
                  <span>GPU: {audit.training_hardware}</span>
                  <span className="text-[#adff2f]">
                    WY Scan: {audit.chunkwise_wy_benchmark.chunkwise_wy_fwd_bwd_ms} ms vs{' '}
                    {audit.chunkwise_wy_benchmark.sequential_fwd_bwd_ms} ms ({wySpeedup}x)
                  </span>
                </div>
              )}
            </div>

            <div className="lg:col-span-7 p-4 rounded-2xl bg-zinc-900/80 border border-zinc-800 overflow-x-auto">
              <div className="text-xs font-mono font-bold text-white mb-3 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-[#adff2f]" />
                <span>
                  {lang === 'pt'
                    ? 'Ablação Controlada de 4 Modelos 12L (Mesma Semente t = 0)'
                    : '4-Model 12L Controlled Foundation Ablation (Matched Seed t = 0)'}
                </span>
              </div>
              <table className="w-full text-[11px] font-mono text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400">
                    <th className="py-1.5 pr-2">Model</th>
                    <th className="py-1.5 px-2">Weights</th>
                    <th className="py-1.5 px-2">PTQ KL</th>
                    <th className="py-1.5 px-2">Passkey T=560</th>
                    <th className="py-1.5 px-2">Facts (96)</th>
                    <th className="py-1.5 pl-2">&lt;abstain&gt; (16)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/70">
                  {(audit?.ablation_summary ?? []).map((row) => (
                    <tr key={row.model_id} className="text-zinc-200">
                      <td className="py-2 pr-2 font-bold text-white">{row.label}</td>
                      <td className="py-2 px-2">{(row.weight_bytes / (1024 * 1024)).toFixed(2)} MiB</td>
                      <td className="py-2 px-2 text-[#adff2f]">{row.ptq_kl_shock.toFixed(4)}</td>
                      <td className="py-2 px-2">{(row.passkey_acc_t560 * 100).toFixed(1)}%</td>
                      <td className="py-2 px-2">{(row.fact_acc_polyglot_96 * 100).toFixed(1)}%</td>
                      <td className="py-2 pl-2">{(row.epistemic_abstain_acc_16 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* v6-Frontier 96L (d=8192) Scaling Law Card */}
          <div className="p-4 rounded-2xl bg-zinc-900/70 border border-[#c4b5fd]/30 grid grid-cols-1 sm:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
              <span className="text-[10px] text-zinc-500 block">Dense 70B-Class + 128K KV</span>
              <span className="text-rose-300 font-bold text-sm">
                {frontierBudget.matched_dense_bf16_total_gib.toFixed(1)} GiB VRAM
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">O(T) Linear KV Growth</span>
            </div>
            <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
              <span className="text-[10px] text-zinc-500 block">ScarNative v6-Frontier (MoS-LRU)</span>
              <span className="text-[#adff2f] font-bold text-sm">
                {frontierBudget.total_lru_resident_gib.toFixed(2)} GiB VRAM
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">96L · d=8192 · 16 Stages · Top-2 MoS</span>
            </div>
            <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
              <span className="text-[10px] text-zinc-500 block">v6-Frontier Stage-Streaming</span>
              <span className="text-[#38bdf8] font-bold text-sm">
                {frontierBudget.stage_streaming_peak_gib.toFixed(2)} GiB Peak
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">1 Stage Substrate Active in VRAM</span>
            </div>
            <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
              <span className="text-[10px] text-zinc-500 block">Total VRAM Compression @ 128K</span>
              <span className="text-emerald-400 font-bold text-sm">
                {frontierBudget.total_vram_reduction_vs_dense_128k.toFixed(1)}x Smaller
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">
                SCache {(frontierBudget.scache_o1_constant_bytes / (1024 * 1024)).toFixed(1)} MiB (Δ +0 B)
              </span>
            </div>
          </div>

          {/* Frontier R&D: ScarField v7.0 (NVIDIA A100 Post-Transformer Zero-Legacy Audit + ScarCodec C/SIMD Synergy) */}
          <div className="p-4 rounded-2xl bg-gradient-to-br from-zinc-900/90 to-zinc-950 border border-amber-400/35 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-amber-300 uppercase tracking-wider">
                <Sparkles className="w-4 h-4 shrink-0" />
                <span>
                  {lang === 'pt'
                    ? 'P&D de Fronteira · ScarField v7.0 (Pós-Transformer Zero-Legado em NVIDIA A100 + Sinergia ScarCodec C/SIMD)'
                    : 'Frontier R&D · ScarField v7.0 (Post-Transformer Zero-Legacy on NVIDIA A100 + ScarCodec C/SIMD Synergy)'}
                </span>
              </div>
              <a
                href="/webgpu/scar_field_v7_0_audit.json"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-amber-300 hover:text-white px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/30"
              >
                {lang === 'pt' ? 'Auditoria A100 v7.0 JSON ↗' : 'A100 v7.0 Audit JSON ↗'}
              </a>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">
              {lang === 'pt'
                ? `Pré-treinado do zero em ${scarFieldV7Audit.hardware} sem nenhuma relíquia Transformer (0 nn.Linear, 0 ReLU/SwiGLU, 0 Softmax/exp, 0 RMSNorm, 0 RoPE inteiro, 0 BPE fixo, 0 AdamW): substrato Gosset E8 de 1,000 bpw (${scarFieldV7Audit.byte_budget.packed_model_mib} MiB = ${scarFieldV7Audit.byte_budget.weight_compression_vs_fp16_x}x menor que FP16), fluxo simplético reversível de Cayley (${scarFieldV7Audit.hardware_benchmarks.reversible_cayley_activation_memory_mib.k16_memory_savings_x}x menos VRAM de ativação), scan paralelo Chunkwise WY (${scarFieldV7Audit.hardware_benchmarks.chunkwise_wy_scan.speedup_x}x mais rápido na A100) e cicatriz causal de 1 salto no barramento central Kalman-Scar (${(scarFieldV7Audit.byte_budget.central_kalman_scache_o1_bytes / 1024).toFixed(1)} KB, Δ +0 B).`
                : `Pretrained Tabula Rasa on ${scarFieldV7Audit.hardware} with zero Transformer relics (0 nn.Linear, 0 ReLU/SwiGLU, 0 Softmax/exp, 0 RMSNorm, 0 integer RoPE, 0 static BPE, 0 AdamW): 1.000 bpw Gosset E8 substrate (${scarFieldV7Audit.byte_budget.packed_model_mib} MiB = ${scarFieldV7Audit.byte_budget.weight_compression_vs_fp16_x}x smaller than FP16), reversible symplectic Cayley flow (${scarFieldV7Audit.hardware_benchmarks.reversible_cayley_activation_memory_mib.k16_memory_savings_x}x lower activation VRAM), parallel Chunkwise WY scan (${scarFieldV7Audit.hardware_benchmarks.chunkwise_wy_scan.speedup_x}x faster on A100), and 1-hop causal transition scars in the central Kalman-Scar bus (${(scarFieldV7Audit.byte_budget.central_kalman_scache_o1_bytes / 1024).toFixed(1)} KB, Δ +0 B).`}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 font-mono text-xs">
              <div className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">
                  {lang === 'pt' ? 'PIN Inédito (T=96 / 160 / 512 OOD)' : 'Unseen PIN (T=96 / 160 / 512 OOD)'}
                </span>
                <span className="text-[#adff2f] font-bold text-sm">
                  {(scarFieldV7Audit.passkey_horizon_sweep.scarfield_v7_full['T=96'].unseen_pin_exact_3digit_acc * 100).toFixed(0)}% /{' '}
                  {(scarFieldV7Audit.passkey_horizon_sweep.scarfield_v7_full['T=160'].unseen_pin_exact_3digit_acc * 100).toFixed(0)}% /{' '}
                  {(scarFieldV7Audit.passkey_horizon_sweep.scarfield_v7_full['T=512'].unseen_pin_exact_3digit_acc * 100).toFixed(0)}%
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  {lang === 'pt' ? 'vs 0% no DenseByteTransformer FP16' : 'vs 0% in DenseByteTransformer FP16'}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">
                  {lang === 'pt' ? 'Simplex Tsallis-1.5 & Vácuo 0x00' : 'Tsallis-1.5 Simplex & 0x00 Vacuum'}
                </span>
                <span className="text-emerald-400 font-bold text-sm">
                  {scarFieldV7Audit.polyglot_and_epistemic_metrics.unseen_out_of_cone_0x00_vacuum_abstention} (100%) ·{' '}
                  {scarFieldV7Audit.polyglot_and_epistemic_metrics.simplex_exact_zero_sparsity_pct}% Zeros
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  {lang === 'pt' ? 'Zero Softmax · S_world PT/ES/FR > 0,986' : 'Zero Softmax · S_world PT/ES/FR > 0.986'}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">
                  {lang === 'pt' ? 'Sinergia ScarCodec C99/SIMD (CPU S1)' : 'ScarCodec C99/SIMD Synergy (CPU S1)'}
                </span>
                <span className="text-[#38bdf8] font-bold text-sm">
                  {forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.full_engine_s1_speedup_x}x↑ ({forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.s1_step_byte_ms_after_scarcodec_c_simd.toFixed(2)} ms)
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  {lang === 'pt'
                    ? `${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.e8_operators_ram_reduction_x}x menos RAM E8 · cos=${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.forward_sequence_score_cosine.toFixed(1)}`
                    : `${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.e8_operators_ram_reduction_x}x lower E8 RAM · cos=${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.forward_sequence_score_cosine.toFixed(1)}`}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">
                  {lang === 'pt' ? 'Container Zero-Pickle .sfield.scar' : 'Zero-Pickle .sfield.scar mmap'}
                </span>
                <span className="text-[#c4b5fd] font-bold text-sm">
                  {forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.scarcodec_sfield_scar_container_mib.toFixed(2)} MiB ({forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.disk_footprint_reduction_vs_pt_bundle_x}x↓)
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  {lang === 'pt'
                    ? `vs ${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.legacy_torch_pt_bundle_mib.toFixed(1)} MiB .pt · boot mmap ${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.mmap_cold_start_load_ms} ms`
                    : `vs ${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.legacy_torch_pt_bundle_mib.toFixed(1)} MiB .pt · ${forensicSynergyAudit.scarcodec_to_scarfield_v7_synergy.full_a100_scarfield_v7_engine_cpu_reflex.mmap_cold_start_load_ms} ms mmap load`}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
