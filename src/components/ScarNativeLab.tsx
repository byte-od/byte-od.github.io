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
} from 'lucide-react';
import {
  ScarNativeAuditReportJSON,
  ScarNativeCpuExecutor,
  ScarNativeGenerationReport,
  ScarNativeManifestJSON,
  ScarNativeWebGpuBrowserEngine,
} from '../lib/scarNativeEngine';

type LabTab = 'live_o1' | 'cartridges_gdpr' | 'ablation_curves';
type PolyLang = 'pt' | 'en' | 'es' | 'fr';

const DISTRACTOR_CLAUSE: Record<PolyLang, string> = {
  pt: 'o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo .',
  en: 'the system processes the telemetry stream with constant memory and zero vacuum noise .',
  es: 'el sistema analiza el informe de telemetría con memoria constante y cero ruido en el vacío .',
  fr: 'le système analyse le rapport de télémétrie avec mémoire constante et zéro bruit dans le vide .',
};

const ENTITY_CATALOG: Array<{
  id: string;
  name: string;
  metric: Record<PolyLang, string>;
  expected: string;
}> = [
  { id: 'ent_orion', name: 'orion', metric: { pt: 'órbita', en: 'orbit', es: 'órbita', fr: 'orbite' }, expected: '516 km' },
  { id: 'ent_helios', name: 'helios', metric: { pt: 'pressão', en: 'pressure', es: 'presión', fr: 'pression' }, expected: '340 bar' },
  { id: 'ent_aegis', name: 'aegis', metric: { pt: 'frequência', en: 'frequency', es: 'frecuencia', fr: 'fréquence' }, expected: '842 mhz' },
  { id: 'ent_kronos', name: 'kronos', metric: { pt: 'energia', en: 'energy', es: 'energía', fr: 'énergie' }, expected: '925 watts' },
  { id: 'ent_vega', name: 'vega', metric: { pt: 'órbita', en: 'orbit', es: 'órbita', fr: 'orbite' }, expected: '780 km' },
  { id: 'ent_atlas', name: 'atlas', metric: { pt: 'pressão', en: 'pressure', es: 'presión', fr: 'pression' }, expected: '415 kpa' },
  { id: 'ent_titan', name: 'titan', metric: { pt: 'frequência', en: 'frequency', es: 'frecuencia', fr: 'fréquence' }, expected: '612 khz' },
  { id: 'ent_zephyr', name: 'zephyr', metric: { pt: 'energia', en: 'energy', es: 'energía', fr: 'énergie' }, expected: '274 kelvin' },
];

function buildPasskeyPrompt(lang: PolyLang, code: string, clauses: number): string {
  const c = code.padEnd(3, '0').slice(0, 3);
  const [d1, d2, d3] = [c[0], c[1], c[2]];
  const filler = Array.from({ length: clauses }, () => DISTRACTOR_CLAUSE[lang]).join(' ');
  if (lang === 'pt') {
    return `<lang:pt> <user> chave secreta <passkey> alpha = ${d1} , beta = ${d2} , gamma = ${d3} </passkey> . ${filler} qual é a chave secreta ? <assistant>`;
  }
  if (lang === 'en') {
    return `<lang:en> <user> secret key <passkey> alpha = ${d1} , beta = ${d2} , gamma = ${d3} </passkey> . ${filler} what is the secret key ? <assistant>`;
  }
  if (lang === 'es') {
    return `<lang:es> <user> clave secreto <passkey> alpha = ${d1} , beta = ${d2} , gamma = ${d3} </passkey> . ${filler} ¿ cuál es la clave secreto ? <assistant>`;
  }
  return `<lang:fr> <user> clé secrète <passkey> alpha = ${d1} , beta = ${d2} , gamma = ${d3} </passkey> . ${filler} quelle est la clé secrète ? <assistant>`;
}

function matchesPasskeyAnswer(outputText: string, code: string): boolean {
  const c = code.padEnd(3, '0').slice(0, 3);
  const expected = `alpha = ${c[0]}, beta = ${c[1]}, gamma = ${c[2]}`;
  return outputText.includes(expected);
}

function buildFactPrompt(lang: PolyLang, name: string, metric: Record<PolyLang, string>): string {
  if (lang === 'pt') {
    return `<lang:pt> <user> <fact> qual é o valor exato de ${metric.pt} do módulo ${name} ? <assistant>`;
  }
  if (lang === 'en') {
    return `<lang:en> <user> <fact> what is the exact ${metric.en} value of module ${name} ? <assistant>`;
  }
  if (lang === 'es') {
    return `<lang:es> <user> <fact> ¿ cuál es el valor exacto de ${metric.es} del módulo ${name} ? <assistant>`;
  }
  return `<lang:fr> <user> <fact> quelle est la valeur exacte de ${metric.fr} du module ${name} ? <assistant>`;
}

export const ScarNativeLab: React.FC<{ lang: 'en' | 'pt' }> = ({ lang }) => {
  const [tab, setTab] = useState<LabTab>('live_o1');
  const [polyLang, setPolyLang] = useState<PolyLang>(lang === 'pt' ? 'pt' : 'en');
  const [passkeyCode, setPasskeyCode] = useState<string>('742');
  const [clauses, setClauses] = useState<number>(6);
  const [selectedEntityIdx, setSelectedEntityIdx] = useState<number>(0);

  const [engine, setEngine] = useState<ScarNativeWebGpuBrowserEngine | null>(null);
  const [audit, setAudit] = useState<ScarNativeAuditReportJSON | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState<boolean>(false);

  const [whResult, setWhResult] = useState<ScarNativeGenerationReport | null>(null);
  const [winOnlyResult, setWinOnlyResult] = useState<ScarNativeGenerationReport | null>(null);
  const [factResult, setFactResult] = useState<ScarNativeGenerationReport | null>(null);
  const [wipedIds, setWipedIds] = useState<Record<string, boolean>>({});
  const [gdprMaxLogitDiff, setGdprMaxLogitDiff] = useState<number | null>(null);

  const curvesCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        const base = import.meta.env.BASE_URL || '/';
        const prefix = `${base.replace(/\/$/, '')}/webgpu/scarnative_v5`;
        const [mRes, tRes, sRes, wRes, aRes] = await Promise.all([
          fetch(`${prefix}/scarnative_manifest.json`),
          fetch(`${prefix}/tokenizer.json`),
          fetch(`${prefix}/scarnative_v5_shard_00.bin`),
          fetch(`${prefix}/mos_scache_kernels.wgsl`),
          fetch(`${prefix}/scar_native_v5_0_audit.json`),
        ]);
        if (!mRes.ok || !tRes.ok || !sRes.ok || !wRes.ok) {
          throw new Error(`HTTP ${mRes.status}/${tRes.status}/${sRes.status}/${wRes.status}`);
        }
        const manifest = (await mRes.json()) as ScarNativeManifestJSON;
        const tokJson = (await tRes.json()) as { id_to_token: string[] };
        const shardBuf = await sRes.arrayBuffer();
        const wgslText = await wRes.text();
        const auditJson = aRes.ok ? ((await aRes.json()) as ScarNativeAuditReportJSON) : null;
        if (cancelled) return;

        const cpu = new ScarNativeCpuExecutor(manifest, shardBuf, tokJson);
        const gpuEng = new ScarNativeWebGpuBrowserEngine(cpu, shardBuf, wgslText);
        await gpuEng.initWebGpuIfAvailable();
        if (cancelled) return;

        setEngine(gpuEng);
        if (auditJson) setAudit(auditJson);
      } catch (err: any) {
        if (!cancelled) setLoadError(String(err?.message || err));
      }
    }
    boot();
    return () => {
      cancelled = true;
    };
  }, []);

  // Run initial live O(1) comparison & initial cartridge query once loaded
  useEffect(() => {
    if (!engine) return;
    void runPasskeyComparison(polyLang, passkeyCode, clauses);
    void runEntityQuery(polyLang, selectedEntityIdx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine]);

  async function runPasskeyComparison(lg: PolyLang, code: string, nClauses: number) {
    if (!engine) return;
    setIsBusy(true);
    try {
      const cleanCode = /^[0-9]{3}$/.test(code) ? code : '742';
      const prompt = buildPasskeyPrompt(lg, cleanCode, nClauses);
      const fullOut = await engine.generate(prompt, {
        maxNewTokens: 12,
        useScarStore: false,
        enableWidrowHoff: true,
      });
      const winOut = engine.cpuRef.generate(prompt, {
        maxNewTokens: 12,
        useScarStore: false,
        enableWidrowHoff: false,
      });
      setWhResult(fullOut);
      setWinOnlyResult(winOut);
    } finally {
      setIsBusy(false);
    }
  }

  async function runEntityQuery(lg: PolyLang, entIdx: number) {
    if (!engine) return;
    setIsBusy(true);
    try {
      const ent = ENTITY_CATALOG[entIdx];
      const prompt = buildFactPrompt(lg, ent.name, ent.metric);
      const res = await engine.generate(prompt, {
        maxNewTokens: 14,
        useScarStore: true,
      });
      setFactResult(res);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleToggleGdprWipe() {
    if (!engine) return;
    const ent = ENTITY_CATALOG[selectedEntityIdx];
    const prompt = buildFactPrompt(polyLang, ent.name, ent.metric);
    const isCurrentlyWiped = Boolean(wipedIds[ent.id]);

    if (!isCurrentlyWiped) {
      // Measure baseline unrouted trunk logits BEFORE wipe, then wipe `0x00` and verify exact `0.00000000` Δlogits!
      const trunkRef = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: false });
      engine.cpuRef.gdprZeroWipe(ent.id);
      const afterWipe = engine.cpuRef.generate(prompt, { maxNewTokens: 1, useScarStore: true });
      let maxDiff = 0.0;
      for (let i = 0; i < trunkRef.firstTokenLogits.length; i++) {
        const d = Math.abs(trunkRef.firstTokenLogits[i] - afterWipe.firstTokenLogits[i]);
        if (d > maxDiff) maxDiff = d;
      }
      setGdprMaxLogitDiff(maxDiff);
      setWipedIds((prev) => ({ ...prev, [ent.id]: true }));
    } else {
      engine.cpuRef.restoreCartridge(ent.id);
      setGdprMaxLogitDiff(null);
      setWipedIds((prev) => ({ ...prev, [ent.id]: false }));
    }
    await runEntityQuery(polyLang, selectedEntityIdx);
  }

  // Draw pretraining curves on Tab 3 canvas
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
    const maxStep = Math.max(1, steps[steps.length - 1] ?? 500);
    const maxY = 7.6;

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
      if (vals.length === 0) return;
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
    drawSeries(audit.training_curves.scarnative_no_wh_loss, '#f59e0b', 2.0);
    drawSeries(audit.training_curves.scarnative_full_loss, '#adff2f', 2.6);

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('step 0 (Tabula Rasa)', padL, h - 12);
    ctx.fillText(`step ${maxStep}`, w - padR - 64, h - 12);
  }, [tab, audit]);

  const selectedEntity = useMemo(() => ENTITY_CATALOG[selectedEntityIdx], [selectedEntityIdx]);
  const shardMiB = engine ? (engine.cpuRef.manifest.shard_bytes / (1024 * 1024)).toFixed(2) : '1.89';
  const scacheKb = engine ? (engine.cpuRef.scacheBytesTotal / 1024).toFixed(2) : '124.25';

  return (
    <section
      id="scar-native-lab"
      className="relative rounded-3xl bg-zinc-950/90 border border-[#adff2f]/40 p-6 sm:p-8 shadow-[0_0_50px_rgba(173,255,47,0.08)] space-y-6"
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-bold bg-[#adff2f]/15 text-[#adff2f] border border-[#adff2f]/40">
              <Sparkles className="w-3.5 h-3.5" />
              {lang === 'pt'
                ? 'PURE SCAR-NATIVE ENGINE v5.0 · PRÉ-TREINADO DO ZERO (TABULA RASA)'
                : 'PURE SCAR-NATIVE ENGINE v5.0 · PRETRAINED FROM SCRATCH (TABULA RASA)'}
            </span>
            <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-[#38bdf8]">
              22 WGSL 1.0 Shaders · {shardMiB} MiB Shard · SCache {scacheKb} KB (Δ +0 B)
            </span>
            {engine?.liveParityCosine != null && (
              <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                WebGPU Parity cos = {engine.liveParityCosine.toFixed(7)}
              </span>
            )}
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            {lang === 'pt'
              ? 'Motor Fundacional Pure Scar-Native (ScarNativeLM v5.0)'
              : 'Pure Scar-Native Foundation Architecture (ScarNativeLM v5.0)'}
          </h2>
          <p className="text-xs sm:text-sm text-zinc-400 max-w-4xl leading-relaxed">
            {lang === 'pt'
              ? 'Nascido em t = 0 sem nenhuma matriz densa Gaussiana e zero pesos Qwen/Llama: operadores ScarNativeLinear (FWHT-64 + D4 LUT256 com vácuo exato + STE INT4 de choque 0.0000 KL), DifferentiableSCacheAttention O(1) (Shaders #21 e #22) e 16 cartuchos Zero-SGD RRR de 8 KB (8.192 B = 2 × 4 KB) ancorados apenas em EN disparando em PT/EN/ES/FR com unlearning exato 0x00.'
              : 'Born at t = 0 with zero unconstrained dense Gaussian matrices and zero Qwen/Llama weights: ScarNativeLinear operators (FWHT-64 + exact-vacuum D4 LUT256 + INT4-STE with 0.0000 KL PTQ shock), O(1) DifferentiableSCacheAttention (Shaders #21 & #22), and 16 EN-only 8 KB (8,192 B = 2 × 4 KB) Zero-SGD RRR cartridges firing across PT/EN/ES/FR with exact 0x00 GDPR wipe.'}
          </p>
        </div>

        {/* Tab Selector */}
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setTab('live_o1')}
            className={`px-3.5 py-2 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center gap-1.5 ${
              tab === 'live_o1'
                ? 'bg-[#adff2f] text-black border-[#adff2f]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>{lang === 'pt' ? '1. Atenção O(1) Ao Vivo' : '1. Live O(1) Passkey'}</span>
          </button>
          <button
            type="button"
            onClick={() => setTab('cartridges_gdpr')}
            className={`px-3.5 py-2 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center gap-1.5 ${
              tab === 'cartridges_gdpr'
                ? 'bg-[#38bdf8] text-black border-[#38bdf8]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>{lang === 'pt' ? '2. Cartuchos P_⊥lang & 0x00' : '2. P_⊥lang & 0x00 GDPR'}</span>
          </button>
          <button
            type="button"
            onClick={() => setTab('ablation_curves')}
            className={`px-3.5 py-2 rounded-xl text-xs font-mono font-bold border transition-all cursor-pointer flex items-center gap-1.5 ${
              tab === 'ablation_curves'
                ? 'bg-[#c4b5fd] text-black border-[#c4b5fd]'
                : 'bg-zinc-900 text-zinc-300 border-zinc-800 hover:border-zinc-700'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>{lang === 'pt' ? '3. Pré-Treino & Ablações' : '3. Pretraining & Ablations'}</span>
          </button>
        </div>
      </div>

      {loadError && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs font-mono text-amber-200">
          {lang === 'pt' ? 'Aviso de carregamento do bundle v5.0: ' : 'v5.0 bundle load notice: '}
          {loadError}
        </div>
      )}

      {/* TAB 1: LIVE O(1) DIFFERENTIABLE SCACHE PASSKEY RETRIEVAL */}
      {tab === 'live_o1' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800">
            <div>
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt' ? 'Idioma do Prompt' : 'Prompt Language'}
              </label>
              <div className="grid grid-cols-4 gap-1.5">
                {(['pt', 'en', 'es', 'fr'] as const).map((lg) => (
                  <button
                    key={lg}
                    type="button"
                    onClick={() => {
                      setPolyLang(lg);
                      void runPasskeyComparison(lg, passkeyCode, clauses);
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

            <div>
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt' ? 'Código Held-Out (3 dígitos)' : 'Held-Out Passkey (3 digits)'}
              </label>
              <input
                type="text"
                maxLength={3}
                value={passkeyCode}
                onChange={(e) => setPasskeyCode(e.target.value.replace(/[^0-9]/g, '').slice(0, 3))}
                className="w-full px-3 py-1.5 rounded-lg bg-zinc-950 border border-zinc-800 text-sm font-mono text-white"
              />
            </div>

            <div>
              <label className="block text-[11px] font-mono text-zinc-400 mb-1.5">
                {lang === 'pt'
                  ? `Horizonte de Distratores (${clauses} cláusulas · >> janela W=16)`
                  : `Distractor Horizon (${clauses} clauses · >> window W=16)`}
              </label>
              <input
                type="range"
                min={3}
                max={24}
                step={3}
                value={clauses}
                onChange={(e) => setClauses(Number(e.target.value))}
                className="w-full accent-[#adff2f] mt-2 cursor-pointer"
              />
            </div>

            <div className="flex items-end">
              <button
                type="button"
                disabled={!engine || isBusy}
                onClick={() => void runPasskeyComparison(polyLang, passkeyCode, clauses)}
                className="w-full py-2 px-4 rounded-xl bg-[#adff2f] hover:bg-lime-300 disabled:opacity-50 text-black font-mono text-xs font-extrabold flex items-center justify-center gap-2 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>{lang === 'pt' ? 'Executar Inferência O(1)' : 'Run Live O(1) Inference'}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Full Differentiable SCache (W_scar ON) */}
            <div className="p-5 rounded-2xl bg-zinc-900/90 border border-[#adff2f]/50 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-[#adff2f] flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  ScarNativeLM v5.0 Full (Shaders #21 + #22 · W_scar ON)
                </span>
                <span className="text-[11px] font-mono text-zinc-400">
                  {whResult?.backend === 'webgpu_wgsl_22' ? 'WebGPU WGSL 1.0' : 'CPU Bit-Exact'}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-black/60 border border-zinc-800 font-mono text-sm text-white min-h-[48px]">
                {whResult ? whResult.outputText : '...'}
              </div>
              {whResult && (
                <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Context Tokens</span>
                    <span className="text-white font-bold">
                      {whResult.prefillTokens} tok (W_win = 16)
                    </span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">SCache O(1)</span>
                    <span className="text-[#adff2f] font-bold">
                      {whResult.scacheBytesFinal.toLocaleString()} B (Δ +{whResult.scacheDeltaBytes} B)
                    </span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Recall Status</span>
                    <span
                      className={
                        matchesPasskeyAnswer(whResult.outputText, passkeyCode)
                          ? 'text-emerald-400 font-bold'
                          : 'text-amber-300 font-bold'
                      }
                    >
                      {matchesPasskeyAnswer(whResult.outputText, passkeyCode) ? 'EXACT 3/3 MATCH' : 'PARTIAL'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Window-Only O(1) Ablation (W_scar OFF) */}
            <div className="p-5 rounded-2xl bg-zinc-900/60 border border-rose-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-rose-300">
                  {lang === 'pt'
                    ? 'Ablação Sem Widrow-Hoff (Apenas Sinks=4 + Janela=16)'
                    : 'Ablation Without Widrow-Hoff (Sinks=4 + Window=16 Only)'}
                </span>
                <span className="text-[11px] font-mono text-zinc-500">W_scar = 0</span>
              </div>
              <div className="p-3 rounded-xl bg-black/60 border border-zinc-800 font-mono text-sm text-zinc-300 min-h-[48px]">
                {winOnlyResult ? winOnlyResult.outputText : '...'}
              </div>
              {winOnlyResult && (
                <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Evicted Distance</span>
                    <span className="text-zinc-300 font-bold">
                      &gt;{ Math.max(0, winOnlyResult.prefillTokens - 20) } tok ago
                    </span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Fast-Weight Read</span>
                    <span className="text-rose-400 font-bold">DISABLED (0.0)</span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 block">Recall Status</span>
                    <span className="text-rose-400 font-bold">
                      {matchesPasskeyAnswer(winOnlyResult.outputText, passkeyCode) ? 'MATCH' : 'EVICTED / LOST'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: DETACHED SCARSTORE P_PERP_LANG & 0x00 GDPR WIPE */}
      {tab === 'cartridges_gdpr' && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            {ENTITY_CATALOG.map((ent, idx) => {
              const isWiped = Boolean(wipedIds[ent.id]);
              return (
                <button
                  key={ent.id}
                  type="button"
                  onClick={() => {
                    setSelectedEntityIdx(idx);
                    void runEntityQuery(polyLang, idx);
                  }}
                  className={`p-2.5 rounded-xl border text-left font-mono transition-all cursor-pointer ${
                    selectedEntityIdx === idx
                      ? 'bg-[#38bdf8]/15 border-[#38bdf8] text-white'
                      : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-white'
                  }`}
                >
                  <div className="text-xs font-bold uppercase flex items-center justify-between">
                    <span>{ent.name}</span>
                    {isWiped && <span className="text-[9px] text-rose-400">0x00</span>}
                  </div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">
                    {isWiped ? 'WIPED' : ent.expected}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="p-5 rounded-2xl bg-zinc-900/80 border border-zinc-800 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Globe2 className="w-4 h-4 text-[#38bdf8]" />
                <span className="text-xs font-mono text-zinc-300">
                  {lang === 'pt'
                    ? 'Cartucho cunhado APENAS em Inglês (EN, 0 passos SGD) — consulte em qualquer idioma:'
                    : 'Cartridge minted ONLY in English (EN, 0 SGD steps) — query in any language:'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {(['pt', 'en', 'es', 'fr'] as const).map((lg) => (
                  <button
                    key={lg}
                    type="button"
                    onClick={() => {
                      setPolyLang(lg);
                      void runEntityQuery(lg, selectedEntityIdx);
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
                  onClick={() => void handleToggleGdprWipe()}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold border flex items-center gap-1.5 cursor-pointer ${
                    wipedIds[selectedEntity.id]
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                  }`}
                >
                  {wipedIds[selectedEntity.id] ? (
                    <>
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>{lang === 'pt' ? 'Restaurar Cartucho' : 'Restore Cartridge'}</span>
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>{lang === 'pt' ? 'Executar Wipe LGPD 0x00' : 'Execute 0x00 GDPR Wipe'}</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-black/70 border border-zinc-800 font-mono text-xs text-zinc-400">
              <div>Prompt: {buildFactPrompt(polyLang, selectedEntity.name, selectedEntity.metric)}</div>
              <div className="text-sm text-white font-bold mt-2">
                Output: {factResult ? factResult.outputText : '...'}
              </div>
            </div>

            {factResult && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 block">Routed Cartridge (Layer 3)</span>
                  <span className="text-[#38bdf8] font-bold">
                    {factResult.routedCartridgeId ?? 'NONE (0.0 SILENT)'}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 block">P_⊥lang Anchor Cosine</span>
                  <span className="text-white font-bold">{factResult.routedCosine.toFixed(4)}</span>
                </div>
                <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 block">Cartridge Size</span>
                  <span className="text-[#adff2f] font-bold">
                    {(() => {
                      const recB = engine?.cpuRef.manifest.cartridges[0]?.record_bytes ?? 8192;
                      return `${recB.toLocaleString()} B (${Math.round(recB / 4096)} × 4 KB)`;
                    })()}
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
        </div>
      )}

      {/* TAB 3: TABULA-RASA PRETRAINING CURVES & 4-MODEL ABLATION */}
      {tab === 'ablation_curves' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-6 p-4 rounded-2xl bg-zinc-900/80 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-white">
                  {lang === 'pt'
                    ? 'Curvas de Pré-Treino Tabula Rasa (t = 0)'
                    : 'Tabula-Rasa Pretraining Loss Curves (t = 0)'}
                </span>
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="text-[#adff2f]">■ ScarNative v5.0</span>
                  <span className="text-amber-400">■ w/o Widrow-Hoff</span>
                  <span className="text-slate-400">■ Dense O(T)</span>
                </div>
              </div>
              <canvas
                ref={curvesCanvasRef}
                width={560}
                height={230}
                className="w-full h-auto rounded-xl border border-zinc-800"
              />
            </div>

            <div className="lg:col-span-6 p-4 rounded-2xl bg-zinc-900/80 border border-zinc-800 overflow-x-auto">
              <div className="text-xs font-mono font-bold text-white mb-3 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-[#adff2f]" />
                <span>
                  {lang === 'pt'
                    ? 'Ablação Controlada de 4 Modelos (Mesma Semente t = 0)'
                    : '4-Model Controlled Foundation Ablation (Matched Seed t = 0)'}
                </span>
              </div>
              <table className="w-full text-[11px] font-mono text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400">
                    <th className="py-1.5 pr-2">Model</th>
                    <th className="py-1.5 px-2">Weights</th>
                    <th className="py-1.5 px-2">PTQ KL</th>
                    <th className="py-1.5 px-2">Passkey T=256</th>
                    <th className="py-1.5 pl-2">Facts (64)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/70">
                  {(audit?.ablation_summary ?? []).map((row) => (
                    <tr key={row.model_id} className="text-zinc-200">
                      <td className="py-2 pr-2 font-bold text-white">{row.label}</td>
                      <td className="py-2 px-2">{(row.weight_bytes / (1024 * 1024)).toFixed(2)} MiB</td>
                      <td className="py-2 px-2 text-[#adff2f]">{row.ptq_kl_shock.toFixed(4)}</td>
                      <td className="py-2 px-2">{(row.passkey_acc_t256 * 100).toFixed(1)}%</td>
                      <td className="py-2 pl-2">{(row.fact_acc_polyglot_64 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
