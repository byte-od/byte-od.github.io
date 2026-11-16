import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { ProjectShowcase } from './components/ProjectShowcase';
import { FaqSection } from './components/FaqSection';
import { ContactPromptConsole } from './components/ContactPromptConsole';
import { TerminalModal } from './components/TerminalModal';
import { AgentViewModal } from './components/AgentViewModal';
import { VisitorTelemetryBadge } from './components/VisitorTelemetryBadge';
import { HostIntegrityBanner } from './components/HostIntegrityBanner';
import { PortfolioAgentWidget } from './components/PortfolioAgentWidget';
import { PhysicalScarLab } from './components/PhysicalScarLab';
import { ScarNativeLab } from './components/ScarNativeLab';
import {
  Github,
  Mail,
  Linkedin,
  ExternalLink,
  Terminal,
  Cpu,
  Radio,
  Bot,
  Blocks,
  PlayCircle,
  Sparkles,
  Brain,
  FileText,
} from 'lucide-react';

export default function App() {
  const [lang, setLang] = useState<'en' | 'pt'>(() => {
    if (typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('pt')) {
      return 'pt';
    }
    return 'en';
  });
  const [isTerminalOpen, setIsTerminalOpen] = useState<boolean>(false);
  const [isAgentModalOpen, setIsAgentModalOpen] = useState<boolean>(false);
  const [agentModalInitialTab, setAgentModalInitialTab] = useState<'llmstxt' | 'jsonld' | 'prompt' | 'mcp'>('llmstxt');

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.altKey && (e.key === 'a' || e.key === 'A')) || (e.ctrlKey && e.shiftKey && (e.key === 'a' || e.key === 'A'))) {
        e.preventDefault();
        setIsAgentModalOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-[#07080a] text-zinc-100 selection:bg-[#adff2f] selection:text-black">
      {/* Host Integrity Banner (renders only on unauthorized mirrors) */}
      <HostIntegrityBanner />

      {/* Subtle Background Grid Texture */}
      <div className="fixed inset-0 pointer-events-none bg-grid-pattern opacity-40 z-0" />

      {/* Navigation */}
      <Navbar
        lang={lang}
        setLang={setLang}
        onOpenTerminal={() => setIsTerminalOpen(true)}
        onOpenAgentView={() => setIsAgentModalOpen(true)}
      />

      {/* Main Content */}
      <main className="relative z-10 pt-28 pb-24 px-6 lg:px-12 max-w-7xl mx-auto space-y-28">
        {/* HERO / OVERVIEW SECTION */}
        <section id="overview" className="pt-6 pb-8 scroll-mt-24">
          <div className="flex items-center gap-2.5 text-xs font-mono text-zinc-400 mb-6 flex-wrap">
            <span className="text-[#adff2f] font-bold">
              {lang === 'pt'
                ? 'ECOSSISTEMA PROPRIETÁRIO · MIXTURE OF SCARS (MoS) & SCACHE O(1)'
                : 'PROPRIETARY ECOSYSTEM · MIXTURE OF SCARS (MoS) & SCACHE O(1)'}
            </span>
            <span aria-hidden="true">·</span>
            <span>Zenodo DOI 10.5281/zenodo.24172784</span>
            <span aria-hidden="true">·</span>
            <VisitorTelemetryBadge
              lang={lang}
              onOpenAgentModal={() => setIsAgentModalOpen(true)}
            />
          </div>

          <div className="space-y-4 mb-6">
            <p className="text-xs sm:text-sm font-mono uppercase tracking-widest text-[#38bdf8]">
              Julio Cesar da Costa Reis Filho
            </p>
            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-[1.08] max-w-5xl">
              {lang === 'pt'
                ? 'Arquitetura de Cicatrizes: Memória O(1) Constante, Streaming Sub-Linear & Síntese 3D Contínua.'
                : 'The Scar Architecture: Constant O(1) Memory, Sub-Linear Streaming & Continuous 3D Synthesis.'}
            </h1>
          </div>

          <div className="max-w-3xl border-l-2 border-[#adff2f]/60 pl-6 my-7">
            <p className="text-base sm:text-lg text-zinc-300 leading-relaxed font-sans">
              {lang === 'pt'
                ? 'Pesquisa aberta e motores WebGPU/PyTorch reais que substituem o KV Cache linear O(T) e matrizes densas monolíticas por decomposição vetorial em cicatrizes: Mixture of Scars (MoS), SCache Associativo O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0 (Tabula Rasa Shared-Stage MoS), PointScarFlow v1.0 e AURA / System One.'
                : 'Open research and live WebGPU/PyTorch engines replacing linear O(T) KV Cache and monolithic dense matrices with directional scar decomposition: Mixture of Scars (MoS), Associative O(1) SCache (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0 (Tabula Rasa Shared-Stage MoS), PointScarFlow v1.0, and AURA / System One.'}
            </p>
          </div>

          {/* 4 Verified Scar Proof Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 py-6 border-y border-zinc-800/80 my-8">
            <div>
              <div className="text-3xl font-bold font-mono text-[#adff2f] tabular-nums">2.40 GB</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt'
                  ? 'Pico de VRAM em 72.7B (80 Camadas MoS)'
                  : '72.7B Peak VRAM (80-Layer MoS Stream)'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-[#38bdf8] tabular-nums">34.8 KB</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt'
                  ? 'SCache O(1) Constante (Δ +0 B vs 32.8 GB)'
                  : 'Constant O(1) SCache (Δ +0 B vs 32.8 GB)'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-[#c4b5fd] tabular-nums">0.0000 KL</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt'
                  ? 'Choque PTQ no ScarNativeLM v6.0 (VQ4-STE)'
                  : 'PTQ Shock in ScarNativeLM v6.0 (VQ4-STE)'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-emerald-400 tabular-nums">38 WGSL</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt'
                  ? 'Compute Shaders no Browser (26 LLM + 12 3D)'
                  : 'Browser Compute Shaders (26 LLM + 12 3D)'}
              </div>
            </div>
          </div>

          {/* Primary Action Bar */}
          <div className="flex flex-wrap items-center gap-3.5 pt-2">
            <button
              type="button"
              data-testid="hero-open-jarvis-btn"
              onClick={() => {
                window.dispatchEvent(new CustomEvent('open-jarvis-agent'));
              }}
              className="px-5 py-2.5 bg-[#adff2f] hover:bg-lime-300 text-black text-xs font-mono font-extrabold rounded-lg transition-all flex items-center gap-2 shadow-[0_0_24px_rgba(173,255,47,0.3)] hover:shadow-[0_0_34px_rgba(173,255,47,0.5)] cursor-pointer"
              title={
                lang === 'pt'
                  ? 'Focar JARVIS Qwen 3B + PointScarFlow 3D (38 Shaders WGSL + SCache O(1))'
                  : 'Focus JARVIS Qwen 3B + PointScarFlow 3D (38 WGSL Shaders + SCache O(1))'
              }
            >
              <Bot className="w-4 h-4" />
              <span>
                {lang === 'pt'
                  ? 'JARVIS + PointScarFlow 3D (Qwen 3B · 38 WGSL)'
                  : 'JARVIS + PointScarFlow 3D (Qwen 3B · 38 WGSL)'}
              </span>
            </button>

            <a
              href="#scar-native-lab"
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-[#adff2f] text-xs font-mono font-bold rounded-lg border border-[#adff2f]/50 hover:border-[#adff2f] transition-all flex items-center gap-2 cursor-pointer"
            >
              <Cpu className="w-4 h-4" />
              <span>Scar-Native Lab v6.0</span>
            </a>

            <a
              href="#physical-scar-lab"
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-white text-xs font-mono rounded-lg border border-zinc-800 hover:border-[#38bdf8]/60 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Radio className="w-4 h-4 text-[#38bdf8]" />
              <span>Physical Scar Lab v4.0</span>
            </a>

            <a
              href="#aura-origin"
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-zinc-200 hover:text-white text-xs font-mono rounded-lg border border-zinc-800 hover:border-purple-400/60 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Brain className="w-4 h-4 text-purple-400" />
              <span>{lang === 'pt' ? 'AURA (Origem 34.8 KB)' : 'AURA (34.8 KB Origin)'}</span>
            </a>

            <button
              type="button"
              onClick={() => {
                setAgentModalInitialTab('llmstxt');
                setIsAgentModalOpen(true);
              }}
              className="px-4 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-zinc-200 hover:text-white text-xs font-mono rounded-lg border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Blocks className="w-4 h-4 text-[#adff2f]" />
              <span>llms.txt &amp; MCP</span>
            </button>

            <a
              href="https://github.com/byte-od/mixture-of-scars"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-white rounded-lg border border-zinc-800 transition-colors"
              title="GitHub · byte-od/mixture-of-scars"
            >
              <Github className="w-4 h-4" />
            </a>

            <a
              href="mailto:reisfilho1116@gmail.com"
              className="p-2.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-[#adff2f] rounded-lg border border-zinc-800 transition-colors"
              title="reisfilho1116@gmail.com"
            >
              <Mail className="w-4 h-4" />
            </a>
          </div>

          {/* Flagship Unified Architecture & PointScarFlow v1.0 Card */}
          <div className="relative mt-10 p-6 sm:p-8 rounded-2xl bg-gradient-to-br from-zinc-900/95 via-zinc-950 to-black border-2 border-[#adff2f]/70 shadow-[0_0_40px_rgba(173,255,47,0.14)] overflow-hidden">
            <div className="absolute top-0 right-0 -mr-20 -mt-20 w-72 h-72 bg-[#adff2f]/10 rounded-full blur-3xl pointer-events-none" />

            <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
              <div className="space-y-3 max-w-3xl">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-bold bg-[#adff2f]/15 text-[#adff2f] border border-[#adff2f]/40 shadow-xs">
                    <Sparkles className="w-3.5 h-3.5 animate-pulse" />
                    {lang === 'pt'
                      ? 'NOVO · SCARNATIVE v6.0 (SHARED-STAGE MoS) + POINTSCARFLOW v1.0 (3D OT FLOW)'
                      : 'NEW · SCARNATIVE v6.0 (SHARED-STAGE MoS) + POINTSCARFLOW v1.0 (3D OT FLOW)'}
                  </span>
                  <a
                    href="https://doi.org/10.5281/zenodo.24172784"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-[11px] font-mono text-[#38bdf8] hover:underline bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800"
                  >
                    <FileText className="w-3 h-3" />
                    <span>DOI 10.5281/zenodo.24172784</span>
                  </a>
                </div>

                <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight leading-tight">
                  {lang === 'pt'
                    ? 'Do Latente P_⊥lang da Camada 23 a Nuvens 3D Contínuas em 86 ms na WebGPU'
                    : 'From Layer-23 P_⊥lang Latents to Continuous 3D Point Manifolds in 86 ms on WebGPU'}
                </h2>

                <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed">
                  {lang === 'pt'
                    ? 'Quando você conversa com o JARVIS (Qwen2.5-3B Track 2.5 ProtectedInt4Scar + SCache O(1)), o estado oculto invariante a idioma na Camada 23 (P_⊥lang) alimenta diretamente o PointScarFlow v1.0 (11.85M params ScarNativeLinear em 3.87 MiB, zero nn.Linear, zero catálogo pré-definido), integrando 1.024 pontos 3D em tela cheia via 12 compute shaders WGSL com cosseno multilíngue > 0.997 (PT/EN/ES/FR).'
                    : 'When you speak to JARVIS (Qwen2.5-3B Track 2.5 ProtectedInt4Scar + O(1) SCache), the language-invariant hidden state at Layer 23 (P_⊥lang) directly conditions PointScarFlow v1.0 (11.85M ScarNativeLinear params in 3.87 MiB, zero nn.Linear, zero preset catalog), integrating 1,024 full-viewport 3D points via 12 WGSL compute shaders with > 0.997 cross-lingual shape cosine (PT/EN/ES/FR).'}
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">PointScarFlow Shard</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#adff2f]">3.87 MiB (11.85M)</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">4-Step ODE WebGPU</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#38bdf8]">86.45 ms · 7.51x OT</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">0x00 GDPR Wipe</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-emerald-400">&lt; 0.2 ms · Δ=0.0</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">Kaggle Benchmark</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#c4b5fd]">0.625 (MoS ≤ 2.2 GB)</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row lg:flex-col gap-3 w-full lg:w-auto shrink-0">
                <a
                  href="https://github.com/byte-od/mixture-of-scars"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-3 rounded-xl bg-[#adff2f] hover:bg-lime-300 text-black font-mono text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-lime-400/20 cursor-pointer"
                >
                  <Github className="w-4 h-4" />
                  <span>{lang === 'pt' ? 'Repositório mixture-of-scars' : 'mixture-of-scars Repository'}</span>
                </a>

                <a
                  href="https://doi.org/10.5281/zenodo.24172784"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-[#38bdf8] font-mono text-xs font-bold border border-[#38bdf8]/40 transition-all flex items-center justify-center gap-2 cursor-pointer"
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>{lang === 'pt' ? 'Papers Zenodo (v1 – v6)' : 'Zenodo Papers (v1 – v6)'}</span>
                </a>

                <a
                  href="https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/MoS_SCache_72B_Live_Test.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-white font-mono text-xs border border-zinc-800 transition-all flex items-center justify-center gap-2 cursor-pointer"
                >
                  <PlayCircle className="w-3.5 h-3.5 text-[#adff2f]" />
                  <span>{lang === 'pt' ? 'Notebook 72.7B no Colab' : '72.7B Colab Notebook'}</span>
                </a>

                <a
                  href="https://www.kaggle.com/competitions/name-that-llm/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2 rounded-xl bg-zinc-950 hover:bg-zinc-900 text-zinc-400 hover:text-white font-mono text-[11px] border border-zinc-850 transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <span>Kaggle Leaderboard (0.625)</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* CHAPTER 01: PURE SCAR-NATIVE FOUNDATION ENGINE LAB (v5.0) */}
        <ScarNativeLab lang={lang} />

        {/* CHAPTER 02: DOMAIN-INDEPENDENT PHYSICAL SCAR LAB (v4.0) */}
        <PhysicalScarLab lang={lang} />

        {/* CHAPTER 03: ORIGIN STORY OF THE SCAR OPERATOR — AURA / SYSTEM ONE */}
        <section id="aura-origin" className="space-y-6 scroll-mt-24">
          <div className="space-y-2 border-b border-zinc-800 pb-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-mono text-[#c4b5fd] uppercase tracking-wider">
                {lang === 'pt'
                  ? '03. História & Origem · Onde Nasceu o Operador Cicatriz (34,8 KB O(1))'
                  : '03. Origin Story · Where the Scar Operator Was Born (34.8 KB O(1))'}
              </span>
              <a
                href="https://github.com/byte-od/system_one"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs font-mono text-zinc-400 hover:text-white inline-flex items-center gap-1.5"
              >
                <Github className="w-3.5 h-3.5" />
                <span>github.com/byte-od/system_one</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
            <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
              {lang === 'pt'
                ? 'O Nascimento do Operador Cicatriz (AURA / System One)'
                : 'The Birth of the Scar Operator (AURA / System One)'}
            </h2>
          </div>

          <div className="p-6 sm:p-8 rounded-3xl bg-zinc-950/85 border border-zinc-800 space-y-5">
            <p className="text-sm sm:text-base text-zinc-200 leading-relaxed">
              {lang === 'pt'
                ? 'A matemática de cicatrizes não começou tentando comprimir modelos de linguagem prontos. Ela nasceu no projeto AURA / System One, ao investigar uma pergunta simples: por que sistemas de IA acumulam gigabytes de histórico repetitivo e ainda assim esquecem o único evento raro que realmente importava? Na biologia e na física, um sistema eficiente não grava cada segundo da própria rotina — ele mantém sua estrutura leve e registra apenas "cicatrizes" permanentes onde sofreu um choque real de surpresa.'
                : 'Scar mathematics did not begin as an attempt to compress off-the-shelf language models. It was born inside AURA / System One while investigating a fundamental question: why do AI systems accumulate gigabytes of repetitive history yet still forget the single rare shock that actually mattered? Physical and biological systems do not record every routine second — they keep a lightweight substrate and etch permanent "scars" only where an unexpected prediction shock strikes.'}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800 space-y-1.5">
                <div className="font-mono font-bold text-[#c4b5fd] uppercase">
                  {lang === 'pt'
                    ? '1. GradientScarMemory (34,8 KB) → SCache O(1)'
                    : '1. GradientScarMemory (34.8 KB) → O(1) SCache'}
                </div>
                <p className="text-zinc-400 leading-relaxed">
                  {lang === 'pt'
                    ? 'Substituímos históricos episódicos que cresciam sem fim por uma matriz associativa fixa de 34.816 bytes modulada pelo choque de surpresa temporal (Surpresa = R_0 − V(s_0)), ignorando estados triviais. É exatamente esse operador que hoje substitui o KV Cache linear O(T) em LLMs de 72.7B, no ScarNativeLM v6.0 e nas caixas-pretas espaciais do ScarCodec.'
                    : 'We replaced unbounded episodic buffers with a fixed 34,816-byte associative matrix gated by temporal prediction surprise (Surprise = R_0 − V(s_0)), filtering out routine states. That exact operator evolved into O(1) SCache across 72.7B LLMs, ScarNativeLM v6.0, and ScarCodec spacecraft black-boxes.'}
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800 space-y-1.5">
                <div className="font-mono font-bold text-[#adff2f] uppercase">
                  {lang === 'pt'
                    ? '2. Batedores Ghost-Jump → Desacoplamento'
                    : '2. Ghost-Jump Scouts → Decoupled Inference'}
                </div>
                <p className="text-zinc-400 leading-relaxed">
                  {lang === 'pt'
                    ? 'Para explorar bifurcações de alta incerteza sem poluir o núcleo reflexivo (~0,23 ms em CPU), criamos Batedores de Fronteira isolados que sondam passos à frente e gravam cicatrizes de choque — originando tanto a decodificação especulativa sem perdas (Ghost-Jump KL = 0,0000) quanto os cartuchos externos Zero-SGD.'
                    : 'To probe high-uncertainty bifurcations ahead in time without polluting the fast reflexive core (~0.23 ms CPU), we built isolated Frontier Ghost-Jump Scouts — directly inspiring both lossless Ghost-Jump speculative decoding (KL = 0.0000) and detached Zero-SGD knowledge cartridges.'}
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800 space-y-1.5">
                <div className="font-mono font-bold text-[#38bdf8] uppercase">
                  {lang === 'pt'
                    ? '3. Tri-Compass → Projetores P_⊥ & MoS'
                    : '3. Tri-Compass → P_⊥ Projectors & MoS'}
                </div>
                <p className="text-zinc-400 leading-relaxed">
                  {lang === 'pt'
                    ? 'A ideia de separar a estrutura invariante do ruído superficial (subtraindo o centro temático antes de ler a assinatura latente) evoluiu para os projetores de subespaço ortogonal P_⊥ — hoje responsáveis por alinhar PT/EN/ES/FR sem retrabalho, cancelar interferência em telescópios e guiar nuvens 3D no PointScarFlow.'
                    : 'Separating invariant structure from surface nuisance (centering cluster means to isolate latent signatures) evolved directly into orthogonal subspace projectors P_⊥ — now powering PT/EN/ES/FR cross-lingual routing, telescope interference nulling, and 3D PointScarFlow conditioning.'}
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CHAPTER 04: SCAR ARCHITECTURE MODULES & PUBLICATIONS */}
        <section id="scar-ecosystem" className="space-y-6 scroll-mt-24">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between border-b border-zinc-800 pb-4 gap-3">
            <div>
              <span className="text-xs font-mono text-[#adff2f] uppercase tracking-wider">
                {lang === 'pt'
                  ? '04. Pilares da Arquitetura de Cicatrizes & Código Aberto'
                  : '04. Scar Architecture Pillars & Open-Source Releases'}
              </span>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight mt-1">
                {lang === 'pt'
                  ? 'Motores, Codecs & Publicações Científicas'
                  : 'Engines, Codecs & Scientific Publications'}
              </h2>
            </div>
            <a
              href="https://github.com/byte-od/mixture-of-scars"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-zinc-400 hover:text-white flex items-center gap-1"
            >
              <span>github.com/byte-od/mixture-of-scars</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          <ProjectShowcase lang={lang} />
        </section>

        {/* CHAPTER 05: TECHNICAL DOSSIER & FAQ */}
        <FaqSection lang={lang} />

        {/* CHAPTER 06: AUTHOR IDENTITY & DIRECT CONTACT */}
        <section id="contact" className="p-8 sm:p-12 bg-zinc-950/80 border border-zinc-800 rounded-3xl relative overflow-hidden scroll-mt-24">
          <div className="relative z-10 max-w-4xl space-y-6">
            <span className="text-xs font-mono text-[#adff2f] uppercase tracking-wider">
              {lang === 'pt' ? '06. AUTORIA & CONTATO DIRETO' : '06. AUTHORSHIP & DIRECT CONTACT'}
            </span>
            <h2 className="text-2xl sm:text-4xl font-black text-white tracking-tight leading-tight">
              Julio Cesar da Costa Reis Filho
            </h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              {lang === 'pt'
                ? 'Criador e pesquisador responsável pela arquitetura Mixture of Scars (MoS), SCache Associativo O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0, PointScarFlow v1.0 e AURA / System One. Aberto a colaborações de pesquisa científica, implantação de motores MoS/SCache em produção (Edge / WebGPU / GPU Clusters) e liderança arquitetural em IA.'
                : 'Creator and principal researcher behind Mixture of Scars (MoS), Associative O(1) SCache (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0, PointScarFlow v1.0, and AURA / System One. Open to research collaborations, production MoS/SCache deployments (Edge / WebGPU / GPU Clusters), and AI architecture leadership.'}
            </p>

            <ContactPromptConsole
              lang={lang}
              onOpenTerminal={() => setIsTerminalOpen(true)}
            />

            <div className="flex flex-wrap items-center gap-4 pt-4 border-t border-zinc-800/80">
              <button
                type="button"
                onClick={() => setIsTerminalOpen(true)}
                className="px-6 py-3 bg-[#adff2f] text-black font-mono text-xs font-bold rounded-xl hover:bg-lime-300 transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-lime-400/10"
              >
                <Terminal className="w-4 h-4" />
                <span>{lang === 'pt' ? 'Abrir Scar Kernel CLI' : 'Launch Scar Kernel CLI'}</span>
              </button>

              <a
                href="mailto:reisfilho1116@gmail.com"
                className="px-6 py-3 bg-zinc-900 hover:bg-zinc-850 text-white font-mono text-xs rounded-xl border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2"
              >
                <Mail className="w-4 h-4 text-[#38bdf8]" />
                <span>reisfilho1116@gmail.com</span>
              </a>

              <a
                href="https://github.com/byte-od/mixture-of-scars"
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-zinc-900 hover:bg-zinc-850 text-zinc-200 hover:text-white font-mono text-xs rounded-xl border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2"
              >
                <Github className="w-4 h-4 text-[#adff2f]" />
                <span>byte-od/mixture-of-scars</span>
              </a>

              <a
                href="https://linkedin.com/in/juliocrfilho"
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-zinc-900 hover:bg-zinc-850 text-zinc-200 hover:text-white font-mono text-xs rounded-xl border border-zinc-800 hover:border-[#0a66c2]/50 transition-all flex items-center gap-2 group"
              >
                <Linkedin className="w-4 h-4 text-[#0a66c2] group-hover:text-[#38bdf8] transition-colors" />
                <span>LinkedIn /in/juliocrfilho</span>
              </a>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-zinc-900/80 py-8 px-6 lg:px-12 text-zinc-500 font-mono text-xs relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span>© 2026 Julio Cesar da Costa Reis Filho</span>
            <span aria-hidden="true">·</span>
            <span className="text-zinc-400">Mixture of Scars (MoS) &amp; SCache O(1)</span>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <button
              type="button"
              onClick={() => setIsAgentModalOpen(true)}
              aria-haspopup="dialog"
              className="text-[#adff2f] hover:text-white transition-colors cursor-pointer flex items-center gap-1.5 font-semibold"
            >
              <Bot className="w-3.5 h-3.5" />
              <span>{lang === 'pt' ? 'Dossiê IA (llms.txt)' : 'AI Dossier (llms.txt)'}</span>
            </button>
            <a
              href="https://doi.org/10.5281/zenodo.24172784"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#38bdf8] transition-colors"
            >
              Zenodo DOI
            </a>
            <a
              href="https://github.com/byte-od/mixture-of-scars"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#adff2f] transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://linkedin.com/in/juliocrfilho"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#38bdf8] transition-colors flex items-center gap-1"
            >
              <Linkedin className="w-3.5 h-3.5 text-[#0a66c2]" />
              <span>LinkedIn</span>
            </a>
            <a href="mailto:reisfilho1116@gmail.com" className="hover:text-[#adff2f] transition-colors">
              {lang === 'pt' ? 'E-mail' : 'Email'}
            </a>
            <button
              type="button"
              onClick={() => setIsTerminalOpen(true)}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Scar CLI
            </button>
          </div>
        </div>
      </footer>

      {/* Scar Kernel Terminal Modal */}
      <TerminalModal
        isOpen={isTerminalOpen}
        onClose={() => setIsTerminalOpen(false)}
        lang={lang}
      />

      {/* Structured AI Agent & LLMs.txt Modal */}
      <AgentViewModal
        isOpen={isAgentModalOpen}
        onClose={() => setIsAgentModalOpen(false)}
        lang={lang}
        initialTab={agentModalInitialTab}
      />

      {/* Sovereign JARVIS + PointScarFlow v1.0 3D Stage (Qwen2.5-3B · 34 WGSL Shaders · SCache O(1)) */}
      <PortfolioAgentWidget lang={lang} />
    </div>
  );
}
