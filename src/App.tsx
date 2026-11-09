import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Navbar } from './components/Navbar';
import { TokenInteractiveText, TokenAssemblyStage } from './components/TokenInteractiveText';
import { TokenizerPlayground } from './components/TokenizerPlayground';
import { CirEngineArchitecture } from './components/CirEngineArchitecture';
import { ProjectShowcase } from './components/ProjectShowcase';
import { InteractiveToolDemos } from './components/InteractiveToolDemos';
import { ExperienceTimeline } from './components/ExperienceTimeline';
import { FaqSection } from './components/FaqSection';
import { ContactPromptConsole } from './components/ContactPromptConsole';
import { TerminalModal } from './components/TerminalModal';
import { AgentViewModal } from './components/AgentViewModal';
import { VisitorTelemetryBadge } from './components/VisitorTelemetryBadge';
import { FloatingTokenMonitor } from './components/FloatingTokenMonitor';
import { AmbientTensorCanvas } from './components/AmbientTensorCanvas';
import { HostIntegrityBanner } from './components/HostIntegrityBanner';
import { PortfolioAgentWidget } from './components/PortfolioAgentWidget';
import {
  Github,
  Mail,
  Linkedin,
  ExternalLink,
  Sliders,
  Terminal,
  Cpu,
  Binary,
  Play,
  RotateCcw,
  CheckCircle2,
  ArrowRight,
  Bot,
  Blocks,
  PlayCircle,
  Sparkles,
  Flame,
} from 'lucide-react';

/**
 * Progressive assembly timeline with an explicit 150ms delay between stages:
 * - 0ms to 500ms: Dwell on Stage 1 (Raw Bytes)
 * - 500ms to 650ms: 150ms transition delay between Bytes and Semi-words
 * - 650ms to 1250ms: Dwell on Stage 2 (BPE Semi-words)
 * - 1250ms to 1400ms: 150ms transition delay between Semi-words and Full Words
 * - 1400ms+: Stage 3 (Full Assembled Words) locked in
 */
function calculateAssemblyProgressWithDelay(elapsedMs: number, totalDuration = 1500): number {
  if (elapsedMs <= 0) return 0.10;
  if (elapsedMs < 500) return 0.10;
  if (elapsedMs < 650) {
    const t = (elapsedMs - 500) / 150;
    return 0.10 + t * (0.50 - 0.10);
  }
  if (elapsedMs < 1250) return 0.50;
  if (elapsedMs < 1400) {
    const t = (elapsedMs - 1250) / 150;
    return 0.50 + t * (1.00 - 0.50);
  }
  return 1.0;
}

/**
 * ScrollAssemblyHeading:
 * When triggered into the viewport reading zone, it smoothly executes a 1.5-second
 * progressive transformation with 150ms delay between each stage.
 */
const ScrollAssemblyHeading: React.FC<{
  text: string;
  stage: TokenAssemblyStage;
  globalProgress: number;
  as?: 'h2' | 'h3' | 'span';
  className?: string;
  lang?: 'en' | 'pt';
}> = ({ text, stage, globalProgress, as = 'h2', className = '', lang = 'en' }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [localProgress, setLocalProgress] = useState<number>(0.10);
  const hasAnimatedRef = useRef<boolean>(false);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (stage !== 'auto') return;

    const startAnimation = () => {
      const startTime = performance.now();
      const duration = 2000; // 2.0s for sequential cascading across words

      const step = (now: number) => {
        const elapsed = now - startTime;
        const ratio = Math.min(1, elapsed / duration);
        setLocalProgress(ratio);

        if (ratio < 1) {
          rafRef.current = requestAnimationFrame(step);
        } else {
          setLocalProgress(1.0);
        }
      };

      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(step);
    };

    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight || 800;

      // Trigger zone: when heading reaches ~75% of viewport height (reading zone)
      const isInView = rect.top <= windowHeight * 0.75 && rect.bottom >= 0;
      const isFarBelow = rect.top > windowHeight * 0.98;

      if (isInView && !hasAnimatedRef.current) {
        hasAnimatedRef.current = true;
        startAnimation();
      } else if (isFarBelow && hasAnimatedRef.current) {
        // Reset when scrolled back above
        hasAnimatedRef.current = false;
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        setLocalProgress(0.10);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [stage]);

  const effectiveProgress = stage === 'auto' ? localProgress : globalProgress;

  return (
    <div ref={containerRef} className="inline-block">
      <TokenInteractiveText
        text={text}
        progress={effectiveProgress}
        stage={stage}
        as={as}
        size="lg"
        className={className}
        lang={lang}
      />
    </div>
  );
};

export default function App() {
  const [assemblyStage, setAssemblyStage] = useState<TokenAssemblyStage>('auto');
  const [scrollProgress, setScrollProgress] = useState<number>(0.10); // Start at ~0.10 so bytes are clearly visible
  const [isReplaying, setIsReplaying] = useState<boolean>(false);
  const [lang, setLang] = useState<'en' | 'pt'>('en');
  const [isTerminalOpen, setIsTerminalOpen] = useState<boolean>(false);
  const [isAgentModalOpen, setIsAgentModalOpen] = useState<boolean>(false);
  const [agentModalInitialTab, setAgentModalInitialTab] = useState<'llmstxt' | 'jsonld' | 'prompt' | 'mcp'>('llmstxt');

  // Keyboard shortcut Alt+A to toggle Agent Mode
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

  const heroAnimatedRef = useRef<boolean>(false);
  const heroRafRef = useRef<number | null>(null);

  // 2.2s Progressive Animation Engine for Hero: decodes each word sequentially one after the other
  const startHeroAnimation = () => {
    setIsReplaying(true);
    const startTime = performance.now();
    const duration = 2200; // 2.2s allows each word to sequentially transform

    const step = (now: number) => {
      const elapsed = now - startTime;
      const ratio = Math.min(1, elapsed / duration);
      setScrollProgress(ratio);

      if (ratio < 1) {
        heroRafRef.current = requestAnimationFrame(step);
      } else {
        setScrollProgress(1.0);
        setIsReplaying(false);
      }
    };

    if (heroRafRef.current) cancelAnimationFrame(heroRafRef.current);
    heroRafRef.current = requestAnimationFrame(step);
  };

  // On page mount: brief 400ms delay to display initial bytes, then sequentially assemble words
  useEffect(() => {
    const mountTimer = setTimeout(() => {
      if (!heroAnimatedRef.current && assemblyStage === 'auto') {
        heroAnimatedRef.current = true;
        startHeroAnimation();
      }
    }, 400);

    return () => clearTimeout(mountTimer);
  }, [assemblyStage]);

  // Manual Replay trigger
  const triggerReplayAnimation = () => {
    heroAnimatedRef.current = true;
    setAssemblyStage('auto');
    startHeroAnimation();
  };

  // Determine current active stage name for telemetry indicator
  const currentStageNum =
    assemblyStage === 'bytes'
      ? 1
      : assemblyStage === 'semiwords'
      ? 2
      : assemblyStage === 'words'
      ? 3
      : scrollProgress < 0.35
      ? 1
      : scrollProgress < 0.72
      ? 2
      : 3;

  // Sample texts evaluated by the floating ML token context telemetry
  const textSamplesForMonitor = useMemo(
    () => [
      'JULIO CESAR DA COSTA REIS FILHO',
      lang === 'pt'
        ? 'Arquiteto de Sistemas Sênior e Engenheiro de LLMs. Especialista em construir modelos de inferência causal do zero (CIR-Engine 503M), clusters distribuídos DDP em PyTorch e plataformas escaláveis em Go, Next.js e Flutter.'
        : 'Senior Systems Architect and Machine Learning Engineer. Specialist in training custom Causal Transformers from scratch (CIR-Engine 503M), orchestrating distributed DDP clusters in PyTorch, and architecting scalable enterprise systems in Go, Next.js, and Flutter.',
      'CIR-ENGINE 503M CAUSAL LLM ARCHITECTURE',
      'DISTRIBUTED DDP TRAINING IN PYTORCH CLUSTER',
      'ENTERPRISE ARCHITECTURE AND GO MICROSERVICES',
      'FULL STACK SYSTEMS WITH NEXTJS AND FLUTTER',
      'BENCHMARKS AND INFERENCE METRICS EVALUATION',
      'CONTACT AND ARCHITECTURE CONSULTATION',
    ],
    [lang]
  );

  return (
    <div className="min-h-screen bg-[#07080a] text-zinc-100 selection:bg-[#adff2f] selection:text-black">
      {/* Host Integrity & Anti-Cloning Banner (renders only if on unauthorized mirror) */}
      <HostIntegrityBanner />

      {/* Background Subtle Grid Texture */}
      <div className="fixed inset-0 pointer-events-none bg-grid-pattern opacity-40 z-0" />

      {/* Ambient Negative-Space Tensor & KV-Cache Substrate */}
      <AmbientTensorCanvas lang={lang} />

      {/* Floating ML Token Telemetry Monitor */}
      <FloatingTokenMonitor textSources={textSamplesForMonitor} lang={lang} />

      {/* Navigation */}
      <Navbar
        assemblyStage={assemblyStage}
        setAssemblyStage={setAssemblyStage}
        lang={lang}
        setLang={setLang}
        onOpenTerminal={() => setIsTerminalOpen(true)}
        onOpenAgentView={() => setIsAgentModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="relative z-10 pt-28 pb-24 px-6 lg:px-12 max-w-7xl mx-auto space-y-32">
        {/* HERO SECTION */}
        <section id="overview" className="pt-6 pb-12">
          {/* Unboxed Metadata Header (frontend-design skill compliant) */}
          <div className="flex items-center gap-2.5 text-xs font-mono text-zinc-400 mb-6 flex-wrap">
            <span className="text-[#adff2f]">
              {lang === 'pt' ? 'STATUS: DECODIFICANDO_ESPAÇO_LATENTE' : 'STATUS: DECODING_LATENT_SPACE'}
            </span>
            <span aria-hidden="true">·</span>
            <span>
              {lang === 'pt' ? 'ARQUITETO DE SISTEMAS & ENGENHEIRO DE LLMs' : 'SYSTEMS ARCHITECT & LLM ENGINEER'}
            </span>
            <span aria-hidden="true">·</span>
            <span>{lang === 'pt' ? 'GLOBAL / REMOTO' : 'GLOBAL / REMOTE'}</span>
            <span aria-hidden="true">·</span>
            <VisitorTelemetryBadge
              lang={lang}
              onOpenAgentModal={() => setIsAgentModalOpen(true)}
            />
          </div>

          {/* Hero Main Headline with 3-Stage Progressive Token Assembly */}
          <div className="space-y-4 mb-8">
            <div className="text-3xl sm:text-5xl lg:text-7xl font-extrabold tracking-tight text-white leading-tight min-h-[90px]">
              <TokenInteractiveText
                text="JULIO CESAR DA COSTA REIS FILHO"
                progress={scrollProgress}
                stage={assemblyStage}
                size="hero"
                as="h1"
                lang={lang}
              />
            </div>
          </div>

          {/* Subtitle / Value Proposition with Token Assembly */}
          <div className="max-w-3xl border-l-2 border-[#adff2f]/40 pl-6 my-8 space-y-3">
            <div className="text-base sm:text-lg text-zinc-300 leading-relaxed font-sans">
              <TokenInteractiveText
                text={
                  lang === 'pt'
                    ? 'Arquiteto de Sistemas Sênior e Engenheiro de LLMs. Especialista em construir modelos de inferência causal do zero (CIR-Engine 503M), clusters distribuídos DDP em PyTorch e plataformas escaláveis em Go, Next.js e Flutter.'
                    : 'Senior Systems Architect and Machine Learning Engineer. Specialist in training custom Causal Transformers from scratch (CIR-Engine 503M), orchestrating distributed DDP clusters in PyTorch, and architecting scalable enterprise systems in Go, Next.js, and Flutter.'
                }
                progress={scrollProgress}
                stage={assemblyStage}
                size="sm"
                as="span"
                lang={lang}
              />
            </div>
          </div>

          {/* 3-STAGE TOKEN ASSEMBLY PIPELINE HUD */}
          <div className="my-8 p-5 bg-zinc-950/85 border border-zinc-800 rounded-2xl shadow-xl max-w-3xl backdrop-blur-md space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-850">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#adff2f]" />
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-white">
                  {lang === 'pt' ? 'Pipeline de Montagem de Tokens por Scroll' : 'Scroll-Driven Token Assembly Pipeline'}
                </span>
              </div>
              <div className="flex items-center gap-2 self-start sm:self-auto">
                <span className="text-xs font-mono text-zinc-400">
                  {lang === 'pt' ? 'Progresso:' : 'Progress:'}
                </span>
                <span className="text-xs font-mono font-bold text-[#adff2f] tabular-nums bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded">
                  {Math.round(scrollProgress * 100)}%
                </span>
                <button
                  onClick={triggerReplayAnimation}
                  disabled={isReplaying}
                  className="flex items-center gap-1.5 text-xs font-mono px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 transition-colors disabled:opacity-50 cursor-pointer"
                  title={lang === 'pt' ? 'Assistir animação de montagem em 3 estágios' : 'Watch the 3-stage assembly animation'}
                >
                  <RotateCcw className={`w-3 h-3 text-[#adff2f] ${isReplaying ? 'animate-spin' : ''}`} />
                  <span>{isReplaying ? (lang === 'pt' ? 'Montando...' : 'Assembling...') : (lang === 'pt' ? 'Repetir' : 'Replay')}</span>
                </button>
              </div>
            </div>

            {/* 3 Step Visual Indicators */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs font-mono">
              {/* Step 1: Raw Bytes */}
              <button
                onClick={() => setAssemblyStage('bytes')}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  currentStageNum === 1
                    ? 'bg-lime-400/10 border-[#adff2f] text-white shadow-xs'
                    : 'bg-zinc-900/40 border-zinc-850 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-[#adff2f]">1. BYTES</span>
                  {currentStageNum === 1 && <span className="w-2 h-2 rounded-full bg-[#adff2f] animate-ping" />}
                </div>
                <div className="text-[11px] text-zinc-400 mt-1 truncate">
                  0x4A 0x75 0x6C 0x69 0x6F
                </div>
                <div className="text-[9px] text-zinc-500 mt-0.5">
                  {lang === 'pt' ? 'Stream UTF-8 em Memória' : 'UTF-8 Memory Stream'}
                </div>
              </button>

              {/* Step 2: Semi-words */}
              <button
                onClick={() => setAssemblyStage('semiwords')}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  currentStageNum === 2
                    ? 'bg-sky-400/10 border-[#38bdf8] text-white shadow-xs'
                    : 'bg-zinc-900/40 border-zinc-850 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-[#38bdf8]">
                    {lang === 'pt' ? '2. SUB-PALAVRAS' : '2. SEMI-WORDS'}
                  </span>
                  {currentStageNum === 2 && <span className="w-2 h-2 rounded-full bg-[#38bdf8] animate-ping" />}
                </div>
                <div className="text-[11px] text-zinc-400 mt-1 truncate">
                  [JUL#102] [IO#982]
                </div>
                <div className="text-[9px] text-zinc-500 mt-0.5">
                  {lang === 'pt' ? 'Tokens de Subpalavras BPE' : 'BPE Subword Tokens'}
                </div>
              </button>

              {/* Step 3: Full Words */}
              <button
                onClick={() => setAssemblyStage('words')}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  currentStageNum === 3
                    ? 'bg-white/10 border-white text-white shadow-xs'
                    : 'bg-zinc-900/40 border-zinc-850 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white">
                    {lang === 'pt' ? '3. PALAVRAS' : '3. WORDS'}
                  </span>
                  {currentStageNum === 3 && <CheckCircle2 className="w-3.5 h-3.5 text-white" />}
                </div>
                <div className="text-[11px] text-zinc-300 mt-1 truncate font-sans">
                  "JULIO CESAR"
                </div>
                <div className="text-[9px] text-zinc-500 mt-0.5 font-mono">
                  {lang === 'pt' ? 'Texto Montado Final' : 'Final Assembled Text'}
                </div>
              </button>
            </div>

            {/* Interactive Scrubber Slider */}
            <div className="space-y-1.5 pt-1">
              <div className="flex justify-between text-[11px] font-mono text-zinc-400">
                <span>{lang === 'pt' ? 'Role a página ou deslize:' : 'Scroll page or drag to scrub assembly:'}</span>
                <span className="text-[#adff2f]">
                  {currentStageNum === 1
                    ? (lang === 'pt' ? 'Estágio 1 (Bytes)' : 'Stage 1 (Bytes)')
                    : currentStageNum === 2
                    ? (lang === 'pt' ? 'Estágio 2 (Sub-palavras)' : 'Stage 2 (Semi-words)')
                    : (lang === 'pt' ? 'Estágio 3 (Palavras Completas)' : 'Stage 3 (Full Words)')}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.02"
                value={scrollProgress}
                onChange={(e) => {
                  setScrollProgress(parseFloat(e.target.value));
                  if (assemblyStage !== 'auto') setAssemblyStage('auto');
                }}
                className="w-full accent-[#adff2f] bg-zinc-800 h-2 rounded cursor-pointer"
              />
            </div>
          </div>

          {/* Unboxed Key Proof Metrics (anti-slop clean text layout) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 py-6 border-y border-zinc-800/80 my-8">
            <div>
              <div className="text-3xl font-bold font-mono text-[#adff2f] tabular-nums">503M</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt' ? 'Parâmetros Transformer Causal' : 'Causal Transformer Parameters'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-[#38bdf8] tabular-nums">~6.200</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt' ? 'Tokens/seg Throughput DDP' : 'Tokens/sec DDP Throughput'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-[#c4b5fd] tabular-nums">
                {lang === 'pt' ? '8+ Anos' : '8+ Years'}
              </div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt' ? 'Arquitetura de Sistemas & Full-Stack' : 'Full-Stack & Systems Architecture'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold font-mono text-emerald-400 tabular-nums">1M+</div>
              <div className="text-xs text-zinc-400 mt-1">
                {lang === 'pt' ? 'Usuários em FinTech Mobile' : 'Users Served in FinTech Mobile'}
              </div>
            </div>
          </div>

          {/* Hero Action Buttons & External Channels */}
          <div className="flex flex-wrap items-center gap-4 pt-4">
            <a
              href="#cir-engine"
              className="px-5 py-2.5 bg-[#adff2f] hover:bg-lime-300 text-black text-xs font-mono font-bold rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-lime-400/10 cursor-pointer"
            >
              <Cpu className="w-4 h-4" />
              <span>{lang === 'pt' ? 'Explorar CIR-Engine 503M' : 'Inspect CIR-Engine 503M'}</span>
            </a>

            <a
              href="#tokenizer-lab"
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-white text-xs font-mono rounded-lg border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Binary className="w-4 h-4 text-[#38bdf8]" />
              <span>{lang === 'pt' ? 'Laboratório de Tokens' : 'Token & Byte Sandbox'}</span>
            </a>

            <button
              onClick={() => setIsTerminalOpen(true)}
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-zinc-300 hover:text-white text-xs font-mono rounded-lg border border-zinc-800 transition-all flex items-center gap-2 cursor-pointer"
              title={lang === 'pt' ? 'Abrir terminal para executar predict_contact_method()' : 'Launch terminal to run predict_contact_method()'}
            >
              <Terminal className="w-4 h-4 text-[#adff2f]" />
              <span>predict_contact_method()</span>
            </button>

            <button
              onClick={() => {
                setAgentModalInitialTab('mcp');
                setIsAgentModalOpen(true);
              }}
              className="px-5 py-2.5 bg-zinc-900 hover:bg-zinc-850 text-white text-xs font-mono rounded-lg border border-purple-500/40 hover:border-purple-400 transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-purple-500/10"
              title={lang === 'pt' ? 'Conectar portfólio via Model Context Protocol (MCP)' : 'Connect portfolio via Model Context Protocol (MCP)'}
            >
              <Blocks className="w-4 h-4 text-purple-400" />
              <span>{lang === 'pt' ? 'Servidor MCP' : 'MCP Server'}</span>
              <span className="px-1.5 py-0.5 text-[9px] bg-purple-500/20 text-purple-300 rounded border border-purple-500/30 font-bold">
                npx
              </span>
            </button>

            <a
              href="https://github.com/byte-od"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-white rounded-lg border border-zinc-800 transition-colors"
              title={lang === 'pt' ? 'Perfil no GitHub (byte-od)' : 'GitHub Profile (byte-od)'}
            >
              <Github className="w-4 h-4" />
            </a>

            <a
              href="mailto:reisfilho1116@gmail.com"
              className="p-2.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-[#adff2f] rounded-lg border border-zinc-800 transition-colors"
              title={lang === 'pt' ? 'Enviar E-mail Direto' : 'Send Direct Email'}
            >
              <Mail className="w-4 h-4" />
            </a>
          </div>

          {/* ========================================================= */}
          {/* ⚡ FLAGSHIP BREAKTHROUGH: 72B LLM ON 16GB GPU + SCACHE     */}
          {/* ========================================================= */}
          <div className="relative mt-8 p-6 sm:p-8 rounded-2xl bg-gradient-to-br from-zinc-900/95 via-zinc-950 to-black border-2 border-[#adff2f]/70 shadow-[0_0_40px_rgba(173,255,47,0.15)] overflow-hidden">
            {/* Ambient backdrop glow */}
            <div className="absolute top-0 right-0 -mr-20 -mt-20 w-72 h-72 bg-[#adff2f]/10 rounded-full blur-3xl pointer-events-none" />

            <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
              <div className="space-y-3 max-w-2xl">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-bold bg-[#adff2f]/15 text-[#adff2f] border border-[#adff2f]/40 shadow-xs">
                    <Sparkles className="w-3.5 h-3.5 animate-pulse" />
                    {lang === 'pt' ? 'BREAKTHROUGH MUNDIAL · NOV 2026' : 'WORLD-FIRST BREAKTHROUGH · NOV 2026'}
                  </span>
                  <span className="text-[11px] font-mono text-zinc-400 bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800">
                    Open-Source &amp; Kaggle Top 5k
                  </span>
                </div>

                <h3 className="text-2xl sm:text-3xl font-black text-white tracking-tight leading-tight">
                  Mixture of Scars (MoS) &amp; SCache
                </h3>

                <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed">
                  {lang === 'pt'
                    ? 'Arquitetura pioneira que resolve simultaneamente os dois maiores limites físicos de LLMs: executa todas as 80 camadas de um 72B em GPUs comuns de 16 GB com pico de 2.40 GB VRAM, e substitui o KV Cache linear O(T) por uma memória associativa O(1) de 34 KB no L1 Cache comprovada em 100.000 tokens com >99.999% de economia.'
                    : 'Pioneering dual systems architecture cracking both physical memory walls of foundation models: executes all 80 transformer layers of 72B parameter geometry on a consumer 16GB GPU with 2.40 GB peak VRAM, and eliminates the linear O(T) KV Cache using an O(1) 34 KB L1 associative memory verified under 100,000 continuous tokens with >99.999% VRAM savings.'}
                </p>

                {/* Stat badges */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">72B Peak VRAM</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#adff2f]">2.40 GB (80 Layers)</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">100k Token SCache</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#38bdf8]">34 KB vs 32.8 GB</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">KV Cache Savings</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-emerald-400">&gt;99.999% VRAM</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <span className="text-[10px] font-mono text-zinc-500 block">Kaggle Leaderboard</span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-[#c4b5fd]">0.625 Verified</span>
                  </div>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex flex-col sm:flex-row lg:flex-col gap-3 w-full lg:w-auto shrink-0">
                <a
                  href="https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-3 rounded-xl bg-[#adff2f] hover:bg-lime-300 text-black font-mono text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-lime-400/20 cursor-pointer"
                >
                  <PlayCircle className="w-4 h-4" />
                  <span>{lang === 'pt' ? 'Executar Live Test no Colab' : 'Launch Live Test on Colab'}</span>
                </a>

                <a
                  href="https://github.com/byte-od/mixture-of-scars"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-3 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-white font-mono text-xs font-bold border border-zinc-750 transition-all flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Github className="w-4 h-4" />
                  <span>{lang === 'pt' ? 'Ver Código no GitHub' : 'Inspect GitHub Repository'}</span>
                </a>

                <a
                  href="https://www.kaggle.com/competitions/name-that-llm/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 rounded-xl bg-zinc-950 hover:bg-zinc-900 text-zinc-400 hover:text-white font-mono text-[11px] border border-zinc-850 transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <span>{lang === 'pt' ? 'Ver no Kaggle Leaderboard (the tod)' : 'Kaggle Leaderboard (the tod)'}</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 1: CIR-ENGINE (CORE AI/ML ARCHITECTURE) */}
        <section id="cir-engine" className="space-y-6">
          <div className="flex items-end justify-between border-b border-zinc-800 pb-4">
            <div>
              <span className="text-xs font-mono text-[#adff2f] uppercase tracking-wider">
                {lang === 'pt' ? '01. Kernel de Inteligência Causal' : '01. Causal Intelligence Kernel'}
              </span>
              <div className="mt-1">
                <ScrollAssemblyHeading
                  text={lang === 'pt' ? 'Arquitetura CIR-Engine & Cir-Jev' : 'CIR-Engine & Cir-Jev Architecture'}
                  stage={assemblyStage}
                  globalProgress={scrollProgress}
                  className="text-3xl sm:text-4xl font-black text-white tracking-tight"
                  lang={lang}
                />
              </div>
            </div>
            <a
              href="https://github.com/byte-od/cir-jev"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-zinc-400 hover:text-white flex items-center gap-1"
            >
              <span>{lang === 'pt' ? 'Repositório GitHub' : 'GitHub Repo'}</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          <CirEngineArchitecture lang={lang} />
        </section>

        {/* SECTION 2: INTERACTIVE TOKENIZER & ATTENTION SANDBOX */}
        <section id="tokenizer-lab" className="space-y-6">
          <div className="flex items-end justify-between border-b border-zinc-800 pb-4">
            <div>
              <span className="text-xs font-mono text-[#38bdf8] uppercase tracking-wider">
                {lang === 'pt' ? '02. Mecânica de Atenção & BPE em Nível de Byte' : '02. Byte-Level BPE & Attention Mechanics'}
              </span>
              <div className="mt-1">
                <ScrollAssemblyHeading
                  text={lang === 'pt' ? 'Tokenizador Interativo & Matriz de Atenção Causal' : 'Live Tokenizer & Causal Attention Matrix'}
                  stage={assemblyStage}
                  globalProgress={scrollProgress}
                  className="text-3xl sm:text-4xl font-black text-white tracking-tight"
                  lang={lang}
                />
              </div>
            </div>
            <span className="text-xs font-mono text-zinc-500 hidden sm:inline">
              {lang === 'pt'
                ? 'UTF-8 · Vocabulário de Subpalavras · Softmax Causal'
                : 'UTF-8 · Subword Vocabulary · Causal Softmax'}
            </span>
          </div>

          <TokenizerPlayground lang={lang} />
        </section>

        {/* SECTION 3: REPOSITORIES MATRIX */}
        <section id="repositories" className="space-y-6">
          <div className="flex items-end justify-between border-b border-zinc-800 pb-4">
            <div>
              <span className="text-xs font-mono text-[#c4b5fd] uppercase tracking-wider">
                {lang === 'pt' ? '03. Ecossistema Open-Source & Ferramental' : '03. Open-Source Ecosystem & Tooling'}
              </span>
              <div className="mt-1">
                <ScrollAssemblyHeading
                  text={lang === 'pt' ? 'Repositórios Latentes & Ferramentas Dev' : 'Latent Repositories & Developer Tools'}
                  stage={assemblyStage}
                  globalProgress={scrollProgress}
                  className="text-3xl sm:text-4xl font-black text-white tracking-tight"
                  lang={lang}
                />
              </div>
            </div>
            <a
              href="https://github.com/byte-od"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-zinc-400 hover:text-white flex items-center gap-1"
            >
              <span>github.com/byte-od</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          <ProjectShowcase lang={lang} />

          {/* Interactive Tool Workbenches */}
          <div id="live-lab" className="pt-8 scroll-mt-24">
            <InteractiveToolDemos lang={lang} />
          </div>
        </section>

        {/* SECTION 4: CAREER TIMELINE & COMPETENCIES */}
        <section id="experience" className="space-y-6">
          <div className="flex items-end justify-between border-b border-zinc-800 pb-4">
            <div>
              <span className="text-xs font-mono text-emerald-400 uppercase tracking-wider">
                {lang === 'pt' ? '04. Trajetória de Engenharia' : '04. Engineering Trajectory'}
              </span>
              <div className="mt-1">
                <ScrollAssemblyHeading
                  text={lang === 'pt' ? 'Logs de Sistema & Experiência Profissional' : 'System Logs & Career Experience'}
                  stage={assemblyStage}
                  globalProgress={scrollProgress}
                  className="text-3xl sm:text-4xl font-black text-white tracking-tight"
                  lang={lang}
                />
              </div>
            </div>
            <span className="text-xs font-mono text-zinc-500">
              {lang === 'pt' ? '8+ Anos de Experiência · 2019 – Presente' : '8+ Years Experience · 2019 – Present'}
            </span>
          </div>

          <ExperienceTimeline lang={lang} />
        </section>

        {/* SECTION 5: FREQUENTLY ASKED QUESTIONS & AEO DOSSIER */}
        <FaqSection lang={lang} />

        {/* SECTION 6: CONTACT / CTA */}
        <section id="contact" className="p-8 sm:p-12 bg-zinc-950/80 border border-zinc-800 rounded-3xl relative overflow-hidden">
          <div className="relative z-10 max-w-4xl space-y-6">
            <span className="text-xs font-mono text-[#adff2f] uppercase tracking-wider">
              {lang === 'pt' ? '[INICIAR_SEQUÊNCIA_DE_CONTATO]' : '[INITIATE_CONTACT_SEQUENCE]'}
            </span>
            <div className="text-2xl sm:text-4xl font-black text-white tracking-tight leading-tight">
              <ScrollAssemblyHeading
                text={
                  lang === 'pt'
                    ? 'Pronto para arquitetar inferência escalável de LLMs e sistemas distribuídos.'
                    : 'Ready to architect scalable LLM inference and distributed systems.'
                }
                stage={assemblyStage}
                globalProgress={scrollProgress}
                className="font-black text-white tracking-tight"
                lang={lang}
              />
            </div>
            <p className="text-sm sm:text-base text-zinc-400 leading-relaxed">
              {lang === 'pt'
                ? 'Disponível para posições de liderança técnica, arquitetura de sistemas distribuídos e engenharia de LLMs/Transformers. Aberto para oportunidades remotas ou relocalização internacional.'
                : 'Available for Technical Leadership, Distributed Systems Architecture, and LLM/Transformer Engineering. Open to remote roles and international relocation.'}
            </p>

            {/* Interactive Contact Prompt Console */}
            <ContactPromptConsole
              lang={lang}
              onOpenTerminal={() => setIsTerminalOpen(true)}
            />

            {/* Direct Channel Coordinates */}
            <div className="flex flex-wrap items-center gap-4 pt-4 border-t border-zinc-800/80">
              <button
                onClick={() => setIsTerminalOpen(true)}
                className="px-6 py-3 bg-[#adff2f] text-black font-mono text-xs font-bold rounded-xl hover:bg-lime-300 transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-lime-400/10"
              >
                <Terminal className="w-4 h-4" />
                <span>{lang === 'pt' ? 'Abrir Terminal Interativo' : 'Launch Interactive Terminal'}</span>
              </button>

              <a
                href="mailto:reisfilho1116@gmail.com"
                className="px-6 py-3 bg-zinc-900 hover:bg-zinc-850 text-white font-mono text-xs rounded-xl border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2"
              >
                <Mail className="w-4 h-4 text-[#38bdf8]" />
                <span>reisfilho1116@gmail.com</span>
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

              <a
                href="https://github.com/byte-od"
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-zinc-900 hover:bg-zinc-850 text-zinc-300 hover:text-white font-mono text-xs rounded-xl border border-zinc-800 hover:border-zinc-700 transition-all flex items-center gap-2"
              >
                <Github className="w-4 h-4" />
                <span>GitHub @byte-od</span>
              </a>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-zinc-900/80 py-8 px-6 lg:px-12 text-zinc-500 font-mono text-xs relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span>© 2026 Julio Cesar da Costa Reis Filho</span>
            <span aria-hidden="true">·</span>
            <span>{lang === 'pt' ? 'O Arquiteto Latente' : 'The Latent Architect'}</span>
            <span aria-hidden="true" className="hidden lg:inline">·</span>
            <span className="hidden lg:inline text-zinc-500 font-mono text-[11px]">
              {lang === 'pt' ? 'Telemetria: Humanos & Agentes Ativos' : 'Telemetry: Active Humans & Agents'}
            </span>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <button
              type="button"
              onClick={() => setIsAgentModalOpen(true)}
              aria-haspopup="dialog"
              className="text-[#adff2f] hover:text-white transition-colors cursor-pointer flex items-center gap-1.5 font-semibold"
              title={lang === 'pt' ? 'Versão para Agentes de IA & LLMs (llms.txt)' : 'AI Agent & LLM View (llms.txt)'}
            >
              <Bot className="w-3.5 h-3.5" />
              <span>{lang === 'pt' ? 'Modo Agentes (llms.txt)' : 'Agent View (llms.txt)'}</span>
            </button>
            <a href="https://linkedin.com/in/juliocrfilho" target="_blank" rel="noopener noreferrer" className="hover:text-[#38bdf8] transition-colors flex items-center gap-1">
              <Linkedin className="w-3.5 h-3.5 text-[#0a66c2]" />
              <span>LinkedIn</span>
            </a>
            <a href="https://github.com/byte-od" target="_blank" rel="noopener noreferrer" className="hover:text-[#adff2f] transition-colors">
              GitHub
            </a>
            <a href="mailto:reisfilho1116@gmail.com" className="hover:text-[#adff2f] transition-colors">
              {lang === 'pt' ? 'E-mail' : 'Email'}
            </a>
            <button onClick={() => setIsTerminalOpen(true)} className="hover:text-white transition-colors cursor-pointer">
              Terminal
            </button>
          </div>
        </div>
      </footer>

      {/* Interactive Terminal Modal */}
      <TerminalModal
        isOpen={isTerminalOpen}
        onClose={() => setIsTerminalOpen(false)}
        lang={lang}
      />

      {/* Accessible AI Agent View & LLMs.txt Modal */}
      <AgentViewModal
        isOpen={isAgentModalOpen}
        onClose={() => setIsAgentModalOpen(false)}
        lang={lang}
        initialTab={agentModalInitialTab}
      />

      {/* Interactive MoS & SCache Conversational Portfolio Agent */}
      <PortfolioAgentWidget lang={lang} />
    </div>
  );
}
