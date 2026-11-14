import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Bot,
  Cpu,
  Database,
  DownloadCloud,
  ExternalLink,
  Github,
  Globe,
  Layers,
  Maximize2,
  Minimize2,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  X,
  Zap,
} from 'lucide-react';
import {
  BrowserMoSWebGPUEngine,
  checkMosShardsCached,
} from '../lib/mosWebgpuEngine';
import {
  BrowserScarStoreHTTP206,
  CapsuleDirectoryEntry,
  CapsuleLink,
  Http206Event,
} from '../lib/scarstoreHttp206Browser';
import { probeWebGPUHardware } from '../lib/webgpuKernel';

export type AgentEngineMode = 'webgpu' | 'reflex';
type PanelTab = 'chat' | 'scarstore';

interface PortfolioAgentWidgetProps {
  lang?: 'en' | 'pt';
}

interface ActiveCartridgeBadge {
  capsuleId: string;
  gate: number;
  fromCache: boolean;
  title?: string;
  titlePt?: string;
  category?: string;
  links?: CapsuleLink[];
}

interface TurnTelemetry {
  scacheBytes: number;
  scacheDeltaBytes: number;
  tokPerSec: number;
  activeCapsules: ActiveCartridgeBadge[];
  turnHttp206Bytes: number;
  confidence: number;
  margin: number;
  prefillMs?: number;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  modeUsed: AgentEngineMode;
  links?: CapsuleLink[];
  telemetry?: TurnTelemetry;
  isSystemNotice?: boolean;
}

const WEBGPU_BASE_URL = `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/webgpu/qwen0p5b`;
const DEFAULT_SCACHE_BYTES = 4332288; // 4230.8 KB (112 sinks + 32 Widrow-Hoff slots + 144 window across 24 layers)

function formatScacheKb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function nowTimeStr(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Instant `< 1 ms` CPU Epistemic Reflex Guide (`0 MB` network) for smartphones / non-WebGPU browsers.
 * Explicitly labeled as `CPU Reflex · 0 MB` (never masquerading as GPU execution).
 */
function buildReflexGuideResponse(
  rawQuery: string,
  fallbackLang: 'en' | 'pt'
): { text: string; links: CapsuleLink[] } {
  const qNorm = rawQuery
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');

  const isEs = /\b(quien|cual|como|explica|principio|teorema|contratar|correo|portafolio|buenos dias|hola)\b/.test(
    qNorm
  );
  const isFr = /\b(qui|quel|quelle|comment|expliquez|principe|theoreme|recruter|bonjour)\b/.test(qNorm);
  const isPt =
    !isEs &&
    !isFr &&
    (/\b(quem|qual|como|explique|fale|sobre|voce|experiencia|contato|contratar|principio|teorema|ola|bom dia|boa tarde|boa noite)\b/.test(
      qNorm
    ) ||
      fallbackLang === 'pt');

  const rLang: 'pt' | 'en' | 'es' | 'fr' = isEs ? 'es' : isFr ? 'fr' : isPt ? 'pt' : 'en';

  const colabLink: CapsuleLink = {
    label: '⚡ Colab Live Proof (72B + SCache O(1))',
    url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
    icon: 'colab',
  };
  const mosLink: CapsuleLink = {
    label: 'GitHub mixture-of-scars',
    url: 'https://github.com/byte-od/mixture-of-scars',
    icon: 'github',
  };
  const auraLink: CapsuleLink = {
    label: 'GitHub system_one (AURA)',
    url: 'https://github.com/byte-od/system_one',
    icon: 'github',
  };
  const linkedinLink: CapsuleLink = {
    label: 'LinkedIn /in/juliocrfilho',
    url: 'https://linkedin.com/in/juliocrfilho',
    icon: 'ext',
  };

  if (
    qNorm.includes('quem e voce') ||
    qNorm.includes('who are you') ||
    qNorm.includes('quien eres') ||
    qNorm.includes('qui etes-vous')
  ) {
    const map = {
      pt: 'Sou o System One, assistente oficial de IA do portfólio de Julio Reis (nunca o próprio Julio). No modo ⚡ WebGPU · WGSL v4.0, executo 24 camadas em shaders WGSL nativos com memória constante SCache O(1) (4230.8 KB, Δ +0 B) e 13 cartuchos universais de 61.25 KB via HTTP 206. No modo atual (🏹 Reflexo S1 · 0 MB), respondo em < 1 ms na CPU.',
      en: "I am System One, the official AI assistant for Julio Reis's portfolio (never Julio himself). In ⚡ WebGPU · WGSL v4.0 mode, I execute 24 Transformer layers via custom WGSL compute shaders with fixed SCache O(1) memory (4230.8 KB, Δ +0 B) and 13 universal 61.25 KB HTTP 206 cartridges. In the current mode (🏹 S1 Reflex · 0 MB), I respond in < 1 ms on CPU.",
      es: 'Soy System One, el asistente oficial de IA del portafolio de Julio Reis. En modo ⚡ WebGPU · WGSL v4.0 ejecuto 24 capas mediante shaders WGSL nativos con memoria constante SCache O(1) (4230.8 KB) y 13 cartuchos HTTP 206 de 61.25 KB.',
      fr: "Je suis System One, l'assistant IA officiel du portfolio de Julio Reis. En mode ⚡ WebGPU · WGSL v4.0, j'exécute 24 couches via des shaders WGSL natifs avec mémoire constante SCache O(1) (4230.8 Ko) et 13 cartouches HTTP 206 de 61.25 Ko.",
    };
    return { text: map[rLang], links: [mosLink, linkedinLink] };
  }

  if (qNorm.includes('mos') || qNorm.includes('scache') || qNorm.includes('mixture of scars') || qNorm.includes('72b') || qNorm.includes('webgpu') || qNorm.includes('wgsl') || qNorm.includes('206')) {
    const map = {
      pt: 'O Mixture of Scars (MoS) executa todas as 80 camadas de um LLM 72B em apenas 2.40 GB de VRAM de pico via Block-64 INT4 + ASVD em streaming, enquanto o SCache O(1) substitui 32.8 GB de KV Cache linear a 100k tokens por memória associativa fixa (Widrow-Hoff). Neste portfólio, o motor WebGPU/WGSL v4.0 compila 15 shaders nativos e baixa cartuchos universais de 61.25 KB sob demanda via HTTP 206!',
      en: 'Mixture of Scars (MoS) runs all 80 layers of a 72B LLM in 2.40 GB peak VRAM via Block-64 INT4 + ASVD streaming, while SCache O(1) replaces 32.8 GB of linear KV Cache at 100k tokens with fixed Widrow-Hoff associative memory. On this portfolio, the WebGPU/WGSL v4.0 engine compiles 15 custom compute shaders and streams 61.25 KB language-invariant cartridges via HTTP 206!',
      es: 'Mixture of Scars (MoS) ejecuta un LLM de 72B en 2.40 GB de VRAM pico mediante Block-64 INT4 + ASVD, mientras SCache O(1) reemplaza 32.8 GB de KV Cache lineal a 100k tokens con memoria asociativa fija y cartuchos HTTP 206 de 61.25 KB.',
      fr: "Mixture of Scars (MoS) exécute un LLM 72B dans 2.40 Go de VRAM via Block-64 INT4 + ASVD, tandis que SCache O(1) remplace 32.8 Go de KV Cache linéaire à 100k tokens par une mémoire associative fixe et des cartouches HTTP 206 de 61.25 Ko.",
    };
    return { text: map[rLang], links: [colabLink, mosLink] };
  }

  if (qNorm.includes('aura') || qNorm.includes('rubik') || qNorm.includes('system 1') || qNorm.includes('sistema 1')) {
    const map = {
      pt: 'O AURA (Universal System 1 Engine) é o motor corporificado de Deep Reinforcement Learning criado por Julio Reis: atinge 0.23 ms de latência P50 em CPU pura com zero busca MCTS sobre 232.2 bilhões de estados (D10) do Cubo de Rubik, treinado por 430M+ passos a >20.000 FPS com Recurrent PPO + Teacher SIL + Online DPO.',
      en: "AURA (Universal System 1 Engine) is Julio Reis's embodied Deep Reinforcement Learning kernel achieving 0.23 ms bare-CPU P50 latency with zero MCTS tree search across 232.2 billion Rubik's Cube D10 states, trained over 430M+ steps at >20,000 FPS.",
      es: 'AURA (Universal System 1 Engine) es el motor de Deep RL de Julio Reis con latencia de 0.23 ms en CPU pura y cero búsqueda MCTS sobre 232.2 mil millones de estados del Cubo de Rubik.',
      fr: "AURA (Universal System 1 Engine) est le moteur Deep RL de Julio Reis atteignant 0.23 ms sur CPU sans recherche MCTS sur 232.2 milliards d'états du Rubik's Cube.",
    };
    return { text: map[rLang], links: [auraLink] };
  }

  if (qNorm.includes('contato') || qNorm.includes('contact') || qNorm.includes('hire') || qNorm.includes('contratar') || qNorm.includes('recruter') || qNorm.includes('email') || qNorm.includes('linkedin')) {
    const map = {
      pt: 'Julio Reis está disponível globalmente (remoto ou relocalização) para posições de Staff/Principal Systems Architect e Senior LLM Systems Engineer:\n• E-mail: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• GitHub: github.com/byte-od',
      en: 'Julio Reis is available globally (remote or relocation) for Staff/Principal Systems Architect and Senior LLM Systems Engineer roles:\n• Email: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• GitHub: github.com/byte-od',
      es: 'Julio Reis está disponible globalmente en remoto o reubicación como Arquitecto de Sistemas Senior e Ingeniero de LLMs:\n• Correo: reisfilho1116@gmail.com | linkedin.com/in/juliocrfilho | github.com/byte-od',
      fr: 'Julio Reis est disponible mondialement en télétravail ou mobilité :\n• E-mail : reisfilho1116@gmail.com | linkedin.com/in/juliocrfilho | github.com/byte-od',
    };
    return {
      text: map[rLang],
      links: [
        { label: 'reisfilho1116@gmail.com', url: 'mailto:reisfilho1116@gmail.com', icon: 'ext' },
        linkedinLink,
      ],
    };
  }

  const bioMap = {
    pt: 'Julio Reis (Julio Cesar da Costa Reis Filho) é Arquiteto de Sistemas Sênior & Engenheiro de LLMs com 8+ anos de experiência em produção (Asapp Desenvolvimento, Opah IT, Arlequim, sistemas bancários de 1M+ usuários), criador do Mixture of Scars (72B em 2.40 GB VRAM), SCache O(1), AURA Deep RL (0.23 ms CPU), CIR-Engine 503M e mddd-cli.\n\nAtive o modo ⚡ WebGPU · WGSL v4.0 acima para executar o modelo neural real e os 12 cartuchos HTTP 206 na sua GPU!',
    en: 'Julio Reis (Julio Cesar da Costa Reis Filho) is a Senior Systems Architect & LLM Engineer with 8+ years of production experience (Asapp Desenvolvimento, Opah IT, Arlequim, 1M+ user fintech runtimes), creator of Mixture of Scars (72B in 2.40 GB VRAM), SCache O(1), AURA Deep RL (0.23 ms CPU), CIR-Engine 503M, and mddd-cli.\n\nSwitch to ⚡ WebGPU · WGSL v4.0 above to run the live WGSL GPU model and 12 HTTP 206 cartridges!',
    es: 'Julio Reis es Arquitecto de Sistemas Senior e Ingeniero de LLMs con 8+ años de experiencia, creador de Mixture of Scars (72B en 2.40 GB VRAM), SCache O(1), AURA Deep RL y CIR-Engine 503M. ¡Activa ⚡ WebGPU · WGSL v4.0 arriba para ejecutar el motor GPU real!',
    fr: "Julio Reis est Architecte Systèmes Senior et Ingénieur LLM avec 8+ ans d'expérience, créateur de Mixture of Scars (72B dans 2.40 Go VRAM), SCache O(1), AURA Deep RL et CIR-Engine 503M. Activez ⚡ WebGPU · WGSL v4.0 ci-dessus pour exécuter le modèle GPU réel !",
  };
  return { text: bioMap[rLang], links: [colabLink, mosLink, linkedinLink] };
}

export const PortfolioAgentWidget: React.FC<PortfolioAgentWidgetProps> = ({ lang = 'en' }) => {
  const [currentLang, setCurrentLang] = useState<'en' | 'pt'>(() => {
    if (lang === 'pt') return 'pt';
    if (typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('pt')) {
      return 'pt';
    }
    return lang;
  });
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [activeTab, setActiveTab] = useState<PanelTab>('chat');
  const [mode, setMode] = useState<AgentEngineMode>('webgpu');

  // Hardware & Engine States
  const [hasWebGPU, setHasWebGPU] = useState<boolean>(true);
  const [adapterLabel, setAdapterLabel] = useState<string>('Detectando WebGPU...');
  const [shardsCached, setShardsCached] = useState<boolean>(false);
  const [engineReady, setEngineReady] = useState<boolean>(false);
  const [engineLoading, setEngineLoading] = useState<boolean>(false);
  const [loadPhaseDetail, setLoadPhaseDetail] = useState<string>('');
  const [loadPercent, setLoadPercent] = useState<number>(0);
  const [loadLoadedMiB, setLoadLoadedMiB] = useState<number>(0);
  const [loadTotalMiB, setLoadTotalMiB] = useState<number>(526.1);
  const [loadedFromCache, setLoadedFromCache] = useState<boolean>(false);
  const [vramWeightsMiB, setVramWeightsMiB] = useState<number>(526.1);
  const [scacheBytes, setScacheBytes] = useState<number>(DEFAULT_SCACHE_BYTES);
  const [lastTokPerSec, setLastTokPerSec] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ScarStore v4.0 HTTP 206 Directory & Network Log
  const [capsules, setCapsules] = useState<CapsuleDirectoryEntry[]>([]);
  const [httpEvents, setHttpEvents] = useState<Http206Event[]>([]);

  // Chat States
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  // Refs for Worker + Main-thread WebGPU fallback + Queued Auto-Boot Query
  const workerRef = useRef<Worker | null>(null);
  const mainThreadEngineRef = useRef<BrowserMoSWebGPUEngine | null>(null);
  const mainThreadRouterRef = useRef<BrowserScarStoreHTTP206 | null>(null);
  const mainThreadAbortRef = useRef<AbortController | null>(null);
  const useMainThreadRef = useRef<boolean>(false);
  const pendingQueryRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const prevPropLangRef = useRef<'en' | 'pt'>(lang);
  useEffect(() => {
    if (prevPropLangRef.current !== lang) {
      prevPropLangRef.current = lang;
      setCurrentLang(lang);
    }
  }, [lang]);

  const buildWelcomeMessage = (targetLang: 'en' | 'pt'): ChatMessage => ({
    id: 'msg-welcome',
    sender: 'agent',
    timestamp: nowTimeStr(),
    modeUsed: 'webgpu',
    isSystemNotice: true,
    text:
      targetLang === 'pt'
        ? 'Bem-vindo ao System One v4.0 — Agente Nativo WebGPU / WGSL do portfólio de Julio Reis!\n\n• Zero Python e Zero ONNX: 15 shaders WGSL 1.0 executam o Qwen2.5-0.5B (MoS Block-64 INT4 + ASVD) direto na sua GPU com memória constante SCache O(1) (4230.8 KB, Δ +0 B).\n• ScarStore HTTP 206 Poliglota: 13 cartuchos universais de 61.25 KB ancorados em inglês (P_⊥lang na Camada 15) baixados sob demanda em PT, EN, ES e FR.\n\nFaça uma pergunta abaixo ou abra a aba "ScarStore HTTP 206 (13)" para testar qualquer cartucho!'
        : "Welcome to System One v4.0 — Julio Reis's Native WebGPU / WGSL Portfolio Agent!\n\n• Zero Python & Zero ONNX: 15 custom WGSL 1.0 compute shaders run Qwen2.5-0.5B (MoS Block-64 INT4 + ASVD) on your GPU with constant SCache O(1) memory (4230.8 KB, Δ +0 B).\n• Polyglot HTTP 206 ScarStore: 13 universal 61.25 KB cartridges anchored in English (P_⊥lang at Layer 15) streamed on demand across PT, EN, ES, and FR.\n\nAsk anything below or open the 'ScarStore HTTP 206 (13)' tab to test any cartridge!",
    links: [
      {
        label: '⚡ Colab Proof (72B MoS in 2.40 GB + SCache O(1))',
        url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
        icon: 'colab',
      },
      {
        label: 'GitHub mixture-of-scars',
        url: 'https://github.com/byte-od/mixture-of-scars',
        icon: 'github',
      },
    ],
  });

  useEffect(() => {
    setMessages((prev) => {
      if (prev.length === 0 || (prev.length === 1 && prev[0].id === 'msg-welcome')) {
        return [buildWelcomeMessage(currentLang)];
      }
      return prev;
    });
  }, [currentLang]);

  // 1. Hardware Probe + Persistent Storage Request
  useEffect(() => {
    let mounted = true;
    (async () => {
      const [probe, cached] = await Promise.all([
        probeWebGPUHardware(),
        checkMosShardsCached(WEBGPU_BASE_URL),
      ]);
      if (!mounted) return;
      setHasWebGPU(probe.supported);
      setAdapterLabel(probe.adapterLabel);
      setShardsCached(cached);
      if (!probe.supported) {
        setMode('reflex');
      }
      if (typeof navigator !== 'undefined' && navigator.storage?.persist) {
        navigator.storage.persist().catch(() => {});
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Helper: Boot Engine on Main Thread (Fallback only if Worker lacks `navigator.gpu`)
  const bootMainThreadWebGPU = async () => {
    if (mainThreadEngineRef.current) {
      setEngineReady(true);
      setEngineLoading(false);
      return mainThreadEngineRef.current;
    }
    useMainThreadRef.current = true;
    setEngineLoading(true);
    setErrorMessage(null);
    try {
      const eng = await BrowserMoSWebGPUEngine.create(WEBGPU_BASE_URL, (prog) => {
        setLoadPercent(prog.pct);
        setLoadPhaseDetail(prog.detail);
        if (prog.loadedBytes !== undefined) {
          setLoadLoadedMiB(Math.round((prog.loadedBytes / (1024 * 1024)) * 10) / 10);
        }
        if (prog.totalBytes !== undefined && prog.totalBytes > 0) {
          setLoadTotalMiB(Math.round((prog.totalBytes / (1024 * 1024)) * 10) / 10);
        }
        if (prog.fromCache !== undefined) {
          setLoadedFromCache(prog.fromCache);
        }
      });
      if (mainThreadRouterRef.current && eng.scarStore) {
        for (const uid of mainThreadRouterRef.current.unlearnedIds) {
          eng.scarStore.unlearnCapsule(uid);
        }
      }
      mainThreadEngineRef.current = eng;
      setAdapterLabel(eng.adapterLabel);
      setScacheBytes(eng.scacheGpuBytes);
      setVramWeightsMiB(Math.round((eng.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100);
      setLoadedFromCache(eng.loadedFromCache);
      setShardsCached(true);
      if (eng.scarStore) {
        setCapsules(eng.scarStore.getCapsulesDirectory());
      }
      setEngineReady(true);
      setEngineLoading(false);
      return eng;
    } catch (err: any) {
      setEngineLoading(false);
      setErrorMessage(err?.message || 'Falha ao inicializar WebGPU.');
      return null;
    }
  };

  // Helper: Run Generation on Main Thread Engine
  const runMainThreadGenerate = async (query: string, msgId: string) => {
    const eng = mainThreadEngineRef.current;
    if (!eng) return;
    setIsGenerating(true);
    const abortCtrl = new AbortController();
    mainThreadAbortRef.current = abortCtrl;
    try {
      const res = await eng.generate(query, {
        maxNewTokens: 260,
        enableScarStore: true,
        signal: abortCtrl.signal,
        onRouting: (routeRes) => {
          const badges: ActiveCartridgeBadge[] = routeRes.active.map((a) => ({
            capsuleId: a.capsuleId,
            gate: Math.round(a.gate * 10000) / 10000,
            fromCache: a.fromCache,
            title: a.meta?.metadata?.title,
            titlePt: a.meta?.metadata?.title_pt,
            category: a.meta?.metadata?.category,
            links: a.meta?.metadata?.links || [],
          }));
          const mergedLinks = badges.flatMap((b) => b.links || []);
          if (routeRes.httpEvents.length > 0) {
            setHttpEvents((prev) => [...routeRes.httpEvents, ...prev].slice(0, 24));
          }
          if (eng.scarStore) {
            setCapsules(eng.scarStore.getCapsulesDirectory());
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msgId
                ? {
                    ...m,
                    links: mergedLinks.length > 0 ? mergedLinks : m.links,
                    telemetry: {
                      scacheBytes: eng.scacheGpuBytes,
                      scacheDeltaBytes: 0,
                      tokPerSec: m.telemetry?.tokPerSec ?? 0,
                      activeCapsules: badges,
                      turnHttp206Bytes: routeRes.turnBytesFetched,
                      confidence: m.telemetry?.confidence ?? 0,
                      margin: m.telemetry?.margin ?? 0,
                    },
                  }
                : m
            )
          );
        },
        onToken: (tokUpd) => {
          setLastTokPerSec(Math.round(tokUpd.tokPerSec * 10) / 10);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msgId
                ? {
                    ...m,
                    text: tokUpd.text,
                    telemetry: {
                      scacheBytes: tokUpd.scacheBytes,
                      scacheDeltaBytes: tokUpd.scacheDeltaBytes,
                      tokPerSec: Math.round(tokUpd.tokPerSec * 10) / 10,
                      activeCapsules: m.telemetry?.activeCapsules || [],
                      turnHttp206Bytes: m.telemetry?.turnHttp206Bytes ?? 0,
                      confidence: tokUpd.confidence,
                      margin: tokUpd.margin,
                    },
                  }
                : m
            )
          );
        },
      });

      setLastTokPerSec(res.decodeTokPerSec);
      if (eng.scarStore) {
        setCapsules(eng.scarStore.getCapsulesDirectory());
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId
            ? {
                ...m,
                text: res.answer,
                telemetry: {
                  scacheBytes: res.scacheBytesFinal,
                  scacheDeltaBytes: res.scacheDeltaBytes,
                  tokPerSec: res.decodeTokPerSec,
                  activeCapsules: m.telemetry?.activeCapsules || [],
                  turnHttp206Bytes: res.turnHttp206Bytes,
                  confidence: res.meanConfidenceP1,
                  margin: res.meanMarginP1P2,
                  prefillMs: res.prefillMs,
                },
              }
            : m
        )
      );
    } catch (err: any) {
      setErrorMessage(err?.message || 'Erro durante geração WebGPU.');
    } finally {
      setIsGenerating(false);
      mainThreadAbortRef.current = null;
    }
  };

  // 2. Initialize Dedicated WebGPU Worker + Preload `< 90 KB` ScarStore Router Index
  useEffect(() => {
    try {
      const worker = new Worker(new URL('../lib/llmWorker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;

      worker.postMessage({
        type: 'init_router',
        payload: { baseUrl: WEBGPU_BASE_URL },
      });

      worker.onmessage = async (e: MessageEvent) => {
        const d = e.data || {};
        if (d.type === 'router_ready') {
          setShardsCached(!!d.shardsCached);
          if (Array.isArray(d.capsules) && d.capsules.length > 0) {
            setCapsules(d.capsules);
          }
        } else if (d.type === 'loading_start') {
          setEngineLoading(true);
          setErrorMessage(null);
          setLoadedFromCache(!!d.wasAlreadyCached);
          setLoadPercent(d.wasAlreadyCached ? 15 : 2);
        } else if (d.type === 'download_progress') {
          setLoadPercent(d.percent ?? 0);
          setLoadPhaseDetail(d.detail || '');
          if (d.loaded) setLoadLoadedMiB(Math.round((d.loaded / (1024 * 1024)) * 10) / 10);
          if (d.total) setLoadTotalMiB(Math.round((d.total / (1024 * 1024)) * 10) / 10);
          if (d.fromCache !== undefined) setLoadedFromCache(!!d.fromCache);
        } else if (d.type === 'ready') {
          setEngineLoading(false);
          setEngineReady(true);
          setShardsCached(true);
          setLoadedFromCache(!!d.fromCache);
          if (d.adapterLabel) setAdapterLabel(d.adapterLabel);
          if (d.scacheBytes) setScacheBytes(d.scacheBytes);
          if (d.vramWeightsMiB) setVramWeightsMiB(d.vramWeightsMiB);
          if (Array.isArray(d.capsules) && d.capsules.length > 0) {
            setCapsules(d.capsules);
          }
          // Flush queued query if user typed before GPU finished compiling!
          if (pendingQueryRef.current) {
            const q = pendingQueryRef.current;
            pendingQueryRef.current = null;
            dispatchWebGPUQuery(q);
          }
        } else if (d.type === 'routing_telemetry') {
          const badges: ActiveCartridgeBadge[] = d.activeCapsules || [];
          const mergedLinks = badges.flatMap((b) => b.links || []);
          if (Array.isArray(d.httpEvents) && d.httpEvents.length > 0) {
            setHttpEvents((prev) => [...d.httpEvents, ...prev].slice(0, 24));
          }
          if (Array.isArray(d.capsules)) {
            setCapsules(d.capsules);
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === d.msgId
                ? {
                    ...m,
                    links: mergedLinks.length > 0 ? mergedLinks : m.links,
                    telemetry: {
                      scacheBytes: m.telemetry?.scacheBytes ?? DEFAULT_SCACHE_BYTES,
                      scacheDeltaBytes: 0,
                      tokPerSec: m.telemetry?.tokPerSec ?? 0,
                      activeCapsules: badges,
                      turnHttp206Bytes: d.turnHttp206Bytes ?? 0,
                      confidence: m.telemetry?.confidence ?? 0,
                      margin: m.telemetry?.margin ?? 0,
                    },
                  }
                : m
            )
          );
        } else if (d.type === 'token') {
          setLastTokPerSec(d.tokPerSec);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === d.msgId
                ? {
                    ...m,
                    text: d.text,
                    telemetry: {
                      scacheBytes: d.scacheBytes ?? DEFAULT_SCACHE_BYTES,
                      scacheDeltaBytes: d.scacheDeltaBytes ?? 0,
                      tokPerSec: d.tokPerSec ?? 0,
                      activeCapsules: m.telemetry?.activeCapsules || [],
                      turnHttp206Bytes: m.telemetry?.turnHttp206Bytes ?? 0,
                      confidence: d.confidence ?? 0,
                      margin: d.margin ?? 0,
                    },
                  }
                : m
            )
          );
        } else if (d.type === 'generation_end') {
          setIsGenerating(false);
          const res = d.result;
          if (res?.decodeTokPerSec) setLastTokPerSec(res.decodeTokPerSec);
          if (Array.isArray(d.capsules)) setCapsules(d.capsules);
          if (res) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === d.msgId
                  ? {
                      ...m,
                      text: res.answer,
                      telemetry: {
                        scacheBytes: res.scacheBytesFinal,
                        scacheDeltaBytes: res.scacheDeltaBytes,
                        tokPerSec: res.decodeTokPerSec,
                        activeCapsules: m.telemetry?.activeCapsules || [],
                        turnHttp206Bytes: res.turnHttp206Bytes,
                        confidence: res.meanConfidenceP1,
                        margin: res.meanMarginP1P2,
                        prefillMs: res.prefillMs,
                      },
                    }
                  : m
              )
            );
          }
        } else if (d.type === 'capsule_unlearned' || d.type === 'capsule_restored') {
          if (Array.isArray(d.capsules)) {
            setCapsules(d.capsules);
          }
        } else if (d.type === 'scache_reset') {
          if (d.scacheBytes) setScacheBytes(d.scacheBytes);
        } else if (d.type === 'error') {
          if (d.code === 'WEBGPU_BOOT_FAILED' && hasWebGPU && !useMainThreadRef.current) {
            // Browser exposes WebGPU on `window` rather than Worker: seamlessly boot on main thread!
            const eng = await bootMainThreadWebGPU();
            if (eng && pendingQueryRef.current) {
              const q = pendingQueryRef.current;
              pendingQueryRef.current = null;
              const msgId = `agent-${Date.now()}`;
              setMessages((prev) => [
                ...prev,
                {
                  id: msgId,
                  sender: 'agent',
                  timestamp: nowTimeStr(),
                  text: '',
                  modeUsed: 'webgpu',
                },
              ]);
              await runMainThreadGenerate(q, msgId);
            }
            return;
          }
          setEngineLoading(false);
          setIsGenerating(false);
          setErrorMessage(d.error || 'Erro no motor WebGPU.');
        }
      };

      return () => {
        worker.terminate();
      };
    } catch {
      // Fallback router load on main thread if Module Workers are blocked
      BrowserScarStoreHTTP206.load(`${WEBGPU_BASE_URL}/scarstore_router.json`)
        .then((r) => {
          mainThreadRouterRef.current = r;
          setCapsules(r.getCapsulesDirectory());
        })
        .catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (isOpen && !isMinimized && activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, isMinimized, activeTab, isGenerating]);

  useEffect(() => {
    if (isOpen && !isMinimized && activeTab === 'chat') {
      setTimeout(() => inputRef.current?.focus(), 120);
    }
  }, [isOpen, isMinimized, activeTab]);

  const handleBootEngine = async () => {
    if (engineReady || engineLoading) return;
    setErrorMessage(null);
    if (useMainThreadRef.current || !workerRef.current) {
      await bootMainThreadWebGPU();
      return;
    }
    workerRef.current.postMessage({
      type: 'load',
      payload: { baseUrl: WEBGPU_BASE_URL },
    });
  };

  const dispatchWebGPUQuery = (query: string) => {
    const msgId = `agent-${Date.now()}`;
    setIsGenerating(true);
    setMessages((prev) => [
      ...prev,
      {
        id: msgId,
        sender: 'agent',
        timestamp: nowTimeStr(),
        text: '',
        modeUsed: 'webgpu',
      },
    ]);

    if (useMainThreadRef.current && mainThreadEngineRef.current) {
      runMainThreadGenerate(query, msgId);
      return;
    }
    workerRef.current?.postMessage({
      type: 'generate',
      payload: {
        query,
        maxNewTokens: 260,
        enableScarStore: true,
        msgId,
      },
    });
  };

  const handleSendMessage = async (overrideText?: string, forceMode?: AgentEngineMode) => {
    const query = (overrideText ?? inputMessage).trim();
    if (!query || isGenerating) return;

    if (
      currentLang === 'en' &&
      /\b(bom dia|boa tarde|boa noite|olá|ola|quem|qual|como|você|voce|fale|explique|obrigado)\b/i.test(query)
    ) {
      setCurrentLang('pt');
    }

    const targetMode = forceMode ?? mode;
    if (forceMode && forceMode !== mode) {
      setMode(forceMode);
    }
    setActiveTab('chat');
    setInputMessage('');
    setErrorMessage(null);

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: nowTimeStr(),
      text: query,
      modeUsed: targetMode,
    };
    setMessages((prev) => [...prev, userMsg]);

    // Mode 2: Explicit `< 1 ms` CPU Reflex Guide (`0 MB` download)
    if (targetMode === 'reflex' || !hasWebGPU) {
      const reflex = buildReflexGuideResponse(query, currentLang);
      setMessages((prev) => [
        ...prev,
        {
          id: `agent-${Date.now()}`,
          sender: 'agent',
          timestamp: nowTimeStr(),
          text: reflex.text,
          links: reflex.links,
          modeUsed: 'reflex',
        },
      ]);
      return;
    }

    // Mode 1: Genuine WebGPU / WGSL v4.0 + ScarStore HTTP 206
    if (engineReady) {
      dispatchWebGPUQuery(query);
      return;
    }

    // Auto-boot GPU on first message & queue the prompt!
    pendingQueryRef.current = query;
    if (!engineLoading) {
      await handleBootEngine();
      if (useMainThreadRef.current && mainThreadEngineRef.current && pendingQueryRef.current) {
        const q = pendingQueryRef.current;
        pendingQueryRef.current = null;
        dispatchWebGPUQuery(q);
      }
    }
  };

  const handleStopGeneration = () => {
    if (useMainThreadRef.current && mainThreadAbortRef.current) {
      mainThreadAbortRef.current.abort();
      return;
    }
    workerRef.current?.postMessage({ type: 'abort' });
  };

  const handleResetConversation = () => {
    if (useMainThreadRef.current && mainThreadEngineRef.current) {
      mainThreadEngineRef.current.resetSCache();
    } else {
      workerRef.current?.postMessage({ type: 'reset_scache' });
    }
    setMessages([buildWelcomeMessage(currentLang)]);
  };

  const handleToggleUnlearn = (capsuleId: string) => {
    if (useMainThreadRef.current) {
      const store = mainThreadEngineRef.current?.scarStore || mainThreadRouterRef.current;
      if (store) {
        if (store.unlearnedIds.has(capsuleId)) {
          store.restoreCapsule(capsuleId);
        } else {
          store.unlearnCapsule(capsuleId);
        }
        setCapsules(store.getCapsulesDirectory());
      }
      return;
    }
    workerRef.current?.postMessage({
      type: 'toggle_unlearn',
      payload: { capsuleId },
    });
  };

  const quickPrompts =
    currentLang === 'pt'
      ? [
          { label: '🤖 Quem é você?', query: 'Quem é você e qual é o seu papel neste portfólio?' },
          { label: '👤 Quem é o Julio?', query: 'Quem é o Julio Reis e qual é a experiência e trajetória profissional dele?' },
          { label: '⚡ MoS 72B & SCache O(1)', query: 'O que é o Mixture of Scars (MoS) e o SCache e como reduzem o consumo de VRAM?' },
          { label: '🌐 WebGPU + HTTP 206', query: 'Como você roda 100% no navegador via WebGPU/WGSL e baixa cartuchos via HTTP 206?' },
          { label: '🧩 Motor AURA (0.23 ms)', query: 'Explique o projeto AURA (Motor Sistema 1 de Deep RL no Cubo de Rubik).' },
          { label: '🔥 Landauer (Física)', query: 'Me fale sobre o princípio de Landauer na física da informação.' },
          { label: '📬 Contratar / Contato', query: 'Como posso entrar em contato ou contratar o Julio Reis (e-mail, LinkedIn e GitHub)?' },
        ]
      : [
          { label: '🤖 Who are you?', query: 'Who are you and what is your role on this portfolio?' },
          { label: '👤 Who is Julio?', query: 'Who is Julio Reis and what is his professional background and career?' },
          { label: '⚡ MoS 72B & SCache O(1)', query: 'What is Mixture of Scars (MoS) and SCache, and how do they slash VRAM usage?' },
          { label: '🌐 WebGPU + HTTP 206', query: 'How do you run 100% in-browser via WebGPU/WGSL and stream cartridges via HTTP 206?' },
          { label: '🧩 AURA RL (0.23 ms)', query: 'Explain the AURA Universal System 1 Deep Reinforcement Learning engine.' },
          { label: '🔥 Landauer Bound', query: "Explain Landauer's principle in the physics of information." },
          { label: '📬 Contact / Hire', query: 'How can I contact or hire Julio Reis (email, LinkedIn, and GitHub)?' },
        ];

  const hotCount = capsules.filter((c) => c.cachedInBrowser && !c.unlearnedWiped).length;
  const wipedCount = capsules.filter((c) => c.unlearnedWiped).length;

  return (
    <>
      {/* FLOATING TRIGGER PILL */}
      <div className="fixed bottom-6 right-6 z-40 flex items-center gap-2">
        {!isOpen && (
          <button
            type="button"
            onClick={() => setIsOpen(true)}
            className="group flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-zinc-950/95 hover:bg-zinc-900 border-2 border-[#adff2f]/70 text-white font-mono text-xs font-bold shadow-[0_0_25px_rgba(173,255,47,0.25)] hover:shadow-[0_0_35px_rgba(173,255,47,0.45)] transition-all duration-200 cursor-pointer hover:scale-105"
            title={
              currentLang === 'pt'
                ? 'Testar Agente Nativo WebGPU / WGSL v4.0 + HTTP 206'
                : 'Test Native WebGPU / WGSL v4.0 + HTTP 206 Agent'
            }
          >
            <div className="relative">
              <Bot className="w-4 h-4 text-[#adff2f]" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f] animate-ping" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f]" />
            </div>
            <span>{currentLang === 'pt' ? 'Testar IA no Browser' : 'Test In-Browser AI'}</span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/35">
              WGSL v4 · HTTP 206
            </span>
          </button>
        )}
      </div>

      {/* EXPANDED AGENT STUDIO PANEL */}
      {isOpen && (
        <div
          className={`fixed bottom-5 right-5 z-50 w-[95vw] sm:w-[500px] bg-zinc-950/95 border-2 border-[#adff2f]/70 rounded-2xl shadow-[0_0_45px_rgba(0,0,0,0.9),0_0_30px_rgba(173,255,47,0.2)] backdrop-blur-xl flex flex-col transition-all duration-200 overflow-hidden ${
            isMinimized ? 'h-[62px]' : 'h-[680px] max-h-[90vh]'
          }`}
        >
          {/* TOP HEADER */}
          <div className="flex items-center justify-between px-3.5 py-2.5 bg-zinc-900/95 border-b border-zinc-800 shrink-0">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="relative p-1.5 rounded-lg bg-[#adff2f]/10 border border-[#adff2f]/30 shrink-0">
                <Bot className="w-4 h-4 text-[#adff2f]" />
                <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-pulse" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <h4 className="text-xs font-mono font-bold text-white tracking-tight">
                    System One · WebGPU v4.0
                  </h4>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/40 font-bold">
                    SCache {formatScacheKb(scacheBytes)} O(1)
                  </span>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-cyan-500/15 text-cyan-300 border border-cyan-500/35 font-bold">
                    HTTP 206 · 61.25 KB
                  </span>
                </div>
                <p className="text-[10px] font-mono text-zinc-400 truncate">
                  {engineReady
                    ? `⚡ WGSL Ativo (${adapterLabel}) · Δ +0 B`
                    : `${hasWebGPU ? '✓' : '⚠️'} ${adapterLabel} · Zero ONNX`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => setCurrentLang((prev) => (prev === 'pt' ? 'en' : 'pt'))}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-mono font-bold bg-zinc-800/80 hover:bg-zinc-700 text-zinc-300 hover:text-white border border-zinc-700/60 transition-colors cursor-pointer"
                title={currentLang === 'pt' ? 'Switch UI labels to English' : 'Mudar rótulos para Português'}
              >
                <Globe className="w-3 h-3 text-[#adff2f]" />
                <span className={currentLang === 'pt' ? 'text-[#adff2f]' : 'text-zinc-500'}>PT</span>
                <span className="text-zinc-600">/</span>
                <span className={currentLang === 'en' ? 'text-[#adff2f]' : 'text-zinc-500'}>EN</span>
              </button>
              <button
                type="button"
                onClick={handleResetConversation}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={currentLang === 'pt' ? 'Resetar SCache O(1) e conversa' : 'Reset SCache O(1) & chat'}
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsMinimized(!isMinimized)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={isMinimized ? 'Expand' : 'Minimize'}
              >
                {isMinimized ? <Maximize2 className="w-3.5 h-3.5" /> : <Minimize2 className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={currentLang === 'pt' ? 'Fechar' : 'Close'}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* ENGINE MODE SELECTOR + VIEW TABS */}
              <div className="px-3 py-2 bg-black/80 border-b border-zinc-800 space-y-2 shrink-0">
                {/* 2 Honest Modes: Native WebGPU WGSL v4.0 vs Instant 0 MB S1 CPU Reflex */}
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    type="button"
                    disabled={!hasWebGPU}
                    onClick={() => {
                      setMode('webgpu');
                      if (!engineReady && !engineLoading) {
                        handleBootEngine();
                      }
                    }}
                    className={`relative px-2.5 py-1.5 rounded-lg border text-left transition-all cursor-pointer flex items-center justify-between ${
                      mode === 'webgpu'
                        ? 'bg-[#adff2f]/15 border-[#adff2f] text-white shadow-[0_0_12px_rgba(173,255,47,0.22)]'
                        : 'bg-zinc-900/80 border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 disabled:opacity-45 disabled:cursor-not-allowed'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-1.5">
                        <Cpu className="w-3 h-3 text-[#adff2f]" />
                        <span className="text-[11px] font-mono font-bold">WebGPU · WGSL v4.0</span>
                      </div>
                      <p className="text-[9px] font-mono text-zinc-400 mt-0.5">
                        {engineReady
                          ? `${vramWeightsMiB} MiB GPU · Pronto`
                          : shardsCached
                          ? '✓ Cache Local (0 MB rede)'
                          : 'Qwen 0.5B INT4 · 15 Shaders'}
                      </p>
                    </div>
                    {hasWebGPU && (
                      <span className="px-1.5 py-0.5 rounded bg-[#adff2f] text-black font-mono font-extrabold text-[8px]">
                        {engineReady ? 'LIVE' : 'GPU'}
                      </span>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => setMode('reflex')}
                    className={`px-2.5 py-1.5 rounded-lg border text-left transition-all cursor-pointer flex items-center justify-between ${
                      mode === 'reflex'
                        ? 'bg-amber-400/15 border-amber-400 text-white'
                        : 'bg-zinc-900/80 border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-1.5">
                        <Zap className="w-3 h-3 text-amber-400" />
                        <span className="text-[11px] font-mono font-bold">
                          {currentLang === 'pt' ? 'Reflexo S1 (CPU)' : 'S1 Reflex (CPU)'}
                        </span>
                      </div>
                      <p className="text-[9px] font-mono text-zinc-400 mt-0.5">
                        0 MB · {'<'} 1 ms · Guia Rápido
                      </p>
                    </div>
                    <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-amber-300 border border-amber-400/30 font-mono font-bold text-[8px]">
                      0 MB
                    </span>
                  </button>
                </div>

                {/* View Switcher: Chat vs ScarStore HTTP 206 */}
                <div className="grid grid-cols-2 gap-1.5 pt-0.5">
                  <button
                    type="button"
                    onClick={() => setActiveTab('chat')}
                    className={`py-1 px-2.5 rounded-md font-mono text-[10px] font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                      activeTab === 'chat'
                        ? 'bg-zinc-800 text-[#adff2f] border border-[#adff2f]/50'
                        : 'bg-zinc-900/60 text-zinc-400 hover:text-white border border-zinc-800'
                    }`}
                  >
                    <Sparkles className="w-3 h-3" />
                    <span>{currentLang === 'pt' ? 'Chat Interativo' : 'Interactive Chat'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab('scarstore')}
                    className={`py-1 px-2.5 rounded-md font-mono text-[10px] font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                      activeTab === 'scarstore'
                        ? 'bg-zinc-800 text-cyan-300 border border-cyan-400/50'
                        : 'bg-zinc-900/60 text-zinc-400 hover:text-white border border-zinc-800'
                    }`}
                  >
                    <Database className="w-3 h-3" />
                    <span>
                      ScarStore HTTP 206 ({capsules.length || 13})
                      {hotCount > 0 ? ` · ${hotCount} RAM` : ''}
                    </span>
                  </button>
                </div>
              </div>

              {/* GPU BOOT BANNER OR SHARD PROGRESS BAR */}
              {mode === 'webgpu' && !engineReady && !engineLoading && hasWebGPU && (
                <div className="px-3.5 py-2.5 bg-zinc-900/90 border-b border-zinc-800 flex items-center justify-between gap-2 shrink-0">
                  <div className="text-[10px] font-mono text-zinc-300 leading-snug">
                    <span className="text-[#adff2f] font-bold">
                      {shardsCached
                        ? currentLang === 'pt'
                          ? '⚡ Pesos MoS INT4 já estão salvos no Cache Persistente (0 MB rede)!'
                          : '⚡ MoS INT4 shards already saved in Persistent Cache (0 MB network)!'
                        : currentLang === 'pt'
                        ? '🚀 Motor WGSL pronto para compilar na sua GPU (cache persistente único).'
                        : '🚀 WGSL engine ready to compile on your GPU (one-time persistent cache).'}
                    </span>
                    <div className="text-zinc-400 text-[9px] mt-0.5">
                      {currentLang === 'pt'
                        ? 'Clique ao lado ou envie qualquer pergunta abaixo para iniciar.'
                        : 'Click alongside or send any prompt below to boot.'}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={handleBootEngine}
                    className="px-3 py-1.5 rounded-lg bg-[#adff2f] hover:bg-lime-300 text-black font-mono text-[10px] font-extrabold shrink-0 cursor-pointer shadow-[0_0_15px_rgba(173,255,47,0.3)]"
                  >
                    {shardsCached
                      ? currentLang === 'pt'
                        ? '⚡ Ativar GPU'
                        : '⚡ Wake GPU'
                      : currentLang === 'pt'
                      ? '⚡ Compilar GPU'
                      : '⚡ Boot GPU'}
                  </button>
                </div>
              )}

              {engineLoading && (
                <div className="p-3 bg-zinc-900/95 border-b border-zinc-800 shrink-0">
                  <div className="flex items-center justify-between text-[11px] font-mono mb-1.5">
                    <span className="text-white font-bold flex items-center gap-1.5 truncate">
                      <DownloadCloud className="w-3.5 h-3.5 text-[#adff2f] animate-pulse shrink-0" />
                      <span className="truncate">
                        {loadPhaseDetail ||
                          (currentLang === 'pt'
                            ? 'Inicializando 15 pipelines WGSL...'
                            : 'Compiling 15 WGSL pipelines...')}
                      </span>
                    </span>
                    <span className="text-[#adff2f] font-bold shrink-0 ml-2">
                      {loadPercent}%
                      {loadLoadedMiB > 0 && !loadedFromCache ? ` (${loadLoadedMiB}/${loadTotalMiB} MiB)` : ''}
                    </span>
                  </div>
                  <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-[#adff2f] h-full transition-all duration-200 shadow-[0_0_8px_#adff2f]"
                      style={{ width: `${Math.max(4, loadPercent)}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-[9px] font-mono text-zinc-400 mt-1.5">
                    <span>
                      {loadedFromCache
                        ? currentLang === 'pt'
                          ? '⚡ 0 MB de rede: carregando direto do CacheStorage para a VRAM.'
                          : '⚡ 0 MB network: streaming from persistent CacheStorage into VRAM.'
                        : currentLang === 'pt'
                        ? 'Alocando SCache O(1) fixo (2733.6 KB) + 15 shards MoS INT4 (< 44 MiB).'
                        : 'Allocating fixed SCache O(1) (2733.6 KB) + 15 MoS INT4 shards (< 44 MiB).'}
                    </span>
                    <span className="text-emerald-400 flex items-center gap-0.5 shrink-0">
                      <ShieldCheck className="w-2.5 h-2.5" />
                      <span>Cache v4</span>
                    </span>
                  </div>
                </div>
              )}

              {errorMessage && (
                <div className="px-3 py-2 bg-rose-950/70 border-b border-rose-800 text-rose-200 text-[10px] font-mono flex items-center justify-between gap-2 shrink-0">
                  <div className="flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    <span>{errorMessage}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setErrorMessage(null);
                      setMode('reflex');
                    }}
                    className="underline text-white font-bold shrink-0 cursor-pointer"
                  >
                    {currentLang === 'pt' ? 'Usar Reflexo S1' : 'Use S1 Reflex'}
                  </button>
                </div>
              )}

              {/* ==================== VIEW 1: CHAT ==================== */}
              {activeTab === 'chat' ? (
                <>
                  <div className="flex-1 overflow-y-auto p-3.5 space-y-3 font-sans text-xs">
                    {messages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                      >
                        <div
                          className={`max-w-[91%] p-3 rounded-xl leading-relaxed ${
                            msg.sender === 'user'
                              ? 'bg-[#adff2f] text-black font-medium shadow-md shadow-lime-400/10'
                              : 'bg-zinc-900/95 text-zinc-100 border border-zinc-800'
                          }`}
                        >
                          <p className="whitespace-pre-line">
                            {msg.text ||
                              (msg.sender === 'agent' && isGenerating
                                ? currentLang === 'pt'
                                  ? 'Roteando Camada 15 (P_⊥lang) & decodificando em WGSL...'
                                  : 'Routing Layer 15 (P_⊥lang) & decoding in WGSL...'
                                : '')}
                          </p>

                          {/* Contextual Portfolio Links */}
                          {msg.links && msg.links.length > 0 && (
                            <div className="mt-2.5 pt-2 border-t border-zinc-800/90 flex flex-wrap gap-2">
                              {msg.links.map((link, idx) => (
                                <a
                                  key={idx}
                                  href={link.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-950/90 border border-[#adff2f]/35 text-[10px] font-mono font-bold text-[#adff2f] hover:bg-[#adff2f]/15 transition-colors"
                                >
                                  {link.icon === 'colab' && <PlayCircle className="w-3 h-3" />}
                                  {link.icon === 'github' && <Github className="w-3 h-3" />}
                                  {(!link.icon || link.icon === 'ext') && <ExternalLink className="w-3 h-3" />}
                                  <span>{link.label}</span>
                                </a>
                              ))}
                            </div>
                          )}

                          {/* Real WGSL + SCache O(1) + HTTP 206 Telemetry Pills */}
                          {msg.telemetry && (
                            <div className="mt-2.5 pt-2 border-t border-zinc-800/80 flex flex-wrap gap-1 text-[9px] font-mono">
                              <span className="px-1.5 py-0.5 rounded bg-[#adff2f]/15 text-[#adff2f] border border-[#adff2f]/30 font-bold">
                                SCache O(1): {formatScacheKb(msg.telemetry.scacheBytes)} (Δ{' '}
                                {msg.telemetry.scacheDeltaBytes >= 0
                                  ? `+${msg.telemetry.scacheDeltaBytes}`
                                  : msg.telemetry.scacheDeltaBytes}{' '}
                                B)
                              </span>
                              {msg.telemetry.tokPerSec > 0 && (
                                <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-bold">
                                  {msg.telemetry.tokPerSec.toFixed(1)} tok/s WGSL
                                </span>
                              )}
                              {msg.telemetry.activeCapsules.length > 0 ? (
                                msg.telemetry.activeCapsules.map((cap) => (
                                  <span
                                    key={cap.capsuleId}
                                    className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-200 border border-purple-400/40 font-bold"
                                  >
                                    🧩 {cap.capsuleId} (g={cap.gate.toFixed(2)}
                                    {cap.fromCache ? ' · RAM' : ' · 206'})
                                  </span>
                                ))
                              ) : (
                                <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                                  0 cartuchos (Invariância 0.0000000000)
                                </span>
                              )}
                              <span className="px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                                HTTP 206: {(msg.telemetry.turnHttp206Bytes / 1024).toFixed(2)} KB
                              </span>
                              {msg.telemetry.confidence > 0 && (
                                <span className="px-1.5 py-0.5 rounded bg-zinc-800/90 text-zinc-300 border border-zinc-700">
                                  S1 p₁={msg.telemetry.confidence.toFixed(3)}
                                </span>
                              )}
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-2 mt-1 px-1 text-[9px] font-mono text-zinc-500">
                          <span>{msg.timestamp}</span>
                          {msg.sender === 'agent' && !msg.isSystemNotice && (
                            <span className={msg.modeUsed === 'webgpu' ? 'text-[#adff2f]' : 'text-amber-400'}>
                              {msg.modeUsed === 'webgpu'
                                ? '⚡ WebGPU · 15 WGSL Shaders'
                                : '🏹 Reflexo S1 · CPU < 1 ms (0 MB)'}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>

                  {/* Quick Prompt Chips */}
                  <div className="px-3 py-1.5 bg-zinc-900/70 border-t border-zinc-800 overflow-x-auto flex items-center gap-1.5 no-scrollbar shrink-0">
                    {quickPrompts.map((p, idx) => (
                      <button
                        key={idx}
                        type="button"
                        disabled={isGenerating}
                        onClick={() => handleSendMessage(p.query)}
                        className="shrink-0 text-[10px] font-mono px-2.5 py-1 rounded-full bg-zinc-950 hover:bg-zinc-800 text-zinc-300 hover:text-[#adff2f] border border-zinc-800 hover:border-[#adff2f]/40 transition-colors cursor-pointer disabled:opacity-50"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                /* ==================== VIEW 2: SCARSTORE HTTP 206 INSPECTOR ==================== */
                <div className="flex-1 overflow-y-auto p-3 space-y-2.5 font-mono text-[11px]">
                  <div className="p-2.5 rounded-xl bg-zinc-900/90 border border-cyan-500/30 text-zinc-300 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-cyan-300 font-bold flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5" />
                        <span>Golden-Slice ScarStore v4.0 (P_⊥lang · k=14)</span>
                      </span>
                      <span className="text-[9px] text-[#adff2f] font-bold">
                        {hotCount} RAM · {wipedCount} LGPD 0x00
                      </span>
                    </div>
                    <p className="text-[10px] text-zinc-400 leading-snug font-sans">
                      {currentLang === 'pt'
                        ? 'Cada cartucho pesa exatos 61.25 KB (Block-64 INT4), ancorado apenas em inglês e roteado na Camada 15. Clique em 🇧🇷 PT, 🇺🇸 EN, 🇪🇸 ES ou 🇫🇷 FR para baixar via HTTP 206 e injetar na GPU!'
                        : 'Each cartridge is exactly 61.25 KB (Block-64 INT4), anchored strictly in English and routed at Layer 15. Click PT, EN, ES, or FR to fetch via HTTP 206 and inject on GPU!'}
                    </p>
                  </div>

                  {capsules.map((cap) => {
                    const poly = cap.metadata?.prompts_polyglot || {};
                    const isWiped = !!cap.unlearnedWiped;
                    const isHot = !!cap.cachedInBrowser && !isWiped;
                    const title =
                      currentLang === 'pt'
                        ? cap.metadata?.title_pt || cap.metadata?.title || cap.capsule_id
                        : cap.metadata?.title || cap.capsule_id;

                    return (
                      <div
                        key={cap.capsule_id}
                        className={`p-2.5 rounded-xl border transition-all ${
                          isWiped
                            ? 'bg-rose-950/20 border-rose-800/50 opacity-75'
                            : isHot
                            ? 'bg-zinc-900/95 border-[#adff2f]/50'
                            : 'bg-zinc-900/75 border-zinc-800 hover:border-zinc-700'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="px-1.5 py-0.2 rounded bg-zinc-800 text-zinc-300 text-[9px]">
                                {cap.metadata?.category || 'Concept'}
                              </span>
                              <span
                                className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
                                  isWiped
                                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                    : isHot
                                    ? 'bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/40'
                                    : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                                }`}
                              >
                                {isWiped ? '🗑️ LGPD 0x00' : isHot ? '⚡ Hot RAM' : '☁️ HTTP 206 · 61.25 KB'}
                              </span>
                              {typeof cap.lastCosine === 'number' && (
                                <span className="text-[9px] text-zinc-400">
                                  cos={cap.lastCosine.toFixed(3)} (τ={cap.tau.toFixed(2)})
                                </span>
                              )}
                            </div>
                            <div className="text-white font-bold text-[11px] mt-1 font-sans">{title}</div>
                            {cap.metadata?.summary && (
                              <p className="text-[10px] text-zinc-400 mt-0.5 leading-snug font-sans">
                                {cap.metadata.summary}
                              </p>
                            )}
                          </div>

                          <button
                            type="button"
                            onClick={() => handleToggleUnlearn(cap.capsule_id)}
                            className={`px-2 py-1 rounded text-[9px] font-bold flex items-center gap-1 shrink-0 cursor-pointer border ${
                              isWiped
                                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/25'
                                : 'bg-rose-500/10 text-rose-300 border-rose-500/30 hover:bg-rose-500/20'
                            }`}
                            title={
                              isWiped
                                ? 'Restaurar cartucho para re-download HTTP 206'
                                : 'LGPD < 0.2 ms: sobrescrever bytes na RAM com 0x00'
                            }
                          >
                            {isWiped ? (
                              <>
                                <RotateCcw className="w-2.5 h-2.5" />
                                <span>Restore</span>
                              </>
                            ) : (
                              <>
                                <Trash2 className="w-2.5 h-2.5" />
                                <span>0x00</span>
                              </>
                            )}
                          </button>
                        </div>

                        {/* 4 Polyglot Test Buttons */}
                        <div className="flex items-center gap-1.5 mt-2 pt-2 border-t border-zinc-800/80 flex-wrap">
                          <span className="text-[9px] text-zinc-500 mr-0.5">
                            {currentLang === 'pt' ? 'Testar em:' : 'Test in:'}
                          </span>
                          {(
                            [
                              ['pt', '🇧🇷 PT'],
                              ['en', '🇺🇸 EN'],
                              ['es', '🇪🇸 ES'],
                              ['fr', '🇫🇷 FR'],
                            ] as const
                          ).map(([code, label]) => {
                            const promptStr = poly[code];
                            if (!promptStr) return null;
                            return (
                              <button
                                key={code}
                                type="button"
                                disabled={isGenerating}
                                onClick={() =>
                                  handleSendMessage(promptStr, hasWebGPU ? 'webgpu' : 'reflex')
                                }
                                className="px-2 py-0.5 rounded bg-zinc-950 hover:bg-[#adff2f] text-zinc-200 hover:text-black border border-zinc-700 hover:border-[#adff2f] text-[9px] font-bold transition-colors cursor-pointer disabled:opacity-45"
                                title={promptStr}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}

                  {/* Live HTTP 206 Byte-Range Audit Log */}
                  {httpEvents.length > 0 && (
                    <div className="p-2.5 rounded-xl bg-black/90 border border-zinc-800 space-y-1">
                      <div className="text-[10px] font-bold text-cyan-300">
                        📡 Log Real RFC 7233 HTTP 206 Partial Content
                      </div>
                      {httpEvents.slice(0, 6).map((ev, idx) => (
                        <div key={idx} className="text-[9px] text-zinc-400 flex items-center justify-between gap-2">
                          <span className="truncate text-zinc-200">
                            [{ev.status}] {ev.capsuleId} · {ev.range}
                          </span>
                          <span className="text-[#adff2f] shrink-0">
                            {ev.kb} KB · {ev.latencyMs} ms
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* FOOTER STATUS + INPUT BAR */}
              <div className="px-3 py-1 bg-black/95 border-t border-zinc-800/90 flex items-center justify-between text-[9px] font-mono text-zinc-400 shrink-0">
                <span className="flex items-center gap-1 text-emerald-400">
                  <ShieldCheck className="w-3 h-3" />
                  <span>
                    {mode === 'webgpu'
                      ? `Zero ONNX · SCache ${formatScacheKb(scacheBytes)} (Δ +0 B)`
                      : 'Reflexo S1 CPU · 0 MB Rede'}
                  </span>
                </span>
                <span className="text-zinc-300">
                  {lastTokPerSec ? `⚡ ${lastTokPerSec.toFixed(1)} tok/s WGSL` : 'P_⊥lang Layer 15 · PT/EN/ES/FR'}
                </span>
              </div>

              <div className="p-3 bg-zinc-950 border-t border-zinc-800 flex items-center gap-2 shrink-0">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder={
                    mode === 'webgpu'
                      ? currentLang === 'pt'
                        ? 'Pergunte em PT, EN, ES ou FR (GPU WGSL + HTTP 206)...'
                        : 'Ask in PT, EN, ES, or FR (WGSL GPU + HTTP 206)...'
                      : currentLang === 'pt'
                      ? 'Pergunte sobre Julio, MoS 72B, AURA...'
                      : 'Ask about Julio, MoS 72B, AURA...'
                  }
                  className="flex-1 bg-zinc-900 border border-zinc-800 focus:border-[#adff2f] rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none transition-colors font-sans"
                />
                {isGenerating ? (
                  <button
                    type="button"
                    onClick={handleStopGeneration}
                    className="p-2 bg-rose-500 hover:bg-rose-400 text-white rounded-lg transition-colors cursor-pointer shrink-0"
                    title={currentLang === 'pt' ? 'Parar decodificação' : 'Stop decoding'}
                  >
                    <Square className="w-3.5 h-3.5" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleSendMessage()}
                    disabled={!inputMessage.trim()}
                    className="p-2 bg-[#adff2f] hover:bg-lime-300 disabled:bg-zinc-800 disabled:text-zinc-600 text-black rounded-lg transition-colors cursor-pointer shrink-0"
                    title={currentLang === 'pt' ? 'Enviar' : 'Send'}
                  >
                    <Send className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};
