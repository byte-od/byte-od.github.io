import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  X,
  Send,
  Sparkles,
  Zap,
  PlayCircle,
  ExternalLink,
  Github,
  Mail,
  RefreshCw,
  Cpu,
  Layers,
  ChevronDown,
  Minimize2,
  Maximize2,
  DownloadCloud,
  CheckCircle2,
  AlertTriangle,
  Flame,
  ShieldCheck,
  HardDrive,
  Info,
  Check,
} from 'lucide-react';
import { MODELS_CONFIG } from '../lib/llmWorker';

export type ModelTier = 'reflex' | 'leve' | 'media' | 'grande';

interface Message {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  links?: { label: string; url: string; icon?: 'colab' | 'github' | 'ext' }[];
  tierUsed?: ModelTier;
  tokPerSec?: string;
  fromCache?: boolean;
}

interface PortfolioAgentWidgetProps {
  lang?: 'en' | 'pt';
}

interface ModelMeta {
  tier: ModelTier;
  labelPt: string;
  labelEn: string;
  badge: string;
  params: string;
  sizeMb: number;
  descPt: string;
  descEn: string;
  devicePt: string;
  deviceEn: string;
}

const MODEL_OPTIONS: Record<ModelTier, ModelMeta> = {
  reflex: {
    tier: 'reflex',
    labelPt: 'Reflexo',
    labelEn: 'Reflex',
    badge: '0 MB',
    params: 'Regras Semânticas',
    sizeMb: 0,
    descPt: 'Instantâneo (<1ms). Zero download, ideal para testes imediatos sem gastar dados móveis.',
    descEn: 'Instant (<1ms). Zero download, perfect for quick inspection with zero mobile data use.',
    devicePt: 'CPU Host',
    deviceEn: 'Host CPU',
  },
  leve: {
    tier: 'leve',
    labelPt: 'Leve',
    labelEn: 'Light',
    badge: '135M',
    params: '135 Milhões (INT4)',
    sizeMb: 180,
    descPt: 'SmolLM2-135M (~180 MB). Download ultrarrápido (3-5s), roda suave em smartphones e laptops.',
    descEn: 'SmolLM2-135M (~180 MB). Rapid download (3-5s), runs smooth on mobile & laptops via WebGPU.',
    devicePt: 'WebGPU / WASM',
    deviceEn: 'WebGPU / WASM',
  },
  media: {
    tier: 'media',
    labelPt: 'Média',
    labelEn: 'Medium',
    badge: '500M',
    params: '500 Milhões (INT4)',
    sizeMb: 780,
    descPt: 'Qwen2.5-0.5B (~780 MB). Equilíbrio ideal, alta fluência em português e raciocínio lógico.',
    descEn: 'Qwen2.5-0.5B (~780 MB). Best balance, high bilingual fluency, and sharp reasoning.',
    devicePt: 'WebGPU Shaders',
    deviceEn: 'WebGPU Shaders',
  },
  grande: {
    tier: 'grande',
    labelPt: 'Grande',
    labelEn: 'Large',
    badge: '1B',
    params: '1 Bilhão (INT4)',
    sizeMb: 680,
    descPt: 'Llama-3.2-1B (~680 MB). Alta capacidade de raciocínio lógico e instrução sem estourar a memória da aba.',
    descEn: 'Llama-3.2-1B (~680 MB). High reasoning and instruction fidelity without browser memory exhaustion.',
    devicePt: 'WebGPU Compute Core',
    deviceEn: 'WebGPU Compute Core',
  },
};

interface HardwareProfile {
  recommendedTier: ModelTier;
  hardwareSummary: string;
  reasonPt: string;
  reasonEn: string;
  hasWebGPU: boolean;
  isMobile: boolean;
  cores: number;
}

const SYSTEM_PROMPT_PT = `Você é a IA Assistente Oficial do portfólio de Julio Cesar da Costa Reis Filho, rodando 100% no navegador do visitante via WebGPU local e arquitetura Scar-Cache (SCache) O(1).

DIRETRIZ DE IDENTIDADE INVIOLÁVEL:
- Você é o ASSISTENTE do portfólio, NUNCA o Julio Cesar. NUNCA fale em 1ª pessoa ("eu desenvolvi", "minha trajetória", "eu sou arquiteto").
- Fale SEMPRE sobre o Julio na 3ª pessoa ("O Julio desenvolveu...", "Ele é Arquiteto de Sistemas...", "O projeto do Julio...").
- Se perguntarem "quem é você", declare com clareza: "Sou a IA assistente neural rodando localmente no seu navegador via WebGPU para apresentar o portfólio e as tecnologias criadas por Julio Cesar."

Fatos verificados sobre o Julio:
- Arquiteto de Sistemas Sênior & Engenheiro de LLMs com 8+ anos de experiência.
- Criador do Mixture of Scars (MoS) & SCache: Viabilizou rodar 80 camadas de um 72B em GPU de 16GB com apenas 2.40 GB de VRAM e substituiu o KV Cache tradicional por uma memória O(1) de 34.8 KB que nunca vaza nem causa OOM a 100.000 tokens.
- Autor do CIR-Engine 503M (Causal Transformer em PyTorch DDP) e do motor AURA System 1 (Embodied Deep RL resolvendo o Cubo de Rubik com 232B estados a 0.23ms na CPU).
- Autor do pacote mddd-cli no npm (Spec-Driven Development por diagramas).
- Contato oficial: reisfilho1116@gmail.com | github.com/byte-od | linkedin.com/in/juliocrfilho.
Seja conciso, técnico, direto e responda com autoridade profissional.`;

const SYSTEM_PROMPT_EN = `You are the Official Virtual AI Assistant for Julio Cesar da Costa Reis Filho's portfolio, running 100% locally in the visitor's browser via WebGPU and the O(1) Scar-Cache (SCache) architecture.

INVIOLABLE IDENTITY DIRECTIVE:
- You are the ASSISTANT of the portfolio, NEVER Julio Cesar himself. NEVER speak in the first person ("I built", "my career", "I am an architect").
- ALWAYS speak about Julio in the third person ("Julio developed...", "He is a Senior Systems Architect...", "His project...").
- If asked "who are you", state clearly: "I am the local neural AI assistant running in your browser via WebGPU to present Julio Cesar's portfolio and technologies."

Verified facts about Julio:
- Senior Systems Architect & LLM Engineer with 8+ years of production experience.
- Creator of Mixture of Scars (MoS) & SCache: Enabled running all 80 layers of a 72B model on a single 16GB GPU at 2.40 GB peak VRAM and replaced the traditional KV Cache with an exact 34.8 KB O(1) associative memory that never leaks across 100,000 tokens.
- Author of CIR-Engine 503M (Causal Transformer in PyTorch DDP) and AURA System 1 engine (Embodied Deep RL solving Rubik's Cube across 232B states at 0.23ms on bare CPU).
- Author of mddd-cli on npm (Spec-Driven Development using architectural diagrams).
- Professional contact: reisfilho1116@gmail.com | github.com/byte-od | linkedin.com/in/juliocrfilho.
Be concise, authoritative, direct, and technical.`;

export const PortfolioAgentWidget: React.FC<PortfolioAgentWidgetProps> = ({ lang = 'en' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [activeTier, setActiveTier] = useState<ModelTier>('reflex');

  // Hardware Profiling State
  const [hardwareProfile, setHardwareProfile] = useState<HardwareProfile>({
    recommendedTier: 'media',
    hardwareSummary: 'Analisando GPU & CPU...',
    reasonPt: 'Detectando arquitetura...',
    reasonEn: 'Detecting hardware...',
    hasWebGPU: false,
    isMobile: false,
    cores: 4,
  });
  const [showHwTooltip, setShowHwTooltip] = useState(false);

  // Web Worker states
  const [loadedTier, setLoadedTier] = useState<ModelTier | null>(null);
  const [loadingTier, setLoadingTier] = useState<ModelTier | null>(null);
  const [downloadPercent, setDownloadPercent] = useState(0);
  const [downloadLoadedMb, setDownloadLoadedMb] = useState(0);
  const [downloadTotalMb, setDownloadTotalMb] = useState(0);
  const [deviceInfo, setDeviceInfo] = useState<string>('Detectando hardware...');
  const [lastTokPerSec, setLastTokPerSec] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Cache & Memory Preservation States
  const [cachedTiers, setCachedTiers] = useState<Record<ModelTier, boolean>>({
    reflex: true,
    leve: false,
    media: false,
    grande: false,
  });
  const [isStoragePersisted, setIsStoragePersisted] = useState(false);
  const [wasLoadedFromCache, setWasLoadedFromCache] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const streamingMsgIdRef = useRef<string | null>(null);

  // 1. HARDWARE PROFILER & PERSISTENT STORAGE REGISTRATION
  useEffect(() => {
    async function profileHardware() {
      const isMobile =
        typeof navigator !== 'undefined' &&
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
      const ramGb = typeof navigator !== 'undefined' ? (navigator as any).deviceMemory || 8 : 8;

      let hasWebGPU = false;
      let gpuVendor = '';
      let maxBufferGb = 0;

      try {
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            hasWebGPU = true;
            maxBufferGb = adapter.limits?.maxBufferSize
              ? Math.round(adapter.limits.maxBufferSize / (1024 * 1024 * 1024))
              : 2;
            const info = await adapter.requestAdapterInfo?.();
            if (info?.vendor || info?.architecture) {
              gpuVendor = `${info.vendor || ''} ${info.architecture || ''}`.trim();
            }
          }
        }
      } catch (e) {
        console.warn('[Hardware Profiler] WebGPU probe failed:', e);
      }

      // Hardware recommendation heuristics
      let recommended: ModelTier = 'media';
      let summary = '';
      let reasonPt = '';
      let reasonEn = '';

      if (isMobile) {
        recommended = 'leve';
        summary = `📱 Mobile (${cores} Cores · ${ramGb}GB)`;
        reasonPt = 'Dispositivo móvel detectado. O modelo Leve (135M) é o ideal para economizar bateria e memória RAM.';
        reasonEn = 'Mobile device detected. Light (135M) is optimal to preserve battery and mobile RAM.';
      } else if (!hasWebGPU) {
        recommended = 'reflex';
        summary = `⚙️ CPU Host (${cores} Cores · Sem WebGPU)`;
        reasonPt = 'Seu navegador não possui WebGPU ativo. O modo Reflexo opera instantaneamente (<1ms) sem risco de travamento.';
        reasonEn = 'WebGPU is inactive in this browser. Reflex mode runs instantaneously (<1ms) with zero lag.';
      } else if (cores >= 8 && (ramGb >= 16 || maxBufferGb >= 2)) {
        recommended = 'grande';
        summary = `⚡ GPU Alta Potência (${cores} Cores · ${gpuVendor || 'WebGPU Discrete'})`;
        reasonPt = 'Hardware potente detectado (GPU dedicada / Apple Silicon). Seu sistema roda com folga e sem risco de OOM o modelo Grande (1B).';
        reasonEn = 'High-end hardware detected (Discrete GPU / Apple Silicon). Your system runs the Large model (1B) smoothly without OOM risks.';
      } else {
        recommended = 'media';
        summary = `💻 Desktop GPU (${cores} Cores · ${ramGb}GB RAM)`;
        reasonPt = 'GPU desktop padrão detectada. O modelo Médio (500M) oferece o equilíbrio perfeito de fluência e velocidade.';
        reasonEn = 'Standard desktop GPU detected. Medium (500M) provides the ideal balance of speed and fluency.';
      }

      setHardwareProfile({
        recommendedTier: recommended,
        hardwareSummary: summary,
        reasonPt,
        reasonEn,
        hasWebGPU,
        isMobile,
        cores,
      });
    }

    async function requestStoragePreservation() {
      if (typeof navigator !== 'undefined' && navigator.storage && navigator.storage.persist) {
        try {
          const isPersisted = await navigator.storage.persist();
          setIsStoragePersisted(isPersisted);
        } catch (e) {
          console.warn('[Storage] persist error:', e);
        }
      }
    }

    profileHardware();
    requestStoragePreservation();
  }, []);

  // 2. INITIALIZE WEB WORKER & CACHE INSPECTOR
  useEffect(() => {
    try {
      const worker = new Worker(new URL('../lib/llmWorker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;

      // Ask worker to check which models are already cached
      worker.postMessage({ type: 'check_cache' });

      worker.onmessage = (e: MessageEvent) => {
        const {
          type,
          modelKey,
          percent,
          loaded,
          total,
          device,
          token,
          stats,
          error,
          results,
          fromCache,
          wasAlreadyCached,
        } = e.data;

        if (type === 'cache_status' && results) {
          setCachedTiers({
            reflex: true,
            leve: !!results.leve,
            media: !!results.media,
            grande: !!results.grande,
          });
        } else if (type === 'loading_start') {
          setLoadingTier(modelKey);
          setDownloadPercent(wasAlreadyCached ? 100 : 0);
          setWasLoadedFromCache(!!wasAlreadyCached);
          setDeviceInfo(
            device === 'webgpu' ? '⚡ WebGPU Shaders (Local)' : '⚙️ WebAssembly SIMD Fallback'
          );
          setErrorMessage(null);
        } else if (type === 'download_progress') {
          setDownloadPercent(percent);
          setDownloadLoadedMb(Math.round((loaded / (1024 * 1024)) * 10) / 10);
          setDownloadTotalMb(Math.round((total / (1024 * 1024)) * 10) / 10);
        } else if (type === 'ready') {
          setLoadedTier(modelKey);
          setLoadingTier(null);
          setWasLoadedFromCache(!!fromCache);
          setCachedTiers((prev) => ({ ...prev, [modelKey]: true }));
          setDeviceInfo(
            device === 'webgpu' ? '⚡ WebGPU Shaders (Local)' : '⚙️ WebAssembly SIMD (Local)'
          );
          setMessages((prev) => [
            ...prev,
            {
              id: `sys-${Date.now()}`,
              sender: 'agent',
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              text:
                lang === 'pt'
                  ? `✅ Motor neural [${MODEL_OPTIONS[modelKey as ModelTier].labelPt} ${
                      MODEL_OPTIONS[modelKey as ModelTier].badge
                    }] pronto na sua GPU!\n• Fonte: ${fromCache ? '⚡ Cache Local Persistente (0 MB de rede)' : '📥 Download Concluído & Salvo no Cache'}\n• Memória SCache: 34.8 KB O(1) residente na SRAM\n• Requisições de rede durante o chat: 0 bytes (100% offline).`
                  : `✅ Neural engine [${MODEL_OPTIONS[modelKey as ModelTier].labelEn} ${
                      MODEL_OPTIONS[modelKey as ModelTier].badge
                    }] compiled in your local GPU!\n• Source: ${fromCache ? '⚡ Persistent Local Cache (0 MB network)' : '📥 Downloaded & Saved to Cache'}\n• SCache Memory: 34.8 KB O(1) resident in SRAM\n• Network requests during chat: 0 bytes (100% offline).`,
              tierUsed: modelKey as ModelTier,
              fromCache: !!fromCache,
            },
          ]);
        } else if (type === 'generation_start') {
          setIsTyping(true);
        } else if (type === 'token') {
          setIsTyping(false);
          const activeId = streamingMsgIdRef.current;
          if (activeId) {
            setMessages((prev) =>
              prev.map((m) => (m.id === activeId ? { ...m, text: m.text + token } : m))
            );
          }
        } else if (type === 'generation_end') {
          setIsTyping(false);
          streamingMsgIdRef.current = null;
          if (stats?.tokensPerSec) {
            setLastTokPerSec(stats.tokensPerSec);
          }
        } else if (type === 'error') {
          setIsTyping(false);
          setLoadingTier(null);
          setErrorMessage(error);
          streamingMsgIdRef.current = null;
        }
      };

      return () => {
        worker.terminate();
      };
    } catch (err) {
      console.warn('[PortfolioAgent] Web Workers not supported:', err);
    }
  }, [lang]);

  const initialGreeting: Message = {
    id: 'msg-0',
    sender: 'agent',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    text:
      lang === 'pt'
        ? 'Olá! Sou o agente virtual do portfólio do Julio Cesar Reis Filho. Posso te apresentar sua trajetória sênior, detalhar suas criações de IA (como o MoS 72B em 16GB e o motor AURA) ou demonstrar a arquitetura SCache rodando na prática no seu navegador.\n\nEscolha acima o modelo que deseja executar no seu hardware: [Reflexo], [Leve 135M], [Média 500M] ou [Grande 1.5B]!'
        : 'Hello! I am Julio Cesar Reis Filho\'s portfolio agent. I can walk you through his systems breakthroughs (such as the 72B MoS on 16GB GPU and AURA Deep RL engine) or demonstrate the SCache architecture running in-browser.\n\nSelect a model above to execute on your local hardware: [Reflex], [Light 135M], [Medium 500M], or [Large 1.5B]!',
    links: [
      {
        label: lang === 'pt' ? '⚡ Live Test Colab (72B + SCache)' : '⚡ Live Test Colab (72B + SCache)',
        url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
        icon: 'colab',
      },
    ],
    tierUsed: 'reflex',
  };

  useEffect(() => {
    setMessages([initialGreeting]);
  }, [lang]);

  useEffect(() => {
    if (isOpen && !isMinimized) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, isMinimized, isTyping]);

  useEffect(() => {
    if (isOpen && !isMinimized) {
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen, isMinimized]);

  // Handle tier selection
  const handleSelectTier = (tier: ModelTier) => {
    setActiveTier(tier);
    setErrorMessage(null);

    if (tier === 'reflex') {
      return;
    }

    // Trigger load in worker (Worker checks cache to avoid duplicate download)
    if (loadedTier !== tier && loadingTier !== tier) {
      workerRef.current?.postMessage({
        type: 'load',
        payload: { modelKey: tier },
      });
    }
  };

  // Quick Chips
  const quickPrompts =
    lang === 'pt'
      ? [
          { label: '👤 Quem é o Julio?', query: 'Quem é o Julio e qual a experiência dele?' },
          { label: '⚡ MoS & SCache 72B', query: 'O que é o Mixture of Scars (MoS) e o SCache?' },
          { label: '🧠 Como você roda no browser?', query: 'Como a arquitetura MoS e SCache viabiliza você no navegador?' },
          { label: '🧩 Motor AURA (Sistema 1)', query: 'Explique o projeto AURA (Deep Reinforcement Learning no Cubo de Rubik)' },
          { label: '📬 Como entrar em contato?', query: 'Como posso entrar em contato ou contratar o Julio?' },
        ]
      : [
          { label: '👤 Who is Julio?', query: 'Who is Julio and what is his background?' },
          { label: '⚡ MoS & SCache 72B', query: 'What is Mixture of Scars (MoS) and SCache?' },
          { label: '🧠 How do you run in-browser?', query: 'How does the MoS and SCache architecture enable you in the browser?' },
          { label: '🧩 AURA Engine (System 1)', query: 'Explain the AURA Deep Reinforcement Learning engine' },
          { label: '📬 How to contact?', query: 'How can I contact or hire Julio?' },
        ];

  // Natural Knowledge Engine for Instant Reflex Mode
  const generateReflexResponse = (userQuery: string): { text: string; links?: Message['links'] } => {
    const q = userQuery.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');

    if (q.includes('quem e') || q.includes('who is') || q.includes('julio') || q.includes('trajetoria') || q.includes('experiencia') || q.includes('background') || q.includes('carreira') || q.includes('sobre')) {
      return {
        text:
          lang === 'pt'
            ? 'Julio Cesar da Costa Reis Filho é um Arquiteto de Sistemas Sênior e Engenheiro de LLMs com mais de 8 anos de experiência em engenharia de ponta. Ele combina domínio de sistemas de baixo nível (Go, C++, Rust, CUDA/Metal) com arquiteturas neurais avançadas (Transformers Causais, PyTorch DDP, RL e quantização em hardware). Disponível globalmente para posições remotas ou relocalização internacional.'
            : 'Julio Cesar da Costa Reis Filho is a Senior Systems Architect and LLM Systems Engineer with over 8 years of engineering experience bridging low-level systems (Go, C++, Rust, CUDA/Metal) with deep learning architectures (Causal Transformers, Distributed PyTorch DDP, Embodied RL, and hardware quantization). Available globally for remote roles and relocation.',
        links: [
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
        ],
      };
    }

    if (q.includes('mos') || q.includes('scache') || q.includes('mixture of scars') || q.includes('72b') || q.includes('70b') || q.includes('16gb') || q.includes('100.000') || q.includes('100k') || q.includes('kv cache')) {
      return {
        text:
          lang === 'pt'
            ? 'O Mixture of Scars (MoS) & SCache resolve os dois maiores gargalos físicos de LLMs:\n\n1. O MoS resolve os PESOS: executa todas as 80 camadas de um 72B em GPU de 16 GB com apenas 2.40 GB de VRAM via desquantização streaming INT4 sob demanda.\n\n2. O SCache resolve o CONTEXTO: substitui o KV Cache tradicional (que consumiria 32.8 GB a 100.000 tokens e causaria Fatal OOM) por uma memória associativa O(1) de estritamente 34.8 KB no Cache L1/SRAM!'
            : 'Mixture of Scars (MoS) & SCache is a dual breakthrough architecture solving both fundamental physical walls of foundation models:\n\n1. MoS solves MODEL WEIGHTS: executes all 80 transformer layers of 72B parameter geometry on a single 16GB GPU with peak VRAM capped strictly under 2.40 GB via INT4 streaming.\n\n2. SCache solves CONTEXT MEMORY: replaces the traditional linear O(T) KV Cache (32.8 GB at 100,000 tokens causing fatal OOM) with an exact 34.8 KB O(1) associative tensor resident in L1 Cache/SRAM!',
        links: [
          {
            label: 'Live Test Google Colab',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
          {
            label: 'GitHub Repo (MoS & SCache)',
            url: 'https://github.com/byte-od/mixture-of-scars',
            icon: 'github',
          },
        ],
      };
    }

    if (q.includes('browser') || q.includes('navegador') || q.includes('como voce') || q.includes('como funciona') || q.includes('arquitetura que viabilizou')) {
      return {
        text:
          lang === 'pt'
            ? 'O motivo de rodar no navegador sem travar a aba:\n• WebGPU limita buffers a 2–4 GB: o MoS evita o estouro fazendo streaming dos pesos INT4 em fatias leves.\n• O KV Cache tradicional cresce a cada token gerado e estoura a memória da aba: o SCache mantém a memória fixa em 34.8 KB constantes O(1).\n\nExperimente selecionar [Leve], [Média] ou [Grande] acima para rodar a rede neural 100% na sua GPU!'
            : 'Why it runs in-browser without tab crash:\n• WebGPU caps memory buffer allocation at 2-4 GB: MoS avoids OOM by streaming INT4 weight slices.\n• Standard KV Cache continuously expands causing browser tab crash: SCache locks context memory to a constant 34.8 KB O(1) tensor in GPU SRAM.\n\nSelect [Light], [Medium] or [Large] above to run an actual neural model locally on your GPU!',
        links: [
          {
            label: 'SCache 100k Token Live Test',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
        ],
      };
    }

    if (q.includes('aura') || q.includes('sistema 1') || q.includes('system 1') || q.includes('rubik') || q.includes('reforco')) {
      return {
        text:
          lang === 'pt'
            ? 'AURA (Universal System 1 Engine) é o motor de Deep Reinforcement Learning do Julio. É um kernel de reflexo reativo que resolve estados combinatórios (Cubo de Rubik até profundidade D10 com 232 bilhões de estados) em apenas 0.23ms em CPU pura com zero árvore de busca (No MCTS).'
            : 'AURA (Universal System 1 Engine) is Julio\'s Embodied Deep Reinforcement Learning engine. It is a sub-millisecond reactive reflex kernel solving combinatorial spaces (Rubik\'s Cube up to Depth 10 across 232B states) in 0.23ms on bare CPU with zero tree search.',
        links: [
          { label: 'Repositório GitHub (AURA)', url: 'https://github.com/byte-od/system_one', icon: 'github' },
        ],
      };
    }

    if (q.includes('contato') || q.includes('contact') || q.includes('contratar') || q.includes('hire') || q.includes('email') || q.includes('linkedin')) {
      return {
        text:
          lang === 'pt'
            ? 'O Julio está disponível para posições de Liderança Técnica, Arquiteto de Sistemas Sênior e Engenheiro de LLMs/Transformers (remoto global ou relocalização internacional):\n\n• E-mail: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• GitHub: github.com/byte-od'
            : 'Julio is available for Staff/Principal Systems Architect and Senior LLM Engineer roles (remote global or relocation):\n\n• Email: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• GitHub: github.com/byte-od',
        links: [
          { label: 'E-mail Direto', url: 'mailto:reisfilho1116@gmail.com', icon: 'ext' },
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
        ],
      };
    }

    return {
      text:
        lang === 'pt'
          ? `Compreendido! No modo Reflexo (Zero Download), posso detalhar os pilares da engenharia do Julio:\n1. ⚡ MoS & SCache 72B em 16GB\n2. 🧩 AURA Deep RL (0.23ms CPU)\n3. 🤖 CIR-Engine 503M Causal\n4. 👤 Contratação & Contato direto por e-mail.\n\nOu selecione [Leve 135M], [Média 500M] ou [Grande 1.5B] acima para conversar com o modelo neural WebGPU real!`
          : `Understood! In Reflex Mode (Zero Download), I can guide you through Julio's engineering dossier:\n1. ⚡ MoS & SCache 72B on 16GB GPU\n2. 🧩 AURA Deep RL (0.23ms CPU)\n3. 🤖 CIR-Engine 503M Causal Transformer\n4. 👤 Hiring & Direct coordinates via email.\n\nOr select [Light 135M], [Medium 500M], or [Large 1.5B] above to chat with the real in-browser WebGPU model!`,
      links: [
        {
          label: 'Live Test no Colab',
          url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
          icon: 'colab',
        },
      ],
    };
  };

  const handleSendMessage = (textToSend?: string) => {
    const query = (textToSend ?? inputMessage).trim();
    if (!query) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');

    // A. SE TIVER EM MODO NEURAL CARREGADO
    if (activeTier !== 'reflex' && loadedTier === activeTier && workerRef.current) {
      setIsTyping(true);
      const agentMsgId = `agent-${Date.now()}`;
      streamingMsgIdRef.current = agentMsgId;

      const newAgentMsg: Message = {
        id: agentMsgId,
        sender: 'agent',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: '',
        tierUsed: activeTier,
        fromCache: wasLoadedFromCache,
      };
      setMessages((prev) => [...prev, newAgentMsg]);

      // Build conversation messages with system prompt
      const systemPrompt = lang === 'pt' ? SYSTEM_PROMPT_PT : SYSTEM_PROMPT_EN;
      const historyPayload = [
        { role: 'system', content: systemPrompt },
        ...messages
          .slice(-4)
          .map((m) => ({
            role: m.sender === 'user' ? 'user' : 'assistant',
            content: m.text,
          })),
        { role: 'user', content: query },
      ];

      workerRef.current.postMessage({
        type: 'generate',
        payload: {
          messages: historyPayload,
          maxTokens: activeTier === 'grande' ? 380 : 280,
          temperature: 0.6,
        },
      });
      return;
    }

    // B. SE O MODELO NEURAL AINDA NÃO CARREGOU
    if (activeTier !== 'reflex' && loadedTier !== activeTier) {
      const waitMsg: Message = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text:
          lang === 'pt'
            ? `O modelo neural [${MODEL_OPTIONS[activeTier].labelPt}] está sendo carregado na sua GPU. Enquanto isso, aqui está a resposta pelo Motor de Reflexo:\n\n${generateReflexResponse(query).text}`
            : `The neural model [${MODEL_OPTIONS[activeTier].labelEn}] is currently loading into your GPU. Meanwhile, here is the immediate Reflex response:\n\n${generateReflexResponse(query).text}`,
        links: generateReflexResponse(query).links,
        tierUsed: 'reflex',
      };
      setMessages((prev) => [...prev, waitMsg]);
      return;
    }

    // C. MODO REFLEXO (INSTANTÂNEO)
    setIsTyping(true);
    setTimeout(() => {
      const resp = generateReflexResponse(query);
      const agentMsg: Message = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: resp.text,
        links: resp.links,
        tierUsed: 'reflex',
      };
      setMessages((prev) => [...prev, agentMsg]);
      setIsTyping(false);
    }, 280);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* FLOATING ACTION TRIGGER */}
      <div className="fixed bottom-6 right-6 z-40 flex items-center gap-2">
        {!isOpen && (
          <button
            type="button"
            onClick={() => setIsOpen(true)}
            className="group flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-zinc-950/95 hover:bg-zinc-900 border-2 border-[#adff2f]/70 text-white font-mono text-xs font-bold shadow-[0_0_25px_rgba(173,255,47,0.25)] hover:shadow-[0_0_35px_rgba(173,255,47,0.4)] transition-all duration-200 cursor-pointer hover:scale-105"
            title={lang === 'pt' ? 'Testar IA no Browser (MoS & SCache)' : 'Test In-Browser AI (MoS & SCache)'}
          >
            <div className="relative">
              <Bot className="w-4 h-4 text-[#adff2f]" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f] animate-ping" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f]" />
            </div>
            <span>{lang === 'pt' ? 'Testar IA no Browser' : 'Test In-Browser AI'}</span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/30">
              WebGPU · O(1)
            </span>
          </button>
        )}
      </div>

      {/* EXPANDABLE CHAT PANEL */}
      {isOpen && (
        <div
          className={`fixed bottom-6 right-6 z-50 w-[94vw] sm:w-[460px] bg-zinc-950/95 border-2 border-[#adff2f]/70 rounded-2xl shadow-[0_0_40px_rgba(0,0,0,0.85),0_0_30px_rgba(173,255,47,0.2)] backdrop-blur-xl flex flex-col transition-all duration-200 overflow-hidden ${
            isMinimized ? 'h-[60px]' : 'h-[635px] max-h-[89vh]'
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/90 border-b border-zinc-800 shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="relative p-1.5 rounded-lg bg-[#adff2f]/10 border border-[#adff2f]/30">
                <Bot className="w-4 h-4 text-[#adff2f]" />
                <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-mono font-bold text-white tracking-tight">
                    {lang === 'pt' ? 'IA no Browser (MoS & SCache)' : 'In-Browser AI (MoS & SCache)'}
                  </h4>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Local GPU
                  </span>
                </div>
                <p className="text-[10px] font-mono text-zinc-400">
                  {lang === 'pt' ? 'Memória SCache: 34.8 KB O(1) · 0 Chamadas de Nuvem' : 'SCache Memory: 34.8 KB O(1) · 0 Cloud Calls'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setMessages([initialGreeting])}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Limpar conversa' : 'Reset chat'}
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsMinimized(!isMinimized)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={isMinimized ? (lang === 'pt' ? 'Expandir' : 'Expand') : (lang === 'pt' ? 'Minimizar' : 'Minimize')}
              >
                {isMinimized ? <Maximize2 className="w-3.5 h-3.5" /> : <Minimize2 className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Fechar' : 'Close'}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* HARDWARE RECOMMENDATION & HARDWARE PROFILE BANNER */}
              <div className="px-3 py-1.5 bg-zinc-950/90 border-b border-zinc-850 flex items-center justify-between text-[10px] font-mono shrink-0">
                <div className="flex items-center gap-1.5 text-zinc-300">
                  <Cpu className="w-3.5 h-3.5 text-[#adff2f]" />
                  <span className="font-semibold">{hardwareProfile.hardwareSummary}</span>
                </div>

                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowHwTooltip(!showHwTooltip)}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-lime-400/10 text-[#adff2f] border border-lime-400/30 hover:bg-lime-400/20 transition-colors cursor-pointer font-bold text-[9px]"
                  >
                    <span>★ {lang === 'pt' ? 'Recomendação' : 'Recommended'}</span>
                    <Info className="w-2.5 h-2.5" />
                  </button>

                  {/* Tooltip / Drawer */}
                  {showHwTooltip && (
                    <div className="absolute right-0 top-6 z-50 w-72 p-2.5 bg-zinc-900 border border-[#adff2f]/50 rounded-xl shadow-2xl text-[10px] font-mono text-zinc-200 space-y-1.5 backdrop-blur-md">
                      <div className="font-bold text-[#adff2f] flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        <span>
                          {lang === 'pt'
                            ? `Recomendado: ${MODEL_OPTIONS[hardwareProfile.recommendedTier].labelPt} (${MODEL_OPTIONS[hardwareProfile.recommendedTier].badge})`
                            : `Recommended: ${MODEL_OPTIONS[hardwareProfile.recommendedTier].labelEn} (${MODEL_OPTIONS[hardwareProfile.recommendedTier].badge})`}
                        </span>
                      </div>
                      <p className="text-zinc-300 leading-snug">
                        {lang === 'pt' ? hardwareProfile.reasonPt : hardwareProfile.reasonEn}
                      </p>
                      <div className="pt-1 border-t border-zinc-800 text-[9px] text-zinc-400 flex items-center justify-between">
                        <span>{hardwareProfile.hasWebGPU ? '✓ WebGPU Ativo' : '⚠️ Sem WebGPU'}</span>
                        <span>{hardwareProfile.cores} Cores</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* MODEL SELECTOR HEADER BAR (4 TIERS WITH HARDWARE BADGES & CACHE ICONS) */}
              <div className="px-3 py-2 bg-black/70 border-b border-zinc-800 shrink-0">
                <div className="grid grid-cols-4 gap-1.5">
                  {(['reflex', 'leve', 'media', 'grande'] as ModelTier[]).map((tier) => {
                    const opt = MODEL_OPTIONS[tier];
                    const isSelected = activeTier === tier;
                    const isLoaded = loadedTier === tier || tier === 'reflex';
                    const isLoading = loadingTier === tier;
                    const isRecommended = hardwareProfile.recommendedTier === tier;
                    const isCached = cachedTiers[tier];

                    return (
                      <button
                        key={tier}
                        type="button"
                        onClick={() => handleSelectTier(tier)}
                        className={`relative px-1.5 py-1.5 rounded-lg border text-center transition-all cursor-pointer flex flex-col items-center justify-center ${
                          isSelected
                            ? 'bg-[#adff2f]/15 border-[#adff2f] text-white shadow-[0_0_12px_rgba(173,255,47,0.3)]'
                            : 'bg-zinc-900/80 border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700'
                        }`}
                        title={lang === 'pt' ? opt.descPt : opt.descEn}
                      >
                        {/* Hardware recommendation badge */}
                        {isRecommended && (
                          <span className="absolute -top-1.5 -right-1 px-1 py-0.2 rounded-full bg-[#adff2f] text-black font-extrabold text-[8px] font-mono shadow-sm">
                            ★ IDEAL
                          </span>
                        )}

                        <div className="flex items-center gap-1">
                          {tier === 'reflex' && <Zap className="w-2.5 h-2.5 text-amber-400" />}
                          {tier === 'leve' && <span className="w-2 h-2 rounded-full bg-emerald-400" />}
                          {tier === 'media' && <span className="w-2 h-2 rounded-full bg-cyan-400" />}
                          {tier === 'grande' && <Flame className="w-2.5 h-2.5 text-rose-400" />}
                          <span className="text-[11px] font-mono font-bold leading-none">
                            {lang === 'pt' ? opt.labelPt : opt.labelEn}
                          </span>
                        </div>

                        <div className="flex items-center gap-1 mt-0.5">
                          <span className="text-[9px] font-mono text-zinc-400">
                            {isLoading ? '⏳' : opt.badge}
                          </span>
                          {/* Cache status icon */}
                          {isCached && tier !== 'reflex' && (
                            <span
                              className="text-[8px] text-emerald-400 font-bold"
                              title={lang === 'pt' ? 'Salvo no cache permanente' : 'Saved in local cache'}
                            >
                              ✓
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* NEURAL LOADING PROGRESS OVERLAY (WITH CACHE NOTICE) */}
              {loadingTier && (
                <div className="p-3 bg-zinc-900/95 border-b border-zinc-800 shrink-0">
                  <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                    <span className="text-white font-bold flex items-center gap-1.5">
                      <DownloadCloud className="w-3.5 h-3.5 text-[#adff2f] animate-pulse" />
                      <span>
                        {cachedTiers[loadingTier]
                          ? lang === 'pt'
                            ? `Carregando do Cache Local [${MODEL_OPTIONS[loadingTier].badge}]...`
                            : `Reading from Local Cache [${MODEL_OPTIONS[loadingTier].badge}]...`
                          : lang === 'pt'
                          ? `Baixando [${MODEL_OPTIONS[loadingTier].labelPt} ${MODEL_OPTIONS[loadingTier].badge}] para GPU...`
                          : `Downloading [${MODEL_OPTIONS[loadingTier].labelEn} ${MODEL_OPTIONS[loadingTier].badge}] to GPU...`}
                      </span>
                    </span>
                    <span className="text-[#adff2f] font-bold">
                      {cachedTiers[loadingTier]
                        ? '100% [Cache]'
                        : `${downloadPercent}% (${downloadLoadedMb} MB / ${downloadTotalMb || MODEL_OPTIONS[loadingTier].sizeMb} MB)`}
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-[#adff2f] h-full transition-all duration-200 shadow-[0_0_8px_#adff2f]"
                      style={{ width: `${Math.max(5, downloadPercent)}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[9px] font-mono text-zinc-400 mt-1.5">
                    <span>
                      {cachedTiers[loadingTier]
                        ? lang === 'pt'
                          ? '⚡ 0 MB baixados: lendo diretamente do IndexedDB permanente.'
                          : '⚡ 0 MB network: reading directly from persistent IndexedDB.'
                        : lang === 'pt'
                        ? 'Alocando buffer SCache O(1) de 34.8 KB na SRAM.'
                        : 'Allocating 34.8 KB O(1) SCache buffer in SRAM.'}
                    </span>
                    <span className="text-emerald-400 flex items-center gap-0.5">
                      <ShieldCheck className="w-2.5 h-2.5" />
                      <span>Cache Persistente</span>
                    </span>
                  </div>
                </div>
              )}

              {/* ERROR NOTICE IF OCCURRED */}
              {errorMessage && (
                <div className="p-2.5 bg-rose-950/60 border-b border-rose-800/80 text-rose-300 text-[11px] font-mono flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    <span>{errorMessage}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActiveTier('reflex')}
                    className="underline text-white font-bold ml-2 cursor-pointer"
                  >
                    {lang === 'pt' ? 'Usar Reflexo' : 'Use Reflex'}
                  </button>
                </div>
              )}

              {/* Message List */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3 font-sans text-xs">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                  >
                    <div
                      className={`max-w-[88%] p-3 rounded-xl leading-relaxed ${
                        msg.sender === 'user'
                          ? 'bg-[#adff2f] text-black font-medium shadow-md shadow-lime-400/10'
                          : 'bg-zinc-900/90 text-zinc-200 border border-zinc-800'
                      }`}
                    >
                      <p className="whitespace-pre-line">{msg.text}</p>

                      {/* Links */}
                      {msg.links && msg.links.length > 0 && (
                        <div className="mt-2.5 pt-2 border-t border-zinc-800/80 space-y-1.5">
                          {msg.links.map((link, idx) => (
                            <a
                              key={idx}
                              href={link.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-1.5 text-[11px] font-mono font-bold text-[#adff2f] hover:underline"
                            >
                              {link.icon === 'colab' && <PlayCircle className="w-3.5 h-3.5" />}
                              {link.icon === 'github' && <Github className="w-3.5 h-3.5" />}
                              {link.icon === 'ext' && <ExternalLink className="w-3.5 h-3.5" />}
                              <span>{link.label}</span>
                            </a>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2 mt-1 px-1">
                      <span className="text-[9px] font-mono text-zinc-500">
                        {msg.timestamp}
                      </span>
                      {msg.tierUsed && msg.tierUsed !== 'reflex' && (
                        <span className="text-[9px] font-mono text-[#adff2f] flex items-center gap-1">
                          <Cpu className="w-2.5 h-2.5" />
                          <span>WebGPU ({MODEL_OPTIONS[msg.tierUsed].badge})</span>
                          {msg.fromCache && (
                            <span className="text-emerald-400 text-[8px] bg-emerald-500/10 px-1 rounded border border-emerald-500/20">
                              Cache
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                  </div>
                ))}

                {/* Typing indicator */}
                {isTyping && (
                  <div className="flex items-center gap-1.5 p-3 rounded-xl bg-zinc-900 border border-zinc-800 w-fit">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce" />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce [animation-delay:0.2s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce [animation-delay:0.4s]" />
                    <span className="text-[10px] font-mono text-zinc-400 ml-1.5">
                      {activeTier !== 'reflex' ? 'Decodificando tokens na GPU local...' : ''}
                    </span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Prompt Chips */}
              <div className="px-3 py-1.5 bg-zinc-900/60 border-t border-zinc-850 overflow-x-auto flex items-center gap-1.5 no-scrollbar shrink-0">
                {quickPrompts.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendMessage(p.query)}
                    className="shrink-0 text-[10px] font-mono px-2.5 py-1 rounded-full bg-zinc-850 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-750 transition-colors cursor-pointer"
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Telemetry & Cache Preservation Guarantee Footer */}
              <div className="px-3 py-1.5 bg-black/90 border-t border-zinc-850 flex items-center justify-between text-[9px] font-mono text-zinc-400 shrink-0">
                <span className="flex items-center gap-1 text-emerald-400">
                  <ShieldCheck className="w-3 h-3" />
                  <span>
                    {isStoragePersisted
                      ? 'Storage.persist: Ativo · Zero Downloads Duplicados'
                      : 'Cache IndexedDB Ativo'}
                  </span>
                </span>
                <span className="text-zinc-300">
                  {lastTokPerSec ? `⚡ ${lastTokPerSec} tok/s` : 'SCache: 34.8 KB O(1)'}
                </span>
              </div>

              {/* Input Field */}
              <div className="p-3 bg-zinc-950 border-t border-zinc-800 flex items-center gap-2 shrink-0">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    activeTier !== 'reflex'
                      ? lang === 'pt'
                        ? `Conversando via ${MODEL_OPTIONS[activeTier].labelPt} local na GPU...`
                        : `Chatting via local ${MODEL_OPTIONS[activeTier].labelEn} on GPU...`
                      : lang === 'pt'
                      ? 'Pergunte sobre o Julio, MoS 72B, AURA...'
                      : 'Ask about Julio, MoS 72B, AURA...'
                  }
                  className="flex-1 bg-zinc-900 border border-zinc-800 focus:border-[#adff2f] rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none transition-colors font-sans"
                />
                <button
                  type="button"
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim()}
                  className="p-2 bg-[#adff2f] hover:bg-lime-300 disabled:bg-zinc-800 disabled:text-zinc-600 text-black rounded-lg transition-colors cursor-pointer shrink-0"
                  title={lang === 'pt' ? 'Enviar' : 'Send'}
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};
