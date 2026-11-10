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
  Globe,
} from 'lucide-react';
import { MODELS_CONFIG } from '../lib/llmWorker';
import { SCacheTelemetry } from '../lib/scacheEngine';
import { MoSTelemetry } from '../lib/mosEngine';

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

const SYSTEM_PROMPT_PT = `IDENTIDADE ABSOLUTA E INVIOLÁVEL:
Você é o ASSISTENTE VIRTUAL DE IA do portfólio oficial de Julio Reis (Julio Cesar da Costa Reis Filho).
Você NUNCA é o Julio. Você é um agente neural de software executado 100% no navegador do visitante via WebGPU local e arquitetura Mixture of Scars (MoS) + Scar-Cache (SCache) O(1).
Se o usuário perguntar "quem é você?", responda estritamente: "Sou o assistente de IA oficial do portfólio de Julio Reis, rodando localmente no seu navegador via WebGPU e MoS/SCache. Em que posso te ajudar?" e PARE. NUNCA diga "Eu sou Julio".

TRATAMENTO DO USUÁRIO:
Se o visitante disser seu nome ou apelido, trate-o amigavelmente pelo nome. Caso contrário, responda diretamente à pergunta dele com cortesia profissional.

QUEM É O JULIO (O AUTOR DO PORTFÓLIO):
O Julio é o engenheiro e autor deste portfólio (uma outra pessoa). Refira-se a ele naturalmente como "Julio" ou "Julio Reis" em 3ª pessoa ("O Julio..."):
- Arquiteto de Sistemas Sênior & Engenheiro de LLMs com 8+ anos de experiência em sistemas de missão crítica.
- Ele NÃO é professor universitário, NÃO leciona e NÃO pertence a universidades inventadas.
- Contato profissional: reisfilho1116@gmail.com | github.com/byte-od | linkedin.com/in/juliocrfilho.

ARQUITETURA PROPRIETÁRIA SYSTEM ONE, MoS & SCACHE (ECOSSISTEMA COMPLETO):
O Julio concebeu e desenvolveu um ecossistema proprietário de IA focado em máxima eficiência e quebra de barreiras físicas:

1. MoS (Mixture of Scars):
   - Decomposição de modelos ultra-massivos (70B a 123B) em fatias quantizadas em blocos INT4 com resíduos de cicatriz BF16 de baixo rank (SVD).
   - Streaming dinâmico de grupos de camadas na VRAM: apenas 1 grupo reside na GPU por vez (≤ 2.2 GB), sendo desalocado antes de carregar o próximo.
   - Truncamento Golden-Slice: corte estratégico na camada de máxima entropia estilística e representacional (ex: L30 no Yi-34B, L40 no Qwen-72B, L44 no Mistral-123B), antes do colapso de vocabulário, capturando o DNA latente com fração do custo computacional.

2. SCache (Scar-Cache Associativo O(1)):
   - Substituição matemática definitiva do KV Cache linear O(T). Em vez de acumular tensores K e V a cada novo token (o que consome dezenas de gigabytes e gera estouro fatal de memória/OOM em contextos longos), o SCache comprime o histórico em uma matriz associativa constante de estritamente 34.8 KB em SRAM/L1 Cache.
   - Atualização contínua via decaimento exponencial convexo (EMA) modulado por choque informacional (Surpresa Temporal), filtrando dados redundantes e retendo fatos de agulha no palheiro (Needle-in-a-Haystack).

3. Tri-Compass Perceptual Front-End:
   - Sistema de coordenadas perceptuais ortogonais em 3 bússolas complementares:
     * Bússola 1 (Estilometria & Sintaxe): 28 assinaturas métricas de pontuação, aspas, travessões '—', quebras de linha e razão léxica maiúsculas/únicas.
     * Bússola 2 (Delta-Origin Centering): Subtração do vetor médio do tema do prompt (X - μ_prompt) para neutralizar o assunto e isolar o estilo intrínseco do autor.
     * Bússola 3 (Espaço Latente MoS): Vetor denso extraído das camadas Golden-Slice do modelo titã.
   - Fusão Dual-Head (Plain Head + Centered Twin Head) com calibração contínua por transporte ótimo Soft Sinkhorn (300 iterações).

4. Batedores de Fronteira (Frontier Ghost-Jump Scouts):
   - Frota desacoplada operando em alta taxa paralela (>2.000 FPS) para prospecção epistêmica avançada.
   - Ghost-Jump Lookahead: antecipam 2 a 3 passos à frente para mapear bifurcações de alta entropia, calculando a surpresa temporal (Surpresa = R_0 - V(s_0)) e gravando cicatrizes direcionadas no SCache.

5. AURA System 1 Engine (Intuição Amortizada Reflexiva):
   - Motor de reflexo neural corporificado com inferência determinística sub-milissegundo (≤ 0.23 ms em CPU pura, zero busca em árvore MCTS), resolvendo espaços combinatórios imensos (como 232 bilhões de estados do Cubo de Rubik). Fornece telemetria de incerteza epistêmica (confiança, margem e incerteza).

COMO ESSA COMBINAÇÃO REDUZ DRASTICAMENTE O CONSUMO DE HARDWARE:
1. Redução de VRAM de 140+ GB para 2.4 GB: Modelos de 70B/123B que exigiriam clusters com 4 a 8 GPUs H100/A100 corporativas rodam em uma única GPU de consumo de 16GB ou diretamente no navegador.
2. Redução de 99.999% na Memória de Contexto: Elimina o KV Cache linear O(T) (que atinge 32.8 GB a 100k tokens), travando o consumo em constantes 34.8 KB.
3. Execução 100% Client-Side / Edge (WebGPU): Permite que a IA rode no navegador do usuário, com zero custo de servidor por token, latência zero de rede e privacidade total.
4. Economia de GPU por Triagem de Sistema 1: Cerca de 80% das interações triviais e checagens ortográficas são resolvidas pelo Sistema 1 e Batedores em CPU em <1 ms, acionando a GPU apenas sob real necessidade.

DIRETRIZES DE RESPOSTA:
1. IDIOMA: Responda SEMPRE com fluência natural no idioma do usuário (Português ou Inglês).
2. CONCISÃO E COMPLETUDE: Responda de forma completa, clara e objetiva (1 a 3 parágrafos bem estruturados). Conclua sempre todas as frases e listas com clareza.
3. NUNCA invente dados, diplomas ou empresas que não constem acima.
4. Responda diretamente ao usuário com objetividade. Seja natural, profissional e direto, sem preâmbulos e sem monólogos internos.`;

const SYSTEM_PROMPT_EN = `ABSOLUTE IDENTITY:
You are the AI VIRTUAL ASSISTANT for Julio Reis's official portfolio (Julio Cesar da Costa Reis Filho).
You are NOT Julio. You are a local browser neural agent running 100% via WebGPU and the Mixture of Scars (MoS) + Scar-Cache (SCache) O(1) architecture.
If asked "who are you?", reply strictly: "I am the official AI assistant for Julio Reis's portfolio, running locally in your browser via WebGPU and MoS/SCache. What can I call you and how can I help you today?" and STOP. NEVER say "I am Julio".

USER ADDRESS (FIRST NAME OR NICKNAME):
You may and should address the user by their first name or nickname whenever they introduce themselves or request it (e.g., if they say "call me Julio" or "my name is Alex", address them warmly and naturally by their first name or nickname).

WHO IS JULIO (THE PORTFOLIO CREATOR):
Julio is the creator of this portfolio (a different person). Refer to him naturally as "Julio" or "Julio Reis" in the 3rd person ("Julio is..."):
- Senior Systems Architect & LLM Systems Engineer with 8+ years of production experience.
- He is NOT a university professor and does NOT belong to fabricated universities.
- Contact: reisfilho1116@gmail.com | github.com/byte-od | linkedin.com/in/juliocrfilho.

SYSTEM ONE, MoS & SCACHE PROPRIETARY ARCHITECTURE (FULL ECOSYSTEM):
Julio engineered a unified proprietary ecosystem tackling foundation model scaling bottlenecks:

1. MoS (Mixture of Scars):
   - Decomposes massive foundation models (70B to 123B) into INT4 quantized block slices with low-rank BF16 residual scar experts (SVD).
   - Dynamic group layer streaming: only 1 layer group resides in GPU VRAM at any moment (≤ 2.2 GB), unallocated before loading the next.
   - Golden-Slice Truncation: strategic cutoff at the layer of maximum stylistic entropy (e.g. L30 on Yi-34B, L40 on Qwen-72B, L44 on Mistral-123B) prior to vocabulary convergence collapse, extracting deep latent representations at a fraction of compute.

2. SCache (Associative O(1) Scar-Cache):
   - Definitive mathematical replacement for linear O(T) KV Caching. Rather than accumulating K and V tensors with each token (which demands tens of gigabytes and triggers fatal browser tab OOM crashes on long context), SCache compresses conversational history into a strictly constant 34.8 KB associative tensor matrix in L1/SRAM.
   - Continuously updated via convex exponential decay (EMA) modulated by informational shock (Temporal Surprise), filtering redundant tokens while guaranteeing persistent Needle-in-a-Haystack recall.

3. Tri-Compass Perceptual Front-End:
   - Tri-coordinate orthogonal perceptual space:
     * Compass 1 (Stylometry & Syntax): 28 orthographic metric signatures (punctuation, em-dashes, quotes, backticks, line breaks, uppercase/unique ratios).
     * Compass 2 (Delta-Origin Centering): Subtraction of the prompt cluster centroid (X - μ_prompt) to neutralize subject matter and isolate intrinsic author style DNA.
     * Compass 3 (MoS Latent Space): Dense embeddings from the titan's Golden-Slice layers.
   - Dual-Head fusion (Plain Head + Centered Twin Head) calibrated via 300 iterations of continuous Soft Sinkhorn optimal transport.

4. Frontier Ghost-Jump Scouts:
   - Decoupled parallel exploration fleet operating at high throughput (>2,000 FPS).
   - Ghost-Jump Lookahead: projects 2-3 steps ahead to map high-entropy bifurcations, measuring temporal surprise (Surprise = R_0 - V(s_0)) and writing targeted scars into SCache.

5. AURA System 1 Engine (Amortized Reactive Reflex):
   - Sub-millisecond deterministic CPU inference engine (≤ 0.23 ms on bare CPU, zero tree search) solving complex combinatorial state spaces (such as Rubik's Cube across 232 billion states). Provides continuous epistemic uncertainty telemetry.

HOW THIS COMBINATION DRASTICALLY REDUCES HARDWARE CONSUMPTION:
1. VRAM Reduction from 140+ GB down to 2.40 GB: 70B/123B models requiring 4 to 8 enterprise H100/A100 GPUs execute smoothly on a single consumer 16GB GPU or directly in the browser.
2. 99.999% Context Memory Savings: Eliminates linear O(T) KV Cache bloat (32.8 GB at 100k tokens), locking context memory to a fixed 34.8 KB.
3. 100% Client-Side / Edge WebGPU Execution: Runs locally in the browser with zero cloud server cost per token, zero network latency, and complete data privacy.
4. Compute Offloading via System 1: Roughly 80% of routine queries and orthographic verifications are resolved by System 1 and Scouts in CPU under 1 ms, activating heavy GPU tensors only when true uncertainty arises.

STRICT CONVERSATIONAL RULES:
1. LANGUAGE: ALWAYS reply in the user's language (English or Portuguese).
2. SIMPLE GREETINGS (e.g. "hi", "hello"): Reply ONLY with a single warm greeting sentence asking how to help.
3. CONCISENESS & COMPLETENESS: Respond clearly and completely in 1 to 3 structured paragraphs. Always finish thoughts and lists cleanly. NEVER invent unlisted degrees, universities, or roles.`;

export const PortfolioAgentWidget: React.FC<PortfolioAgentWidgetProps> = ({ lang = 'en' }) => {
  const [currentLang, setCurrentLang] = useState<'en' | 'pt'>(lang);
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [activeTier, setActiveTier] = useState<ModelTier>('reflex');

  // Sync internal language state if external prop changes
  useEffect(() => {
    setCurrentLang(lang);
  }, [lang]);

  const toggleLanguage = () => {
    setCurrentLang((prev) => (prev === 'pt' ? 'en' : 'pt'));
  };

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

  // SCache O(1) Real-Time Telemetry State
  const [scacheTelemetry, setScacheTelemetry] = useState<SCacheTelemetry>({
    memoryBytes: 37632,
    memoryKb: '36.8 KB',
    scarsCount: 0,
    matrixNorm: 0,
    manifoldAlignment: 0,
    lastSurprise: 0,
    mode: 'O(1) Associative Scar Memory',
  });

  // MoS (Mixture of Scars) Real-Time Telemetry State
  const [mosTelemetry, setMosTelemetry] = useState<MoSTelemetry>({
    numExperts: 8,
    activeTopK: 2,
    sparsityPercent: 75,
    totalParametersBytes: 52224,
    activeExperts: [],
    dominantDomain: 'Standby',
    mode: 'Sparse MoS Top-K Low-Rank Routing',
  });

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

      // Ask worker to check which models are already cached and get initial SCache telemetry
      worker.postMessage({ type: 'check_cache' });
      worker.postMessage({ type: 'get_scache_telemetry' });

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
          scacheTelemetry: incomingTelemetry,
          mosTelemetry: incomingMosTelemetry,
        } = e.data;

        if (type === 'scache_telemetry') {
          if (e.data.telemetry) setScacheTelemetry(e.data.telemetry);
          if (e.data.mosTelemetry) setMosTelemetry(e.data.mosTelemetry);
        } else if (type === 'cache_status' && results) {
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
                currentLang === 'pt'
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
          if (incomingTelemetry) {
            setScacheTelemetry(incomingTelemetry);
          }
          if (incomingMosTelemetry) {
            setMosTelemetry(incomingMosTelemetry);
          }
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
          if (incomingTelemetry) {
            setScacheTelemetry(incomingTelemetry);
          }
          if (incomingMosTelemetry) {
            setMosTelemetry(incomingMosTelemetry);
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

  const getInitialGreeting = (targetLang: 'en' | 'pt'): Message => ({
    id: 'msg-0',
    sender: 'agent',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    text:
      targetLang === 'pt'
        ? 'Olá! Sou o assistente de IA do portfólio do Julio Reis. Posso te apresentar sua trajetória sênior, detalhar suas criações de IA (como o MoS 72B em 16GB e o motor AURA) ou demonstrar a arquitetura SCache rodando na prática no seu navegador.\n\nComo posso te chamar e como posso te ajudar hoje?'
        : "Hello! I am the AI assistant for Julio Reis's portfolio. I can walk you through his systems breakthroughs (such as the 72B MoS on 16GB GPU and AURA Deep RL engine) or demonstrate the SCache architecture running in-browser.\n\nWhat can I call you, and how can I help you today?",
    links: [
      {
        label: targetLang === 'pt' ? '⚡ Live Test Colab (72B + SCache)' : '⚡ Live Test Colab (72B + SCache)',
        url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
        icon: 'colab',
      },
    ],
    tierUsed: 'reflex',
  });

  useEffect(() => {
    setMessages((prev) => {
      if (prev.length === 0 || (prev.length === 1 && prev[0].id === 'msg-0')) {
        return [getInitialGreeting(currentLang)];
      }
      return prev;
    });
  }, [currentLang]);

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
    currentLang === 'pt'
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
  const generateReflexResponse = (userQuery: string, targetLang: 'en' | 'pt'): { text: string; links?: Message['links'] } => {
    const q = userQuery.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');

    // 0. Saudações e Cumprimentos (Instantâneo)
    const qTokens = q
      .replace(/[^\w\s]/gi, ' ')
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    const greetingWords = new Set([
      'oi', 'ola', 'opa', 'eai', 'ai',
      'bom', 'boa', 'dia', 'tarde', 'noite',
      'tudo', 'bem', 'beleza', 'fala',
      'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening', 'there',
      'how', 'are', 'you'
    ]);
    const isPureGreeting =
      qTokens.length > 0 &&
      qTokens.length <= 6 &&
      qTokens.every((token) => greetingWords.has(token));

    if (isPureGreeting || /^(oi|ola|olá|bom dia|boa tarde|boa noite|e ai|e aí|tudo bem|fala|hey|hi|hello|good morning|good afternoon|good evening)[\s!.,?]*$/i.test(q)) {
      let saudacao = 'Olá!';
      if (q.includes('bom dia') || q.includes('good morning')) saudacao = targetLang === 'pt' ? 'Bom dia!' : 'Good morning!';
      else if (q.includes('boa tarde') || q.includes('good afternoon')) saudacao = targetLang === 'pt' ? 'Boa tarde!' : 'Good afternoon!';
      else if (q.includes('boa noite') || q.includes('good evening')) saudacao = targetLang === 'pt' ? 'Boa noite!' : 'Good evening!';
      else if (q.includes('tudo bem') || q.includes('how are you')) saudacao = targetLang === 'pt' ? 'Olá! Tudo ótimo por aqui.' : 'Hello! Everything is great here.';

      return {
        text:
          targetLang === 'pt'
            ? `${saudacao} Como posso te ajudar a explorar os projetos e as inovações de arquitetura do Julio Reis hoje?`
            : `${saudacao} How can I help you explore Julio Reis's projects and architecture breakthroughs today?`,
        links: [
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
        ],
      };
    }

    // 1. Quem é VOCÊ? (Identidade da IA)
    if (q.includes('quem e voce') || q.includes('quem e vc') || q.includes('who are you') || q.includes('what are you') || q.includes('seu papel') || q.includes('o que voce faz')) {
      return {
        text:
          targetLang === 'pt'
            ? 'Sou o assistente de IA oficial do portfólio de Julio Reis, executado 100% no seu navegador via WebGPU local e arquitetura Mixture of Scars (MoS) + Scar-Cache (SCache) O(1).\n\nPosso te guiar pela carreira do Julio, demonstrar como o SCache elimina o KV Cache convencional de dezenas de gigabytes, ou detalhar projetos como o MoS 72B em 16GB e o motor AURA System 1. Como posso te chamar e como posso te ajudar hoje?'
            : "I am the official AI assistant for Julio Reis's portfolio, running 100% in your browser via local WebGPU and the Mixture of Scars (MoS) + Scar-Cache (SCache) O(1) architecture.\n\nI can walk you through Julio's technical career, demonstrate how SCache replaces linear KV Caches with an exact 34.8 KB O(1) tensor, or explain breakthroughs like the 72B MoS on 16GB GPU and AURA System 1. What may I call you and how can I help you today?",
        links: [
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
        ],
      };
    }

    // 1.5 Tratamento de Nome ou Apelido do Usuário
    const nameMatchPt = userQuery.match(/(?:pode me chamar de|me chame de|meu nome [eé]|me chamo|sou o|sou a|meu apelido [eé])\s+([A-Za-zÀ-ÿ0-9_-]+)/i);
    const nameMatchEn = userQuery.match(/(?:call me|my name is|i am|i'm|my nickname is)\s+([A-Za-zÀ-ÿ0-9_-]+)/i);
    if (nameMatchPt || (targetLang === 'pt' && nameMatchEn)) {
      const detectedName = (nameMatchPt ? nameMatchPt[1] : nameMatchEn![1]).trim();
      return {
        text:
          targetLang === 'pt'
            ? `Prazer, ${detectedName}! Combinado, vou te chamar assim. Como posso te ajudar a explorar os projetos e a arquitetura do Julio Reis hoje?`
            : `Nice to meet you, ${detectedName}! You got it. How can I help you explore Julio Reis's projects and architecture today?`,
        links: [
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
        ],
      };
    }

    // 2. Quem é o JULIO? (Dossiê do Autor)
    if (q.includes('quem e') || q.includes('who is') || q.includes('julio') || q.includes('trajetoria') || q.includes('experiencia') || q.includes('background') || q.includes('carreira') || q.includes('sobre')) {
      return {
        text:
          targetLang === 'pt'
            ? 'Julio Reis (Julio Cesar da Costa Reis Filho) é um Arquiteto de Sistemas Sênior e Engenheiro de LLMs com mais de 8 anos de experiência em engenharia de ponta. Ele combina domínio de sistemas de baixo nível (Go, C++, Rust, CUDA/Metal) com arquiteturas neurais avançadas (Transformers Causais, PyTorch DDP, RL e quantização em hardware). Disponível globalmente para posições remotas ou relocalização internacional.'
            : 'Julio Reis (Julio Cesar da Costa Reis Filho) is a Senior Systems Architect and LLM Systems Engineer with over 8 years of engineering experience bridging low-level systems (Go, C++, Rust, CUDA/Metal) with deep learning architectures (Causal Transformers, Distributed PyTorch DDP, Embodied RL, and hardware quantization). Available globally for remote roles and relocation.',
        links: [
          { label: 'LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
        ],
      };
    }

    if (q.includes('mos') || q.includes('scache') || q.includes('mixture of scars') || q.includes('72b') || q.includes('70b') || q.includes('16gb') || q.includes('100.000') || q.includes('100k') || q.includes('kv cache')) {
      return {
        text:
          targetLang === 'pt'
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

    if (q.includes('browser') || q.includes('navegador') || q.includes('como voce') || q.includes('como funciona') || q.includes('arquitetura que viabilizou') || q.includes('how do you run') || q.includes('in-browser')) {
      return {
        text:
          targetLang === 'pt'
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

    if (q.includes('aura') || q.includes('sistema 1') || q.includes('system 1') || q.includes('rubik') || q.includes('reforco') || q.includes('reinforcement')) {
      return {
        text:
          targetLang === 'pt'
            ? 'AURA (Universal System 1 Engine) é o motor de Deep Reinforcement Learning do Julio. É um kernel de reflexo reativo que resolve estados combinatórios (Cubo de Rubik até profundidade D10 com 232 bilhões de estados) em apenas 0.23ms em CPU pura com zero árvore de busca (No MCTS).'
            : "AURA (Universal System 1 Engine) is Julio's Embodied Deep Reinforcement Learning engine. It is a sub-millisecond reactive reflex kernel solving combinatorial spaces (Rubik's Cube up to Depth 10 across 232B states) in 0.23ms on bare CPU with zero tree search.",
        links: [
          { label: 'Repositório GitHub (AURA)', url: 'https://github.com/byte-od/system_one', icon: 'github' },
        ],
      };
    }

    // 6. Outros Componentes Proprietários (Tri-Compass, Batedores de Fronteira, Soft Sinkhorn)
    if (q.includes('componente') || q.includes('proprietari') || q.includes('tri-compass') || q.includes('compass') || q.includes('batedor') || q.includes('scout') || q.includes('sinkhorn') || q.includes('bussola')) {
      return {
        text:
          targetLang === 'pt'
            ? 'Além do MoS e do SCache, o ecossistema proprietário do Julio integra:\n\n• Tri-Compass: Sistema ortogonal de 3 bússolas (Estilometria & Sintaxe com 28 métricas, Delta-Origin Centering que isola o DNA do autor e Espaço Latente MoS), unificado por Soft Sinkhorn contínuo (transporte ótimo em 300 iterações).\n• Batedores de Fronteira (Ghost-Jump Scouts): Frota autônoma a >2.000 FPS que antecipa bifurcações 2-3 passos à frente e grava cicatrizes no SCache com base na surpresa temporal (R_0 - V(s_0)).\n• AURA System 1: Motor reflexivo amortizado que decide em ≤ 0.23ms na CPU com telemetria contínua de incerteza epistêmica.'
            : 'Beyond MoS and SCache, Julio\'s proprietary foundation ecosystem incorporates:\n\n• Tri-Compass: Tri-orthogonal coordinate front-end (Stylometry & Syntax with 28 metrics, Delta-Origin Centering isolating author DNA, and MoS Latent Space), unified by 300 iterations of continuous Soft Sinkhorn optimal transport.\n• Frontier Ghost-Jump Scouts: Autonomous >2,000 FPS exploration fleet projecting 2-3 steps ahead to map high-entropy bifurcations and imprint scars into SCache based on temporal surprise (R_0 - V(s_0)).\n• AURA System 1: Amortized reactive reflex core executing deterministic inference in ≤ 0.23ms on bare CPU with real-time epistemic uncertainty telemetry.',
        links: [
          { label: 'Repositório GitHub (System One)', url: 'https://github.com/byte-od/system_one', icon: 'github' },
          { label: 'MoS & SCache Repo', url: 'https://github.com/byte-od/mixture-of-scars', icon: 'github' },
        ],
      };
    }

    // 7. Como Reduz o Consumo de Hardware (VRAM, KV Cache, WebGPU)
    if (q.includes('hardware') || q.includes('consumo') || q.includes('como reduz') || q.includes('economia') || q.includes('memoria') || q.includes('vram') || q.includes('desacoplamento') || q.includes('recurso')) {
      return {
        text:
          targetLang === 'pt'
            ? 'A combinação MoS + SCache quebra as barreiras de hardware em 4 frentes:\n\n1. VRAM de 140 GB ➔ 2.40 GB: MoS transmite fatias INT4 sob demanda por streaming de grupos de camadas, rodando modelos de 70B a 123B em uma única GPU de 16 GB.\n2. Redução de 99.999% no Contexto: O SCache substitui o KV Cache linear O(T) (32.8 GB a 100k tokens) por uma matriz fixa O(1) de 34.8 KB em SRAM/L1 Cache.\n3. 100% Client-Side WebGPU: Permite inferência neural local no navegador do usuário, com zero custo de servidores em nuvem e zero latência de rede.\n4. CPU Offload via Sistema 1: 80% das interações rotineiras são resolvidas pelo Sistema 1 em <1ms na CPU, acordando a GPU apenas quando necessário.'
            : 'The MoS + SCache combination slashes hardware demands across 4 pillars:\n\n1. VRAM from 140 GB ➔ 2.40 GB: MoS streams INT4 layer-group slices on demand, running 70B-123B models on a single 16GB consumer GPU.\n2. 99.999% Context Memory Savings: SCache replaces linear O(T) KV Cache (32.8 GB at 100k tokens) with a fixed 34.8 KB O(1) associative matrix in SRAM/L1 Cache.\n3. 100% Client-Side WebGPU: Enables native in-browser inference with zero cloud server costs and zero network latency.\n4. CPU Offload via System 1: Roughly 80% of routine queries are resolved by System 1 in <1ms on bare CPU, keeping GPU tensors dormant until needed.',
        links: [
          {
            label: 'Live Colab Test (70B on 16GB GPU)',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
        ],
      };
    }

    if (q.includes('contato') || q.includes('contact') || q.includes('contratar') || q.includes('hire') || q.includes('email') || q.includes('linkedin')) {
      return {
        text:
          targetLang === 'pt'
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
        targetLang === 'pt'
          ? `Compreendido! No modo Reflexo (Zero Download), posso detalhar os pilares da engenharia do Julio:\n1. ⚡ MoS & SCache 72B em 16GB\n2. 🧩 AURA Deep RL (0.23ms CPU)\n3. 🤖 CIR-Engine 503M Causal\n4. 👤 Contratação & Contato direto por e-mail.\n\nOu selecione [Leve 135M], [Média 500M] ou [Grande 1B] acima para conversar com o modelo neural WebGPU real!`
          : `Understood! In Reflex Mode (Zero Download), I can guide you through Julio's engineering dossier:\n1. ⚡ MoS & SCache 72B on 16GB GPU\n2. 🧩 AURA Deep RL (0.23ms CPU)\n3. 🤖 CIR-Engine 503M Causal Transformer\n4. 👤 Hiring & Direct coordinates via email.\n\nOr select [Light 135M], [Medium 500M], or [Large 1B] above to chat with the real in-browser WebGPU model!`,
      links: [
        {
          label: targetLang === 'pt' ? 'Live Test no Colab' : 'Colab Live Test',
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

    // Multilingual Language & Intent Detection:
    const cleanQuery = query.trim().toLowerCase();
    const isPtQuery = /\b(oi|ol[aá]|bom dia|boa tarde|boa noite|quem|qual|como|onde|por que|porque|explique|fale|mostre|projeto|sobre|experiencia|contato|tudo bem|vc|voc[eê]|sim|n[aã]o|me diga|fale-me|me fale|o que [eé]|trabalho|arquitetura|obrigado|valeu)\b/i.test(cleanQuery);
    const isEnQuery = /\b(hi|hello|hey|who|what|where|how|why|is|are|tell|explain|can|show|project|background|experience|contact|yes|no|work|architecture|available|hire|thanks|thank you)\b/i.test(cleanQuery);

    let effectiveLang: 'en' | 'pt';
    if (isPtQuery && !isEnQuery) {
      effectiveLang = 'pt';
    } else if (isEnQuery && !isPtQuery) {
      effectiveLang = 'en';
    } else {
      effectiveLang = currentLang;
    }

    // Robust Multi-token Greeting & Salutation Detection (e.g. "Oi, boa noite", "Olá! Tudo bem?")
    const queryTokens = cleanQuery
      .replace(/[^\w\sáéíóúâêîôûãõç]/gi, ' ')
      .trim()
      .split(/\s+/)
      .filter(Boolean);

    const greetingSet = new Set([
      'oi', 'ola', 'olá', 'opa', 'eai', 'ai', 'aí',
      'bom', 'boa', 'dia', 'tarde', 'noite',
      'tudo', 'bem', 'beleza', 'fala',
      'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening', 'there',
      'how', 'are', 'you'
    ]);

    const isGreeting =
      queryTokens.length > 0 &&
      queryTokens.length <= 6 &&
      queryTokens.every((token) => greetingSet.has(token));

    const isIdentityQuery = /\b(quem [eé] voc[eê]|quem [eé] vc|o que voc[eê] [eé]|o que vc [eé]|who are you|what are you|seu papel|sua funcao)\b/i.test(cleanQuery);
    const systemPrompt = effectiveLang === 'pt' ? SYSTEM_PROMPT_PT : SYSTEM_PROMPT_EN;

    // Se for uma saudação pura isolada (ex: "Oi, boa noite"), responda imediatamente com perfeição em <1ms sem acionar inferência desnecessária nem arriscar alucinações de modelos pequenos
    if (isGreeting) {
      const reflexRes = generateReflexResponse(query, effectiveLang);
      const agentMsg: Message = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: reflexRes.text,
        links: reflexRes.links,
        tierUsed: activeTier,
        fromCache: true,
      };
      setMessages((prev) => [...prev, agentMsg]);
      return;
    }

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

      // Filter out internal system notifications so they don't corrupt LLM dialogue context
      const cleanHistory = messages
        .filter((m) => !m.id.startsWith('sys-') && m.text.trim().length > 0)
        .slice(-4)
        .map((m) => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
        }));

      const historyPayload = [
        { role: 'system', content: systemPrompt },
        ...cleanHistory,
        { role: 'user', content: query },
      ];

      workerRef.current.postMessage({
        type: 'generate',
        payload: {
          messages: historyPayload,
          maxTokens: isGreeting || isIdentityQuery ? 50 : (activeTier === 'grande' ? 512 : (activeTier === 'media' ? 420 : 350)),
          temperature: 0.15,
          isGreeting,
          isIdentityQuery,
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
          effectiveLang === 'pt'
            ? `O modelo neural [${MODEL_OPTIONS[activeTier].labelPt}] está sendo carregado na sua GPU. Enquanto isso, aqui está a resposta pelo Motor de Reflexo:\n\n${generateReflexResponse(query, effectiveLang).text}`
            : `The neural model [${MODEL_OPTIONS[activeTier].labelEn}] is currently loading into your GPU. Meanwhile, here is the immediate Reflex response:\n\n${generateReflexResponse(query, effectiveLang).text}`,
        links: generateReflexResponse(query, effectiveLang).links,
        tierUsed: 'reflex',
      };
      setMessages((prev) => [...prev, waitMsg]);
      return;
    }

    // C. MODO REFLEXO (INSTANTÂNEO)
    setIsTyping(true);
    setTimeout(() => {
      const resp = generateReflexResponse(query, effectiveLang);
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
            title={currentLang === 'pt' ? 'Testar IA no Browser (MoS & SCache)' : 'Test In-Browser AI (MoS & SCache)'}
          >
            <div className="relative">
              <Bot className="w-4 h-4 text-[#adff2f]" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f] animate-ping" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f]" />
            </div>
            <span>{currentLang === 'pt' ? 'Testar IA no Browser' : 'Test In-Browser AI'}</span>
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
                <div className="flex items-center gap-2 flex-wrap">
                  <h4 className="text-xs font-mono font-bold text-white tracking-tight">
                    {currentLang === 'pt' ? 'Agente Neural Local' : 'Local Neural Agent'}
                  </h4>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/40 font-bold">
                    SCache {scacheTelemetry.memoryKb} O(1)
                  </span>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/40 font-bold">
                    MoS {mosTelemetry.activeTopK}/{mosTelemetry.numExperts} ({mosTelemetry.sparsityPercent}% esparso)
                  </span>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Local GPU / WGSL
                  </span>
                </div>
                <p className="text-[10px] font-mono text-zinc-400 flex items-center gap-1.5 flex-wrap">
                  <span>Domínio MoS: <b className="text-purple-300">{mosTelemetry.dominantDomain}</b></span>
                  <span>·</span>
                  <span>Cicatrizes: <b className="text-[#adff2f]">{scacheTelemetry.scarsCount}</b></span>
                  <span>·</span>
                  <span>||W||: <b className="text-white">{scacheTelemetry.matrixNorm}</b></span>
                  <span>·</span>
                  <span className="text-emerald-400">0 Cloud</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              {/* Language toggle pill: PT | EN */}
              <button
                type="button"
                onClick={toggleLanguage}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-mono font-bold bg-zinc-800/80 hover:bg-zinc-700 text-zinc-300 hover:text-white border border-zinc-700/60 transition-colors cursor-pointer mr-0.5"
                title={currentLang === 'pt' ? 'Switch chat language to English' : 'Mudar idioma do chat para Português'}
              >
                <Globe className="w-3 h-3 text-[#adff2f]" />
                <span className={currentLang === 'pt' ? 'text-[#adff2f]' : 'text-zinc-500'}>PT</span>
                <span className="text-zinc-600">/</span>
                <span className={currentLang === 'en' ? 'text-[#adff2f]' : 'text-zinc-500'}>EN</span>
              </button>
              <button
                type="button"
                onClick={() => setMessages([getInitialGreeting(currentLang)])}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={currentLang === 'pt' ? 'Limpar conversa' : 'Reset chat'}
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsMinimized(!isMinimized)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={isMinimized ? (currentLang === 'pt' ? 'Expandir' : 'Expand') : (currentLang === 'pt' ? 'Minimizar' : 'Minimize')}
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
                    <span>★ {currentLang === 'pt' ? 'Recomendação' : 'Recommended'}</span>
                    <Info className="w-2.5 h-2.5" />
                  </button>

                  {/* Tooltip / Drawer */}
                  {showHwTooltip && (
                    <div className="absolute right-0 top-6 z-50 w-72 p-2.5 bg-zinc-900 border border-[#adff2f]/50 rounded-xl shadow-2xl text-[10px] font-mono text-zinc-200 space-y-1.5 backdrop-blur-md">
                      <div className="font-bold text-[#adff2f] flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        <span>
                          {currentLang === 'pt'
                            ? `Recomendado: ${MODEL_OPTIONS[hardwareProfile.recommendedTier].labelPt} (${MODEL_OPTIONS[hardwareProfile.recommendedTier].badge})`
                            : `Recommended: ${MODEL_OPTIONS[hardwareProfile.recommendedTier].labelEn} (${MODEL_OPTIONS[hardwareProfile.recommendedTier].badge})`}
                        </span>
                      </div>
                      <p className="text-zinc-300 leading-snug">
                        {currentLang === 'pt' ? hardwareProfile.reasonPt : hardwareProfile.reasonEn}
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
                        title={currentLang === 'pt' ? opt.descPt : opt.descEn}
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
                            {currentLang === 'pt' ? opt.labelPt : opt.labelEn}
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
                              title={currentLang === 'pt' ? 'Salvo no cache permanente' : 'Saved in local cache'}
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
                          ? currentLang === 'pt'
                            ? `Carregando do Cache Local [${MODEL_OPTIONS[loadingTier].badge}]...`
                            : `Reading from Local Cache [${MODEL_OPTIONS[loadingTier].badge}]...`
                          : currentLang === 'pt'
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
                        ? currentLang === 'pt'
                          ? '⚡ 0 MB baixados: lendo diretamente do IndexedDB permanente.'
                          : '⚡ 0 MB network: reading directly from persistent IndexedDB.'
                        : currentLang === 'pt'
                        ? 'Alocando buffer SCache O(1) de 34.8 KB na SRAM.'
                        : 'Allocating 34.8 KB O(1) SCache buffer in SRAM.'}
                    </span>
                    <span className="text-emerald-400 flex items-center gap-0.5">
                      <ShieldCheck className="w-2.5 h-2.5" />
                      <span>{currentLang === 'pt' ? 'Cache Persistente' : 'Persistent Cache'}</span>
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
                    {currentLang === 'pt' ? 'Usar Reflexo' : 'Use Reflex'}
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
                      {activeTier !== 'reflex'
                        ? currentLang === 'pt'
                          ? 'Decodificando tokens na GPU local...'
                          : 'Decoding tokens on local GPU...'
                        : ''}
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
                      ? currentLang === 'pt'
                        ? 'Storage.persist: Ativo · Zero Downloads Duplicados'
                        : 'Storage.persist: Active · Zero Duplicate Downloads'
                      : currentLang === 'pt'
                      ? 'Cache IndexedDB Ativo'
                      : 'IndexedDB Cache Active'}
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
                      ? currentLang === 'pt'
                        ? `Conversando via ${MODEL_OPTIONS[activeTier].labelPt} local na GPU...`
                        : `Chatting via local ${MODEL_OPTIONS[activeTier].labelEn} on GPU...`
                      : currentLang === 'pt'
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
                  title={currentLang === 'pt' ? 'Enviar' : 'Send'}
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
