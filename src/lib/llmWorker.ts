import { pipeline, env, TextStreamer, InterruptableStoppingCriteria } from '@huggingface/transformers';
import { SCacheMemoryCore } from './scacheEngine';
import { MixtureOfScarsCore } from './mosEngine';

// Configuration for in-browser client execution
env.allowLocalModels = false;
env.useBrowserCache = true;

// Instanciação dos Motores Matemáticos Proprietários:
// 1. SCache: Memória Associativa Fast-Weights Hebbiana O(1) (~36.8 KB)
// 2. MoS: Roteador Esparso de Micro-Especialistas em Cicatriz (Top-K Low-Rank)
const scacheCore = new SCacheMemoryCore(96);
const mosCore = new MixtureOfScarsCore(96, 8, 2);

interface ModelConfig {
  id: string;
  name: string;
  dtype: 'q4' | 'q4f16' | 'fp16' | 'fp32';
}

export const MODELS_CONFIG: Record<string, ModelConfig> = {
  leve: {
    id: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    name: 'SmolLM2-135M-Instruct (INT4)',
    dtype: 'q4',
  },
  media: {
    id: 'onnx-community/Qwen2.5-0.5B-Instruct',
    name: 'Qwen2.5-0.5B-Instruct (INT4)',
    dtype: 'q4',
  },
  grande: {
    id: 'onnx-community/Llama-3.2-1B-Instruct',
    name: 'Llama-3.2-1B-Instruct (INT4)',
    dtype: 'q4',
  },
};

let currentPipeline: any = null;
let currentModelKey: string | null = null;
let isGenerating = false;
let isLoading = false;

// Check if WebGPU is available in this worker context
async function checkWebGPU(): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const adapter = await (navigator as any).gpu.requestAdapter();
      return !!adapter;
    }
  } catch (e) {
    console.warn('[LLM Worker] WebGPU check failed, falling back to WASM:', e);
  }
  return false;
}

// Check if model files already exist in local browser cache (zero download guarantee)
async function isModelCached(modelId: string): Promise<boolean> {
  try {
    if (typeof self !== 'undefined' && 'caches' in self) {
      const cache = await self.caches.open('transformers-cache');
      const keys = await cache.keys();
      return keys.some((req) => req.url.includes(modelId) && req.url.includes('.onnx'));
    }
  } catch (e) {
    console.warn('[LLM Worker] Cache inspection error:', e);
  }
  return false;
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  // 0. TELEMETRIA DO SCACHE O(1) & MOS
  if (type === 'get_scache_telemetry' || type === 'get_mos_scache_telemetry') {
    self.postMessage({
      type: 'scache_telemetry',
      telemetry: scacheCore.getTelemetry(),
      mosTelemetry: mosCore.getTelemetry(),
    });
    return;
  }

  // 1. VERIFICAR STATUS DO CACHE (GARANTIA DE NÃO RE-BAIXAR)
  if (type === 'check_cache') {
    try {
      const results: Record<string, boolean> = {};
      for (const [key, cfg] of Object.entries(MODELS_CONFIG)) {
        results[key] = await isModelCached(cfg.id);
      }
      self.postMessage({ type: 'cache_status', results });
    } catch (err) {
      console.warn('[LLM Worker] Failed to check cache status:', err);
    }
    return;
  }

  // 2. CARREGAR MODELO
  if (type === 'load') {
    const { modelKey } = payload;
    const targetConfig = MODELS_CONFIG[modelKey];

    if (!targetConfig) {
      self.postMessage({ type: 'error', error: `Modelo inválido: ${modelKey}` });
      return;
    }

    // Se já está residente na memória da GPU
    if (currentPipeline && currentModelKey === modelKey) {
      self.postMessage({
        type: 'ready',
        modelKey,
        name: targetConfig.name,
        device: 'memory_resident',
        fromCache: true,
      });
      return;
    }

    // Trava de concorrência: impede downloads paralelos duplicados
    if (isLoading) {
      console.warn('[LLM Worker] Carregamento já em andamento. Ignorando requisição concorrente.');
      return;
    }

    isLoading = true;

    try {
      // Liberar pipeline anterior da VRAM antes de carregar o novo (preservação de memória da aba)
      if (currentPipeline) {
        currentPipeline = null;
        currentModelKey = null;
      }

      const hasWebGPU = await checkWebGPU();
      const preferredDevice = hasWebGPU ? 'webgpu' : 'wasm';
      const wasAlreadyCached = await isModelCached(targetConfig.id);

      self.postMessage({
        type: 'loading_start',
        modelKey,
        device: preferredDevice,
        wasAlreadyCached,
      });

      // Progress callback tracking download percentage
      const progressCallback = (p: any) => {
        if (p.status === 'progress' && p.total > 0) {
          const percent = Math.min(100, Math.round((p.loaded / p.total) * 100));
          self.postMessage({
            type: 'download_progress',
            modelKey,
            percent,
            loaded: p.loaded,
            total: p.total,
            file: p.file || 'weights',
          });
        } else if (p.status === 'initiate') {
          self.postMessage({
            type: 'download_initiate',
            modelKey,
            file: p.file,
          });
        } else if (p.status === 'ready') {
          self.postMessage({
            type: 'download_ready',
            modelKey,
            file: p.file,
          });
        }
      };

      currentPipeline = await pipeline('text-generation', targetConfig.id, {
        dtype: targetConfig.dtype,
        device: preferredDevice,
        progress_callback: progressCallback,
      });

      currentModelKey = modelKey;

      self.postMessage({
        type: 'ready',
        modelKey,
        name: targetConfig.name,
        device: preferredDevice,
        fromCache: wasAlreadyCached,
      });
    } catch (err: any) {
      console.error('[LLM Worker] Failed to load model:', err);
      self.postMessage({
        type: 'error',
        error: err?.message || 'Falha ao compilar shaders WebGPU ou ler tensores do cache.',
      });
    } finally {
      isLoading = false;
    }
    return;
  }

  // 3. GERAR TEXTO
  if (type === 'generate') {
    if (!currentPipeline) {
      self.postMessage({ type: 'error', error: 'Nenhum modelo neural carregado na GPU.' });
      return;
    }

    if (isGenerating) {
      self.postMessage({ type: 'error', error: 'Uma inferência já está em andamento.' });
      return;
    }

    const { messages, maxTokens = 160, temperature = 0.25, isGreeting = false, isIdentityQuery = false } = payload;
    isGenerating = true;

    try {
      let tokenCount = 0;
      let fullGeneratedText = '';
      const startTime = performance.now();

      const lastUserMsg = Array.isArray(messages) && messages.length > 0
        ? messages[messages.length - 1]?.content || ''
        : '';

      // 1. Projeção Semântica e Roteamento Mixture of Scars (MoS Top-K)
      const userEmbedding = scacheCore.embedText(lastUserMsg);
      const mosResult = mosCore.routeAndProject(userEmbedding);

      // 2. Consulta Associativa SCache O(1)
      const scacheRetrieval = scacheCore.retrieve(lastUserMsg);

      const stoppingCriteria = new InterruptableStoppingCriteria();
      let hasInterrupted = false;

      const STOP_TOKENS = [
        '<|eot_id|>',
        '<|end_of_text|>',
        '<|im_end|>',
        '</s>',
        '<|end|>',
        '[END]',
        'User:',
        'Visitor:',
        'Usuário:',
      ];

      const streamer = new TextStreamer(currentPipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          if (hasInterrupted) return;

          tokenCount++;
          fullGeneratedText += chunk;

          // 0. Trava de Segurança Anti-Personificação Imediata
          // Se o modelo tentar falar "Eu sou Julio" em 1ª pessoa, interceptamos na hora
          const trimmedGen = fullGeneratedText.trim();
          if (/^(eu sou julio|eu sou o julio|eu me chamo julio|me chamo julio)/i.test(trimmedGen)) {
            hasInterrupted = true;
            stoppingCriteria.interrupt();
            self.postMessage({
              type: 'token',
              token: ' o assistente de IA oficial do portfólio de Julio Reis, executado no seu navegador via WebGPU e MoS/SCache. Como posso te ajudar?',
            });
            return;
          }
          if (/^(i am julio|i'm julio|my name is julio)/i.test(trimmedGen)) {
            hasInterrupted = true;
            stoppingCriteria.interrupt();
            self.postMessage({
              type: 'token',
              token: " the official AI assistant for Julio Reis's portfolio, running locally via WebGPU and MoS/SCache. How can I help you today?",
            });
            return;
          }

          // 1. Checagem de tokens de parada residuais no texto decodificado
          for (const stopTok of STOP_TOKENS) {
            if (fullGeneratedText.includes(stopTok)) {
              hasInterrupted = true;
              stoppingCriteria.interrupt();
              return;
            }
          }

          // 2. Trava anti-alucinação / anti-prolixidade em saudações e consultas de identidade
          if ((isGreeting || isIdentityQuery) && tokenCount >= 18 && /[.!?]\s*$/.test(fullGeneratedText.trim())) {
            hasInterrupted = true;
            stoppingCriteria.interrupt();
          }

          // 3. Trava de segurança para evitar loop infinito sem pontuação (diarreia de buzzwords)
          const wordsWithoutPunct = fullGeneratedText.split(/[.!?\n]/).pop()?.trim().split(/\s+/) || [];
          if (wordsWithoutPunct.length > 35) {
            hasInterrupted = true;
            stoppingCriteria.interrupt();
          }

          self.postMessage({
            type: 'token',
            token: chunk,
          });
        },
      });

      self.postMessage({
        type: 'generation_start',
        scacheResonance: scacheRetrieval.resonance,
        scacheTelemetry: scacheCore.getTelemetry(),
        mosTelemetry: mosResult.telemetry,
      });

      // Lista abrangente de EOS token IDs para Llama 3.2, Qwen 2.5 e SmolLM2
      const eosTokenIds = [128001, 128009, 151643, 151645, 2, 0];

      await currentPipeline(messages, {
        max_new_tokens: isGreeting ? Math.min(maxTokens, 45) : maxTokens,
        temperature: Math.max(0.10, Math.min(0.25, temperature)),
        do_sample: temperature > 0.05,
        repetition_penalty: 1.08,
        top_k: 25,
        top_p: 0.85,
        eos_token_id: eosTokenIds,
        stopping_criteria: [stoppingCriteria],
        streamer: streamer,
      });

      // Consolidação na Memória Associativa SCache O(1)
      if (lastUserMsg && fullGeneratedText) {
        scacheCore.absorbInteraction(lastUserMsg, fullGeneratedText);
      }

      const totalTimeMs = performance.now() - startTime;
      const tokensPerSec = totalTimeMs > 0 ? (tokenCount / (totalTimeMs / 1000)).toFixed(1) : '0';

      self.postMessage({
        type: 'generation_end',
        stats: {
          tokenCount,
          timeMs: Math.round(totalTimeMs),
          tokensPerSec,
        },
        scacheTelemetry: scacheCore.getTelemetry(),
        mosTelemetry: mosCore.getTelemetry(),
      });
    } catch (err: any) {
      console.error('[LLM Worker] Generation error:', err);
      self.postMessage({
        type: 'error',
        error: err?.message || 'Erro durante a decodificação autorregressiva.',
      });
    } finally {
      isGenerating = false;
    }
    return;
  }

  // 4. DESCARREGAR PIPELINE PARA LIMPAR MEMÓRIA
  if (type === 'unload') {
    currentPipeline = null;
    currentModelKey = null;
    self.postMessage({ type: 'unloaded' });
    return;
  }
};
