import { pipeline, env, TextStreamer } from '@huggingface/transformers';

// Configuration for in-browser client execution
env.allowLocalModels = false;
env.useBrowserCache = true;

interface ModelConfig {
  id: string;
  name: string;
  dtype: 'q4' | 'q4f16' | 'fp16' | 'fp32';
}

export const MODELS_CONFIG: Record<string, ModelConfig> = {
  leve: {
    id: 'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA',
    name: 'SmolLM2-135M-Instruct (INT4)',
    dtype: 'q4',
  },
  media: {
    id: 'onnx-community/Qwen2.5-0.5B-Instruct',
    name: 'Qwen2.5-0.5B-Instruct (INT4)',
    dtype: 'q4',
  },
  grande: {
    id: 'onnx-community/Qwen2.5-1.5B-Instruct',
    name: 'Qwen2.5-1.5B-Instruct (INT4)',
    dtype: 'q4',
  },
};

let currentPipeline: any = null;
let currentModelKey: string | null = null;
let isGenerating = false;

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

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  // 1. CARREGAR MODELO
  if (type === 'load') {
    const { modelKey } = payload;
    const targetConfig = MODELS_CONFIG[modelKey];

    if (!targetConfig) {
      self.postMessage({ type: 'error', error: `Modelo inválido: ${modelKey}` });
      return;
    }

    if (currentPipeline && currentModelKey === modelKey) {
      self.postMessage({
        type: 'ready',
        modelKey,
        name: targetConfig.name,
        device: 'cached',
      });
      return;
    }

    try {
      const hasWebGPU = await checkWebGPU();
      const preferredDevice = hasWebGPU ? 'webgpu' : 'wasm';

      self.postMessage({
        type: 'loading_start',
        modelKey,
        device: preferredDevice,
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
      });
    } catch (err: any) {
      console.error('[LLM Worker] Failed to load model:', err);
      self.postMessage({
        type: 'error',
        error: err?.message || 'Falha ao compilar shaders WebGPU ou baixar tensores.',
      });
    }
  }

  // 2. GERAR TEXTO
  if (type === 'generate') {
    if (!currentPipeline) {
      self.postMessage({ type: 'error', error: 'Nenhum modelo neural carregado na GPU.' });
      return;
    }

    if (isGenerating) {
      self.postMessage({ type: 'error', error: 'Uma inferência já está em andamento.' });
      return;
    }

    const { messages, maxTokens = 350, temperature = 0.6 } = payload;
    isGenerating = true;

    try {
      let tokenCount = 0;
      const startTime = performance.now();

      const streamer = new TextStreamer(currentPipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          tokenCount++;
          self.postMessage({
            type: 'token',
            token: chunk,
          });
        },
      });

      self.postMessage({ type: 'generation_start' });

      await currentPipeline(messages, {
        max_new_tokens: maxTokens,
        temperature: temperature,
        do_sample: temperature > 0.0,
        streamer: streamer,
      });

      const totalTimeMs = performance.now() - startTime;
      const tokensPerSec = totalTimeMs > 0 ? (tokenCount / (totalTimeMs / 1000)).toFixed(1) : '0';

      self.postMessage({
        type: 'generation_end',
        stats: {
          tokenCount,
          timeMs: Math.round(totalTimeMs),
          tokensPerSec,
        },
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
  }

  // 3. DESCARREGAR
  if (type === 'unload') {
    currentPipeline = null;
    currentModelKey = null;
    self.postMessage({ type: 'unloaded' });
  }
};
