/**
 * Dedicated WebGPU Worker for Portfolio Agent v4.0
 *
 * Runs `BrowserMoSWebGPUEngine` (15 custom WGSL 1.0 compute shaders + `SCache O(1)` +
 * `GoldenSliceScarStoreV4` `HTTP 206` universal cartridges) off the main UI thread so
 * 3D portfolio canvases remain locked at 60 FPS.
 *
 * Zero `@huggingface/transformers`, Zero ONNX Runtime, Zero linear O(T) KV Cache.
 */

import {
  BrowserMoSWebGPUEngine,
  checkMosShardsCached,
} from './mosWebgpuEngine';
import { BrowserScarStoreHTTP206 } from './scarstoreHttp206Browser';

let engine: BrowserMoSWebGPUEngine | null = null;
let standaloneRouter: BrowserScarStoreHTTP206 | null = null;
let isLoading = false;
let isGenerating = false;
let currentAbort: AbortController | null = null;

async function ensureRouterLoaded(baseUrl: string): Promise<BrowserScarStoreHTTP206 | null> {
  if (engine && engine.scarStore) return engine.scarStore;
  if (standaloneRouter) return standaloneRouter;
  try {
    const cleanBase = baseUrl.replace(/\/$/, '');
    standaloneRouter = await BrowserScarStoreHTTP206.load(`${cleanBase}/scarstore_router.json`);
    return standaloneRouter;
  } catch (err) {
    console.warn('[MoS WebGPU Worker] Router preload notice:', err);
    return null;
  }
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data || {};
  const baseUrl: string = payload?.baseUrl || '/webgpu/qwen0p5b';

  // 1. Check Cache + Preload Lightweight `< 90 KB` Golden-Slice Router Index (`0 B` cartridge payload!)
  if (type === 'init_router') {
    const [shardsCached, router] = await Promise.all([
      checkMosShardsCached(baseUrl),
      ensureRouterLoaded(baseUrl),
    ]);
    self.postMessage({
      type: 'router_ready',
      shardsCached,
      capsules: router ? router.getCapsulesDirectory() : [],
      recordKb: router ? Math.round((router.recordBytes / 1024) * 100) / 100 : 61.25,
      kLang: router ? router.kLang : 14,
      midLayerIdx: router ? router.midLayerIdx : 15,
    });
    return;
  }

  // 2. Boot Native WebGPU / WGSL Engine (`15` Compute Pipelines + `SCache O(1)`)
  if (type === 'load') {
    if (engine) {
      self.postMessage({
        type: 'ready',
        adapterLabel: engine.adapterLabel,
        scacheBytes: engine.scacheGpuBytes,
        scacheKb: engine.scacheKbFormatted,
        vramWeightsMiB: Math.round((engine.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100,
        fromCache: true,
        capsules: engine.scarStore ? engine.scarStore.getCapsulesDirectory() : [],
      });
      return;
    }
    if (isLoading) return;
    isLoading = true;

    try {
      const preCached = await checkMosShardsCached(baseUrl);
      self.postMessage({
        type: 'loading_start',
        wasAlreadyCached: preCached,
      });

      engine = await BrowserMoSWebGPUEngine.create(baseUrl, (prog) => {
        self.postMessage({
          type: 'download_progress',
          phase: prog.phase,
          percent: prog.pct,
          detail: prog.detail,
          loaded: prog.loadedBytes ?? 0,
          total: prog.totalBytes ?? 0,
          fromCache: prog.fromCache ?? preCached,
        });
      });

      // Preserve any unlearn state applied before engine boot
      if (standaloneRouter && engine.scarStore) {
        for (const uid of standaloneRouter.unlearnedIds) {
          engine.scarStore.unlearnCapsule(uid);
        }
      }

      self.postMessage({
        type: 'ready',
        adapterLabel: engine.adapterLabel,
        scacheBytes: engine.scacheGpuBytes,
        scacheKb: engine.scacheKbFormatted,
        vramWeightsMiB: Math.round((engine.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100,
        fromCache: engine.loadedFromCache,
        capsules: engine.scarStore ? engine.scarStore.getCapsulesDirectory() : [],
      });
    } catch (err: any) {
      console.error('[MoS WebGPU Worker] Engine boot error:', err);
      self.postMessage({
        type: 'error',
        code: 'WEBGPU_BOOT_FAILED',
        error: err?.message || 'Falha ao inicializar o motor WebGPU/WGSL.',
      });
    } finally {
      isLoading = false;
    }
    return;
  }

  // 3. Generate Streaming Response on GPU (`MoS INT4+ASVD` + `SCache O(1)` + `HTTP 206` Cartridges)
  if (type === 'generate') {
    if (!engine) {
      self.postMessage({ type: 'error', error: 'Motor WebGPU/WGSL ainda não está carregado na GPU.' });
      return;
    }
    if (isGenerating) {
      self.postMessage({ type: 'error', error: 'Uma geração já está em andamento na GPU.' });
      return;
    }

    const {
      query,
      maxNewTokens = 260,
      enableScarStore = true,
      resetCache = false,
      msgId,
    } = payload || {};

    isGenerating = true;
    currentAbort = new AbortController();

    try {
      const result = await engine.generate(query, {
        maxNewTokens,
        enableScarStore,
        resetCache,
        signal: currentAbort.signal,
        onRouting: (routeRes) => {
          self.postMessage({
            type: 'routing_telemetry',
            msgId,
            activeCapsules: routeRes.active.map((a) => ({
              capsuleId: a.capsuleId,
              gate: Math.round(a.gate * 10000) / 10000,
              fromCache: a.fromCache,
              title: a.meta?.metadata?.title,
              titlePt: a.meta?.metadata?.title_pt,
              category: a.meta?.metadata?.category,
              links: a.meta?.metadata?.links || [],
            })),
            turnHttp206Bytes: routeRes.turnBytesFetched,
            httpEvents: routeRes.httpEvents,
            capsules: engine?.scarStore ? engine.scarStore.getCapsulesDirectory() : [],
          });
        },
        onToken: (tokUpdate) => {
          self.postMessage({
            type: 'token',
            msgId,
            text: tokUpdate.text,
            deltaToken: tokUpdate.deltaToken,
            generatedTokens: tokUpdate.generatedTokens,
            tokPerSec: Math.round(tokUpdate.tokPerSec * 10) / 10,
            scacheBytes: tokUpdate.scacheBytes,
            scacheDeltaBytes: tokUpdate.scacheDeltaBytes,
            confidence: tokUpdate.confidence,
            margin: tokUpdate.margin,
          });
        },
      });

      self.postMessage({
        type: 'generation_end',
        msgId,
        result,
        capsules: engine.scarStore ? engine.scarStore.getCapsulesDirectory() : [],
      });
    } catch (err: any) {
      console.error('[MoS WebGPU Worker] Generation error:', err);
      self.postMessage({
        type: 'error',
        msgId,
        error: err?.message || 'Erro durante a decodificação WebGPU/WGSL.',
      });
    } finally {
      isGenerating = false;
      currentAbort = null;
    }
    return;
  }

  // 4. Abort Active Decode
  if (type === 'abort') {
    if (currentAbort) {
      currentAbort.abort();
    }
    return;
  }

  // 5. Reset SCache O(1) Conversation State
  if (type === 'reset_scache') {
    if (engine) {
      engine.resetSCache();
    }
    self.postMessage({
      type: 'scache_reset',
      scacheBytes: engine ? engine.scacheGpuBytes : 2799232,
      scacheKb: engine ? engine.scacheKbFormatted : '2733.6 KB',
    });
    return;
  }

  // 6. LGPD `< 0.2 ms` Physical `0x00` Unlearn or Restore of a Cartridge
  if (type === 'toggle_unlearn') {
    const capsuleId: string = payload?.capsuleId;
    const store = engine?.scarStore || standaloneRouter;
    if (store && capsuleId) {
      if (store.unlearnedIds.has(capsuleId)) {
        store.restoreCapsule(capsuleId);
        self.postMessage({
          type: 'capsule_restored',
          capsuleId,
          capsules: store.getCapsulesDirectory(),
        });
      } else {
        const wipe = store.unlearnCapsule(capsuleId);
        self.postMessage({
          type: 'capsule_unlearned',
          ...wipe,
          capsules: store.getCapsulesDirectory(),
        });
      }
    }
    return;
  }

  // 7. Unload Engine from VRAM
  if (type === 'unload') {
    if (engine) {
      engine.destroy();
      engine = null;
    }
    self.postMessage({ type: 'unloaded' });
  }
};
