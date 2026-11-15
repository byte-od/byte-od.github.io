/**
 * Dedicated WebGPU Worker for Portfolio Agent v4.2 (JARVIS Qwen2.5-3B Neural Engine)
 *
 * Runs `BrowserMoSWebGPUEngine` (20 custom WGSL 1.0 compute shaders + `VQ4-Scar` + `ProtectedInt4Scar` +
 * `SCache O(1)` + `GoldenSliceScarStoreV4` `HTTP 206` universal `140 KB` cartridges) off the main UI thread
 * so 3D portfolio canvases and the Jarvis Neural Face remain locked at 60 FPS.
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
let activeGeneratePromise: Promise<void> | null = null;

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
  const baseUrl: string = payload?.baseUrl || '/webgpu/qwen3b';

  // 1. Check Cache + Preload Lightweight Golden-Slice Router Index (`0 B` cartridge payload!)
  if (type === 'init_router') {
    const [shardsCached, router] = await Promise.all([
      checkMosShardsCached(baseUrl),
      ensureRouterLoaded(baseUrl),
    ]);
    self.postMessage({
      type: 'router_ready',
      shardsCached,
      capsules: router ? router.getCapsulesDirectory() : [],
      recordKb: router ? Math.round((router.recordBytes / 1024) * 100) / 100 : 140.0,
      kLang: router ? router.kLang : 14,
      midLayerIdx: router ? router.midLayerIdx : 23,
    });
    return;
  }

  // 2. Boot Native Qwen 3B WebGPU / WGSL Engine (`20` Compute Pipelines + `SCache O(1)`)
  if (type === 'load') {
    if (engine) {
      self.postMessage({
        type: 'ready',
        adapterLabel: engine.adapterLabel,
        scacheBytes: engine.scacheGpuBytes,
        scacheKb: engine.scacheKbFormatted,
        vramWeightsMiB: Math.round((engine.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100,
        pointScarReady: Boolean(engine.pointScarFlow),
        pointScarShardMiB: engine.pointScarFlow
          ? Math.round((engine.pointScarFlow.shardBytes / (1024 * 1024)) * 100) / 100
          : 0,
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
        pointScarReady: Boolean(engine.pointScarFlow),
        pointScarShardMiB: engine.pointScarFlow
          ? Math.round((engine.pointScarFlow.shardBytes / (1024 * 1024)) * 100) / 100
          : 0,
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

  // 3. Generate Streaming Response on GPU (`MoS VQ4-Scar` + `SCache O(1)` + `HTTP 206` Cartridges)
  if (type === 'generate') {
    if (!engine) {
      self.postMessage({ type: 'error', error: 'Motor WebGPU/WGSL ainda não está carregado na GPU.' });
      return;
    }
    // If an aborted turn is finishing its ~80ms GPU readback, abort & await cleanly instead of failing
    if (isGenerating && activeGeneratePromise) {
      currentAbort?.abort();
      try {
        await activeGeneratePromise;
      } catch {
        // Ignore prior aborted turn cleanup
      }
    }

    const {
      query,
      maxNewTokens = 148,
      enableScarStore = true,
      resetCache = false,
      msgId,
    } = payload || {};

    isGenerating = true;
    const abortCtrl = new AbortController();
    currentAbort = abortCtrl;

    const runTurn = async () => {
      try {
        if (!engine) throw new Error('Motor WebGPU/WGSL indisponível.');
        const result = await engine.generate(query, {
          maxNewTokens,
          enableScarStore,
          resetCache,
          signal: abortCtrl.signal,
          onPrefillProgress: (done, total) => {
            self.postMessage({
              type: 'prefill_progress',
              msgId,
              done,
              total,
            });
          },
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
              routingDiagnostics: routeRes.diagnostics,
              turnHttp206Bytes: routeRes.turnBytesFetched,
              httpEvents: routeRes.httpEvents,
              epistemicMode: routeRes.epistemicMode,
              pointScarMidF32: routeRes.pointScarSample?.midCloudF32 ?? null,
              pointScarFinalF32: routeRes.pointScarSample?.finalCloudF32 ?? null,
              pointScarMs: routeRes.pointScarSample?.latencyMs ?? null,
              pointScarSteps: routeRes.pointScarSample?.numSteps ?? null,
              pointScarShardBytes: routeRes.pointScarSample?.shardBytes ?? null,
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
              uncertainty: tokUpdate.uncertainty,
              horizonScale: tokUpdate.horizonScale,
              top1TokenId: tokUpdate.top1TokenId,
            });
          },
        });

        if (abortCtrl.signal.aborted) {
          self.postMessage({
            type: 'generation_aborted',
            msgId,
          });
          return;
        }

        self.postMessage({
          type: 'generation_end',
          msgId,
          result,
          capsules: engine.scarStore ? engine.scarStore.getCapsulesDirectory() : [],
        });
      } catch (err: any) {
        if (err?.name === 'AbortError' || abortCtrl.signal.aborted) {
          self.postMessage({
            type: 'generation_aborted',
            msgId,
          });
          return;
        }
        console.error('[MoS WebGPU Worker] Generation error:', err);
        const lost = Boolean(engine?.isDeviceLost || err?.code === 'WEBGPU_DEVICE_LOST');
        if (lost && engine) {
          engine.destroy();
          engine = null;
        }
        self.postMessage({
          type: 'error',
          code: lost ? 'WEBGPU_DEVICE_LOST' : undefined,
          msgId,
          error: err?.message || 'Erro durante a decodificação WebGPU/WGSL.',
        });
      } finally {
        if (currentAbort === abortCtrl) {
          isGenerating = false;
          currentAbort = null;
        }
      }
    };

    activeGeneratePromise = runTurn();
    await activeGeneratePromise;
    activeGeneratePromise = null;
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
      scacheBytes: engine ? engine.scacheGpuBytes : 15345792,
      scacheKb: engine ? engine.scacheKbFormatted : '14986.1 KB',
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
