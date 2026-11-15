import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ExternalLink,
  Eye,
  Github,
  Layers,
  Maximize2,
  Minimize2,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Send,
  Shrink,
  Sparkles,
  Square,
  Trash2,
  Volume2,
  VolumeX,
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
  GoldenSliceRoutingDiagnostics,
  Http206Event,
} from '../lib/scarstoreHttp206Browser';
import {
  advanceSessionAdaptation,
  computeJarvisFaceState,
  createInitialSessionAdaptation,
  detectPolyglotLanguage,
  PolyglotLang,
  SessionAdaptationState,
} from '../lib/jarvisNeuralController';
import { JarvisFaceRig, JarvisMiniFaceOrb } from './JarvisFaceRig';
import { JarvisFloatingPointStage } from './JarvisFloatingPointStage';
import { probeWebGPUHardware } from '../lib/webgpuKernel';

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

const LANG_BADGES: Record<PolyglotLang, { flag: string; code: string }> = {
  pt: { flag: '🇧🇷', code: 'PT' },
  en: { flag: '🇺🇸', code: 'EN' },
  es: { flag: '🇪🇸', code: 'ES' },
  fr: { flag: '🇫🇷', code: 'FR' },
};

const WEBGPU_BASE_URL = `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/webgpu/qwen3b`;
const DEFAULT_SCACHE_BYTES = 15345792; // 14986.1 KB (112 sinks + 32 Widrow-Hoff slots + 144 window across 36 Qwen2.5-3B layers)

function formatScacheKb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export const PortfolioAgentWidget: React.FC<PortfolioAgentWidgetProps> = ({ lang = 'en' }) => {
  const [currentLang, setCurrentLang] = useState<'en' | 'pt'>(() => {
    if (lang === 'pt') return 'pt';
    if (typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('pt')) {
      return 'pt';
    }
    return lang;
  });

  // Floating Face Visibility State (compact corner floating face, expanded centered stage, or mini-orb tuck)
  const [isMinimized, setIsMinimized] = useState<boolean>(false);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [viewportSize, setViewportSize] = useState<{ w: number; h: number }>(() => ({
    w: typeof window !== 'undefined' ? window.innerWidth : 1280,
    h: typeof window !== 'undefined' ? window.innerHeight : 840,
  }));
  const [showInspector, setShowInspector] = useState<boolean>(false);
  const [captionDismissed, setCaptionDismissed] = useState<boolean>(false);

  // Unified Qwen WebGPU Hardware & Engine States
  const [hasWebGPU, setHasWebGPU] = useState<boolean>(true);
  const [adapterLabel, setAdapterLabel] = useState<string>('WebGPU');
  const [shardsCached, setShardsCached] = useState<boolean>(false);
  const [engineReady, setEngineReady] = useState<boolean>(false);
  const [engineLoading, setEngineLoading] = useState<boolean>(false);
  const [loadPhaseDetail, setLoadPhaseDetail] = useState<string>('');
  const [loadPercent, setLoadPercent] = useState<number>(0);
  const [loadLoadedMiB, setLoadLoadedMiB] = useState<number>(0);
  const [loadTotalMiB, setLoadTotalMiB] = useState<number>(1601.6);
  const [loadedFromCache, setLoadedFromCache] = useState<boolean>(false);
  const [scacheBytes, setScacheBytes] = useState<number>(DEFAULT_SCACHE_BYTES);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ScarStore v4.0 HTTP 206 Directory & Network Log
  const [capsules, setCapsules] = useState<CapsuleDirectoryEntry[]>([]);
  const [httpEvents, setHttpEvents] = useState<Http206Event[]>([]);

  // Input & Ephemeral Live Spoken Caption (Zero Chat Log / Zero History)
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [hasPendingQuery, setHasPendingQuery] = useState<boolean>(false);
  const [prefillStatus, setPrefillStatus] = useState<{ done: number; total: number } | null>(null);
  const [spokenLinks, setSpokenLinks] = useState<CapsuleLink[]>([]);
  const [latestTelemetry, setLatestTelemetry] = useState<TurnTelemetry | null>(null);

  // Sovereign JARVIS Live Qwen Neural-Motor Telemetry (`ZERO Math.random()`)
  const [sessionAdapt, setSessionAdapt] = useState<SessionAdaptationState>(() =>
    createInitialSessionAdaptation(currentLang)
  );
  const [lastUserQuery, setLastUserQuery] = useState<string>('');
  const [liveRoutingDiag, setLiveRoutingDiag] = useState<GoldenSliceRoutingDiagnostics | null>(null);
  const [liveActiveCapsuleIds, setLiveActiveCapsuleIds] = useState<string[]>([]);
  const [liveMaxGate, setLiveMaxGate] = useState<number>(0);
  const [liveTokenState, setLiveTokenState] = useState<{
    deltaToken: string;
    currentAgentText: string;
    generatedTokens: number;
    tokPerSec: number;
    confidence: number;
    margin: number;
    horizonScale: number;
  }>({
    deltaToken: '',
    currentAgentText: '',
    generatedTokens: 0,
    tokPerSec: 0,
    confidence: 0.92,
    margin: 0.84,
    horizonScale: 1.0,
  });
  const [unlearnShockActive, setUnlearnShockActive] = useState<boolean>(false);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(false);
  const [pointFieldEnabled, setPointFieldEnabled] = useState<boolean>(true);
  const [pointScarState, setPointScarState] = useState<{
    midF32: Float32Array | null;
    finalF32: Float32Array | null;
    latencyMs: number | null;
    steps: number;
    shardMiB: number;
    seq: number;
  }>({
    midF32: null,
    finalF32: null,
    latencyMs: null,
    steps: 4,
    shardMiB: 3.87,
    seq: 0,
  });

  // Refs for Qwen WebGPU Worker + Main-thread WebGPU fallback + Queued Auto-Boot Query
  const workerRef = useRef<Worker | null>(null);
  const mainThreadEngineRef = useRef<BrowserMoSWebGPUEngine | null>(null);
  const mainThreadRouterRef = useRef<BrowserScarStoreHTTP206 | null>(null);
  const mainThreadAbortRef = useRef<AbortController | null>(null);
  const useMainThreadRef = useRef<boolean>(false);
  const autoBootTriggeredRef = useRef<boolean>(false);
  const pendingQueryRef = useRef<string | null>(null);
  const activeMsgIdRef = useRef<string | null>(null);
  const abortWatchdogRef = useRef<number | null>(null);
  const spawnWorkerRef = useRef<(() => Worker | null) | null>(null);
  const lastUserQueryRef = useRef<string>('');
  const activeCapsulesRef = useRef<ActiveCartridgeBadge[]>([]);
  const voiceEnabledRef = useRef<boolean>(false);
  const unlearnTimerRef = useRef<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    voiceEnabledRef.current = voiceEnabled;
    if (!voiceEnabled && typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, [voiceEnabled]);

  useEffect(() => {
    const onResize = () => {
      setViewportSize({ w: window.innerWidth, h: window.innerHeight });
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    if (!isExpanded) return;
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        setIsExpanded(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isExpanded]);

  useEffect(() => {
    return () => {
      if (unlearnTimerRef.current) window.clearTimeout(unlearnTimerRef.current);
      if (abortWatchdogRef.current) window.clearTimeout(abortWatchdogRef.current);
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const speakPolyglotUtterance = useCallback(
    (rawAnswer: string, queryForLang: string) => {
      if (!voiceEnabledRef.current || typeof window === 'undefined' || !('speechSynthesis' in window)) {
        return;
      }
      try {
        window.speechSynthesis.cancel();
        const spkLang = detectPolyglotLanguage(queryForLang || rawAnswer, currentLang);
        const bcp47Map: Record<PolyglotLang, string> = {
          pt: 'pt-BR',
          en: 'en-US',
          es: 'es-ES',
          fr: 'fr-FR',
        };
        const cleanSpeech = rawAnswer
          .replace(/[⚡🏹🧩🛡️📡🗑️☁️🇧🇷🇺🇸🇪🇸🇫🇷•*_`]/g, '')
          .replace(/\s+/g, ' ')
          .trim();
        if (!cleanSpeech) return;
        const utter = new SpeechSynthesisUtterance(cleanSpeech);
        utter.lang = bcp47Map[spkLang];
        utter.rate = 1.04;
        utter.pitch = 0.95;
        window.speechSynthesis.speak(utter);
      } catch {
        // Ignore browser speech synthesis permission blocks
      }
    },
    [currentLang]
  );

  const prevPropLangRef = useRef<'en' | 'pt'>(lang);
  useEffect(() => {
    if (prevPropLangRef.current !== lang) {
      prevPropLangRef.current = lang;
      setCurrentLang(lang);
    }
  }, [lang]);

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
      if (typeof navigator !== 'undefined' && navigator.storage?.persist) {
        navigator.storage.persist().catch(() => {});
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Helper: Boot Qwen WebGPU Engine on Main Thread (Fallback only if Worker lacks `navigator.gpu`)
  const bootMainThreadWebGPU = useCallback(async () => {
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
      setErrorMessage(err?.message || 'Falha ao inicializar Qwen WebGPU.');
      return null;
    }
  }, []);

  // Helper: Run Qwen Generation on Main Thread Engine
  const runMainThreadGenerate = useCallback(
    async (query: string) => {
      const eng = mainThreadEngineRef.current;
      if (!eng) return;
      setIsGenerating(true);
      setCaptionDismissed(false);
      setPrefillStatus(null);
      const abortCtrl = new AbortController();
      mainThreadAbortRef.current = abortCtrl;
      let turnFetchedBytes = 0;

      try {
        const res = await eng.generate(query, {
          maxNewTokens: 148,
          enableScarStore: true,
          signal: abortCtrl.signal,
          onPrefillProgress: (done, total) => {
            setPrefillStatus({ done, total });
          },
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
            activeCapsulesRef.current = badges;
            const activeIds = badges.map((b) => b.capsuleId);
            setLiveActiveCapsuleIds(activeIds);
            setLiveMaxGate(badges.reduce((mx, b) => Math.max(mx, b.gate), 0));
            if (routeRes.diagnostics) {
              setLiveRoutingDiag(routeRes.diagnostics);
            }
            if (routeRes.pointScarSample) {
              const ps = routeRes.pointScarSample;
              setPointScarState((prev) => ({
                midF32: ps.midCloudF32,
                finalF32: ps.finalCloudF32,
                latencyMs: ps.latencyMs,
                steps: ps.numSteps,
                shardMiB: Math.round((ps.shardBytes / (1024 * 1024)) * 100) / 100,
                seq: prev.seq + 1,
              }));
            }
            turnFetchedBytes = routeRes.turnBytesFetched;
            setSpokenLinks(badges.flatMap((b) => b.links || []));
            if (routeRes.httpEvents.length > 0) {
              setHttpEvents((prev) => [...routeRes.httpEvents, ...prev].slice(0, 20));
            }
            if (eng.scarStore) {
              setCapsules(eng.scarStore.getCapsulesDirectory());
            }
            setLatestTelemetry({
              scacheBytes: eng.scacheGpuBytes,
              scacheDeltaBytes: 0,
              tokPerSec: 0,
              activeCapsules: badges,
              turnHttp206Bytes: routeRes.turnBytesFetched,
              confidence: 0,
              margin: 0,
            });
          },
          onToken: (tokUpd) => {
            setPrefillStatus(null);
            const roundedTokSec = Math.round(tokUpd.tokPerSec * 10) / 10;
            setLiveTokenState((prev) => ({
              deltaToken: tokUpd.deltaToken,
              currentAgentText: tokUpd.text,
              generatedTokens: tokUpd.generatedTokens,
              tokPerSec: roundedTokSec,
              confidence: 0.82 * prev.confidence + 0.18 * tokUpd.confidence,
              margin: 0.82 * prev.margin + 0.18 * tokUpd.margin,
              horizonScale: tokUpd.horizonScale,
            }));
            setLatestTelemetry({
              scacheBytes: tokUpd.scacheBytes,
              scacheDeltaBytes: tokUpd.scacheDeltaBytes,
              tokPerSec: roundedTokSec,
              activeCapsules: activeCapsulesRef.current,
              turnHttp206Bytes: turnFetchedBytes,
              confidence: tokUpd.confidence,
              margin: tokUpd.margin,
            });
          },
        });

        if (abortCtrl.signal.aborted) return;

        if (res.routingDiagnostics) {
          setLiveRoutingDiag(res.routingDiagnostics);
        }
        setLiveTokenState((prev) => ({
          ...prev,
          deltaToken: '',
          currentAgentText: res.answer,
          generatedTokens: res.generatedTokens,
          tokPerSec: res.decodeTokPerSec,
          confidence: res.meanConfidenceP1,
          margin: res.meanMarginP1P2,
        }));
        setLatestTelemetry({
          scacheBytes: res.scacheBytesFinal,
          scacheDeltaBytes: res.scacheDeltaBytes,
          tokPerSec: res.decodeTokPerSec,
          activeCapsules: activeCapsulesRef.current,
          turnHttp206Bytes: res.turnHttp206Bytes,
          confidence: res.meanConfidenceP1,
          margin: res.meanMarginP1P2,
          prefillMs: res.prefillMs,
        });
        setSessionAdapt((prev) =>
          advanceSessionAdaptation(prev, {
            userQuery: query,
            activatedCapsuleIds: activeCapsulesRef.current.map((b) => b.capsuleId),
            turnMeanConfidence: res.meanConfidenceP1,
            fallbackLang: currentLang,
          })
        );
        speakPolyglotUtterance(res.answer, query);

        if (eng.scarStore) {
          setCapsules(eng.scarStore.getCapsulesDirectory());
        }
      } catch (err: any) {
        if (err?.name === 'AbortError' || abortCtrl.signal.aborted) {
          return;
        }
        if (eng.isDeviceLost || err?.code === 'WEBGPU_DEVICE_LOST') {
          eng.destroy();
          mainThreadEngineRef.current = null;
          setEngineReady(false);
          setEngineLoading(false);
        }
        setErrorMessage(err?.message || 'Erro durante geração Qwen WebGPU.');
      } finally {
        setPrefillStatus(null);
        setIsGenerating(false);
        setLiveTokenState((prev) => ({ ...prev, deltaToken: '' }));
        mainThreadAbortRef.current = null;
      }
    },
    [currentLang, speakPolyglotUtterance]
  );

  const dispatchQwenQuery = useCallback(
    (query: string) => {
      if (abortWatchdogRef.current) {
        window.clearTimeout(abortWatchdogRef.current);
        abortWatchdogRef.current = null;
      }
      const msgId = `qwen-${Date.now()}-${Math.floor(performance.now() * 1000)}`;
      activeMsgIdRef.current = msgId;
      setIsGenerating(true);
      setCaptionDismissed(false);
      setPrefillStatus(null);
      activeCapsulesRef.current = [];
      setSpokenLinks([]);
      setLiveTokenState((prev) => ({
        ...prev,
        deltaToken: '',
        currentAgentText: '',
        generatedTokens: 0,
        confidence: 0.9,
        margin: 0.82,
      }));

      if (useMainThreadRef.current && mainThreadEngineRef.current) {
        void runMainThreadGenerate(query);
        return;
      }
      workerRef.current?.postMessage({
        type: 'generate',
        payload: {
          query,
          maxNewTokens: 148,
          enableScarStore: true,
          msgId,
        },
      });
    },
    [runMainThreadGenerate]
  );

  const handleBootEngine = useCallback(async () => {
    if (engineReady || engineLoading || !hasWebGPU) return;
    setErrorMessage(null);
    if (useMainThreadRef.current || !workerRef.current) {
      await bootMainThreadWebGPU();
      return;
    }
    workerRef.current.postMessage({
      type: 'load',
      payload: { baseUrl: WEBGPU_BASE_URL },
    });
  }, [bootMainThreadWebGPU, engineLoading, engineReady, hasWebGPU]);

  // 2. Initialize Dedicated Qwen WebGPU Worker + Auto-Wake if Shards Already Cached in Browser
  // CRITICAL: Empty dependency array `[]` so changing UI language (`currentLang`) NEVER calls `worker.terminate()`!
  const dispatchQwenRef = useRef(dispatchQwenQuery);
  dispatchQwenRef.current = dispatchQwenQuery;
  const bootMainThreadRef = useRef(bootMainThreadWebGPU);
  bootMainThreadRef.current = bootMainThreadWebGPU;
  const runMainThreadGenRef = useRef(runMainThreadGenerate);
  runMainThreadGenRef.current = runMainThreadGenerate;
  const speakRef = useRef(speakPolyglotUtterance);
  speakRef.current = speakPolyglotUtterance;
  const currentLangRef = useRef(currentLang);
  currentLangRef.current = currentLang;
  const hasWebGPURef = useRef(hasWebGPU);
  hasWebGPURef.current = hasWebGPU;

  useEffect(() => {
    const spawnWorker = (): Worker | null => {
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
          if (worker !== workerRef.current) return;
          const d = e.data || {};
          if (d.type === 'router_ready') {
            const cached = !!d.shardsCached;
            setShardsCached(cached);
            if (Array.isArray(d.capsules) && d.capsules.length > 0) {
              setCapsules(d.capsules);
            }
            // Seamless Zero-Network Auto-Boot when Qwen shards are already in CacheStorage!
            if (cached && hasWebGPURef.current && !autoBootTriggeredRef.current) {
              autoBootTriggeredRef.current = true;
              worker.postMessage({
                type: 'load',
                payload: { baseUrl: WEBGPU_BASE_URL },
              });
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
            if (d.pointScarShardMiB) {
              setPointScarState((prev) => ({ ...prev, shardMiB: d.pointScarShardMiB }));
            }
            if (Array.isArray(d.capsules) && d.capsules.length > 0) {
              setCapsules(d.capsules);
            }
            // Immediately run user's queued prompt on Qwen!
            if (pendingQueryRef.current) {
              const q = pendingQueryRef.current;
              pendingQueryRef.current = null;
              setHasPendingQuery(false);
              dispatchQwenRef.current(q);
            }
          } else if (d.type === 'prefill_progress') {
            if (d.msgId && d.msgId !== activeMsgIdRef.current) return;
            setPrefillStatus({ done: d.done ?? 0, total: d.total ?? 1 });
          } else if (d.type === 'routing_telemetry') {
            if (d.msgId && d.msgId !== activeMsgIdRef.current) return;
            const badges: ActiveCartridgeBadge[] = d.activeCapsules || [];
            activeCapsulesRef.current = badges;
            const activeIds = badges.map((b) => b.capsuleId);
            setLiveActiveCapsuleIds(activeIds);
            setLiveMaxGate(badges.reduce((mx, b) => Math.max(mx, b.gate), 0));
            if (d.routingDiagnostics) {
              setLiveRoutingDiag(d.routingDiagnostics);
            }
            if (d.pointScarFinalF32) {
              setPointScarState((prev) => ({
                midF32: d.pointScarMidF32 ? new Float32Array(d.pointScarMidF32) : null,
                finalF32: new Float32Array(d.pointScarFinalF32),
                latencyMs: typeof d.pointScarMs === 'number' ? d.pointScarMs : prev.latencyMs,
                steps: typeof d.pointScarSteps === 'number' ? d.pointScarSteps : 4,
                shardMiB:
                  typeof d.pointScarShardBytes === 'number'
                    ? Math.round((d.pointScarShardBytes / (1024 * 1024)) * 100) / 100
                    : prev.shardMiB,
                seq: prev.seq + 1,
              }));
            }
            setSpokenLinks(badges.flatMap((b) => b.links || []));
            if (Array.isArray(d.httpEvents) && d.httpEvents.length > 0) {
              setHttpEvents((prev) => [...d.httpEvents, ...prev].slice(0, 20));
            }
            if (Array.isArray(d.capsules)) {
              setCapsules(d.capsules);
            }
            setLatestTelemetry((prev) => ({
              scacheBytes: prev?.scacheBytes ?? DEFAULT_SCACHE_BYTES,
              scacheDeltaBytes: 0,
              tokPerSec: prev?.tokPerSec ?? 0,
              activeCapsules: badges,
              turnHttp206Bytes: d.turnHttp206Bytes ?? 0,
              confidence: prev?.confidence ?? 0,
              margin: prev?.margin ?? 0,
            }));
          } else if (d.type === 'token') {
            if (d.msgId && d.msgId !== activeMsgIdRef.current) return;
            setPrefillStatus(null);
            setLiveTokenState((prev) => ({
              deltaToken: d.deltaToken ?? '',
              currentAgentText: d.text ?? '',
              generatedTokens: d.generatedTokens ?? 0,
              tokPerSec: d.tokPerSec ?? 0,
              confidence: 0.82 * prev.confidence + 0.18 * (d.confidence ?? 0.85),
              margin: 0.82 * prev.margin + 0.18 * (d.margin ?? 0.7),
              horizonScale: d.horizonScale ?? 1.0,
            }));
            setLatestTelemetry((prev) => ({
              scacheBytes: d.scacheBytes ?? DEFAULT_SCACHE_BYTES,
              scacheDeltaBytes: d.scacheDeltaBytes ?? 0,
              tokPerSec: d.tokPerSec ?? 0,
              activeCapsules: activeCapsulesRef.current,
              turnHttp206Bytes: prev?.turnHttp206Bytes ?? 0,
              confidence: d.confidence ?? 0,
              margin: d.margin ?? 0,
            }));
          } else if (d.type === 'generation_end') {
            if (d.msgId && d.msgId !== activeMsgIdRef.current) return;
            activeMsgIdRef.current = null;
            setPrefillStatus(null);
            setIsGenerating(false);
            const res = d.result;
            if (Array.isArray(d.capsules)) setCapsules(d.capsules);
            if (res) {
              if (res.routingDiagnostics) {
                setLiveRoutingDiag(res.routingDiagnostics);
              }
              setLiveTokenState((prev) => ({
                ...prev,
                deltaToken: '',
                currentAgentText: res.answer || '',
                generatedTokens: res.generatedTokens ?? prev.generatedTokens,
                tokPerSec: res.decodeTokPerSec ?? prev.tokPerSec,
                confidence: res.meanConfidenceP1 ?? prev.confidence,
                margin: res.meanMarginP1P2 ?? prev.margin,
              }));
              setLatestTelemetry({
                scacheBytes: res.scacheBytesFinal,
                scacheDeltaBytes: res.scacheDeltaBytes,
                tokPerSec: res.decodeTokPerSec,
                activeCapsules: activeCapsulesRef.current,
                turnHttp206Bytes: res.turnHttp206Bytes,
                confidence: res.meanConfidenceP1,
                margin: res.meanMarginP1P2,
                prefillMs: res.prefillMs,
              });
              setSessionAdapt((prev) =>
                advanceSessionAdaptation(prev, {
                  userQuery: lastUserQueryRef.current,
                  activatedCapsuleIds: activeCapsulesRef.current.map((b) => b.capsuleId),
                  turnMeanConfidence: res.meanConfidenceP1 ?? 0.88,
                  fallbackLang: currentLangRef.current,
                })
              );
              speakRef.current(res.answer || '', lastUserQueryRef.current);
            } else {
              setLiveTokenState((prev) => ({ ...prev, deltaToken: '' }));
            }
          } else if (d.type === 'generation_aborted') {
            if (abortWatchdogRef.current) {
              window.clearTimeout(abortWatchdogRef.current);
              abortWatchdogRef.current = null;
            }
            setPrefillStatus(null);
            if (!activeMsgIdRef.current || d.msgId === activeMsgIdRef.current) {
              activeMsgIdRef.current = null;
              setIsGenerating(false);
              setLiveTokenState((prev) => ({ ...prev, deltaToken: '' }));
            }
          } else if (d.type === 'capsule_unlearned' || d.type === 'capsule_restored') {
            if (Array.isArray(d.capsules)) {
              setCapsules(d.capsules);
            }
          } else if (d.type === 'scache_reset') {
            if (d.scacheBytes) setScacheBytes(d.scacheBytes);
          } else if (d.type === 'error') {
            if (abortWatchdogRef.current) {
              window.clearTimeout(abortWatchdogRef.current);
              abortWatchdogRef.current = null;
            }
            if (d.code === 'WEBGPU_BOOT_FAILED' && hasWebGPURef.current && !useMainThreadRef.current) {
              const eng = await bootMainThreadRef.current();
              if (eng && pendingQueryRef.current) {
                const q = pendingQueryRef.current;
                pendingQueryRef.current = null;
                setHasPendingQuery(false);
                await runMainThreadGenRef.current(q);
              }
              return;
            }
            if (d.code === 'WEBGPU_DEVICE_LOST') {
              setEngineReady(false);
            }
            activeMsgIdRef.current = null;
            setPrefillStatus(null);
            setEngineLoading(false);
            setIsGenerating(false);
            setLiveTokenState((prev) => ({ ...prev, deltaToken: '' }));
            setErrorMessage(d.error || 'Erro no motor Qwen WebGPU.');
          }
        };

        return worker;
      } catch {
        BrowserScarStoreHTTP206.load(`${WEBGPU_BASE_URL}/scarstore_router.json`)
          .then((r) => {
            mainThreadRouterRef.current = r;
            setCapsules(r.getCapsulesDirectory());
          })
          .catch(() => {});
        return null;
      }
    };

    spawnWorkerRef.current = spawnWorker;
    spawnWorker();

    return () => {
      spawnWorkerRef.current = null;
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const handleSendMessage = useCallback(
    async (overrideText?: string) => {
      const query = (overrideText ?? inputMessage).trim();
      if (!query || isGenerating) return;

      lastUserQueryRef.current = query;
      setLastUserQuery(query);
      setInputMessage('');
      setErrorMessage(null);
      setCaptionDismissed(false);

      const detectedQueryLang = detectPolyglotLanguage(query, currentLang);
      if (detectedQueryLang === 'pt' && currentLang !== 'pt') {
        setCurrentLang('pt');
      }

      if (!hasWebGPU) {
        setErrorMessage(
          currentLang === 'pt'
            ? 'WebGPU necessário: este navegador não expõe navigator.gpu para executar o Qwen 3B (20 Shaders WGSL).'
            : 'WebGPU required: this browser does not expose navigator.gpu to run Qwen 3B (20 WGSL Shaders).'
        );
        return;
      }

      if (engineReady) {
        dispatchQwenQuery(query);
        return;
      }

      // Queue query and boot Qwen WebGPU immediately!
      pendingQueryRef.current = query;
      setHasPendingQuery(true);
      if (!engineLoading) {
        await handleBootEngine();
        if (useMainThreadRef.current && mainThreadEngineRef.current && pendingQueryRef.current) {
          const q = pendingQueryRef.current;
          pendingQueryRef.current = null;
          setHasPendingQuery(false);
          dispatchQwenQuery(q);
        }
      }
    },
    [currentLang, dispatchQwenQuery, engineLoading, engineReady, handleBootEngine, hasWebGPU, inputMessage, isGenerating]
  );

  // Global custom event listener so Hero CTA button focuses/summons the floating JARVIS face
  const handleSendRef = useRef(handleSendMessage);
  handleSendRef.current = handleSendMessage;
  useEffect(() => {
    const onOpenJarvis = (ev: Event) => {
      const detail = (ev as CustomEvent<{ prompt?: string }>).detail;
      setIsMinimized(false);
      setTimeout(() => {
        inputRef.current?.focus();
        if (detail?.prompt) {
          void handleSendRef.current(detail.prompt);
        }
      }, 60);
    };
    window.addEventListener('open-jarvis-agent', onOpenJarvis);
    return () => window.removeEventListener('open-jarvis-agent', onOpenJarvis);
  }, []);

  const handleStopGeneration = useCallback(() => {
    const wasWorkerGenerating = isGenerating && !useMainThreadRef.current;

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    pendingQueryRef.current = null;
    setHasPendingQuery(false);
    activeMsgIdRef.current = null;
    setIsGenerating(false);
    setPrefillStatus(null);
    setLiveTokenState((prev) => {
      if (!prev.currentAgentText.trim()) {
        setCaptionDismissed(true);
      }
      return { ...prev, deltaToken: '' };
    });

    if (useMainThreadRef.current) {
      mainThreadAbortRef.current?.abort();
      return;
    }

    workerRef.current?.postMessage({ type: 'abort' });

    // Safety watchdog: only recycle the worker if GPU queue failed to acknowledge abort within 2.5s
    if (wasWorkerGenerating && workerRef.current) {
      if (abortWatchdogRef.current) window.clearTimeout(abortWatchdogRef.current);
      const targetWorker = workerRef.current;
      abortWatchdogRef.current = window.setTimeout(() => {
        abortWatchdogRef.current = null;
        if (workerRef.current === targetWorker && activeMsgIdRef.current === null) {
          targetWorker.terminate();
          setEngineReady(false);
          setEngineLoading(false);
          autoBootTriggeredRef.current = false;
          spawnWorkerRef.current?.();
        }
      }, 2500);
    }
  }, [isGenerating]);

  const handleResetSCache = () => {
    handleStopGeneration();
    if (useMainThreadRef.current && mainThreadEngineRef.current) {
      mainThreadEngineRef.current.resetSCache();
    } else {
      workerRef.current?.postMessage({ type: 'reset_scache' });
    }
    setLastUserQuery('');
    lastUserQueryRef.current = '';
    setLiveRoutingDiag(null);
    setLiveActiveCapsuleIds([]);
    setLiveMaxGate(0);
    setSpokenLinks([]);
    setLatestTelemetry(null);
    setCaptionDismissed(true);
    setPointScarState((prev) => ({
      ...prev,
      midF32: null,
      finalF32: null,
      latencyMs: null,
      seq: 0,
    }));
    setLiveTokenState((prev) => ({
      ...prev,
      deltaToken: '',
      currentAgentText: '',
      generatedTokens: 0,
    }));
    setSessionAdapt(createInitialSessionAdaptation(currentLang));
  };

  const handleToggleUnlearn = (capsuleId: string) => {
    const targetCap = capsules.find((c) => c.capsule_id === capsuleId);
    const willWipe = !targetCap?.unlearnedWiped;

    if (willWipe) {
      setUnlearnShockActive(true);
      if (unlearnTimerRef.current) window.clearTimeout(unlearnTimerRef.current);
      unlearnTimerRef.current = window.setTimeout(() => {
        setUnlearnShockActive(false);
      }, 1800);
      setSessionAdapt((prev) => ({
        ...prev,
        unlearnWipeCount: prev.unlearnWipeCount + 1,
        lastUnlearnedCapsuleId: capsuleId,
      }));
    }

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

  // Pure deterministic JARVIS biomechanical state (`ZERO Math.random()`) wired strictly to Qwen
  const jarvisFace = useMemo(
    () =>
      computeJarvisFaceState({
        hasWebGPU,
        engineReady,
        engineLoading,
        loadPercent,
        isGenerating,
        liveInputText: inputMessage,
        lastUserQuery,
        currentAgentText: liveTokenState.currentAgentText,
        deltaToken: liveTokenState.deltaToken,
        generatedTokens: liveTokenState.generatedTokens,
        tokPerSec: liveTokenState.tokPerSec,
        confidence: liveTokenState.confidence,
        margin: liveTokenState.margin,
        horizonScale: liveTokenState.horizonScale,
        scacheBytes,
        scacheDeltaBytes: 0,
        activeCapsuleIds: liveActiveCapsuleIds,
        maxActiveGate: liveMaxGate,
        routingDiagnostics: liveRoutingDiag,
        session: sessionAdapt,
        unlearnShockActive,
        fallbackUiLang: currentLang,
      }),
    [
      hasWebGPU,
      engineReady,
      engineLoading,
      loadPercent,
      isGenerating,
      inputMessage,
      lastUserQuery,
      liveTokenState,
      scacheBytes,
      liveActiveCapsuleIds,
      liveMaxGate,
      liveRoutingDiag,
      sessionAdapt,
      unlearnShockActive,
      currentLang,
    ]
  );

  const langBadge = LANG_BADGES[jarvisFace.detectedLang] || LANG_BADGES.en;
  const rawPostureText =
    jarvisFace.postureLabel[currentLang === 'pt' ? 'pt' : jarvisFace.detectedLang] ||
    jarvisFace.postureLabel.en;
  const [displayedPostureText, setDisplayedPostureText] = useState<string>(rawPostureText);
  const lastPostureSwapMsRef = useRef<number>(0);

  useEffect(() => {
    if (!isGenerating || unlearnShockActive || engineLoading) {
      lastPostureSwapMsRef.current = performance.now();
      setDisplayedPostureText(rawPostureText);
      return;
    }
    const elapsed = performance.now() - lastPostureSwapMsRef.current;
    const minHoldMs = 1200;
    if (elapsed >= minHoldMs) {
      lastPostureSwapMsRef.current = performance.now();
      setDisplayedPostureText(rawPostureText);
      return;
    }
    const timer = window.setTimeout(() => {
      lastPostureSwapMsRef.current = performance.now();
      setDisplayedPostureText(rawPostureText);
    }, minHoldMs - elapsed);
    return () => window.clearTimeout(timer);
  }, [rawPostureText, isGenerating, unlearnShockActive, engineLoading]);

  const postureText = displayedPostureText;
  const hotCount = capsules.filter((c) => c.cachedInBrowser && !c.unlearnedWiped).length;
  const wipedCount = capsules.filter((c) => c.unlearnedWiped).length;

  const canStopNow = isGenerating || (engineLoading && hasPendingQuery);
  const hasCaptionContent =
    !captionDismissed &&
    (isGenerating ||
      (engineLoading && hasPendingQuery) ||
      liveTokenState.currentAgentText.trim().length > 0);

  const expandedFaceW = Math.round(
    Math.max(250, Math.min(390, viewportSize.w - 40, viewportSize.h * 0.46))
  );
  const faceWidth = isExpanded ? expandedFaceW : 214;
  const faceHeight = isExpanded ? Math.round(expandedFaceW * (218 / 214)) : 218;

  // Minimized Corner Orb State (if user explicitly tucks JARVIS away)
  if (isMinimized) {
    return (
      <div className="fixed bottom-5 right-5 z-50">
        <button
          type="button"
          data-testid="jarvis-minimized-orb"
          onClick={() => {
            setIsMinimized(false);
            setTimeout(() => inputRef.current?.focus(), 80);
          }}
          className="group relative flex items-center justify-center p-1.5 rounded-full bg-slate-950/75 hover:bg-slate-900/90 backdrop-blur-md border border-sky-400/55 shadow-[0_0_18px_rgba(56,189,248,0.28)] transition-all duration-200 cursor-pointer hover:scale-105"
          title={currentLang === 'pt' ? 'Restaurar rosto flutuante JARVIS (Qwen)' : 'Restore floating JARVIS face (Qwen)'}
        >
          <JarvisMiniFaceOrb face={jarvisFace} size={46} />
          <span
            className="absolute top-1 right-1 w-2.5 h-2.5 rounded-full border border-black"
            style={{ backgroundColor: jarvisFace.primaryHex }}
          />
        </button>
      </div>
    );
  }

  return (
    <>
      {/* FULL-VIEWPORT FLOATING 3D POINT-SCAR GENERATIVE STAGE (VQ4-Scar OT Rectified Flow) */}
      <JarvisFloatingPointStage
        enabled={pointFieldEnabled}
        face={jarvisFace}
        isGenerating={isGenerating}
        isExpanded={isExpanded}
        activeCapsuleIds={liveActiveCapsuleIds}
        unlearnShockActive={unlearnShockActive}
        pointScarMidF32={pointScarState.midF32}
        pointScarFinalF32={pointScarState.finalF32}
        pointScarMs={pointScarState.latencyMs}
        pointScarSeq={pointScarState.seq}
        lang={currentLang}
      />

      {/* HOLOGRAPHIC BACKDROP WHEN EXPANDED (click outside or press Esc to return to corner) */}
      {isExpanded && (
        <div
          data-testid="jarvis-expanded-backdrop"
          onClick={() => setIsExpanded(false)}
          className="fixed inset-0 z-40 bg-slate-950/78 backdrop-blur-xl transition-opacity duration-200"
          aria-hidden="true"
        />
      )}

      <div
        data-testid="jarvis-floating-widget"
        className={
          isExpanded
            ? 'fixed inset-0 z-50 flex flex-col items-center justify-center p-4 sm:p-6 gap-2.5 pointer-events-none'
            : 'fixed bottom-5 right-5 z-50 flex flex-col items-center gap-1.5 pointer-events-none max-w-[94vw]'
        }
      >
        {/* OPTIONAL FLOATING TELEMETRY & SCARSTORE HTTP 206 POPOVER (closed by default) */}
        {showInspector && (
          <div
            data-testid="jarvis-wires-popover"
            className={`pointer-events-auto overflow-y-auto rounded-2xl bg-slate-950/95 backdrop-blur-xl border border-sky-900/60 shadow-[0_18px_45px_rgba(0,0,0,0.92),0_0_24px_rgba(56,189,248,0.16)] p-3 font-mono text-[10px] text-zinc-200 space-y-2.5 ${
              isExpanded
                ? 'w-[min(92vw,380px)] max-h-[38vh] lg:fixed lg:right-7 lg:top-1/2 lg:-translate-y-1/2 lg:w-[376px] lg:max-h-[78vh]'
                : 'w-[326px] sm:w-[356px] max-h-[52vh]'
            }`}
          >
          <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/90">
            <span className="font-bold text-sky-300 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" />
              <span>Qwen 3B · 20-WGSL · ScarStore 206</span>
            </span>
            <button
              type="button"
              onClick={() => setShowInspector(false)}
              className="p-0.5 text-zinc-400 hover:text-white cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Live Biometric Wires (Zero Math.random Proof) */}
          <div className="grid grid-cols-3 gap-1.5 text-[9px]">
            <div className="p-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
              <div className="text-zinc-500">Qwen p₁ / Δp</div>
              <div className="font-bold text-white">
                {(jarvisFace.epistemicConfidence * 100).toFixed(0)}% / {jarvisFace.epistemicMargin.toFixed(2)}
              </div>
            </div>
            <div className="p-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
              <div className="text-zinc-500 flex items-center gap-0.5">
                <Eye className="w-2.5 h-2.5 text-sky-300" />
                <span>Pupil / Furrow</span>
              </div>
              <div className="font-bold text-sky-300">
                {(jarvisFace.pupilDilation * 100).toFixed(0)}% / {jarvisFace.browFurrow.toFixed(2)}
              </div>
            </div>
            <div className="p-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
              <div className="text-zinc-500">Viseme · RNG</div>
              <div className="font-bold text-cyan-300">{jarvisFace.viseme.code} · 0%</div>
            </div>
          </div>

          {/* PointScarFlow VQ4 Continuous 3D Point-Cloud Telemetry */}
          <div className="p-2 rounded-lg bg-slate-900/85 border border-sky-900/50 flex items-center justify-between text-[9px]">
            <span className="text-cyan-300 font-bold flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              <span>PointScarFlow VQ4 ({pointScarState.shardMiB.toFixed(2)} MiB)</span>
            </span>
            <span className="text-zinc-300">
              1024 pts · {pointScarState.steps}-OT
              {pointScarState.latencyMs !== null ? ` · ${pointScarState.latencyMs.toFixed(1)} ms` : ' · Ready'}
            </span>
          </div>

          {/* Layer-23 P_⊥lang Universal Cartridge Spectrum Bars */}
          <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-[9px]">
              <span className="text-sky-300 font-bold flex items-center gap-1">
                <Layers className="w-3 h-3" />
                <span>L23 P_⊥lang ({capsules.length || 40} × 140 KB)</span>
              </span>
              <span className="text-zinc-400">
                {hotCount} RAM · {wipedCount} 0x00
              </span>
            </div>
            <div className="flex items-end gap-[1.5px] h-3 pt-0.5">
              {(liveRoutingDiag?.anchorCosines?.length
                ? liveRoutingDiag.anchorCosines
                : new Array(capsules.length || 40).fill({ cosine: 0.14, gate: 0 })
              ).map((item, idx) => (
                <div
                  key={idx}
                  className="flex-1 rounded-xs transition-all duration-150"
                  style={{
                    height: `${Math.max(20, Math.min(100, Math.round(Math.max(0, item.cosine) * 100)))}%`,
                    backgroundColor: item.gate > 0.01 ? jarvisFace.primaryHex : '#1e293b',
                  }}
                />
              ))}
            </div>
          </div>

          {/* Compact Capsule List with Polyglot Test & LGPD 0x00 Wipe */}
          <div className="space-y-1.5 max-h-[210px] overflow-y-auto pr-0.5">
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
                  className={`p-2 rounded-lg border ${
                    isWiped
                      ? 'bg-rose-950/25 border-rose-800/50'
                      : isHot
                      ? 'bg-slate-900/95 border-sky-400/50'
                      : 'bg-slate-900/70 border-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="text-[10px] font-sans font-bold text-white truncate">{title}</span>
                    <button
                      type="button"
                      onClick={() => handleToggleUnlearn(cap.capsule_id)}
                      className={`px-1.5 py-0.5 rounded text-[8.5px] font-bold flex items-center gap-0.5 shrink-0 cursor-pointer border ${
                        isWiped
                          ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                          : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                      }`}
                      title="LGPD < 0.2 ms physical 0x00 wipe / restore"
                    >
                      {isWiped ? <RotateCcw className="w-2.5 h-2.5" /> : <Trash2 className="w-2.5 h-2.5" />}
                      <span>{isWiped ? 'Restore' : '0x00'}</span>
                    </button>
                  </div>
                  <div className="flex items-center gap-1 mt-1.5">
                    {(
                      [
                        ['pt', '🇧🇷'],
                        ['en', '🇺🇸'],
                        ['es', '🇪🇸'],
                        ['fr', '🇫🇷'],
                      ] as const
                    ).map(([code, flag]) => {
                      const promptStr = poly[code];
                      if (!promptStr) return null;
                      return (
                        <button
                          key={code}
                          type="button"
                          disabled={isGenerating}
                          onClick={() => {
                            setShowInspector(false);
                            void handleSendMessage(promptStr);
                          }}
                          className="px-1.5 py-0.5 rounded bg-slate-950 hover:bg-sky-400 text-zinc-300 hover:text-slate-950 border border-slate-700 text-[8.5px] font-bold transition-colors cursor-pointer disabled:opacity-45"
                          title={promptStr}
                        >
                          {flag} {code.toUpperCase()}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {httpEvents.length > 0 && (
            <div className="p-2 rounded-lg bg-black/80 border border-slate-800 space-y-0.5 text-[8.5px] text-zinc-400">
              <div className="text-sky-300 font-bold">📡 HTTP 206 Partial Content</div>
              {httpEvents.slice(0, 3).map((ev, idx) => (
                <div key={idx} className="flex justify-between gap-2 truncate">
                  <span className="truncate">
                    [{ev.status}] {ev.capsuleId}
                  </span>
                  <span className="text-cyan-300 shrink-0">{ev.kb} KB</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* EPHEMERAL HOLOGRAPHIC SPOKEN CAPTION (Replaces Chat Window — Only Visible When Speaking/Answering) */}
      {hasCaptionContent && (
        <div
          data-testid="jarvis-spoken-caption"
          className={`pointer-events-auto relative overflow-y-auto rounded-2xl bg-slate-950/92 backdrop-blur-xl border border-sky-900/70 shadow-[0_14px_38px_rgba(0,0,0,0.9),0_0_24px_rgba(56,189,248,0.16)] text-zinc-100 leading-relaxed transition-all ${
            isExpanded
              ? 'w-[min(92vw,580px)] max-h-[28vh] px-5 py-3.5 text-sm'
              : 'w-[306px] sm:w-[334px] max-h-[215px] px-3.5 py-2.5 text-xs'
          }`}
        >
          <button
            type="button"
            onClick={() => setCaptionDismissed(true)}
            className="float-right ml-2 p-0.5 text-zinc-500 hover:text-white cursor-pointer"
            title={currentLang === 'pt' ? 'Ocultar legenda' : 'Dismiss caption'}
          >
            <X className="w-3.5 h-3.5" />
          </button>

          <p
            className={`whitespace-pre-line font-sans text-zinc-100 ${
              isExpanded ? 'text-[13.5px] sm:text-[14px]' : 'text-[12px]'
            }`}
          >
            {liveTokenState.currentAgentText ||
              (engineLoading
                ? currentLang === 'pt'
                  ? `Sincronizando Qwen 3B na GPU (${loadPercent}%)...`
                  : `Syncing Qwen 3B on GPU (${loadPercent}%)...`
                : prefillStatus && prefillStatus.total > 0
                ? currentLang === 'pt'
                  ? `Roteando Camada 23 (${prefillStatus.done}/${prefillStatus.total} tokens) no Qwen 3B...`
                  : `Routing Layer 23 (${prefillStatus.done}/${prefillStatus.total} tokens) on Qwen 3B...`
                : currentLang === 'pt'
                ? 'Roteando Camada 23 (P_⊥lang) & decodificando no Qwen 3B...'
                : 'Routing Layer 23 (P_⊥lang) & decoding on Qwen 3B...')}
          </p>

          {spokenLinks.length > 0 && (
            <div className="mt-2 pt-1.5 border-t border-slate-800/80 flex flex-wrap gap-1.5">
              {spokenLinks.map((link, idx) => (
                <a
                  key={idx}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-900 border border-sky-400/40 text-[9.5px] font-mono font-bold text-sky-300 hover:bg-sky-400/15 transition-colors"
                >
                  {link.icon === 'colab' && <PlayCircle className="w-2.5 h-2.5" />}
                  {link.icon === 'github' && <Github className="w-2.5 h-2.5" />}
                  {(!link.icon || link.icon === 'ext') && <ExternalLink className="w-2.5 h-2.5" />}
                  <span>{link.label}</span>
                </a>
              ))}
            </div>
          )}

          {latestTelemetry && (
            <div className="mt-2 pt-1.5 border-t border-slate-800/70 flex items-center flex-wrap gap-1 text-[8.5px] font-mono text-zinc-400">
              <span className="text-sky-300 font-bold">
                SCache {formatScacheKb(latestTelemetry.scacheBytes)} (Δ +0 B)
              </span>
              {latestTelemetry.tokPerSec > 0 && (
                <span>· {latestTelemetry.tokPerSec.toFixed(1)} tok/s</span>
              )}
              {latestTelemetry.activeCapsules.length > 0 ? (
                latestTelemetry.activeCapsules.map((c) => (
                  <span key={c.capsuleId} className="text-cyan-200 font-bold">
                    · 🧩 {c.capsuleId.replace(/^portfolio_|^concept_|^ow_/, '')}
                  </span>
                ))
              ) : (
                <span className="text-amber-300/90 font-bold">
                  · 🛡️{' '}
                  {jarvisFace.detectedLang === 'pt'
                    ? '0 Cartuchos (Escopo 3B)'
                    : jarvisFace.detectedLang === 'es'
                    ? '0 Cartuchos (Alcance 3B)'
                    : jarvisFace.detectedLang === 'fr'
                    ? '0 Cartouche (Périmètre 3B)'
                    : '0 Cartridges (3B Scoped)'}
                </span>
              )}
              {latestTelemetry.margin > 0 && (
                <span>· Δp {latestTelemetry.margin.toFixed(2)}</span>
              )}
              {pointFieldEnabled && pointScarState.latencyMs !== null && (
                <span className="text-sky-300 font-bold">
                  · ✦ 1024 pts ({pointScarState.latencyMs.toFixed(0)} ms)
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* HONEST HARDWARE / ENGINE ERROR PILL (if any) */}
      {errorMessage && (
        <div
          className={`pointer-events-auto rounded-xl bg-rose-950/90 backdrop-blur-md border border-rose-700/80 px-3 py-2 text-[10px] font-mono text-rose-200 flex items-start justify-between gap-2 shadow-xl ${
            isExpanded ? 'w-[min(92vw,520px)]' : 'w-[306px] sm:w-[334px]'
          }`}
        >
          <div className="flex items-start gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-rose-300 hover:text-white cursor-pointer shrink-0"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* FRAMELESS FLOATING 3D NEURAL CONSTELLATION JARVIS FACE + CROWN MICRO-CONTROLS */}
      <div className="pointer-events-auto relative flex flex-col items-center">
        {/* Subtle Crown Status & Micro-Action Bar */}
        <div
          className={`flex items-center gap-1 -mb-1.5 z-10 rounded-full bg-slate-950/80 backdrop-blur-md border border-sky-900/70 shadow-[0_6px_18px_rgba(0,0,0,0.8)] font-mono ${
            isExpanded ? 'px-3 py-1 text-[10px]' : 'px-2.5 py-0.5 text-[9px]'
          }`}
        >
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{
              backgroundColor: jarvisFace.primaryHex,
              boxShadow: `0 0 8px ${jarvisFace.primaryHex}`,
            }}
          />
          <span
            className={`text-sky-100 font-bold truncate ${isExpanded ? 'max-w-[240px]' : 'max-w-[155px]'}`}
            title={`${adapterLabel} · ${postureText}`}
          >
            {langBadge.flag} {engineLoading ? `Qwen GPU ${loadPercent}%` : postureText}
          </span>

          <button
            type="button"
            data-testid="jarvis-pointscar-toggle"
            onClick={() => setPointFieldEnabled((v) => !v)}
            className={`p-1 rounded-full transition-colors cursor-pointer ${
              pointFieldEnabled ? 'text-sky-300' : 'text-zinc-500 hover:text-white'
            }`}
            title={
              currentLang === 'pt'
                ? 'Alternar pontos flutuantes 3D neurais (PointScarFlow VQ4)'
                : 'Toggle neural 3D floating points (PointScarFlow VQ4)'
            }
          >
            <Sparkles className="w-3 h-3" />
          </button>

          <button
            type="button"
            onClick={() => setVoiceEnabled((v) => !v)}
            className={`p-1 rounded-full transition-colors cursor-pointer ${
              voiceEnabled ? 'text-sky-300' : 'text-zinc-400 hover:text-white'
            }`}
            title={currentLang === 'pt' ? 'Alternar voz neural' : 'Toggle voice synthesis'}
          >
            {voiceEnabled ? <Volume2 className="w-3 h-3" /> : <VolumeX className="w-3 h-3" />}
          </button>

          <button
            type="button"
            data-testid="jarvis-wires-toggle"
            onClick={() => setShowInspector((v) => !v)}
            className={`p-1 rounded-full transition-colors cursor-pointer ${
              showInspector ? 'text-cyan-300' : 'text-zinc-400 hover:text-white'
            }`}
            title={
              currentLang === 'pt'
                ? 'Inspecionar fios matemáticos Qwen & ScarStore HTTP 206'
                : 'Inspect Qwen mathematical wires & ScarStore HTTP 206'
            }
          >
            <Activity className="w-3 h-3" />
          </button>

          <button
            type="button"
            onClick={handleResetSCache}
            className="p-1 rounded-full text-zinc-400 hover:text-white transition-colors cursor-pointer"
            title={currentLang === 'pt' ? 'Limpar SCache O(1)' : 'Reset SCache O(1)'}
          >
            <RefreshCw className="w-2.5 h-2.5" />
          </button>

          <button
            type="button"
            data-testid="jarvis-expand-toggle"
            onClick={() => {
              setIsExpanded((prev) => !prev);
              setTimeout(() => inputRef.current?.focus(), 60);
            }}
            className={`p-1 rounded-full transition-colors cursor-pointer ${
              isExpanded ? 'text-sky-300 hover:text-white' : 'text-zinc-400 hover:text-white'
            }`}
            title={
              isExpanded
                ? currentLang === 'pt'
                  ? 'Recolher JARVIS (Esc)'
                  : 'Collapse JARVIS (Esc)'
                : currentLang === 'pt'
                ? 'Expandir JARVIS'
                : 'Expand JARVIS'
            }
          >
            {isExpanded ? <Shrink className="w-3 h-3" /> : <Maximize2 className="w-2.5 h-2.5" />}
          </button>

          <button
            type="button"
            onClick={() => {
              setIsExpanded(false);
              setIsMinimized(true);
            }}
            className="p-1 rounded-full text-zinc-400 hover:text-white transition-colors cursor-pointer"
            title={currentLang === 'pt' ? 'Minimizar para orbe' : 'Minimize to orb'}
          >
            <Minimize2 className="w-2.5 h-2.5" />
          </button>
        </div>

        {/* Modular 3D Holographic Human Constellation Face (`JarvisFaceRig`) */}
        <JarvisFaceRig
          face={jarvisFace}
          isSpeaking={isGenerating}
          isBooting={engineLoading}
          bootPercent={loadPercent}
          width={faceWidth}
          height={faceHeight}
          onClick={() => inputRef.current?.focus()}
        />

        {/* Subtle GPU Wake / Boot Progress Pill right beneath the constellation collar */}
        {engineLoading ? (
          <div className="-mt-2 mb-0.5 px-2.5 py-0.5 rounded-full bg-slate-950/90 border border-sky-500/45 text-[9px] font-mono text-sky-300 shadow">
            ⚡ {loadPhaseDetail ? loadPhaseDetail.slice(0, 32) : 'Qwen WGSL'} · {loadPercent}%
            {loadLoadedMiB > 0 && !loadedFromCache ? ` (${loadLoadedMiB}/${loadTotalMiB} MiB)` : ''}
          </div>
        ) : (
          !engineReady &&
          hasWebGPU && (
            <button
              type="button"
              onClick={() => void handleBootEngine()}
              className="-mt-2 mb-0.5 px-2.5 py-0.5 rounded-full bg-slate-950/90 hover:bg-slate-900 border border-sky-400/50 text-[9px] font-mono font-bold text-sky-300 flex items-center gap-1 shadow-[0_0_14px_rgba(56,189,248,0.22)] cursor-pointer transition-colors"
              title={
                shardsCached
                  ? '0 MB network: load Qwen 3B from browser CacheStorage into GPU'
                  : 'Compile 20 WGSL shaders & load Qwen 3B (1,601.57 MiB ProtectedInt4Scar) into GPU'
              }
            >
              <Zap className="w-2.5 h-2.5" />
              <span>
                {shardsCached
                  ? currentLang === 'pt'
                    ? 'Acordar Qwen 3B GPU (0 MB rede)'
                    : 'Wake Qwen 3B GPU (0 MB net)'
                  : currentLang === 'pt'
                  ? 'Pré-carregar Qwen 3B GPU'
                  : 'Preload Qwen 3B GPU'}
              </span>
            </button>
          )
        )}
      </div>

      {/* SLEEK FLOATING TEXT INPUT FIELD DIRECTLY UNDERNEATH JARVIS'S CONSTELLATION COLLAR */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void handleSendMessage();
        }}
        className={`pointer-events-auto flex items-center gap-1.5 rounded-full bg-slate-950/92 backdrop-blur-xl border border-sky-400/60 focus-within:border-cyan-300 focus-within:shadow-[0_0_26px_rgba(56,189,248,0.35)] shadow-[0_12px_32px_rgba(0,0,0,0.9)] transition-all ${
          isExpanded
            ? 'w-[min(92vw,520px)] pl-5 pr-2 py-2'
            : 'w-[294px] sm:w-[328px] pl-4 pr-1.5 py-1.5'
        }`}
      >
        <input
          ref={inputRef}
          data-testid="jarvis-floating-input"
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={
            currentLang === 'pt'
              ? 'Fale com o JARVIS (Qwen 3B)...'
              : 'Speak to JARVIS (Qwen 3B)...'
          }
          className={`flex-1 min-w-0 bg-transparent text-white placeholder-slate-400 focus:outline-none font-sans ${
            isExpanded ? 'text-sm' : 'text-xs'
          }`}
        />

        {canStopNow ? (
          <button
            type="button"
            data-testid="jarvis-stop-button"
            onClick={handleStopGeneration}
            className={`${
              isExpanded ? 'w-8 h-8' : 'w-7 h-7'
            } rounded-full bg-rose-500 hover:bg-rose-400 text-white flex items-center justify-center transition-colors cursor-pointer shrink-0`}
            title={currentLang === 'pt' ? 'Parar geração agora' : 'Stop generation now'}
          >
            <Square className="w-3 h-3" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!inputMessage.trim()}
            className={`${
              isExpanded ? 'w-8 h-8' : 'w-7 h-7'
            } rounded-full bg-sky-400 hover:bg-cyan-300 disabled:bg-slate-800 disabled:text-slate-600 text-slate-950 flex items-center justify-center transition-colors cursor-pointer shrink-0 shadow-[0_0_12px_rgba(56,189,248,0.45)] disabled:shadow-none`}
            title={currentLang === 'pt' ? 'Enviar para o Qwen' : 'Send to Qwen'}
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        )}
      </form>
      </div>
    </>
  );
};
