import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  ExternalLink,
  Brain,
  Dices,
  Activity,
  Trophy,
  Zap,
} from 'lucide-react';
import {
  RubiksCubeSim,
  CartPoleSim,
  LunarLanderSim,
  MountainCarSim,
  AcrobotSim,
  System1AgentWeb,
  ensureOnnxRuntime,
  type DecisionResult,
} from '../lib/system1';
import { RubiksCube3D } from './RubiksCube3D';

interface ActionLogItem {
  id: number;
  name: string;
  confidence: number;
  uncertainty: number;
  latencyMs: number;
  isUncertain: boolean;
}

export type DemoKey = 'rubiks_atomic' | 'lunarlander' | 'mountaincar' | 'cartpole' | 'acrobot';

export const SystemOneHudDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang = 'en' }) => {
  const [activeDemo, setActiveDemo] = useState<DemoKey>('rubiks_atomic');
  const [cubeViewMode, setCubeViewMode] = useState<'3d' | '2d'>('3d');
  const [cubeState, setCubeState] = useState<{
    state: Int32Array;
    lastAction: string;
    steps: number;
    alignedCount: number;
    isSolved: boolean;
  }>({
    state: new Int32Array(54),
    lastAction: 'SCRAMBLED',
    steps: 0,
    alignedCount: 0,
    isSolved: false,
  });

  const [nonCubeSteps, setNonCubeSteps] = useState<number>(0);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [statusDesc, setStatusDesc] = useState<string>(
    lang === 'pt' ? 'Pronto para avaliar' : 'Ready to evaluate'
  );

  const [decision, setDecision] = useState<DecisionResult>({
    action: 0,
    confidence: 0,
    uncertainty: 0,
    calibration: 0.0,
    isUncertain: false,
    latencyMs: 0,
  });

  const [isAutoCalibrate, setIsAutoCalibrate] = useState<boolean>(false);
  const [calibrationValue, setCalibrationValue] = useState<number>(0.0);

  const [actionLog, setActionLog] = useState<ActionLogItem[]>([]);
  const [fps, setFps] = useState<number>(60);
  const [cubeScrambleDepth, setCubeScrambleDepth] = useState<number>(2);

  const [manifestData, setManifestData] = useState<Record<string, any>>({});

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simRef = useRef<RubiksCubeSim | CartPoleSim | LunarLanderSim | MountainCarSim | AcrobotSim | null>(null);
  const agentRef = useRef<System1AgentWeb | null>(null);
  const animRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number>(0);
  const lastUiUpdateTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const isRunningRef = useRef<boolean>(false);
  const lastRewardRef = useRef<number>(0.0);
  const prevScoreRef = useRef<number>(0.0);
  const actionHistoryRef = useRef<number[]>([]);
  const peakRawRef = useRef<number>(0);
  const visitedStatesRef = useRef<string[]>([]);

  const isRubiksDemo = (k: DemoKey) => k === 'rubiks_atomic';

  const getCubeScore = (c: any): number => {
    if (!c) return 0.0;
    if (typeof c.getScore === 'function') return c.getScore();
    if (typeof c.getRawAlignedCount === 'function') {
      return Math.max(0.0, Math.min(1.0, (c.getRawAlignedCount() - 6) / 48.0));
    }
    return 0.0;
  };

  const GITHUB_REPO = 'JulioCRFilho/system_one';

  useEffect(() => {
    const fetchManifest = async () => {
      try {
        const baseUrl = import.meta.env.BASE_URL?.endsWith('/')
          ? import.meta.env.BASE_URL
          : `${import.meta.env.BASE_URL || ''}/`;
        const res = await fetch(`${baseUrl}models/manifest.json?v=${Date.now()}`);
        if (res.ok) {
          const data = await res.json();
          if (data?.models) {
            setManifestData(data.models);
          }
        }
      } catch (_) {}
    };
    fetchManifest();
  }, []);

  const DEMO_CONFIGS: Record<DemoKey, any> = useMemo(
    () => ({
      rubiks_atomic: {
        id: 'rubiks_atomic',
        name: "Rubik's Cube",
        shortTitle: "Rubik's Cube",
        tag: lang === 'pt' ? '12 Giros Atômicos' : '12 Atomic Moves',
        icon: '🧩',
        file: 's1_rubiks_atomic.onnx',
        checkpointSource: 's1_rubikscube_v0_trained_v6.pt',
        checkpointVersion: 'v6 (450M+ passos, D10)',
        size: '~6.4 MB',
        envName: 'RubiksCube-v0',
        actionSpace: "12 Giros Atômicos (U, U', D, D', F, F', B, B', R, R', L, L')",
        obsSpace: '324-dim One-Hot Vector (54 stickers × 6 cores)',
        description:
          lang === 'pt'
            ? 'Agente neural AURA-v6 com LeanResidualTrunk 512D (1.6M parâmetros). Treinado em 450M+ passos com 16 professores SIL assimétricos e DPO contrastivo, alcançando profundidade D10.'
            : 'AURA-v6 neural agent with LeanResidualTrunk 512D (1.6M parameters). Trained across 450M+ steps with 16 asymmetric SIL teachers and contrastive DPO, unlocking Depth 10.',
      },
      lunarlander: {
        id: 'lunarlander',
        name: lang === 'pt' ? 'LunarLander-v3 (Pouso Espacial)' : 'LunarLander-v3 (Lunar Landing)',
        shortTitle: 'LunarLander-v3',
        tag: lang === 'pt' ? 'Pouso Lunar (4)' : 'Moon Landing (4)',
        icon: '🚀',
        file: 's1_lunarlander_v3.onnx',
        checkpointSource: 's1_lunarlander_v3_trained.pt',
        checkpointVersion: 'v1',
        size: '~2.8 MB',
        envName: 'LunarLander-v3',
        actionSpace: '4 Ações Discretas (Noop, Propulsor Esq, Principal, Dir)',
        obsSpace: '8-dim Vetor de Estado [x, y, ẋ, ẏ, θ, θ̇, legL, legR]',
        description:
          lang === 'pt'
            ? 'Módulo lunar controlando propulsores com física gravitacional, partículas de fogo e amortecimento de pouso sobre a plataforma.'
            : 'Lunar lander controlling main and side thrusters with gravity physics, exhaust particles, and smooth landing contact.',
      },
      mountaincar: {
        id: 'mountaincar',
        name: lang === 'pt' ? 'MountainCar-v0 (Controle de Inércia)' : 'MountainCar-v0 (Inertia Control)',
        shortTitle: 'MountainCar-v0',
        tag: lang === 'pt' ? 'Inércia / Colina (3)' : 'Inertia Climb (3)',
        icon: '🏎️',
        file: 's1_mountaincar_v0.onnx',
        checkpointSource: 's1_mountaincar_v0_trained.pt',
        checkpointVersion: 'v1',
        size: '~2.8 MB',
        envName: 'MountainCar-v0',
        actionSpace: '3 Ações Discretas (Push Left, Coast, Push Right)',
        obsSpace: '2-dim Cinemática [Posição, Velocidade]',
        description:
          lang === 'pt'
            ? 'Problema clássico de recompensa esparsa. O carro precisa acumular energia potencial voltando de ré para vencer a gravidade e chegar à bandeira.'
            : 'Sparse-reward classic task. The rover must build momentum by reversing up the opposite hill to climb and reach the goal flag.',
      },
      cartpole: {
        id: 'cartpole',
        name: lang === 'pt' ? 'CartPole-v1 (Controle Clássico)' : 'CartPole-v1 (Classic Control)',
        shortTitle: 'CartPole-v1',
        tag: lang === 'pt' ? 'Haste Invertida (2)' : 'Inverted Pole (2)',
        icon: '⚖️',
        file: 's1_cartpole.onnx',
        checkpointSource: 's1_cartpole_v1_trained.pt',
        checkpointVersion: 'v1',
        size: '~2.8 MB',
        envName: 'CartPole-v1',
        actionSpace: '2 Ações Discretas (Push Left / Right)',
        obsSpace: '4-dim Física Contínua [x, ẋ, θ, θ̇]',
        description:
          lang === 'pt'
            ? 'Equilíbrio dinâmico de haste invertida com física contínua. Latência de inferência ultrabaixa para controle crítico sem jitter.'
            : 'Dynamic inverted pendulum balancing with continuous physics. Ultra-low latency inference for real-time control without jitter.',
      },
      acrobot: {
        id: 'acrobot',
        name: lang === 'pt' ? 'Acrobot-v1 (Pêndulo Duplo)' : 'Acrobot-v1 (Double Pendulum)',
        shortTitle: 'Acrobot-v1',
        tag: lang === 'pt' ? 'Pêndulo Duplo (3)' : 'Double Pendulum (3)',
        icon: '🤸',
        file: 's1_acrobot_v1.onnx',
        checkpointSource: 's1_acrobot_v1_trained.pt',
        checkpointVersion: 'v1',
        size: '~2.8 MB',
        envName: 'Acrobot-v1',
        actionSpace: '3 Ações Discretas (Torque -1, 0, +1)',
        obsSpace: '6-dim Cinemática [cos, sin, cos, sin, θ̇₁, θ̇₂]',
        description:
          lang === 'pt'
            ? 'Robô acrobático de duas juntas com motor atuando apenas na articulação do meio. O agente aprende balanço harmônico para ultrapassar a meta.'
            : 'Two-link underactuated acrobat robot with torque only at the elbow joint. Learns harmonic resonance swing to reach the target line.',
      },
    }),
    [lang]
  );

  const RUBIKS_BENCHMARK_LEVELS = useMemo(() => {
    return [
      { depth: 1, name: 'D1', winRate: '100.0', rateColor: 'text-emerald-400', avgMoves: 1.0, states: '12', status: lang === 'pt' ? 'Ótimo Exato' : 'Exact Optimal' },
      { depth: 2, name: 'D2', winRate: '100.0', rateColor: 'text-emerald-400', avgMoves: 2.0, states: '114', status: lang === 'pt' ? 'Ótimo Exato' : 'Exact Optimal' },
      { depth: 3, name: 'D3', winRate: '100.0', rateColor: 'text-emerald-400', avgMoves: 2.9, states: '1.066', status: lang === 'pt' ? 'Ótimo Exato' : 'Exact Optimal' },
      { depth: 4, name: 'D4', winRate: '100.0', rateColor: 'text-emerald-400', avgMoves: 3.9, states: '9.324', status: lang === 'pt' ? 'Ótimo Exato' : 'Exact Optimal' },
      { depth: 5, name: 'D5', winRate: '88.0', rateColor: 'text-emerald-400', avgMoves: 7.2, states: '84.288', status: lang === 'pt' ? 'Quasi-Ótimo' : 'Near-Optimal' },
      { depth: 6, name: 'D6', winRate: '77.0', rateColor: 'text-cyan-400', avgMoves: 11.5, states: '757.512', status: lang === 'pt' ? 'Alta Precisão' : 'High Precision' },
      { depth: 7, name: 'D7', winRate: '60.0', rateColor: 'text-cyan-400', avgMoves: 18.7, states: '6.77M', status: lang === 'pt' ? 'Comutadores' : 'Commutators' },
      { depth: 8, name: 'D8', winRate: '36.0', rateColor: 'text-amber-400', avgMoves: 24.6, states: '61.3M', status: lang === 'pt' ? 'Teacher SIL' : 'Teacher SIL' },
      { depth: 9, name: 'D9', winRate: '17.0', rateColor: 'text-amber-400', avgMoves: 32.3, states: '550M', status: lang === 'pt' ? 'Fronteira D9' : 'D9 Frontier' },
      { depth: 10, name: 'D10', winRate: '12.0', rateColor: 'text-purple-400', avgMoves: 38.0, states: '232.2B', status: lang === 'pt' ? 'Fronteira 232B' : '232B Frontier' },
    ];
  }, [lang]);

  const activeLvlInfo = useMemo(() => {
    const found = RUBIKS_BENCHMARK_LEVELS.find((lvl) => lvl.depth === cubeScrambleDepth);
    return found || {
      depth: cubeScrambleDepth,
      name: `D${cubeScrambleDepth}`,
      winRate: cubeScrambleDepth <= 4 ? '100.0' : cubeScrambleDepth === 5 ? '88.0' : cubeScrambleDepth === 6 ? '77.0' : cubeScrambleDepth === 7 ? '60.0' : cubeScrambleDepth === 8 ? '36.0' : cubeScrambleDepth === 9 ? '17.0' : '12.0',
      rateColor: 'text-zinc-300',
      avgMoves: (cubeScrambleDepth * 3.8).toFixed(1),
      states: `${cubeScrambleDepth} giros`,
      status: lang === 'pt' ? 'Customizado' : 'Custom',
    };
  }, [RUBIKS_BENCHMARK_LEVELS, cubeScrambleDepth, lang]);

  useEffect(() => {
    isRunningRef.current = isRunning;
  }, [isRunning]);

  const initSim = (demoKey: DemoKey, depthOverride?: number) => {
    if (isRubiksDemo(demoKey)) {
      const cube = new RubiksCubeSim();
      const depth = depthOverride ?? (cubeScrambleDepth || 4);
      cube.scramble(depth, 'atomic');
      simRef.current = cube;
      setCubeState({
        state: new Int32Array(cube.state),
        lastAction: cube.lastAction,
        steps: cube.steps,
        alignedCount: cube.getAlignedCount(),
        isSolved: cube.isSolved(),
      });
    } else if (demoKey === 'lunarlander') {
      const lander = new LunarLanderSim();
      simRef.current = lander;
    } else if (demoKey === 'mountaincar') {
      const car = new MountainCarSim();
      simRef.current = car;
    } else if (demoKey === 'acrobot') {
      const acrobot = new AcrobotSim();
      simRef.current = acrobot;
    } else {
      const cp = new CartPoleSim();
      simRef.current = cp;
    }

    if (canvasRef.current && simRef.current) {
      simRef.current.render(canvasRef.current);
    }

    actionHistoryRef.current = [];
    lastRewardRef.current = 0.0;
    setNonCubeSteps(0);
    prevScoreRef.current = isRubiksDemo(demoKey) && simRef.current ? getCubeScore(simRef.current) : 0.0;
    peakRawRef.current = isRubiksDemo(demoKey) && simRef.current ? (simRef.current as RubiksCubeSim).getRawAlignedCount() : 0;
    visitedStatesRef.current = isRubiksDemo(demoKey) && simRef.current ? [(simRef.current as RubiksCubeSim).getStateHash()] : [];
    if (agentRef.current) {
      agentRef.current.resetMemory();
    }
  };

  useEffect(() => {
    if (isRubiksDemo(activeDemo) && canvasRef.current && simRef.current) {
      simRef.current.render(canvasRef.current);
    }
  }, [cubeViewMode, activeDemo]);

  const loadModel = async (demoKey: DemoKey) => {
    setIsLoading(true);
    setLoadError(null);
    setStatusDesc(lang === 'pt' ? 'Inicializando Wasm SIMD...' : 'Initializing Wasm SIMD...');

    try {
      const ort = await ensureOnnxRuntime();
      const cfg = DEMO_CONFIGS[demoKey];

      const baseUrl = import.meta.env.BASE_URL?.endsWith('/')
        ? import.meta.env.BASE_URL
        : `${import.meta.env.BASE_URL || ''}/`;
      const localUrl = `${baseUrl}models/${cfg.file}?v=${Date.now()}`;
      const cdnUrl = `https://cdn.jsdelivr.net/gh/${GITHUB_REPO}@master/web/models/${cfg.file}`;

      let session: any = null;
      try {
        session = await ort.InferenceSession.create(localUrl, {
          executionProviders: ['wasm'],
        });
      } catch (localErr) {
        console.warn('Falha ao carregar modelo local, tentando CDN do GitHub...', localErr);
        session = await ort.InferenceSession.create(cdnUrl, {
          executionProviders: ['wasm'],
        });
      }

      const preferredHx = manifestData[activeDemo]?.hx_dim || (activeDemo === 'rubiks_atomic' ? 512 : 256);
      agentRef.current = new System1AgentWeb(session, ort, preferredHx);
      setIsLoading(false);
      setStatusDesc(lang === 'pt' ? 'Pronto para avaliar' : 'Ready to evaluate');
    } catch (err: any) {
      console.error('Erro ao carregar modelo ONNX:', err);
      setIsLoading(false);
      setLoadError(err?.message || (lang === 'pt' ? 'Erro ao carregar pesos neurais' : 'Failed to load weights'));
      setStatusDesc(lang === 'pt' ? 'Falha no carregamento' : 'Load failed');
    }
  };

  const stopEvaluation = () => {
    setIsRunning(false);
    isRunningRef.current = false;
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  };

  useEffect(() => {
    stopEvaluation();
    initSim(activeDemo, cubeScrambleDepth);
    loadModel(activeDemo);

    return () => {
      stopEvaluation();
    };
  }, [activeDemo]);

  const stepLoop = async (timestamp: number) => {
    if (!isRunningRef.current) return;

    frameCountRef.current++;
    if (timestamp - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsTimeRef.current = timestamp;
    }

    const interval =
      isRubiksDemo(activeDemo)
        ? 250
        : activeDemo === 'lunarlander'
        ? 35
        : activeDemo === 'acrobot'
        ? 35
        : 30;

    if (timestamp - lastStepTimeRef.current >= interval) {
      lastStepTimeRef.current = timestamp;

      const sim = simRef.current;
      const agent = agentRef.current;
      const canvas = canvasRef.current;

      if (sim && agent) {
        try {
          let obs: Float32Array;
          if (isRubiksDemo(activeDemo)) {
            obs = (sim as RubiksCubeSim).getOneHot();
          } else if (activeDemo === 'lunarlander') {
            obs = (sim as LunarLanderSim).getState();
          } else if (activeDemo === 'mountaincar') {
            obs = (sim as MountainCarSim).getState();
          } else if (activeDemo === 'acrobot') {
            obs = (sim as AcrobotSim).getState();
          } else {
            obs = (sim as CartPoleSim).getState();
          }

          const avoidList: number[] = [];
          const aHist = actionHistoryRef.current;

          const isDet = !isAutoCalibrate && calibrationValue === 0.0;
          const dec = await agent.actWithConfidence(obs, lastRewardRef.current, {
            calibration: isAutoCalibrate ? 'auto' : calibrationValue,
            autoCalibrate: isAutoCalibrate,
            deterministic: isDet,
            valueRerank: isRubiksDemo(activeDemo),
            simulateStep: isRubiksDemo(activeDemo) ? (act) => (sim as RubiksCubeSim).simulateStep(act) : undefined,
          });
          actionHistoryRef.current.push(dec.action);
          if (actionHistoryRef.current.length > 20) actionHistoryRef.current.shift();

          let currentActionName = '';

          if (isRubiksDemo(activeDemo)) {
            const cube = sim as RubiksCubeSim;
            currentActionName = RubiksCubeSim.ATOMIC_MOVES[dec.action] || `MOVE_${dec.action}`;
            cube.applyAtomic(currentActionName, true);

            visitedStatesRef.current.push(cube.getStateHash());
            if (visitedStatesRef.current.length > 24) visitedStatesRef.current.shift();

            const curScore = getCubeScore(cube);
            const deltaScore = curScore - prevScoreRef.current;
            prevScoreRef.current = curScore;
            let stepReward = deltaScore * 5.0 - 0.02;
            if (cube.isSolved()) {
              stepReward += 10.0;
            }
            lastRewardRef.current = stepReward;

            setCubeState({
              state: new Int32Array(cube.state),
              lastAction: currentActionName,
              steps: cube.steps,
              alignedCount: cube.getAlignedCount(),
              isSolved: cube.isSolved(),
            });

            if (cube.isSolved()) {
              stopEvaluation();
              setStatusDesc(lang === 'pt' ? '🏆 Cubo Resolvido!' : '🏆 Solved!');
            } else {
              setStatusDesc(lang === 'pt' ? '⚡ Avaliando...' : '⚡ Evaluating...');
            }

            setDecision(dec);
            const actionEntryName = cube.isSolved()
              ? (lang === 'pt' ? '🏆 VITÓRIA (RESOLVIDO)' : '🏆 VICTORY (SOLVED)')
              : currentActionName;
            setActionLog((prev) => [
              {
                id: Date.now() + Math.random(),
                name: actionEntryName,
                confidence: cube.isSolved() ? 1.0 : dec.confidence,
                uncertainty: cube.isSolved() ? 0.0 : dec.uncertainty,
                latencyMs: dec.latencyMs,
                isUncertain: dec.isUncertain,
              },
              ...prev.slice(0, 4),
            ]);
          } else if (activeDemo === 'lunarlander') {
            const lander = sim as LunarLanderSim;
            const actionNames = ['ORBITANDO', 'PROP_ESQ ◀', 'PROP_PRINCIPAL ▲', 'PROP_DIR ▶'];
            currentActionName = actionNames[dec.action] || `ACT_${dec.action}`;
            const res = lander.step(dec.action);

            if (res.done) {
              lander.reset();
              agent.resetMemory();
            }

            const now = performance.now();
            if (now - lastUiUpdateTimeRef.current >= 100) {
              lastUiUpdateTimeRef.current = now;
              setDecision(dec);
              setNonCubeSteps(res.steps);
              setStatusDesc(
                lander.isLanded
                  ? (lang === 'pt' ? '🏆 Pousou com Sucesso!' : '🏆 Landed Safely!')
                  : (lang === 'pt' ? '⚡ Em voo...' : '⚡ In flight...')
              );

              setActionLog((prev) => [
                {
                  id: Date.now() + Math.random(),
                  name: currentActionName,
                  confidence: dec.confidence,
                  uncertainty: dec.uncertainty,
                  latencyMs: dec.latencyMs,
                  isUncertain: dec.isUncertain,
                },
                ...prev.slice(0, 4),
              ]);
            }
          } else if (activeDemo === 'mountaincar') {
            const car = sim as MountainCarSim;
            const actionNames = ['PUSH_LEFT ◀', 'COAST', 'PUSH_RIGHT ▶'];
            currentActionName = actionNames[dec.action] || `ACT_${dec.action}`;
            const res = car.step(dec.action);

            if (res.done) {
              car.reset();
              agent.resetMemory();
            }

            const now = performance.now();
            if (now - lastUiUpdateTimeRef.current >= 100) {
              lastUiUpdateTimeRef.current = now;
              setDecision(dec);
              setNonCubeSteps(res.steps);
              setStatusDesc(
                car.isGoalReached
                  ? (lang === 'pt' ? '🏆 Meta Alcançada!' : '🏆 Goal Reached!')
                  : (lang === 'pt' ? '⚡ Em aceleração...' : '⚡ Accelerating...')
              );

              setActionLog((prev) => [
                {
                  id: Date.now() + Math.random(),
                  name: currentActionName,
                  confidence: dec.confidence,
                  uncertainty: dec.uncertainty,
                  latencyMs: dec.latencyMs,
                  isUncertain: dec.isUncertain,
                },
                ...prev.slice(0, 4),
              ]);
            }
          } else if (activeDemo === 'acrobot') {
            const acrobot = sim as AcrobotSim;
            const actionNames = ['TORQUE_ESQ ◀', 'SEM_TORQUE', 'TORQUE_DIR ▶'];
            currentActionName = actionNames[dec.action] || `ACT_${dec.action}`;
            const res = acrobot.step(dec.action);

            if (res.done) {
              acrobot.reset();
              agent.resetMemory();
            }

            const now = performance.now();
            if (now - lastUiUpdateTimeRef.current >= 100) {
              lastUiUpdateTimeRef.current = now;
              setDecision(dec);
              setNonCubeSteps(res.steps);
              setStatusDesc(
                acrobot.isGoalReached
                  ? (lang === 'pt' ? '🏆 Swing Concluído!' : '🏆 Swing Completed!')
                  : (lang === 'pt' ? '⚡ Em balanço...' : '⚡ Swinging...')
              );

              setActionLog((prev) => [
                {
                  id: Date.now() + Math.random(),
                  name: currentActionName,
                  confidence: dec.confidence,
                  uncertainty: dec.uncertainty,
                  latencyMs: dec.latencyMs,
                  isUncertain: dec.isUncertain,
                },
                ...prev.slice(0, 4),
              ]);
            }
          } else {
            const cp = sim as CartPoleSim;
            currentActionName = dec.action === 1 ? 'PUSH_RIGHT' : 'PUSH_LEFT';
            const res = cp.step(dec.action);

            if (res.done) {
              cp.reset();
              agent.resetMemory();
            }

            const now = performance.now();
            if (now - lastUiUpdateTimeRef.current >= 100) {
              lastUiUpdateTimeRef.current = now;
              setDecision(dec);
              setNonCubeSteps(res.steps);
              setStatusDesc(lang === 'pt' ? '⚡ Equilibrando...' : '⚡ Balancing...');

              setActionLog((prev) => [
                {
                  id: Date.now() + Math.random(),
                  name: currentActionName,
                  confidence: dec.confidence,
                  uncertainty: dec.uncertainty,
                  latencyMs: dec.latencyMs,
                  isUncertain: dec.isUncertain,
                },
                ...prev.slice(0, 4),
              ]);
            }
          }

          if (canvas) {
            sim.render(canvas);
          }
        } catch (e) {
          console.error('Erro no passo neural:', e);
        }
      }
    }

    animRef.current = requestAnimationFrame(stepLoop);
  };

  const startEvaluation = () => {
    if (!agentRef.current || isLoading) return;
    if (isRubiksDemo(activeDemo) && simRef.current) {
      const cube = simRef.current as RubiksCubeSim;
      visitedStatesRef.current = [cube.getStateHash()];
      actionHistoryRef.current = [];
      lastRewardRef.current = 0.0;
      prevScoreRef.current = getCubeScore(cube);
      if (agentRef.current) agentRef.current.resetMemory();
    }
    setIsRunning(true);
    isRunningRef.current = true;
    animRef.current = requestAnimationFrame(stepLoop);
  };

  const toggleEvaluation = () => {
    if (isRunning) {
      stopEvaluation();
      setStatusDesc(lang === 'pt' ? '⏸️ Pausado' : '⏸️ Paused');
    } else {
      startEvaluation();
      setStatusDesc(lang === 'pt' ? '⚡ Avaliando...' : '⚡ Evaluating...');
    }
  };

  const scrambleEnv = (depthOverride?: number) => {
    if (isRubiksDemo(activeDemo) && simRef.current) {
      const cube = simRef.current as RubiksCubeSim;
      const depth = depthOverride ?? cubeScrambleDepth;
      const moves = cube.scramble(depth, 'atomic');
      if (canvasRef.current) {
        cube.render(canvasRef.current);
      }
      setCubeState({
        state: new Int32Array(cube.state),
        lastAction: cube.lastAction,
        steps: cube.steps,
        alignedCount: cube.getAlignedCount(),
        isSolved: cube.isSolved(),
      });
      actionHistoryRef.current = [];
      lastRewardRef.current = 0.0;
      prevScoreRef.current = getCubeScore(cube);
      visitedStatesRef.current = [cube.getStateHash()];
      if (agentRef.current) agentRef.current.resetMemory();
      setStatusDesc(
        lang === 'pt'
          ? `🎲 Embaralhado [D${depth}]`
          : `🎲 Scrambled [D${depth}]`
      );
    }
  };

  const resetEnv = () => {
    if (simRef.current) {
      simRef.current.reset();
      setNonCubeSteps(0);
      if (canvasRef.current) {
        simRef.current.render(canvasRef.current);
      }
      if (isRubiksDemo(activeDemo)) {
        const cube = simRef.current as RubiksCubeSim;
        setCubeState({
          state: new Int32Array(cube.state),
          lastAction: 'RESET',
          steps: 0,
          alignedCount: cube.getAlignedCount(),
          isSolved: cube.isSolved(),
        });
      }
      if (agentRef.current) agentRef.current.resetMemory();
      setActionLog([]);
      setStatusDesc(lang === 'pt' ? 'Pronto para avaliar' : 'Ready to evaluate');
    }
  };

  const currentCfg = DEMO_CONFIGS[activeDemo];

  const displaySteps = isRubiksDemo(activeDemo) ? cubeState.steps : nonCubeSteps;

  const displayAction = isRubiksDemo(activeDemo)
    ? (cubeState.lastAction === 'SCRAMBLED' ? (lang === 'pt' ? 'INÍCIO' : 'START') : (cubeState.lastAction === 'RESET' ? 'RESET' : cubeState.lastAction))
    : (actionLog[0]?.name || '--');

  const card1Data = useMemo(() => {
    if (isRubiksDemo(activeDemo)) {
      const pct = Math.min(100, Math.max(0, (cubeState.alignedCount / 54) * 100));
      return {
        label: lang === 'pt' ? 'ALINHAMENTO' : 'ALIGNMENT',
        val: `${cubeState.alignedCount}`,
        max: '/54',
        badge: `${pct.toFixed(0)}%`,
        color: cubeState.isSolved ? 'text-emerald-400' : 'text-cyan-400',
        sub: cubeState.isSolved
          ? (lang === 'pt' ? '🏆 100% Resolvido' : '🏆 100% Solved')
          : (lang === 'pt' ? `${54 - cubeState.alignedCount} facetas restantes` : `${54 - cubeState.alignedCount} stickers left`),
      };
    }
    if (activeDemo === 'lunarlander') {
      const lander = simRef.current as LunarLanderSim | null;
      const fuel = lander?.fuel ?? 100;
      return {
        label: lang === 'pt' ? 'COMBUSTÍVEL' : 'FUEL',
        val: `${fuel.toFixed(0)}`,
        max: '%',
        badge: lander?.isLanded ? (lang === 'pt' ? 'POUSADO' : 'LANDED') : (lang === 'pt' ? 'EM VOO' : 'IN FLIGHT'),
        color: lander?.isLanded ? 'text-emerald-400' : 'text-cyan-400',
        sub: lander?.isLanded
          ? (lang === 'pt' ? '🏆 Pouso seguro' : '🏆 Safe landing')
          : (lang === 'pt' ? 'Propulsão ativa' : 'Thrusters active'),
      };
    }
    if (activeDemo === 'mountaincar') {
      const car = simRef.current as MountainCarSim | null;
      const pos = car?.position ?? -0.5;
      const pct = Math.max(0, Math.min(100, ((pos + 1.2) / 1.7) * 100));
      return {
        label: 'MOMENTUM',
        val: `${pct.toFixed(0)}`,
        max: '%',
        badge: car?.isGoalReached ? (lang === 'pt' ? 'META' : 'GOAL') : (lang === 'pt' ? 'INÉRCIA' : 'INERTIA'),
        color: car?.isGoalReached ? 'text-emerald-400' : 'text-amber-400',
        sub: car?.isGoalReached
          ? (lang === 'pt' ? '🏆 Topo alcançado' : '🏆 Summit reached')
          : (lang === 'pt' ? 'Balanço cinético' : 'Kinetic swing'),
      };
    }
    if (activeDemo === 'acrobot') {
      const acrobot = simRef.current as AcrobotSim | null;
      return {
        label: lang === 'pt' ? 'ALTURA' : 'HEIGHT',
        val: `${acrobot?.isGoalReached ? '100' : displaySteps}`,
        max: acrobot?.isGoalReached ? '%' : (lang === 'pt' ? ' movs' : ' steps'),
        badge: acrobot?.isGoalReached ? (lang === 'pt' ? 'META' : 'GOAL') : 'SWING',
        color: acrobot?.isGoalReached ? 'text-emerald-400' : 'text-purple-400',
        sub: acrobot?.isGoalReached
          ? (lang === 'pt' ? '🏆 Alvo superado' : '🏆 Goal cleared')
          : (lang === 'pt' ? 'Ressonância dupla' : 'Double resonance'),
      };
    }
    // cartpole
    return {
      label: lang === 'pt' ? 'EQUILÍBRIO' : 'BALANCE',
      val: `${displaySteps}`,
      max: (lang === 'pt' ? ' movs' : ' steps'),
      badge: displaySteps > 200 ? (lang === 'pt' ? 'EXCELENTE' : 'EXCELLENT') : (lang === 'pt' ? 'ATIVO' : 'ACTIVE'),
      color: 'text-emerald-400',
      sub: lang === 'pt' ? 'Haste estabilizada' : 'Stabilized pole',
    };
  }, [activeDemo, cubeState, displaySteps, lang]);

  return (
    <div className="space-y-6">
      {/* Top Header & Identity */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-5 rounded-2xl bg-zinc-950/80 border border-zinc-800 shadow-xl backdrop-blur-md">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-amber-500 flex items-center justify-center font-bold text-white shadow-lg">
              ⚡
            </span>
            <h3 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
              AURA — System 1 Engine
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono font-bold tracking-wider uppercase">
                {lang === 'pt' ? 'LIVE EVALUATION WEB' : 'LIVE WEB EVALUATION'}
              </span>
            </h3>
          </div>
          <p className="text-xs text-zinc-400 max-w-2xl leading-relaxed">
            {lang === 'pt'
              ? 'AURA (Autonomous Universal Reflex Architecture) — Agente Neural Recorrente Amortizado (GRU Lean + Tri-Compass) executando inferência reflexiva sub-milissegundo em tempo real direto no seu navegador com ONNX Runtime Web (Wasm SIMD).'
              : 'AURA (Autonomous Universal Reflex Architecture) — Amortized Recurrent Neural Agent (Lean GRU + Tri-Compass) performing sub-millisecond reflex inference in real-time inside your browser using ONNX Runtime Web (Wasm SIMD).'}
          </p>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          <a
            href={`https://github.com/${GITHUB_REPO}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3.5 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-xs font-mono text-zinc-300 hover:text-white transition flex items-center gap-1.5 shadow-sm"
          >
            <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
            <span>GitHub ({GITHUB_REPO})</span>
          </a>
        </div>
      </div>

      {/* Main Grid: Left Viewport (7 Cols) | Right Specs & Architecture (5 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* ================= COLUNA ESQUERDA: VIEWPORT & TELEMETRIA (7 Cols) ================= */}
        <div className="lg:col-span-7 space-y-4">

          {/* Viewport Card */}
          <div className="p-4 rounded-2xl bg-zinc-950/80 border border-zinc-800 shadow-2xl flex flex-col gap-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
                  }`}
                />
                <span className="text-xs font-mono font-bold text-zinc-200 flex items-center gap-1.5">
                  <span>{currentCfg.icon}</span>
                  <span>{currentCfg.name}</span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                {isRubiksDemo(activeDemo) && (
                  <div className="flex items-center gap-0.5 bg-zinc-900 p-0.5 rounded-lg border border-zinc-800">
                    <button
                      type="button"
                      onClick={() => setCubeViewMode('3d')}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition cursor-pointer ${
                        cubeViewMode === '3d'
                          ? 'bg-cyan-500 text-black shadow'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                      title={lang === 'pt' ? 'Visualização 3D interativa (WebGL)' : 'Interactive 3D view (WebGL)'}
                    >
                      3D
                    </button>
                    <button
                      type="button"
                      onClick={() => setCubeViewMode('2d')}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition cursor-pointer ${
                        cubeViewMode === '2d'
                          ? 'bg-cyan-500 text-black shadow'
                          : 'text-zinc-400 hover:text-white'
                      }`}
                      title={lang === 'pt' ? 'Visualização 2D desdobrada em cruz' : 'Unfolded 2D net view'}
                    >
                      2D Net
                    </button>
                  </div>
                )}
                <span className="text-[10px] font-mono tabular-nums text-cyan-400 bg-cyan-950/70 px-2 py-0.5 rounded border border-cyan-800 min-w-[54px] text-center inline-block">
                  {fps} FPS
                </span>
                <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                  {currentCfg.size}
                </span>
              </div>
            </div>

            {/* Viewport 3D Rubik's Cube (WebGL) */}
            {isRubiksDemo(activeDemo) && (
              <div
                className={`relative rounded-xl overflow-hidden min-h-[320px] ${
                  cubeViewMode === '3d' ? 'block' : 'hidden'
                }`}
              >
                <RubiksCube3D
                  state={cubeState.state}
                  lastAction={cubeState.lastAction}
                  steps={cubeState.steps}
                  alignedCount={cubeState.alignedCount}
                  isSolved={cubeState.isSolved}
                  isRunning={isRunning}
                  lang={lang}
                  scrambleSequence={
                    simRef.current && isRubiksDemo(activeDemo)
                      ? (simRef.current as RubiksCubeSim).scrambleSequence?.join(' ')
                      : undefined
                  }
                />

                {/* Loader Overlay */}
                {isLoading && (
                  <div className="absolute inset-0 bg-[#07090e]/85 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-xs font-mono text-cyan-400 z-10 rounded-xl">
                    <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                    <span>
                      {lang === 'pt'
                        ? 'Carregando pesos neurais ONNX (Wasm SIMD)...'
                        : 'Loading ONNX neural weights (Wasm SIMD)...'}
                    </span>
                    <span className="text-[10px] text-zinc-500">{currentCfg.file}</span>
                  </div>
                )}

                {/* Error Overlay */}
                {loadError && (
                  <div className="absolute inset-0 bg-[#07090e]/90 flex flex-col items-center justify-center gap-2 p-4 text-center z-10 rounded-xl">
                    <span className="text-red-400 text-xs font-mono">⚠️ {loadError}</span>
                    <button
                      onClick={() => loadModel(activeDemo)}
                      className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-xs font-mono rounded text-zinc-200 border border-zinc-700 transition"
                    >
                      {lang === 'pt' ? 'Tentar novamente' : 'Retry'}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Viewport 2D Canvas (desdobramento em cruz sincronizado ou simulação física clássica) */}
            <div
              className={`relative bg-[#07090e] rounded-xl overflow-hidden border border-zinc-800/80 flex items-center justify-center min-h-[320px] ${
                !isRubiksDemo(activeDemo) || cubeViewMode === '2d' ? 'block' : 'hidden'
              }`}
            >
              <canvas
                ref={canvasRef}
                width={480}
                height={320}
                className="w-full h-auto max-h-[320px] object-contain block"
              />

              {/* Loader Overlay */}
              {isLoading && (
                <div className="absolute inset-0 bg-[#07090e]/85 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-xs font-mono text-cyan-400 z-10">
                  <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <span>
                    {lang === 'pt'
                      ? 'Carregando pesos neurais ONNX (Wasm SIMD)...'
                      : 'Loading ONNX neural weights (Wasm SIMD)...'}
                  </span>
                  <span className="text-[10px] text-zinc-500">{currentCfg.file}</span>
                </div>
              )}

              {/* Error Overlay */}
              {loadError && (
                <div className="absolute inset-0 bg-[#07090e]/90 flex flex-col items-center justify-center gap-2 p-4 text-center z-10">
                  <span className="text-red-400 text-xs font-mono">⚠️ {loadError}</span>
                  <button
                    onClick={() => loadModel(activeDemo)}
                    className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-xs font-mono rounded text-zinc-200 border border-zinc-700 transition"
                  >
                    {lang === 'pt' ? 'Tentar novamente' : 'Retry'}
                  </button>
                </div>
              )}
            </div>

            {/* Interactive Control Buttons */}
            <div className="flex items-center justify-between pt-1 gap-2 flex-wrap">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={toggleEvaluation}
                  disabled={isLoading || !!loadError}
                  className={`px-3.5 py-1.5 text-xs font-bold font-mono rounded-lg transition flex items-center gap-1.5 shadow cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
                    isRunning
                      ? 'bg-amber-600 hover:bg-amber-500 text-white'
                      : 'bg-cyan-600 hover:bg-cyan-500 text-white'
                  }`}
                >
                  {isRunning ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  <span>{isRunning ? (lang === 'pt' ? 'Pausar' : 'Pause') : (lang === 'pt' ? 'Iniciar' : 'Start')}</span>
                </button>

                {isRubiksDemo(activeDemo) && (
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => scrambleEnv()}
                      disabled={isLoading}
                      className="px-3 py-1.5 text-xs font-medium font-mono rounded-lg bg-zinc-900 hover:bg-zinc-800 text-amber-300 border border-zinc-700 transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                      title={lang === 'pt' ? 'Embaralha o cubo com rotações ortogonais reais' : 'Scramble cube with real orthogonal rotations'}
                    >
                      <Dices className="w-3.5 h-3.5" />
                      <span>{lang === 'pt' ? 'Embaralhar' : 'Scramble'}</span>
                    </button>

                    <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-700 rounded-lg px-2 py-1 shadow-sm">
                      <span className="text-[10px] text-zinc-400 font-mono select-none">
                        {lang === 'pt' ? 'Passos:' : 'Steps:'}
                      </span>
                      <input
                        type="number"
                        min={1}
                        value={cubeScrambleDepth}
                        onChange={(e) => {
                          const val = parseInt(e.target.value, 10);
                          const newDepth = Math.max(1, isNaN(val) ? 1 : val);
                          setCubeScrambleDepth(newDepth);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            scrambleEnv(cubeScrambleDepth);
                          }
                        }}
                        disabled={isLoading}
                        className="w-12 bg-transparent text-xs font-mono text-center text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500 rounded py-0.5"
                        title={lang === 'pt' ? 'Número aberto de passos de embaralhamento sem limite' : 'Open scramble step count without upper limit'}
                      />
                    </div>
                  </div>
                )}

                <button
                  type="button"
                  onClick={resetEnv}
                  disabled={isLoading}
                  className="px-3 py-1.5 text-xs font-medium font-mono rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-700 transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              </div>

              <span className="text-[11px] font-mono text-zinc-400 bg-zinc-900/80 px-2.5 py-1 rounded border border-zinc-800 truncate min-w-[130px] text-right inline-block">
                {statusDesc}
              </span>
            </div>
          </div>

          {/* Níveis Conquistados & Taxas de Vitória (Curriculum D1–D10) */}
          {isRubiksDemo(activeDemo) && (
            <div className="p-3 sm:p-3.5 rounded-2xl bg-zinc-950/90 border border-zinc-800 shadow-xl space-y-2.5 font-mono">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-zinc-200 flex items-center gap-1.5">
                  <Trophy className="w-3.5 h-3.5 text-amber-400" />
                  <span>
                    {lang === 'pt'
                      ? 'TAXA DE VITÓRIA POR NÍVEL CONQUISTADO (AURA D1–D10)'
                      : 'CONQUERED LEVEL WIN RATE BENCHMARK (AURA D1–D10)'}
                  </span>
                </span>
                <span className="text-[10px] text-zinc-500">
                  {lang === 'pt' ? 'Clique em um nível para testar' : 'Click a level to test'}
                </span>
              </div>

              {/* Grid Interativa de Níveis com Win Rate */}
              <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5">
                {RUBIKS_BENCHMARK_LEVELS.map((lvl) => {
                  const isCurrent = cubeScrambleDepth === lvl.depth;
                  return (
                    <button
                      key={lvl.depth}
                      type="button"
                      onClick={() => {
                        stopEvaluation();
                        setCubeScrambleDepth(lvl.depth);
                        scrambleEnv(lvl.depth);
                      }}
                      className={`px-1 py-1.5 rounded-lg border text-center transition cursor-pointer flex flex-col items-center justify-between min-h-[52px] ${
                        isCurrent
                          ? 'bg-amber-950/70 border-amber-400 text-amber-200 shadow-lg shadow-amber-950/60 ring-1 ring-amber-400/60'
                          : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700 hover:bg-zinc-900'
                      }`}
                      title={`${lvl.name}: ${lvl.winRate}% ${lang === 'pt' ? 'de vitória' : 'win rate'} (${lvl.states} ${lang === 'pt' ? 'estados' : 'states'}, ~${lvl.avgMoves} ${lang === 'pt' ? 'movs' : 'moves'})`}
                    >
                      <span className="text-[11px] font-bold text-white tracking-tight">{lvl.name}</span>
                      <span className={`text-[10px] font-bold tabular-nums ${lvl.rateColor}`}>
                        {lvl.winRate}%
                      </span>
                      <span className="text-[8px] text-zinc-500">{lvl.avgMoves}m</span>
                    </button>
                  );
                })}
              </div>

              {/* Detalhe Dinâmico do Nível Selecionado */}
              <div className="flex items-center justify-between text-[11px] text-zinc-400 bg-zinc-900/80 px-3 py-2 rounded-lg border border-zinc-800">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-white font-bold flex items-center gap-1">
                    <Zap className="w-3 h-3 text-amber-400 fill-current" />
                    <span>{activeLvlInfo.name}:</span>
                  </span>
                  <span className="text-emerald-400 font-bold">
                    {activeLvlInfo.winRate}% {lang === 'pt' ? 'Taxa de Vitória' : 'Win Rate'}
                  </span>
                  <span className="text-zinc-600">•</span>
                  <span>
                    ~{activeLvlInfo.avgMoves} {lang === 'pt' ? 'movs médios' : 'avg moves'}
                  </span>
                  <span className="text-zinc-600">•</span>
                  <span>
                    {lang === 'pt' ? 'Espaço de Estados' : 'State Space'}: <strong className="text-cyan-300 font-mono">{activeLvlInfo.states}</strong>
                  </span>
                </div>
                <span className="text-[10px] text-amber-400/90 font-bold uppercase tracking-wider hidden md:inline bg-amber-950/40 px-2 py-0.5 rounded border border-amber-900/50">
                  {activeLvlInfo.status}
                </span>
              </div>
            </div>
          )}

          {/* Status & Telemetry Cards (4-Grid Não-Redundante) */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
            {/* Card 1: Progresso / Alinhamento da Tarefa */}
            <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 shadow-md flex flex-col justify-between h-[116px] min-h-[116px] overflow-hidden select-none">
              <div>
                <div className="h-5 flex items-center justify-between">
                  <span className="text-[10px] text-zinc-400 font-mono uppercase tracking-wider whitespace-nowrap truncate">
                    {card1Data.label}
                  </span>
                  <span className={`text-[9px] font-mono font-bold ${card1Data.color}`}>
                    {card1Data.badge}
                  </span>
                </div>
                <div className={`text-xl font-bold font-mono tabular-nums ${card1Data.color} mt-1 flex items-baseline gap-0.5`}>
                  <span>{card1Data.val}</span>
                  <span className="text-xs text-zinc-500 font-normal">{card1Data.max}</span>
                </div>
              </div>
              <span className="text-[9px] text-zinc-500 font-mono mt-1 whitespace-nowrap truncate block">
                {card1Data.sub}
              </span>
            </div>

            {/* Card 2: Passos do Episódio */}
            <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 shadow-md flex flex-col justify-between h-[116px] min-h-[116px] overflow-hidden select-none">
              <div>
                <div className="h-5 flex items-center justify-between">
                  <span className="text-[10px] text-zinc-400 font-mono uppercase tracking-wider whitespace-nowrap truncate">
                    {lang === 'pt' ? 'PASSOS DO EPISÓDIO' : 'EPISODE STEPS'}
                  </span>
                  <span className="text-[9px] font-mono text-zinc-500">
                    {isRubiksDemo(activeDemo) ? `D${cubeScrambleDepth}` : 'EP'}
                  </span>
                </div>
                <div className="text-xl font-bold font-mono tabular-nums text-amber-300 mt-1 flex items-baseline gap-1">
                  <span className="w-14 inline-block">{displaySteps}</span>
                  <span className="text-xs text-amber-500 font-normal">{lang === 'pt' ? 'movs' : 'steps'}</span>
                </div>
              </div>
              <span className="text-[9px] text-zinc-500 font-mono mt-1 whitespace-nowrap truncate block">
                {isRubiksDemo(activeDemo)
                  ? (lang === 'pt' ? `Alvo: D${cubeScrambleDepth} (~${activeLvlInfo.avgMoves} médios)` : `Target: D${cubeScrambleDepth} (~${activeLvlInfo.avgMoves} avg)`)
                  : (lang === 'pt' ? 'Horizonte reflexivo contínuo' : 'Continuous reflex horizon')}
              </span>
            </div>

            {/* Card 3: Última Ação & Confiança */}
            <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 shadow-md flex flex-col justify-between h-[116px] min-h-[116px] overflow-hidden select-none">
              <div>
                <div className="h-5 flex items-center justify-between gap-1">
                  <span className="text-[10px] text-zinc-400 font-mono uppercase tracking-wider whitespace-nowrap truncate">
                    {lang === 'pt' ? 'ÚLTIMA AÇÃO' : 'LAST ACTION'}
                  </span>
                  <span
                    className={`text-[8px] font-mono px-1 py-0.2 rounded border font-bold shrink-0 ${
                      decision.isUncertain
                        ? 'bg-amber-950/80 text-amber-300 border-amber-800'
                        : 'bg-cyan-950/80 text-cyan-300 border-cyan-800'
                    }`}
                  >
                    {decision.isUncertain ? 'S2 TRIGGER' : 'O(1) REFLEX'}
                  </span>
                </div>
                <div className="text-xl font-bold font-mono tabular-nums text-white mt-1 flex items-baseline justify-between">
                  <span className="text-cyan-300 font-bold truncate max-w-[90px]">{displayAction}</span>
                  <span className="text-xs text-emerald-400 font-mono font-bold">
                    {decision.confidence > 0 ? (decision.confidence * 100).toFixed(0) : '--'}%
                  </span>
                </div>
              </div>
              <span className="text-[9px] text-zinc-500 font-mono mt-1 whitespace-nowrap truncate block">
                {decision.confidence > 0
                  ? (lang === 'pt' ? `Certeza: ${(decision.confidence * 100).toFixed(1)}%` : `Certainty: ${(decision.confidence * 100).toFixed(1)}%`)
                  : (lang === 'pt' ? 'Aguardando decisão' : 'Awaiting decision')}
              </span>
            </div>

            {/* Card 4: Latência de Inferência */}
            <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 shadow-md flex flex-col justify-between h-[116px] min-h-[116px] overflow-hidden select-none">
              <div>
                <div className="h-5 flex items-center justify-between">
                  <span className="text-[10px] text-zinc-400 font-mono uppercase tracking-wider whitespace-nowrap truncate">
                    {lang === 'pt' ? 'LATÊNCIA' : 'LATENCY'}
                  </span>
                  <span className="text-[9px] font-mono text-cyan-400">Wasm SIMD</span>
                </div>
                <div className="text-xl font-bold font-mono tabular-nums text-cyan-400 mt-1 flex items-baseline gap-1">
                  <span className="w-16 inline-block">{decision.latencyMs > 0 ? decision.latencyMs.toFixed(2) : '0.18'}</span>
                  <span className="text-xs text-cyan-500 font-normal">ms</span>
                </div>
              </div>
              <span className="text-[9px] text-zinc-500 font-mono mt-1 whitespace-nowrap truncate block">
                {lang === 'pt' ? 'Sub-milissegundo CPU' : 'Sub-millisecond CPU'}
              </span>
            </div>
          </div>

        </div>

        {/* ================= COLUNA DIREITA: CONFIGURAÇÃO, LOG & SPECS (5 Cols) ================= */}
        <div className="lg:col-span-5 space-y-4">

          {/* Seletor de Demonstração & Checkpoint */}
          <div className="p-4 rounded-2xl bg-zinc-950/80 border border-zinc-800 shadow-xl space-y-3">
            <h4 className="text-xs font-bold text-zinc-200 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <span>⚙️</span>
              <span>{lang === 'pt' ? 'SELECIONAR AMBIENTE' : 'SELECT ENVIRONMENT'}</span>
            </h4>

            <div className="grid grid-cols-2 gap-2">
              {(['rubiks_atomic', 'lunarlander', 'mountaincar', 'cartpole', 'acrobot'] as DemoKey[]).map((key) => {
                const cfg = DEMO_CONFIGS[key];
                const isSelected = activeDemo === key;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setActiveDemo(key)}
                    className={`p-2.5 rounded-xl border text-left text-xs font-mono transition cursor-pointer flex flex-col gap-1 relative overflow-hidden ${
                      isSelected
                        ? 'bg-cyan-950/50 border-cyan-400 text-white shadow-lg shadow-cyan-950/50 ring-1 ring-cyan-500/50'
                        : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700 hover:bg-zinc-900/90'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="text-base">{cfg.icon}</span>
                      {isSelected && (
                        <span className="flex h-2 w-2 relative">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400"></span>
                        </span>
                      )}
                    </div>
                    <span className="font-bold truncate text-[12px]">{cfg.shortTitle}</span>
                    <span className="text-[10px] text-zinc-500 truncate">{cfg.tag}</span>
                  </button>
                );
              })}
            </div>

            {/* Descrição do Ambiente Ativo */}
            <p className="text-[11px] text-zinc-400 leading-relaxed bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-850">
              {currentCfg.description}
            </p>

            {/* Calibração Estocástica & Homeostase */}
            <div className="pt-2 border-t border-zinc-800 space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-zinc-300 font-semibold flex items-center gap-1.5">
                  <span>⚡</span>
                  <span>{lang === 'pt' ? 'Calibração Estocástica' : 'Stochastic Calibration'}</span>
                </span>
                <span className="text-[11px] text-cyan-400 font-mono">
                  {isAutoCalibrate ? '⚡ Auto (Homeostase)' : `${calibrationValue.toFixed(2)} [Fixo]`}
                </span>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  type="button"
                  onClick={() => setIsAutoCalibrate(true)}
                  className={`px-2 py-1 text-[10px] font-mono rounded border transition cursor-pointer ${
                    isAutoCalibrate
                      ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white'
                  }`}
                >
                  ⚡ Auto
                </button>
                <button
                  type="button"
                  onClick={() => { setIsAutoCalibrate(false); setCalibrationValue(0.0); }}
                  className={`px-2 py-1 text-[10px] font-mono rounded border transition cursor-pointer ${
                    !isAutoCalibrate && calibrationValue === 0.0
                      ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white'
                  }`}
                >
                  0.0 [Det]
                </button>
                <button
                  type="button"
                  onClick={() => { setIsAutoCalibrate(false); setCalibrationValue(0.5); }}
                  className={`px-2 py-1 text-[10px] font-mono rounded border transition cursor-pointer ${
                    !isAutoCalibrate && calibrationValue === 0.5
                      ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white'
                  }`}
                >
                  0.5 [Calib]
                </button>
                <button
                  type="button"
                  onClick={() => { setIsAutoCalibrate(false); setCalibrationValue(1.0); }}
                  className={`px-2 py-1 text-[10px] font-mono rounded border transition cursor-pointer ${
                    !isAutoCalibrate && calibrationValue === 1.0
                      ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white'
                  }`}
                >
                  1.0 [Estoc]
                </button>
              </div>
              <div className="flex items-center gap-2 pt-1">
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={isAutoCalibrate ? (decision.calibration ?? 0.5) : calibrationValue}
                  onChange={(e) => {
                    setIsAutoCalibrate(false);
                    setCalibrationValue(parseFloat(e.target.value));
                  }}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Arquitetura Universal System 1 */}
          <div className="p-4 rounded-2xl bg-zinc-950/80 border border-zinc-800 shadow-xl space-y-2.5 text-xs">
            <div className="font-bold text-cyan-300 font-mono flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5 text-cyan-400" />
              <span>{lang === 'pt' ? 'ARQUITETURA AURA (SISTEMA 1)' : 'AURA ARCHITECTURE (SYSTEM 1)'}</span>
            </div>
            <div className="space-y-1.5 text-[11px] text-zinc-400 leading-relaxed font-mono">
              <div className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>
                  <strong className="text-zinc-200">Tronco Recorrente:</strong> GRU + ResMLP (256 dimensões latentes) com memória temporal de curto prazo.
                </span>
              </div>
              <div className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>
                  <strong className="text-zinc-200">Cálculo Amortizado:</strong> Ação reflexiva direta em O(1) sem busca combinatória exponencial em árvore.
                </span>
              </div>
              <div className="flex items-start gap-1.5">
                <span className="text-cyan-400">•</span>
                <span>
                  <strong className="text-zinc-200">Homeostase Termodinâmica:</strong> Auto-regulação da temperatura de amostragem por momentum de recompensa (System 1 puro).
                </span>
              </div>
            </div>
          </div>

          {/* Log Recente de Decisões */}
          <div className="p-4 rounded-2xl bg-zinc-950/80 border border-zinc-800 shadow-xl space-y-2.5">
            <div className="flex items-center justify-between text-xs font-mono font-bold text-zinc-200">
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-amber-400" />
                <span>{lang === 'pt' ? 'DECISÕES NEURAIS RECENTES' : 'RECENT NEURAL DECISIONS'}</span>
              </span>
              <span className="text-[10px] text-zinc-500 font-normal">
                {actionLog.length} passos
              </span>
            </div>

            <div className="space-y-1.5 min-h-[120px]">
              {actionLog.length === 0 ? (
                <div className="h-[120px] flex items-center justify-center text-[11px] text-zinc-600 font-mono">
                  {lang === 'pt' ? 'Aguardando início da avaliação...' : 'Waiting for evaluation to start...'}
                </div>
              ) : (
                actionLog.map((item) => (
                  <div
                    key={item.id}
                    className="p-1.5 rounded-lg bg-zinc-900/70 border border-zinc-850 flex items-center justify-between text-[11px] font-mono"
                  >
                    <span className="font-bold text-white flex items-center gap-1.5 truncate max-w-[140px]">
                      <span className="text-cyan-400">⚡</span>
                      <span>{item.name}</span>
                    </span>
                    <div className="flex items-center gap-2 text-[10px]">
                      <span className="text-emerald-400">{(item.confidence * 100).toFixed(0)}% conf</span>
                      <span className="text-zinc-500">|</span>
                      <span className="text-cyan-300">{item.latencyMs.toFixed(2)}ms</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Especificações Técnicas de Runtime */}
          <div className="p-3.5 rounded-xl bg-zinc-950 border border-zinc-800/90 text-[11px] font-mono text-zinc-400 space-y-1.5">
            <div className="flex justify-between">
              <span>Runtime Engine:</span>
              <span className="text-cyan-400 font-bold">ONNX Runtime Web (Wasm SIMD)</span>
            </div>
            <div className="flex justify-between">
              <span>Modelo Wasm:</span>
              <span className="text-white truncate max-w-[160px]">{currentCfg.file}</span>
            </div>
            <div className="flex justify-between">
              <span>Checkpoint (.pt):</span>
              <span className="text-emerald-400 font-semibold truncate max-w-[185px]">
                {manifestData[activeDemo]?.checkpoint_source || currentCfg.checkpointSource || `${currentCfg.file.replace('.onnx', '')}_trained.pt`}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Versão / Treino:</span>
              <span className="text-purple-400 truncate max-w-[160px]">
                {manifestData[activeDemo]?.checkpoint_version === 'v1' && activeDemo === 'rubiks_atomic'
                  ? 'v1 (4M passos)'
                  : (manifestData[activeDemo]?.checkpoint_version || currentCfg.checkpointVersion || 'v1 (Amortizado)')}
              </span>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};
