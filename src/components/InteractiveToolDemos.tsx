import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Layers,
  Box,
  GitBranch,
  Play,
  Pause,
  Check,
  FileText,
  Terminal,
  Lock,
  Globe,
  GitCommit,
  Table,
  Activity,
  RefreshCw,
  Zap,
  Crosshair,
  Cpu,
  Gauge,
  Sliders,
  Eye,
  X,
  Search,
  CheckCircle2,
  Flame,
  Award,
  HardDrive,
  Copy,
  ExternalLink,
  Key,
  Sparkles,
  DollarSign,
  CloudUpload,
  FileSpreadsheet,
  ArrowRight,
  Plus,
} from 'lucide-react';
import { SystemOneHudDemo } from './SystemOneHudDemo';

interface InteractiveToolDemosProps {
  lang?: 'en' | 'pt';
}

export const InteractiveToolDemos: React.FC<InteractiveToolDemosProps> = ({ lang = 'en' }) => {
  const [activeTab, setActiveTab] = useState<'system_one' | 'flutter_scene' | 'gittrack' | 'register_red' | 'mad' | 'mddd'>('system_one');

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-2xl p-4 sm:p-6 lg:p-8 backdrop-blur-xl shadow-2xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-zinc-800">
        <div>
          <h3 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="text-[#adff2f] font-mono text-sm">[LIVE_LAB]</span>
            {lang === 'pt' ? 'Bancada Interativa de Ferramentas Open Source' : 'Interactive Open-Source Tool Workbenches'}
          </h3>
          <p className="text-xs text-zinc-400 mt-1">
            {lang === 'pt'
              ? 'Interaja diretamente com a avaliação neural em tempo real do system_one (ONNX Runtime Web / Wasm SIMD), o renderizador 3D do Flutter Scene, o GitTrack (Rastreamento & IA VIP), o Register-red (Pipeline Apple Numbers & Drive), o MAD-cli (Doc Viva) e o mddd-cli (Spec-Driven Dev).'
              : 'Directly test real-time neural evaluation of system_one (ONNX Runtime Web / Wasm SIMD), Flutter Scene 3D Shaders, GitTrack (Commit Tracking & VIP AI), Register-red (Apple Numbers & Drive Pipeline), MAD-cli (Auto-Doccing), and mddd-cli (Spec-Driven Dev).'}
          </p>
        </div>

        {/* Tab Switcher - Ordered: system_one, Flutter_scene, GitTrack, Register-red, MAD-cli, mddd-cli */}
        <div className="flex flex-wrap items-center gap-1 p-1 bg-zinc-900 border border-zinc-800 rounded-lg">
          <button
            type="button"
            onClick={() => setActiveTab('system_one')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'system_one' ? 'bg-zinc-800 text-amber-400 font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            <span>system_one (HUD)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('flutter_scene')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer ${
              activeTab === 'flutter_scene' ? 'bg-zinc-800 text-[#38bdf8] font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            Flutter Scene (3D)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('gittrack')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer ${
              activeTab === 'gittrack' ? 'bg-zinc-800 text-[#c4b5fd] font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'GitTrack (Rastreamento & IA)' : 'GitTrack (Tracking & AI)'}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('register_red')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'register_red' ? 'bg-zinc-800 text-[#f87171] font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-[#f87171]" />
            <span>Register-red (Numbers)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('mad')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer ${
              activeTab === 'mad' ? 'bg-zinc-800 text-[#34d399] font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'MAD-cli (Doc Viva)' : 'MAD-cli (Auto-Doccing)'}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('mddd')}
            className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors cursor-pointer ${
              activeTab === 'mddd' ? 'bg-zinc-800 text-[#adff2f] font-bold shadow-sm' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {lang === 'pt' ? 'mddd-cli (Spec-Driven Dev)' : 'mddd-cli (Spec-Driven Dev)'}
          </button>
        </div>
      </div>

      {/* Tab 1: system_one High-Definition HUD & Real Telemetry */}
      {activeTab === 'system_one' && <SystemOneHudDemo lang={lang} />}

      {/* Tab 2: flutter_scene 3D Canvas */}
      {activeTab === 'flutter_scene' && <FlutterSceneDemo lang={lang} />}

      {/* Tab 3: GitTrack Telemetry */}
      {activeTab === 'gittrack' && <GitTrackDemo lang={lang} />}

      {/* Tab 4: Register-red Apple Numbers & Google Drive Financial Pipeline */}
      {activeTab === 'register_red' && <RegisterRedDemo lang={lang} />}

      {/* Tab 5: MAD-cli Mermaid Auto-Doccing with MAD Tags */}
      {activeTab === 'mad' && <MadCliDemo lang={lang} />}

      {/* Tab 6: mddd-cli Spec-Driven Development (Diagram + Decision Matrix) */}
      {activeTab === 'mddd' && <MdddCliDemo lang={lang} />}
    </div>
  );
};

/* =========================================================================
   1. FLUTTER_SCENE 3D SHADER DEMO
   ========================================================================= */
const FlutterSceneDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [wireframe, setWireframe] = useState<boolean>(true);
  const [bloom, setBloom] = useState<boolean>(true);
  const [rotSpeed, setRotSpeed] = useState<number>(1.2);
  const [colorMode, setColorMode] = useState<'cyan' | 'matrix' | 'sunset'>('cyan');
  const animFrameRef = useRef<number | null>(null);
  const angleRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      angleRef.current += 0.015 * rotSpeed;
      const a = angleRef.current;
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;

      ctx.fillStyle = '#060913';
      ctx.fillRect(0, 0, w, h);

      // Grid background
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += 24) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 24) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // 3D Cube Vertices
      const size = 68;
      const vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
      ];

      // Rotation matrix
      const cosY = Math.cos(a);
      const sinY = Math.sin(a);
      const cosX = Math.cos(a * 0.7);
      const sinX = Math.sin(a * 0.7);

      const projected = vertices.map(([vx, vy, vz]) => {
        // Rot Y
        const x1 = vx * cosY + vz * sinY;
        const z1 = -vx * sinY + vz * cosY;
        // Rot X
        const y2 = vy * cosX - z1 * sinX;
        const z2 = vy * sinX + z1 * cosX;
        // Project
        const fov = 200;
        const scale = fov / (fov + z2 * size * 0.7 + 140);
        return {
          x: cx + x1 * size * scale,
          y: cy + y2 * size * scale,
          z: z2,
        };
      });

      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];

      // Bloom glow
      if (bloom) {
        const glowColor =
          colorMode === 'cyan' ? '#38bdf8' : colorMode === 'matrix' ? '#adff2f' : '#f43f5e';
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 18;
      } else {
        ctx.shadowBlur = 0;
      }

      // Draw wireframe
      ctx.lineWidth = 2;
      ctx.strokeStyle =
        colorMode === 'cyan' ? '#38bdf8' : colorMode === 'matrix' ? '#adff2f' : '#fb7185';

      edges.forEach(([i, j]) => {
        const p1 = projected[i];
        const p2 = projected[j];
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      });

      // Reset shadow
      ctx.shadowBlur = 0;

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [wireframe, bloom, rotSpeed, colorMode]);

  return (
    <div className="mt-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Left: 3D Canvas Viewport */}
        <div className="lg:col-span-7 bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 flex flex-col items-center">
          <div className="w-full flex items-center justify-between pb-3 mb-2 border-b border-zinc-800 text-xs font-mono text-zinc-400">
            <span className="flex items-center gap-1.5 text-[#38bdf8] font-bold">
              <Box className="w-4 h-4" />
              <span>Flutter Scene Mesh Viewport</span>
            </span>
            <span className="text-emerald-400 font-bold">60 FPS · Metal/Vulkan</span>
          </div>

          <div className="relative w-full aspect-video max-h-72 rounded-lg overflow-hidden bg-[#060913] border border-zinc-800 flex items-center justify-center">
            <canvas ref={canvasRef} width={480} height={280} className="w-full h-full object-contain" />
            <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 backdrop-blur-sm text-[10px] font-mono text-zinc-400">
              FragmentShader: spatial_viewport.frag
            </div>
          </div>

          {/* Interactive controls */}
          <div className="w-full grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4 text-xs font-mono">
            <button
              type="button"
              onClick={() => setWireframe(!wireframe)}
              className={`p-2 rounded-lg border text-center transition-all cursor-pointer ${
                wireframe ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300' : 'bg-zinc-950 border-zinc-800 text-zinc-400'
              }`}
            >
              Wireframe: {wireframe ? 'ON' : 'OFF'}
            </button>
            <button
              type="button"
              onClick={() => setBloom(!bloom)}
              className={`p-2 rounded-lg border text-center transition-all cursor-pointer ${
                bloom ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300' : 'bg-zinc-950 border-zinc-800 text-zinc-400'
              }`}
            >
              Bloom: {bloom ? 'ON' : 'OFF'}
            </button>
            <button
              type="button"
              onClick={() => setRotSpeed((s) => (s >= 2.5 ? 0.5 : s + 0.7))}
              className="p-2 rounded-lg bg-zinc-950 border border-zinc-800 hover:border-zinc-700 text-zinc-300 text-center transition-all cursor-pointer"
            >
              Speed: {rotSpeed.toFixed(1)}x
            </button>
            <button
              type="button"
              onClick={() => setColorMode((c) => (c === 'cyan' ? 'matrix' : c === 'matrix' ? 'sunset' : 'cyan'))}
              className="p-2 rounded-lg bg-zinc-950 border border-zinc-800 hover:border-zinc-700 text-zinc-300 text-center transition-all cursor-pointer"
            >
              Palette: {colorMode.toUpperCase()}
            </button>
          </div>
        </div>

        {/* Right: Technical Explanation */}
        <div className="lg:col-span-5 space-y-4 font-mono text-xs">
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-3">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-[#38bdf8]">▹</span>
              <span>{lang === 'pt' ? 'Fork 3D & Animações de Scene' : '3D Fork & Scene Animations'}</span>
            </h4>
            <p className="text-zinc-300 font-sans leading-relaxed text-xs">
              {lang === 'pt'
                ? 'Fork de alta performance com contribuições focadas na esteira de animações de scenes e renderização 3D nativa sobre Impeller (Metal no iOS/macOS e Vulkan no Android/Linux). Elimina sobrecargas de ponte entre o código Dart e shaders GLSL/MSL compilados em tempo real.'
                : 'High-performance fork with contributions focused on scene animation pipelines and native 3D mesh rendering atop Impeller (Metal on iOS/macOS and Vulkan on Android/Linux), avoiding bridge overheads between Dart code and compiled GLSL/MSL shaders.'}
            </p>
            <div className="p-2.5 rounded bg-black/80 border border-zinc-800 text-[11px] text-zinc-400 space-y-1">
              <div className="text-cyan-400 font-bold">// Dart + Impeller Pipeline:</div>
              <div>Scene scene = Scene();</div>
              <div>Node meshNode = Node.mesh(cubeMesh);</div>
              <div>scene.root.add(meshNode);</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* =========================================================================
   2. GITTRACK TELEMETRY DEMO
   ========================================================================= */
const GitTrackDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang }) => {
  const [hasToken, setHasToken] = useState<boolean>(true);
  const [filterPrivacy, setFilterPrivacy] = useState<'all' | 'public' | 'private'>('all');
  const [isAnalyzingAi, setIsAnalyzingAi] = useState<boolean>(false);
  const [showAiAnalysis, setShowAiAnalysis] = useState<boolean>(false);

  const commits = [
    {
      hash: 'e89a12c',
      message: 'feat(ast): Implement invariant token validation in System 1',
      diff: '+142 -8',
      isPrivate: false,
      date: '2h ago',
    },
    {
      hash: '7c392f1',
      message: 'refactor(core): Microservice clean architecture domain boundary isolation',
      diff: '+215 -64',
      isPrivate: true,
      repo: 'enterprise/payment-gateway',
      date: '1d ago',
    },
    {
      hash: '4f2910b',
      message: 'perf(impeller): Optimize memory buffer allocation for Metal 3D shaders',
      diff: '+86 -14',
      isPrivate: false,
      date: '2d ago',
    },
    {
      hash: 'a10e42d',
      message: 'security(auth): Implement token-based biometric IPC session validation',
      diff: '+112 -19',
      isPrivate: true,
      repo: 'fintech/banking-native-sdk',
      date: '3d ago',
    },
    {
      hash: '9b772c1',
      message: 'docs(mddd): Add decision matrix compilation to Mermaid CLI spec',
      diff: '+320 -15',
      isPrivate: false,
      date: '4d ago',
    },
  ];

  const handleRequestAiAnalysis = () => {
    setIsAnalyzingAi(true);
    setTimeout(() => {
      setIsAnalyzingAi(false);
      setShowAiAnalysis(true);
    }, 600);
  };

  const filtered = commits.filter((c) => {
    if (filterPrivacy === 'public') return !c.isPrivate;
    if (filterPrivacy === 'private') return c.isPrivate;
    return true;
  });

  return (
    <div className="mt-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Commit Stream & Token Configuration */}
        <div className="lg:col-span-7 bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-4">
          {/* Header & Filter Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-800">
            <span className="text-xs font-mono font-bold text-white flex items-center gap-2">
              <GitCommit className="w-4 h-4 text-[#c4b5fd]" />
              <span>{lang === 'pt' ? 'Rastreamento Granular de Commits' : 'Granular Commit Tracking'}</span>
            </span>
            <div className="flex gap-1 text-[11px] font-mono">
              {(['all', 'public', 'private'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setFilterPrivacy(mode)}
                  className={`px-2 py-0.5 rounded cursor-pointer transition ${
                    filterPrivacy === mode
                      ? 'bg-[#c4b5fd] text-black font-bold'
                      : 'bg-zinc-950 text-zinc-400 hover:text-white border border-zinc-800'
                  }`}
                >
                  {mode === 'all'
                    ? (lang === 'pt' ? 'TODOS' : 'ALL')
                    : mode === 'public'
                    ? (lang === 'pt' ? 'PÚBLICOS' : 'PUBLIC')
                    : (lang === 'pt' ? 'PRIVADOS' : 'PRIVATE')}
                </button>
              ))}
            </div>
          </div>

          {/* Token Access Simulation Header */}
          <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 text-xs font-mono">
            <div className="flex items-center gap-2">
              <Key className={`w-3.5 h-3.5 ${hasToken ? 'text-emerald-400' : 'text-zinc-500'}`} />
              <span className="text-zinc-400">{lang === 'pt' ? 'Token Repos Privados:' : 'Private Repo Token:'}</span>
              <code className="text-zinc-300 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800 text-[11px]">
                {hasToken ? 'ghp_••••••••••••94b2' : (lang === 'pt' ? 'Não informado' : 'Not provided')}
              </code>
            </div>
            <button
              type="button"
              onClick={() => setHasToken(!hasToken)}
              className={`px-2.5 py-1 rounded text-[11px] font-mono transition cursor-pointer self-start sm:self-auto ${
                hasToken
                  ? 'bg-emerald-500/10 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/20'
                  : 'bg-zinc-800 border border-zinc-700 text-zinc-300 hover:text-white'
              }`}
            >
              {hasToken
                ? (lang === 'pt' ? '✓ Token Ativo (Clique p/ Remover)' : '✓ Token Active (Click to Remove)')
                : (lang === 'pt' ? '+ Incluir Token Privado' : '+ Include Private Token')}
            </button>
          </div>

          {/* Commits Stream */}
          <div className="space-y-2">
            {filtered.map((c, i) => {
              const isLockedPrivate = c.isPrivate && !hasToken;
              return (
                <div
                  key={i}
                  className={`p-3 rounded-lg border text-xs font-mono flex items-center justify-between gap-3 ${
                    c.isPrivate
                      ? isLockedPrivate
                        ? 'bg-zinc-950/80 border-zinc-850 text-zinc-500'
                        : 'bg-amber-950/20 border-amber-500/30 text-amber-200'
                      : 'bg-zinc-950 border-zinc-800 text-zinc-300'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    {c.isPrivate ? (
                      <Lock className={`w-3.5 h-3.5 shrink-0 ${isLockedPrivate ? 'text-zinc-500' : 'text-amber-400'}`} />
                    ) : (
                      <GitBranch className="w-3.5 h-3.5 text-[#c4b5fd] shrink-0" />
                    )}
                    <span className="text-zinc-500 font-bold shrink-0">{c.hash}</span>
                    <span className="truncate">
                      {isLockedPrivate
                        ? (lang === 'pt' ? '🔒 [Repositório Privado · Inclua o token para visualizar]' : '🔒 [Private Repo · Include token to view]')
                        : c.message}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 text-[10px]">
                    {!isLockedPrivate && (
                      <span className="text-emerald-400/90 font-mono hidden sm:inline">{c.diff}</span>
                    )}
                    <span className="text-zinc-500">{c.date}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* VIP AI Commit Analysis Action */}
          <div className="pt-2">
            <button
              type="button"
              onClick={handleRequestAiAnalysis}
              disabled={isAnalyzingAi}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-amber-500/20 via-purple-500/20 to-sky-500/20 hover:from-amber-500/30 hover:to-sky-500/30 border border-amber-500/40 text-amber-300 hover:text-white rounded-lg text-xs font-mono font-bold flex items-center justify-center gap-2 transition cursor-pointer disabled:opacity-50"
            >
              <Sparkles className={`w-3.5 h-3.5 text-amber-400 ${isAnalyzingAi ? 'animate-spin' : ''}`} />
              <span>
                {isAnalyzingAi
                  ? (lang === 'pt' ? 'Processando Análise por IA dos Commits...' : 'Processing AI Commit Analysis...')
                  : (lang === 'pt' ? 'Solicitar Análise Detalhada por IA (Plano VIP)' : 'Request Detailed AI Commit Analysis (VIP Plan)')}
              </span>
            </button>
          </div>

          {/* VIP AI Analysis Report Box */}
          {showAiAnalysis && (
            <div className="p-3.5 rounded-lg bg-amber-950/20 border border-amber-500/40 space-y-2.5 font-mono text-xs">
              <div className="flex items-center justify-between text-amber-300 font-bold pb-2 border-b border-amber-500/30">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  <span>{lang === 'pt' ? 'Relatório de IA GitTrack (Plano VIP)' : 'GitTrack AI Report (VIP Plan)'}</span>
                </span>
                <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30">
                  {lang === 'pt' ? 'SCORE: 98/100' : 'SCORE: 98/100'}
                </span>
              </div>
              <p className="text-zinc-300 font-sans text-xs leading-relaxed">
                {lang === 'pt'
                  ? 'A análise de IA detectou alta consistência arquitetural com commits atômicos, mensagens padronizadas em Conventional Commits e separação estrita entre domínio e infraestrutura. Os repositórios privados desbloqueados via token apresentam conformidade total com Clean Architecture e zero vazamento de credenciais.'
                  : 'AI analysis detected high architectural consistency with atomic commits, conventional commit syntax, and strict separation between domain and infrastructure. Private repos unlocked via token exhibit full Clean Architecture compliance with zero credential leakage.'}
              </p>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[10px] text-zinc-400">
                <div className="p-1.5 rounded bg-black/40 border border-zinc-800">
                  <span className="text-zinc-500 block">{lang === 'pt' ? 'Padrão' : 'Pattern'}</span>
                  <span className="text-white font-bold">Clean Arch</span>
                </div>
                <div className="p-1.5 rounded bg-black/40 border border-zinc-800">
                  <span className="text-zinc-500 block">{lang === 'pt' ? 'Atomicidade' : 'Atomicity'}</span>
                  <span className="text-emerald-400 font-bold">100%</span>
                </div>
                <div className="p-1.5 rounded bg-black/40 border border-zinc-800">
                  <span className="text-zinc-500 block">{lang === 'pt' ? 'Risco' : 'Risk'}</span>
                  <span className="text-sky-400 font-bold">{lang === 'pt' ? 'Mínimo' : 'Minimal'}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: Technical Details & Metrics Deck */}
        <div className="lg:col-span-5 space-y-3 font-mono text-xs">
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-3">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-[#c4b5fd]">▹</span>
              <span>{lang === 'pt' ? 'Rastreamento Detalhado & IA VIP' : 'Granular Tracking & VIP AI'}</span>
            </h4>
            <p className="text-zinc-300 font-sans text-xs leading-relaxed">
              {lang === 'pt'
                ? 'O GitTrack rastreia minuciosamente o fluxo de commits do desenvolvedor, permitindo a inclusão segura de token de acesso pessoal (PAT) para indexar repositórios privados com proteção e mascaramento. No plano VIP, é possível solicitar análises profundas de qualidade e impacto via IA.'
                : 'GitTrack provides granular tracking of developer commit streams, supporting Personal Access Tokens (PAT) to safely index private repositories with secret masking. On the VIP plan, developers can request deep AI analysis auditing commit quality and architectural impact.'}
            </p>
            <div className="grid grid-cols-2 gap-2 pt-2">
              <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                <div className="text-[10px] text-zinc-500">{lang === 'pt' ? 'Rastreamento' : 'Tracking'}</div>
                <div className="text-base font-bold text-white">{lang === 'pt' ? 'Detalhado' : 'Granular'}</div>
              </div>
              <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                <div className="text-[10px] text-zinc-500">{lang === 'pt' ? 'Repos Privados' : 'Private Repos'}</div>
                <div className="text-base font-bold text-emerald-400">{lang === 'pt' ? 'Via Token' : 'Token Access'}</div>
              </div>
              <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                <div className="text-[10px] text-zinc-500">{lang === 'pt' ? 'Análise IA' : 'AI Analysis'}</div>
                <div className="text-base font-bold text-amber-400">{lang === 'pt' ? 'Plano VIP' : 'VIP Tier'}</div>
              </div>
              <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                <div className="text-[10px] text-zinc-500">{lang === 'pt' ? 'Motor' : 'Engine'}</div>
                <div className="text-base font-bold text-[#c4b5fd]">Go Core</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* =========================================================================
   4. REGISTER-RED APPLE NUMBERS & GOOGLE DRIVE PIPELINE DEMO
   ========================================================================= */
interface BillItem {
  id: string;
  descriptionPt: string;
  descriptionEn: string;
  categoryPt: string;
  categoryEn: string;
  dueDatePt: string;
  dueDateEn: string;
  amount: number;
  paid: boolean;
}

const DEFAULT_OCTOBER_BILLS: BillItem[] = [
  {
    id: '1',
    descriptionPt: 'Aluguel & Condomínio',
    descriptionEn: 'Rent & HOA Fees',
    categoryPt: 'Casa',
    categoryEn: 'Housing',
    dueDatePt: '10/10',
    dueDateEn: 'Oct 10',
    amount: 3200,
    paid: true,
  },
  {
    id: '2',
    descriptionPt: 'Servidor Cloud & GCP',
    descriptionEn: 'Cloud Server & GCP',
    categoryPt: 'Dev / Nuvem',
    categoryEn: 'Dev / Cloud',
    dueDatePt: '15/10',
    dueDateEn: 'Oct 15',
    amount: 850,
    paid: true,
  },
  {
    id: '3',
    descriptionPt: 'Internet Fibra Óptica 1Gbps',
    descriptionEn: 'Gigabit Fiber Internet',
    categoryPt: 'Serviços',
    categoryEn: 'Utilities',
    dueDatePt: '18/10',
    dueDateEn: 'Oct 18',
    amount: 180,
    paid: true,
  },
  {
    id: '4',
    descriptionPt: 'Contabilidade & DAS PJ',
    descriptionEn: 'Corporate Accounting & Taxes',
    categoryPt: 'Operação',
    categoryEn: 'Operations',
    dueDatePt: '20/10',
    dueDateEn: 'Oct 20',
    amount: 650,
    paid: false,
  },
  {
    id: '5',
    descriptionPt: 'Energia Elétrica COPEL',
    descriptionEn: 'Electric Utility COPEL',
    categoryPt: 'Casa',
    categoryEn: 'Housing',
    dueDatePt: '22/10',
    dueDateEn: 'Oct 22',
    amount: 340,
    paid: false,
  },
  {
    id: '6',
    descriptionPt: 'Saúde & Academia',
    descriptionEn: 'Healthcare & Fitness',
    categoryPt: 'Saúde',
    categoryEn: 'Health',
    dueDatePt: '25/10',
    dueDateEn: 'Oct 25',
    amount: 220,
    paid: false,
  },
];

const RegisterRedDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang = 'en' }) => {
  const [monthKey, setMonthKey] = useState<'oct' | 'nov'>('oct');
  const [bills, setBills] = useState<BillItem[]>(DEFAULT_OCTOBER_BILLS);
  const [income, setIncome] = useState<number>(18500);
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [lastSyncTime, setLastSyncTime] = useState<string | null>('18:05:06');
  const [syncAlert, setSyncAlert] = useState<string | null>(null);
  const [activeSideTab, setActiveSideTab] = useState<'architecture' | 'logs' | 'export'>('architecture');
  const [exportedFormat, setExportedFormat] = useState<string | null>(null);

  const currentMonth = useMemo(() => {
    if (monthKey === 'oct') {
      return lang === 'pt' ? 'Outubro 2026' : 'October 2026';
    }
    return lang === 'pt' ? 'Novembro 2026' : 'November 2026';
  }, [monthKey, lang]);

  const [apiLogs, setApiLogs] = useState<string[]>(() => [
    '[FASTAPI] Uvicorn worker running on http://127.0.0.1:8000 (v0.115.0)',
    '[GOOGLE-DRIVE] Direct mount ~/Library/CloudStorage/.../Numbers connected',
    '[NUMBERS-PARSER] 18 .numbers sheets indexed. Cache initialized.',
    lang === 'pt'
      ? '[GET /api/months] HTTP 200 OK — Conjunto ativo: Outubro 2026'
      : '[GET /api/months] HTTP 200 OK — Active dataset: October 2026',
  ]);

  // Sync API log language when lang toggles if user switches language
  useEffect(() => {
    setApiLogs([
      lang === 'pt'
        ? '[FASTAPI] Uvicorn worker rodando em http://127.0.0.1:8000 (v0.115.0)'
        : '[FASTAPI] Uvicorn worker running on http://127.0.0.1:8000 (v0.115.0)',
      lang === 'pt'
        ? '[GOOGLE-DRIVE] Ponto de montagem ~/Library/CloudStorage/.../Numbers conectado'
        : '[GOOGLE-DRIVE] Direct mount ~/Library/CloudStorage/.../Numbers connected',
      lang === 'pt'
        ? '[NUMBERS-PARSER] 18 planilhas .numbers indexadas. Cache inicializado.'
        : '[NUMBERS-PARSER] 18 .numbers sheets indexed. Cache initialized.',
      lang === 'pt'
        ? `[GET /api/months] HTTP 200 OK — Conjunto ativo: ${monthKey === 'oct' ? 'Outubro 2026' : 'Novembro 2026'}`
        : `[GET /api/months] HTTP 200 OK — Active dataset: ${monthKey === 'oct' ? 'October 2026' : 'November 2026'}`,
    ]);
  }, [lang, monthKey]);

  // Derived financial metrics
  const totalExpenses = useMemo(() => bills.reduce((acc, b) => acc + b.amount, 0), [bills]);
  const totalPaid = useMemo(() => bills.filter((b) => b.paid).reduce((acc, b) => acc + b.amount, 0), [bills]);
  const totalPending = useMemo(() => bills.filter((b) => !b.paid).reduce((acc, b) => acc + b.amount, 0), [bills]);
  const netBalance = useMemo(() => income - totalExpenses, [income, totalExpenses]);
  const allPaid = useMemo(() => bills.length > 0 && bills.every((b) => b.paid), [bills]);

  const formatCurrency = (val: number) => {
    if (lang === 'pt') {
      return `R$ ${val.toLocaleString('pt-BR')},00`;
    }
    return `R$ ${val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const togglePaid = (id: string) => {
    setBills((prev) =>
      prev.map((item) => (item.id === id ? { ...item, paid: !item.paid } : item))
    );
  };

  const handleSyncGoogleDrive = () => {
    setIsSyncing(true);
    setSyncAlert(null);
    setTimeout(() => {
      const now = new Date().toLocaleTimeString();
      setIsSyncing(false);
      setLastSyncTime(now);
      const msg =
        lang === 'pt'
          ? `Planilha '${currentMonth}.numbers' regravada no Google Drive com sucesso!`
          : `Spreadsheet '${currentMonth}.numbers' synchronized with Google Drive!`;
      setSyncAlert(msg);
      setApiLogs((logs) => [
        `[${now}] [POST /api/sync-drive] HTTP 200 OK — Flush to ~/Library/CloudStorage/.../${currentMonth.replace(' ', '_')}.numbers`,
        ...logs.slice(0, 7),
      ]);
      setTimeout(() => setSyncAlert(null), 4000);
    }, 700);
  };

  const handleCloneMonth = () => {
    if (monthKey === 'nov') {
      // Reset back to October
      setMonthKey('oct');
      setBills(DEFAULT_OCTOBER_BILLS);
      setIncome(18500);
      const now = new Date().toLocaleTimeString();
      setApiLogs((logs) => [
        lang === 'pt'
          ? `[${now}] [GET /api/months/outubro-2026] Linha de base de Outubro restaurada`
          : `[${now}] [GET /api/months/october-2026] Restored October baseline`,
        ...logs.slice(0, 7),
      ]);
      return;
    }

    // Clone into November
    const rolloverSavings = Math.max(0, netBalance);
    const clonedBills: BillItem[] = DEFAULT_OCTOBER_BILLS.map((item) => ({
      ...item,
      id: `nov-${item.id}`,
      dueDatePt: item.dueDatePt.replace('/10', '/11'),
      dueDateEn: item.dueDateEn.replace('Oct', 'Nov'),
      paid: false,
    }));

    setMonthKey('nov');
    setBills(clonedBills);
    setIncome(18500 + rolloverSavings);

    const now = new Date().toLocaleTimeString();
    setSyncAlert(
      lang === 'pt'
        ? `Novembro 2026 gerado com sucesso! Economias transferidas: ${formatCurrency(rolloverSavings)}`
        : `November 2026 generated! Rollover balance carried forward: ${formatCurrency(rolloverSavings)}`
    );
    setApiLogs((logs) => [
      lang === 'pt'
        ? `[${now}] [POST /api/months/clone] Criado 'Novembro 2026'. Propagadas 6 despesas fixas recorrentes. Status resetados.`
        : `[${now}] [POST /api/months/clone] Created 'November 2026'. Propagated 6 fixed recurring bills. Reset statuses.`,
      lang === 'pt'
        ? `[${now}] [ROLLOVER] Saldo positivo de ${formatCurrency(rolloverSavings)} transferido como saldo inicial.`
        : `[${now}] [ROLLOVER] Rolled over positive balance ${formatCurrency(rolloverSavings)} into new month baseline.`,
      lang === 'pt'
        ? `[${now}] [NUMBERS-WRITE] Novo arquivo criado ~/GoogleDrive/Numbers/Novembro_2026.numbers`
        : `[${now}] [NUMBERS-WRITE] Created new file ~/GoogleDrive/Numbers/November_2026.numbers`,
      ...logs.slice(0, 6),
    ]);
    setTimeout(() => setSyncAlert(null), 4500);
  };

  const handleExport = (fmt: 'numbers' | 'xlsx') => {
    setExportedFormat(fmt);
    const now = new Date().toLocaleTimeString();
    setApiLogs((logs) => [
      lang === 'pt'
        ? `[${now}] [DOWNLOAD] Pacote de arquivo ${fmt === 'numbers' ? 'Apple Numbers (.numbers)' : 'Excel (.xlsx via openpyxl)'} gerado para ${currentMonth}`
        : `[${now}] [DOWNLOAD] Generated ${fmt === 'numbers' ? 'Apple Numbers (.numbers)' : 'Excel (.xlsx via openpyxl)'} file package for ${currentMonth}`,
      ...logs.slice(0, 7),
    ]);
    setTimeout(() => setExportedFormat(null), 2500);
  };

  return (
    <div className="mt-6 space-y-6">
      {/* Top Banner: Month Selector, Sync CTA & Clone CTA */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-4 rounded-xl bg-zinc-900/70 border border-zinc-800">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="p-2 rounded-lg bg-red-500/10 border border-red-500/30 text-[#f87171]">
              <FileSpreadsheet className="w-4 h-4" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono font-bold text-white">{currentMonth}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-red-500/15 text-[#f87171] border border-red-500/30 font-semibold">
                  .numbers Native
                </span>
              </div>
              <span className="text-[11px] text-zinc-400 font-mono">
                {lang === 'pt' ? 'Planilha integrada ao Google Drive' : 'Spreadsheet connected to Google Drive'}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Sync Button */}
          <button
            type="button"
            onClick={handleSyncGoogleDrive}
            disabled={isSyncing}
            className="px-3.5 py-1.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 border border-red-500/40 text-[#f87171] hover:text-white text-xs font-mono font-bold flex items-center gap-2 transition cursor-pointer disabled:opacity-50"
          >
            <CloudUpload className={`w-3.5 h-3.5 ${isSyncing ? 'animate-bounce' : ''}`} />
            <span>
              {isSyncing
                ? (lang === 'pt' ? 'Sincronizando...' : 'Syncing...')
                : (lang === 'pt' ? 'Sincronizar no Google Drive' : 'Sync to Google Drive')}
            </span>
          </button>

          {/* Smart Clone Button */}
          <button
            type="button"
            onClick={handleCloneMonth}
            className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-200 hover:text-white text-xs font-mono font-bold flex items-center gap-1.5 transition cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5 text-[#adff2f]" />
            <span>
              {monthKey === 'oct'
                ? (lang === 'pt' ? '+ Novo Mês (Clonagem Inteligente)' : '+ New Month (Smart Clone)')
                : (lang === 'pt' ? 'Voltar para Outubro 2026' : 'Restore October Baseline')}
            </span>
          </button>

          {lastSyncTime && (
            <span className="text-[10px] font-mono text-zinc-500 self-center hidden sm:inline">
              {lang === 'pt' ? `Última sincronia: ${lastSyncTime}` : `Last sync: ${lastSyncTime}`}
            </span>
          )}
        </div>
      </div>

      {/* Sync Alert Banner */}
      {syncAlert && (
        <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-xs font-mono flex items-center justify-between animate-in fade-in">
          <span className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{syncAlert}</span>
          </span>
          <span className="text-[10px] text-emerald-400/80">Google Drive · 200 OK</span>
        </div>
      )}

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            {lang === 'pt' ? 'Entradas do Mês' : 'Monthly Income'}
          </span>
          <div className="text-base sm:text-lg font-mono font-bold text-emerald-400 mt-1 tabular-nums">
            {formatCurrency(income)}
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {monthKey === 'nov'
              ? (lang === 'pt' ? 'Base + Economias' : 'Base + Rollover')
              : (lang === 'pt' ? 'Receita Total' : 'Gross Revenue')}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            {lang === 'pt' ? 'Total Despesas' : 'Total Expenses'}
          </span>
          <div className="text-base sm:text-lg font-mono font-bold text-red-400 mt-1 tabular-nums">
            {formatCurrency(totalExpenses)}
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {bills.length} {lang === 'pt' ? 'contas listadas' : 'accounts listed'}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            {lang === 'pt' ? 'Total Pago' : 'Total Paid'}
          </span>
          <div className="text-base sm:text-lg font-mono font-bold text-[#38bdf8] mt-1 tabular-nums">
            {formatCurrency(totalPaid)}
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {bills.filter((b) => b.paid).length} {lang === 'pt' ? 'quitadas' : 'settled'}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            {lang === 'pt' ? 'Pendente a Pagar' : 'Pending to Pay'}
          </span>
          <div className="text-base sm:text-lg font-mono font-bold text-amber-400 mt-1 tabular-nums">
            {formatCurrency(totalPending)}
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {bills.filter((b) => !b.paid).length} {lang === 'pt' ? 'pendentes' : 'open'}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800 col-span-2 sm:col-span-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            {lang === 'pt' ? 'Saldo Restante' : 'Net Surplus'}
          </span>
          <div className="text-base sm:text-lg font-mono font-bold text-[#adff2f] mt-1 tabular-nums">
            {formatCurrency(netBalance)}
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {lang === 'pt' ? 'Disponível p/ poupar' : 'Free cash flow'}
          </span>
        </div>
      </div>

      {/* Main Grid: Interactive Table + Architectural Sub-panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 cols: Table of Accounts & Payment Actions */}
        <div className="lg:col-span-7 bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
            <span className="text-xs font-mono font-bold text-white flex items-center gap-2">
              <Table className="w-4 h-4 text-[#f87171]" />
              <span>{lang === 'pt' ? 'Contas do Mês & Baixa Rápida' : 'Monthly Bills & 1-Click Settlement'}</span>
            </span>
            <span className="text-[11px] font-mono text-zinc-400">
              {lang === 'pt' ? 'Clique no status para liquidar' : 'Click status to toggle payment'}
            </span>
          </div>

          {/* Celebratory Banner if 100% Paid */}
          {allPaid && (
            <div className="p-3 rounded-lg bg-[#adff2f]/10 border border-[#adff2f]/40 text-[#adff2f] text-xs font-mono flex items-center gap-2 animate-pulse">
              <Award className="w-4 h-4 shrink-0" />
              <span>
                {lang === 'pt'
                  ? '🎉 Todas as contas do mês foram liquidadas com sucesso!'
                  : '🎉 100% Cleared! All bills for this month have been settled.'}
              </span>
            </div>
          )}

          {/* Bills List */}
          <div className="space-y-2">
            {bills.map((bill) => (
              <div
                key={bill.id}
                className={`p-2.5 rounded-lg border transition-all flex items-center justify-between gap-3 text-xs font-mono ${
                  bill.paid
                    ? 'bg-zinc-950/60 border-zinc-850 opacity-90'
                    : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700 shadow-xs'
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <button
                    type="button"
                    onClick={() => togglePaid(bill.id)}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase transition cursor-pointer shrink-0 ${
                      bill.paid
                        ? 'bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/25'
                        : 'bg-amber-500/15 border border-amber-500/40 text-amber-300 hover:bg-amber-500/25'
                    }`}
                    title={lang === 'pt' ? 'Clique para alternar PAGO / PENDENTE' : 'Click to toggle PAID / PENDING'}
                  >
                    {bill.paid ? (lang === 'pt' ? '✓ PAGO' : '✓ PAID') : (lang === 'pt' ? 'PENDENTE' : 'PENDING')}
                  </button>

                  <div className="min-w-0">
                    <span className={`block font-sans text-xs font-medium truncate ${bill.paid ? 'text-zinc-400 line-through' : 'text-zinc-100'}`}>
                      {lang === 'pt' ? bill.descriptionPt : bill.descriptionEn}
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      {lang === 'pt' ? bill.categoryPt : bill.categoryEn} · {lang === 'pt' ? `Vence ${bill.dueDatePt}` : `Due ${bill.dueDateEn}`}
                    </span>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <span className={`font-mono font-bold tabular-nums ${bill.paid ? 'text-zinc-400' : 'text-white'}`}>
                    {formatCurrency(bill.amount)}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-850 flex items-center justify-between text-[11px] font-mono text-zinc-400">
            <span>
              {lang === 'pt' ? 'Formato de gravação em disco:' : 'Disk write format:'}
            </span>
            <code className="text-[#f87171] bg-red-950/30 px-2 py-0.5 rounded border border-red-500/20">
              Numbers v13+ (.numbers package directory)
            </code>
          </div>
        </div>

        {/* Right 5 cols: Architecture, API Logs & Multi-format Export */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-3">
            {/* Sub-tab Switcher */}
            <div className="flex items-center gap-1 p-1 bg-zinc-950 border border-zinc-800 rounded-lg">
              <button
                type="button"
                onClick={() => setActiveSideTab('architecture')}
                className={`flex-1 py-1 text-[11px] font-mono rounded transition cursor-pointer ${
                  activeSideTab === 'architecture'
                    ? 'bg-zinc-800 text-white font-bold'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {lang === 'pt' ? 'Arquitetura' : 'Architecture'}
              </button>
              <button
                type="button"
                onClick={() => setActiveSideTab('logs')}
                className={`flex-1 py-1 text-[11px] font-mono rounded transition cursor-pointer ${
                  activeSideTab === 'logs'
                    ? 'bg-zinc-800 text-[#38bdf8] font-bold'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {lang === 'pt' ? 'Logs FastAPI' : 'FastAPI Logs'}
              </button>
              <button
                type="button"
                onClick={() => setActiveSideTab('export')}
                className={`flex-1 py-1 text-[11px] font-mono rounded transition cursor-pointer ${
                  activeSideTab === 'export'
                    ? 'bg-zinc-800 text-[#adff2f] font-bold'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {lang === 'pt' ? 'Exportar' : 'Export'}
              </button>
            </div>

            {/* Sub-tab 1: Architecture Highlights */}
            {activeSideTab === 'architecture' && (
              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-850 space-y-2">
                  <div className="text-[#f87171] font-bold flex items-center gap-1.5">
                    <span>▹</span>
                    <span>{lang === 'pt' ? 'Leitura & Escrita Nativa Apple Numbers' : 'Native Apple Numbers I/O'}</span>
                  </div>
                  <p className="text-zinc-300 font-sans text-xs leading-relaxed">
                    {lang === 'pt'
                      ? 'Utiliza numbers-parser em Python para decodificar diretamente o container Snappy/Protobuf do formato .numbers proprietário da Apple, recalculando somas sem passar por intermediários.'
                      : 'Leverages Python numbers-parser to inspect Apple’s proprietary Snappy/Protobuf container, modifying tables and recalculating formulas with zero loss.'}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-850 space-y-2">
                  <div className="text-[#38bdf8] font-bold flex items-center gap-1.5">
                    <span>▹</span>
                    <span>{lang === 'pt' ? 'Sincronização com Google Drive' : 'Google Drive Cloud Sync'}</span>
                  </div>
                  <p className="text-zinc-300 font-sans text-xs leading-relaxed">
                    {lang === 'pt'
                      ? 'Integrado diretamente ao diretório de montagem do Google Drive no macOS, regravando a planilha original na nuvem para visualização instantânea no iPhone, iPad e Mac.'
                      : 'Directly hooks into Google Drive mount paths on macOS, flushing atomic writes to cloud storage for instant reflection across Apple ecosystem devices.'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[10px]">
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                    <span className="text-zinc-500 block">Backend</span>
                    <span className="text-white font-bold">FastAPI / Python 3.11</span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                    <span className="text-zinc-500 block">Frontend</span>
                    <span className="text-[#f87171] font-bold">React 19 / Vite</span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                    <span className="text-zinc-500 block">Parser Core</span>
                    <span className="text-white font-bold">numbers-parser</span>
                  </div>
                  <div className="p-2 rounded bg-zinc-950 border border-zinc-850">
                    <span className="text-zinc-500 block">Excel Export</span>
                    <span className="text-emerald-400 font-bold">openpyxl</span>
                  </div>
                </div>
              </div>
            )}

            {/* Sub-tab 2: FastAPI Logs Stream */}
            {activeSideTab === 'logs' && (
              <div className="space-y-2 font-mono text-[11px]">
                <div className="flex items-center justify-between pb-1 text-zinc-500 text-[10px]">
                  <span>stdout / uvicorn.log</span>
                  <span className="text-emerald-400">● LIVE</span>
                </div>
                <div className="p-3 rounded-lg bg-black/90 border border-zinc-800 space-y-1.5 max-h-[220px] overflow-y-auto">
                  {apiLogs.map((log, idx) => (
                    <div key={idx} className="text-zinc-300 leading-tight">
                      <span className="text-zinc-600 mr-1.5">$</span>
                      <span className={log.includes('200') || log.includes('201') ? 'text-emerald-400' : log.includes('FASTAPI') ? 'text-sky-400' : 'text-zinc-300'}>
                        {log}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Sub-tab 3: Multi-format Export Actions */}
            {activeSideTab === 'export' && (
              <div className="space-y-3 font-mono text-xs">
                <p className="text-zinc-400 font-sans text-xs">
                  {lang === 'pt'
                    ? 'Baixe a planilha ativa em formatos nativos para edição offline ou compartilhamento corporativo:'
                    : 'Download active dataset in native formats for offline analysis or team distribution:'}
                </p>

                <div className="space-y-2">
                  <button
                    type="button"
                    onClick={() => handleExport('numbers')}
                    className="w-full p-2.5 rounded-lg bg-zinc-950 hover:bg-zinc-850 border border-zinc-800 hover:border-zinc-700 text-zinc-200 flex items-center justify-between transition cursor-pointer"
                  >
                    <span className="flex items-center gap-2">
                      <FileSpreadsheet className="w-4 h-4 text-[#f87171]" />
                      <span>{currentMonth}.numbers (Apple Numbers)</span>
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      {exportedFormat === 'numbers' ? (lang === 'pt' ? '✓ Gerado' : '✓ Generated') : 'Download'}
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleExport('xlsx')}
                    className="w-full p-2.5 rounded-lg bg-zinc-950 hover:bg-zinc-850 border border-zinc-800 hover:border-zinc-700 text-zinc-200 flex items-center justify-between transition cursor-pointer"
                  >
                    <span className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-emerald-400" />
                      <span>{currentMonth}.xlsx (Microsoft Excel)</span>
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      {exportedFormat === 'xlsx' ? (lang === 'pt' ? '✓ Gerado' : '✓ Generated') : 'Download'}
                    </span>
                  </button>
                </div>

                <div className="pt-2 flex items-center justify-between text-[11px] text-zinc-400">
                  <a
                    href="https://github.com/byte-od/Register-red"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[#f87171] hover:underline flex items-center gap-1"
                  >
                    <span>{lang === 'pt' ? 'Ver repositório Register-red no GitHub' : 'View Register-red repository on GitHub'}</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/* =========================================================================
   5. MAD-CLI (AUTO-DOCCING) DEMO
   ========================================================================= */
const MadCliDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang }) => {
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [docState, setDocState] = useState<'synced' | 'pending'>('synced');

  const handleSync = () => {
    setIsSyncing(true);
    setTimeout(() => {
      setIsSyncing(false);
      setDocState('synced');
    }, 700);
  };

  return (
    <div className="mt-6 space-y-6 font-mono text-xs">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Living Documentation Markdown View */}
        <div className="lg:col-span-7 bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
            <span className="font-bold text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-[#34d399]" />
              <span>README.md (com Tags MAD Vivas)</span>
            </span>
            <button
              type="button"
              onClick={handleSync}
              disabled={isSyncing}
              className="px-2.5 py-1 rounded bg-[#34d399]/20 hover:bg-[#34d399]/30 text-[#34d399] border border-[#34d399]/40 font-bold transition cursor-pointer flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3 h-3 ${isSyncing ? 'animate-spin' : ''}`} />
              <span>{isSyncing ? 'Sincronizando...' : 'mad-cli sync'}</span>
            </button>
          </div>

          <div className="p-3 rounded-lg bg-black/80 border border-zinc-800 text-[11px] text-zinc-300 space-y-1.5 font-mono">
            <span className="text-zinc-500">&lt;!-- MAD:ARCHITECTURE:START --&gt;</span>
            <div className="pl-3 border-l border-[#34d399]/40 text-emerald-300">
              ```mermaid<br />
              graph LR<br />
              &nbsp;&nbsp;AST_Parser --&gt; FastPath_Kernel<br />
              &nbsp;&nbsp;FastPath_Kernel --&gt;|Sub-800µs| System_1<br />
              &nbsp;&nbsp;FastPath_Kernel --&gt;|Causal| CIR_503M<br />
              ```
            </div>
            <span className="text-zinc-500">&lt;!-- MAD:ARCHITECTURE:END --&gt;</span>
          </div>
          <div className="flex items-center justify-between text-[11px] text-zinc-500 pt-1">
            <span>Status: <strong className="text-emerald-400">100% Sincronizado com o Código</strong></span>
            <span>CLI: v1.4.2</span>
          </div>
        </div>

        {/* Right: Technical Explanation */}
        <div className="lg:col-span-5 space-y-3">
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-3">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-[#34d399]">▹</span>
              <span>{lang === 'pt' ? 'Documentação Viva com Markdown Tags' : 'Living Documentation with Markdown Tags'}</span>
            </h4>
            <p className="text-zinc-300 font-sans text-xs leading-relaxed">
              {lang === 'pt'
                ? 'O MAD-cli garante que a arquitetura no README nunca fique defasada em relação ao código real. Ele analisa o AST dos arquivos TypeScript/Dart e regera os diagramas Mermaid nos blocos demarcados a cada commit.'
                : 'MAD-cli ensures README architectures never drift from reality. It parses TypeScript/Dart ASTs and regenerates Mermaid diagrams inside bounded comment blocks on every commit.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

/* =========================================================================
   4. MDDD-CLI (SPEC-DRIVEN DEV) DEMO
   ========================================================================= */
const MdddCliDemo: React.FC<{ lang?: 'en' | 'pt' }> = ({ lang }) => {
  const [specTab, setSpecTab] = useState<'diagram' | 'matrix'>('diagram');

  return (
    <div className="mt-6 space-y-6 font-mono text-xs">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
            <span className="font-bold text-white flex items-center gap-2">
              <Table className="w-4 h-4 text-[#adff2f]" />
              <span>mddd-cli Spec Engine</span>
            </span>
            <div className="flex gap-1 text-[11px]">
              <button
                type="button"
                onClick={() => setSpecTab('diagram')}
                className={`px-2.5 py-1 rounded cursor-pointer transition ${
                  specTab === 'diagram' ? 'bg-[#adff2f] text-black font-bold' : 'bg-zinc-950 text-zinc-400 hover:text-white border border-zinc-800'
                }`}
              >
                Diagrama Mermaid
              </button>
              <button
                type="button"
                onClick={() => setSpecTab('matrix')}
                className={`px-2.5 py-1 rounded cursor-pointer transition ${
                  specTab === 'matrix' ? 'bg-[#adff2f] text-black font-bold' : 'bg-zinc-950 text-zinc-400 hover:text-white border border-zinc-800'
                }`}
              >
                Matriz de Decisão
              </button>
            </div>
          </div>

          {specTab === 'diagram' ? (
            <div className="p-3 rounded-lg bg-black/80 border border-zinc-800 text-[11px] text-[#adff2f] space-y-1">
              <div>classDiagram</div>
              <div>&nbsp;&nbsp;direction TB</div>
              <div>&nbsp;&nbsp;class UserEntity &#123;</div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;+UUID id</div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;+String email</div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;+validatePolicy() bool</div>
              <div>&nbsp;&nbsp;&#125;</div>
              <div>&nbsp;&nbsp;class OrderService --&gt; UserEntity : requires</div>
            </div>
          ) : (
            <div className="p-3 rounded-lg bg-black/80 border border-zinc-800 text-[11px] text-zinc-300 space-y-1">
              <div className="text-zinc-500 font-bold">| Decisão Arquitetural | Alternativa Avaliada | Veredito | Justificativa Técnica |</div>
              <div className="text-white">| PagedAttention | Sparse Attention | Aprovado | Reduz VRAM em 4.2x sem perda de contexto |</div>
              <div className="text-white">| BPTT Recorrente | Feed-Forward Puro | Aprovado | Mantém memória temporal em jogos Gym |</div>
            </div>
          )}
        </div>

        <div className="lg:col-span-5 space-y-3">
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-3">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-[#adff2f]">▹</span>
              <span>{lang === 'pt' ? 'Desenvolvimento Orientado a Especificação' : 'Spec-Driven Development'}</span>
            </h4>
            <p className="text-zinc-300 font-sans text-xs leading-relaxed">
              {lang === 'pt'
                ? 'O mddd-cli compila especificações arquiteturais e diagramas de classe diretamente em boilerplate estruturado, interfaces e testes unitários com conformidade garantida.'
                : 'mddd-cli compiles architectural specs and class diagrams directly into verified boilerplates, interfaces, and unit tests.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
