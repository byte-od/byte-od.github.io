import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Rotate3d, Compass } from 'lucide-react';
import { RubiksCubeSim } from '../lib/system1';

interface RubiksCube3DProps {
  state: Int32Array;
  lastAction: string;
  steps: number;
  alignedCount: number;
  isSolved: boolean;
  isRunning: boolean;
  lang?: 'en' | 'pt';
  scrambleSequence?: string;
}

// Mapeamento matemático de posições para índices de adesivos em RubiksCubeSim
const getSlotIndex = (pos: [number, number, number], norm: [number, number, number]): number => {
  const [x, y, z] = pos;
  const [nx, ny, nz] = norm;
  let face = 0, r = 0, c = 0;
  if (ny === 1) { face = 0; r = 1 - z; c = x + 1; }       // U (Branco)
  else if (ny === -1) { face = 1; r = z + 1; c = x + 1; }  // D (Amarelo)
  else if (nz === 1) { face = 2; r = 1 - y; c = x + 1; }   // F (Verde)
  else if (nz === -1) { face = 3; r = 1 - y; c = 1 - x; }  // B (Azul)
  else if (nx === 1) { face = 4; r = 1 - y; c = 1 - z; }   // R (Vermelho)
  else if (nx === -1) { face = 5; r = 1 - y; c = z + 1; }  // L (Laranja)
  return face * 9 + (r * 3 + c);
};

const COLOR_MAP: Record<number, number> = {
  0: 0xf5f5fa, // U: Branco gelo
  1: 0xfacc15, // D: Amarelo vibrante
  2: 0x22c55e, // F: Verde esmeralda
  3: 0x3b82f6, // B: Azul elétrico
  4: 0xef4444, // R: Vermelho rubi
  5: 0xf97316, // L: Laranja neon
};

const PLASTIC_COLOR = 0x111318; // Plástico preto fosco speedcube

export const RubiksCube3D: React.FC<RubiksCube3DProps> = ({
  state,
  lastAction,
  steps,
  alignedCount,
  isSolved,
  lang = 'en',
  scrambleSequence,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const cubeGroupRef = useRef<THREE.Group | null>(null);
  const cubiesRef = useRef<Array<{ mesh: THREE.Mesh; materials: THREE.MeshStandardMaterial[]; pos: [number, number, number] }>>([]);

  const isDraggingRef = useRef(false);
  const prevMousePos = useRef({ x: 0, y: 0 });
  const rotationVelocity = useRef({ x: 0, y: 0.003 });
  const [autoRotate, setAutoRotate] = useState(true);

  // Inicializa a cena 3D
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth || 480;
    const height = 320;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    camera.position.set(4.2, 3.6, 5.2);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    rendererRef.current = renderer;

    // Iluminação balanceada Cyber-Studio
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight1.position.set(5, 8, 6);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 1.0);
    dirLight2.position.set(-6, -4, -5);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xadff2f, 0.8, 10);
    pointLight.position.set(0, 4, 0);
    scene.add(pointLight);

    // Grupo do Cubo Mágico
    const cubeGroup = new THREE.Group();
    scene.add(cubeGroup);
    cubeGroupRef.current = cubeGroup;

    // Cria os 26 cubies (3x3x3 exceto o centro oco)
    const cubies: Array<{ mesh: THREE.Mesh; materials: THREE.MeshStandardMaterial[]; pos: [number, number, number] }> = [];
    const cubieSize = 0.94;
    const geom = new THREE.BoxGeometry(cubieSize, cubieSize, cubieSize);

    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
          if (x === 0 && y === 0 && z === 0) continue; // Pula o miolo interno invisível

          // Cria os 6 materiais para as faces: [+X, -X, +Y, -Y, +Z, -Z]
          const materials: THREE.MeshStandardMaterial[] = [];
          for (let f = 0; f < 6; f++) {
            materials.push(
              new THREE.MeshStandardMaterial({
                color: PLASTIC_COLOR,
                roughness: 0.35,
                metalness: 0.1,
              })
            );
          }

          const mesh = new THREE.Mesh(geom, materials);
          mesh.position.set(x, y, z);
          cubeGroup.add(mesh);

          cubies.push({ mesh, materials, pos: [x, y, z] });
        }
      }
    }
    cubiesRef.current = cubies;

    // Loop de Animação
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (cubeGroupRef.current) {
        if (!isDraggingRef.current && autoRotate) {
          cubeGroupRef.current.rotation.y += rotationVelocity.current.y;
          cubeGroupRef.current.rotation.x += rotationVelocity.current.x * 0.2;
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    // Redimensionamento reativo garantindo sincronia com transições de visibilidade
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      if (!w) return;
      cameraRef.current.aspect = w / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, height);
    };

    window.addEventListener('resize', handleResize);
    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined' && containerRef.current) {
      resizeObserver = new ResizeObserver(() => handleResize());
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      resizeObserver?.disconnect();
      geom.dispose();
      cubies.forEach((c) => c.materials.forEach((m) => m.dispose()));
      renderer.dispose();
    };
  }, []);

  // Atualiza as cores dos 54 adesivos em tempo real conforme o vetor `state` da inferência neural
  useEffect(() => {
    if (cubiesRef.current.length === 0 || state.length < 54) return;

    for (const { materials, pos } of cubiesRef.current) {
      const [x, y, z] = pos;

      // Face 0: +X (Right)
      if (x === 1) {
        const slot = getSlotIndex([1, y, z], [1, 0, 0]);
        const colorId = state[slot];
        materials[0].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[0].color.setHex(PLASTIC_COLOR);
      }

      // Face 1: -X (Left)
      if (x === -1) {
        const slot = getSlotIndex([-1, y, z], [-1, 0, 0]);
        const colorId = state[slot];
        materials[1].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[1].color.setHex(PLASTIC_COLOR);
      }

      // Face 2: +Y (Up)
      if (y === 1) {
        const slot = getSlotIndex([x, 1, z], [0, 1, 0]);
        const colorId = state[slot];
        materials[2].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[2].color.setHex(PLASTIC_COLOR);
      }

      // Face 3: -Y (Down)
      if (y === -1) {
        const slot = getSlotIndex([x, -1, z], [0, -1, 0]);
        const colorId = state[slot];
        materials[3].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[3].color.setHex(PLASTIC_COLOR);
      }

      // Face 4: +Z (Front)
      if (z === 1) {
        const slot = getSlotIndex([x, y, 1], [0, 0, 1]);
        const colorId = state[slot];
        materials[4].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[4].color.setHex(PLASTIC_COLOR);
      }

      // Face 5: -Z (Back)
      if (z === -1) {
        const slot = getSlotIndex([x, y, -1], [0, 0, -1]);
        const colorId = state[slot];
        materials[5].color.setHex(COLOR_MAP[colorId] ?? PLASTIC_COLOR);
      } else {
        materials[5].color.setHex(PLASTIC_COLOR);
      }
    }
  }, [state]);

  // Interatividade de mouse / toque para girar o cubo livremente em 3D
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    isDraggingRef.current = true;
    prevMousePos.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDraggingRef.current || !cubeGroupRef.current) return;

    const deltaX = e.clientX - prevMousePos.current.x;
    const deltaY = e.clientY - prevMousePos.current.y;
    prevMousePos.current = { x: e.clientX, y: e.clientY };

    cubeGroupRef.current.rotation.y += deltaX * 0.008;
    cubeGroupRef.current.rotation.x += deltaY * 0.008;

    // Limita o pitch vertical para não inverter
    cubeGroupRef.current.rotation.x = Math.max(-1.4, Math.min(1.4, cubeGroupRef.current.rotation.x));
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    isDraggingRef.current = false;
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // Ignora erro se não capturado
    }
  }, []);

  const resetView = () => {
    if (cubeGroupRef.current) {
      cubeGroupRef.current.rotation.set(0.3, -0.6, 0);
    }
  };

  const scorePct = isSolved
    ? '100.0'
    : Math.min(100, Math.max(0, (alignedCount / 54) * 100)).toFixed(1);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[320px] bg-[#07090e] rounded-xl overflow-hidden border border-zinc-800/80 select-none cursor-grab active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      title={lang === 'pt' ? 'Clique e arraste para inspecionar o Cubo em 3D' : 'Click and drag to orbit 3D cube'}
    >
      <canvas ref={canvasRef} className="w-full h-full block" />

      {/* HUD Overlay Superior: Controles de Câmera & Scramble */}
      <div className="absolute top-2.5 left-3 right-3 flex items-center justify-between pointer-events-none">
        <div className="pointer-events-auto">
          {scrambleSequence && steps === 0 && (
            <span className="px-2 py-0.5 rounded bg-zinc-900/80 border border-zinc-700 text-[10px] font-mono text-amber-300 shadow">
              🎲 {scrambleSequence}
            </span>
          )}
        </div>

        {/* Botão de resetar câmera e toggle rotação */}
        <div className="flex items-center gap-1.5 pointer-events-auto">
          <button
            type="button"
            onClick={() => setAutoRotate((prev) => !prev)}
            className={`p-1.5 rounded-lg border text-[10px] font-mono transition cursor-pointer ${
              autoRotate
                ? 'bg-cyan-950/80 border-cyan-700 text-cyan-300'
                : 'bg-zinc-900/80 border-zinc-700 text-zinc-400 hover:text-white'
            }`}
            title={lang === 'pt' ? 'Alternar auto-rotação da câmera' : 'Toggle auto-rotation'}
          >
            <Compass className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={resetView}
            className="p-1.5 rounded-lg bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 text-[10px] font-mono transition cursor-pointer"
            title={lang === 'pt' ? 'Centralizar ângulo da câmera' : 'Center camera angle'}
          >
            {lang === 'pt' ? 'Centralizar' : 'Center'}
          </button>
        </div>
      </div>

      {/* Status da Ação Inferior */}
      <div className="absolute bottom-2.5 left-3 right-3 flex items-center justify-between pointer-events-none text-xs font-mono">
        <div>
          {isSolved && (
            <div className="px-2.5 py-1 rounded-md border font-bold bg-emerald-950/90 border-emerald-500 text-emerald-300 shadow-lg">
              {lang === 'pt' ? '🏆 CUBO RESOLVIDO' : '🏆 CUBE SOLVED'}
            </div>
          )}
        </div>

        <span className="text-[9px] text-zinc-500 hidden sm:inline bg-zinc-950/60 px-2 py-0.5 rounded border border-zinc-850">
          {lang === 'pt' ? 'Arraste para girar a visão 3D' : 'Drag to orbit 3D view'}
        </span>
      </div>
    </div>
  );
};
