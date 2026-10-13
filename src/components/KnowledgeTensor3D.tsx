import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import {
  Cpu,
  Layers,
  Code2,
  GraduationCap,
  Sparkles,
  RotateCcw,
  Play,
  Pause,
  Maximize2,
  Minimize2,
  Search,
  Compass,
  Zap,
  Info,
  ExternalLink,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { getSkillCategories } from '../data/portfolio-data';

interface KnowledgeTensor3DProps {
  lang?: 'en' | 'pt';
  onSelectSkill?: (skillName: string) => void;
}

export interface TensorNode {
  id: string;
  name: string;
  level: string;
  categoryIndex: number;
  categoryTitle: string;
  color: string;
  glowColor: string;
  position: [number, number, number];
  sphericalPosition: [number, number, number];
  cubePosition: [number, number, number];
  connections: string[];
}

export const KnowledgeTensor3D: React.FC<KnowledgeTensor3DProps> = ({
  lang = 'en',
  onSelectSkill,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [activeCluster, setActiveCluster] = useState<number | 'all'>('all');
  const [projectionMode, setProjectionMode] = useState<'latent' | 'spherical' | 'cube'>('latent');
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const isAutoRotatingRef = useRef(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<TensorNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<TensorNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Keep isAutoRotatingRef in sync with state
  useEffect(() => {
    isAutoRotatingRef.current = isAutoRotating;
  }, [isAutoRotating]);

  const skillCategories = useMemo(() => getSkillCategories(lang), [lang]);

  // Construct high-dimensional clustered nodes
  const nodes: TensorNode[] = useMemo(() => {
    const rawNodes: TensorNode[] = [];

    // Cluster Centroids in 3D Latent Space
    const centroids = [
      { x: -14, y: 6, z: -5, color: '#adff2f', glow: '#adff2f' }, // 0: ML & AI (Lime)
      { x: 14, y: 7, z: -6, color: '#38bdf8', glow: '#38bdf8' },  // 1: Architecture & Systems (Sky)
      { x: 0, y: -11, z: 8, color: '#c084fc', glow: '#c084fc' },  // 2: Languages & Tech (Violet)
      { x: 0, y: 15, z: 6, color: '#fbbf24', glow: '#fbbf24' },   // 3: Academic & Education (Gold)
    ];

    skillCategories.forEach((cat, catIdx) => {
      const centroid = centroids[catIdx] || { x: 0, y: 0, z: 0, color: '#ffffff', glow: '#ffffff' };
      const count = cat.skills.length;

      cat.skills.forEach((skill, sIdx) => {
        const id = `${catIdx}-${sIdx}-${skill.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}`;

        // Organic latent position around centroid
        const angle = (sIdx / count) * Math.PI * 2;
        const radius = 4 + (sIdx % 3) * 2.2;
        const elevation = ((sIdx % 4) - 1.5) * 2.5;

        const lx = centroid.x + Math.cos(angle) * radius;
        const ly = centroid.y + elevation;
        const lz = centroid.z + Math.sin(angle) * radius;

        // Spherical Manifold position (RoPE positional representation)
        const totalIdx = catIdx * 12 + sIdx;
        const phi = Math.acos(-1 + (2 * totalIdx) / 40);
        const theta = Math.sqrt(40 * Math.PI) * phi;
        const sphereRadius = 18;
        const sx = sphereRadius * Math.cos(theta) * Math.sin(phi);
        const sy = sphereRadius * Math.sin(theta) * Math.sin(phi);
        const sz = sphereRadius * Math.cos(phi);

        // Hypercube orthogonal tensor position
        const cx = ((catIdx % 2 === 0 ? -1 : 1) * 12) + ((sIdx % 3) - 1) * 4;
        const cy = (catIdx < 2 ? 1 : -1) * 10 + Math.floor(sIdx / 3) * 3 - 4;
        const cz = ((sIdx % 4) - 1.5) * 6;

        rawNodes.push({
          id,
          name: skill.name,
          level: skill.level || '',
          categoryIndex: catIdx,
          categoryTitle: cat.title,
          color: centroid.color,
          glowColor: centroid.glow,
          position: [lx, ly, lz],
          sphericalPosition: [sx, sy, sz],
          cubePosition: [cx, cy, cz],
          connections: [],
        });
      });
    });

    // Semantic Cross-Cluster & Intra-Cluster Synaptic Bridges
    const nodeMap = new Map(rawNodes.map((n) => [n.name.toLowerCase(), n]));

    const connect = (nameA: string, nameB: string) => {
      const a = Array.from(nodeMap.values()).find((n) => n.name.toLowerCase().includes(nameA.toLowerCase()));
      const b = Array.from(nodeMap.values()).find((n) => n.name.toLowerCase().includes(nameB.toLowerCase()));
      if (a && b && a.id !== b.id) {
        if (!a.connections.includes(b.id)) a.connections.push(b.id);
        if (!b.connections.includes(a.id)) b.connections.push(a.id);
      }
    };

    // Intra-cluster proximity loops
    rawNodes.forEach((node, i) => {
      // Connect to neighbor in same category
      const sameCat = rawNodes.filter((n) => n.categoryIndex === node.categoryIndex && n.id !== node.id);
      if (sameCat.length > 0) {
        const neighbor = sameCat[i % sameCat.length];
        if (!node.connections.includes(neighbor.id)) {
          node.connections.push(neighbor.id);
        }
      }
    });

    // Meaningful Cross-Cluster Synaptic Bridges
    connect('Causal LLMs', 'PyTorch & torchrun');
    connect('Causal LLMs', 'Grouped-Query');
    connect('Causal LLMs', 'Rotary Position');
    connect('PyTorch & torchrun', 'CUDA & Apple Metal');
    connect('PyTorch & torchrun', 'DistributedDataParallel');
    connect('RMSNorm', 'CUDA & Apple Metal');
    connect('Distributed Microservices', 'Go (Golang)');
    connect('Clean Architecture', 'Go (Golang)');
    connect('Clean Architecture', 'TypeScript');
    connect('Clean Architecture', 'Mermaid Diagram-Driven');
    connect('Multi-Agent AI', 'Autonomous Tool-Use');
    connect('Multi-Agent AI', 'Clean Architecture');
    connect('Data Science', 'Causal LLMs');
    connect('Data Science', 'Python (PyTorch');
    connect('Machine Learning & Deep Learning', 'PyTorch & torchrun');
    connect('Machine Learning & Deep Learning', 'CUDA & Apple Metal');
    connect('Go Programming', 'Go (Golang)');
    connect('Next.js Specialist', 'Next.js');
    connect('Dart / Flutter', 'C++ & Metal');
    connect('C++ & Metal', 'CUDA & Apple Metal');
    connect('Autonomous Tool-Use', 'Byte-Level BPE');

    return rawNodes;
  }, [skillCategories]);

  // Three.js Scene, Camera, Renderer references
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const nodeMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const linesGroupRef = useRef<THREE.Group | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Interaction State
  const mouseRef = useRef({ x: 0, y: 0, down: false, startX: 0, startY: 0 });
  const activePointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinchDistRef = useRef<number | null>(null);
  const dragStartPosRef = useRef({ x: 0, y: 0 });
  const hasDraggedRef = useRef(false);
  const rotationRef = useRef({ x: 0.15, y: 0.35, targetX: 0.15, targetY: 0.35 });
  const zoomRef = useRef({ dist: 38, targetDist: 38 });
  const cameraTargetRef = useRef(new THREE.Vector3(0, 0, 0));
  const currentCameraTargetRef = useRef(new THREE.Vector3(0, 0, 0));

  // Initialize Three.js WebGL Scene
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 520;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.fog = new THREE.FogExp2(0x07080a, 0.018);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    cameraRef.current = camera;
    camera.position.set(0, 5, zoomRef.current.dist);

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    rendererRef.current = renderer;
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xadff2f, 1.2);
    dirLight1.position.set(20, 30, 20);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 1.0);
    dirLight2.position.set(-20, -20, -20);
    scene.add(dirLight2);

    // Subtle 3D Coordinate Grid Plane
    const gridHelper = new THREE.GridHelper(48, 24, 0x1f2937, 0x111827);
    gridHelper.position.y = -16;
    scene.add(gridHelper);

    // Background Latent Particles
    const particleCount = 240;
    const particleGeometry = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 80;
      particlePositions[i + 1] = (Math.random() - 0.5) * 60;
      particlePositions[i + 2] = (Math.random() - 0.5) * 80;
    }
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMaterial = new THREE.PointsMaterial({
      color: 0xadff2f,
      size: 0.6,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
    });
    const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particleSystem);

    // Group for Knowledge Tensor Edges (Synapses)
    const linesGroup = new THREE.Group();
    linesGroupRef.current = linesGroup;
    scene.add(linesGroup);

    // Group for Tensor Node Meshes
    const nodesGroup = new THREE.Group();
    scene.add(nodesGroup);

    // Create Spheres for each Tensor Node
    const sphereGeometry = new THREE.SphereGeometry(0.85, 24, 24);
    const nodeMeshes = new Map<string, THREE.Mesh>();

    nodes.forEach((node) => {
      const color = new THREE.Color(node.color);
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.35,
        roughness: 0.2,
        metalness: 0.8,
      });

      const mesh = new THREE.Mesh(sphereGeometry, material);
      mesh.position.set(...node.position);
      mesh.userData = { nodeId: node.id, nodeData: node };

      // Outer Glow Halo Ring
      const haloGeo = new THREE.RingGeometry(1.1, 1.35, 16);
      const haloMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      const halo = new THREE.Mesh(haloGeo, haloMat);
      halo.lookAt(camera.position);
      mesh.add(halo);

      nodesGroup.add(mesh);
      nodeMeshes.set(node.id, mesh);
    });
    nodeMeshesRef.current = nodeMeshes;

    // Build Lines (Synaptic tensor edges)
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x52525b,
      transparent: true,
      opacity: 0.2,
      blending: THREE.AdditiveBlending,
    });

    const activeLineMaterial = new THREE.LineBasicMaterial({
      color: 0xadff2f,
      transparent: true,
      opacity: 0.8,
      linewidth: 2,
    });

    const drawnPairs = new Set<string>();
    nodes.forEach((node) => {
      node.connections.forEach((targetId) => {
        const pairKey = [node.id, targetId].sort().join('--');
        if (drawnPairs.has(pairKey)) return;
        drawnPairs.add(pairKey);

        const targetNode = nodes.find((n) => n.id === targetId);
        if (!targetNode) return;

        const points = [
          new THREE.Vector3(...node.position),
          new THREE.Vector3(...targetNode.position),
        ];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(lineGeo, lineMaterial);
        line.userData = { from: node.id, to: targetId, pairKey };
        linesGroup.add(line);
      });
    });

    // Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    const resizeObserver = new ResizeObserver(() => handleResize());
    resizeObserver.observe(container);

    // Non-passive native wheel listener to reliably prevent webpage scroll and allow smooth 3D zoom
    const handleNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      let dy = e.deltaY;
      if (e.deltaMode === 1) {
        dy *= 20; // lines
      } else if (e.deltaMode === 2) {
        dy *= 60; // pages
      }

      if (e.ctrlKey) {
        // macOS trackpad pinch gesture sends wheel with ctrlKey
        dy *= 2.0;
      }

      const zoomSpeed = 0.035;
      zoomRef.current.targetDist = Math.max(
        10,
        Math.min(75, zoomRef.current.targetDist + dy * zoomSpeed)
      );
    };

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });

    // Safari on macOS trackpad gesture handling
    const handleGestureStart = (e: Event) => e.preventDefault();
    const handleGestureChange = (e: any) => {
      e.preventDefault();
      if (typeof e.scale === 'number') {
        const factor = (1 - e.scale) * 15;
        zoomRef.current.targetDist = Math.max(
          10,
          Math.min(75, zoomRef.current.targetDist + factor)
        );
      }
    };
    canvas.addEventListener('gesturestart', handleGestureStart, { passive: false });
    canvas.addEventListener('gesturechange', handleGestureChange, { passive: false });

    // Animation Loop
    let clock = new THREE.Clock();

    const animate = () => {
      const delta = clock.getDelta();

      // Smooth Auto-Rotation if active and user not dragging
      if (isAutoRotatingRef.current && !mouseRef.current.down) {
        rotationRef.current.targetY += delta * 0.25;
      }

      // Smooth Camera Interpolation (Damping)
      rotationRef.current.x += (rotationRef.current.targetX - rotationRef.current.x) * 0.08;
      rotationRef.current.y += (rotationRef.current.targetY - rotationRef.current.y) * 0.08;
      zoomRef.current.dist += (zoomRef.current.targetDist - zoomRef.current.dist) * 0.08;

      currentCameraTargetRef.current.lerp(cameraTargetRef.current, 0.06);

      const radX = rotationRef.current.x;
      const radY = rotationRef.current.y;
      const dist = zoomRef.current.dist;

      camera.position.x = currentCameraTargetRef.current.x + dist * Math.sin(radY) * Math.cos(radX);
      camera.position.y = currentCameraTargetRef.current.y + dist * Math.sin(radX);
      camera.position.z = currentCameraTargetRef.current.z + dist * Math.cos(radY) * Math.cos(radX);
      camera.lookAt(currentCameraTargetRef.current);

      // Keep Halos facing the camera
      nodeMeshes.forEach((mesh) => {
        const halo = mesh.children[0];
        if (halo) halo.quaternion.copy(camera.quaternion);
      });

      // Slowly rotate particle dust
      particleSystem.rotation.y += delta * 0.02;

      renderer.render(scene, camera);
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      resizeObserver.disconnect();
      canvas.removeEventListener('wheel', handleNativeWheel);
      canvas.removeEventListener('gesturestart', handleGestureStart);
      canvas.removeEventListener('gesturechange', handleGestureChange);
      renderer.dispose();
      sphereGeometry.dispose();
      particleGeometry.dispose();
      particleMaterial.dispose();
      lineMaterial.dispose();
      activeLineMaterial.dispose();
    };
  }, [nodes]);

  // Update Node Positions based on Projection Mode (Latent vs Spherical vs Cube)
  useEffect(() => {
    const nodeMeshes = nodeMeshesRef.current;
    const linesGroup = linesGroupRef.current;
    if (!nodeMeshes || nodeMeshes.size === 0) return;

    nodes.forEach((node) => {
      const mesh = nodeMeshes.get(node.id);
      if (!mesh) return;

      const targetPos =
        projectionMode === 'latent'
          ? node.position
          : projectionMode === 'spherical'
          ? node.sphericalPosition
          : node.cubePosition;

      // Smoothly animate mesh positions
      mesh.position.set(...targetPos);
    });

    // Update line vertices
    if (linesGroup) {
      linesGroup.children.forEach((child) => {
        const line = child as THREE.Line;
        const fromNode = nodes.find((n) => n.id === line.userData.from);
        const toNode = nodes.find((n) => n.id === line.userData.to);
        if (!fromNode || !toNode) return;

        const p1 =
          projectionMode === 'latent'
            ? fromNode.position
            : projectionMode === 'spherical'
            ? fromNode.sphericalPosition
            : fromNode.cubePosition;

        const p2 =
          projectionMode === 'latent'
            ? toNode.position
            : projectionMode === 'spherical'
            ? toNode.sphericalPosition
            : toNode.cubePosition;

        line.geometry.setFromPoints([new THREE.Vector3(...p1), new THREE.Vector3(...p2)]);
      });
    }
  }, [projectionMode, nodes]);

  // Filter Highlight & Active Cluster update
  useEffect(() => {
    const nodeMeshes = nodeMeshesRef.current;
    const linesGroup = linesGroupRef.current;
    if (!nodeMeshes) return;

    const currentFocusId = selectedNode?.id || hoveredNode?.id;

    nodes.forEach((node) => {
      const mesh = nodeMeshes.get(node.id);
      if (!mesh) return;

      const matchesCluster = activeCluster === 'all' || node.categoryIndex === activeCluster;
      const matchesSearch =
        !searchQuery.trim() ||
        node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.level.toLowerCase().includes(searchQuery.toLowerCase());

      const isConnectedToFocus =
        currentFocusId &&
        (node.id === currentFocusId || node.connections.includes(currentFocusId));

      const isHighlighted = matchesCluster && matchesSearch;

      const mat = mesh.material as THREE.MeshStandardMaterial;
      const halo = mesh.children[0] as THREE.Mesh;
      const haloMat = halo?.material as THREE.MeshBasicMaterial;

      if (!isHighlighted) {
        // Dimmed node
        mesh.scale.set(0.6, 0.6, 0.6);
        mat.emissiveIntensity = 0.05;
        mat.opacity = 0.2;
        mat.transparent = true;
        if (haloMat) haloMat.opacity = 0.05;
      } else if (isConnectedToFocus) {
        // Connected to focused node
        mesh.scale.set(1.4, 1.4, 1.4);
        mat.emissiveIntensity = 1.0;
        mat.opacity = 1.0;
        mat.transparent = false;
        if (haloMat) haloMat.opacity = 0.7;
      } else {
        // Normal active node
        mesh.scale.set(1.0, 1.0, 1.0);
        mat.emissiveIntensity = 0.35;
        mat.opacity = 1.0;
        mat.transparent = false;
        if (haloMat) haloMat.opacity = 0.3;
      }
    });

    // Update lines highlight
    if (linesGroup) {
      linesGroup.children.forEach((child) => {
        const line = child as THREE.Line;
        const lineMat = line.material as THREE.LineBasicMaterial;
        const { from, to } = line.userData;

        const isLineActive =
          currentFocusId && (from === currentFocusId || to === currentFocusId);

        if (isLineActive) {
          lineMat.color.setHex(0xadff2f);
          lineMat.opacity = 0.9;
        } else {
          lineMat.color.setHex(0x52525b);
          lineMat.opacity = 0.15;
        }
      });
    }
  }, [activeCluster, searchQuery, hoveredNode, selectedNode, nodes]);

  const handleZoomIn = useCallback(() => {
    zoomRef.current.targetDist = Math.max(10, zoomRef.current.targetDist - 6);
  }, []);

  const handleZoomOut = useCallback(() => {
    zoomRef.current.targetDist = Math.min(75, zoomRef.current.targetDist + 6);
  }, []);

  const handleResetCamera = useCallback(() => {
    rotationRef.current.targetX = 0.15;
    rotationRef.current.targetY = 0.35;
    zoomRef.current.targetDist = 38;
    cameraTargetRef.current.set(0, 0, 0);
    setSelectedNode(null);
    setHoveredNode(null);
  }, []);

  // Keyboard navigation (+ / - for zoom, 0 or R for reset) when container is focused
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        handleZoomIn();
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        handleZoomOut();
      } else if (e.key === '0' || e.key === 'r' || e.key === 'R') {
        handleResetCamera();
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleZoomIn, handleZoomOut, handleResetCamera]);

  // Raycasting & Pointer Interaction Handlers
  const handlePointerDown = (e: React.PointerEvent) => {
    activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      // ignore if pointer capture fails
    }

    if (activePointersRef.current.size === 1) {
      mouseRef.current.down = true;
      mouseRef.current.startX = e.clientX;
      mouseRef.current.startY = e.clientY;
      dragStartPosRef.current = { x: e.clientX, y: e.clientY };
      hasDraggedRef.current = false;
    } else if (activePointersRef.current.size === 2) {
      // Multi-touch pinch start
      mouseRef.current.down = false;
      const points = Array.from(activePointersRef.current.values());
      pinchDistRef.current = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    const camera = cameraRef.current;
    const scene = sceneRef.current;
    if (!canvas || !camera || !scene) return;

    if (activePointersRef.current.has(e.pointerId)) {
      activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }

    // Two-finger pinch-to-zoom on touchscreens / trackpads
    if (activePointersRef.current.size === 2 && pinchDistRef.current !== null) {
      const points = Array.from(activePointersRef.current.values());
      const currentDist = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
      const diff = pinchDistRef.current - currentDist;
      zoomRef.current.targetDist = Math.max(
        10,
        Math.min(75, zoomRef.current.targetDist + diff * 0.12)
      );
      pinchDistRef.current = currentDist;
      hasDraggedRef.current = true;
      return;
    }

    // Single pointer dragging / orbiting
    if (mouseRef.current.down && activePointersRef.current.size === 1) {
      const deltaX = e.clientX - mouseRef.current.startX;
      const deltaY = e.clientY - mouseRef.current.startY;
      mouseRef.current.startX = e.clientX;
      mouseRef.current.startY = e.clientY;

      if (Math.hypot(e.clientX - dragStartPosRef.current.x, e.clientY - dragStartPosRef.current.y) > 5) {
        hasDraggedRef.current = true;
      }

      rotationRef.current.targetY -= deltaX * 0.007;
      rotationRef.current.targetX = Math.max(
        -Math.PI / 2.2,
        Math.min(Math.PI / 2.2, rotationRef.current.targetX - deltaY * 0.007)
      );
      return;
    }

    // Raycast on hover (when not dragging)
    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), camera);

    const meshes = Array.from(nodeMeshesRef.current.values());
    const intersects = raycaster.intersectObjects(meshes, false);

    if (intersects.length > 0) {
      const hitMesh = intersects[0].object as THREE.Mesh;
      const hitNode = hitMesh.userData.nodeData as TensorNode;
      setHoveredNode(hitNode);
      canvas.style.cursor = 'pointer';
    } else {
      setHoveredNode(null);
      canvas.style.cursor = 'grab';
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    activePointersRef.current.delete(e.pointerId);
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }

    if (activePointersRef.current.size === 0) {
      mouseRef.current.down = false;
      pinchDistRef.current = null;
    } else if (activePointersRef.current.size === 1) {
      const remaining = Array.from(activePointersRef.current.values())[0];
      mouseRef.current.startX = remaining.x;
      mouseRef.current.startY = remaining.y;
      pinchDistRef.current = null;
    }
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    // If user dragged to orbit or performed pinch zoom, ignore click
    if (hasDraggedRef.current) {
      hasDraggedRef.current = false;
      return;
    }

    const canvas = canvasRef.current;
    const camera = cameraRef.current;
    if (!canvas || !camera) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), camera);

    const meshes = Array.from(nodeMeshesRef.current.values());
    const intersects = raycaster.intersectObjects(meshes, false);

    if (intersects.length > 0) {
      const hitMesh = intersects[0].object as THREE.Mesh;
      const hitNode = hitMesh.userData.nodeData as TensorNode;
      setSelectedNode(hitNode);
      if (onSelectSkill) onSelectSkill(hitNode.name);

      // Lerp camera target to clicked node
      cameraTargetRef.current.set(...hitMesh.position.toArray());
      zoomRef.current.targetDist = Math.min(zoomRef.current.targetDist, 24);
    } else {
      setSelectedNode(null);
      // Preserve current user zoom distance! Do not reset zoomRef.current.targetDist to 38.
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    const camera = cameraRef.current;
    if (!canvas || !camera) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), camera);

    const meshes = Array.from(nodeMeshesRef.current.values());
    const intersects = raycaster.intersectObjects(meshes, false);

    if (intersects.length > 0) {
      const hitMesh = intersects[0].object as THREE.Mesh;
      const hitNode = hitMesh.userData.nodeData as TensorNode;
      setSelectedNode(hitNode);
      if (onSelectSkill) onSelectSkill(hitNode.name);
      cameraTargetRef.current.set(...hitMesh.position.toArray());
      zoomRef.current.targetDist = 18;
    } else {
      // Double click empty space: zoom in
      zoomRef.current.targetDist = Math.max(10, zoomRef.current.targetDist - 10);
    }
  };

  const activeFocusNode = selectedNode || hoveredNode;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      className={`relative w-full rounded-2xl bg-[#07080a] border border-zinc-800 shadow-2xl overflow-hidden font-mono select-none outline-none focus:border-zinc-700 transition-colors ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none' : 'h-[580px] lg:h-[640px]'
      }`}
    >
      {/* 3D WebGL Canvas */}
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onClick={handleCanvasClick}
        onDoubleClick={handleDoubleClick}
        className="w-full h-full block cursor-grab active:cursor-grabbing touch-none"
      />

      {/* Top Floating HUD: Title, Clusters, Projection Mode */}
      <div className="absolute top-3 sm:top-4 left-3 sm:left-4 right-3 sm:right-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 pointer-events-none">
        {/* Title and Telemetry Badge */}
        <div className="flex items-center gap-2 pointer-events-auto flex-wrap">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-950/90 border border-zinc-800 text-xs shadow-2xl backdrop-blur-md">
            <Sparkles className="w-3.5 h-3.5 text-[#adff2f]" />
            <span className="font-bold text-white tracking-wide">
              {lang === 'pt' ? 'TENSOR 3D DE COMPETÊNCIAS' : '3D KNOWLEDGE TENSOR'}
            </span>
            <span className="text-[10px] text-[#adff2f] bg-[#adff2f]/10 px-1.5 py-0.2 rounded font-bold">
              {nodes.length} nós
            </span>
          </div>

          {/* Projection Modes */}
          <div className="flex items-center p-0.5 rounded-lg bg-zinc-950/90 border border-zinc-800 text-[10px]">
            <button
              type="button"
              onClick={() => setProjectionMode('latent')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                projectionMode === 'latent' ? 'bg-zinc-800 text-[#adff2f] font-bold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Espaço Latente de Agrupamento Semântico' : 'Latent Semantic Proximity Space'}
            >
              {lang === 'pt' ? 'Latente' : 'Latent'}
            </button>
            <button
              type="button"
              onClick={() => setProjectionMode('spherical')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                projectionMode === 'spherical' ? 'bg-zinc-800 text-[#38bdf8] font-bold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Projeção Esférica RoPE' : 'Spherical RoPE Manifold'}
            >
              {lang === 'pt' ? 'Esfera RoPE' : 'RoPE Sphere'}
            </button>
            <button
              type="button"
              onClick={() => setProjectionMode('cube')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                projectionMode === 'cube' ? 'bg-zinc-800 text-purple-300 font-bold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Hipercubo Tensorial Ortogonal' : 'Orthogonal Tensor Grid'}
            >
              {lang === 'pt' ? 'Hipercubo' : 'Hypercube'}
            </button>
          </div>

          {/* Navigation Instruction Pill */}
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-zinc-950/90 border border-zinc-800 text-[10px] text-zinc-400 backdrop-blur-md">
            <Compass className="w-3 h-3 text-[#adff2f]" />
            <span>{lang === 'pt' ? 'Arraste p/ orbitar 360° · Scroll ou +/- p/ zoom' : 'Drag to orbit 360° · Scroll or +/- to zoom'}</span>
          </div>
        </div>

        {/* Camera Utilities & Search */}
        <div className="flex items-center gap-1.5 pointer-events-auto self-end sm:self-auto">
          {/* Quick Search Input */}
          <div className="relative">
            <Search className="w-3 h-3 text-zinc-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              maxLength={50}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={lang === 'pt' ? 'Buscar nó...' : 'Find node...'}
              className="w-24 sm:w-32 pl-7 pr-2 py-1 rounded-lg bg-zinc-950/90 border border-zinc-800 text-[10px] text-white placeholder-zinc-500 focus:outline-hidden focus:border-[#adff2f]/60 backdrop-blur-md"
            />
          </div>

          {/* Zoom In Button */}
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-1.5 rounded-lg bg-zinc-950/90 border border-zinc-800 text-zinc-400 hover:text-[#adff2f] hover:border-[#adff2f]/40 transition-colors cursor-pointer backdrop-blur-md"
            title={lang === 'pt' ? 'Aproximar Zoom (+) · Scroll p/ cima' : 'Zoom In (+) · Scroll up'}
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          {/* Zoom Out Button */}
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-1.5 rounded-lg bg-zinc-950/90 border border-zinc-800 text-zinc-400 hover:text-[#adff2f] hover:border-[#adff2f]/40 transition-colors cursor-pointer backdrop-blur-md"
            title={lang === 'pt' ? 'Afastar Zoom (-) · Scroll p/ baixo' : 'Zoom Out (-) · Scroll down'}
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          {/* Auto Rotate Toggle */}
          <button
            type="button"
            onClick={() => {
              setIsAutoRotating((prev) => {
                const next = !prev;
                isAutoRotatingRef.current = next;
                return next;
              });
            }}
            className={`p-1.5 rounded-lg border text-xs transition-colors cursor-pointer backdrop-blur-md ${
              isAutoRotating
                ? 'bg-zinc-950/90 border-[#adff2f]/40 text-[#adff2f]'
                : 'bg-zinc-950/90 border-zinc-800 text-zinc-500 hover:text-white'
            }`}
            title={lang === 'pt' ? 'Alternar Rotação Automática' : 'Toggle Auto-Rotation'}
          >
            {isAutoRotating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>

          {/* Reset Camera */}
          <button
            type="button"
            onClick={handleResetCamera}
            className="p-1.5 rounded-lg bg-zinc-950/90 border border-zinc-800 text-zinc-400 hover:text-white transition-colors cursor-pointer backdrop-blur-md"
            title={lang === 'pt' ? 'Resetar Câmera 3D (0 ou R)' : 'Reset Camera View (0 or R)'}
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Fullscreen Toggle */}
          <button
            type="button"
            onClick={() => setIsFullscreen((prev) => !prev)}
            className="p-1.5 rounded-lg bg-zinc-950/90 border border-zinc-800 text-zinc-400 hover:text-white transition-colors cursor-pointer backdrop-blur-md"
            title={isFullscreen ? (lang === 'pt' ? 'Sair da tela cheia' : 'Exit fullscreen') : (lang === 'pt' ? 'Tela cheia' : 'Fullscreen')}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Cluster Navigation Chips Bar */}
      <div className="absolute bottom-3 sm:bottom-4 left-3 sm:left-4 right-3 sm:right-4 z-10 flex flex-wrap items-center gap-1.5 pointer-events-auto">
        <button
          type="button"
          onClick={() => setActiveCluster('all')}
          className={`px-2.5 py-1 text-[10px] font-mono rounded-lg transition-colors cursor-pointer backdrop-blur-md ${
            activeCluster === 'all'
              ? 'bg-[#adff2f] text-black font-bold shadow-xs shadow-lime-400/30'
              : 'bg-zinc-950/90 text-zinc-400 hover:text-white border border-zinc-800'
          }`}
        >
          {lang === 'pt' ? 'Todos os Clusters' : 'All Clusters'} ({nodes.length})
        </button>

        {skillCategories.map((cat, idx) => {
          const isSelected = activeCluster === idx;
          const colors = ['#adff2f', '#38bdf8', '#c084fc', '#fbbf24'];
          const catColor = colors[idx] || '#ffffff';

          return (
            <button
              type="button"
              key={idx}
              onClick={() => {
                setActiveCluster(isSelected ? 'all' : idx);
                // Center on cluster centroid
                const centroids = [
                  [-14, 6, -5],
                  [14, 7, -6],
                  [0, -11, 8],
                  [0, 15, 6],
                ];
                if (!isSelected) {
                  cameraTargetRef.current.set(...(centroids[idx] as [number, number, number]));
                  zoomRef.current.targetDist = 26;
                } else {
                  cameraTargetRef.current.set(0, 0, 0);
                  zoomRef.current.targetDist = 38;
                }
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-mono rounded-lg transition-all cursor-pointer backdrop-blur-md ${
                isSelected
                  ? 'bg-zinc-900 text-white font-bold border'
                  : 'bg-zinc-950/90 text-zinc-400 hover:text-white border border-zinc-850'
              }`}
              style={{
                borderColor: isSelected ? catColor : undefined,
                boxShadow: isSelected ? `0 0 12px ${catColor}33` : undefined,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: catColor }} />
              <span>{cat.title}</span>
              <span className="text-[9px] opacity-60">({cat.skills.length})</span>
            </button>
          );
        })}
      </div>

      {/* Holographic Active Node Inspection HUD Card (Top Right / Floating) */}
      {activeFocusNode && (
        <div className="absolute top-16 right-3 sm:right-4 z-20 w-72 sm:w-80 p-3.5 rounded-2xl bg-zinc-950/95 border border-zinc-800 shadow-2xl backdrop-blur-xl animate-in fade-in zoom-in-95 duration-150 space-y-2.5 pointer-events-auto">
          {/* Header */}
          <div className="flex items-start justify-between border-b border-zinc-850 pb-2">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: activeFocusNode.color }} />
                <span className="text-[10px] uppercase font-bold tracking-wider" style={{ color: activeFocusNode.color }}>
                  {activeFocusNode.categoryTitle}
                </span>
              </div>
              <h4 className="text-sm font-bold text-white mt-0.5 leading-snug">{activeFocusNode.name}</h4>
            </div>

            {selectedNode && (
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="p-1 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-850 transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Desafixar' : 'Unpin'}
              >
                ×
              </button>
            )}
          </div>

          {/* Node Specs */}
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="p-2 rounded-lg bg-zinc-900/60 border border-zinc-850 space-y-0.5">
              <span className="text-zinc-500 uppercase tracking-wider">{lang === 'pt' ? 'Proficiência' : 'Proficiency'}</span>
              <div className="font-bold text-zinc-200 truncate">{activeFocusNode.level || 'Expert / Production'}</div>
            </div>

            <div className="p-2 rounded-lg bg-zinc-900/60 border border-zinc-850 space-y-0.5">
              <span className="text-zinc-500 uppercase tracking-wider">{lang === 'pt' ? 'Sinapses Tensoriais' : 'Tensor Synapses'}</span>
              <div className="font-bold text-[#adff2f]">{activeFocusNode.connections.length} {lang === 'pt' ? 'arestas' : 'edges'}</div>
            </div>
          </div>

          {/* Coordinate Vector */}
          <div className="p-2 rounded-lg bg-zinc-900/40 border border-zinc-850 text-[10px] space-y-0.5">
            <span className="text-zinc-500">{lang === 'pt' ? 'Vetor de Coordenadas Latentes' : 'Latent Tensor Embedding'}:</span>
            <div className="font-mono text-zinc-400 truncate">
              [{activeFocusNode.position.map((v) => v.toFixed(1)).join(', ')}]
            </div>
          </div>

          {/* Connected Knowledge Synapses Chips */}
          <div className="space-y-1">
            <span className="text-[10px] text-zinc-400 font-semibold flex items-center gap-1">
              <Zap className="w-3 h-3 text-[#adff2f]" />
              <span>{lang === 'pt' ? 'Conexões Próximas:' : 'Connected Synapses:'}</span>
            </span>

            <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto pr-1">
              {activeFocusNode.connections.map((targetId) => {
                const targetNode = nodes.find((n) => n.id === targetId);
                if (!targetNode) return null;

                return (
                  <button
                    key={targetId}
                    type="button"
                    onClick={() => {
                      setSelectedNode(targetNode);
                      cameraTargetRef.current.set(...targetNode.position);
                    }}
                    className="px-1.5 py-0.5 rounded text-[9px] bg-zinc-900 hover:bg-zinc-850 border border-zinc-800 hover:border-zinc-700 text-zinc-300 transition-colors cursor-pointer truncate max-w-[180px]"
                  >
                    {targetNode.name}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Helper hint */}
          <div className="text-[9px] text-zinc-500 pt-1 border-t border-zinc-850/80 flex items-center justify-between">
            <span>{selectedNode ? (lang === 'pt' ? 'Nó fixado' : 'Node pinned') : (lang === 'pt' ? 'Clique p/ fixar câmera' : 'Click to lock camera')}</span>
            <span className="text-zinc-600">{lang === 'pt' ? 'Arraste p/ orbitar · Scroll ou +/- p/ zoom' : 'Drag to orbit · Scroll or +/- to zoom'}</span>
          </div>
        </div>
      )}
    </div>
  );
};
