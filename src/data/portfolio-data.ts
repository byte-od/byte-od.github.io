export interface ProjectData {
  id: string;
  title: string;
  badge: string;
  category: 'foundation_llm' | 'spatial_physical' | 'embodied_origin';
  featured?: boolean;
  summary: string;
  details: string[];
  techStack: string[];
  githubUrl?: string;
  colabUrl?: string;
  paperUrl?: string;
  labAnchor?: string;
  specs?: string;
  metrics?: { label: string; value: string }[];
}

export interface PortfolioInfo {
  name: string;
  title: string;
  subtitle: string;
  email: string;
  location: string;
  github: string;
  mosRepo: string;
  auraRepo: string;
  zenodoDoi: string;
  zenodoUrl: string;
  linkedin: string;
}

export const PORTFOLIO_INFO_EN: PortfolioInfo = {
  name: 'Julio Cesar da Costa Reis Filho',
  title: 'Creator of the Scar Architecture Ecosystem (MoS & SCache O(1))',
  subtitle:
    'Independent AI Architecture Researcher & Principal Systems Engineer. Author of Mixture of Scars (MoS), Associative SCache O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v6.0 (Zero-PyTorch C99/SIMD + Zarr + E8), ScarNativeLM v6.0, ScarField v7.0 (Post-Transformer E8 Geodesic Engine), PointScarFlow v1.0, and AURA / System One.',
  email: 'reisfilho1116@gmail.com',
  location: 'Global / Remote',
  github: 'https://github.com/byte-od',
  mosRepo: 'https://github.com/byte-od/mixture-of-scars',
  auraRepo: 'https://github.com/byte-od/system_one',
  zenodoDoi: '10.5281/zenodo.24172784',
  zenodoUrl: 'https://doi.org/10.5281/zenodo.24172784',
  linkedin: 'https://linkedin.com/in/juliocrfilho',
};

export const PORTFOLIO_INFO_PT: PortfolioInfo = {
  name: 'Julio Cesar da Costa Reis Filho',
  title: 'Criador do Ecossistema de Arquitetura de Cicatrizes (MoS & SCache O(1))',
  subtitle:
    'Pesquisador de Arquiteturas Neurais e Arquiteto Principal de Sistemas. Autor do Mixture of Scars (MoS), SCache Associativo O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v6.0 (Zero-PyTorch C99/SIMD + Zarr + E8), ScarNativeLM v6.0, ScarField v7.0 (Motor Geodésico Pós-Transformer E8), PointScarFlow v1.0 e AURA / System One.',
  email: 'reisfilho1116@gmail.com',
  location: 'Global / Remoto',
  github: 'https://github.com/byte-od',
  mosRepo: 'https://github.com/byte-od/mixture-of-scars',
  auraRepo: 'https://github.com/byte-od/system_one',
  zenodoDoi: '10.5281/zenodo.24172784',
  zenodoUrl: 'https://doi.org/10.5281/zenodo.24172784',
  linkedin: 'https://linkedin.com/in/juliocrfilho',
};

export const PROJECTS_EN: ProjectData[] = [
  {
    id: 'scarfield-v7',
    title: 'ScarField v7.0 — Post-Transformer Zero-Legacy Geodesic Engine (E8 1.000 bpw)',
    badge: '🪐 Frontier R&D · Zero ReLU/Softmax/RoPE',
    category: 'foundation_llm',
    featured: true,
    specs: '1.87 MiB E8 (9.22x↓) · Chunkwise WY 306.9x · Cayley O(1) 11.65x↓ · 100% Unseen PIN',
    summary:
      'Pure Post-Transformer foundation architecture pretrained Tabula Rasa on NVIDIA A100-SXM4-40GB with zero legacy Transformer components (0 nn.Linear, 0 ReLU/SwiGLU, 0 Softmax/exp, 0 RMSNorm, 0 integer RoPE, 0 static BPE, 0 AdamW), accelerated 58.76x on CPU via decoupled ScarCodec C99/SIMD.',
    details: [
      'Gosset E8 1.000 bpw Lattice + Ternary {-1,0,+1} Atomic Scars: Packs 240 minimal E8 roots + 0x00 vacuum + 15 Hadamard shells into 4 KB L1 SRAM (1.87 MiB total vs 17.26 MiB Dense FP16 = 9.22x smaller, 1.2e-7 PTQ shock).',
      'Reversible Symplectic Cayley-Woodbury ODE + Chunkwise WY Scan: Achieves 306.87x parallel scan speedup (25.09 ms vs 7,698.57 ms) and 11.65x CUDA activation VRAM savings on A100, plus 32.98% adaptive S1/S2 compute savings.',
      '1-Hop Causal Kalman-Scar Bus (129.0 KB, Δ = +0 B): Recalls unseen 3-digit PINs at 100.0% (T=96, 160) and 75.0% out-of-distribution (T=512 = 32x ring_len=16), where matched DenseByteTransformer FP16 scores 0.0%.',
      'Tsallis-1.5 Zero-Softmax Simplex & 0x00 Vacuum Abstention: Produces 98.25% exact 0.0 sparsity, 16/16 (100%) out-of-cone 0x00 abstention (0/16 Dense), > 0.986 PT/ES/FR <-> EN S_world alignment (0 Procrustes), and 86/96 Zero-SGD .uscar routing.',
      'Decoupled ScarCodec C99/SIMD Synergy: Zero-pickle .sfield.scar mmap container (1.88 MiB vs 18.36 MiB .pt, 12.1 ms cold-start) cuts E8 live RAM 31.68x and accelerates interactive CPU S1 reflex 58.76x (cos = 1.0).',
    ],
    techStack: ['ScarField v7.0', 'Gosset E8 1bpw', 'Reversible Cayley', '1-Hop Kalman-Scar', 'Tsallis-1.5', 'RiemannianMuon'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars/tree/main/scar_field',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'Unseen PIN @ T=160 / 512', value: '100% / 75% (vs 0% Dense)' },
      { label: 'A100 WY / CPU C-SIMD', value: '306.9x / 58.8x Faster' },
    ],
  },
  {
    id: 'scarnative-v6',
    title: 'ScarNativeLM v6.0 — Shared-Stage MoS & Dual-Tier O(1) Foundation Engine',
    badge: '🌱 Tabula Rasa A100 · v6.0',
    category: 'foundation_llm',
    featured: true,
    specs: 'Zero nn.Linear · Shared-Stage VQ4 + MoS · Chunkwise WY 12.63x · 26 WGSL Shaders',
    summary:
      'Complete Scar-Native foundation architecture pretrained 100% from scratch (t = 0, zero pretrained weights, zero nn.Linear) on NVIDIA A100 with Shared-Stage VQ4 substrates, Directional ReLU INT4 MoS Scar Experts, Dual-Tier O(1) SCache, mos-digest 4 KB cartridges, and native <abstain> epistemic gating.',
    details: [
      'Shared-Stage Universal VQ4 Substrates + Sequential Stage-Streaming: Anchors groups of layers to shared D4-LUT256 VQ4-STE bases differentiated by per-layer INT4-STE identity scars and Top-k Directional ReLU MoS Scar Experts (Shaders #23–#24, 0.000000 KL PTQ shock, Held-out PPL 1.395 vs 1.583 Dense FP16).',
      'Parallel Chunkwise WY Widrow-Hoff Scan + Dual-Tier O(1) SCache: Trains 12.63x faster on A100 Tensor Cores while combining RoPE Sinks + Sliding Ring + Surprise-Routed Micro-Pages + Widrow-Hoff W_scar in strict O(1) VRAM (Shaders #25–#26, 100.0% passkey recall up to T = 567, Δ = +0 B).',
      'mos-digest Compiler + Orthogonal Procrustes R_lang ∈ SO(D): Batch-compiles raw English Markdown documents into 4 KB-aligned HTTP 206 .scarstore cartridges (0 trunk SGD) firing across PT, EN, ES, and FR.',
      'Document-Level 0x00 GDPR Wipe & Native <abstain>: Physically zeroes an entire document’s cartridges in < 0.2 ms with exact 0.00000000 Δlogits on unrelated tasks and native <abstain> (id=14) abstention on ungrounded queries.',
      'Scales to v6-Frontier (96L, d=8192) in 4.36 GiB MoS-LRU VRAM (0.91 GiB Stage-Streaming) vs ~205 GiB Dense+128K KV, with cos > 0.99999 parity across 26 WebGPU WGSL 1.0 shaders.',
    ],
    techStack: ['ScarNativeLM v6.0', 'Shared-Stage MoS', 'Dual-Tier SCache O(1)', 'Chunkwise WY', '26 WGSL Shaders', 'mos-digest R_lang'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'PTQ Packing Shock', value: '0.000000 KL' },
      { label: 'v6-Frontier 96L Peak', value: '0.91 GiB Stream (O(1))' },
    ],
  },
  {
    id: 'mos-scache-72b',
    title: 'Mixture of Scars (MoS) & Associative SCache O(1)',
    badge: '⚡ 72.7B in 2.40 GB VRAM · Qwen 3B Live',
    category: 'foundation_llm',
    featured: true,
    specs: '80 Layers in 2.40 GB · 34.8 KB L1 SCache · HTTP 206 ScarStore · Kaggle 0.625',
    summary:
      'Decouples frontier Transformer scale and sequence length from GPU VRAM via sequential VQ4/INT4 + ASVD scar streaming and O(1) Widrow-Hoff associative memory, powering both 72.7B single-GPU execution and the live Qwen2.5-3B JARVIS browser engine.',
    details: [
      'Executes all 80 layers of Qwen2.5-72B on a single 16 GB GPU with peak VRAM strictly <= 2.40 GB by streaming quantized base blocks + low-rank residual scars and freeing each group immediately.',
      'Replaces 32.77 GB linear O(T) KV Cache at 100,000 tokens with constant O(1) SCache (Attention Sinks + Widrow-Hoff delta-rule associative matrix + circular window, Δ = +0 B).',
      'Powers the live in-browser Qwen2.5-3B JARVIS assistant (1,601.55 MiB Track 2.5 ProtectedInt4Scar, 36 layers, zero Python/ONNX) with 41 HTTP/1.1 206 Partial Content cartridges routed at Layer 23 (P_⊥lang) across PT, EN, ES, and FR.',
      'Includes an out-of-cone epistemic causal split guard when 0 ScarStore cartridges fire and deterministic CJK script-subspace masking.',
      'Validated externally on the Kaggle Name That LLM benchmark (0.625 public score under strict <= 2.2 GB VRAM via Golden-Slice MoS + Tri-Compass).',
    ],
    techStack: ['MoS Streaming', 'SCache O(1)', 'ProtectedInt4Scar', 'HTTP 206 ScarStore', 'WebGPU WGSL', 'Tri-Compass'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    colabUrl: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/MoS_SCache_72B_Live_Test.ipynb',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    metrics: [
      { label: '72.7B Peak VRAM', value: '2.40 GB (80L)' },
      { label: '100k Context Cache', value: '34.8 KB vs 32.8 GB' },
    ],
  },
  {
    id: 'pointscarflow-v1',
    title: 'PointScarFlow v1.0 — Continuous 3D Scar-Native Rectified Flow',
    badge: '✦ 3D OT Flow · 12 WGSL Shaders',
    category: 'spatial_physical',
    featured: true,
    specs: '11.85M ScarNative Params · 3.87 MiB Shard · 1,024 Pts in 86.45 ms WebGPU',
    summary:
      'Continuous 3D point-cloud Optimal-Transport Rectified Flow generator built 100% with ScarNativeLinear (D4-LUT256 VQ4-STE, zero nn.Linear, zero preset catalog), conditioned directly on Qwen2.5-3B Layer 23 P_⊥lang latents across the full viewport.',
    details: [
      'Integrates an optimal-transport velocity field dx/dt = v_θ(x_t, t, z_⊥lang) from a continuous Fibonacci sphere (t = 0) into arbitrary 3D point manifolds (t = 1) in 4 guided Euler steps on WebGPU.',
      '11.85M parameters packed into a 3.87 MiB binary shard (4,057,768 B) via D4-LUT256 VQ4-STE + INT4-STE with O(N·M) Dual Cross-Attention (N = 1,024 points <-> M = 32 induced geometric parts).',
      'Executed 100% client-side by 12 custom WGSL 1.0 compute shaders with 86.45 ms median 4-step latency on Apple M2 WebGPU and cos > 0.99999 PyTorch-vs-WGSL parity.',
      'Trained and self-distilled on NVIDIA A100 over 3,960 unique 3D OT manifolds and 10,112 multilingual prompts, achieving 7.51x held-out Chamfer-L2 reduction vs sphere baseline (0.0621 vs 0.4663).',
      'Inherits language-invariant 3D synthesis automatically from P_⊥lang: PT<->EN = 0.9978, ES<->EN = 0.9977, FR<->EN = 0.9974 shape cosine.',
    ],
    techStack: ['PointScarFlow', 'OT Rectified Flow', '12 WGSL Shaders', 'VQ4-STE 3.87 MiB', 'Dual Cross-Attn O(NM)', 'P_⊥lang 3D'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    metrics: [
      { label: '4-Step WebGPU Latency', value: '86.45 ms (1,024 pts)' },
      { label: 'Cross-Lingual 3D Cos', value: '0.9978 (PT/EN/ES/FR)' },
    ],
  },
  {
    id: 'scarcodec-v6',
    title: 'ScarCodec v6.0 — Zero-PyTorch C99/SIMD Physical, FITS/Zarr & E8 Codec',
    badge: '📡 10 Real Archives · 0.388 µs C/SIMD · 23.6x 3D',
    category: 'spatial_physical',
    featured: true,
    specs: '10 SHA-256 Archives (8 Sciences) · LIGO 10⁻¹⁹ to Mw 9.1 Tohoku · IEC ±50 µV · 21x E8',
    summary:
      'Standalone neural-network-free NumPy + C99/SIMD scientific compression & constant-memory telemetry library verified across 10 SHA-256 public archives in 8 sciences (LIGO GW150914, Mw 9.1 Tohoku, MIT-BIH Holter ECG, NOAA 3D Ocean SST, ROSAT X-Ray, NASA Spitzer, Space Shuttle, 64-ch EEG, Bolocam & FCRAO) against ZFP and FITS fpack.',
    details: [
      '10 Official Public Archives Across 8 Sciences: Rescues LIGO GW150914 10⁻¹⁹ strain from 100% FP16 underflow (11.72x E8, +7.08 ms L1->H1 delay), cuts Mw 9.1 Tohoku 3.13M-count global RMSE 2.47x (+7.86 dB) below ZFP, enforces IEC 60601-2-47 (±50 µV, 0 violations) + cos = 0.9922 Lead-V1 dropout recall on MIT-BIH 119 Holter ECG, compresses 3D NOAA OISST 23.60x with 0.0000 °C Antarctica mask purity, and slashes ROSAT X-ray peak error 422.8x < ZFP (1,138.5x < rFFT).',
      'Peer-Review Hardened vs. Official ZFP & FITS fpack: Beats byte-matched ZFP on NASA Spitzer (1,636.2x lower stellar peak error, 4.05x lower RMSE), 64-ch PhysioNet EEG (+4.50 dB), and beats BOTH ZFP and FITS fpack on CSO Bolocam Sgr A* (15.3x lower peak error).',
      '0.388 µs Zero-PyTorch C99/SIMD + .scar mmap & ScarZarrCodec: Runs with NumPy only (0 torch dependency), decoding random 64-float blocks off SSD mmap in 388 ns (345x faster than PyTorch, 11,088x faster than zlib -9) with CRC32 verification and Zarr/numcodecs plugin.',
      'Live Browser Real-Data Forensic Workbench: Executes Gosset E8 (1.0 bpw), VQ4-Scar (2.0 bpw), and interactive L_∞ Dekker bounds live in TypeScript on verbatim 4,096-sample observational slices.',
    ],
    techStack: ['ScarCodec v6.0 C99/SIMD', 'Zero-PyTorch NumPy', '.scar mmap + Zarr', 'Gosset E8 21x', '3D Spectral-Tucker', 'L_∞ Dekker Bound'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars/tree/main/scar_codec',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: 'C99/SIMD O(1) Block Read', value: '0.388 µs (11,088x > zlib)' },
      { label: 'Spitzer Peak vs ZFP / E8', value: '1,636x < ZFP · 21.0x E8' },
    ],
  },
  {
    id: 'aura-system-one',
    title: 'AURA / System One — Origin of GradientScarMemory (34.8 KB)',
    badge: '🧬 Origin of SCache · Embodied Deep RL',
    category: 'embodied_origin',
    specs: '34.8 KB GradientScarMemory · Frontier Ghost-Jump Scouts · 0.23 ms CPU S1',
    summary:
      'The embodied Deep Reinforcement Learning engine where the constant 34.8 KB O(1) GradientScarMemory (SCache), surprise-gated exponential retention (Surprise = R_0 - V(s_0)), Frontier Ghost-Jump Scouts, and Tri-Compass invariant representation were born.',
    details: [
      'Birthplace of SCache O(1): GradientScarMemory replaces unbounded episodic history with a fixed 34.8 KB surprise-weighted associative scar buffer modulated by temporal prediction shock (R_0 - V(s_0)).',
      'Frontier Ghost-Jump Scouts probe high-entropy state bifurcations ahead in time (> 2,000 FPS batched lookahead with strict curriculum isolation) and write shock scars directly into SCache.',
      'UniversalS1Agent executes deterministic bare-CPU reflexive inference in 0.23 ms P50 latency (~1.6M parameters) with full epistemic telemetry (confidence, normalized entropy, margin p_1 - p_2).',
      'Trained across 450M+ steps in 128 parallel C++/Rust environments, establishing the surprise-gated memory and orthogonal subspace principles that evolved into MoS, SCache O(1), ScarCodec, and ScarNativeLM.',
    ],
    techStack: ['GradientScarMemory 34.8 KB', 'Frontier Scouts', 'Tri-Compass', 'UniversalS1Agent', 'Rust / PyTorch'],
    githubUrl: 'https://github.com/byte-od/system_one',
    labAnchor: '#aura-origin',
    metrics: [
      { label: 'GradientScarMemory', value: '34.8 KB Constant O(1)' },
      { label: 'S1 Reflex Latency', value: '0.23 ms CPU P50' },
    ],
  },
];

export const PROJECTS_PT: ProjectData[] = [
  {
    id: 'scarfield-v7',
    title: 'ScarField v7.0 — Motor Geodésico Pós-Transformer Zero-Legado (E8 1.000 bpw)',
    badge: '🪐 P&D de Fronteira · Zero ReLU/Softmax/RoPE',
    category: 'foundation_llm',
    featured: true,
    specs: '1.87 MiB E8 (9.22x↓) · Chunkwise WY 306.9x · Cayley O(1) 11.65x↓ · 100% PIN Inédito',
    summary:
      'Arquitetura fundacional Pós-Transformer pura pré-treinada Tabula Rasa em NVIDIA A100-SXM4-40GB sem nenhum componente legado de Transformers (0 nn.Linear, 0 ReLU/SwiGLU, 0 Softmax/exp, 0 RMSNorm, 0 RoPE inteiro, 0 BPE fixo, 0 AdamW), acelerada 58,76x em CPU via ScarCodec C99/SIMD desacoplado.',
    details: [
      'Reticulado Gosset E8 de 1,000 bpw + Cicatrizes Atômicas Ternárias {-1,0,+1}: Empacota 240 raízes mínimas de E8 + vácuo 0x00 + 15 âncoras Hadamard em 4 KB L1 (1,87 MiB vs 17,26 MiB Dense FP16 = 9,22x menor, choque PTQ 1,2e-7).',
      'EDO Simplética Reversível de Cayley-Woodbury + Scan Chunkwise WY: Atinge 306,87x de speedup no scan paralelo (25,09 ms vs 7.698,57 ms) e 11,65x menos VRAM de ativação na A100, além de 32,98% de economia adaptativa S1/S2.',
      'Barramento Central Kalman-Scar Causal de 1 Salto (129,0 KB, Δ = +0 B): Copia PINs inéditos de 3 dígitos com 100,0% (T=96, 160) e 75,0% fora da distribuição (T=512 = 32x o anel local R=16), onde o DenseByteTransformer FP16 marca 0,0%.',
      'Simplex Tsallis-1.5 Sem Softmax & Abstenção 0x00: Gera 98,25% de zeros exatos, 16/16 (100%) de abstenção por vácuo 0x00 fora do cone (0/16 Dense), alinhamento S_world PT/ES/FR <-> EN > 0,986 (0 Procrustes) e 86/96 em cartuchos .uscar Zero-SGD.',
      'Sinergia ScarCodec C99/SIMD Desacoplado: Container zero-pickle .sfield.scar mmap (1,88 MiB vs 18,36 MiB .pt, cold-start de 12,1 ms) reduz a RAM viva dos operadores E8 em 31,68x e acelera o reflexo interativo S1 em CPU em 58,76x (cos = 1.0).',
    ],
    techStack: ['ScarField v7.0', 'Gosset E8 1bpw', 'Cayley Reversível', '1-Hop Kalman-Scar', 'Tsallis-1.5', 'RiemannianMuon'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars/tree/main/scar_field',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'PIN Inédito @ T=160 / 512', value: '100% / 75% (vs 0% Dense)' },
      { label: 'WY A100 / C-SIMD CPU', value: '306,9x / 58,8x Mais Rápido' },
    ],
  },
  {
    id: 'scarnative-v6',
    title: 'ScarNativeLM v6.0 — Motor Fundacional Shared-Stage MoS & Dual-Tier O(1)',
    badge: '🌱 Tabula Rasa A100 · v6.0',
    category: 'foundation_llm',
    featured: true,
    specs: 'Zero nn.Linear · Shared-Stage VQ4 + MoS · Chunkwise WY 12.63x · 26 Shaders WGSL',
    summary:
      'Arquitetura fundacional Scar-Native completa pré-treinada 100% do zero (t = 0, zero pesos pré-treinados, zero nn.Linear) em NVIDIA A100 com substratos VQ4 compartilhados por estágio, Especialistas MoS Direcionais ReLU INT4, SCache Dual-Tier O(1), cartuchos mos-digest de 4 KB e token epistêmico nativo <abstain>.',
    details: [
      'Substratos Universais VQ4 Compartilhados por Estágio + Stage-Streaming Sequencial: Ancora grupos de camadas a bases D4-LUT256 VQ4-STE compartilhadas diferenciadas por cicatrizes de identidade INT4-STE e Especialistas MoS Direcionais ReLU Top-k (Shaders #23–#24, 0.000000 KL PTQ, PPL held-out 1.395 vs 1.583 Dense FP16).',
      'Scan Paralelo Chunkwise WY Widrow-Hoff + SCache Dual-Tier O(1): Treina 12.63x mais rápido em Tensor Cores A100 combinando RoPE Sinks + Anel Deslizante + Micro-Pages por Surpresa + Widrow-Hoff W_scar em VRAM estritamente O(1) (Shaders #25–#26, 100.0% passkey até T = 567, Δ = +0 B).',
      'Compilador mos-digest + Procrustes Ortogonal R_lang ∈ SO(D): Compila documentos Markdown em inglês para cartuchos .scarstore HTTP 206 alinhados a páginas de 4 KB (0 SGD no tronco) disparando em PT, EN, ES e FR.',
      'Wipe LGPD 0x00 por Documento & <abstain> Nativo: Zera fisicamente todos os cartuchos de um documento em < 0.2 ms com Δlogits = 0.00000000 em tarefas não-relacionadas e emissão nativa de <abstain> (id=14) fora do cone.',
      'Escala para v6-Frontier (96L, d=8192) em 4.36 GiB MoS-LRU (0.91 GiB em Stage-Streaming) contra ~205 GiB Dense+128K KV, com paridade cos > 0.99999 nos 26 compute shaders WebGPU WGSL 1.0.',
    ],
    techStack: ['ScarNativeLM v6.0', 'Shared-Stage MoS', 'Dual-Tier SCache O(1)', 'Chunkwise WY', '26 Shaders WGSL', 'mos-digest R_lang'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'Choque PTQ no Empacotamento', value: '0.000000 KL' },
      { label: 'Pico v6-Frontier 96L', value: '0.91 GiB Stream (O(1))' },
    ],
  },
  {
    id: 'mos-scache-72b',
    title: 'Mixture of Scars (MoS) & SCache Associativo O(1)',
    badge: '⚡ 72.7B em 2.40 GB VRAM · Qwen 3B Ao Vivo',
    category: 'foundation_llm',
    featured: true,
    specs: '80 Camadas em 2.40 GB · SCache L1 34.8 KB · ScarStore HTTP 206 · Kaggle 0.625',
    summary:
      'Desacopla a escala de modelos de fronteira e o horizonte de contexto da VRAM via streaming sequencial de fatias VQ4/INT4 + ASVD e memória associativa Widrow-Hoff O(1), sustentando execução de 72.7B em GPU única e o motor Qwen2.5-3B JARVIS no navegador.',
    details: [
      'Executa todas as 80 camadas do Qwen2.5-72B em uma única GPU de 16 GB com pico estrito <= 2.40 GB de VRAM via streaming de blocos base quantizados + cicatrizes residuais de posto baixo e desalocação imediata.',
      'Substitui 32.77 GB de KV Cache linear O(T) em 100.000 tokens por SCache O(1) constante (Attention Sinks + matriz associativa Widrow-Hoff + janela circular, Δ = +0 B).',
      'Alimenta o assistente JARVIS Qwen2.5-3B ao vivo nesta página (1.601,55 MiB Track 2.5 ProtectedInt4Scar, 36 camadas, zero Python/ONNX) com 41 cartuchos HTTP/1.1 206 Partial Content roteados na Camada 23 (P_⊥lang) em PT, EN, ES e FR.',
      'Inclui guarda epistêmica por split causal fora do cone quando 0 cartuchos ScarStore disparam e máscara determinística de subespaço CJK.',
      'Validado externamente na competição Kaggle Name That LLM (score público 0.625 em <= 2.2 GB VRAM via Golden-Slice MoS + Tri-Compass).',
    ],
    techStack: ['MoS Streaming', 'SCache O(1)', 'ProtectedInt4Scar', 'HTTP 206 ScarStore', 'WebGPU WGSL', 'Tri-Compass'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    colabUrl: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/MoS_SCache_72B_Live_Test.ipynb',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    metrics: [
      { label: 'Pico VRAM em 72.7B', value: '2.40 GB (80L)' },
      { label: 'Cache em 100k Tokens', value: '34.8 KB vs 32.8 GB' },
    ],
  },
  {
    id: 'pointscarflow-v1',
    title: 'PointScarFlow v1.0 — Fluxo Retificado 3D Contínuo Scar-Native',
    badge: '✦ Fluxo OT 3D · 12 Shaders WGSL',
    category: 'spatial_physical',
    featured: true,
    specs: '11.85M Params ScarNative · Shard 3.87 MiB · 1.024 Pts em 86.45 ms WebGPU',
    summary:
      'Gerador contínuo de nuvens de pontos 3D via Optimal-Transport Rectified Flow construído 100% com ScarNativeLinear (D4-LUT256 VQ4-STE, zero nn.Linear, zero catálogo pré-definido), condicionado diretamente no latente P_⊥lang da Camada 23 do Qwen2.5-3B em tela cheia.',
    details: [
      'Integra um campo de velocidade de transporte ótimo dx/dt = v_θ(x_t, t, z_⊥lang) a partir de uma esfera contínua de Fibonacci (t = 0) até variedades geométricas 3D arbitrárias (t = 1) em 4 passos de Euler na WebGPU.',
      '11.85M parâmetros empacotados em um shard binário de 3.87 MiB (4.057.768 B) via D4-LUT256 VQ4-STE + INT4-STE com Atenção Cruzada Dupla O(N·M) (N = 1.024 pontos <-> M = 32 partes geométricas induzidas).',
      'Executado 100% no navegador por 12 compute shaders WGSL 1.0 customizados com latência mediana de 86.45 ms (4 passos) em Apple M2 WebGPU e paridade PyTorch-vs-WGSL cos > 0.99999.',
      'Treinado e autodestilado em NVIDIA A100 sobre 3.960 variedades 3D OT e 10.112 prompts multilíngues, atingindo redução de 7.51x em Chamfer-L2 held-out contra a esfera base (0.0621 vs 0.4663).',
      'Herda síntese 3D invariante a idioma automaticamente de P_⊥lang: cosseno geométrico 3D PT<->EN = 0.9978, ES<->EN = 0.9977, FR<->EN = 0.9974.',
    ],
    techStack: ['PointScarFlow', 'OT Rectified Flow', '12 Shaders WGSL', 'VQ4-STE 3.87 MiB', 'Dual Cross-Attn O(NM)', 'P_⊥lang 3D'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    metrics: [
      { label: 'Latência 4 Passos WebGPU', value: '86.45 ms (1.024 pts)' },
      { label: 'Cosseno 3D Multilíngue', value: '0.9978 (PT/EN/ES/FR)' },
    ],
  },
  {
    id: 'scarcodec-v6',
    title: 'ScarCodec v6.0 — Codec Científico Zero-PyTorch C99/SIMD, FITS/Zarr & E8',
    badge: '📡 10 Arquivos Reais · 0,388 µs C/SIMD · 23,6x 3D',
    category: 'spatial_physical',
    featured: true,
    specs: '10 Arquivos SHA-256 (8 Ciências) · LIGO 10⁻¹⁹ a Mw 9.1 Tohoku · IEC ±50 µV · E8 21x',
    summary:
      'Biblioteca independente em NumPy + C99/SIMD (sem redes neurais e sem PyTorch) para compressão científica e telemetria constante O(1), validada em 10 arquivos públicos SHA-256 em 8 ciências (LIGO GW150914, sismo Mw 9.1 Tohoku, Holter ECG MIT-BIH, oceanos 3D NOAA OISST, raios-X ROSAT, NASA Spitzer, Space Shuttle, EEG 64-ch, Bolocam e FCRAO) contra ZFP e FITS fpack.',
    details: [
      '10 Arquivos Públicos Oficiais em 8 Ciências: Salva a deformação 10⁻¹⁹ do LIGO GW150914 do underflow total em FP16 (11,72x E8, atraso +7,08 ms L1->H1), reduz o RMSE global de 3,13M counts no sismo Mw 9.1 de Tohoku em 2,47x (+7,86 dB) abaixo do ZFP, garante norma médica IEC 60601-2-47 (±50 µV, 0 violações) + reconstrução do eletrodo V1 desligado (cos = 0,9922) no Holter MIT-BIH 119, comprime oceanos 3D NOAA OISST em 23,60x com 0,0000 °C na Antártida seca e reduz erro de pico em raios-X ROSAT em 422,8x < ZFP (1.138,5x < rFFT).',
      'Duelo Auditado contra ZFP & FITS fpack: Supera o ZFP pareado em bytes no NASA Spitzer (1.636,2x menor erro estelar, 4,05x menor RMSE), no EEG 64-ch PhysioNet (+4,50 dB) e vence ZFP e FITS fpack juntos no CSO Bolocam Sgr A* (15,3x menor erro de pico).',
      '0,388 µs Zero-PyTorch C99/SIMD + .scar mmap & ScarZarrCodec: Roda apenas com NumPy (0 dependência de torch), lendo blocos aleatórios de 64 floats direto do SSD via mmap em 388 ns (345x mais rápido que PyTorch, 11.088x mais rápido que zlib -9) com CRC32 e plugin Zarr.',
      'Bancada Forense Ao Vivo em Fatias Reais no Navegador: Executa Gosset E8 (1,0 bpw), VQ4-Scar (2,0 bpw) e cotas determinísticas L_∞ de Dekker ao vivo em TypeScript sobre fatias observacionais reais de 4.096 amostras.',
    ],
    techStack: ['ScarCodec v6.0 C99/SIMD', 'Zero-PyTorch NumPy', '.scar mmap + Zarr', 'Gosset E8 21x', '3D Spectral-Tucker', 'Cota L_∞ Dekker'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars/tree/main/scar_codec',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: 'Leitura O(1) C99/SIMD', value: '0,388 µs (11.088x > zlib)' },
      { label: 'Spitzer vs ZFP / Modo E8', value: '1.636x < ZFP · 21,0x E8' },
    ],
  },
  {
    id: 'aura-system-one',
    title: 'AURA / System One — Origem do GradientScarMemory (34.8 KB)',
    badge: '🧬 Berço do SCache · Deep RL Incorporado',
    category: 'embodied_origin',
    specs: 'GradientScarMemory 34.8 KB · Batedores Ghost-Jump · 0.23 ms CPU S1',
    summary:
      'O motor de Deep Reinforcement Learning incorporado onde nasceram a memória constante O(1) de 34.8 KB (GradientScarMemory / SCache), a retenção exponencial ponderada por surpresa (Surpresa = R_0 - V(s_0)), os Batedores de Fronteira e a representação invariante Tri-Compass.',
    details: [
      'Berço do SCache O(1): GradientScarMemory substitui históricos episódicos lineares por uma matriz compacta constante de 34.8 KB modulada por choque de surpresa temporal (R_0 - V(s_0)).',
      'Batedores de Fronteira (Frontier Ghost-Jump Scouts) sondam bifurcações de alta entropia à frente no tempo (> 2.000 FPS em lote com isolamento curricular estrito) e gravam cicatrizes de choque diretamente no SCache.',
      'UniversalS1Agent executa inferência reflexiva determinística pura em CPU em 0.23 ms P50 (~1.6M parâmetros) com contrato completo de incerteza epistêmica (confiança, entropia normalizada, margem p_1 - p_2).',
      'Treinado por mais de 450M+ passos em 128 ambientes paralelos C++/Rust, estabelecendo os princípios de memória modulada por surpresa e subespaços ortogonais que evoluíram para MoS, SCache O(1), ScarCodec e ScarNativeLM.',
    ],
    techStack: ['GradientScarMemory 34.8 KB', 'Frontier Scouts', 'Tri-Compass', 'UniversalS1Agent', 'Rust / PyTorch'],
    githubUrl: 'https://github.com/byte-od/system_one',
    labAnchor: '#aura-origin',
    metrics: [
      { label: 'GradientScarMemory', value: '34.8 KB Constante O(1)' },
      { label: 'Latência Reflexiva S1', value: '0.23 ms CPU P50' },
    ],
  },
];

export function getPortfolioInfo(lang: 'en' | 'pt' = 'en'): PortfolioInfo {
  return lang === 'pt' ? PORTFOLIO_INFO_PT : PORTFOLIO_INFO_EN;
}

export function getProjects(lang: 'en' | 'pt' = 'en'): ProjectData[] {
  return lang === 'pt' ? PROJECTS_PT : PROJECTS_EN;
}
