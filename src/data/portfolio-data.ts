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
    'Independent AI Architecture Researcher & Principal Systems Engineer. Author of Mixture of Scars (MoS), Associative SCache O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0 (Tabula Rasa Shared-Stage MoS), PointScarFlow v1.0, and AURA / System One (GradientScarMemory).',
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
    'Pesquisador de Arquiteturas Neurais e Arquiteto Principal de Sistemas. Autor do Mixture of Scars (MoS), SCache Associativo O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v6.0 (Tabula Rasa Shared-Stage MoS), PointScarFlow v1.0 e AURA / System One (GradientScarMemory).',
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
    id: 'scarcodec-v4',
    title: 'ScarCodec v4.0 — Physical & Hardware Signal/Tensor Codec',
    badge: '📡 5 Public NASA/NIH Archives + Live Lab',
    category: 'spatial_physical',
    specs: 'Spitzer 2,094x HDR · NASA Shuttle 6/6 Faults · PhysioNet 64-Ch · 34.8 KB O(1)',
    summary:
      'Neural-network-free mathematical physics & hardware compression suite validated both live in-browser on controlled PDE/array phantoms and unmodified on 5 SHA-256-verified public observational & spacecraft archives (NASA Spitzer IRAC, NASA Space Shuttle, CSO Bolocam Sgr A*, FCRAO 14-m Radio, and NIH/PhysioNet 64-Ch).',
    details: [
      'Real Public Astronomy FITS Archives (NASA Spitzer IRAC, CSO Bolocam Sgr A*, FCRAO 14-m 13CO): Compresses IEEE-754 floats by 11.77x–12.27x (where zlib -9 stalls at 1.10x–1.14x), decodes random 64-float blocks 23.2x–34.7x faster (Δ = 0.0), and discovers an empirical HDR contrast scaling law over byte-matched FP16 2D rFFT (4.6x -> 45.9x -> 2,093.6x lower stellar peak photometry error).',
      'Real NASA Space Shuttle & NIH/PhysioNet 64-Channel O(1) Telemetry: Retains 6/6 (100%) ultra-rare 0.0414% Bypass Valve faults in NASA Space Shuttle radiator telemetry (0/6 in equal-byte circular ring) and reconstructs 32 masked odd electrodes from 32 even electrodes at cos = 0.9967 inside exact 34,816 B (Δ = +0 B), plus +13.25 dB raw C3 motor-cortex Laplacian SNR gain via cross-run blind P_⊥.',
      'Controlled Live Browser Proofs (Zero Oracle): Blind noisy-covariance RFI excision under 1,000x (-60 dB) interference (cos = 0.9753 native / 0.9837 browser), 2-of-6 partial-cue black-box recall (cos = 0.9999), and conservative Rusanov 2D Burgers shockwave PDE (188.2x lower RMSE than byte-matched FP16 2D rFFT, exact 0.000000 vacuum).',
    ],
    techStack: ['ScarCodec v4.0', 'NASA Spitzer & Shuttle', 'PhysioNet 64-Ch P_⊥', 'LUT256 Exact Vacuum', '34.8 KB SCache O(1)'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: 'Spitzer IRAC Peak Error', value: '2,093.6x < FP16 rFFT' },
      { label: 'NASA Shuttle Bpv Faults', value: '6/6 (100%) in O(1)' },
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
      'Trained across 450M+ steps in 128 parallel C++/Rust environments, solving Rubik’s Cube scrambles up to Depth 10 (232.2B states) with zero MCTS tree search — running live in-browser via ONNX Wasm SIMD.',
    ],
    techStack: ['GradientScarMemory 34.8 KB', 'Frontier Scouts', 'Tri-Compass', 'UniversalS1Agent', 'Rust / PyTorch', 'ONNX Wasm SIMD'],
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
    id: 'scarcodec-v4',
    title: 'ScarCodec v4.0 — Codec Físico & de Hardware por Cicatrizes',
    badge: '📡 5 Arquivos Reais NASA/NIH + Lab Ao Vivo',
    category: 'spatial_physical',
    specs: 'Spitzer 2.094x HDR · NASA Shuttle 6/6 Panes · PhysioNet 64-Ch · 34.8 KB O(1)',
    summary:
      'Suíte de física matemática aplicada e compressão de hardware sem redes neurais, validada tanto ao vivo no navegador em bancadas controladas de EDPs/antenas quanto sem modificações em 5 bases científicas públicas verificadas por SHA-256 (NASA Spitzer IRAC, Ônibus Espacial da NASA, CSO Bolocam Sgr A*, Radiotelescópio FCRAO 14-m e NIH/PhysioNet 64 canais).',
    details: [
      'Arquivos Astronômicos Públicos FITS Reais (NASA Spitzer IRAC, CSO Bolocam Sgr A*, FCRAO 14-m 13CO): Comprime floats IEEE-754 em 11.77x–12.27x (onde zlib -9 trava em 1.10x–1.14x), lê blocos aleatórios de 64 amostras 23.2x–34.7x mais rápido (Δ = 0.0) e comprova uma lei empírica monotônica de contraste HDR sobre rFFT 2D FP16 pareada em bytes (4.6x -> 45.9x -> 2.093,6x menor erro fotométrico estelar).',
      'Telemetria Real O(1) no Ônibus Espacial da NASA & NIH/PhysioNet 64 Canais: Retém 6/6 (100%) das falhas críticas raras (0.0414%) de Válvula de Bypass nos radiadores do Space Shuttle (0/6 no buffer circular de mesmo tamanho) e reconstrói 32 eletrodos ímpares ocultos a partir dos 32 pares com cos = 0.9967 em exatos 34.816 B (Δ = +0 B), além de +13.25 dB de ganho SNR no Laplaciano motor C3 bruto via P_⊥ entre sessões.',
      'Bancadas Controladas Ao Vivo no Navegador (Zero Oráculo): Excisão cega de RFI 1.000x (-60 dB) por covariância ruidosa (cos = 0.9753 nativo / 0.9837 navegador), evocação parcial 2-de-6 na caixa-preta (cos = 0.9999) e EDP conservativa de Burgers 2D via Rusanov (RMSE 188.2x menor que rFFT 2D FP16 pareada em bytes, erro 0.000000 no vácuo).',
    ],
    techStack: ['ScarCodec v4.0', 'NASA Spitzer & Shuttle', 'PhysioNet 64-Ch P_⊥', 'LUT256 Vácuo Exato', 'SCache 34.8 KB O(1)'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: 'Erro Estelar Spitzer IRAC', value: '2.093,6x < rFFT FP16' },
      { label: 'Panes NASA Space Shuttle', value: '6/6 (100%) em O(1)' },
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
      'Treinado por mais de 450M+ passos em 128 ambientes paralelos C++/Rust, resolvendo embaralhamentos do Cubo de Rubik até Profundidade 10 (232.2B estados) sem busca em árvore MCTS — rodando ao vivo no navegador via ONNX Wasm SIMD.',
    ],
    techStack: ['GradientScarMemory 34.8 KB', 'Frontier Scouts', 'Tri-Compass', 'UniversalS1Agent', 'Rust / PyTorch', 'ONNX Wasm SIMD'],
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
