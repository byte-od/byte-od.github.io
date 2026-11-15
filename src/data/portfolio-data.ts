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
    'Independent AI Architecture Researcher & Principal Systems Engineer. Author of Mixture of Scars (MoS), Associative SCache O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v5.0 (Tabula Rasa), PointScarFlow v1.0, and AURA / System One (GradientScarMemory).',
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
    'Pesquisador de Arquiteturas Neurais e Arquiteto Principal de Sistemas. Autor do Mixture of Scars (MoS), SCache Associativo O(1) (34.8 KB), D4-LUT256 VQ4-Scar, ScarCodec v4.0, ScarNativeLM v5.0 (Tabula Rasa), PointScarFlow v1.0 e AURA / System One (GradientScarMemory).',
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
    id: 'scarnative-v5',
    title: 'ScarNativeLM v5.0 — Pure Scar-Native Foundation Engine',
    badge: '🌱 Tabula Rasa · Zenodo v5.0',
    category: 'foundation_llm',
    featured: true,
    specs: 'Zero nn.Linear · D4-LUT256 VQ4-STE · 0.0000 KL PTQ · 22 WGSL Shaders',
    summary:
      'Foundation language architecture pretrained 100% from scratch (step t = 0, zero pretrained weights, zero dense nn.Linear layers) directly on the discrete scar manifold with DifferentiableSCacheAttention O(1) and Zero-SGD RRR P_⊥lang cartridges.',
    details: [
      'ScarNativeLinear replaces every dense matrix with Equilibration d_eq + FWHT-64 + D4-LUT256 VQ4-STE + INT4-STE low-rank scars, eliminating post-training quantization shock (0.000000 KL divergence upon binary packing).',
      'End-to-end DifferentiableSCacheAttention trains directly over fixed Sinks + Circular Window + Pre-RoPE Widrow-Hoff Associative Matrix W_scar in strict O(1) memory (Δ = +0 B at any horizon T -> ∞).',
      'LanguageNullspaceProjector (P_⊥lang) decouples mutable facts from the frozen trunk: 4 KB-aligned INT4 cartridges minted via closed-form Reduced-Rank Regression (0 SGD steps) in English fire seamlessly across PT, EN, ES, and FR.',
      'Cryptographic 0x00 physical overwrite erases individual factual cartridges in < 0.2 ms with exact bit-zero trunk collateral damage (Δlogits = 0.00000000).',
      'Exact numerical parity (cos > 0.99999) verified between PyTorch and 22 custom WebGPU WGSL 1.0 compute shaders.',
    ],
    techStack: ['ScarNativeLinear', 'VQ4-STE', 'SCache O(1)', 'WebGPU WGSL 1.0', 'PyTorch', 'P_⊥lang RRR'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'PTQ Packing Shock', value: '0.000000 KL' },
      { label: 'SCache Growth', value: 'Δ +0 B (O(1))' },
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
    badge: '📡 Domain-Independent · Zenodo v4.0',
    category: 'spatial_physical',
    specs: 'Radio RFI P_⊥ · 34.8 KB Black-Box Telemetry · 12.05x HDR CFD Vacuum=0.0',
    summary:
      'Proves that the Scar Decomposition Operator S(X) = P_⊥(Q_VQ4(R · D^-1 X) + U · V^T) + W_scar is a universal mathematical compression and anomaly-retention primitive for physical sensors, radio interferometry, and scientific PDEs beyond NLP.',
    details: [
      'Radio-Interferometric RFI Excision: Null-space projector P_⊥ excises 1,000x non-stationary satellite bursts while recovering weak pulsar dispersion sweeps with cos = 0.9923 at 5.75x array compression (where 2-bit VLBI collapses to cos = 0.08).',
      '34.8 KB O(1) Aerospace Black-Box Telemetry: Retains 5/5 transient precursor fault signatures across T = 8,000+ sensor frames with exact Δ = +0 B memory growth (cos = 0.9989), where equal-byte circular ring buffers retain 0/5.',
      '12.05x High-Dynamic-Range CFD Shockwave Compression: Entry-0 exact null vector in LUT256 guarantees bit-exact 0.00000000 quiescent vacuum error while preserving Mach shock discontinuities with O(1) random block access.',
    ],
    techStack: ['ScarCodec v4.0', 'Nullspace P_⊥', 'LUT256 Exact Vacuum', '34.8 KB Black-Box', 'O(1) Block Decode'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: '1,000x RFI Recovery', value: 'cos = 0.9923 (5.75x)' },
      { label: 'CFD HDR Compression', value: '12.05x (0.0 Vacuum)' },
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
    id: 'scarnative-v5',
    title: 'ScarNativeLM v5.0 — Motor Fundacional Puro de Cicatrizes',
    badge: '🌱 Tabula Rasa · Zenodo v5.0',
    category: 'foundation_llm',
    featured: true,
    specs: 'Zero nn.Linear · D4-LUT256 VQ4-STE · 0.0000 KL PTQ · 22 Shaders WGSL',
    summary:
      'Arquitetura fundacional pré-treinada 100% do zero (passo t = 0, zero pesos pré-treinados, zero camadas densas nn.Linear) diretamente na variedade discreta de cicatrizes com DifferentiableSCacheAttention O(1) e cartuchos Zero-SGD RRR em P_⊥lang.',
    details: [
      'ScarNativeLinear substitui toda matriz densa por Equilibração d_eq + FWHT-64 + D4-LUT256 VQ4-STE + cicatrizes INT4-STE de posto baixo, eliminando o choque de quantização pós-treino (0.000000 KL ao empacotar em binário).',
      'DifferentiableSCacheAttention treina ponta a ponta sobre Sinks + Janela Circular + Matriz Associativa Widrow-Hoff pré-RoPE W_scar em memória estritamente O(1) (Δ = +0 B em qualquer horizonte T -> ∞).',
      'LanguageNullspaceProjector (P_⊥lang) desacopla fatos mutáveis do tronco congelado: cartuchos INT4 alinhados a 4 KB cunhados via Regressão de Posto Reduzido fechada (0 passos de SGD) em inglês disparam nativamente em PT, EN, ES e FR.',
      'Sobrescrita física criptográfica 0x00 apaga cartuchos factuais individuais em < 0.2 ms com zero dano colateral ao tronco (Δlogits = 0.00000000).',
      'Paridade numérica exata (cos > 0.99999) verificada entre PyTorch e 22 compute shaders WebGPU WGSL 1.0 customizados.',
    ],
    techStack: ['ScarNativeLinear', 'VQ4-STE', 'SCache O(1)', 'WebGPU WGSL 1.0', 'PyTorch', 'P_⊥lang RRR'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#scar-native-lab',
    metrics: [
      { label: 'Choque PTQ no Empacotamento', value: '0.000000 KL' },
      { label: 'Crescimento do SCache', value: 'Δ +0 B (O(1))' },
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
    badge: '📡 Independente de Domínio · Zenodo v4.0',
    category: 'spatial_physical',
    specs: 'Rádio RFI P_⊥ · Caixa-Preta 34.8 KB · CFD HDR 12.05x Vácuo=0.0',
    summary:
      'Demonstra que o Operador de Decomposição por Cicatrizes S(X) = P_⊥(Q_VQ4(R · D^-1 X) + U · V^T) + W_scar é uma primitiva matemática universal de compressão e retenção de anomalias para sensores físicos, radiointerferometria e EDPs científicas além de NLP.',
    details: [
      'Excisão de RFI em Radiointerferometria: O projetor de subespaço nulo P_⊥ anula rajadas não-estacionárias 1.000x mais intensas de satélites recuperando pulsos fracos de pulsares com cos = 0.9923 e compressão 5.75x (onde VLBI 2-bit colapsa para cos = 0.08).',
      'Caixa-Preta Aeroespacial O(1) em 34.8 KB: Retém 5/5 assinaturas precursoras de falha ao longo de T = 8.000+ quadros de telemetria com crescimento exato Δ = +0 B (cos = 0.9989), onde um buffer circular de mesmo tamanho retém 0/5.',
      'Compressão 12.05x em Campos de Choque CFD HDR: O vetor nulo exato na entrada 0 da LUT256 garante erro zero absoluto (0.00000000) em regiões de vácuo calmo preservando frentes de choque de Mach com decodificação aleatória O(1) por bloco.',
    ],
    techStack: ['ScarCodec v4.0', 'Subespaço Nulo P_⊥', 'LUT256 Vácuo Exato', 'Caixa-Preta 34.8 KB', 'Decode O(1)'],
    githubUrl: 'https://github.com/byte-od/mixture-of-scars',
    paperUrl: 'https://doi.org/10.5281/zenodo.24172784',
    labAnchor: '#physical-scar-lab',
    metrics: [
      { label: 'Recuperação sob RFI 1.000x', value: 'cos = 0.9923 (5.75x)' },
      { label: 'Compressão CFD HDR', value: '12.05x (Vácuo 0.0)' },
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
