export interface ProjectData {
  id: string;
  title: string;
  badge: string;
  category: 'ml_ai' | 'developer_tooling' | 'mobile_graphics';
  specs?: string;
  summary: string;
  details: string[];
  techStack: string[];
  githubUrl?: string;
  packageUrl?: string;
  interactiveType?: 'transformer_sim' | 'mermaid_demo' | 'graphics_demo' | 'git_telemetry';
  metrics?: { label: string; value: string }[];
}

export interface ExperienceData {
  company: string;
  role: string;
  period: string;
  location: string;
  type: string;
  highlights: string[];
  technologies: string[];
}

export interface SkillCategory {
  title: string;
  skills: { name: string; level?: string; highlight?: boolean }[];
}

export const PORTFOLIO_INFO = {
  name: 'Julio Cesar da Costa Reis Filho',
  handle: 'byte-od',
  title: 'Senior Systems Architect & LLM Engineer',
  location: 'Curitiba, Paraná, Brazil (Open to Remote & Relocation)',
  email: 'reisfilho1116@gmail.com',
  phone: '+55 41 9 9626 8203',
  github: 'https://github.com/byte-od',
  linkedin: 'https://linkedin.com/in/juliocrfilho',
  languages: [
    { name: 'English', level: 'Proficient C2 (EF SET 75/100)' },
    { name: 'Portuguese', level: 'Native' },
    { name: 'Spanish', level: 'Intermediate' },
  ],
  bio: 'Systems Architect and Machine Learning Engineer with 8+ years of engineering experience spanning custom Causal Transformers (CIR-Engine 503M) trained from scratch, distributed DDP clusters, autonomous tool-use orchestration, and high-concurrency microservices in Go, Next.js, and Flutter.',
};

export function getPortfolioInfo(lang: 'en' | 'pt' = 'en') {
  if (lang === 'pt') {
    return {
      ...PORTFOLIO_INFO,
      title: 'Arquiteto de Sistemas Sênior & Engenheiro de LLMs',
      location: 'Curitiba, Paraná, Brasil (Aberto para Remoto & Relocalização)',
      languages: [
        { name: 'Inglês', level: 'Proficiente C2 (EF SET 75/100)' },
        { name: 'Português', level: 'Nativo' },
        { name: 'Espanhol', level: 'Intermediário' },
      ],
      bio: 'Arquiteto de Sistemas e Engenheiro de Machine Learning com 8+ anos de experiência cobrindo Transformers Causais proprietários (CIR-Engine 503M) treinados do zero, clusters distribuídos DDP em PyTorch, orquestração autônoma de ferramentas e microsserviços de alta concorrência em Go, Next.js e Flutter.',
    };
  }
  return PORTFOLIO_INFO;
}

export const PROJECTS: ProjectData[] = [
  {
    id: 'cir-engine',
    title: 'CIR-Engine & Cir-Jev',
    badge: 'Core ML Architecture',
    category: 'ml_ai',
    specs: '503M Parameters · 16 Layers · 1536 Dim · 12 Heads (4 KV Heads)',
    summary: 'Proprietary 503M-parameter causal autoregressive Transformer designed, trained, and benchmarked from scratch in PyTorch without pre-baked high-level wrappers.',
    details: [
      'Implemented modern attention and normalization techniques including Rotary Position Embeddings (RoPE), RMSNorm, and Scaled Dot-Product Attention (SDPA/FlashAttention).',
      'Architected distributed training pipeline via torchrun & PyTorch DistributedDataParallel (DDP) across NVIDIA Tesla T4 and L4 clusters with mixed-precision (FP16/AMP).',
      'Achieved throughput of ~6,200 tokens/s and cut epoch runtime by 66% (from 180 min to 61 min).',
      'Formulated strict canonical communication grammar ([OP:QUERY], [OP:SOLVE], [OP:RESULT] [VAL: ...]) with 98–100% format compliance and zero anthropomorphic drift.',
      'Autonomous runtime tool-use interceptor: generates structured subroutine calls ([OP:CALC]) triggering deterministic Python execution loop returning verified outputs in O(1) with 100% calculation accuracy.',
      'SFT Prompt Loss Masking (labels = -100) preventing overfitting on reasoning templates.',
      'Headless MLOps linking VS Code, Git, and Kaggle API to autonomously dispatch training jobs and evaluate local inference via Apple Silicon Metal Performance Shaders (MPS) at ~0.19s latency.'
    ],
    techStack: ['PyTorch', 'DistributedDataParallel (DDP)', 'CUDA', 'RoPE', 'FlashAttention', 'Kaggle API', 'Apple Metal (MPS)'],
    githubUrl: 'https://github.com/byte-od/cir-jev',
    interactiveType: 'transformer_sim',
    metrics: [
      { label: 'Parameters', value: '503M' },
      { label: 'DDP Throughput', value: '~6,200 t/s' },
      { label: 'Epoch Time Cut', value: '-66%' },
      { label: 'Tool Accuracy', value: '100% O(1)' },
    ],
  },
  {
    id: 'mddd-cli',
    title: 'mddd-cli (Spec-Driven Dev via Diagrams & Matrices)',
    badge: 'Developer Tooling & NPM Package',
    category: 'developer_tooling',
    specs: 'Spec-Driven Development · Executable Diagrams & Decision Matrices · NPM Tool',
    summary: 'Spec-Driven Development CLI framework that elevates architectural diagrams and formal decision matrices—instead of ambiguous pure text—as the executable specification to compile production Clean Architecture systems.',
    details: [
      'Pioneers Spec-Driven Development (SDD): replaces vague, drift-prone pure text specifications with formal diagram ASTs and deterministic decision truth tables.',
      'Compiles visual domain topologies and decision matrices directly into Clean Architecture entities, use-cases, and HTTP/gRPC contracts with 100% branch verification.',
      'Published on npm for global developer consumption; guarantees mathematical alignment between architecture specs and production code.'
    ],
    techStack: ['TypeScript', 'Node.js', 'Mermaid.js', 'AST Parsing', 'Decision Matrices', 'CLI', 'Clean Architecture'],
    githubUrl: 'https://github.com/byte-od/mermaid-diagram-driven-development',
    packageUrl: 'https://www.npmjs.com/package/mddd-cli',
    interactiveType: 'mermaid_demo',
    metrics: [
      { label: 'Paradigm', value: 'Spec-Driven Dev' },
      { label: 'Spec Input', value: 'Diagrams & Matrices' },
      { label: 'Code Output', value: 'Clean Architecture' },
    ],
  },
  {
    id: 'mad-cli',
    title: 'MAD-cli (Mermaid Auto-Doccing)',
    badge: 'Living Documentation & Architecture CLI',
    category: 'developer_tooling',
    specs: 'Code-to-Mermaid Living Docs · MAD Tags Parser',
    summary: 'CLI and automation tool that parses special MAD comment tags (<!-- MAD:... -->, @mad) across codebases and transforms them into living, auto-updating Mermaid architecture diagrams directly within repository Markdown documentation.',
    details: [
      'Engineered to eliminate documentation rot by extracting living Mermaid sequence, class, and architecture flowcharts directly from AST and comment tags in Go, Dart, TypeScript, and Python.',
      'Scans and synchronizes <!-- MAD:SEQUENCE -->, <!-- MAD:FLOWCHART -->, and <!-- MAD:ARCHITECTURE --> blocks on every commit or CI pipeline run.',
      'Forms a continuous bidirectional architecture loop with mddd-cli: mddd-cli compiles diagrams to code, while MAD-cli extracts living code reality back into verifiable diagrams.',
      'Published open-source developer tooling for clean architectural visibility across distributed teams.'
    ],
    techStack: ['Go', 'TypeScript', 'AST Parsing', 'Mermaid.js', 'Living Documentation', 'Git Hooks', 'CLI'],
    githubUrl: 'https://github.com/byte-od/mad',
    metrics: [
      { label: 'Syntax Engine', value: 'MAD Tags' },
      { label: 'Sync Loop', value: 'Code-to-Diagram' },
    ],
  },
  {
    id: 'system_one',
    title: 'system_one',
    badge: 'Fast-Path Heuristic Engine & Dual-Process AI',
    category: 'ml_ai',
    specs: 'Sub-10ms Inference · Dual-Process Cognitive Layer · Fast-Path Routing',
    summary: 'Fast-path heuristic inference kernel and reactive cognitive layer (System 1) designed as the low-latency companion to the 503M CIR-Engine. Dispatches intuitive, sub-10ms classifications before triggering deliberate causal reasoning.',
    details: [
      'Implements dual-process cognitive architecture (System 1 fast heuristic dispatch vs System 2 slow causal deliberation).',
      'Sub-10ms local evaluation loop leveraging INT8/FP8 weight quantization, optimized vector embeddings, and zero-allocation memory buffers.',
      'Dynamically routes queries: handles high-probability heuristic paths instantly, while dispatching high-complexity reasoning graphs to the 503M CIR-Engine.',
      'Integrates with CIR-Engine KV-cache and canonical grammar loops for seamless token handoff.'
    ],
    techStack: ['Python', 'PyTorch', 'Quantization (INT8/FP8)', 'Vector Embeddings', 'Apple MPS / CUDA', 'Inference Optimization'],
    githubUrl: 'https://github.com/byte-od/system_one',
    metrics: [
      { label: 'Latency', value: '<10ms Local' },
      { label: 'Architecture', value: 'System 1 Fast-Path' },
    ],
  },
  {
    id: 'flutter-scene-fork',
    title: 'flutter_scene (Fork & Animation Contributions)',
    badge: 'Graphics & Scene Animations',
    category: 'mobile_graphics',
    specs: '3D Scene Graph · Scene Animations · Custom Shaders',
    summary: 'Custom fork of Flutter’s 3D graphics engine (flutter_scene) with open-source contributions focused on scene animation pipelines, keyframe playback, low-latency framebuffers, and custom Metal/Vulkan shaders.',
    details: [
      'Contributed optimizations and enhancements to scene animation handling, playback interpolation, and 3D transform hierarchies.',
      'Engineered custom shader pipelines and memory buffer optimizations for real-time 3D asset rendering.',
      'Demonstrates deep fluency in low-level rendering (C++, shaders) seamlessly unified with high-level mobile applications (Dart/Flutter).'
    ],
    techStack: ['Dart', 'C++', 'Flutter Scene', 'Metal/Vulkan', 'GLSL/MSL Shaders', 'Scene Animations'],
    githubUrl: 'https://github.com/byte-od/flutter_scene',
    interactiveType: 'graphics_demo',
    metrics: [
      { label: 'Target', value: '60-120 FPS' },
      { label: 'Focus', value: 'Scene Animations' },
      { label: 'Substrate', value: 'Metal / Vulkan' },
    ],
  },
  {
    id: 'gittrack',
    title: 'GitTrack',
    badge: 'Granular Tracking & VIP AI Insights',
    category: 'developer_tooling',
    specs: 'Detailed Commit Tracking · Private Token Access · AI Insights (VIP)',
    summary: 'Granular developer commit tracking platform with support for secure personal access tokens to index private repositories, featuring comprehensive AI-driven commit and architectural impact analysis on the VIP tier.',
    details: [
      'Delivers granular, commit-by-commit developer telemetry tracking code evolution and architectural progression in real time.',
      'Supports secure Personal Access Token (PAT) injection to unlock and track private repositories with sensitive data masking.',
      'VIP Plan: On-demand deep AI analysis evaluating commit quality, architectural consistency, and potential regressions.',
      'Built in Go for low-latency git log streaming and local SQLite indexing.'
    ],
    techStack: ['Go', 'Git Plumbing APIs', 'Personal Access Tokens', 'AI Code Analysis', 'SQLite', 'CLI'],
    interactiveType: 'git_telemetry',
    metrics: [
      { label: 'Tracking', value: 'Granular Commits' },
      { label: 'Private Repos', value: 'Token Access' },
      { label: 'AI Review', value: 'VIP Plan' },
      { label: 'Engine', value: 'Golang Core' },
    ],
  },
  {
    id: 'register-red',
    title: 'Register-red',
    badge: 'Full-Stack Financial Architecture',
    category: 'developer_tooling',
    specs: 'Apple Numbers (.numbers) · Google Drive Sync · Email Polling Automation · FastAPI & React 19',
    summary: 'Modern financial management and budget projection web panel with bi-directional Apple Numbers (.numbers) parsing, live Google Drive cloud synchronization, automated bank email polling, and intelligent balance reconciliation.',
    details: [
      'Native parsing, updating, and mutation of Apple Numbers (.numbers) files via Python (numbers-parser) and styled Excel (.xlsx) exports via openpyxl.',
      'Automated bank email polling pipeline (Nubank/IMAP/Gmail): cleans complex HTML with zero-width character stripping (\\u200c, \\ufeff), extracts transaction amounts with exact decimal precision (R$ XX,XX), isolates recipients after "para", and performs smart reconciliation against planned bills with anti-duplication guards.',
      'Direct integration with Google Drive cloud storage for real-time synchronization, instant cloud saves, and automated directory indexing.',
      'Intelligent multi-month budget cloning: auto-propagates recurring fixed obligations (rent, utilities, tuition, accounting), resets statuses, recalculates due dates, and carries forward positive rollover balances.',
      'Interactive one-click payment reconciliation with dynamic KPI cards (Income, Outflows, Paid, Pending, Net Surplus) and festive celebration triggers.',
      'Category distribution analytics (Home vs. Personal vs. Credit), Top 5 expense ranking, and 31-day accumulated timeline projection.',
      'High-performance full-stack architecture pairing a FastAPI asynchronous backend with React 19, TypeScript, Tailwind CSS, and Vite.'
    ],
    techStack: ['Python 3.11', 'FastAPI', 'numbers-parser', 'Google Drive API', 'IMAP / Email Polling', 'React 19', 'TypeScript', 'Tailwind CSS', 'openpyxl'],
    githubUrl: 'https://github.com/byte-od/Register-red',
    metrics: [
      { label: 'Sync Pipeline', value: 'Google Drive' },
      { label: 'Email Polling', value: 'Nubank Auto' },
      { label: 'Format', value: '.numbers Native' },
      { label: 'Backend', value: 'FastAPI Async' },
      { label: 'Frontend', value: 'React 19' },
    ],
  },
];

export function getProjects(lang: 'en' | 'pt' = 'en'): ProjectData[] {
  if (lang !== 'pt') return PROJECTS;

  return [
    {
      id: 'cir-engine',
      title: 'CIR-Engine & Cir-Jev',
      badge: 'Arquitetura ML Central',
      category: 'ml_ai',
      specs: '503M Parâmetros · 16 Camadas · 1536 Dim · 12 Cabeças (4 KV)',
      summary: 'Transformer causal autorregressivo proprietário de 503M parâmetros projetado, treinado e avaliado do zero em PyTorch sem wrappers de alto nível pré-fabricados.',
      details: [
        'Implementação de técnicas modernas de atenção e normalização, incluindo Rotary Position Embeddings (RoPE), RMSNorm e Scaled Dot-Product Attention (SDPA/FlashAttention).',
        'Arquitetura de pipeline de treinamento distribuído via torchrun & PyTorch DistributedDataParallel (DDP) em clusters NVIDIA Tesla T4 e L4 com precisão mista (FP16/AMP).',
        'Alcançou throughput de ~6.200 tokens/s e redução de 66% no tempo de época (de 180 min para 61 min).',
        'Formulou gramática estrita canônica de comunicação ([OP:QUERY], [OP:SOLVE], [OP:RESULT] [VAL: ...]) com 98–100% de conformidade de formato e zero desvio antropomórfico.',
        'Interceptor autônomo de uso de ferramentas em tempo de execução: emite chamadas estruturadas de sub-rotina ([OP:CALC]) acionando execução determinística em Python com retorno verificado em O(1) e 100% de precisão de cálculo.',
        'SFT Prompt Loss Masking (labels = -100) prevenindo sobreajuste (overfitting) nos templates de raciocínio.',
        'MLOps headless conectando VS Code, Git e Kaggle API para despachar jobs de treinamento autonomamente e validar inferência local via Apple Silicon Metal (MPS) com latência de ~0,19s.'
      ],
      techStack: ['PyTorch', 'DistributedDataParallel (DDP)', 'CUDA', 'RoPE', 'FlashAttention', 'Kaggle API', 'Apple Metal (MPS)'],
      githubUrl: 'https://github.com/byte-od/cir-jev',
      interactiveType: 'transformer_sim',
      metrics: [
        { label: 'Parâmetros', value: '503M' },
        { label: 'Throughput DDP', value: '~6.200 t/s' },
        { label: 'Redução de Época', value: '-66%' },
        { label: 'Precisão Ferramentas', value: '100% O(1)' },
      ],
    },
    {
      id: 'mddd-cli',
      title: 'mddd-cli (Spec-Driven Dev via Diagramas & Matrizes)',
      badge: 'Ferramenta Dev & Pacote NPM',
      category: 'developer_tooling',
      specs: 'Spec-Driven Development · Diagramas & Matrizes Executáveis · Pacote NPM',
      summary: 'Framework CLI de Spec-Driven Development (SDD) que utiliza diagramas arquiteturais e matrizes de decisão formais — ao invés de puro texto ambíguo — como a especificação executável para compilar sistemas em Clean Architecture.',
      details: [
        'Pioneiro em Spec-Driven Development (SDD): substitui especificações vagas e sujeitas a desvios em puro texto por diagramas com AST validável e matrizes de decisão determinísticas.',
        'Compila topologias visuais de domínio e matrizes de decisão diretamente em entidades de Clean Architecture, casos de uso e contratos gRPC/TypeScript com 100% de cobertura de branches.',
        'Publicado no npm para consumo global; assegura alinhamento rigoroso entre a especificação arquitetural e o código implementado.'
      ],
      techStack: ['TypeScript', 'Node.js', 'Mermaid.js', 'Parsing AST', 'Matrizes de Decisão', 'CLI', 'Clean Architecture'],
      githubUrl: 'https://github.com/byte-od/mermaid-diagram-driven-development',
      packageUrl: 'https://www.npmjs.com/package/mddd-cli',
      interactiveType: 'mermaid_demo',
      metrics: [
        { label: 'Paradigma', value: 'Spec-Driven Dev' },
        { label: 'Entrada Spec', value: 'Diagramas & Matrizes' },
        { label: 'Saída Código', value: 'Clean Architecture' },
      ],
    },
    {
      id: 'mad-cli',
      title: 'MAD-cli (Mermaid Auto-Doccing)',
      badge: 'Documentação Viva & CLI de Arquitetura',
      category: 'developer_tooling',
      specs: 'Código-para-Mermaid Vivo · Parser de Tags MAD',
      summary: 'Ferramenta CLI e automação que analisa tags de comentário especiais MAD (<!-- MAD:... -->, @mad) em bases de código e as transforma em diagramas de arquitetura Mermaid vivos e auto-atualizáveis diretamente na documentação Markdown.',
      details: [
        'Projetado para eliminar a obsolescência de documentação ao extrair diagramas Mermaid de sequência, classes e fluxogramas diretamente da AST e tags em Go, Dart, TypeScript e Python.',
        'Verifica e sincroniza blocos <!-- MAD:SEQUENCE -->, <!-- MAD:FLOWCHART --> e <!-- MAD:ARCHITECTURE --> a cada commit ou execução de pipeline CI.',
        'Forma um ciclo bidirecional contínuo de arquitetura com o mddd-cli: mddd-cli compila diagramas para código, enquanto MAD-cli extrai a realidade viva do código de volta para diagramas verificáveis.',
        'Ferramenta open-source para visibilidade arquitetural transparente entre equipes de engenharia distribuídas.'
      ],
      techStack: ['Go', 'TypeScript', 'Parsing AST', 'Mermaid.js', 'Documentação Viva', 'Git Hooks', 'CLI'],
      githubUrl: 'https://github.com/byte-od/mad',
      metrics: [
        { label: 'Motor de Sintaxe', value: 'Tags MAD' },
        { label: 'Ciclo de Sincronia', value: 'Código-para-Diagrama' },
      ],
    },
    {
      id: 'system_one',
      title: 'system_one',
      badge: 'Motor Heurístico Fast-Path & IA de Processo Duplo',
      category: 'ml_ai',
      specs: 'Inferência Sub-10ms · Camada Cognitiva Dupla · Roteamento Fast-Path',
      summary: 'Kernel de inferência heurística ultra-rápida e camada cognitiva reativa (Sistema 1) projetado como companheiro de baixíssima latência do CIR-Engine 503M. Executa classificações intuitivas sub-10ms antes de acionar o raciocínio causal deliberado.',
      details: [
        'Implementa arquitetura cognitiva de processo duplo (despacho heurístico rápido Sistema 1 vs deliberação causal profunda Sistema 2).',
        'Loop de avaliação local sub-10ms aproveitando quantização de pesos INT8/FP8, embeddings vetoriais otimizados e buffers de memória com zero-alocação.',
        'Roteia consultas dinamicamente: resolve caminhos heurísticos de alta probabilidade instantaneamente, despachando grafos de raciocínio de alta complexidade para o CIR-Engine 503M.',
        'Integração direta com o KV-cache do CIR-Engine e loops de gramática canônica para repasse contínuo de tokens.'
      ],
      techStack: ['Python', 'PyTorch', 'Quantização (INT8/FP8)', 'Embeddings Vetoriais', 'Apple MPS / CUDA', 'Otimização de Inferência'],
      githubUrl: 'https://github.com/byte-od/system_one',
      metrics: [
        { label: 'Latência', value: '<10ms Local' },
        { label: 'Arquitetura', value: 'Sistema 1 Fast-Path' },
      ],
    },
    {
      id: 'flutter-scene-fork',
      title: 'flutter_scene (Fork & Contribuições em Animações)',
      badge: 'Computação Gráfica & Animações de Scenes',
      category: 'mobile_graphics',
      specs: 'Grafo de Cena 3D · Animações de Scenes · Shaders Customizados',
      summary: 'Fork customizado do motor gráfico 3D do Flutter (flutter_scene) com contribuições focadas na esteira de animações de scenes, interpolação de keyframes, framebuffers de baixa latência e shaders em Metal/Vulkan.',
      details: [
        'Contribuições com foco no pipeline de animações de scenes, reprodução de keyframes e hierarquias de transformação 3D.',
        'Engenharia de pipelines customizados de shaders e otimizações de buffer de memória para renderização de ativos 3D em tempo real.',
        'Demonstra fluência profunda em renderização de baixo nível (C++, shaders) unificada de forma transparente a aplicações mobile de alto nível (Dart/Flutter).'
      ],
      techStack: ['Dart', 'C++', 'Flutter Scene', 'Metal/Vulkan', 'Shaders GLSL/MSL', 'Animações de Scenes'],
      githubUrl: 'https://github.com/byte-od/flutter_scene',
      interactiveType: 'graphics_demo',
      metrics: [
        { label: 'Alvo', value: '60-120 FPS' },
        { label: 'Foco', value: 'Animações de Scenes' },
        { label: 'Substrato', value: 'Metal / Vulkan' },
      ],
    },
    {
      id: 'gittrack',
      title: 'GitTrack',
      badge: 'Rastreamento Granular & Análise IA VIP',
      category: 'developer_tooling',
      specs: 'Rastreamento Detalhado de Commits · Token Privado · Análise IA (VIP)',
      summary: 'Plataforma de rastreamento detalhado de commits para desenvolvedores, com suporte a token de acesso para repositórios privados e geração de análises aprofundadas de impacto e qualidade via IA no plano VIP.',
      details: [
        'Rastreamento minucioso do fluxo de commits e evolução arquitetural do desenvolvedor em tempo real.',
        'Inclusão de token de acesso pessoal (PAT) para indexação segura de repositórios privados com mascaramento de dados sensíveis.',
        'Plano VIP: Solicitação de análises detalhadas por Inteligência Artificial sobre a qualidade, impacto e coerência arquitetural dos commits.',
        'Construído com núcleo em Go para processamento de alto rendimento de logs e índices em SQLite.'
      ],
      techStack: ['Go', 'APIs Git Plumbing', 'Tokens de Acesso Pessoal', 'Análise de Código por IA', 'SQLite', 'CLI'],
      interactiveType: 'git_telemetry',
      metrics: [
        { label: 'Rastreamento', value: 'Commits Detalhados' },
        { label: 'Repos Privados', value: 'Acesso via Token' },
        { label: 'Análise de IA', value: 'Plano VIP' },
        { label: 'Motor', value: 'Núcleo em Golang' },
      ],
    },
    {
      id: 'register-red',
      title: 'Register-red',
      badge: 'Gestão Financeira & Nuvem',
      category: 'developer_tooling',
      specs: 'Apple Numbers (.numbers) · Sincronia Google Drive · Polling Automático de E-mails · FastAPI & React 19',
      summary: 'Painel web moderno de gestão financeira e projeção orçamentária com integração bidirecional nativa para planilhas Apple Numbers (.numbers) no Google Drive, automação de polling de e-mails bancários e reconciliação em tempo real.',
      details: [
        'Leitura e mutação nativa de arquivos Apple Numbers (.numbers) com Python (numbers-parser) e exportação em Excel estilizado (.xlsx) via openpyxl.',
        'Automação de polling de e-mails bancários (Nubank/IMAP): higienização de HTML preservando centavos, extração de valores com precisão decimal exata (R$ XX,XX), identificação de favorecido pós-"para" e baixa automática em contas planejadas com proteção contra duplicação.',
        'Sincronização direta com Google Drive em nuvem para gravação atômica na planilha original e indexação contínua de diretórios.',
        'Clonagem inteligente de próximos meses: replica despesas fixas recorrentes, reseta pendências, ajusta vencimentos e transfere economias remanescentes como saldo do mês anterior.',
        'Controle de liquidação e baixa de contas em 1 clique com atualização instantânea de KPIs (Entradas, Despesas, Pago, Pendente, Saldo Restante) e animações festivas de conclusão.',
        'Métricas visuais com distribuição por categorias (Casa, Pessoal, Crédito), ranking Top 5 despesas e cronograma diário acumulado de vencimentos.',
        'Arquitetura full-stack combinando backend assíncrono em FastAPI e interface reativa moderna em React 19, TypeScript, Tailwind CSS e Vite.'
      ],
      techStack: ['Python 3.11', 'FastAPI', 'numbers-parser', 'Google Drive API', 'Polling de E-mails / IMAP', 'React 19', 'TypeScript', 'Tailwind CSS', 'openpyxl'],
      githubUrl: 'https://github.com/byte-od/Register-red',
      metrics: [
        { label: 'Pipeline Sync', value: 'Google Drive' },
        { label: 'Automação', value: 'Polling E-mails' },
        { label: 'Formato', value: '.numbers Nativo' },
        { label: 'Backend', value: 'FastAPI Async' },
        { label: 'Frontend', value: 'React 19' },
      ],
    },
  ];
}

export const EXPERIENCES: ExperienceData[] = [
  {
    company: 'Asapp Desenvolvimento',
    role: 'Principal Architect & Founder',
    period: '2026 – Present',
    location: 'Curitiba & Pontal do Paraná, Brazil',
    type: 'Full-time / Leadership',
    highlights: [
      'Spearhead software architecture and technology strategy, delivering microservices, cloud-native platforms, and high-performance cross-platform applications using Go, Next.js, and Flutter.',
      'Architected Appfy, a modular micro-apps platform and marketplace leveraging distributed services on Google Cloud Platform (GCP).',
      'Created and maintain open-source developer tooling, including mddd-cli (Mermaid Diagram Driven Development CLI) on npm.',
      'Integrate multi-agent AI development workflows (Cursor, Cline, Roo Code, LiteLLM) to accelerate code generation, automated review, and testing pipelines.'
    ],
    technologies: ['Go (Golang)', 'Next.js', 'Flutter', 'Google Cloud Platform (GCP)', 'mddd-cli', 'Multi-Agent LLMs'],
  },
  {
    company: 'Opah IT',
    role: 'Senior Software Engineer / Mobile Specialist',
    period: '2026 – Present',
    location: 'Remote',
    type: 'Consulting',
    highlights: [
      'Consult and engineer mission-critical, high-performance Flutter/Android solutions for enterprise clients.',
      'Enforce robust architecture design patterns, code quality standards, and CI/CD best practices across distributed engineering teams.',
      'Grant security of 3rd-party libraries for mobile applications with Dex-guard, Obfuscation, and Encryption.'
    ],
    technologies: ['Flutter', 'Android', 'Clean Architecture', 'CI/CD', 'Dex-guard', 'Security Hardening'],
  },
  {
    company: 'Arlequim Technologies',
    role: 'Flutter Specialist & Multiplatform Lead',
    period: '07/2025 – Present',
    location: 'Curitiba, Brazil',
    type: 'Full-time',
    highlights: [
      'Technical leadership in the development of multiplatform solutions, integrating high-performance Flutter front-ends.',
      'Implementation of Technical SEO and AEO (Answer Engine Optimization) strategies to optimize content discoverability by search engines and AI-based answer engines.',
      'Migration of state management and dependency injection systems, resulting in increased stability and reduced technical debt.',
      'Configuration of CI/CD pipelines focusing on automated testing (Golden Image) to ensure visual integrity at scale.'
    ],
    technologies: ['Flutter', 'Technical SEO', 'AEO', 'Golden Image Tests', 'CI/CD Pipelines', 'GCP'],
  },
  {
    company: 'Asapp Desenvolvimento (Consulting Era)',
    role: 'Founder / Lead Developer',
    period: '10/2023 – 03/2024',
    location: 'Curitiba, Brazil',
    type: 'Leadership',
    highlights: [
      'Founded and led a consultancy focused on high-complexity Desktop, Mobile, and Web applications.',
      'Developed web applications with a focus on hybrid rendering, ensuring maximum performance and superior indexing for corporate clients.',
      'Led the refactoring of MVP to V2 for strategic projects, resulting in an App Store rating increase from 4.2 to 4.87.'
    ],
    technologies: ['Next.js', 'Flutter', 'Hybrid Rendering', 'SSR/ISR', 'Core Web Vitals'],
  },
  {
    company: 'Grupo Data',
    role: 'Flutter Architect & Technical Lead',
    period: '11/2021 – 10/2022',
    location: 'Curitiba, Brazil',
    type: 'Contract',
    highlights: [
      'Responsible for the maintenance and evolution of the "Tim Beta" mobile application, serving over 1 million active users.',
      'Orchestrated the complete refactoring of the mobile application to align with the critical enterprise backend migration from .NET to Java.'
    ],
    technologies: ['Flutter', '1M+ Active Users', 'Java', '.NET Migration', 'Clean Architecture'],
  },
  {
    company: 'Grupo GFT',
    role: 'Mobile Developer & Tech Lead',
    period: '07/2020 – 11/2021',
    location: 'Curitiba, Brazil',
    type: 'Full-time',
    highlights: [
      'Development of critical features for tier-1 financial institutions (Banco Original and Banco Votorantim) using Java, Kotlin, and Flutter.',
      'Pioneered the implementation of custom native keyboard SDKs with IPC communication for secure Flutter financial applications.'
    ],
    technologies: ['Flutter', 'Kotlin', 'Java', 'Banking FinTech', 'IPC Native Keyboards'],
  },
  {
    company: 'DevMaker Mobile Apps',
    role: 'Mobile Developer',
    period: '04/2019 – 07/2020',
    location: 'Curitiba, Brazil',
    type: 'Full-time',
    highlights: [
      'Android (Java/Kotlin), iOS (Swift), and React Native engineering.',
      'Acted as the company’s first React Native developer, leading new client projects and training internal engineering teams.'
    ],
    technologies: ['React Native', 'Swift', 'Kotlin', 'iOS', 'Android'],
  },
];

export function getExperiences(lang: 'en' | 'pt' = 'en'): ExperienceData[] {
  if (lang !== 'pt') return EXPERIENCES;

  return [
    {
      company: 'Asapp Desenvolvimento',
      role: 'Arquiteto Principal & Fundador',
      period: '2026 – Presente',
      location: 'Curitiba & Pontal do Paraná, Brasil',
      type: 'Tempo Integral / Liderança',
      highlights: [
        'Liderança técnica em arquitetura de software e estratégia tecnológica, entregando microsserviços, plataformas nativas em nuvem e aplicações multiplataforma de alta performance com Go, Next.js e Flutter.',
        'Arquitetou a Appfy, uma plataforma e marketplace modular de micro-apps aproveitando serviços distribuídos no Google Cloud Platform (GCP).',
        'Criou e mantém ferramentas open-source para desenvolvedores, incluindo o mddd-cli (CLI de desenvolvimento dirigido a diagramas Mermaid) no npm.',
        'Integração de fluxos de trabalho com IA multi-agente (Cursor, Cline, Roo Code, LiteLLM) acelerando pipelines de geração de código, testes e revisão automatizada.'
      ],
      technologies: ['Go (Golang)', 'Next.js', 'Flutter', 'Google Cloud Platform (GCP)', 'mddd-cli', 'LLMs Multi-Agente'],
    },
    {
      company: 'Opah IT',
      role: 'Engenheiro de Software Sênior / Especialista Mobile',
      period: '2026 – Presente',
      location: 'Remoto',
      type: 'Consultoria',
      highlights: [
        'Consultoria e engenharia de soluções críticas de alta performance em Flutter/Android para clientes corporativos de grande porte.',
        'Aplicação de padrões rigorosos de Clean Architecture, diretrizes de qualidade de código e melhores práticas de CI/CD entre equipes distribuídas.',
        'Garantia de segurança de bibliotecas de terceiros para aplicações mobile com Dex-guard, ofuscação avançada e criptografia de ponta a ponta.'
      ],
      technologies: ['Flutter', 'Android', 'Clean Architecture', 'CI/CD', 'Dex-guard', 'Blindagem de Segurança'],
    },
    {
      company: 'Arlequim Technologies',
      role: 'Especialista Flutter & Líder Multiplataforma',
      period: '07/2025 – Presente',
      location: 'Curitiba, Brasil',
      type: 'Tempo Integral',
      highlights: [
        'Liderança técnica no desenvolvimento de soluções multiplataforma com front-ends Flutter de alto rendimento.',
        'Implementação de estratégias de SEO Técnico e AEO (Answer Engine Optimization) para otimização de descoberta de conteúdo por buscadores e motores de resposta de IA.',
        'Migração de arquiteturas de gerenciamento de estado e injeção de dependência, aumentando estabilidade e eliminando débitos técnicos críticos.',
        'Configuração de pipelines de CI/CD focados em testes automatizados de Golden Image para assegurar integridade visual em escala.'
      ],
      technologies: ['Flutter', 'SEO Técnico', 'AEO', 'Testes Golden Image', 'Pipelines CI/CD', 'GCP'],
    },
    {
      company: 'Asapp Desenvolvimento (Fase de Consultoria)',
      role: 'Fundador / Desenvolvedor Líder',
      period: '10/2023 – 03/2024',
      location: 'Curitiba, Brasil',
      type: 'Liderança',
      highlights: [
        'Fundação e liderança de consultoria focada em aplicações corporativas de alta complexidade para Desktop, Mobile e Web.',
        'Desenvolvimento de aplicações web com foco em renderização híbrida, garantindo máxima performance e indexação superior para clientes corporativos.',
        'Liderança na refatoração de MVP para V2 em projetos estratégicos, elevando a avaliação na App Store de 4.2 para 4.87.'
      ],
      technologies: ['Next.js', 'Flutter', 'Renderização Híbrida', 'SSR/ISR', 'Core Web Vitals'],
    },
    {
      company: 'Grupo Data',
      role: 'Arquiteto Flutter & Líder Técnico',
      period: '11/2021 – 10/2022',
      location: 'Curitiba, Brasil',
      type: 'Contrato',
      highlights: [
        'Responsável pela manutenção e evolução da aplicação móvel "Tim Beta", atendendo mais de 1 milhão de usuários ativos.',
        'Orquestrou a refatoração completa da aplicação mobile para alinhamento com a migração crítica do backend corporativo de .NET para Java.'
      ],
      technologies: ['Flutter', '1M+ Usuários Ativos', 'Java', 'Migração .NET', 'Clean Architecture'],
    },
    {
      company: 'Grupo GFT',
      role: 'Desenvolvedor Mobile & Líder Técnico',
      period: '07/2020 – 11/2021',
      location: 'Curitiba, Brasil',
      type: 'Tempo Integral',
      highlights: [
        'Desenvolvimento de funcionalidades críticas para instituições financeiras tier-1 (Banco Original e Banco Votorantim) usando Java, Kotlin e Flutter.',
        'Pioneirismo na implementação de SDKs nativos customizados de teclado com comunicação IPC segura para aplicações bancárias em Flutter.'
      ],
      technologies: ['Flutter', 'Kotlin', 'Java', 'FinTech Bancária', 'Teclados Nativos IPC'],
    },
    {
      company: 'DevMaker Mobile Apps',
      role: 'Desenvolvedor Mobile',
      period: '04/2019 – 07/2020',
      location: 'Curitiba, Brasil',
      type: 'Tempo Integral',
      highlights: [
        'Engenharia mobile em Android (Java/Kotlin), iOS (Swift) e React Native.',
        'Atuou como o primeiro desenvolvedor React Native da empresa, liderando novos projetos de clientes e capacitando as equipes internas.'
      ],
      technologies: ['React Native', 'Swift', 'Kotlin', 'iOS', 'Android'],
    },
  ];
}

export const SKILL_CATEGORIES: SkillCategory[] = [
  {
    title: 'Machine Learning & AI',
    skills: [
      { name: 'Causal LLMs & Transformers', level: 'Core Architect', highlight: true },
      { name: 'PyTorch & torchrun (DDP)', level: 'Cluster Scale', highlight: true },
      { name: 'CUDA & Apple Metal (MPS)', level: 'Acceleration', highlight: true },
      { name: 'Rotary Position Embeddings (RoPE)', level: 'Positional Logic', highlight: true },
      { name: 'RMSNorm & SDPA / FlashAttention', level: 'Kernel Opt', highlight: true },
      { name: 'DistributedDataParallel (DDP)', level: 'Multi-GPU Cluster', highlight: true },
      { name: 'Grouped-Query Attention (GQA)', level: 'Memory Efficient', highlight: true },
      { name: 'Byte-Level BPE Tokenization', level: 'Custom Tokenizer', highlight: true },
      { name: 'SFT Prompt Loss Masking', level: 'Fine-Tuning', highlight: true },
      { name: 'Autonomous Tool-Use Interception', level: 'Agentic Tooling', highlight: true },
      { name: 'Kaggle API Automation', level: 'MLOps Pipelines', highlight: true },
    ],
  },
  {
    title: 'Software Architecture & Systems',
    skills: [
      { name: 'Distributed Microservices', level: 'Enterprise Go', highlight: true },
      { name: 'Clean Architecture & SOLID', level: 'Domain Driven', highlight: true },
      { name: 'Multi-Agent AI Pipelines', level: 'Agentic Workflows', highlight: true },
      { name: 'Spec-Driven Dev (mddd-cli)', level: 'Author & Maintainer', highlight: true },
      { name: 'Micro-apps Architecture', level: 'Modular Scale', highlight: true },
      { name: 'CI/CD & Golden Image Testing', level: 'Automated QA', highlight: true },
      { name: 'Technical SEO & AEO (Answer Engine)', level: 'AI Search Opt', highlight: true },
    ],
  },
  {
    title: 'Languages & Core Tech',
    skills: [
      { name: 'Python (PyTorch / Data Science)', level: 'Senior ML', highlight: true },
      { name: 'Go (Golang)', level: 'Principal Services', highlight: true },
      { name: 'Dart / Flutter (Mobile, Desktop, Web)', level: 'Lead / 8+ Years', highlight: true },
      { name: 'TypeScript / JavaScript', level: 'Full-Stack Modern', highlight: true },
      { name: 'Next.js (App Router, SSR, ISR)', level: 'Web Architect', highlight: true },
      { name: 'Java & Kotlin', level: 'Android & Enterprise', highlight: true },
      { name: 'Swift (iOS Native)', level: 'iOS Native', highlight: true },
      { name: 'SQL & PostgreSQL', level: 'Relational Scale', highlight: true },
      { name: 'Firestore / Redis / MongoDB', level: 'Distributed Caching', highlight: true },
      { name: 'C++ & Metal/Vulkan Shaders', level: 'Low-Level Graphics', highlight: true },
    ],
  },
  {
    title: 'Education & Honors',
    skills: [
      { name: 'B.Sc. in Data Science', level: 'Higher Education', highlight: true },
      { name: 'Postgraduate: Machine Learning & Deep Learning', level: 'Specialization', highlight: true },
      { name: 'EF SET English Certificate C2 Proficient (75/100)', level: 'CEFR C2 Native Equiv', highlight: true },
      { name: 'Go Programming: The Complete Bootcamp', level: 'Certified', highlight: true },
      { name: 'Next.js Specialist (SSR, ISR, Core Web Vitals)', level: 'Certified Specialist', highlight: true },
    ],
  },
];

export function getSkillCategories(lang: 'en' | 'pt' = 'en'): SkillCategory[] {
  if (lang !== 'pt') return SKILL_CATEGORIES;

  return [
    {
      title: 'Machine Learning & IA',
      skills: [
        { name: 'LLMs Causais & Transformers', level: 'Arquiteto Central', highlight: true },
        { name: 'PyTorch & torchrun (DDP)', level: 'Escala em Cluster', highlight: true },
        { name: 'CUDA & Apple Metal (MPS)', level: 'Aceleração de Hardware', highlight: true },
        { name: 'Rotary Position Embeddings (RoPE)', level: 'Lógica Posicional', highlight: true },
        { name: 'RMSNorm & SDPA / FlashAttention', level: 'Otimização de Kernel', highlight: true },
        { name: 'DistributedDataParallel (DDP)', level: 'Cluster Multi-GPU', highlight: true },
        { name: 'Grouped-Query Attention (GQA)', level: 'Eficiência de Memória', highlight: true },
        { name: 'Tokenização BPE em Nível de Byte', level: 'Tokenizador Próprio', highlight: true },
        { name: 'SFT Prompt Loss Masking', level: 'Ajuste Fino (SFT)', highlight: true },
        { name: 'Interceptação Autônoma de Ferramentas', level: 'Tooling Agêntico', highlight: true },
        { name: 'Automação Kaggle API', level: 'Pipelines MLOps', highlight: true },
      ],
    },
    {
      title: 'Arquitetura de Software & Sistemas',
      skills: [
        { name: 'Microsserviços Distribuídos', level: 'Go Corporativo', highlight: true },
        { name: 'Clean Architecture & SOLID', level: 'Orientado a Domínio', highlight: true },
        { name: 'Pipelines de IA Multi-Agente', level: 'Workflows Agênticos', highlight: true },
        { name: 'Spec-Driven Dev (mddd-cli)', level: 'Autor & Mantenedor', highlight: true },
        { name: 'Arquitetura de Micro-aplicações', level: 'Escala Modular', highlight: true },
        { name: 'CI/CD & Testes Golden Image', level: 'QA Automatizado', highlight: true },
        { name: 'SEO Técnico & AEO (Answer Engine)', level: 'Otimização p/ IA', highlight: true },
      ],
    },
    {
      title: 'Linguagens & Tecnologias Centrais',
      skills: [
        { name: 'Python (PyTorch / Ciência de Dados)', level: 'ML Sênior', highlight: true },
        { name: 'Go (Golang)', level: 'Serviços Principais', highlight: true },
        { name: 'Dart / Flutter (Mobile, Desktop, Web)', level: 'Líder / 8+ Anos', highlight: true },
        { name: 'TypeScript / JavaScript', level: 'Full-Stack Moderno', highlight: true },
        { name: 'Next.js (App Router, SSR, ISR)', level: 'Arquiteto Web', highlight: true },
        { name: 'Java & Kotlin', level: 'Android & Corporativo', highlight: true },
        { name: 'Swift (iOS Nativo)', level: 'iOS Nativo', highlight: true },
        { name: 'SQL & PostgreSQL', level: 'Escala Relacional', highlight: true },
        { name: 'Firestore / Redis / MongoDB', level: 'Cache Distribuído', highlight: true },
        { name: 'C++ & Shaders Metal/Vulkan', level: 'Gráficos Baixo Nível', highlight: true },
      ],
    },
    {
      title: 'Formação & Certificações',
      skills: [
        { name: 'Bacharelado em Ciência de Dados', level: 'Ensino Superior', highlight: true },
        { name: 'Pós-Graduação: Machine Learning & Deep Learning', level: 'Especialização', highlight: true },
        { name: 'Certificado de Inglês EF SET C2 Proficiente (75/100)', level: 'Equivalente Nativo', highlight: true },
        { name: 'Programação Go: Bootcamp Completo', level: 'Certificado', highlight: true },
        { name: 'Especialista Next.js (SSR, ISR, Core Web Vitals)', level: 'Especialista Certificado', highlight: true },
      ],
    },
  ];
}
