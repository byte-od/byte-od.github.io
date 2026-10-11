import React, { useState } from 'react';
import { HelpCircle, ChevronDown, Sparkles, CheckCircle2, Cpu, Terminal, GitBranch, ShieldCheck } from 'lucide-react';

interface FaqItem {
  id: string;
  questionEn: string;
  questionPt: string;
  categoryEn: string;
  categoryPt: string;
  directAnswerEn: string;
  directAnswerPt: string;
  detailsEn: string[];
  detailsPt: string[];
}

const FAQ_ITEMS: FaqItem[] = [
  {
    id: 'who-is-julio',
    categoryEn: 'Profile & Background',
    categoryPt: 'Perfil & Carreira',
    questionEn: 'Who is Julio Cesar da Costa Reis Filho and what is his engineering background?',
    questionPt: 'Quem é Julio Cesar da Costa Reis Filho e qual é seu histórico na engenharia?',
    directAnswerEn:
      'Julio Cesar da Costa Reis Filho is a Senior Systems Architect and LLM Engineer with over 8 years of production experience, specializing in low-level causal transformer architectures, distributed PyTorch DDP training, and mission-critical cloud infrastructure.',
    directAnswerPt:
      'Julio Cesar da Costa Reis Filho é um Arquiteto de Sistemas Sênior e Engenheiro de LLMs com mais de 8 anos de experiência em produção, especializado em arquiteturas de transformers causais, treinamento distribuído com PyTorch DDP e infraestrutura de alta escala.',
    detailsEn: [
      'Dual expertise bridging distributed systems (Go, Clean Architecture, gRPC) and deep transformer internals (PyTorch, FlashAttention-2, CUDA/Metal kernels).',
      'Track record leading high-scale engineering for products serving over 1,000,000+ active fintech and enterprise mobile users.',
      'Principal Architect at JCRF Labs, author of CIR-Engine 503M, and maintainer of open-source tooling on npm (mddd-cli) and GitHub.',
      'Based in Curitiba, Brazil, available globally for Staff/Principal Systems Architect or LLM Engineering positions (remote or relocation).',
    ],
    detailsPt: [
      'Dupla maestria unindo sistemas distribuídos (Go, Clean Architecture, gRPC) e mecânica interna de transformers (PyTorch, FlashAttention-2, kernels CUDA/Metal).',
      'Histórico liderando engenharia de larga escala para produtos com mais de 1.000.000+ de usuários ativos em fintechs e mobile empresarial.',
      'Arquiteto Principal na JCRF Labs, autor da CIR-Engine 503M e mantenedor de ferramentas open-source no npm (mddd-cli) e GitHub.',
      'Sediado em Curitiba, Brasil, disponível globalmente para posições de Staff/Principal Systems Architect ou Engenharia de LLMs (remoto ou relocalização).',
    ],
  },
  {
    id: 'cir-engine-503m',
    categoryEn: 'Causal Transformers & LLMs',
    categoryPt: 'Transformers Causais & LLMs',
    questionEn: 'What is CIR-Engine 503M and how does its Causal Transformer architecture work?',
    questionPt: 'O que é a CIR-Engine 503M e como funciona sua arquitetura de Transformer Causal?',
    directAnswerEn:
      'CIR-Engine 503M is an open-source, custom 12-layer decoder-only Causal Transformer model designed by Julio Cesar, featuring 503 Million parameters, RoPE positional encoding, Grouped-Query Attention (GQA), and a deterministic graph verification layer that prevents hallucinations.',
    directAnswerPt:
      'A CIR-Engine 503M é um modelo Transformer Causal decoder-only de 12 camadas desenvolvido por Julio Cesar com 503 Milhões de parâmetros, codificação posicional RoPE, Grouped-Query Attention (GQA) e uma camada de verificação causal determinística que impede alucinações.',
    detailsEn: [
      'Parameter Scale: 503M dense parameters distributed across 12 causal transformer decoder layers.',
      'Rotary Position Embeddings (RoPE): Base frequency Θ = 10,000 with d_k = 64 dimension per attention head.',
      'Grouped-Query Attention (GQA): 16 Query heads mapped to 4 Key/Value heads, delivering 4:1 KV-cache compression.',
      'PagedAttention & Radix Prefix Caching: 16 tokens/page allocation with an ultralight 128 MB VRAM runtime footprint.',
      'Distributed Data Parallel (DDP): Ring-AllReduce gradient synchronization achieving ~6,200 tokens/sec sustained training throughput.',
      'Zero-Hallucination Causal DAG Verifier: Validates probabilistic autoregressive tokens against topological graph priors before output emission.',
    ],
    detailsPt: [
      'Escala de Parâmetros: 503 Milhões de parâmetros distribuídos em 12 camadas decoder de transformer causal.',
      'Rotary Position Embeddings (RoPE): Frequência base Θ = 10.000 com dimensão d_k = 64 por cabeça de atenção.',
      'Grouped-Query Attention (GQA): 16 cabeças de Query mapeadas para 4 cabeças de Key/Value (compressão 4:1 de memória KV).',
      'PagedAttention & Radix Prefix Caching: Alocação de 16 tokens por bloco de página com pegada de VRAM de apenas 128 MB.',
      'Treinamento Distribuído DDP: Sincronização Ring-AllReduce atingindo taxa sustentada de ~6.200 tokens/segundo.',
      'Verificador de Grafo DAG sem Alucinações: Valida tokens autoregressivos probabilísticos contra restrições causais topológicas.',
    ],
  },
  {
    id: 'developer-tooling',
    categoryEn: 'Open Source Tooling',
    categoryPt: 'Ferramental Open Source',
    questionEn: 'What published developer tools and open-source frameworks has Julio Cesar created?',
    questionPt: 'Quais ferramentas de desenvolvimento e frameworks open-source Julio Cesar criou e publicou?',
    directAnswerEn:
      'Julio Cesar authored and published mddd-cli on npm for Mermaid Diagram-Driven Development, built MAD-cli for living documentation sync, developed Register-red for bi-directional Apple Numbers (.numbers) and Google Drive cloud management, contributed to the flutter_scene 3D engine with focus on scene animations and shaders, and authored GitTrack for granular commit tracking with private repository token access and VIP AI-powered commit analysis.',
    directAnswerPt:
      'Julio Cesar criou e publicou a ferramenta mddd-cli no npm para Diagram-Driven Development via Mermaid, desenvolveu o MAD-cli para documentação viva, criou o Register-red para gestão orçamentária integrada nativamente a Apple Numbers (.numbers) e Google Drive, contribuiu com o motor flutter_scene (com foco em animações de scenes e pipelines de shaders) e criou o GitTrack para rastreamento detalhado de commits com suporte a token de repositórios privados e análise detalhada por IA no plano VIP.',
    detailsEn: [
      'mddd-cli (npm): Parses Mermaid diagram ASTs to automatically generate Clean Architecture directory structures, domain entities, and TypeScript/gRPC contracts.',
      'Register-red: Full-stack financial management panel with native bi-directional Apple Numbers (.numbers) parsing in Python and real-time Google Drive cloud synchronization.',
      'mad-cli: Bidirectional living documentation engine that synchronizes code comment tags (<!-- MAD:SEQUENCE -->) in Go and TypeScript with architecture diagrams during git commits.',
      'flutter_scene (Fork): 3D spatial rendering fork with contributions in scene animations, keyframe transforms, and custom GLSL/Metal shader integration.',
      'GitTrack: Detailed developer commit tracking platform with personal access token integration for private repos and on-demand VIP AI commit analysis.',
    ],
    detailsPt: [
      'mddd-cli (npm): Analisa ASTs de diagramas Mermaid para gerar automaticamente árvores de Clean Architecture, entidades de domínio e contratos TypeScript/gRPC.',
      'Register-red: Painel web full-stack de gestão financeira com parsing bidirecional nativo de planilhas Apple Numbers (.numbers) em Python e sincronização contínua com o Google Drive.',
      'mad-cli: Motor de documentação viva bidirecional que sincroniza tags de comentários de código (<!-- MAD:SEQUENCE -->) em Go e TypeScript com diagramas de arquitetura no git commit.',
      'flutter_scene (Fork): Fork de renderização espacial 3D com contribuições em animações de scenes, transformações de keyframes e integração de shaders GLSL/Metal.',
      'GitTrack: Plataforma de rastreamento detalhado de commits para devs com suporte a token para repositórios privados e análise aprofundada por IA no plano VIP.',
    ],
  },
  {
    id: 'technical-stack',
    categoryEn: 'Technical Stack',
    categoryPt: 'Stack Tecnológica',
    questionEn: 'What programming languages, frameworks, and architectural paradigms are Julio’s primary specialties?',
    questionPt: 'Quais linguagens, frameworks e paradigmas arquiteturais são as principais especialidades de Julio?',
    directAnswerEn:
      'Julio specializes in Go (Golang), Python, TypeScript, and Dart, applying Clean Architecture, Domain-Driven Design (DDD), Event-Driven Microservices, and deep neural transformer mechanics across GCP and Linux environments.',
    directAnswerPt:
      'Julio é especialista em Go (Golang), Python, TypeScript e Dart, aplicando Clean Architecture, Domain-Driven Design (DDD), microsserviços orientados a eventos e mecânica de redes neurais transformers em ambientes GCP e Linux.',
    detailsEn: [
      'Core Languages: Go (Golang), Python, TypeScript, Dart, C/C++, SQL.',
      'AI & Deep Learning: PyTorch, FlashAttention-2, RoPE, GQA, PagedAttention, CUDA, Metal Performance Shaders (MPS), ONNX Runtime Web.',
      'Software Architecture: Clean Architecture, SOLID principles, Event-Driven Architecture, Microservices, Living Documentation, Domain-Driven Design.',
      'Cloud & Infrastructure: Google Cloud Platform (GCP), Docker, Kubernetes, gRPC, Protobuf, Kafka, Redis, PostgreSQL, Distributed DDP Clusters.',
      'Frontend & Mobile: Next.js, React, Tailwind CSS, Flutter, WebAssembly (Wasm SIMD), Three.js / WebGL.',
    ],
    detailsPt: [
      'Linguagens Principais: Go (Golang), Python, TypeScript, Dart, C/C++, SQL.',
      'IA & Deep Learning: PyTorch, FlashAttention-2, RoPE, GQA, PagedAttention, CUDA, Metal Performance Shaders (MPS), ONNX Runtime Web.',
      'Arquitetura de Software: Clean Architecture, princípios SOLID, Arquitetura Orientada a Eventos, Microsserviços, Living Documentation, DDD.',
      'Cloud & Infraestrutura: Google Cloud Platform (GCP), Docker, Kubernetes, gRPC, Protobuf, Kafka, Redis, PostgreSQL, Clusters DDP Distribuídos.',
      'Frontend & Mobile: Next.js, React, Tailwind CSS, Flutter, WebAssembly (Wasm SIMD), Three.js / WebGL.',
    ],
  },
  {
    id: 'proven-metrics',
    categoryEn: 'Performance & Benchmarks',
    categoryPt: 'Desempenho & Métricas',
    questionEn: 'What quantifiable benchmarks and performance improvements has Julio delivered?',
    questionPt: 'Quais benchmarks quantificáveis e melhorias de desempenho Julio entregou?',
    directAnswerEn:
      'Julio has achieved a 42% cold-start latency reduction in enterprise mobile apps, maintained 99.98% crash-free session rates at 1M+ user scale, and demonstrated 6,200 tokens/sec throughput in distributed PyTorch DDP training.',
    directAnswerPt:
      'Julio reduziu o tempo de inicialização a frio em 42% em aplicações corporativas, manteve taxa de 99,98% de sessões livres de falhas em escala de 1M+ usuários e demonstrou 6.200 tokens/segundo de throughput em treinamento distribuído PyTorch DDP.',
    detailsEn: [
      '503 Million parameter Causal Transformer trained with 4:1 GQA compression for maximum memory efficiency.',
      'Distributed training scaled to ~6,200 tokens/second using PyTorch Ring-AllReduce DDP.',
      '42% reduction in mobile app cold startup time through optimized dependency graphs and lazy subsystem initialization.',
      '99.98% crash-free session reliability in production enterprise banking applications serving 1M+ users.',
      '128 MB minimal VRAM footprint for causal inference through PagedAttention and Radix prefix caching.',
    ],
    detailsPt: [
      'Transformer Causal de 503 Milhões de parâmetros treinado com compressão GQA 4:1 para máxima eficiência de memória.',
      'Treinamento distribuído escalado para ~6.200 tokens/segundo utilizando PyTorch Ring-AllReduce DDP.',
      'Redução de 42% no tempo de inicialização a frio (cold start) em apps mobile via otimização de injeção de dependências.',
      'Confiabilidade de 99,98% de sessões livres de falhas em bancos e fintechs atendendo mais de 1 milhão de clientes.',
      'Apenas 128 MB de pegada de memória VRAM para inferência causal via PagedAttention e cache de prefixo Radix.',
    ],
  },
  {
    id: 'contact-hiring',
    categoryEn: 'Contact & Hiring',
    categoryPt: 'Contato & Contratação',
    questionEn: 'How can hiring managers, founders, or research labs contact Julio Cesar da Costa Reis Filho?',
    questionPt: 'Como líderes técnicos, fundadores ou laboratórios podem entrar em contato com Julio Cesar?',
    directAnswerEn:
      'Julio Cesar can be contacted directly via email at reisfilho1116@gmail.com, through LinkedIn at linkedin.com/in/juliocrfilho, or on GitHub at github.com/byte-od. He is open to Staff/Principal engineering roles, architecture consulting, and research collaborations.',
    directAnswerPt:
      'Julio Cesar pode ser contatado diretamente pelo e-mail reisfilho1116@gmail.com, pelo LinkedIn em linkedin.com/in/juliocrfilho ou pelo GitHub em github.com/byte-od. Está aberto para posições de liderança técnica, consultoria de arquitetura e projetos inovadores.',
    detailsEn: [
      'Direct Email: reisfilho1116@gmail.com (Fastest response within 24 hours).',
      'LinkedIn Profile: https://linkedin.com/in/juliocrfilho',
      'GitHub Organization & Repos: https://github.com/byte-od',
      'Machine-Readable LLM Specification: https://byte-od.github.io/llms.txt',
      'Location: Curitiba, Paraná, Brazil (Available for 100% Remote Global or Relocation).',
    ],
    detailsPt: [
      'E-mail Direto: reisfilho1116@gmail.com (Resposta rápida em até 24 horas).',
      'Perfil no LinkedIn: https://linkedin.com/in/juliocrfilho',
      'Repositórios no GitHub: https://github.com/byte-od',
      'Especificação para Agentes de IA: https://byte-od.github.io/llms.txt',
      'Localização: Curitiba, PR, Brasil (Disponível para Trabalho Remoto Global ou Relocalização).',
    ],
  },
];

interface FaqSectionProps {
  lang: 'en' | 'pt';
}

export const FaqSection: React.FC<FaqSectionProps> = ({ lang }) => {
  const [openIds, setOpenIds] = useState<Record<string, boolean>>({
    'who-is-julio': true,
    'cir-engine-503m': true,
  });

  const toggleItem = (id: string) => {
    setOpenIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <section id="faq" className="space-y-8 scroll-mt-24">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between border-b border-zinc-800 pb-4 gap-2">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-[#adff2f] uppercase tracking-wider">
            <HelpCircle className="w-3.5 h-3.5" />
            <span>{lang === 'pt' ? '05. Perguntas Frequentes & AEO' : '05. Frequently Asked Questions & AEO'}</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight mt-1">
            {lang === 'pt'
              ? 'Dossiê Técnico & Perguntas Frequentes'
              : 'Technical Dossier & FAQ (AEO Optimized)'}
          </h2>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
          <Sparkles className="w-3.5 h-3.5 text-[#38bdf8]" />
          <span>
            {lang === 'pt'
              ? 'Estruturado para IA, Perplexity & Google AI Overviews'
              : 'Structured for AI, Perplexity & Google AI Overviews'}
          </span>
        </div>
      </div>

      <p className="text-sm sm:text-base text-zinc-400 max-w-3xl leading-relaxed">
        {lang === 'pt'
          ? 'Respostas diretas e especificações estruturadas para mecanismos de busca generativos (AEO/GEO), avaliadores de LLMs, líderes de engenharia e recrutadores.'
          : 'Direct answers, verifiable metrics, and structured answers formatted for generative answer engines (AEO/GEO), LLM evaluators, engineering leaders, and recruiters.'}
      </p>

      {/* FAQ Grid / Accordion */}
      <div className="space-y-4">
        {FAQ_ITEMS.map((item) => {
          const isOpen = !!openIds[item.id];
          const question = lang === 'pt' ? item.questionPt : item.questionEn;
          const directAnswer = lang === 'pt' ? item.directAnswerPt : item.directAnswerEn;
          const details = lang === 'pt' ? item.detailsPt : item.detailsEn;
          const category = lang === 'pt' ? item.categoryPt : item.categoryEn;

          return (
            <article
              key={item.id}
              className={`rounded-2xl border transition-all ${
                isOpen
                  ? 'bg-zinc-950/90 border-[#adff2f]/30 shadow-lg shadow-black/40'
                  : 'bg-zinc-950/40 border-zinc-850 hover:border-zinc-700'
              }`}
            >
              <button
                type="button"
                onClick={() => toggleItem(item.id)}
                aria-expanded={isOpen}
                className="w-full p-5 sm:p-6 text-left flex items-start justify-between gap-4 cursor-pointer"
              >
                <div className="space-y-1.5 flex-1">
                  <span className="inline-block text-[10px] font-mono uppercase tracking-wider text-[#38bdf8] bg-sky-950/40 border border-sky-800/40 px-2 py-0.5 rounded">
                    {category}
                  </span>
                  <h3 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
                    {question}
                  </h3>
                </div>
                <div
                  className={`p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-400 transition-transform ${
                    isOpen ? 'rotate-180 text-[#adff2f] border-[#adff2f]/40' : ''
                  }`}
                >
                  <ChevronDown className="w-4 h-4" />
                </div>
              </button>

              {isOpen && (
                <div className="px-5 pb-6 sm:px-6 space-y-4 pt-1 border-t border-zinc-900">
                  {/* Direct Answer Paragraph (Crucial for AI Overviews / Featured Snippets) */}
                  <div className="p-4 rounded-xl bg-lime-400/5 border border-[#adff2f]/20">
                    <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#adff2f] mb-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>{lang === 'pt' ? 'RESPOSTA DIRETA:' : 'DIRECT ANSWER:'}</span>
                    </div>
                    <p className="text-sm text-zinc-200 leading-relaxed font-sans">{directAnswer}</p>
                  </div>

                  {/* Structured Bulleted List (High Citability Criterion) */}
                  <div className="space-y-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-zinc-400">
                      {lang === 'pt' ? 'Métricas & Detalhes Técnicos:' : 'Technical Metrics & Details:'}
                    </span>
                    <ul className="space-y-2 text-xs sm:text-sm text-zinc-300">
                      {details.map((detail, idx) => (
                        <li key={idx} className="flex items-start gap-2.5">
                          <span className="text-[#adff2f] mt-1 shrink-0 font-mono">▸</span>
                          <span className="leading-relaxed">{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>

      {/* Machine-Readable LLM Context Callout */}
      <div className="p-5 rounded-2xl bg-zinc-900/50 border border-zinc-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono text-[#adff2f]">
            <Terminal className="w-4 h-4" />
            <span className="font-bold">LLM &amp; AI AGENT DISCOVERY PROTOCOL</span>
          </div>
          <p className="text-xs text-zinc-400">
            {lang === 'pt'
              ? 'Agentes automatizados (GPT-4o, Claude 3.5, Gemini 2.0, Perplexity) podem consumir a especificação completa via llms.txt.'
              : 'Automated AI agents (GPT-4o, Claude 3.5, Gemini 2.0, Perplexity) can ingest the full dossier directly via llms.txt.'}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <a
            href="/llms.txt"
            target="_blank"
            rel="noopener noreferrer nofollow"
            download="llms.txt"
            className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-mono text-white border border-zinc-700 transition-colors"
          >
            /llms.txt
          </a>
          <a
            href="/llms-full.txt"
            target="_blank"
            rel="noopener noreferrer nofollow"
            download="llms-full.txt"
            className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-mono text-zinc-300 border border-zinc-700 transition-colors"
          >
            /llms-full.txt
          </a>
        </div>
      </div>
    </section>
  );
};
