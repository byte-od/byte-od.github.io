import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  X,
  Send,
  Sparkles,
  Zap,
  PlayCircle,
  ExternalLink,
  Github,
  Mail,
  RefreshCw,
  Cpu,
  Layers,
  ChevronDown,
  Minimize2,
  Maximize2,
} from 'lucide-react';

interface Message {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  links?: { label: string; url: string; icon?: 'colab' | 'github' | 'ext' }[];
}

interface PortfolioAgentWidgetProps {
  lang?: 'en' | 'pt';
}

export const PortfolioAgentWidget: React.FC<PortfolioAgentWidgetProps> = ({ lang = 'en' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const initialGreeting: Message = {
    id: 'msg-0',
    sender: 'agent',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    text:
      lang === 'pt'
        ? 'Olá! Sou o agente virtual do portfólio do Julio Cesar Reis Filho. Posso te apresentar sua trajetória sênior, detalhar suas criações de IA (como o MoS 72B em 16GB e o motor AURA) ou explicar a arquitetura de Scar Cache que viabilizou minha presença aqui no seu navegador. Sobre o que gostaria de conversar?'
        : 'Hello! I am Julio Cesar Reis Filho\'s portfolio agent. I can introduce Julio\'s background, walk you through his core systems breakthroughs (such as the 72B MoS on 16GB GPU and AURA Deep RL engine), or explain the Scar Cache architecture that powers this conversation directly in your browser. What would you like to explore?',
    links: [
      {
        label: lang === 'pt' ? '⚡ Live Test Colab (72B + SCache)' : '⚡ Live Test Colab (72B + SCache)',
        url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
        icon: 'colab',
      },
    ],
  };

  // Reset or change language
  useEffect(() => {
    setMessages([initialGreeting]);
  }, [lang]);

  // Auto-scroll
  useEffect(() => {
    if (isOpen && !isMinimized) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, isMinimized, isTyping]);

  // Focus input when open
  useEffect(() => {
    if (isOpen && !isMinimized) {
      setTimeout(() => inputRef.current?.focus(), 150);
      setHasUnread(false);
    }
  }, [isOpen, isMinimized]);

  // Quick Chips
  const quickPrompts =
    lang === 'pt'
      ? [
          { label: '👤 Quem é o Julio?', query: 'Quem é o Julio e qual a experiência dele?' },
          { label: '⚡ MoS & SCache 72B', query: 'O que é o Mixture of Scars (MoS) e o SCache?' },
          { label: '🧠 Como você roda no browser?', query: 'Como a arquitetura MoS e SCache viabiliza você no navegador?' },
          { label: '🧩 Motor AURA (Sistema 1)', query: 'Explique o projeto AURA (Deep Reinforcement Learning no Cubo de Rubik)' },
          { label: '📬 Como entrar em contato?', query: 'Como posso entrar em contato ou contratar o Julio?' },
        ]
      : [
          { label: '👤 Who is Julio?', query: 'Who is Julio and what is his background?' },
          { label: '⚡ MoS & SCache 72B', query: 'What is Mixture of Scars (MoS) and SCache?' },
          { label: '🧠 How do you run in-browser?', query: 'How does the MoS and SCache architecture enable you in the browser?' },
          { label: '🧩 AURA Engine (System 1)', query: 'Explain the AURA Deep Reinforcement Learning engine' },
          { label: '📬 How to contact?', query: 'How can I contact or hire Julio?' },
        ];

  // Natural Knowledge Engine
  const generateAgentResponse = (userQuery: string): { text: string; links?: Message['links'] } => {
    const q = userQuery.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');

    // 1. Quem é o Julio / Bio / Experiência
    if (q.includes('quem e') || q.includes('who is') || q.includes('julio') || q.includes('trajetoria') || q.includes('experiencia') || q.includes('background') || q.includes('carreira') || q.includes('sobre')) {
      if (lang === 'pt') {
        return {
          text: 'Julio Cesar da Costa Reis Filho é um Arquiteto de Sistemas Sênior e Engenheiro de LLMs com mais de 8 anos de experiência em engenharia de software de ponta. Ele combina domínio profundo de sistemas de baixo nível (Go, C++, Rust, CUDA/Metal) com arquiteturas neurais avançadas (Transformers Causais, PyTorch DDP, RL e quantização em hardware). É baseado em Curitiba/Pontal do Paraná, Brasil, e está aberto a posições remotas globais ou relocalização internacional.',
          links: [
            { label: 'LinkedIn do Julio', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
            { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
          ],
        };
      }
      return {
        text: 'Julio Cesar da Costa Reis Filho is a Senior Systems Architect and LLM Systems Engineer with over 8 years of engineering experience. He bridges low-level systems programming (Go, C++, Rust, CUDA/Metal) with deep learning architectures (Causal Transformers, Distributed PyTorch DDP, Embodied RL, and hardware-level quantization). Based in Curitiba, Brazil, he is open to global remote roles and international relocation.',
        links: [
          { label: "Julio's LinkedIn", url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
          { label: 'GitHub @byte-od', url: 'https://github.com/byte-od', icon: 'github' },
        ],
      };
    }

    // 2. MoS & SCache 72B / 100k tokens
    if (q.includes('mos') || q.includes('scache') || q.includes('mixture of scars') || q.includes('72b') || q.includes('70b') || q.includes('16gb') || q.includes('100.000') || q.includes('100k') || q.includes('kv cache') || q.includes('kv-cache')) {
      if (lang === 'pt') {
        return {
          text: 'O Mixture of Scars (MoS) & SCache é uma arquitetura pioneira que resolve simultaneamente os dois maiores gargalos físicos de LLMs:\n\n1. O MoS resolve os PESOS: executa todas as 80 camadas de um 72B (dimensão 8.192, MLP 29.568) em uma única GPU de 16 GB consumindo apenas 2.40 GB de VRAM via desquantização streaming INT4 em tempo real.\n\n2. O SCache resolve o CONTEXTO: substitui o KV Cache tradicional (que explodiria para 32.8 GB a 100.000 tokens e causaria Fatal OOM) por uma memória associativa multi-slot constante O(1) de apenas 34.8 KB residente no Cache L1, com mais de 99.999% de economia de VRAM!',
          links: [
            {
              label: 'Abrir Live Test no Google Colab',
              url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
              icon: 'colab',
            },
            {
              label: 'Repositório GitHub (MoS & SCache)',
              url: 'https://github.com/byte-od/mixture-of-scars',
              icon: 'github',
            },
          ],
        };
      }
      return {
        text: 'Mixture of Scars (MoS) & SCache is a dual breakthrough systems architecture solving both fundamental physical walls of giant foundation models:\n\n1. MoS solves MODEL WEIGHTS: executes all 80 transformer layers of 72B parameter geometry (8192 hidden dim, 29568 MLP dim) on a single 16GB GPU with peak VRAM capped strictly under 2.40 GB via real-time INT4 streaming dequantization.\n\n2. SCache solves CONTEXT MEMORY: replaces the traditional linear O(T) KV Cache (which demands 32.8 GB VRAM at 100,000 tokens causing fatal OOM) with an exact 34.8 KB O(1) multi-slot associative tensor strictly resident in L1 Cache, saving >99.999% VRAM!',
        links: [
          {
            label: 'Launch Live Test on Google Colab',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
          {
            label: 'GitHub Repo (MoS & SCache)',
            url: 'https://github.com/byte-od/mixture-of-scars',
            icon: 'github',
          },
        ],
      };
    }

    // 3. Como roda no browser / Como o agente funciona
    if (q.includes('browser') || q.includes('navegador') || q.includes('como voce') || q.includes('como funciona') || q.includes('arquitetura que viabilizou') || q.includes('how do you run') || q.includes('how it works')) {
      if (lang === 'pt') {
        return {
          text: 'Essa é a beleza do design: em LLMs convencionais, rodar no navegador trava a aba porque o WebGPU tem um limite de buffer de 2 a 4 GB e o KV Cache incha a cada token gerado. Com o SCache, a memória do contexto é de apenas 34.8 KB (cabendo 100% na Shared Memory SRAM da GPU), mantendo a memória da aba perfeitamente plana mesmo após 100.000 tokens! E com o MoS, os pesos transitam em blocos INT4 por streaming via OPFS local sem acúmulo de VRAM.',
          links: [
            {
              label: 'Ver Live Test do SCache (100k tokens)',
              url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
              icon: 'colab',
            },
          ],
        };
      }
      return {
        text: 'This highlights the elegance of the architecture: standard in-browser LLMs crash Chrome/Safari tabs because WebGPU enforces a 2-4 GB buffer allocation ceiling and the KV Cache continuously inflates. With SCache, context memory is strictly 34.8 KB (fitting 100% inside GPU Workgroup Shared Memory SRAM), keeping tab memory completely flat across 100,000 tokens! With MoS, weights stream in INT4 blocks via local OPFS without VRAM bloat.',
        links: [
          {
            label: 'Inspect SCache 100k Token Live Test',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
        ],
      };
    }

    // 4. AURA / Sistema 1 / Rubik
    if (q.includes('aura') || q.includes('sistema 1') || q.includes('system 1') || q.includes('rubik') || q.includes('reforco') || q.includes('reinforcement')) {
      if (lang === 'pt') {
        return {
          text: 'AURA (Universal System 1 Engine) é o motor flagship de Deep Reinforcement Learning do Julio. É um kernel de reflexo reativo que resolve estados combinatórios enganosos (Cubo de Rubik até profundidade D10 com 232 bilhões de estados) em apenas 0.23ms em CPU pura sem árvore de busca (Zero MCTS). Combina política recorrente GRU, treinamento com 16 professores SIL e exportação WebAssembly/ONNX.',
          links: [
            { label: 'Repositório GitHub (AURA / System 1)', url: 'https://github.com/byte-od/system_one', icon: 'github' },
          ],
        };
      }
      return {
        text: 'AURA (Universal System 1 Engine) is Julio\'s flagship Embodied Deep Reinforcement Learning engine. It is a sub-millisecond reactive reflex kernel solving deceptive combinatorial spaces (Rubik\'s Cube up to Depth 10 across 232 Billion states) in 0.23ms on bare CPU with zero tree search (No MCTS). It combines recurrent GRU policy, 16-worker teacher SIL, and client-side ONNX/WebAssembly export.',
        links: [
          { label: 'GitHub Repository (AURA / System 1)', url: 'https://github.com/byte-od/system_one', icon: 'github' },
        ],
      };
    }

    // 5. CIR-Engine / Cir-Jev / Causal
    if (q.includes('cir') || q.includes('cir-engine') || q.includes('503m') || q.includes('causal') || q.includes('transformer')) {
      if (lang === 'pt') {
        return {
          text: 'O CIR-Engine & Cir-Jev é um Causal Transformer autorregressivo de 503 milhões de parâmetros projetado e treinado do zero em PyTorch sem bibliotecas prontas. Conta com Grouped-Query Attention (GQA), Rotary Embeddings (RoPE), throughput DDP de ~6.200 tokens/seg e um verificador causal determinístico em grafo que impede alucinações em chamadas de ferramentas.',
          links: [
            { label: 'Repositório CIR-Jev no GitHub', url: 'https://github.com/byte-od/cir-jev', icon: 'github' },
          ],
        };
      }
      return {
        text: 'CIR-Engine & Cir-Jev is a proprietary 503M-parameter causal autoregressive Transformer built and trained completely from scratch in PyTorch without high-level wrappers. It features Grouped-Query Attention (GQA), RoPE, ~6,200 t/s DDP throughput, and a deterministic causal DAG verifier eliminating hallucinations during tool use.',
        links: [
          { label: 'CIR-Jev on GitHub', url: 'https://github.com/byte-od/cir-jev', icon: 'github' },
        ],
      };
    }

    // 6. Ferramentas / mddd-cli / mad-cli / npm
    if (q.includes('mddd') || q.includes('mad') || q.includes('cli') || q.includes('npm') || q.includes('ferramenta') || q.includes('tool')) {
      if (lang === 'pt') {
        return {
          text: 'O Julio é o criador e mantenedor do mddd-cli (publicado no npm com "npx mddd-cli --init"), que compila diagramas formais de arquitetura e matrizes de decisão diretamente em código Clean Architecture sem desvios de especificação. Também criou o mad-cli, um parser bidirecional de documentação viva que atualiza diagramas Mermaid a cada git commit.',
          links: [
            { label: 'Pacote mddd-cli no NPM', url: 'https://www.npmjs.com/package/mddd-cli', icon: 'ext' },
            { label: 'Repositório mddd-cli', url: 'https://github.com/byte-od/mddd-cli', icon: 'github' },
          ],
        };
      }
      return {
        text: 'Julio is the creator and maintainer of mddd-cli (published on npm as "npx mddd-cli --init"), which compiles formal architecture diagrams and decision matrices into production Clean Architecture trees without specification drift. He also created mad-cli, a living documentation parser keeping Mermaid diagrams automatically synchronized on git commit.',
        links: [
          { label: 'mddd-cli on NPM', url: 'https://www.npmjs.com/package/mddd-cli', icon: 'ext' },
          { label: 'mddd-cli Repository', url: 'https://github.com/byte-od/mddd-cli', icon: 'github' },
        ],
      };
    }

    // 7. Contato / Contratação / Email / Vagas
    if (q.includes('contato') || q.includes('contact') || q.includes('contratar') || q.includes('hire') || q.includes('email') || q.includes('linkedin') || q.includes('salario') || q.includes('vaga') || q.includes('entrevista')) {
      if (lang === 'pt') {
        return {
          text: 'O Julio está disponível para posições de Liderança Técnica, Arquiteto de Sistemas Sênior e Engenheiro de LLMs/Transformers (remoto global ou relocalização internacional). Você pode contatá-lo diretamente pelos canais oficiais:\n\n• E-mail: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• Telefone/WhatsApp: +55 41 9 9626 8203\n• GitHub: github.com/byte-od',
          links: [
            { label: 'Enviar E-mail Direto', url: 'mailto:reisfilho1116@gmail.com', icon: 'ext' },
            { label: 'Perfil no LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
          ],
        };
      }
      return {
        text: 'Julio is actively open to Principal/Lead Systems Architect and Senior LLM Systems Engineer opportunities (global remote or international relocation). You can reach him directly through his primary coordinates:\n\n• Direct Email: reisfilho1116@gmail.com\n• LinkedIn: linkedin.com/in/juliocrfilho\n• Phone: +55 41 9 9626 8203\n• GitHub: github.com/byte-od',
        links: [
          { label: 'Send Direct Email', url: 'mailto:reisfilho1116@gmail.com', icon: 'ext' },
          { label: 'Connect on LinkedIn', url: 'https://linkedin.com/in/juliocrfilho', icon: 'ext' },
        ],
      };
    }

    // 8. Saudação / Resposta Padrão
    if (q.includes('ola') || q.includes('oi') || q.includes('hello') || q.includes('hi') || q.includes('hey') || q.includes('bom dia') || q.includes('boa tarde') || q.includes('boa noite')) {
      if (lang === 'pt') {
        return {
          text: 'Olá! Prazer em falar com você. Estou pronto para responder sobre as criações de IA do Julio (MoS 72B, SCache, AURA), sua carreira de 8+ anos em sistemas distribuídos de alta escala ou os detalhes técnicos dos projetos. O que você gostaria de saber?',
        };
      }
      return {
        text: 'Hello! Great to connect with you. I am ready to answer any questions about Julio\'s AI systems breakthroughs (MoS 72B, SCache, AURA), his 8+ years architecting distributed systems, or the engineering blueprints behind these projects. What would you like to explore?',
      };
    }

    // Fallback conversacional
    if (lang === 'pt') {
      return {
        text: `Compreendo sua pergunta sobre "${userQuery}". Como agente de sistemas do Julio, posso te direcionar para as principais áreas do portfólio:\n\n1. ⚡ **MoS & SCache**: Rodando modelos de 72B em GPUs de 16GB e eliminando o KV Cache a 100k tokens.\n2. 🧩 **AURA (Sistema 1)**: Motor de Deep RL resolvendo 232B de estados em 0.23ms em CPU.\n3. 🤖 **CIR-Engine 503M**: Causal Transformer treinado do zero em PyTorch.\n4. 👤 **Trajetória & Contato**: 8+ anos de arquitetura de software, liderança técnica e coordenadas diretas.\n\nQual desses tópicos deseja explorar em detalhes?`,
        links: [
          {
            label: 'Executar Live Test no Colab',
            url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
            icon: 'colab',
          },
        ],
      };
    }
    return {
      text: `Understood your query regarding "${userQuery}". As Julio's systems agent, I can guide you through the core pillars of his engineering dossier:\n\n1. ⚡ **MoS & SCache**: Running 72B LLMs on 16GB GPUs and eliminating the KV Cache across 100k tokens.\n2. 🧩 **AURA (System 1 Engine)**: Deep RL reflex kernel solving 232B combinatorial states in 0.23ms on bare CPU.\n3. 🤖 **CIR-Engine 503M**: Causal Transformer built and trained from scratch in PyTorch.\n4. 👤 **Background & Hiring**: 8+ years of high-concurrency systems architecture and direct coordinates.\n\nWhich of these would you like to dive into?`,
      links: [
        {
          label: 'Launch Live Test on Colab',
          url: 'https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb',
          icon: 'colab',
        },
      ],
    };
  };

  const handleSendMessage = (textToSend?: string) => {
    const query = (textToSend ?? inputMessage).trim();
    if (!query) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setIsTyping(true);

    // Realistic token streaming latency
    setTimeout(() => {
      const response = generateAgentResponse(query);
      const agentMsg: Message = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: response.text,
        links: response.links,
      };
      setMessages((prev) => [...prev, agentMsg]);
      setIsTyping(false);
      if (!isOpen) setHasUnread(true);
    }, 450);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* ========================================================= */}
      {/* FLOATING ACTION TRIGGER (BOTTOM-RIGHT CORNER)             */}
      {/* ========================================================= */}
      <div className="fixed bottom-6 right-6 z-40 flex items-center gap-2">
        {!isOpen && (
          <button
            type="button"
            onClick={() => setIsOpen(true)}
            className="group flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-zinc-950/95 hover:bg-zinc-900 border-2 border-[#adff2f]/70 text-white font-mono text-xs font-bold shadow-[0_0_25px_rgba(173,255,47,0.25)] hover:shadow-[0_0_35px_rgba(173,255,47,0.4)] transition-all duration-200 cursor-pointer hover:scale-105"
            title={lang === 'pt' ? 'Conversar com o Agente SCache' : 'Chat with SCache Agent'}
          >
            <div className="relative">
              <Bot className="w-4 h-4 text-[#adff2f]" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f] animate-ping" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#adff2f]" />
            </div>
            <span>{lang === 'pt' ? 'Agente MoS & SCache' : 'MoS & SCache Agent'}</span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#adff2f]/20 text-[#adff2f] border border-[#adff2f]/30">
              O(1)
            </span>
          </button>
        )}
      </div>

      {/* ========================================================= */}
      {/* EXPANDABLE CHAT PANEL                                     */}
      {/* ========================================================= */}
      {isOpen && (
        <div
          className={`fixed bottom-6 right-6 z-50 w-[92vw] sm:w-[420px] bg-zinc-950/95 border-2 border-[#adff2f]/70 rounded-2xl shadow-[0_0_40px_rgba(0,0,0,0.8),0_0_30px_rgba(173,255,47,0.2)] backdrop-blur-xl flex flex-col transition-all duration-200 overflow-hidden ${
            isMinimized ? 'h-[60px]' : 'h-[580px] max-h-[85vh]'
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/90 border-b border-zinc-800 shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="relative p-1.5 rounded-lg bg-[#adff2f]/10 border border-[#adff2f]/30">
                <Bot className="w-4 h-4 text-[#adff2f]" />
                <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-mono font-bold text-white tracking-tight">
                    {lang === 'pt' ? 'Agente MoS & SCache' : 'MoS & SCache Agent'}
                  </h4>
                  <span className="px-1 py-0.2 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    online
                  </span>
                </div>
                <p className="text-[10px] font-mono text-zinc-400">
                  {lang === 'pt' ? '34 KB L1 Cache Memory · Zero KV Leak' : '34 KB L1 Cache Memory · Zero KV Leak'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setMessages([initialGreeting])}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Limpar conversa' : 'Reset chat'}
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsMinimized(!isMinimized)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={isMinimized ? (lang === 'pt' ? 'Expandir' : 'Expand') : (lang === 'pt' ? 'Minimizar' : 'Minimize')}
              >
                {isMinimized ? <Maximize2 className="w-3.5 h-3.5" /> : <Minimize2 className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Fechar' : 'Close'}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Chat Body (Hidden when minimized) */}
          {!isMinimized && (
            <>
              {/* Message List */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3 font-sans text-xs">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                  >
                    <div
                      className={`max-w-[88%] p-3 rounded-xl leading-relaxed ${
                        msg.sender === 'user'
                          ? 'bg-[#adff2f] text-black font-medium shadow-md shadow-lime-400/10'
                          : 'bg-zinc-900/90 text-zinc-200 border border-zinc-800'
                      }`}
                    >
                      <p className="whitespace-pre-line">{msg.text}</p>

                      {/* Attached Links if any */}
                      {msg.links && msg.links.length > 0 && (
                        <div className="mt-2.5 pt-2 border-t border-zinc-800/80 space-y-1.5">
                          {msg.links.map((link, idx) => (
                            <a
                              key={idx}
                              href={link.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-1.5 text-[11px] font-mono font-bold text-[#adff2f] hover:underline"
                            >
                              {link.icon === 'colab' && <PlayCircle className="w-3.5 h-3.5" />}
                              {link.icon === 'github' && <Github className="w-3.5 h-3.5" />}
                              {link.icon === 'ext' && <ExternalLink className="w-3.5 h-3.5" />}
                              <span>{link.label}</span>
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                    <span className="text-[9px] font-mono text-zinc-500 mt-1 px-1">
                      {msg.timestamp}
                    </span>
                  </div>
                ))}

                {/* Typing indicator */}
                {isTyping && (
                  <div className="flex items-center gap-1.5 p-3 rounded-xl bg-zinc-900 border border-zinc-800 w-fit">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce" />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce [animation-delay:0.2s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#adff2f] animate-bounce [animation-delay:0.4s]" />
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Prompt Chips */}
              <div className="px-3 py-2 bg-zinc-900/60 border-t border-zinc-850 overflow-x-auto flex items-center gap-1.5 no-scrollbar shrink-0">
                {quickPrompts.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendMessage(p.query)}
                    className="shrink-0 text-[10px] font-mono px-2.5 py-1 rounded-full bg-zinc-850 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-750 transition-colors cursor-pointer"
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Input Field */}
              <div className="p-3 bg-zinc-950 border-t border-zinc-800 flex items-center gap-2 shrink-0">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    lang === 'pt' ? 'Pergunte sobre o Julio, MoS 72B, AURA...' : 'Ask about Julio, MoS 72B, AURA...'
                  }
                  className="flex-1 bg-zinc-900 border border-zinc-800 focus:border-[#adff2f] rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none transition-colors font-sans"
                />
                <button
                  type="button"
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim()}
                  className="p-2 bg-[#adff2f] hover:bg-lime-300 disabled:bg-zinc-800 disabled:text-zinc-600 text-black rounded-lg transition-colors cursor-pointer shrink-0"
                  title={lang === 'pt' ? 'Enviar' : 'Send'}
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};
