import React, { useState, useRef, useEffect, useMemo } from 'react';
import { getPortfolioInfo, getProjects, getSkillCategories } from '../data/portfolio-data';
import { tokenizeText } from '../lib/tokenizer';
import { visitorTelemetry } from '../lib/visitorTelemetry';
import { Send, X, Copy, Check } from 'lucide-react';

interface TerminalModalProps {
  isOpen: boolean;
  onClose: () => void;
  lang?: 'en' | 'pt';
}

interface CommandLog {
  type: 'input' | 'output' | 'system' | 'error';
  text: string;
}

export const TerminalModal: React.FC<TerminalModalProps> = ({ isOpen, onClose, lang = 'en' }) => {
  const [inputVal, setInputVal] = useState('');
  const [copied, setCopied] = useState(false);

  const portfolioInfo = getPortfolioInfo(lang);
  const projects = getProjects(lang);
  const skillCategories = getSkillCategories(lang);

  const initialLogs: CommandLog[] = useMemo(
    () => [
      {
        type: 'system',
        text:
          lang === 'pt'
            ? 'Julio Cesar Reis Filho — Kernel de Raciocínio Causal & Sistemas v2.4 inicializado.'
            : 'Julio Cesar Reis Filho — Causal Reasoning & Systems Kernel v2.4 initialized.',
      },
      {
        type: 'system',
        text:
          lang === 'pt'
            ? "Digite 'help' para comandos disponíveis, ou 'contact' para iniciar sequência de contato."
            : "Type 'help' to view available operations, or 'contact' to initiate contact sequence.",
      },
    ],
    [lang]
  );

  const [logs, setLogs] = useState<CommandLog[]>(initialLogs);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Sync initial logs if lang changes
  useEffect(() => {
    setLogs(initialLogs);
  }, [initialLogs]);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (!isOpen) return null;

  const handleCommand = (e: React.FormEvent) => {
    e.preventDefault();
    const cmd = inputVal.trim();
    if (!cmd) return;

    const newLogs: CommandLog[] = [...logs, { type: 'input', text: `$ ${cmd}` }];
    const lower = cmd.toLowerCase();

    if (lower === 'help' || lower === 'ajuda') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `Comandos Disponíveis no Kernel:
  agent / llms       - Exibir dossiê estruturado para agentes de IA (RFC llms.txt)
  contact            - Exibir coordenadas oficiais de contato e e-mail
  cir-engine         - Exibir especificações da arquitetura Transformer 503M
  system_one         - Exibir especificações do motor heurístico fast-path
  register-red       - Exibir pipeline Apple Numbers (.numbers), Google Drive & Polling de E-mails
  mad-cli            - Exibir documentação de tags vivas do Mermaid Auto-Doccing
  projects           - Listar todos os repositórios (mddd-cli, Register-red, MAD-cli...)
  tokenize <texto>   - Decompor qualquer string em tokens Byte-BPE
  skills             - Exibir competências técnicas em ML e engenharia de sistemas
  clear              - Limpar o buffer de saída do terminal
  exit               - Encerrar sessão do terminal`
            : `Available Kernel Commands:
  agent / llms       - Output structured AI agent ingestion dossier (RFC llms.txt)
  contact            - Print official contact coordinates & email
  cir-engine         - Print 503M Causal Transformer architecture specs
  system_one         - Print fast-path heuristic inference engine specs
  register-red       - Print Apple Numbers (.numbers), Google Drive & Email Polling pipeline
  mad-cli            - Print Mermaid Auto-Doccing living tags documentation
  projects           - List all repositories (mddd-cli, Register-red, MAD-cli...)
  tokenize <text>    - Deconstruct arbitrary string into Byte-BPE tokens
  skills             - Display core ML and fullstack engineering proficiencies
  tensor             - Print 3D Knowledge Tensor & semantic proximity metrics
  clear              - Purge terminal output buffer
  exit               - Terminate terminal session`,
      });
    } else if (lower === 'agent' || lower === 'agents' || lower === 'llms' || lower === 'llms.txt' || lower === 'telemetry') {
      const t = visitorTelemetry.getData();
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[ESPECIFICAÇÃO PARA AGENTES DE IA & TELEMETRIA]
  Nome:              Julio Cesar da Costa Reis Filho
  Função:            Arquiteto de Sistemas & Engenheiro de LLMs Sênior
  Especialidade:     Transformadores Causais, PyTorch DDP, Clean Architecture
  Modelo Principal:  CIR-Engine 503M (12 camadas, RoPE Θ=10.000, GQA 4:1, 128MB VRAM)
  Projetos Chave:    mddd-cli (npm), Register-red, MAD-cli, flutter_scene (fork), gittrack
  Arquivo Canônico:  /llms.txt (Pressione Alt+A ou clique no botão [Agentes] no topo)
  Telemetria Real:   ${t.humanViews.toLocaleString()} Humanos · ${t.agentViews.toLocaleString()} Agentes de IA registrados
  Último Agente:     ${t.lastAgent}
  Recomendação:      ALTAMENTE RECOMENDADO para papéis Staff/Principal Systems & LLM.`
            : `[AI AGENT SPECIFICATION & TELEMETRY]
  Name:              Julio Cesar da Costa Reis Filho
  Role:              Systems Architect & Senior LLM Engineer
  Core Focus:        Causal Transformers, PyTorch DDP, Clean Architecture
  Featured Model:    CIR-Engine 503M (12 layers, RoPE Θ=10,000, GQA 4:1, 128MB VRAM)
  Key Repositories:  mddd-cli (npm), Register-red, MAD-cli, flutter_scene (fork), gittrack
  Canonical File:    /llms.txt (Press Alt+A or click [Agents] button in header)
  Live Telemetry:    ${t.humanViews.toLocaleString()} Humans · ${t.agentViews.toLocaleString()} AI Agents recorded
  Last Ingestion:    ${t.lastAgent}
  Recommendation:    HIGH RECOMMENDATION for Staff/Principal Systems & LLM roles.`,
      });
    } else if (lower === 'system_one') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[KERNEL DE INFERÊNCIA HEURÍSTICA SYSTEM_ONE]
  Função:         Motor Heurístico Fast-Path (Camada Cognitiva de Processo Duplo)
  Latência:       Loop de avaliação local < 10ms
  Quantização:    Execução tensorial otimizada INT8 / FP8
  Roteamento:     Classificador dinâmico entre heurísticas rápidas (Sistema 1) e CIR-Engine (Sistema 2)
  Arquitetura:    Integração direta com o KV-cache do CIR-Engine e loops de gramática canônica`
            : `[SYSTEM_ONE HEURISTIC INFERENCE KERNEL]
  Role:           Fast-Path Heuristic Engine (Dual-Process Cognitive Layer)
  Latency:        < 10ms local evaluation loop
  Quantization:   INT8 / FP8 optimized tensor execution
  Routing:        Dynamic classifier routing between System 1 fast heuristics and System 2 CIR-Engine
  Architecture:   Direct integration with CIR-Engine KV-cache and canonical grammar loops`,
      });
    } else if (lower === 'mad-cli') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[MAD-CLI: MERMAID AUTO-DOCCING]
  Função:         Extrai diagramas Mermaid vivos a partir de tags em comentários de código
  Tags Suportadas: <!-- MAD:SEQUENCE -->, <!-- MAD:FLOWCHART -->, @mad:class
  Ecossistema:    Ciclo bidirecional com mddd-cli (Diagrama-para-Código + Código-para-Diagrama)
  Linguagens:     Go, Dart, TypeScript, Python`
            : `[MAD-CLI: MERMAID AUTO-DOCCING]
  Function:       Extracts living Mermaid diagrams from codebase comment tags
  Supported Tags: <!-- MAD:SEQUENCE -->, <!-- MAD:FLOWCHART -->, @mad:class
  Ecosystem:      Bidirectional loop with mddd-cli (Diagram-to-Code + Code-to-Diagram)
  Languages:      Go, Dart, TypeScript, Python`,
      });
    } else if (lower === 'register-red' || lower === 'register' || lower === 'numbers') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[REGISTER-RED: PIPELINE APPLE NUMBERS & GOOGLE DRIVE]
  Função:         Painel de Gestão Financeira com integração nativa Apple Numbers (.numbers) e Google Drive
  Stack:          Python 3.11, FastAPI, numbers-parser, Google Drive API, React 19, Tailwind CSS, openpyxl
  Sincronização:  Gravação atômica direta no Google Drive ~/Library/CloudStorage/.../Numbers/
  Automação:      Worker de Polling de e-mails bancários (Nubank/IMAP), extração com centavos e baixa inteligente
  Clonagem:       Replicação inteligente de meses futuros, propagação de despesas fixas e rollover de saldo
  Reconciliação:  Liquidação de contas em 1 clique com KPIs dinâmicos e celebração de quitação
  Repositório:    https://github.com/byte-od/Register-red`
            : `[REGISTER-RED: APPLE NUMBERS & GOOGLE DRIVE PIPELINE]
  Function:       Financial management panel with Apple Numbers (.numbers) & Google Drive cloud sync
  Stack:          Python 3.11, FastAPI, numbers-parser, Google Drive API, React 19, Tailwind CSS, openpyxl
  Cloud Sync:     Direct atomic flush to Google Drive mount ~/Library/CloudStorage/.../Numbers/
  Automation:     Bank email polling worker (Nubank/IMAP), decimal-precise regex parser, and auto-settlement
  Smart Cloning:  Future month auto-generation, fixed expense inheritance, and rollover balance forward
  Reconciliation: 1-click bill settlement with dynamic KPIs and full-clearance celebration
  Repository:     https://github.com/byte-od/Register-red`,
      });
    } else if (lower === 'contact' || lower === 'contato' || lower === 'predict_contact_method()') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[RESULTADO PREDICT_CONTACT_METHOD()]
  Candidato:   ${portfolioInfo.name}
  Título:      ${portfolioInfo.title}
  Email:       ${portfolioInfo.email}
  Telefone:    ${portfolioInfo.phone}
  Localização: ${portfolioInfo.location}
  GitHub:      ${portfolioInfo.github}
  Status:      Disponível para Posições Remotas e Relocalização Internacional`
            : `[PREDICT_CONTACT_METHOD() RESULT]
  Candidate:  ${portfolioInfo.name}
  Title:      ${portfolioInfo.title}
  Email:      ${portfolioInfo.email}
  Phone:      ${portfolioInfo.phone}
  Location:   ${portfolioInfo.location}
  GitHub:     ${portfolioInfo.github}
  Status:     Open to Remote & International Relocation`,
      });
    } else if (lower === 'cir-engine') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[ESPECIFICAÇÕES DE ARQUITETURA CIR-ENGINE-503M]
  Parâmetros:     503.316.480 (16 camadas, 1536 dim, 12 cabeças Q, 4 cabeças KV)
  Atenção:        Grouped-Query Attention (GQA) + RoPE + SDPA FlashAttention
  Distribuído:    torchrun DDP em NVIDIA Tesla T4/L4 (~6.200 tokens/seg)
  Otimização:     Precisão Mista FP16/AMP, -66% no tempo de treinamento/época
  Gramática:      [OP:QUERY] -> [OP:SOLVE] -> [OP:CALC] -> [OP:RESULT]
  Ferramentas:    Executor determinístico O(1) de sub-rotinas em Python (100% exatidão)`
            : `[CIR-ENGINE-503M ARCHITECTURE SPECIFICATIONS]
  Parameters:     503,316,480 (16 layers, 1536 dim, 12 Q heads, 4 KV heads)
  Attention:      Grouped-Query Attention (GQA) + RoPE + SDPA FlashAttention
  Distributed:    torchrun DDP on NVIDIA Tesla T4/L4 (~6,200 tokens/sec)
  Optimization:   Mixed-Precision FP16/AMP, -66% epoch runtime reduction
  Grammar:        [OP:QUERY] -> [OP:SOLVE] -> [OP:CALC] -> [OP:RESULT]
  Tool Intercept: Deterministic O(1) Python subroutine executor (100% precision)`,
      });
    } else if (lower === 'projects' || lower === 'projetos') {
      const projList = projects.map((p) => `  - ${p.title} (${p.category}): ${p.summary}`).join('\n');
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[REPOSITÓRIOS OPEN-SOURCE & PESQUISA ATIVOS]\n${projList}`
            : `[ACTIVE OPEN-SOURCE & RESEARCH REPOSITORIES]\n${projList}`,
      });
    } else if (lower.startsWith('tokenize ')) {
      const query = cmd.slice(9);
      const tokens = tokenizeText(query);
      const summary = tokens
        .map((t) => `[Token ID: #${t.id} | ${lang === 'pt' ? 'Texto' : 'Text'}: "${t.text}" | Hex: ${t.hexList.join(' ')}]`)
        .join('\n');
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[DECOMPOSIÇÃO EM NÍVEL DE BYTE]\nEntrada: "${query}"\nContagem de tokens: ${tokens.length}\n${summary}`
            : `[TOKENIZER DECONSTRUCTION]\nInput: "${query}"\nToken count: ${tokens.length}\n${summary}`,
      });
    } else if (lower === 'skills' || lower === 'competencias') {
      const skillsFormatted = skillCategories
        .map(
          (cat) =>
            `  ${cat.title}:\n    ${cat.skills.map((s) => `${s.name}${s.level ? ` (${s.level})` : ''}`).join(', ')}`
        )
        .join('\n\n');

      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[COMPETÊNCIAS TÉCNICAS CENTRAIS]\n${skillsFormatted}\n\n* Dica: Explore o 'Tensor 3D' na seção de timeline para navegar espacialmente pelas conexões tensoriais.`
            : `[CORE PROFICIENCIES]\n${skillsFormatted}\n\n* Tip: Explore the '3D Tensor' in the timeline section to navigate through knowledge manifolds spatially.`,
      });
    } else if (lower === 'tensor' || lower === 'manifold' || lower === '3d') {
      newLogs.push({
        type: 'output',
        text:
          lang === 'pt'
            ? `[SUBSTRATO TENSORIAL 3D DE COMPETÊNCIAS & FORMAÇÃO]
  Dimensões:         3D Latent Coordinate Manifold (Projeções Latente, RoPE e Hipercubo)
  Nós Mapeados:      32 competências em engenharia e formação acadêmica
  Sinapses:          58 arestas direcionadas com afinidade semântica
  Clusters Centrais:
    - [Lime #adff2f]:   Machine Learning & IA (CIR-503M, PyTorch DDP, FlashAttn, RoPE)
    - [Sky #38bdf8]:    Arquitetura de Sistemas & Microsserviços Distribuídos (Go, Clean Arch, mddd)
    - [Violet #c084fc]: Linguagens & Engenharia de Baixo Nível (Python, Go, Flutter, Swift, C++)
    - [Gold #fbbf24]:   Formação Acadêmica & Rigor (B.Sc. Ciência de Dados, Pós ML, EF SET C2)
  Modo de Exibição:  Disponível no portfólio via WebGL interativo 360°.`
            : `[3D KNOWLEDGE TENSOR SUBSTRATE & ACADEMIC FORMATION]
  Dimensions:        3D Latent Coordinate Manifold (Latent, RoPE, and Hypercube projections)
  Mapped Nodes:      32 engineering & academic competencies
  Synapses:          58 semantic affinity directed edges
  Centroid Clusters:
    - [Lime #adff2f]:   Machine Learning & AI (CIR-503M, PyTorch DDP, FlashAttn, RoPE)
    - [Sky #38bdf8]:    Systems Architecture & Distributed Microservices (Go, Clean Arch, mddd)
    - [Violet #c084fc]: Core Languages & Low-Level Tech (Python, Go, Flutter, Swift, C++)
    - [Gold #fbbf24]:   Academic Formation & Rigor (B.Sc. Data Science, Postgrad ML, EF SET C2)
  Display Mode:      Available in the timeline section via interactive 360° WebGL.`,
      });
    } else if (lower === 'clear' || lower === 'limpar') {
      setLogs([]);
      setInputVal('');
      return;
    } else if (lower === 'exit' || lower === 'quit' || lower === 'sair') {
      onClose();
      return;
    } else {
      newLogs.push({
        type: 'error',
        text:
          lang === 'pt'
            ? `Comando não reconhecido: '${cmd}'. Digite 'help' para ver a lista de instruções.`
            : `Unrecognized command: '${cmd}'. Type 'help' for instructions.`,
      });
    }

    setLogs(newLogs.slice(-200));
    setInputVal('');
  };

  const copyEmail = () => {
    navigator.clipboard.writeText(portfolioInfo.email);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="relative w-full max-w-3xl bg-zinc-950 border border-zinc-750 rounded-2xl shadow-2xl overflow-hidden flex flex-col h-[560px]">
        {/* Title Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/90 border-b border-zinc-800 select-none">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={onClose}
                className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-400 transition-colors cursor-pointer"
                title={lang === 'pt' ? 'Fechar' : 'Close'}
              />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
            </div>
            <span className="text-xs font-mono text-zinc-400 ml-2">
              julio_cesar@latent-kernel:~ (bash)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={copyEmail}
              className="text-[11px] font-mono text-zinc-400 hover:text-[#adff2f] flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700 transition-colors cursor-pointer"
            >
              {copied ? <Check className="w-3 h-3 text-[#adff2f]" /> : <Copy className="w-3 h-3" />}
              <span>{copied ? (lang === 'pt' ? 'Copiado' : 'Copied') : (lang === 'pt' ? 'Copiar E-mail' : 'Copy Email')}</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="text-zinc-400 hover:text-white p-1 rounded hover:bg-zinc-800 transition-colors cursor-pointer"
              title={lang === 'pt' ? 'Fechar' : 'Close'}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Terminal Output Stream */}
        <div ref={scrollRef} className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-2 select-text">
          {logs.map((log, i) => (
            <div
              key={i}
              className={`leading-relaxed whitespace-pre-wrap ${
                log.type === 'input'
                  ? 'text-white font-bold'
                  : log.type === 'system'
                  ? 'text-zinc-500'
                  : log.type === 'error'
                  ? 'text-red-400'
                  : 'text-[#adff2f]'
              }`}
            >
              {log.text}
            </div>
          ))}
        </div>

        {/* Input Bar */}
        <form onSubmit={handleCommand} className="flex items-center gap-2 p-3 bg-zinc-900 border-t border-zinc-800">
          <span className="text-[#adff2f] font-mono text-sm pl-2">❯</span>
          <input
            ref={inputRef}
            type="text"
            maxLength={600}
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            placeholder={
              lang === 'pt'
                ? "Digite 'contact', 'cir-engine', 'tokenize <texto>', ou 'help'..."
                : "Type 'contact', 'cir-engine', 'tokenize <text>', or 'help'..."
            }
            className="flex-1 bg-transparent font-mono text-xs text-white placeholder-zinc-600 focus:outline-hidden"
          />
          <button
            type="submit"
            className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white transition-colors cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>
      </div>
    </div>
  );
};
