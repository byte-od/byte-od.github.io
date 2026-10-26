import React, { useState, useRef, useEffect } from 'react';
import {
  Terminal,
  Globe,
  Bot,
  Menu,
  ChevronRight,
  Cpu,
  Layers,
  GitBranch,
  Briefcase,
  Send,
  Compass,
  Sparkles,
  FlaskConical,
  Boxes,
  Zap,
  HelpCircle,
} from 'lucide-react';
import { TokenAssemblyStage } from './TokenInteractiveText';

interface NavbarProps {
  assemblyStage: TokenAssemblyStage;
  setAssemblyStage: (stage: TokenAssemblyStage) => void;
  lang: 'en' | 'pt';
  setLang: (lang: 'en' | 'pt') => void;
  onOpenTerminal: () => void;
  onOpenAgentView: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  assemblyStage,
  setAssemblyStage,
  lang,
  setLang,
  onOpenTerminal,
  onOpenAgentView,
}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const closeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setIsMenuOpen(true);
  };

  const handleMouseLeave = () => {
    closeTimeoutRef.current = setTimeout(() => {
      setIsMenuOpen(false);
    }, 200);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      if (closeTimeoutRef.current) {
        clearTimeout(closeTimeoutRef.current);
      }
    };
  }, []);

  // Featured Highlighted Interactive Modules (Core Competencies & Live Lab)
  const featuredHighlights = [
    {
      href: '#live-lab',
      label: 'Live Lab',
      badge: lang === 'pt' ? 'INTERATIVO' : 'INTERACTIVE',
      desc:
        lang === 'pt'
          ? 'HUD AURA (Sistema 1), Register-red, Flutter Scene fork 3D, GitTrack, MAD-cli & mddd-cli'
          : 'AURA (System 1 HUD), Register-red, Flutter Scene fork 3D, GitTrack, MAD-cli & mddd-cli',
      icon: FlaskConical,
      accent: '#adff2f',
      glowBorder: 'border-[#adff2f]/50 hover:border-[#adff2f]',
      glowBg: 'from-[#adff2f]/15 via-zinc-900 to-zinc-950',
      badgeBg: 'bg-[#adff2f]/15 text-[#adff2f] border-[#adff2f]/40',
    },
    {
      href: '#core-competencies',
      label: 'Core Competencies',
      badge: lang === 'pt' ? 'TENSOR 3D' : '3D TENSOR',
      desc:
        lang === 'pt'
          ? '32 competências de IA, arquitetura & grafo semântico 3D'
          : '32 engineering competencies & semantic 3D tensor manifold',
      icon: Boxes,
      accent: '#38bdf8',
      glowBorder: 'border-[#38bdf8]/50 hover:border-[#38bdf8]',
      glowBg: 'from-[#38bdf8]/15 via-zinc-900 to-zinc-950',
      badgeBg: 'bg-[#38bdf8]/15 text-[#38bdf8] border-[#38bdf8]/40',
    },
  ];

  const navItems = [
    {
      href: '#overview',
      label: lang === 'pt' ? 'Visão Geral' : 'Overview',
      desc: lang === 'pt' ? 'Filosofia de Arquitetura & Apresentação' : 'Architecture Philosophy & Intro',
      icon: Compass,
      tag: '01',
    },
    {
      href: '#cir-engine',
      label: 'CIR-Engine 503M',
      desc: lang === 'pt' ? 'Causal LLM, RoPE & FlashAttention' : 'Causal LLM, RoPE & FlashAttention',
      icon: Cpu,
      tag: '02',
    },
    {
      href: '#tokenizer-lab',
      label: lang === 'pt' ? 'Laboratório de Tokens' : 'Tokenizer Lab',
      desc: lang === 'pt' ? 'Byte-BPE & Matriz de Atenção em Tempo Real' : 'Real-time Byte-BPE & Attention Matrix',
      icon: Layers,
      tag: '03',
    },
    {
      href: '#repositories',
      label: lang === 'pt' ? 'Repositórios Open Source' : 'Open Source Repositories',
      desc: lang === 'pt' ? 'mddd-cli, Register-red, MAD-cli, flutter_scene, gittrack' : 'mddd-cli, Register-red, MAD-cli, flutter_scene, gittrack',
      icon: GitBranch,
      tag: '04',
    },
    {
      href: '#experience',
      label: lang === 'pt' ? 'Trajetória & Formação' : 'Experience & Education',
      desc: lang === 'pt' ? 'Carreira Sênior, Pós-Graduação & Especialidades' : 'Senior Career, Postgrad & Specialties',
      icon: Briefcase,
      tag: '05',
    },
    {
      href: '#faq',
      label: lang === 'pt' ? 'Perguntas Frequentes (FAQ)' : 'Technical FAQ (AEO)',
      desc: lang === 'pt' ? 'Dossiê Técnico & Otimização AEO/GEO' : 'AEO Technical Dossier & Q&A',
      icon: HelpCircle,
      tag: '06',
    },
    {
      href: '#contact',
      label: lang === 'pt' ? 'Console de Transmissão' : 'Dispatch Console',
      desc: lang === 'pt' ? 'Contato Direto, Parcerias e Propostas' : 'Direct Inquiry & Proposals',
      icon: Send,
      tag: '07',
    },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-[#07080a]/85 backdrop-blur-xl border-b border-zinc-800/80 px-6 lg:px-12 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Zone 1: Wordmark & Hamburger Menu Trigger on Hover */}
        <div className="flex items-center gap-3 sm:gap-4">
          <a
            href="#"
            className="font-mono font-bold tracking-tight text-white hover:text-[#adff2f] transition-colors text-base flex items-center gap-1.5"
          >
            <span>JULIO_CESAR</span>
            <span className="text-[#adff2f] animate-pulse">_</span>
          </a>

          {/* Quick Navigation Hamburger Menu Trigger on Hover */}
          <div
            className="relative"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <button
              type="button"
              onClick={() => setIsMenuOpen((prev) => !prev)}
              aria-expanded={isMenuOpen}
              aria-label={lang === 'pt' ? 'Menu de navegação rápida' : 'Quick navigation menu'}
              className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border font-mono text-xs transition-all cursor-pointer ${
                isMenuOpen
                  ? 'bg-zinc-800 border-[#adff2f]/60 text-white shadow-sm shadow-lime-400/10'
                  : 'bg-zinc-900/90 border-zinc-800 text-zinc-300 hover:text-white hover:border-zinc-700'
              }`}
            >
              <Menu
                className={`w-4 h-4 transition-transform duration-200 ${
                  isMenuOpen ? 'text-[#adff2f] rotate-90' : 'text-zinc-400'
                }`}
              />
              <span className="font-semibold text-[11px] tracking-wide">
                {lang === 'pt' ? 'Navegação' : 'Menu'}
              </span>
            </button>

            {/* Hover Flyout Dropdown */}
            {isMenuOpen && (
              <div
                className="absolute top-full left-0 mt-2 w-84 sm:w-[440px] max-h-[85vh] overflow-y-auto bg-zinc-950/95 border border-zinc-800 rounded-2xl shadow-2xl backdrop-blur-xl p-3 sm:p-4 z-50 animate-in fade-in zoom-in-95 duration-150"
                role="menu"
              >
                {/* Header */}
                <div className="flex items-center justify-between px-1 pb-2.5 mb-2.5 border-b border-zinc-800/80">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#adff2f]" />
                    <span>
                      {lang === 'pt'
                        ? 'Navegação Rápida do Portfólio'
                        : 'Quick Portfolio Navigation'}
                    </span>
                  </span>
                  <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest">
                    LATENT_SYSTEM v2.5
                  </span>
                </div>

                {/* FEATURED HIGHLIGHTS: Live Lab & Core Competencies */}
                <div className="mb-3 space-y-2">
                  <div className="flex items-center justify-between px-1">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-300 font-bold flex items-center gap-1.5">
                      <Zap className="w-3 h-3 text-[#adff2f] animate-pulse" />
                      <span>{lang === 'pt' ? 'Módulos em Destaque' : 'Featured Modules'}</span>
                    </span>
                    <span className="text-[9px] font-mono text-[#adff2f] bg-[#adff2f]/10 border border-[#adff2f]/30 px-1.5 py-0.2 rounded-full">
                      TOP_PICKS
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {featuredHighlights.map((item) => {
                      const Icon = item.icon;
                      return (
                        <a
                          key={item.href}
                          href={item.href}
                          onClick={() => setIsMenuOpen(false)}
                          className={`group relative p-3 rounded-xl bg-gradient-to-br ${item.glowBg} border ${item.glowBorder} transition-all cursor-pointer flex flex-col justify-between shadow-lg shadow-black/40`}
                          role="menuitem"
                        >
                          <div>
                            <div className="flex items-center justify-between gap-1 mb-1.5">
                              <div className="flex items-center gap-1.5">
                                <div
                                  className="p-1 rounded-md bg-zinc-900/90 border border-zinc-700/80"
                                  style={{ color: item.accent }}
                                >
                                  <Icon className="w-3.5 h-3.5" />
                                </div>
                                <span className="text-xs font-mono font-bold text-white group-hover:underline">
                                  {item.label}
                                </span>
                              </div>
                              <span
                                className={`text-[8px] font-mono font-bold px-1.5 py-0.5 rounded-full border ${item.badgeBg}`}
                              >
                                {item.badge}
                              </span>
                            </div>
                            <p className="text-[11px] text-zinc-300 leading-snug line-clamp-2 font-sans">
                              {item.desc}
                            </p>
                          </div>

                          <div
                            className="flex items-center gap-1 text-[10px] font-mono mt-2.5 self-end transition-colors font-medium"
                            style={{ color: item.accent }}
                          >
                            <span>{lang === 'pt' ? 'Acessar agora' : 'Launch now'}</span>
                            <ChevronRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                          </div>
                        </a>
                      );
                    })}
                  </div>
                </div>

                {/* ALL SYSTEM CHAPTERS */}
                <div className="pt-2 border-t border-zinc-800/80">
                  <div className="px-1 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                    {lang === 'pt' ? 'Todos os Capítulos' : 'All Chapters'}
                  </div>
                  <div className="space-y-1">
                    {navItems.map((item) => {
                      const Icon = item.icon;
                      return (
                        <a
                          key={item.href}
                          href={item.href}
                          onClick={() => setIsMenuOpen(false)}
                          className="group flex items-start gap-2.5 p-2 rounded-xl hover:bg-zinc-900/90 border border-transparent hover:border-zinc-800 transition-all cursor-pointer text-left"
                          role="menuitem"
                        >
                          <div className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 group-hover:text-[#adff2f] group-hover:border-[#adff2f]/40 transition-colors mt-0.5">
                            <Icon className="w-3.5 h-3.5" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-mono font-bold text-zinc-200 group-hover:text-white transition-colors">
                                {item.label}
                              </span>
                              <span className="text-[9px] font-mono text-zinc-600 group-hover:text-[#adff2f] transition-colors">
                                {item.tag}
                              </span>
                            </div>
                            <p className="text-[11px] text-zinc-400 line-clamp-1 mt-0.5 font-sans">
                              {item.desc}
                            </p>
                          </div>
                          <ChevronRight className="w-3.5 h-3.5 text-zinc-600 group-hover:text-[#adff2f] group-hover:translate-x-0.5 transition-all self-center shrink-0 opacity-0 group-hover:opacity-100" />
                        </a>
                      );
                    })}
                  </div>
                </div>

                {/* Footer bar */}
                <div className="mt-2.5 pt-2 border-t border-zinc-800/80 flex items-center justify-between px-1 text-[10px] font-mono text-zinc-500">
                  <span>
                    {lang === 'pt' ? 'Pressione Esc para fechar' : 'Press Esc to close'}
                  </span>
                  <span className="text-[#adff2f]">julio_cesar@kernel</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Zone 3: Primary interactive controls */}
        <div className="flex items-center gap-2.5">
          {/* 3-Stage Token Assembly Segmented Control */}
          <div className="hidden sm:flex items-center p-0.5 bg-zinc-900 border border-zinc-800 rounded-lg text-[11px] font-mono">
            <button
              type="button"
              onClick={() => setAssemblyStage('auto')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                assemblyStage === 'auto' ? 'bg-zinc-800 text-[#adff2f] font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Montagem Automática por Rolagem' : 'Auto Assembly via Scroll'}
            >
              {lang === 'pt' ? 'Auto Scroll' : 'Scroll Auto'}
            </button>
            <button
              type="button"
              onClick={() => setAssemblyStage('bytes')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                assemblyStage === 'bytes' ? 'bg-zinc-800 text-[#adff2f] font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Estágio 1: Bytes Brutos' : 'Stage 1: Raw Bytes'}
            >
              1. Bytes
            </button>
            <button
              type="button"
              onClick={() => setAssemblyStage('semiwords')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                assemblyStage === 'semiwords' ? 'bg-zinc-800 text-[#38bdf8] font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Estágio 2: Sub-palavras / Tokens BPE' : 'Stage 2: BPE Semi-words / Subwords'}
            >
              {lang === 'pt' ? '2. Sub-palavras' : '2. Semi-words'}
            </button>
            <button
              type="button"
              onClick={() => setAssemblyStage('words')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer ${
                assemblyStage === 'words' ? 'bg-zinc-800 text-white font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
              title={lang === 'pt' ? 'Estágio 3: Palavras Completas' : 'Stage 3: Full Assembled Words'}
            >
              {lang === 'pt' ? '3. Palavras' : '3. Words'}
            </button>
          </div>

          {/* Language Toggle */}
          <button
            type="button"
            onClick={() => setLang(lang === 'en' ? 'pt' : 'en')}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-300 transition-colors cursor-pointer"
            title={lang === 'pt' ? 'Alternar Idioma (EN / PT)' : 'Switch Language (PT / EN)'}
          >
            <Globe className="w-3.5 h-3.5 text-[#adff2f]" />
            <span className="font-bold text-[#adff2f]">{lang.toUpperCase()}</span>
          </button>

          {/* Accessible AI Agent Version Button (llms.txt & MCP) */}
          <button
            type="button"
            onClick={onOpenAgentView}
            aria-haspopup="dialog"
            aria-label={lang === 'pt' ? 'Abrir versão estruturada para agentes de IA e Servidor MCP (llms.txt & MCP)' : 'Open machine-readable AI agent version & MCP server (llms.txt & MCP)'}
            className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-700/80 hover:border-[#adff2f] text-xs font-mono text-zinc-200 hover:text-white transition-all cursor-pointer focus:ring-2 focus:ring-[#adff2f] focus:outline-hidden group"
            title={lang === 'pt' ? 'Versão para Agentes de IA & Servidor MCP (llms.txt & MCP)' : 'AI Agent & MCP Server View (llms.txt & MCP)'}
          >
            <Bot className="w-3.5 h-3.5 text-[#adff2f] group-hover:scale-110 transition-transform" />
            <span className="font-semibold">{lang === 'pt' ? 'Agentes & MCP' : 'Agents & MCP'}</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-zinc-800 text-[#adff2f] hidden lg:inline">
              MCP
            </span>
          </button>

          {/* Contact / Terminal Action */}
          <button
            type="button"
            onClick={onOpenTerminal}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#adff2f] text-black font-mono text-xs font-bold hover:bg-lime-300 transition-all cursor-pointer whitespace-nowrap shadow-sm shadow-lime-400/20"
          >
            <Terminal className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Terminal</span>
          </button>
        </div>
      </div>
    </header>
  );
};
