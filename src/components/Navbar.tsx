import React, { useState, useRef, useEffect } from 'react';
import {
  Terminal,
  Globe,
  Bot,
  Menu,
  ChevronRight,
  Cpu,
  Radio,
  GitBranch,
  Send,
  Compass,
  Sparkles,
  FlaskConical,
  Brain,
  Zap,
  HelpCircle,
} from 'lucide-react';

interface NavbarProps {
  lang: 'en' | 'pt';
  setLang: (lang: 'en' | 'pt') => void;
  onOpenTerminal: () => void;
  onOpenAgentView: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  lang,
  setLang,
  onOpenTerminal,
  onOpenAgentView,
}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const closeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  const featuredHighlights = [
    {
      href: '#scar-native-lab',
      label: 'Scar-Native Lab v5.0',
      badge: 'TABULA RASA',
      desc:
        lang === 'pt'
          ? 'ScarNativeLM v5.0 do zero (VQ4-STE 0.0000 KL), SCache O(1) Δ +0 B, P_⊥lang & 0x00 GDPR'
          : 'Tabula-Rasa ScarNativeLM v5.0 (VQ4-STE 0.0000 KL), O(1) SCache Δ +0 B, P_⊥lang & 0x00 GDPR',
      icon: FlaskConical,
      accent: '#adff2f',
      glowBorder: 'border-[#adff2f]/50 hover:border-[#adff2f]',
      glowBg: 'from-[#adff2f]/15 via-zinc-900 to-zinc-950',
      badgeBg: 'bg-[#adff2f]/15 text-[#adff2f] border-[#adff2f]/40',
    },
    {
      href: '#aura-origin',
      label: 'AURA · System One',
      badge: '34.8 KB SCACHE',
      desc:
        lang === 'pt'
          ? 'Berço do GradientScarMemory O(1) (34.8 KB), Batedores Ghost-Jump & S1 0.23 ms ao vivo'
          : 'Birthplace of 34.8 KB O(1) GradientScarMemory, Ghost-Jump Scouts & live 0.23 ms S1 HUD',
      icon: Brain,
      accent: '#38bdf8',
      glowBorder: 'border-[#38bdf8]/50 hover:border-[#38bdf8]',
      glowBg: 'from-[#38bdf8]/15 via-zinc-900 to-zinc-950',
      badgeBg: 'bg-[#38bdf8]/15 text-[#38bdf8] border-[#38bdf8]/40',
    },
  ];

  const navItems = [
    {
      href: '#overview',
      label: lang === 'pt' ? 'Visão Geral & JARVIS 3D' : 'Overview & JARVIS 3D',
      desc:
        lang === 'pt'
          ? 'MoS 72.7B, SCache O(1), PointScarFlow 3D & Qwen 3B WebGPU'
          : 'MoS 72.7B, SCache O(1), PointScarFlow 3D & Qwen 3B WebGPU',
      icon: Compass,
      tag: '01',
    },
    {
      href: '#scar-native-lab',
      label: 'Scar-Native Lab v5.0',
      desc:
        lang === 'pt'
          ? 'D4-LUT256 VQ4-STE, DifferentiableSCacheAttention O(1) & RRR P_⊥lang'
          : 'D4-LUT256 VQ4-STE, O(1) DifferentiableSCacheAttention & RRR P_⊥lang',
      icon: Cpu,
      tag: '02',
    },
    {
      href: '#physical-scar-lab',
      label: 'Physical Scar Lab v4.0',
      desc:
        lang === 'pt'
          ? 'ScarCodec: Rádio RFI P_⊥, Caixa-Preta 34.8 KB & CFD HDR 12.05x'
          : 'ScarCodec: Radio RFI P_⊥, 34.8 KB Black-Box & 12.05x HDR CFD',
      icon: Radio,
      tag: '03',
    },
    {
      href: '#aura-origin',
      label: 'AURA / System One',
      desc:
        lang === 'pt'
          ? 'Origem do GradientScarMemory (34.8 KB) & Deep RL 0.23 ms'
          : 'Origin of GradientScarMemory (34.8 KB) & 0.23 ms Deep RL',
      icon: Brain,
      tag: '04',
    },
    {
      href: '#scar-ecosystem',
      label: lang === 'pt' ? 'Ecossistema & Papers Zenodo' : 'Scar Ecosystem & Zenodo Papers',
      desc:
        lang === 'pt'
          ? 'ScarNativeLM, MoS, PointScarFlow v1.0, ScarCodec & System One'
          : 'ScarNativeLM, MoS, PointScarFlow v1.0, ScarCodec & System One',
      icon: GitBranch,
      tag: '05',
    },
    {
      href: '#faq',
      label: lang === 'pt' ? 'Dossiê Técnico Scar (FAQ)' : 'Scar Technical Dossier (FAQ)',
      desc:
        lang === 'pt'
          ? 'Garantias Matemáticas O(1), VQ4-Scar, P_⊥lang & AEO'
          : 'O(1) Mathematical Guarantees, VQ4-Scar, P_⊥lang & AEO',
      icon: HelpCircle,
      tag: '06',
    },
    {
      href: '#contact',
      label: lang === 'pt' ? 'Autoria & Contato Direto' : 'Author & Direct Contact',
      desc:
        lang === 'pt'
          ? 'Julio Cesar Reis Filho · Pesquisa, Parcerias & Engenharia'
          : 'Julio Cesar Reis Filho · Research, Collaborations & Engineering',
      icon: Send,
      tag: '07',
    },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-[#07080a]/85 backdrop-blur-xl border-b border-zinc-800/80 px-6 lg:px-12 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Zone 1: Wordmark & Navigation Flyout */}
        <div className="flex items-center gap-3 sm:gap-4">
          <a
            href="#overview"
            className="font-mono font-bold tracking-tight text-white hover:text-[#adff2f] transition-colors text-base flex items-center gap-1.5"
          >
            <span>SCAR_ARCHITECTURE</span>
            <span className="text-[#adff2f] animate-pulse">_</span>
          </a>

          <div
            className="relative"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <button
              type="button"
              onClick={() => setIsMenuOpen((prev) => !prev)}
              aria-expanded={isMenuOpen}
              aria-label={lang === 'pt' ? 'Menu do Ecossistema de Cicatrizes' : 'Scar Ecosystem navigation menu'}
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
                {lang === 'pt' ? 'Capítulos' : 'Chapters'}
              </span>
            </button>

            {isMenuOpen && (
              <div
                className="absolute top-full left-0 mt-2 w-84 sm:w-[440px] max-h-[85vh] overflow-y-auto bg-zinc-950/95 border border-zinc-800 rounded-2xl shadow-2xl backdrop-blur-xl p-3 sm:p-4 z-50 animate-in fade-in zoom-in-95 duration-150"
                role="menu"
              >
                <div className="flex items-center justify-between px-1 pb-2.5 mb-2.5 border-b border-zinc-800/80">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#adff2f]" />
                    <span>
                      {lang === 'pt'
                        ? 'Ecossistema Mixture of Scars & SCache'
                        : 'Mixture of Scars & SCache Ecosystem'}
                    </span>
                  </span>
                  <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest">
                    MOS + SCACHE v5.0
                  </span>
                </div>

                <div className="mb-3 space-y-2">
                  <div className="flex items-center justify-between px-1">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-300 font-bold flex items-center gap-1.5">
                      <Zap className="w-3 h-3 text-[#adff2f] animate-pulse" />
                      <span>{lang === 'pt' ? 'Motores Ao Vivo' : 'Live Engines'}</span>
                    </span>
                    <span className="text-[9px] font-mono text-[#adff2f] bg-[#adff2f]/10 border border-[#adff2f]/30 px-1.5 py-0.2 rounded-full">
                      ZERO_FACADE
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
                            <span>{lang === 'pt' ? 'Abrir lab' : 'Open lab'}</span>
                            <ChevronRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                          </div>
                        </a>
                      );
                    })}
                  </div>
                </div>

                <div className="pt-2 border-t border-zinc-800/80">
                  <div className="px-1 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                    {lang === 'pt' ? 'Capítulos da Arquitetura' : 'Architecture Chapters'}
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

                <div className="mt-2.5 pt-2 border-t border-zinc-800/80 flex items-center justify-between px-1 text-[10px] font-mono text-zinc-500">
                  <span>{lang === 'pt' ? 'Pressione Esc para fechar' : 'Press Esc to close'}</span>
                  <span className="text-[#adff2f]">julio_reis@scar-kernel</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Zone 2: Quick Lab Links on Desktop + Primary Controls */}
        <div className="flex items-center gap-2.5">
          <div className="hidden xl:flex items-center gap-1 p-1 bg-zinc-900/90 border border-zinc-800 rounded-lg text-[11px] font-mono">
            <a
              href="#scar-native-lab"
              className="px-2.5 py-1 rounded text-zinc-300 hover:text-[#adff2f] hover:bg-zinc-800 transition-colors"
            >
              ScarNative v5.0
            </a>
            <a
              href="#physical-scar-lab"
              className="px-2.5 py-1 rounded text-zinc-300 hover:text-[#38bdf8] hover:bg-zinc-800 transition-colors"
            >
              ScarCodec v4.0
            </a>
            <a
              href="#aura-origin"
              className="px-2.5 py-1 rounded text-zinc-300 hover:text-purple-300 hover:bg-zinc-800 transition-colors"
            >
              AURA 34.8 KB
            </a>
          </div>

          <button
            type="button"
            onClick={() => window.dispatchEvent(new CustomEvent('open-jarvis-agent'))}
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-sky-950/60 border border-sky-400/40 hover:border-sky-300 text-xs font-mono text-sky-200 hover:text-white transition-colors cursor-pointer"
            title={
              lang === 'pt'
                ? 'Focar JARVIS Qwen 3B + PointScarFlow 3D (34 Shaders WGSL)'
                : 'Focus JARVIS Qwen 3B + PointScarFlow 3D (34 WGSL Shaders)'
            }
          >
            <Sparkles className="w-3.5 h-3.5 text-[#38bdf8]" />
            <span className="font-bold">JARVIS 3D</span>
          </button>

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

          {/* AI Agent Version Button (llms.txt & MCP) */}
          <button
            type="button"
            onClick={onOpenAgentView}
            aria-haspopup="dialog"
            aria-label={
              lang === 'pt'
                ? 'Abrir dossiê estruturado Scar para agentes de IA (llms.txt & MCP)'
                : 'Open structured Scar dossier for AI agents (llms.txt & MCP)'
            }
            className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-700/80 hover:border-[#adff2f] text-xs font-mono text-zinc-200 hover:text-white transition-all cursor-pointer focus:ring-2 focus:ring-[#adff2f] focus:outline-hidden group"
            title={lang === 'pt' ? 'Dossiê para Agentes de IA (llms.txt & MCP)' : 'AI Agent Dossier (llms.txt & MCP)'}
          >
            <Bot className="w-3.5 h-3.5 text-[#adff2f] group-hover:scale-110 transition-transform" />
            <span className="font-semibold">{lang === 'pt' ? 'llms.txt' : 'llms.txt'}</span>
          </button>

          {/* Scar Kernel Terminal Action */}
          <button
            type="button"
            onClick={onOpenTerminal}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#adff2f] text-black font-mono text-xs font-bold hover:bg-lime-300 transition-all cursor-pointer whitespace-nowrap shadow-sm shadow-lime-400/20"
          >
            <Terminal className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Scar CLI</span>
          </button>
        </div>
      </div>
    </header>
  );
};
