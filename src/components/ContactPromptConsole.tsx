import React, { useState } from 'react';
import {
  Copy,
  Check,
  Terminal,
  Sparkles,
  MessageSquare,
  ArrowRight,
  Mail,
  Zap,
} from 'lucide-react';

interface ContactPromptConsoleProps {
  lang?: 'en' | 'pt';
  onOpenTerminal?: () => void;
}

export const ContactPromptConsole: React.FC<ContactPromptConsoleProps> = ({
  lang = 'en',
  onOpenTerminal,
}) => {
  const [promptText, setPromptText] = useState('');
  const [senderInfo, setSenderInfo] = useState('');
  const [copied, setCopied] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const presets = [
    {
      label: lang === 'pt' ? '🧬 Colaboração Científica / Pesquisa Scar' : '🧬 Scar Research Collaboration',
      text:
        lang === 'pt'
          ? 'Olá Julio, acompanhei os papers no Zenodo e os motores ScarNativeLM v5.0, MoS, SCache O(1) e PointScarFlow v1.0. Gostaria de conversar sobre colaboração científica / benchmarking conjunto.'
          : 'Hello Julio, I reviewed your Zenodo papers and live engines (ScarNativeLM v5.0, MoS, O(1) SCache, PointScarFlow v1.0) and would like to discuss a research or benchmarking collaboration.',
    },
    {
      label: lang === 'pt' ? '⚡ Implantação MoS & SCache O(1) (WebGPU / GPU)' : '⚡ MoS & SCache O(1) Deployment',
      text:
        lang === 'pt'
          ? 'Olá Julio, temos um desafio de inferência sob orçamento estrito de VRAM / contexto longo e gostaríamos de conversar sobre aplicar Mixture of Scars (MoS) e SCache O(1) em nossa infraestrutura.'
          : 'Hi Julio, we have a strict-VRAM / long-horizon inference challenge and would love to discuss deploying Mixture of Scars (MoS) and O(1) SCache in our stack.',
    },
    {
      label: lang === 'pt' ? '🚀 Liderança em Arquitetura de IA' : '🚀 Principal AI Architecture Role',
      text:
        lang === 'pt'
          ? 'Olá Julio, vimos os motores WebGPU/WGSL e PyTorch do ecossistema de cicatrizes e gostaríamos de conversar sobre uma posição de Principal AI Systems Architect em nossa equipe.'
          : 'Hi Julio, we explored your WebGPU/WGSL and PyTorch Scar Architecture engines and would like to discuss a Principal AI Systems Architect opportunity with our team.',
    },
  ];

  const handleSelectPreset = (text: string) => {
    setPromptText(text);
    setStatusMessage(null);
  };

  const generateMailtoPayload = () => {
    const recipient = 'reisfilho1116@gmail.com';
    const cleanSender = senderInfo.replace(/[\r\n\t\0]/g, ' ').trim().slice(0, 100);
    const cleanPrompt = promptText.trim().slice(0, 2000);

    const subjectStr =
      lang === 'pt'
        ? `[SCAR_ECOSYSTEM] Contato de ${cleanSender || 'Visitante'}`
        : `[SCAR_ECOSYSTEM] Inquiry from ${cleanSender || 'Visitor'}`;

    const bodyContent = `${cleanPrompt}\n\n---\n${
      lang === 'pt' ? 'Remetente' : 'Sender'
    }: ${cleanSender || (lang === 'pt' ? 'Não informado' : 'Not specified')}\n${
      lang === 'pt' ? 'Origem' : 'Origin'
    }: Mixture of Scars & SCache O(1) Portfolio (byte-od.github.io)\nTimestamp: ${new Date().toISOString()}`;

    const subject = encodeURIComponent(subjectStr);
    const body = encodeURIComponent(bodyContent);
    const mailto = `mailto:${recipient}?subject=${subject}&body=${body}`;

    return { mailto, bodyContent };
  };

  const handleDispatch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptText.trim()) return;

    const { mailto, bodyContent } = generateMailtoPayload();

    if (mailto.length > 1900) {
      navigator.clipboard.writeText(bodyContent);
      setStatusMessage(
        lang === 'pt'
          ? '✓ Mensagem longa copiada para a área de transferência. Abrindo e-mail...'
          : '✓ Long message copied to clipboard. Opening email...'
      );
    } else {
      setStatusMessage(
        lang === 'pt'
          ? '✓ Transmissão gerada. Abrindo seu cliente de e-mail...'
          : '✓ Transmission generated. Opening your email client...'
      );
    }

    window.location.href = mailto;
  };

  const handleCopyPrompt = () => {
    if (!promptText.trim()) return;

    const cleanSender = senderInfo.replace(/[\r\n\t\0]/g, ' ').trim().slice(0, 100);
    const cleanPrompt = promptText.trim().slice(0, 2000);

    const fullPayload = `${cleanPrompt}\n\n[${lang === 'pt' ? 'Remetente' : 'Sender'}: ${
      cleanSender || (lang === 'pt' ? 'Não especificado' : 'Not specified')
    }]`;

    navigator.clipboard.writeText(fullPayload);
    setCopied(true);
    setStatusMessage(
      lang === 'pt' ? '✓ Mensagem copiada para a área de transferência!' : '✓ Message copied to clipboard!'
    );

    setTimeout(() => {
      setCopied(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }, 2000);
  };

  return (
    <div className="w-full mt-6 bg-zinc-900/90 border border-zinc-800 rounded-2xl p-5 sm:p-7 shadow-2xl relative font-mono space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#adff2f] animate-pulse" />
          <span className="text-xs uppercase font-bold text-white tracking-wider flex items-center gap-1.5">
            <MessageSquare className="w-3.5 h-3.5 text-[#adff2f]" />
            <span>{lang === 'pt' ? 'TRANSMISSÃO_DIRETA_SCAR' : 'DIRECT_SCAR_DISPATCH'}</span>
          </span>
          <span className="text-[10px] text-zinc-500 bg-zinc-800/80 px-2 py-0.5 rounded border border-zinc-700/50">
            UTF-8
          </span>
        </div>

        <div className="flex items-center gap-2 text-[11px] text-zinc-400">
          <span className="text-[#38bdf8]">dest:</span>
          <span className="text-zinc-200">reisfilho1116@gmail.com</span>
        </div>
      </div>

      <div className="space-y-2">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-[#adff2f]" />
          <span>{lang === 'pt' ? 'Tópicos Rápidos:' : 'Quick Topics:'}</span>
        </span>
        <div className="flex flex-wrap gap-1.5">
          {presets.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSelectPreset(p.text)}
              className="text-[11px] px-2.5 py-1 rounded-lg bg-zinc-950/80 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer text-left"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleDispatch} className="space-y-4">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[10px] text-zinc-400">
            <label htmlFor="contact-prompt-input" className="text-zinc-400">
              {lang === 'pt' ? '> Mensagem sobre pesquisa, engenharia ou parceria:' : '> Message regarding research, engineering, or collaboration:'}
            </label>
            <span className="text-zinc-500">{promptText.length} chars</span>
          </div>

          <textarea
            id="contact-prompt-input"
            rows={4}
            maxLength={2000}
            value={promptText}
            onChange={(e) => {
              setPromptText(e.target.value);
              if (statusMessage) setStatusMessage(null);
            }}
            placeholder={
              lang === 'pt'
                ? 'Ex: Olá Julio, gostaríamos de discutir os resultados do ScarNativeLM v5.0 / MoS / PointScarFlow...'
                : 'E.g., Hi Julio, we would love to discuss ScarNativeLM v5.0 / MoS / PointScarFlow...'
            }
            className="w-full p-3.5 bg-zinc-950 border border-zinc-800 focus:border-[#adff2f]/70 rounded-xl text-xs sm:text-sm text-zinc-100 placeholder-zinc-600 focus:outline-hidden transition-all resize-y font-mono leading-relaxed"
          />
        </div>

        <div className="space-y-1.5">
          <label htmlFor="contact-sender-input" className="text-[10px] text-zinc-400 block">
            {lang === 'pt'
              ? '> Seu Nome / Laboratório / Empresa & E-mail (opcional):'
              : '> Your Name / Lab / Company & Email (optional):'}
          </label>
          <input
            id="contact-sender-input"
            type="text"
            maxLength={100}
            value={senderInfo}
            onChange={(e) => setSenderInfo(e.target.value)}
            placeholder={
              lang === 'pt'
                ? 'Ex: Dra. Ana · AI Lab (ana@instituto.org)'
                : 'E.g., Dr. Alex · AI Research Lab (alex@lab.org)'
            }
            className="w-full px-3.5 py-2.5 bg-zinc-950 border border-zinc-800 focus:border-[#38bdf8]/60 rounded-xl text-xs text-zinc-200 placeholder-zinc-600 focus:outline-hidden transition-all font-mono"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="submit"
              disabled={!promptText.trim()}
              className="px-5 py-2.5 bg-[#adff2f] hover:bg-lime-300 disabled:opacity-40 disabled:hover:bg-[#adff2f] text-black text-xs font-bold rounded-xl transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-lime-400/10"
            >
              <Mail className="w-3.5 h-3.5" />
              <span>{lang === 'pt' ? 'Transmitir via E-mail' : 'Dispatch via Email'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={handleCopyPrompt}
              disabled={!promptText.trim()}
              className="px-4 py-2.5 bg-zinc-950 hover:bg-zinc-800 disabled:opacity-40 text-zinc-300 hover:text-white text-xs rounded-xl border border-zinc-800 transition-all flex items-center gap-1.5 cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-[#adff2f]" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? (lang === 'pt' ? 'Copiado!' : 'Copied!') : (lang === 'pt' ? 'Copiar Texto' : 'Copy Text')}</span>
            </button>
          </div>

          {onOpenTerminal && (
            <button
              type="button"
              onClick={onOpenTerminal}
              className="text-[11px] text-zinc-400 hover:text-[#adff2f] transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>scar_kernel --interactive</span>
            </button>
          )}
        </div>

        {statusMessage && (
          <div className="p-2.5 rounded-lg bg-[#adff2f]/10 border border-[#adff2f]/30 text-[11px] text-[#adff2f] flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 shrink-0" />
            <span>{statusMessage}</span>
          </div>
        )}
      </form>
    </div>
  );
};
