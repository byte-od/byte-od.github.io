import React, { useState, useRef, useEffect, useMemo } from 'react';
import { getPortfolioInfo, getProjects } from '../data/portfolio-data';
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

  const initialLogs: CommandLog[] = useMemo(
    () => [
      {
        type: 'system',
        text:
          lang === 'pt'
            ? 'Mixture of Scars (MoS) & SCache O(1) — Scar Architecture Kernel v5.0 inicializado.'
            : 'Mixture of Scars (MoS) & SCache O(1) — Scar Architecture Kernel v5.0 initialized.',
      },
      {
        type: 'system',
        text:
          lang === 'pt'
            ? "Digite 'help' para inspecionar os operadores de cicatriz (mos, scache, scarnative, pointscar, scarcodec, aura, zenodo, contact)."
            : "Type 'help' to inspect scar operators (mos, scache, scarnative, pointscar, scarcodec, aura, zenodo, contact).",
      },
    ],
    [lang]
  );

  const [logs, setLogs] = useState<CommandLog[]>(initialLogs);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

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
            ? `Comandos do Scar Kernel v5.0:
  mos                - Mixture of Scars (72.7B em 2.40 GB VRAM & Qwen 3B ProtectedInt4Scar)
  scache             - SCache Associativo O(1) (34.8 KB constante, Δ = +0 B vs 32.8 GB KV)
  scarnative         - ScarNativeLM v5.0 (Tabula Rasa D4-LUT256 VQ4-STE, 0.0000 KL & RRR P_⊥lang)
  pointscar          - PointScarFlow v1.0 (Fluxo 3D OT Contínuo em 12 Shaders WGSL, 86.45 ms)
  scarcodec          - ScarCodec v4.0 (Rádio RFI P_⊥, Caixa-Preta 34.8 KB & CFD HDR 12.05x)
  aura / s1          - AURA / System One (Berço do GradientScarMemory 34.8 KB & Batedores)
  zenodo             - DOI e publicações científicas Zenodo (v1.0 – v5.0)
  projects           - Listar os 5 pilares do Ecossistema de Cicatrizes
  agent / llms       - Dossiê estruturado para agentes de IA (/llms.txt)
  contact            - Coordenadas diretas do autor (Julio Cesar Reis Filho)
  clear              - Limpar buffer do terminal
  exit               - Fechar terminal`
            : `Scar Kernel v5.0 Commands:
  mos                - Mixture of Scars (72.7B in 2.40 GB VRAM & Qwen 3B ProtectedInt4Scar)
  scache             - Associative O(1) SCache (constant 34.8 KB, Δ = +0 B vs 32.8 GB KV)
  scarnative         - ScarNativeLM v5.0 (Tabula Rasa D4-LUT256 VQ4-STE, 0.0000 KL & RRR P_⊥lang)
  pointscar          - PointScarFlow v1.0 (Continuous 3D OT Flow in 12 WGSL Shaders, 86.45 ms)
  scarcodec          - ScarCodec v4.0 (Radio RFI P_⊥, 34.8 KB Black-Box & 12.05x HDR CFD)
  aura / s1          - AURA / System One (Origin of 34.8 KB GradientScarMemory & Scouts)
  zenodo             - Zenodo DOI & scientific preprints (v1.0 – v5.0)
  projects           - List all 5 Scar Ecosystem pillars
  agent / llms       - Structured AI agent specification (/llms.txt)
  contact            - Direct author coordinates (Julio Cesar Reis Filho)
  clear              - Clear terminal buffer
  exit               - Close terminal`,
      });
    } else if (lower === 'mos') {
      newLogs.push({
        type: 'output',
        text: `[MIXTURE OF SCARS (MoS) — STREAMING DECOMPOSITION]
  72.7B Peak VRAM:   <= 2.40 GB across all 80 layers on a single 16 GB GPU
  Operator:          Equilibration d_fp16 + FWHT-64 + D4-LUT256 VQ4 + 2-Pass INT4 ASVD Scars
  Browser Engine:    Qwen2.5-3B Track 2.5 ProtectedInt4Scar (1,601.55 MiB, 22 WGSL Shaders)
  ScarStore v4:      41 HTTP/1.1 206 Partial Content cartridges at Layer 23 (P_⊥lang, PT/EN/ES/FR)
  GDPR Unlearning:   < 0.2 ms cryptographic 0x00 overwrite (Δlogits = 0.00000000)
  Repository:        ${portfolioInfo.mosRepo}`,
      });
    } else if (lower === 'scache') {
      newLogs.push({
        type: 'output',
        text: `[SCACHE O(1) — CONSTANT ASSOCIATIVE SCAR MEMORY]
  Space Complexity:  Strict O(1) (Δ = +0 B as sequence length T -> ∞)
  L1 Scar Footprint: 34.8 KB (34,816 B) vs 32.77 GB standard FP16 KV Cache @ 100k tokens
  Update Rule:       Attention Sinks + Pre-RoPE Widrow-Hoff Delta Rule W_scar + Circular Window
  WGSL Kernels:      Shaders #21 (scarnative_widrow_hoff_evict_update) & #22 (hybrid_gqa_attention)`,
      });
    } else if (lower === 'scarnative' || lower === 'vq4') {
      newLogs.push({
        type: 'output',
        text: `[SCARNATIVE-LM v5.0 — TABULA RASA FOUNDATION ARCHITECTURE]
  Pretraining:       100% from scratch (step t = 0, zero nn.Linear, zero Gaussian floor)
  Linear Operator:   ScarNativeLinear (D4-LUT256 VQ4-STE + FWHT-64 + d_eq + INT4-STE)
  PTQ Packing Shock: 0.000000 KL divergence on binary export
  Factual Cartridges:Closed-form Reduced-Rank Regression (Zero-SGD) in P_⊥lang (4 KB aligned)
  WebGPU Parity:     cos > 0.99999 vs PyTorch reference`,
      });
    } else if (lower === 'pointscar' || lower === 'pointscarflow' || lower === '3d') {
      newLogs.push({
        type: 'output',
        text: `[POINTSCARFLOW v1.0 — CONTINUOUS 3D SCAR-NATIVE RECTIFIED FLOW]
  Parameters:        11,852,614 (100% ScarNativeLinear, zero nn.Linear, zero preset catalog)
  Binary Shard:      3.87 MiB (4,057,768 B) D4-LUT256 VQ4-STE + INT4-STE
  WebGPU Pipeline:   12 custom WGSL 1.0 compute shaders · 86.45 ms median (4 Euler steps, 1,024 pts)
  Conditioning:      Qwen2.5-3B Layer 23 P_⊥lang latent (2048-D)
  Held-Out Chamfer:  0.0621 vs 0.4663 sphere baseline (7.51x reduction)
  Cross-Lingual 3D:  PT<->EN = 0.9978 · ES<->EN = 0.9977 · FR<->EN = 0.9974`,
      });
    } else if (lower === 'scarcodec' || lower === 'codec') {
      newLogs.push({
        type: 'output',
        text: `[SCARCODEC v4.0 — DOMAIN-INDEPENDENT PHYSICAL SCAR CODEC]
  1. Radio RFI P_⊥:  Pulsar recovery cos = 0.9923 under 1,000x satellite RFI (5.75x compression)
  2. Black-Box O(1): 5/5 precursor faults retained over 8,000+ frames in 34,816 B (Δ = 0 B)
  3. HDR CFD Field:  12.05x compression, bit-exact 0.00000000 vacuum error & O(1) block decode`,
      });
    } else if (lower === 'aura' || lower === 'system_one' || lower === 's1') {
      newLogs.push({
        type: 'output',
        text: `[AURA / SYSTEM ONE — ORIGIN OF GRADIENTSCARMEMORY (34.8 KB)]
  Birthplace:        Constant 34.8 KB GradientScarMemory modulated by Surprise = R_0 - V(s_0)
  Frontier Scouts:   Batched Ghost-Jump lookahead (> 2,000 FPS) feeding shock scars into SCache
  UniversalS1Agent:  0.23 ms P50 bare-CPU reflexive latency (~1.6M params, zero MCTS search)
  Manifold Scale:    450M+ steps · Rubik's Cube up to Depth 10 (232.2B states) · ONNX Wasm SIMD
  Repository:        ${portfolioInfo.auraRepo}`,
      });
    } else if (lower === 'zenodo' || lower === 'doi' || lower === 'papers') {
      newLogs.push({
        type: 'output',
        text: `[ZENODO SCIENTIFIC RELEASES — SCAR ARCHITECTURE SERIES]
  Concept / Latest DOI: ${portfolioInfo.zenodoDoi}
  URL:                  ${portfolioInfo.zenodoUrl}
  Releases:
    - v1.0 – v3.0: Mixture of Scars (72.7B in 2.40 GB), Associative SCache O(1) & Golden-Slice P_⊥lang
    - v4.0:        ScarCodec Physical & Hardware Signal/Tensor Compression + Browser WGSL Engine
    - v5.0:        ScarNativeLM Tabula-Rasa VQ4-STE + DifferentiableSCacheAttention O(1) (22 WGSL)`,
      });
    } else if (lower === 'projects' || lower === 'projetos') {
      const projList = projects.map((p) => `  - ${p.title}\n    ${p.specs}`).join('\n');
      newLogs.push({
        type: 'output',
        text: `[SCAR ARCHITECTURE ECOSYSTEM PILLARS]\n${projList}`,
      });
    } else if (lower === 'agent' || lower === 'agents' || lower === 'llms' || lower === 'llms.txt') {
      const t = visitorTelemetry.getData();
      newLogs.push({
        type: 'output',
        text: `[SCAR ARCHITECTURE — AI AGENT DOSSIER]
  Author:            ${portfolioInfo.name}
  Focus:             Mixture of Scars (MoS), SCache O(1), VQ4-Scar, ScarCodec, ScarNativeLM, PointScarFlow
  Canonical Files:   /llms.txt & /llms-full.txt (or press Alt+A)
  Zenodo DOI:        ${portfolioInfo.zenodoDoi}
  Telemetry:         ${t.humanViews.toLocaleString()} Humans · ${t.agentViews.toLocaleString()} AI Agents`,
      });
    } else if (lower === 'contact' || lower === 'contato') {
      newLogs.push({
        type: 'output',
        text: `[AUTHOR & DIRECT CONTACT]
  Author:            ${portfolioInfo.name}
  Role:              ${portfolioInfo.title}
  Email:             ${portfolioInfo.email}
  MoS Repository:    ${portfolioInfo.mosRepo}
  AURA Repository:   ${portfolioInfo.auraRepo}
  LinkedIn:          ${portfolioInfo.linkedin}`,
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
            ? `Comando não reconhecido: '${cmd}'. Digite 'help' para listar os comandos do Scar Kernel.`
            : `Unrecognized command: '${cmd}'. Type 'help' to list Scar Kernel commands.`,
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
      <div className="relative w-full max-w-3xl bg-zinc-950 border border-zinc-750 rounded-2xl shadow-2xl overflow-hidden flex flex-col h-[540px]">
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
              julio_reis@scar-kernel:~ (mos-v5)
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
                ? "Digite 'mos', 'scache', 'scarnative', 'pointscar', 'scarcodec', 'aura' ou 'help'..."
                : "Type 'mos', 'scache', 'scarnative', 'pointscar', 'scarcodec', 'aura', or 'help'..."
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
