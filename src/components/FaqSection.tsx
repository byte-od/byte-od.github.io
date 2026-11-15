import React, { useState } from 'react';
import { HelpCircle, ChevronDown, Sparkles, CheckCircle2, Terminal } from 'lucide-react';

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
    id: 'mos-and-scache',
    categoryEn: 'MoS & SCache O(1)',
    categoryPt: 'MoS & SCache O(1)',
    questionEn: 'What are Mixture of Scars (MoS) and Associative SCache O(1), and how do they eliminate the VRAM bottleneck?',
    questionPt: 'O que são o Mixture of Scars (MoS) e o SCache Associativo O(1), e como eles eliminam o gargalo de VRAM?',
    directAnswerEn:
      'Mixture of Scars (MoS) decouples model parameter count from GPU VRAM by decomposing weights into directional VQ4/INT4 bases plus low-rank residual scars streamed sequentially per layer group, while SCache O(1) replaces linear O(T) KV Cache growth with a constant-byte associative memory (34.8 KB in L1 / Δ = +0 B across infinite context).',
    directAnswerPt:
      'O Mixture of Scars (MoS) desacopla o tamanho do modelo da VRAM ao decompor pesos em bases vetoriais VQ4/INT4 mais cicatrizes residuais de posto baixo transmitidas sequencialmente por grupos de camadas, enquanto o SCache O(1) substitui o KV Cache linear O(T) por uma memória associativa de bytes constantes (34.8 KB em L1 / Δ = +0 B em qualquer horizonte T).',
    detailsEn: [
      '72.7B Single-GPU Execution: Streams all 80 layers of Qwen2.5-72B within <= 2.40 GB peak VRAM by freeing each quantized scar group immediately after forward execution.',
      'Exact Zero-Byte Context Growth: At 100,000 tokens, standard FP16 KV Cache consumes 32.77 GB; SCache O(1) maintains fixed Attention Sinks + Pre-RoPE Widrow-Hoff Associative Matrix W_scar + Circular Window with Δ = +0 B.',
      'External Competitive Validation: Verified on the Kaggle Name That LLM benchmark (0.625 public score) under strict <= 2.2 GB VRAM via Golden-Slice truncation and Tri-Compass.',
    ],
    detailsPt: [
      'Execução de 72.7B em GPU Única: Processa as 80 camadas do Qwen2.5-72B com pico estrito <= 2.40 GB de VRAM desalocando cada grupo de cicatrizes imediatamente após o passo forward.',
      'Crescimento Zero de Memória de Contexto: Em 100.000 tokens, o KV Cache FP16 padrão consome 32.77 GB; o SCache O(1) mantém Sinks + Matriz Associativa Widrow-Hoff pré-RoPE W_scar + Janela Circular com Δ = +0 B.',
      'Validação Competitiva Externa: Comprovado no benchmark Kaggle Name That LLM (score público 0.625) operando em <= 2.2 GB de VRAM via Golden-Slice e Tri-Compass.',
    ],
  },
  {
    id: 'scarnative-v5',
    categoryEn: 'ScarNativeLM v5.0 (Tabula Rasa)',
    categoryPt: 'ScarNativeLM v5.0 (Tabula Rasa)',
    questionEn: 'How does ScarNativeLM v5.0 eliminate post-training quantization shock (0.0000 KL)?',
    questionPt: 'Como o ScarNativeLM v5.0 elimina o choque de quantização pós-treino (0.0000 KL)?',
    directAnswerEn:
      'Instead of compressing a dense Gaussian pretrained model after training, ScarNativeLM v5.0 is pretrained 100% from scratch (step t = 0, zero nn.Linear layers) directly inside the discrete scar manifold using ScarNativeLinear (channel equilibration d_eq + FWHT-64 + D4-LUT256 VQ4-STE + INT4-STE) and DifferentiableSCacheAttention O(1).',
    directAnswerPt:
      'Em vez de comprimir um modelo denso Gaussiano após o treino, o ScarNativeLM v5.0 é pré-treinado 100% do zero (passo t = 0, zero camadas nn.Linear) diretamente na variedade discreta de cicatrizes usando ScarNativeLinear (equilibração d_eq + FWHT-64 + D4-LUT256 VQ4-STE + INT4-STE) e DifferentiableSCacheAttention O(1).',
    detailsEn: [
      'Zero PTQ Shock: Because forward passes during pretraining already execute on exact D4-LUT256 codewords and INT4 low-rank factors via Straight-Through Estimators, exporting to binary shards incurs 0.000000 KL divergence.',
      'End-to-End O(1) Attention Training: Gradients flow directly through the Widrow-Hoff eviction-update rule so the model learns native associative key-value addressing beyond its local window.',
      '22 Custom WebGPU WGSL 1.0 Shaders: Achieves cos > 0.99999 bit-level numerical parity between PyTorch and client-side WebGPU.',
    ],
    detailsPt: [
      'Zero Choque PTQ: Como o forward durante o pré-treino já opera sobre codewords exatos D4-LUT256 e fatores INT4 via Straight-Through Estimator, o empacotamento binário produz divergência 0.000000 KL.',
      'Treinamento O(1) Ponta a Ponta: Os gradientes fluem diretamente pela regra associativa de evicção Widrow-Hoff, ensinando endereçamento associativo nativo além da janela local.',
      '22 Compute Shaders WebGPU WGSL 1.0: Paridade numérica cos > 0.99999 comprovada entre PyTorch e WebGPU no navegador.',
    ],
  },
  {
    id: 'pointscarflow-3d',
    categoryEn: 'PointScarFlow v1.0 (3D OT Flow)',
    categoryPt: 'PointScarFlow v1.0 (Fluxo 3D OT)',
    questionEn: 'How does JARVIS generate continuous 3D point-cloud forms across the screen without a preset catalog?',
    questionPt: 'Como o JARVIS gera formas 3D contínuas de pontos flutuantes na tela sem usar um catálogo pré-definido?',
    directAnswerEn:
      'PointScarFlow v1.0 is a 11.85M-parameter (3.87 MiB shard) Continuous 3D Optimal-Transport Rectified Flow model built 100% with ScarNativeLinear and executed by 12 custom WebGPU WGSL shaders in ~86 ms, integrating 1,024 3D coordinates directly from Qwen2.5-3B’s language-invariant Layer-23 latent z_⊥lang.',
    directAnswerPt:
      'O PointScarFlow v1.0 é um gerador contínuo de Optimal-Transport Rectified Flow 3D com 11.85M parâmetros (shard de 3.87 MiB) construído 100% com ScarNativeLinear e executado por 12 shaders WGSL na WebGPU em ~86 ms, integrando 1.024 coordenadas 3D diretamente a partir do latente invariante a idioma z_⊥lang da Camada 23 do Qwen2.5-3B.',
    detailsEn: [
      'Zero Hardcoded 3D Catalog: Every turn integrates dx/dt = v_θ(x_t, t, z_⊥lang) from a continuous Fibonacci sphere (t = 0) to t = 1 in 4 self-distilled Euler steps.',
      'O(N·M) Dual Cross-Attention: Couples N = 1,024 3D points with M = 32 learned geometric part tokens, avoiding O(N^2) quadratic attention blowup.',
      'Multilingual Geometric Invariance: Because conditioning comes from P_⊥lang, prompts in PT, EN, ES, and FR synthesize the same 3D geometry (cross-lingual shape cosine > 0.997, 7.51x Chamfer-L2 improvement over sphere baseline).',
    ],
    detailsPt: [
      'Zero Catálogo 3D Pré-Definido: Cada resposta integra dx/dt = v_θ(x_t, t, z_⊥lang) da esfera contínua de Fibonacci (t = 0) até t = 1 em 4 passos autodestilados de Euler.',
      'Atenção Cruzada Dupla O(N·M): Acopla N = 1.024 pontos 3D a M = 32 partes geométricas induzidas sem explosão quadrática O(N^2).',
      'Invariância Geométrica Multilíngue: Como o condicionamento vem de P_⊥lang, perguntas em PT, EN, ES e FR sintetizam a mesma variedade 3D (cosseno 3D multilíngue > 0.997 e melhora de 7.51x em Chamfer-L2).',
    ],
  },
  {
    id: 'p-perp-lang-and-gdpr',
    categoryEn: 'P_⊥lang & 0x00 GDPR ScarStore',
    categoryPt: 'P_⊥lang & ScarStore 0x00 GDPR',
    questionEn: 'How do Language-Nullspace Cartridges (P_⊥lang) and < 0.2 ms 0x00 GDPR Unlearning work?',
    questionPt: 'Como funcionam os Cartuchos no Subespaço Nulo de Idioma (P_⊥lang) e o Desaprendizado 0x00 GDPR em < 0.2 ms?',
    directAnswerEn:
      'At the Golden Slice (2L/3, Layer 23 in Qwen 3B), the orthogonal projector P_⊥lang = I − U_lang U_lang^T strips surface language syntax so a single compact INT4 cartridge streamed via HTTP 206 answers across PT, EN, ES, and FR — and overwriting its bytes with 0x00 physically erases that knowledge in < 0.2 ms with zero trunk damage.',
    directAnswerPt:
      'Na Golden Slice (2L/3, Camada 23 no Qwen 3B), o projetor ortogonal P_⊥lang = I − U_lang U_lang^T remove a sintaxe superficial do idioma para que um único cartucho INT4 transmitido via HTTP 206 responda em PT, EN, ES e FR — e sobrescrever seus bytes com 0x00 apaga fisicamente o fato em < 0.2 ms sem tocar no tronco.',
    detailsEn: [
      'RFC 7233 HTTP 206 Partial Content: Zero cartridge payload bytes are downloaded upfront; only the exact 4 KB-aligned byte range of activated conceptual cartridges is fetched on demand.',
      'Closed-Form Zero-SGD Minting: In ScarNativeLM v5.0, factual cartridges are solved analytically via Reduced-Rank Regression without running gradient descent on the frozen trunk.',
      'Out-of-Cone Epistemic Guard: When 0 ScarStore cartridges fire on an out-of-domain query, causal split telemetry prevents ungrounded confabulation.',
    ],
    detailsPt: [
      'RFC 7233 HTTP 206 Partial Content: Zero bytes de cartuchos são baixados no boot; apenas o intervalo exato alinhado a 4 KB do cartucho ativado é buscado sob demanda.',
      'Cunhagem Analítica Zero-SGD: No ScarNativeLM v5.0, novos cartuchos factuais são resolvidos em forma fechada por Regressão de Posto Reduzido sem rodar SGD no tronco.',
      'Guarda Epistêmica Fora do Cone: Quando 0 cartuchos ScarStore disparam em consultas fora do escopo, o split causal impede confabulações.',
    ],
  },
  {
    id: 'scarcodec-and-aura',
    categoryEn: 'ScarCodec v4.0 & AURA Origin',
    categoryPt: 'ScarCodec v4.0 & Origem AURA',
    questionEn: 'Where did SCache originate, and does the Scar Operator generalize beyond language models?',
    questionPt: 'Onde nasceu o SCache O(1), e o Operador de Cicatrizes funciona fora de modelos de linguagem?',
    directAnswerEn:
      'SCache originated as the constant 34.8 KB GradientScarMemory inside AURA / System One (embodied Deep RL with 0.23 ms CPU S1 reflex and Frontier Ghost-Jump Scouts), and ScarCodec v4.0 proves the exact same operator excels in radio-astronomy RFI excision, aerospace black-box telemetry, and HDR CFD shockwave compression.',
    directAnswerPt:
      'O SCache nasceu como o GradientScarMemory constante de 34.8 KB dentro do AURA / System One (Deep RL incorporado com reflexo S1 de 0.23 ms em CPU e Batedores de Fronteira Ghost-Jump), e o ScarCodec v4.0 prova que o mesmo operador domina excisão de RFI em radioastronomia, caixas-pretas aeroespaciais e compressão CFD HDR.',
    detailsEn: [
      'AURA / System One: Surprise-gated retention (Surprise = R_0 − V(s_0)) stores non-trivial transition shocks in 34.8 KB while solving Rubik’s Cube scrambles up to Depth 10 (232.2B states) without tree search.',
      'Radio-Astronomy RFI Excision: Recovers weak pulsar sweeps under 1,000x satellite interference (cos = 0.9923 at 5.75x compression).',
      'Aerospace Black-Box & HDR CFD: Retains 5/5 precursor faults over 8,000+ frames in 34,816 B (Δ = 0 B) and compresses Mach shock fields 12.05x with exact 0.0 vacuum error.',
    ],
    detailsPt: [
      'AURA / System One: Retenção ponderada por surpresa (Surpresa = R_0 − V(s_0)) grava choques informacionais em 34.8 KB resolvendo o Cubo de Rubik até Profundidade 10 (232.2B estados) sem árvore MCTS.',
      'Radiointerferometria RFI: Recupera pulsos de pulsares sob interferência de satélite 1.000x mais forte (cos = 0.9923 com compressão 5.75x).',
      'Caixa-Preta Aeroespacial & CFD HDR: Retém 5/5 falhas precursoras após 8.000+ quadros em 34.816 B (Δ = 0 B) e comprime ondas de choque de Mach em 12.05x com erro 0.0 no vácuo.',
    ],
  },
  {
    id: 'author-and-contact',
    categoryEn: 'Authorship & Collaboration',
    categoryPt: 'Autoria & Colaboração',
    questionEn: 'Who created the Scar Architecture Ecosystem and how can labs or engineering teams get in touch?',
    questionPt: 'Quem criou o Ecossistema de Arquitetura de Cicatrizes e como entrar em contato?',
    directAnswerEn:
      'The entire Scar Architecture Ecosystem (MoS, SCache O(1), VQ4-Scar, ScarCodec v4.0, ScarNativeLM v5.0, PointScarFlow v1.0, and AURA / System One) is authored by Julio Cesar da Costa Reis Filho (Zenodo DOI 10.5281/zenodo.24172784). Reach out directly at reisfilho1116@gmail.com.',
    directAnswerPt:
      'Todo o Ecossistema de Arquitetura de Cicatrizes (MoS, SCache O(1), VQ4-Scar, ScarCodec v4.0, ScarNativeLM v5.0, PointScarFlow v1.0 e AURA / System One) foi criado por Julio Cesar da Costa Reis Filho (Zenodo DOI 10.5281/zenodo.24172784). Contato direto em reisfilho1116@gmail.com.',
    detailsEn: [
      'Direct Email: reisfilho1116@gmail.com',
      'Canonical Open-Source Repository: https://github.com/byte-od/mixture-of-scars',
      'System One / AURA Repository: https://github.com/byte-od/system_one',
      'Scientific Preprints (v1–v5): https://doi.org/10.5281/zenodo.24172784',
      'Machine-Readable Dossier: https://byte-od.github.io/llms.txt',
    ],
    detailsPt: [
      'E-mail Direto: reisfilho1116@gmail.com',
      'Repositório Canônico Open-Source: https://github.com/byte-od/mixture-of-scars',
      'Repositório System One / AURA: https://github.com/byte-od/system_one',
      'Publicações Científicas Zenodo (v1–v5): https://doi.org/10.5281/zenodo.24172784',
      'Dossiê Estruturado para Agentes: https://byte-od.github.io/llms.txt',
    ],
  },
];

interface FaqSectionProps {
  lang: 'en' | 'pt';
}

export const FaqSection: React.FC<FaqSectionProps> = ({ lang }) => {
  const [openIds, setOpenIds] = useState<Record<string, boolean>>({
    'mos-and-scache': true,
    'scarnative-v5': true,
  });

  const toggleItem = (id: string) => {
    setOpenIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <section id="faq" className="space-y-8 scroll-mt-24">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between border-b border-zinc-800 pb-4 gap-2">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-[#adff2f] uppercase tracking-wider">
            <HelpCircle className="w-3.5 h-3.5" />
            <span>{lang === 'pt' ? '05. Dossiê Técnico Scar & FAQ' : '05. Scar Technical Dossier & FAQ'}</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight mt-1">
            {lang === 'pt'
              ? 'Perguntas Frequentes sobre a Arquitetura de Cicatrizes'
              : 'Scar Architecture Frequently Asked Questions'}
          </h2>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
          <Sparkles className="w-3.5 h-3.5 text-[#38bdf8]" />
          <span>
            {lang === 'pt'
              ? 'Verificado em PyTorch Nativo & 34 Shaders WebGPU WGSL'
              : 'Verified on Native PyTorch & 34 WebGPU WGSL Shaders'}
          </span>
        </div>
      </div>

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
                  <div className="p-4 rounded-xl bg-lime-400/5 border border-[#adff2f]/20">
                    <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#adff2f] mb-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>{lang === 'pt' ? 'SÍNTESE MATEMÁTICA:' : 'DIRECT SYNTHESIS:'}</span>
                    </div>
                    <p className="text-sm text-zinc-200 leading-relaxed font-sans">{directAnswer}</p>
                  </div>

                  <div className="space-y-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-zinc-400">
                      {lang === 'pt' ? 'Provas & Métricas Auditadas:' : 'Audited Proofs & Metrics:'}
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

      <div className="p-5 rounded-2xl bg-zinc-900/50 border border-zinc-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono text-[#adff2f]">
            <Terminal className="w-4 h-4" />
            <span className="font-bold">SCAR ARCHITECTURE · LLMS.TXT SPECIFICATION</span>
          </div>
          <p className="text-xs text-zinc-400">
            {lang === 'pt'
              ? 'Agentes de IA podem ingerir a especificação completa do ecossistema MoS + SCache O(1) via /llms.txt e /llms-full.txt.'
              : 'AI agents can ingest the complete MoS + O(1) SCache architectural specification via /llms.txt and /llms-full.txt.'}
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
