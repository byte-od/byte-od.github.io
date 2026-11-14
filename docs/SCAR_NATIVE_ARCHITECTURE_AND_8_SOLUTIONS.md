# Arquitetura Unificada de Cicatrizes (`Nativo + WebGPU`) e Mapa das 8 Soluções Estruturais

> **Diretriz Normativa do Ecossistema (`System One` + `Mixture of Scars` + `SCache O(1)`):**
> A substituição da quantização escalar Gaussiana legada pelo mecanismo **`VQ4-Scar` (`Micro-Cicatrizes Vetoriais 4D LUT256` + `Equilibração de Canal` + `WHT-64` + `Cicatrizes ASVD Comprimidas em 2 Passos`)** **NÃO é exclusiva para Web/Browser**.
> Ela constitui um **primitivo universal de todo o ecossistema** — operando com formato binário unificado tanto no **Motor Nativo (`PyTorch / Metal MPS / CUDA / C++`)** quanto no **Motor Web (`WebGPU / WGSL 1.0`)**.
> **Regra Obrigatória de Engenharia:** Toda evolução de operadores de cicatriz deve ser validada e testada **simultaneamente em AMBOS os ambientes (`Nativo` E `WebGPU`)**, exigindo paridade numérica (`cosine > 0.99999`) e validação poliglota *held-out* (`PT`, `EN`, `ES`, `FR`).

---

## Parte I — Síntese das Provas Científicas e Empíricas (`Qwen2.5-3B` em `PT / EN / ES / FR`)

### 1. Anatomia Forense do Ruído em Modelos Poliglotas (`scripts/forensic_2bit_noise_anatomy.py` & `scripts/test_anisotropic_polyglot_3b.py`)
1. **Assimetria Extrema Peso vs. Ativação:**
   - Os pesos $W$ parecem quase Gaussianos (`W_kurt = 3,38..5,44`) porque empilham milhares de conceitos sobrepostos (Teorema Central do Limite), mas as ativações $X$ possuem *outliers* violentos concentrados em poucos canais (`X_kurt = 10.556` na `L0.down_proj` até `80.252` na `L35.down_proj`, pico `607x` acima do RMS).
2. **Efeito Cascata nas Camadas de *Attention Sink* (`L0..L4`):**
   - Comprimir apenas `L0..L5` em 2 bits escalares colapsa o modelo inteiro (`KL = 11,2941`, `Top-1 = 0,0%`), pois corrói os *Attention Sinks* onde mora a âncora sintática do idioma do usuário.
3. **Tríade de Projeções Críticas:**
   - `mlp.gate_proj` (`KL = 10,46`, chave não-linear `SiLU`), `mlp.down_proj` (`KL = 9,78`) e `self_attn.o_proj` (`KL = 9,38`) concentram a explosão de ruído, enquanto `q/k/v_proj` e `up_proj` são muito mais resilientes.
4. **Equação da Amplificação de *Outliers* na Rotação Hadamard (`WHT`) sem Equilibração:**
   - Canais $j^*$ com ativação gigante ($|x_{j^*}| \sim 500$) costumam ter colunas de peso pequenas ($\|W_{:, j^*}\| \sim 10^{-4}$). Aplicar Hadamard `WHT` direto em $W$ mistura $W_{:, j^*}$ com vizinhos `70x` maiores e injeta ruído `70x` maior de volta no canal multiplicado por `500`!
   - **Solução (`d_fp16`):** Equilibrar as colunas por $d_j = (\text{RMS}(X_j)/\text{med})^\beta$ antes da `WHT` e dividir o vetor de entrada $x$ uma única vez por $d_j$ elimina essa amplificação custando apenas `2 * K` bytes por matriz (`3,1 MiB` no modelo `3B` inteiro).
5. **Por que Grades Gaussianas Escalares (`NF3`) Falham fora do Inglês e por que `INT4` Puro Vaza Chinês:**
   - Grades pares simétricas (`NF3`) **não possuem o `0.0` exato** (injetando chiado em pesos nulos) e, ao comprimir `embed_tokens`, fazem o Inglês engolir sub-palavras acentuadas de `PT/ES/FR`.
   - Mesmo em `Block-64 INT4` puro (`1,53 GiB` sem cicatrizes `ASVD`), o `Qwen2.5-3B` **vazou sub-tokens de Chinês no meio do Francês no token ~90** (`"...interactions微观的量子力学..."`), provando que **escudos de cicatriz `ASVD` poliglotas são indispensáveis** até em 4 bits.

---

### 2. A Superioridade do `VQ4-Scar` + Cura Residual `INT4-ASVD` em 2 Passos (`scripts/test_vector_scar_codebook_3b.py` & `scripts/test_sub1gb_vq4_compressed_scars_3b.py`)

Em vez de esmagar escalares individuais em gavetas Gaussianas, cada matriz é decomposta no operador unificado **Nativo + WebGPU**:

$$\widehat{W} = \Big(\text{FWHT}_{64}^{-1}\big(\alpha_b \cdot \text{LUT}_{256}[\text{codes}]\big) \oslash d_{\text{fp16}}\Big) + Q_{\text{INT4}}(B_{\text{scar}}) \, Q_{\text{INT4}}(A_{\text{scar}})$$

- **`LUT256` (`256 × vec4<f16>` = `2 KB` por matriz, `4 KB` em `var<workgroup>` WGSL):**
  - Entrada `0` travada no **vetor nulo exato `(0, 0, 0, 0)`**;
  - Inicializada com espigas esparsas de eixo único + **reticulado ótimo $D_4$** (*kissing number* 24 em 4D) + hipercubos e refinada via **GPU Lloyd-Max ponderado por energia $\alpha_b^2$**.
- **Cura Residual em 2 Passos (*2-Pass Residual Scar Healing* com Cicatrizes `INT4` `3,76x` menores):**
  - Na Camada 18 (*held-out*, `637` tokens inéditos nos 4 idiomas, `0%` de sobreposição com calibração), a cicatriz `INT4` em 2 passos (`6,48 MiB`/matriz) **bateu a própria cicatriz `FP16` de 16 bits (`7,65 MiB`/matriz)**: erro caiu de `15,59%` para **`15,33%`** na `gate_proj` (`14,82%` em `r=96`) e de `25,81%` para **`25,45%`** na `down_proj` (`24,68%` em `r=96`)!

#### Fronteira de Pareto Verificada no `Qwen2.5-3B-Instruct` (`6,17 GB` FP16 original) em `PT / EN / ES / FR`

| Regime / Configuração | Tamanho Físico | Compressão vs FP16 | KL Médio (4 idiomas) | Top-1 vs FP16 | Resultado Empírico Poliglota (`PT / EN / ES / FR`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *(Legado)* Gaussiana `2/3-Bit` + WHT | `1.027 MiB` (`1,003 GiB`) | `5,9x` | `3.7905` | `27,6%` | Colapsou em Inglês repetitivo nos 4 idiomas. |
| *(Legado)* Gaussiana `NF3` 3-Bit + WHT | `1.196 MiB` (`1,168 GiB`) | `5,0x` | `2.9290` | `38,7%` | **PT** travou em loop e **FR** respondeu em Inglês. |
| **Opção 1A:** `VQ4 + 2-Pass INT4-Scar` (`embed INT3`) | `943 MiB` (`0,921 GiB`) | `6,4x` | `1.7728` | `56,2%` | Piso físico: comprimir `embed_tokens` em `INT3` faz ES escorregar para PT no fim da geração. |
| **Opção 1B:** **`VQ4 + 2-Pass INT4-Scar` (`embed INT4`)** | **`1.016 MiB` (`0,992 GiB`)** | **`5,9x`** | **`1.5561`** | **`59,3%`** (`PT 64,0%`) | **Fura `< 1,00 GiB` mantendo `PT`, `EN`, `ES` e `FR` 100% firmes no idioma do início ao fim!** |
| **Config 2:** `VQ4-Scar` Híbrido (`FP16` Scars) | `1.176 MiB` (`1,149 GiB`) | `5,1x` | `1.4343` | `61,5%` | Menor que `NF3` com `> 2x` menor `KL` e fluência nos 4 idiomas. |
| **Trilha 2.4-A (Lean):** `VQ4 + INT4-Scar + gate_proj INT4` | `1.150 MiB` (`1,123 GiB`) | `5,4x` | `1.4698` | `60,5%` (`PT 65,1%`) | Paridade física `Nativo <-> WebGPU` `cos = 1.000000`; exige `r=96` nas bordas `L34..35` (`up_proj`) contra colisão CJK. |
| **Trilha 2.4-B (Triad):** `VQ4 + INT4-Scar + gate/o_proj INT4` | `1.190 MiB` (`1,163 GiB`) | `5,2x` | `1.5111` | `63,2%` (`PT 68,6%`) | Menor que `NF3` com salto de `Top-1` para `63,2%` (`FR/EN/ES` cristalinos; identificou sensibilidade CJK de `up_proj` em `L34..35`). |
| **Config 3 (Roadmap #2):** **`MoS INT4 + ASVD Poliglota`** | **`1.631 MiB` (`1,593 GiB`)** | **`3,7x`** | **`0.6938`** | **`74,5%`** (`ES 80,4%`) | **Elimina 100% do vazamento `FR -> ZH`** e entrega rigor de livro-texto nos 4 idiomas. |

---

## Parte II — Mapa Direto das 8 Soluções Estruturais (Onde Nosso Mecanismo Melhora ou Substitui o Legado)

### 1. Na Camada de Pesos, Compiladores e Silício (Substituição da Quantização Escalar Gaussiana)
- **Mecanismo Legado:** Quantização escalar Gaussiana (`INT4`, `NF4`, `GPTQ`, `AWQ`) — arredonda cada peso isoladamente; colapsa abaixo de 3 bits, carece de zero exato em grades pares e amplifica *outliers* de ativação.
- **Nossa Solução (`VQ4-Scar` + Cura Residual em 2 Passos — Nativo & WebGPU):**
  - Substitui gavetas escalares por **Dicionários de Micro-Cicatrizes Direcionais (`LUT256` em $\mathbb{R}^4$ com `(0,0,0,0)` exato + reticulado $D_4$)** + equilibração `d_fp16` + rotação ortonormal `FWHT-64` aplicada apenas **uma vez no vetor de entrada $x$** (`0,002 ms`) + escudos residuais `INT4-ASVD`.
  - **Impacto em Hardware (CUDA / Metal / WebGPU / NPUs Mobile):** A tabela `LUT256` inteira cabe em **`4 KB`** de SRAM/memória compartilhada (`var<workgroup>` / shared memory L1), convertendo largura de banda de DRAM em leitura em registrador (`1 u32 = 16 pesos`). No modo **Nativo (`72B MoS`)**, permite reduzir ainda mais o grupo ativo em VRAM e o footprint em disco sem depender de quantizadores off-the-shelf.

### 2. Na Memória Contextual Contínua (Substituição do `KV Cache` Linear $\mathcal{O}(T)$)
- **Mecanismo Legado:** `KV Cache` linear (`PagedAttention`, `vLLM`, `RingAttention`) — cresce indefinidamente com a conversa ($\mathcal{O}(T)$), estoura VRAM e degrada a atenção no meio do contexto (*lost-in-the-middle*).
- **Nossa Solução (`SCache O(1)` guiado por Choque Informacional):**
  - Combina **Attention Sinks permanentes** (blindando System Prompt e identidade) + **Janela Deslizante Local** + **Slots Associativos + Fast-Weights Widrow-Hoff** promovidos por surpresa epistêmica ($\Delta = +0\text{ B}$ em turnos infinitos).
  - **Onde substitui de vez:** Agentes `24/7` nativos em servidor/desktop, assistentes de sistema operacional, robôs contínuos, NPCs persistentes e agentes em navegador.

### 3. Na Injeção de Conhecimento Dinâmico (Substituição do `RAG` Textual Pesado)
- **Mecanismo Legado:** `RAG` tradicional (`Modelo de Embedding separado` + `Vector DB` + colar milhares de tokens brutos no prompt) — encarece o *prefill*, aumenta latência, consome contexto e alucina quando *chunks* colidem.
- **Nossa Solução (`Latent HTTP 206 / mmap ScarStore` + Fatia Dourada):**
  - **Zero banco de texto no prompt e zero modelo externo de embedding.** O estado oculto da camada intermediária (`Golden Slice`) roteia em microssegundos e carrega via `mmap` (Nativo) ou `HTTP 206 Range` (Web) apenas **`61 KB – 140 KB`** do operador $(B_m, A_m)$ direto na sinapse, garantindo **invariância matemática `0.0000000000` fora do cone**.

### 4. Em Aprendizado Contínuo e Governança LGPD/GDPR (Substituição de `LoRA / Fine-Tuning` com SGD)
- **Mecanismo Legado:** `LoRA / SFT` via *backpropagation* + Adam (`3x` VRAM, horas de GPU, esquecimento catastrófico e impossibilidade prática de remover um único fato sob auditoria).
- **Nossa Solução (`Zero-SGD Closed-Form RRR Scars` + `Sleep-Consolidation Bridge Scars`):**
  - **Aprendizado instantâneo sem gradiente:** Regressão de posto reduzido regularizada em forma fechada em segundos.
  - **Desaprendizado exato (*True Machine Unlearning*):** Direito ao esquecimento vira deletar um cartucho de `61 KB`, restaurando matematicamente o bit exato anterior sem retreinar o modelo.
  - **Consolidação de Sono (`Bridge Cartridges`):** Fusão algébrica offline de duas cicatrizes para raciocínio composicional entre domínios.

### 5. Na Escala de Modelos Gigantes (Evolução de `MoE` Residente para `Streaming MoS`)
- **Mecanismo Legado:** `Mixture of Experts (MoE)` clássico — exige dezenas ou centenas de GBs de RAM/VRAM residentes porque cada especialista é um bloco MLP denso gigantesco.
- **Nossa Solução (`Streaming Mixture of Scars` — Local Disk/NVMe `mmap` no Nativo + `HTTP 206` na Web):**
  - Um único **Tronco Sintático/Lógico enxuto** + milhares ou milhões de **Especialistas de Cicatriz ultra-leves** transmitidos sob demanda da NVMe (Nativo) ou CDN estática (Web), desacoplando o conhecimento total do sistema da memória física da GPU.

### 6. Em Sistemas Poliglotas Globais (Substituição de Duplicação/Tradução Multilíngue)
- **Mecanismo Legado:** Treinar adaptadores duplicados por idioma ou encadear tradutores lentos (`PT -> EN -> PT`).
- **Nossa Solução (Projetor Ortogonal de Espaço Nulo de Idioma $P_{\perp\text{lang}} = I - U_{\text{lang}} U_{\text{lang}}^\top$):**
  - Remove cirurgicamente as ~`14` direções sintáticas de idioma na camada intermediária: um cartucho ancorado uma única vez em Inglês dispara nativamente em `PT`, `EN`, `ES` ou `FR` e gera a resposta direto na língua do usuário.

### 7. Em *Test-Time Compute* e Tomada de Decisão (Substituição de Cadeias Verbosas Caras)
- **Mecanismo Legado:** Rodar um LLM massivo por milhares de tokens de tentativa-e-erro (*Chain/Tree-of-Thought* bruto) até para rotas triviais.
- **Nossa Solução (`UniversalS1Agent` `< 1 ms` + `Batedores de Fronteira Ghost-Jump`):**
  - Intuição reflexiva ultrarrápida em CPU/GPU nativa (`act_fast <= 1 ms`) + frota paralela de batedores que sondam bifurcações 2–3 passos à frente (`> 2.000 FPS`), medem choque temporal ($R_0 - V(s_0)$), gravam cicatrizes de alerta no `SCache` e só escalam para o motor deliberativo S2 quando a margem epistêmica $p_1 - p_2$ cai.

### 8. Derivações Transversais Além de LLMs (Mídia, Robótica e Multi-Tenant Corporativo)
1. **Difusão de Imagem, Áudio e Vídeo (Nativo CUDA/MPS & WebGPU):** Cartuchos de `80 KB` de estilo, voz ou personagem substituindo LoRAs de `200 MB`, combinados com `SCache O(1)` temporal para geração de vídeo longo sem estourar VRAM.
2. **Robótica Embarcada e *Embodied AI* (`Motor Skill Scars`):** Tronco reflexivo S1 fixo em NPU embarcada acoplando cartuchos cinemáticos de `60 KB` (*"piso escorregadio"*, *"carga frágil"*) em milissegundos.
3. **Servidores Corporativos *Multi-Tenant Zero-Trust* (Nativo):** Uma única GPU serve milhares de clientes isolados injetando por requisição apenas o cartucho criptografado de `60 KB` daquele tenant em memória volátil, com zero contaminação cruzada nos pesos base.

---

## Parte III — Protocolo de Execução e Testes Unificados (`Nativo + WebGPU`)

Para garantir integridade absoluta entre os motores **Nativo** e **Web**, toda implementação segue este contrato:
1. **Módulo Canônico Nativo (`scar_cache/vq4_scar_mos.py`):**
   - Empacota qualquer `nn.Linear` em formato binário unificado (`codes_u32`, `scales_f16`, `lut256_f16`, `d_eq_f16`, `a_scar_int4`, `b_scar_int4`) via `VQ4ScarLinear` e `quantize_matrix_vq4_scar_2pass`, executando nativamente em `PyTorch (CPU / Metal MPS / CUDA)`.
2. **Kernels Canônicos WebGPU (`mos_scache_kernels.wgsl`):**
   - `vq4_equilibrate_fwht64`: Aplica `(x / d_fp16) * rademacher` + `FWHT-64` ortonormal em `6` estágios de memória compartilhada uma única vez por vetor de entrada.
   - `mos_fused_vq4_int4scar_gemv`: Carrega `LUT256` em `4 KB` de `var<workgroup>`, decodifica `16` pesos por `u32` coalescido e funde a cicatriz `INT4 B_scar` + `bias` sem materializar matrizes intermediárias.
3. **Suíte Obrigatória de Paridade Dupla (`tests/test_vq4_scar_native_and_webgpu.py`):**
   - Valida simultaneamente:
     - **(A) Motor Nativo (`PyTorch CPU/MPS`):** Orçamento exato de bytes, ganho do `2-Pass INT4-ASVD` sobre `1-Pass` e sobre escalar `2-bit` em ativações reais *held-out* (`PT/EN/ES/FR`).
     - **(B) Motor WebGPU Físico (`wgpu` WGSL 1.0):** Compilação pura WGSL 1.0 na GPU física e paridade numérica (`cosine_similarity > 0.999999`, erro máximo `< 1e-3`) contra o `VQ4ScarLinear` nativo em projeções reais do `Qwen2.5-3B`.
