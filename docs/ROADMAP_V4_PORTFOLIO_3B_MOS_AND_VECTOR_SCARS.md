# Roadmap Próximo: Portfólio WebGPU v4.0 com `Qwen2.5-3B MoS` Completo (`INT4 + ASVD Scars` Poliglota) + `HTTP 206` + `SCache O(1)`

> **Status:** Arquitetura WGSL 1.0 validada (`hidden_dim = 2048`, `9.472 B < 16.384 B` *workgroup shared memory*) e agendada como **próximo deploy de produção** após o ciclo atual de P&D em **Micro-Cicatrizes Vetoriais (`vec4` Codebooks)**.

---

## 1. Motivação Empírica (Descobertas Forenses em 4 Idiomas: `PT / EN / ES / FR`)

Nos experimentos forenses (`scripts/forensic_2bit_noise_anatomy.py` e `scripts/test_anisotropic_polyglot_3b.py`) executados sobre o `Qwen2.5-3B-Instruct`, mapeamos com precisão cirúrgica onde o ruído de quantização nasce e como ele afeta modelos poliglotas:

1. **Assimetria Extrema Peso vs. Ativação:**
   - Os pesos $W$ têm curtose quase Gaussiana (`3,38` a `5,44`), mas as ativações de entrada $X$ exibem *outliers* massivos de canal (`X_kurt = 10.556` na `L0.down_proj` até `80.252` na `L35.down_proj`, com pico `607x` acima do RMS).
2. **Localização por Profundidade e Módulo:**
   - Camadas iniciais (`L0..L5`) em baixa precisão colapsam todo o modelo (`KL = 11,29`, `Top-1 = 0%`) por corrupção dos *Attention Sinks* e efeito cascata não-linear nas 30 camadas seguintes.
   - As 3 projeções críticas onde o ruído explode são `mlp.gate_proj` (chave não-linear `SiLU`), `mlp.down_proj` e `self_attn.o_proj` (escrita direta no fluxo residual).
3. **Por que `Block-64 INT4` Puro NÃO Basta sem Cicatrizes `MoS (ASVD)` em Modelos Poliglotas:**
   - No `Regime 3` (`Block-64 INT4` puro sem cicatrizes `ASVD`, `1,53 GiB`), `PT`, `EN` e `ES` mantiveram alta fidelidade (`KL médio = 1,0583`, `Top-1 = 72,8%`), **porém em Francês (`FR`) por volta do token ~90 ocorreu vazamento de sub-token Chinês** (`"...interactions微观的量子力学..."`).
   - **Conclusão Arquitetural:** Os escudos de cicatriz **`ASVD` ($W \approx Q_{\text{INT4}} + B_{\text{scar}} A_{\text{scar}}$)** ponderados pelas ativações poliglotas (`+50 a +80 MiB`) são indispensáveis para blindar as fronteiras de decisão entre `PT`, `EN`, `ES` e `FR` contra o atrator pré-treinado Chinês/Inglês do tronco `Qwen2.5`.

---

## 2. Especificação de Deploy (Roadmap Próximo — Opção 2)

### 2.1 Motor de Inferência Nativo no Navegador (`Zero-Python / Zero-ONNX`)
- **Modelo Base:** `Qwen2.5-3B-Instruct` decomposto genuinamente em **Mixture of Scars (`MoS`)**:
  - Base `Block-64 INT4` (`34 B` por bloco de 64 pesos = `4,25 bpw`) + **Escudos `ASVD` Poliglotas** (`rank = 32..64` FP16 nas projeções `gate_proj`, `down_proj`, `o_proj` e camadas de fronteira), totalizando **`~1,60 GiB`** divididos em *shards* `<= 44 MiB`.
- **Shaders Computacionais:** 15 *compute shaders* **WGSL 1.0** (`mos_scache.wgsl` / `mos_scache_kernels.wgsl`) já compatíveis nativamente com `hidden_dim = 2048`, `num_heads = 16`, `num_kv_heads = 2`, `head_dim = 128`.
- **Memória Contextual `SCache O(1)`:**
  - Complexidade estritamente constante ($\Delta = +0\text{ B}$ em turnos infinitos).
  - `num_sink_tokens = 112` (blindando permanentemente os `81` tokens ChatML do *System Prompt* de 3ª pessoa e regra poliglota), `window_size = 256`, `num_slots = 32` (`288 <= 512` chaves totais em memória compartilhada WGSL).

### 2.2 Loja Poliglota de Cartuchos Conceituais (`GoldenSliceScarStoreV4` `HTTP 206`)
- **13 Cartuchos Universais Invariantes a Idioma** ancorados no espaço nulo de idioma $P_{\perp\text{lang}} = I - U_{\text{lang}} U_{\text{lang}}^\top$ na fatia dourada intermediária (`mid_layer_idx = 18` para `3B`, `140,00 KB` em `Block-64 INT4` por cartucho):
  1. `portfolio_agent_identity` — Identidade *System One* e saudação sem se passar por Julio Reis
  2. `portfolio_agent_capabilities` — Capacidades locais WebGPU, privacidade offline e atualizações via `HTTP 206`
  3. `portfolio_julio_bio_career` — Trajetória sênior de Julio Reis
  4. `portfolio_mos_scache_72b` — Prova `72B MoS` em `2,40 GB` VRAM + `SCache O(1)`
  5. `portfolio_browser_webgpu_wgsl` — Arquitetura dos 15 shaders WGSL + `HTTP 206` no navegador
  6. `portfolio_aura_system_one` — Motor `AURA` / `UniversalS1Agent` (`0,23 ms` em Deep RL)
  7. `portfolio_tricompass_scouts` — `Tri-Compass`, *Frontier Ghost-Jump Scouts* e *Soft Sinkhorn*
  8. `portfolio_cir_engine_503m` — `CIR-Engine 503M` treinado do zero
  9. `portfolio_devtools_fullstack` — Ecossistema `mddd-cli`, `MAD-cli`, `flutter_scene`, `GitTrack`, `Register-red`
  10. `portfolio_contact_hire` — Contato oficial, disponibilidade remota global e realocação
  11. `concept_landauer_principle` — Limite termodinâmico de Landauer ($k_B T \ln 2$)
  12. `concept_immerman_szelepcsenyi` — Teorema de Immerman–Szelepcsényi ($\text{NSPACE}(s(n)) = \text{co-NSPACE}(s(n))$)
  13. `bridge_landauer_immerman` — Ponte conjuntiva de consolidação de sono

### 2.3 Topologia de Hospedagem CDN (`GitHub Pages` + `Hugging Face Hub HTTP 206`)
- Como o GitHub Pages impõe teto de `1 GB` por artefato de deploy:
  - O frontend React/Vite + Worker TypeScript + Shaders WGSL permanecem em `byte-od.github.io` (`< 5 MB`).
  - Os *shards* `.bin` (`<= 44 MiB`) e o arquivo `capsules.scarstore` do `Qwen2.5-3B MoS` (`Config 3`: `1,593 GiB`, `KL médio = 0,6938`, `Top-1 = 74,5%`, `0%` de vazamento `FR -> ZH` verificado em `scripts/test_vector_scar_codebook_3b.py`) — ou da variante compacta de **Micro-Cicatrizes Vetoriais `vec4`** caso atinja paridade poliglota `< 1 GiB` — são servidos via repositório público no **Hugging Face Hub** (`CORS *` + suporte nativo a `HTTP 206 Partial Content`) com persistência local na `CacheStorage` do navegador (`'mos-webgpu-v4-assets-v1'`).

### 2.4 Trilha Anotada: `VQ4-Scar` Unificado (`Nativo + WebGPU`) — `< 1,00 GiB` e `~1,10 GiB`
> Ver documento arquitetural completo e Mapa das 8 Soluções em `docs/SCAR_NATIVE_ARCHITECTURE_AND_8_SOLUTIONS.md`.

- **Escopo Unificado (`Nativo + WebGPU`):** O mecanismo `VQ4-Scar` opera em **todo o ecossistema** (`PyTorch / Metal MPS / CUDA` via `scar_cache/vq4_scar_mos.py` E `WebGPU / WGSL 1.0` via `vq4_equilibrate_fwht64` + `mos_fused_vq4_int4scar_gemv`), exigindo sempre testes duplos de paridade (`tests/test_vq4_scar_native_and_webgpu.py`).
- **Resultados Verificados (`scripts/test_vector_scar_codebook_3b.py` & `scripts/test_sub1gb_vq4_compressed_scars_3b.py`):**
  1. **Cura Residual em 2 Passos com Cicatrizes `INT4` (`3,76x` menores):** Na `L18` *held-out* (`637` tokens inéditos nos 4 idiomas), `VQ4 + INT4-ASVD (r=64, 2-Pass)` (`6,48 MiB`/matriz) **superou** a cicatriz `FP16` (`7,65 MiB`/matriz), baixando o erro de `15,59% -> 15,33%` (`gate_proj`) e `25,81% -> 25,45%` (`down_proj`) — chegando a `14,82%` e `24,68%` em `r=96` (`6,69 MiB`).
  2. **Opção 1B (`0,992 GiB` / `1.015,98 MiB` — `< 1,00 GiB`!):** Com `embed_tokens` protegido em `INT4`, furou a barreira de `1,00 GiB` mantendo `PT` (`64,0%` Top-1), `ES` (`61,3%`), `FR` (`56,3%`) e `EN` (`55,6%`) 100% firmes no idioma correto do início ao fim (`KL médio = 1,5561`).
  3. **Opção 1A (`942,87 MiB` / `0,921 GiB`):** Mostrou o piso físico — comprimir `embed_tokens` em `INT3` faz `ES` escorregar para `PT` nos tokens finais, provando que `embed_tokens` deve permanecer em `Block-64 INT4`.
  4. **Evolução `~1,10 GiB` Anotada:** Reinvestir os `~135 MiB` poupados pelas cicatrizes `INT4` blindando `mlp.gate_proj` em `INT4` para eliminar deslizes lexicais finos mantendo `~1,08–1,12 GiB`.

---

## 3. Checklist de Aceite Pré-Deploy (`Zero Shortcuts` — Nativo & WebGPU)
- [ ] **Paridade Dupla `Nativo <-> WebGPU`:** `cosine_similarity > 0.99999` entre `VQ4ScarLinear` (PyTorch CPU/MPS/CUDA) e `mos_fused_vq4_int4scar_gemv` (WGSL 1.0 físico).
- [ ] **Auditoria Poliglota (`PT / EN / ES / FR`):** `52/52` consultas de cartucho + perguntas abertas de física, matemática e programação respondidas **100% no idioma do usuário**, com **zero vazamento de Chinês ou Inglês** em sequências longas (`>= 200` tokens).
- [ ] **Invariância Fora do Cone:** `10/10` controles gerais com `0` cartuchos disparados e desvio de logits `0.0000000000`.
- [ ] **Auditoria Física WebGPU / WGSL:** Verificação em GPU real (`wgpu` + navegador) atestando `SCache O(1)` com `Δ +0 B` ao longo de 5+ turnos consecutivos e guarda anti-repetição de 6-gramas ativa.
