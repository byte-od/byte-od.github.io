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
  - Os *shards* `.bin` (`<= 44 MiB`) e o arquivo `capsules.scarstore` do `Qwen2.5-3B MoS` (ou da variante compacta de **Micro-Cicatrizes Vetoriais `vec4`** caso atinja paridade poliglota `< 1 GiB`) são servidos via repositório público no **Hugging Face Hub** (`CORS *` + suporte nativo a `HTTP 206 Partial Content`) com persistência local na `CacheStorage` do navegador (`'mos-webgpu-v4-assets-v1'`).

---

## 3. Checklist de Aceite Pré-Deploy (`Zero Shortcuts`)
- [ ] **Auditoria Poliglota (`PT / EN / ES / FR`):** `52/52` consultas de cartucho + perguntas abertas de física, matemática e programação respondidas **100% no idioma do usuário**, com **zero vazamento de Chinês ou Inglês** em sequências longas (`>= 200` tokens).
- [ ] **Invariância Fora do Cone:** `10/10` controles gerais com `0` cartuchos disparados e desvio de logits `0.0000000000`.
- [ ] **Auditoria Física WebGPU / WGSL:** Verificação em GPU real (`wgpu` + navegador) atestando `SCache O(1)` com `Δ +0 B` ao longo de 5+ turnos consecutivos e guarda anti-repetição de 6-gramas ativa.
