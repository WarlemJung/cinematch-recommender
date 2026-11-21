# Relatório de Homologação Técnica Independente de Release — CineMatch v2.0.1

**Data da Homologação:** 20 de Novembro de 2026  
**Equipe Auditora:** Release Engineering, QA, MLOps, Machine Learning Engineering & DevOps (Auditoria Independente)  
**Repositório:** `https://github.com/WarlemJung/cinematch-recommender`  
**Versão Candidata:** `2.0.1`  
**Branch de Origem:** `main`  
**Branch de Trabalho:** `fix/cinematch-production-hardening`  
**Parecer Final:** **`APPROVED WITH OBSERVATIONS`**

---

## 1. Escopo e Metodologia

Esta homologação técnica independente foi conduzida de forma **estritamente não destrutiva** sobre o código-fonte real e o ambiente de execução local/containerizado do repositório `WarlemJung/cinematch-recommender`, sem alterar código-fonte, sem recalibrar hiperparâmetros para inflar métricas e sem executar `git commit`, `git push`, `git merge`, `git tag` ou deploy externo.

A metodologia de verificação abrangeu:
1. **Inspeção Direta de Controle de Versão e CI:** Consulta à árvore de trabalho Git local (`git status`, `git diff HEAD`, `git log`) e consulta direta à API REST pública do GitHub Actions (`/repos/WarlemJung/cinematch-recommender/actions/runs` e `/jobs`) para confrontar a configuração YAML com as execuções reais nos runners do GitHub.
2. **Reexecução Independente da Suíte de Qualidade:** Execução local de `ruff`, `mypy`, `pytest` com medição de cobertura de código e branches (`pytest-cov`), `pip check` e `pip-audit`.
3. **Auditoria Matemática e Reprodução Experimental do Benchmark:** Inspeção linha a linha de `scripts/train_models.py`, `src/cinematch/models/evaluation.py`, `src/cinematch/models/hybrid.py`, `src/cinematch/models/collaborative.py`, `src/cinematch/models/content_based.py` e `src/cinematch/models/demographic.py`, seguida de dupla execução independente com `seed=42` sobre o catálogo real (`45.430` filmes) e **análise estatística via bootstrap pareado por consulta** ($B = 10.000$ reamostragens, IC 95%, teste *t* pareado e teste dos postos sinalizados de Wilcoxon).
4. **Instrumentação de Memória no Kernel Linux do Container Docker:** Coleta de telemetria real de memória (`VmRSS`, `RssAnon`, `RssFile`, `Pss`, `Pss_Anon`, `Pss_File`, `Shared_Clean`, `Shared_Dirty`, `Private_Clean`, `Private_Dirty`) diretamente de `/proc/<pid>/status` e `/proc/<pid>/smaps_rollup` para o processo mestre e para os workers do Gunicorn, antes e após uma bateria de 50 requisições HTTP concorrentes dentro do container.
5. **Auditoria de Contrato REST e Segurança:** Comparação estrutural via `git diff HEAD` entre os contratos da `v2.0.0` e da `v2.0.1`, inspeção de cabeçalhos HTTP, CORS, tratamento de exceções, permissões do `Dockerfile` e varredura de segredos nos commits existentes e na árvore de trabalho.

### Resumo Consolidado de Classificação por Item

| # | Item Auditado | Classificação | Resumo Técnico |
|---|---|---|---|
| **1** | **Execução do CI no GitHub Actions (Matriz 3.11, 3.12, 3.13)** | **`NOT VERIFIED`** | Matriz configurada em `.github/workflows/ci.yml`, porém as alterações da `v2.0.1` ainda não foram comitadas/enviadas (`git push`) ao GitHub. O único run remoto (`44826189135`) rodou na `main` (`v2.0.0`) apenas em Python `3.13`. |
| **2** | **Qualidade de Código e Testes Locais (Ruff, Mypy, Pytest, Coverage)** | **`PASS`** | `Ruff`: 0 erros; `Mypy`: 0 erros em 33 arquivos; `Pytest`: 31/31 aprovados; Cobertura real: **86,42%** (sem omissão de `main.py` [97%] ou `wsgi.py` [100%]). |
| **3** | **Reprodutibilidade e Metodologia do Benchmark (`seed=42`)** | **`PASS WITH OBSERVATIONS`** | Resultados 100% reproduzidos até a 4ª casa decimal. `collection_name` em holdout estrito (`collection=0`), mas `genres_list` é compartilhado entre feature e filtro secundário do oráculo, e a calibração de pesos usou o mesmo subconjunto de 200 consultas. |
| **4** | **Análise Estatística (Bootstrap Pareado por Consulta, IC 95%)** | **`PASS WITH OBSERVATIONS`** | Ganhos do `Hybrid` sobre `Content-Based` são estatisticamente significativos (IC 95% exclui zero) em **Recall@10** ($p = 0.018$), **Diversity** ($p = 0.017$) e **Coverage**, mas o IC 95% cruza zero em **Precision@10**, **NDCG@10** e **MAP@10** para $Q = 200$. |
| **5** | **Auditoria de Memória e Docker (`gunicorn --preload` & CoW)** | **`PASS WITH OBSERVATIONS`** | `gunicorn --preload` operacional com UID `10001` e volume `:ro`. CoW compartilha ~542 MB no boot (`PSS` total de `595 MB`), mas 5 cópias em memória de `movies_df` e mutações de `ob_refcnt` do CPython geram ~113–197 MB de `Private_Dirty` por worker após carga (`790–920 MiB` no container). |
| **6** | **Compatibilidade de Contratos REST (`v2.0.0` vs. `v2.0.1`)** | **`PASS WITH OBSERVATIONS`** | 100% retrocompatível em rotas, métodos, parâmetros, códigos HTTP e tratamento de erros. Apenas adição compatível do campo `collaborative_distinct_users` em `/api/health` e leve defasagem documental no exemplo estático de `docs/API.md`. |
| **7** | **Segurança, Dependências e Configuração de Produção** | **`PASS`** | `pip check` e `pip-audit` com 0 vulnerabilidades; bloqueio de `debug=True` em produção; zero vazamento de stack trace; CORS restrito; usuário não-root `10001:10001`; nenhum segredo nos 2 commits do histórico Git ou na working tree. |

---

## 2. Commit e Branch Efetivamente Auditados

A inspeção do estado real do repositório Git revelou um ponto operacional crítico que deve ser registrado com exatidão:

- **Branch ativa local:** `fix/cinematch-production-hardening`
- **Commit `HEAD` atual da branch:** `f9292c58c33aead63c826445159a25bda973b746` (`feat(v2.0.0): evolve into CineMatch modular hybrid movie recommendation engine`)
- **Commit ancestral inicial:** `9406f9df9e5a03f2da457c205359cc1730f13a39`
- **Tags Git existentes:** Nenhuma (`git tag -l` vazio).
- **Branches remotas (`origin`):** Apenas `remotes/origin/main` apontando para `f9292c58c33aead63c826445159a25bda973b746`. A branch `fix/cinematch-production-hardening` **ainda não possui commit próprio nem foi enviada (`push`) ao remoto**.
- **Estado da Árvore de Trabalho (`Working Tree`):** Todas as correções da versão `2.0.1` encontram-se como **modificações locais não comitadas** sobre o commit `f9292c5`:
  - **27 arquivos rastreados modificados (`M`):** `.env.example`, `.github/workflows/ci.yml`, `CHANGELOG.md`, `Dockerfile`, `README.md`, `README_pt.md`, `docker-compose.yml`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `docs/MODEL_EVALUATION.md`, `pyproject.toml`, `requirements.txt`, `scripts/train_models.py`, `src/cinematch/__init__.py`, `src/cinematch/config.py`, `src/cinematch/data/loader.py`, `src/cinematch/main.py`, `src/cinematch/models/__init__.py`, `src/cinematch/models/collaborative.py`, `src/cinematch/models/evaluation.py`, `src/cinematch/models/hybrid.py`, `src/cinematch/services/recommendation_service.py`, `src/cinematch/utils/exceptions.py`, `tests/integration/test_api.py`, `tests/unit/test_data.py`, `tests/unit/test_evaluation_and_config.py`, `tests/unit/test_models.py`.
  - **2 caminhos não rastreados (`??`):** `src/cinematch/wsgi.py` e `docs/audits/`.

---

## 3. Evidências de CI (GitHub Actions)

**Classificação do Item:** **`NOT VERIFIED`** *(configuração YAML presente localmente, mas matriz não executada nos runners do GitHub Actions para a v2.0.1)*

### 3.1. Inspeção do Workflow Local (`.github/workflows/ci.yml`)
O arquivo `.github/workflows/ci.yml` na árvore de trabalho define o workflow `CineMatch CI` acionado em `push` para `[main, master, "fix/**"]` e `pull_request` para `[main, master]`, com a matriz:
```yaml
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
```
E executa sequencialmente: `py_compile`/`compileall`, `ruff check`, `mypy`, `pytest tests/unit -v`, `pytest tests/integration -v`, `pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80`, `pip check` e `pip-audit --skip-editable`.

### 3.2. Consulta Real à API do GitHub Actions
A consulta direta ao endpoint `https://api.github.com/repos/WarlemJung/cinematch-recommender/actions/runs` retornou **`total_count: 1`**. Não existem runs para a branch `fix/cinematch-production-hardening` porque as alterações da `v2.0.1` ainda não foram comitadas nem enviadas via `git push`.

| Campo | Evidência Registrada no GitHub Actions |
|---|---|
| **Workflow Run ID** | `44826189135` |
| **Workflow Run URL** | https://github.com/WarlemJung/cinematch-recommender/actions/runs/44826189135 |
| **Job ID** | `136599273710` |
| **Job URL** | https://github.com/WarlemJung/cinematch-recommender/actions/runs/44826189135/job/136599273710 |
| **Nome do Job** | `Lint, Typecheck, Test & Security Audit` |
| **Branch / Commit SHA** | `main` / `f9292c58c33aead63c826445159a25bda973b746` (`v2.0.0`) |
| **Runner / Ambiente** | `GitHub Actions 1000000003` (`ubuntu-latest`) |
| **Timestamp de Criação** | `2026-11-20T22:06:58Z` |
| **Status / Conclusão** | `completed` / `success` |
| **Versões de Python Executadas** | **Apenas Python `3.13`** (`Step 3: Set up Python 3.13 -> success`) |
| **Python 3.11 e Python 3.12** | **Não executados no GitHub Actions nem localmente** (no host Windows local estão instalados apenas Python `3.13.7` e `3.14`) |

> [!WARNING]
> Em conformidade com a diretriz de homologação (*"Não considerar a matriz aprovada apenas porque está configurada no YAML"*), a verificação de execução multi-versão em Python `3.11`, `3.12` e `3.13` no CI é classificada como **`NOT VERIFIED`**. O commit e o push da branch `fix/cinematch-production-hardening` são pré-requisitos obrigatórios antes do merge na `main` para que o GitHub Actions comprove a aprovação nos três ambientes.

---

## 4. Evidências dos Testes e Qualidade de Código

**Classificação do Item:** **`PASS`**

Todas as ferramentas de verificação estática e testes automatizados foram reexecutadas localmente sobre o código-fonte real da árvore de trabalho em **Windows 11 / Python 3.13.7** (`pytest-9.2.0`, `cov-7.1.0`):

1. **Ruff Linter (`python -m ruff check src tests scripts app.py recommender.py`):**
   - **Resultado:** `All checks passed!` (`0` erros, exit code `0`).
2. **Mypy Type Checker (`python -m mypy src tests scripts app.py recommender.py`):**
   - **Resultado:** `Success: no issues found in 33 source files` (exit code `0`).
3. **Suíte de Testes e Cobertura (`python -m pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80 -v`):**
   - **Resultado:** **`31 passed in 15.07s`** (`11` testes de integração em `tests/integration/test_api.py`, `8` testes unitários em `tests/unit/test_data.py`, `6` em `tests/unit/test_evaluation_and_config.py`, `6` em `tests/unit/test_models.py`).
   - **Exclusões de Cobertura (`omit`):** Confirmada a remoção de `omit = ["src/cinematch/main.py"]` em `pyproject.toml`. Nenhum arquivo de `src/cinematch` é omitido.
   - **Cobertura Total Medida (Statements + Branches):** **`86.42%`** (`1.604` statements, `151` missed; `472` branches, `117` parciais):

| Módulo | Statements | Missed | Branches | BrPart | Cobertura |
|---|---:|---:|---:|---:|---:|
| `src/cinematch/__init__.py` | 2 | 0 | 0 | 0 | **100%** |
| `src/cinematch/api/__init__.py` | 4 | 0 | 0 | 0 | **100%** |
| `src/cinematch/api/errors.py` | 33 | 4 | 4 | 0 | **89%** |
| `src/cinematch/api/routes.py` | 95 | 6 | 20 | 6 | **90%** |
| `src/cinematch/api/schemas.py` | 58 | 9 | 6 | 1 | **78%** |
| `src/cinematch/config.py` | 101 | 11 | 26 | 4 | **87%** |
| `src/cinematch/data/__init__.py` | 4 | 0 | 0 | 0 | **100%** |
| `src/cinematch/data/loader.py` | 82 | 17 | 20 | 5 | **76%** |
| `src/cinematch/data/preprocessing.py` | 141 | 17 | 58 | 17 | **83%** |
| `src/cinematch/data/validation.py` | 29 | 0 | 4 | 0 | **100%** |
| `src/cinematch/main.py` | 60 | 1 | 6 | 1 | **97%** |
| `src/cinematch/models/__init__.py` | 6 | 0 | 0 | 0 | **100%** |
| `src/cinematch/models/collaborative.py` | 140 | 11 | 34 | 9 | **87%** |
| `src/cinematch/models/content_based.py` | 170 | 19 | 70 | 23 | **82%** |
| `src/cinematch/models/demographic.py` | 105 | 17 | 42 | 13 | **78%** |
| `src/cinematch/models/evaluation.py` | 194 | 10 | 68 | 9 | **93%** |
| `src/cinematch/models/hybrid.py` | 153 | 14 | 54 | 11 | **88%** |
| `src/cinematch/services/__init__.py` | 3 | 0 | 0 | 0 | **100%** |
| `src/cinematch/services/movie_service.py` | 79 | 8 | 34 | 10 | **84%** |
| `src/cinematch/services/recommendation_service.py` | 67 | 5 | 12 | 5 | **87%** |
| `src/cinematch/utils/__init__.py` | 3 | 0 | 0 | 0 | **100%** |
| `src/cinematch/utils/exceptions.py` | 27 | 0 | 0 | 0 | **100%** |
| `src/cinematch/utils/logging.py` | 31 | 2 | 10 | 3 | **88%** |
| `src/cinematch/wsgi.py` | 17 | 0 | 4 | 0 | **100%** |
| **TOTAL** | **1604** | **151** | **472** | **117** | **86.42%** |

---

## 5. Resultados e Auditoria Metodológica do Benchmark

**Classificação do Item:** **`PASS WITH OBSERVATIONS`**

### 5.1. Verificação de Reprodutibilidade (`seed = 42`)
Duas execuções independentes da função `evaluate_catalog_models(movies_df, k=10, max_queries=200, random_state=42)` sobre `data/movies_metadata.csv` (`45.430` filmes válidos após limpeza de `45.466` linhas brutas) produziram dicionários de métricas **100% idênticos** (`deterministic_seed_42 = True`), confirmando os números reportados em `docs/MODEL_EVALUATION.md` e `FINAL_AUDIT_V2.0.1.md`:

| Modelo (`K = 10`, `Q = 200`, `seed = 42`) | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Intra-List Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` | `0.0171` | `0.9829` |
| **Content-Based Filtering** | `0.2045` | `0.4231` | `0.4244` | `0.3531` | `0.0350` | `0.2508` | `0.7492` |
| **Hybrid Recommendation (0.75/0.15/0.10)** | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | **`0.0354`** | `0.2485` | **`0.7515`** |

### 5.2. Reprodução da Análise de Sensibilidade dos Pesos do Hybrid
A execução de `evaluate_hybrid_weight_sensitivity(movies_df, k=10, max_queries=200, random_state=42)` reproduziu exatamente os valores documentados:

| Pesos Ativos $(\hat{w}_{\text{content}} / \hat{w}_{\text{demographic}})$ | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Intra-List Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid (0.90 / 0.10)** | `0.2055` | `0.4281` | `0.4263` | `0.3543` | `0.0352` | `0.2495` | `0.7505` |
| **Hybrid (0.85 / 0.15)** | `0.2065` | `0.4352` | `0.4286` | `0.3551` | `0.0354` | `0.2487` | `0.7513` |
| **Hybrid Padrão (0.8333 / 0.1667)** | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | `0.0354` | `0.2485` | `0.7515` |
| **Hybrid (0.75 / 0.25)** | `0.2065` | `0.4348` | `0.4267` | `0.3530` | **`0.0356`** | `0.2453` | `0.7547` |
| **Hybrid (0.65 / 0.35)** | `0.2005` | `0.4233` | `0.4222` | `0.3507` | `0.0355` | `0.2379` | `0.7621` |
| **Hybrid (0.50 / 0.50)** | `0.1895` | `0.3983` | `0.4009` | `0.3303` | `0.0326` | `0.2049` | **`0.7951`** |

### 5.3. Auditoria da Construção das Consultas, Oráculo, Métricas e Data Leakage
1. **Conjunto de Consultas e Relevance Oracle (`build_relevance_oracle`):**
   - Filtra filmes com `collection_name` não vazio pertencentes a coleções com $\ge 3$ filmes no catálogo (`min_group_size=3`).
   - Para cada filme $q$ da coleção, define $R(q)$ como os demais filmes $m \neq q$ da mesma coleção que compartilham pelo menos um gênero em `genres_list` (ou quando ambos não possuem gênero cadastrado).
   - No catálogo de `45.430` filmes, esse critério gera **`2.735` consultas elegíveis** com $R(q) \neq \emptyset$.
   - A amostragem seleciona `200` consultas sem reposição via `np.random.default_rng(42).choice(2735, size=200, replace=False)` ordenadas por índice.
2. **Avaliação de Data Leakage e Separação Treino/Avaliação:**
   - **Holdout de Feature Principal (Correto):** No benchmark, `ContentBasedRecommender` é ajustado com `ContentFeatureWeights(overview=2, genres=3, tagline=2, collection=0, production=1, language=1)`. O atributo `collection_name` é completamente omitido dos documentos TF-IDF (`collection=0`), e o próprio item consultado $q$ recebe score `-1.0` (autoexclusão estrita).
   - **Observação Metodológica 1 — Sobreposição Parcial do Filtro Secundário (`genres_list`):** Embora pertencer à mesma franquia (`collection_name`, oculto do modelo) seja a condição necessária primária do oráculo, a condição secundária ($G_m \cap G_q \neq \emptyset$) utiliza `genres_list`, que simultaneamente faz parte das features do TF-IDF com peso `genres=3`.
   - **Observação Metodológica 2 — Seleção de Hiperparâmetros no Conjunto Avaliado:** A escolha dos pesos padrão `(0.75, 0.15, 0.10)` e do limiar de *thematic relevance gating* (`0.25` em `np.clip(personalized_signal / 0.25, 0.0, 1.0)`) foi calibrada avaliando o mesmo subconjunto de `Q = 200` consultas (`seed = 42`), sem particionar as `2.735` consultas do oráculo em conjuntos disjuntos de validação (tuning) e teste (avaliação final).
3. **Implementação Matemática das Métricas:**
   - As funções `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `average_precision_at_k`, `catalog_coverage`, `intra_list_similarity` e `recommendation_diversity` em `src/cinematch/models/evaluation.py` estão implementadas sem erros matemáticos.

---

## 6. Análise Estatística (Bootstrap Pareado por Consulta)

**Classificação do Item:** **`PASS WITH OBSERVATIONS`**

Para verificar se a superioridade numérica do **Hybrid Recommendation (`0.75/0.15/0.10` $\rightarrow$ `0.8333/0.1667`)** sobre o **Content-Based Filtering** puro nas `Q = 200` consultas (`seed = 42`) possui significância estatística, foi executado um **bootstrap pareado por consulta** com **$B = 10.000$ reamostragens com reposição** (`seed = 42`), acompanhado do teste *t* de Student pareado e do teste não-paramétrico dos postos sinalizados de Wilcoxon para a diferença $\Delta(q) = \text{Métrica}_{\text{Hybrid}}(q) - \text{Métrica}_{\text{Content-Based}}(q)$:

| Métrica (`K = 10`, `Q = 200`) | Content-Based (Média) | Hybrid v2.0.1 (Média) | Diferença Pareada $\Delta$ (`Hybrid - CB`) | Ganho Relativo | Erro Padrão Bootstrap (SE) | IC 95% Bootstrap `[2.5%, 97.5%]` | Vitórias / Empates / Derrotas (`Hybrid` vs `CB`) | $p$-valor (Bootstrap Bilateral) | $p$-valor (Wilcoxon Pareado) | Significativo a 95% ($\alpha = 0.05$)? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Precision@10** | `0.204500` | `0.207000` | `+0.002500` | `+1.22%` | `0.001792` | `[-0.001000, +0.006000]` | `9 / 187 / 4` | `0.1140` | `0.0323` | **Não** (IC 95% inclui `0`; significativo apenas por Wilcoxon) |
| **Recall@10** | `0.423054` | `0.435846` | **`+0.012792`** | **`+3.02%`** | `0.005986` | **`[+0.001957, +0.025419]`** | `9 / 187 / 4` | **`0.0180`** | **`0.0295`** | **Sim (IC 95% > 0)** |
| **NDCG@10** | `0.424417` | `0.429045` | `+0.004628` | `+1.09%` | `0.002642` | `[-0.000375, +0.010032]` | `22 / 160 / 18` | `0.0702` | `0.2113` | **Não** (marginal a 90%, IC 95% cruza `0` em `-0.000375`) |
| **MAP@10** | `0.353080` | `0.355818` | `+0.002738` | `+0.78%` | `0.002046` | `[-0.001172, +0.006820]` | `22 / 160 / 18` | `0.1712` | `0.2560` | **Não** (IC 95% inclui `0`) |
| **Intra-List Diversity** | `0.749199` | `0.751504` | **`+0.002305`** | **`+0.31%`** | `0.000973` | **`[+0.000421, +0.004253]`** | `72 / 92 / 36` | **`0.0170`** | **`0.0003`** | **Sim (IC 95% > 0)** |
| **Catalog Coverage** | `0.034999` | `0.035395` | **`+0.000396`** | **`+1.13%`** | `0.000067` | **`[+0.000110, +0.000352]`** | *Métrica global* | **`< 0.001`** | — | **Sim (IC 95% > 0)** |

### Interpretação Técnica da Análise Estatística
1. **Ausência de Regressão e Eliminação da Degradação da v2.0.0:** Na versão `2.0.0`, o modelo Híbrido apresentava forte degradação estatística frente ao Content-Based (`NDCG@10 = 0.3292` vs. `0.4108`). Na `v2.0.1`, a normalização relativa à consulta e o gating temático eliminaram completamente essa regressão: a diferença média observada $\Delta$ é positiva em todas as 6 métricas.
2. **Ganhos Estatisticamente Significativos ($\alpha = 0.05$):** O modelo Híbrido supera o Content-Based com intervalo de confiança de 95% estritamente positivo em **`Recall@10`** (`+1.28 p.p.`, $+3.02\%$ relativo, IC 95% `[+0.0020, +0.0254]`), **`Intra-List Diversity`** (72 consultas com maior diversidade contra 36, IC 95% `[+0.0004, +0.0043]`) e **`Catalog Coverage`** (`1.608` vs. `1.590` itens únicos recomendados).
3. **Estabilidade Conservadora no Topo da Lista:** Como o gating temático ($\gamma(q, i) = \min(1, \tilde{S}_c / 0.25)$) preserva fortemente a ordenação do Content-Based para candidatos com alta similaridade lexical, em `187` das `200` consultas o conjunto de acertos no top-10 permanece inalterado (`9` vitórias e `4` derrotas em hits absolutos; `22` vitórias, `160` empates e `18` derrotas em ordenação `NDCG@10`/`MAP@10`). Por isso, os ganhos pontuais em `Precision@10` (`+0.0025`), `NDCG@10` (`+0.0046`) e `MAP@10` (`+0.0027`) possuem intervalos de confiança de 95% que ainda incluem o zero para $Q = 200$.

---

## 7. Evidências de Memória e Docker

**Classificação do Item:** **`PASS WITH OBSERVATIONS`**

### 7.1. Topologia de Processos no Container (`cinematch-recommender`)
O container `cinematch-recommender` (imagem `recommender-cinematch` baseada em `python:3.13-slim`, ID `fd8a71d436e1`) executa sob o usuário não privilegiado `uid=10001(cinematch) gid=10001(cinematch)` com o comando:
```text
/usr/local/bin/python3.13 /usr/local/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --graceful-timeout 30 --preload --access-logfile - --error-logfile - cinematch.wsgi:app
```
- **Processo Mestre:** `PID 1` (`PPid: 0`, `Threads: 2`).
- **Quantidade de Workers:** `2` workers síncronos (`PID 53` e `PID 54` na medição de boot recente; `PID 66` e `PID 67` na sessão anterior de 2 horas de uptime, ambos com `PPid: 1`, `Threads: 1`).

### 7.2. Medições Quantitativas de `RSS`, `PSS` e *Copy-on-Write* (`/proc/<pid>/smaps_rollup`)
Foram coletadas medições exatas do kernel Linux em três estágios:
1. **Estágio A (Recém-inicializado com `--preload`, antes de tráfego de aplicação):**
2. **Estágio B (Imediatamente após 50 requisições HTTP concorrentes distribuídas entre todos os endpoints REST):**
3. **Estágio C (Após ~2 horas de execução contínua e health checks periódicos):**

| Estágio / Processo | `VmRSS` / `Rss` (kB) | `RssAnon` (kB) | `RssFile` (kB) | `Pss` (kB) | `Pss_Anon` (kB) | `Shared_Dirty` (kB) | `Shared_Clean` (kB) | `Private_Dirty` (kB) | `docker stats` MemUsage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Boot Inicial — Master (`PID 1`)** | `601.844` | `545.924` | `55.552` | `236.909` | `185.847` | `540.296` | `7.092` | `5.720` | **`568,6 MiB`** |
| **A. Boot Inicial — Worker 1 (`PID 53`)** | `551.520` | `546.000` | `5.436` | `186.204` | `184.661` | `542.660` | `5.508` | `3.352` | *(Soma PSS: `595,0 MiB`)* |
| **A. Boot Inicial — Worker 2 (`PID 54`)** | `551.520` | `546.000` | `5.436` | `186.122` | `184.579` | `542.824` | `5.508` | `3.188` | *(Soma RSS: `1.664,9 MiB`)* |
| **B. Pós-50 Req. — Master (`PID 1`)** | `601.844` | `545.924` | `55.552` | `304.354` | `259.503` | `431.880` | `16.156` | `114.136` | **`790,0 MiB`** |
| **B. Pós-50 Req. — Worker 1 (`PID 53`)** | `561.832` | `546.000` | `15.804` | `268.038` | `263.027` | `424.832` | `15.808` | `121.184` | *(Soma PSS: `816,7 MiB`)* |
| **B. Pós-50 Req. — Worker 2 (`PID 54`)** | `561.824` | `546.000` | `15.676` | `263.878` | `258.875` | `433.136` | `15.808` | `112.880` | *(Soma RSS: `1.685,1 MiB`)* |
| **C. Uptime 2h — Master (`PID 1`)** | `603.748` | `545.344` | `58.112` | `353.790` | `306.961` | `362.180` | `16.940` | `183.236` | **`920,1 MiB`** |
| **C. Uptime 2h — Worker 1 (`PID 66`)** | `562.020` | `545.400` | `16.404` | `313.032` | `307.759` | `360.568` | `16.612` | `184.840` | *(Soma PSS: `962,9 MiB`)* |
| **C. Uptime 2h — Worker 2 (`PID 67`)** | `562.028` | `545.400` | `16.404` | `319.168` | `313.895` | `348.312` | `16.612` | `197.104` | *(Soma RSS: `1.687,3 MiB`)* |

### 7.3. Análise de Copy-on-Write, Comportamento Pós-Requisições e Duplicações Internas
1. **Efeito Comprovado do `--preload` e Copy-on-Write (CoW):**
   - Sem `--preload`, 3 processos independentes consumiriam a soma de seus `RSS` (`~1.665 MiB`).
   - Com `--preload`, imediatamente após o `fork()`, **`98,3%` da memória anônima de cada worker (`542,7 MB` de `546,0 MB`) é `Shared_Dirty`** com o processo mestre, resultando em apenas `~3,2 MB` de `Private_Dirty` por worker e um consumo físico real (`docker stats`) de **`568,6 MiB`** (economia imediata de `~1,09 GiB`).
2. **Estabilidade do Heap Anônimo (`RssAnon`) Pós-Requisições e Efeito de `ob_refcnt` no CPython:**
   - Após 50 requisições HTTP concorrentes (todas respondidas com `HTTP 200 OK`), o `RssAnon` de ambos os workers permaneceu **exatamente em `546.000 kB` (crescimento líquido de `0 kB` no heap anônimo)**, comprovando ausência de vazamento de memória (*memory leak*) durante a inferência esparsa sob demanda (`linear_kernel`).
   - Contudo, o `Private_Dirty` de cada worker subiu de `~3,3 MB` para `~113–121 MB` (e `~185–197 MB` após 2h), elevando o uso do container de `568,6 MiB` para `790,0 MiB` (estabilizando em `920,1 MiB`). Isso ocorre porque a leitura de objetos Python (`str`, `list`) dentro de DataFrames `pandas` incrementa e decrementa contadores de referência (`PyObject.ob_refcnt`) nos cabeçalhos dos objetos, sujando as páginas de 4 KB correspondentes e disparando *page faults* de Copy-on-Write no kernel Linux. Ainda assim, **`~348–433 MB` por worker permanecem compartilhados (`Shared_Dirty`)**.
3. **Identificação de Duplicação Interna de DataFrames no Processo Mestre (Observação de Engenharia):**
   - Embora a matriz TF-IDF esparsa (`45.430 x 25.000`, `float32`, `1.590.008` elementos não nulos) ocupe apenas **`12,304 MB`**, o processo mestre aloca **`~533 MiB` de `RssAnon`**.
   - A inspeção do código-fonte revelou a causa: durante a inicialização em `create_app()`, o DataFrame `movies_df` (`45.430` linhas contendo listas e strings Python) é **copiado 5 vezes em memória** dentro do mesmo processo:
     1. `LoadedDataset.movies_df` (`src/cinematch/data/loader.py:97`)
     2. `MovieService._df = movies_df.copy().reset_index(drop=True)` (`src/cinematch/services/movie_service.py:22`)
     3. `DemographicRecommender._df = movies_df.copy().reset_index(drop=True)` (`src/cinematch/models/demographic.py:117`)
     4. `ContentBasedRecommender._df = movies_df.copy().reset_index(drop=True)` (`src/cinematch/models/content_based.py:157`)
     5. `HybridRecommender._df = movies_df.copy().reset_index(drop=True)` (`src/cinematch/models/hybrid.py:107`)
   - Adicionalmente, `MovieService.search_movies()` executa `candidates = self._df.copy()` (`src/cinematch/services/movie_service.py:113`) a cada requisição em `/api/movies/search`. Embora o consumo total (`790–920 MiB`) seja seguro e inferior a 6% dos `16,5 GB` que uma matriz densa exigiria, eliminar as cópias redundantes de `movies_df` em versão futura (`v2.0.2`) reduzirá o footprint base do container em ~60%.

---

## 8. Matriz de Compatibilidade REST (`v2.0.0` vs. `v2.0.1`)

**Classificação do Item:** **`PASS WITH OBSERVATIONS`**

A comparação via `git diff HEAD` dos módulos `src/cinematch/api/routes.py`, `src/cinematch/api/schemas.py`, `src/cinematch/api/errors.py` e `src/cinematch/services/movie_service.py` confirmou **zero alterações incompatíveis (*breaking changes*)**.

| Rota & Método HTTP | Parâmetros & Schema Pydantic | Status HTTP (`v2.0.0` $\rightarrow$ `v2.0.1`) | Contrato JSON de Resposta | Comportamento em Estado `degraded` | Status de Compatibilidade |
|---|---|---|---|---|---|
| `GET /` | Nenhum | `200 OK` | HTML (`index.html`) + headers `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy` | `200 OK` (serve interface web estática) | **100% Compatível (`PASS`)** |
| `GET /api/health` | Nenhum | `200 OK` (saudável) / `503 Service Unavailable` (degradado) | `status`, `service`, `version` (`"2.0.1"`), `catalog_size`, `models`, `data_quality`. Adicionado campo aditivo `models.collaborative_distinct_users` (`int`). | Retorna `HTTP 503` com `{"status": "degraded", "service": "...", "error": "..."}` quando dataset falha ou `RecommendationService` falha na inicialização. | **Compatível / Aditivo (`PASS`)** |
| `GET /api/trending` | `TrendingQueryParams` (`top_n`, `page`, `page_size`, `genre`, `min_year`, `max_year`, `min_rating`) | `200 OK` / `400` / `503` | `items`, `total`, `page`, `page_size`, `total_pages`, `parameters` (`mean_vote_c`, `min_votes_m`, `quantile`, `qualifying_count`) | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/genres` | Nenhum | `200 OK` / `503` | `{"items": [{"name": str, "count": int}], "total": int}` | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/movies/search` | `SearchQueryParams` (`q`, `page`, `page_size`, `genre`, `min_year`, `max_year`, `min_rating`) | `200 OK` / `400` / `503` | `items`, `total`, `page`, `page_size`, `total_pages` | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/movies/<int:movie_id>` | Path param `movie_id` (`int`) | `200 OK` / `404` / `503` | Objeto completo do filme (`movie_id`, `title`, `release_year`, `genres`, `tagline`, `collection_name`, `production_companies`, `poster_url`, etc.) | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/recommend/<int:movie_id>` | Path `movie_id` + `RecommendationQueryParams` (`top_n`) | `200 OK` / `400` / `404` / `503` | `source_movie`, `algorithm: "content_based_tfidf"`, `total`, `items` (com `similarity_score`) | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/recommend` | Query `title` ou `movie_id` + `top_n` | `200 OK` / `400` / `404` / `503` | Idêntico a `/api/recommend/<movie_id>` | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **100% Compatível (`PASS`)** |
| `GET /api/recommendations/hybrid/<int:movie_id>` | Path `movie_id` + `RecommendationQueryParams` (`top_n`, `genre`, `min_year`, `max_year`, `min_rating`) | `200 OK` / `400` / `404` / `503` | `source_movie`, `algorithm: "hybrid_weighted_ensemble"`, `collaborative_active`, `active_weights`, `total`, `items` (`final_score`, `individual_scores`, `active_weights`, `explanation`) | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **Compatível (`PASS WITH OBSERVATIONS`¹)** |
| `GET /api/recommendations/hybrid` | Query `title` ou `movie_id` + `RecommendationQueryParams` | `200 OK` / `400` / `404` / `503` | Idêntico a `/api/recommendations/hybrid/<movie_id>` | Retorna `HTTP 503` com `error_code: "SYSTEM_NOT_INITIALIZED"` | **Compatível (`PASS WITH OBSERVATIONS`¹)** |
| **Tratamento Global de Erros** | `ErrorResponseSchema` (`errors.py`) | `400`, `404`, `422`, `500`, `503` | `{"error": str, "error_code": str, "status_code": int, "details": optional}` | Sem exposição de stack trace em nenhuma rota | **100% Compatível (`PASS`)** |

> ¹ **Observação sobre `/api/recommendations/hybrid` e `docs/API.md`:** O schema JSON é 100% idêntico ao da `v2.0.0`. Os valores numéricos de `active_weights` em modo cold-start passaram de `{"content": 0.7222, "demographic": 0.2778, "collaborative": 0.0}` (`v2.0.0`) para `{"content": 0.8333, "demographic": 0.1667, "collaborative": 0.0}` (`v2.0.1`) devido à nova configuração padrão `(0.75, 0.15, 0.10)`. Notou-se que o exemplo estático em `docs/API.md` (linhas 168–171 e 198–200) ainda exibe os números `0.7222` e `0.2778` da `v2.0.0`.

---

## 9. Auditoria de Segurança e Dependências

**Classificação do Item:** **`PASS`**

1. **Integridade de Pacotes (`python -m pip check`):**
   - **Resultado:** `No broken requirements found.` (exit code `0`).
2. **Auditoria de Vulnerabilidades (`python -m pip_audit -r requirements.txt`):**
   - **Resultado:** `No known vulnerabilities found` (exit code `0`).
3. **Verificação de Configuração de Produção (`src/cinematch/config.py` e `src/cinematch/main.py`):**
   - `AppConfig` define `env: str = "production"` e `debug: bool = False` por padrão.
   - `AppConfig.from_env()` força `debug_flag = False` quando `env_name == "production"`, e `AppConfig.validate()` (agora chamado explicitamente dentro de `create_app()`) lança `ConfigurationError("DEBUG mode cannot be enabled when ENV is 'production'.")`.
4. **Prevenção contra Exposição de Stack Traces (`src/cinematch/api/errors.py`):**
   - O handler `@app.errorhandler(Exception)` intercepta qualquer exceção não tratada, registra o traceback apenas no logger interno do servidor e retorna ao cliente exclusivamente `{"error": "An unexpected internal server error occurred.", "error_code": "INTERNAL_SERVER_ERROR", "status_code": 500}`.
5. **Restrição de CORS e Cabeçalhos HTTP (`src/cinematch/main.py`):**
   - `CORS(app, resources={r"/api/*": {"origins": app_config.cors_origins}})` restringe o compartilhamento de recursos às origens explicitamente configuradas (`http://localhost:5000,http://127.0.0.1:5000` por padrão), sem uso de curinga `*`.
   - Middleware `@app.after_request` injeta `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` e `Referrer-Policy: strict-origin-when-cross-origin`.
   - `sanitize_poster_path()` bloqueia URLs absolutas, payloads `javascript:` e extensões não permitidas via regex `^/[A-Za-z0-9_\-]+\.(jpg|jpeg|png|webp)$`.
6. **Inspeção de `Dockerfile`, `docker-compose.yml` e Permissões:**
   - Criação de grupo e usuário dedicado sem privilégios de root: `groupadd --gid 10001 cinematch && useradd --uid 10001 --gid cinematch ...`.
   - `USER cinematch` ativo (`uid=10001(cinematch) gid=10001(cinematch)` verificado em execução).
   - Dataset montado como somente-leitura (`./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`), impedindo adulteração do arquivo a partir do container.
   - `.dockerignore` exclui `.git`, `.env`, `data/*.csv`, `movies_metadata.csv`, `tests` e `docs`.
7. **Verificação de Segredos Versionados (Histórico Git Completo e Working Tree):**
   - Como o repositório possui exatamente 2 commits em todo o seu histórico (`9406f9df9e5a03f2da457c205359cc1730f13a39` e `f9292c58c33aead63c826445159a25bda973b746`), foi possível auditar **100% do histórico Git existente** (`git log -p --all`) além de todos os arquivos da árvore de trabalho atual.
   - Nenhum segredo, token, senha ou chave de API (`TMDB_API_KEY` está vazio em `.env.example`) foi encontrado nos commits ou nos arquivos locais. O arquivo `.env` está devidamente listado em `.gitignore` e `.dockerignore`.

---

## 10. Limitações Conhecidas

1. **Ausência de Dataset Público de Avaliações Individuais (`ratings.csv`) por Padrão:** O repositório distribui/utiliza `data/movies_metadata.csv`, que contém apenas estatísticas agregadas (`vote_average`, `vote_count`). Por princípio de integridade científica, o sistema não sintetiza avaliações falsas; assim, o `CollaborativeRecommender` opera em modo *Cold-Start Standby* na configuração padrão (sendo testado com matrizes de interação na suíte automatizada e ativado dinamicamente quando `CINEMATCH_RATINGS_PATH` aponta para um arquivo `ratings.csv` real).
2. **Incompatibilidade Nativa do Gunicorn com Windows:** O servidor `gunicorn` utiliza chamadas de sistema POSIX (`fcntl`, `pwd`, `grp`, `fork`), portanto só pode ser executado em Linux/macOS ou via container Docker. Em ambiente Windows nativo, a execução local utiliza o servidor de desenvolvimento Flask (`python -m cinematch.main`).
3. **Conflito de Porta Local `5000` no Host de Homologação:** Durante a auditoria, identificou-se um processo Python local (`PID 9376`) escutando em `0.0.0.0:5000` no host Windows, concorrendo com o port-forwarding do Docker na interface loopback do host. As medições de carga do container foram executadas diretamente dentro do namespace de rede do container para garantir isolamento total.
4. **Poder Estatístico para Diferenças Pequenas de Ranking em $Q = 200$:** Devido ao gating temático conservador, o modelo Híbrido preserva a lista do Content-Based na maioria das consultas (`160` empates em `200` consultas para `NDCG@10`), fazendo com que os ganhos médios em `Precision@10` (`+0.0025`), `NDCG@10` (`+0.0046`) e `MAP@10` (`+0.0027`) tenham intervalos de confiança de 95% que ainda cruzam zero em uma amostra de `200` consultas (embora `Recall@10`, `Diversity` e `Coverage` apresentem ganhos estatisticamente significativos a 95%).

---

## 11. Lista de Pendências

### Pendências Pré-Merge (Operacionais)
- [ ] **PEND-01 (Obrigatória antes do Merge):** Realizar o `git add` e `git commit` das alterações da versão `2.0.1` na branch `fix/cinematch-production-hardening` (incluindo `src/cinematch/wsgi.py` e os relatórios em `docs/audits/`), executar `git push -u origin fix/cinematch-production-hardening` e aguardar a conclusão com sucesso dos **3 jobs da matriz do GitHub Actions (`Python 3.11`, `Python 3.12` e `Python 3.13`)**.

### Recomendações Não-Bloqueantes (Melhoria Contínua / `v2.0.2`)
- [ ] **PEND-02 (Documentação):** Atualizar os dois blocos de exemplo JSON estático em `docs/API.md` (linhas 168–171 e 198–200) para refletir os novos pesos normalizados padrão da `v2.0.1` (`"content": 0.8333, "demographic": 0.1667` em vez de `0.7222 / 0.2778`).
- [ ] **PEND-03 (Otimização de Memória / CoW):** Compartilhar uma única referência imutável do DataFrame `movies_df` entre `LoadedDataset`, `MovieService`, `DemographicRecommender`, `ContentBasedRecommender` e `HybridRecommender` (eliminando as 5 cópias `.copy()` criadas na inicialização e o `.copy()` por requisição em `MovieService.search_movies()`), reduzindo o consumo base de RAM de `~568 MiB` (boot) / `~790–920 MiB` (pós-carga) para `< 250 MiB`.
- [ ] **PEND-04 (Metodologia de Benchmark):** Em futuros ajustes de hiperparâmetros do modelo Híbrido, dividir as `2.735` consultas elegíveis do oráculo em conjuntos separados de *validação* (para tuning de pesos e threshold de gating) e *teste* (para avaliação final), ou ampliar $Q$ para todas as `2.735` consultas elegíveis a fim de estreitar os intervalos de confiança de `NDCG@10` e `MAP@10`.

---

## 12. Parecer Final

### **`APPROVED WITH OBSERVATIONS`**

**Justificativa Técnica:**  
A versão candidata **CineMatch v2.0.1** corrige de forma efetiva, limpa e comprovada todas as deficiências arquiteturais e matemáticas apontadas na auditoria da versão `2.0.0`:
1. O servidor WSGI de produção (`gunicorn --preload cinematch.wsgi:app`) com singleton thread-safe por processo está funcional, validado em container Linux com usuário não privilegiado (`UID 10001`) e volume somente-leitura (`:ro`).
2. A integridade matemática do `CollaborativeRecommender` (deduplicação por `timestamp`, qualificação por `nunique` de usuários e truncamento de cossenos não positivos em `0.0`), do oráculo `build_relevance_oracle` (coleção em holdout + interseção de gênero) e da calibração/gating do `HybridRecommender` foi confirmada por inspeção de código e reprodução determinística (`seed = 42`).
3. A suíte de 31 testes automatizados atinge **86,42% de cobertura real** (sem exclusão de `main.py` ou `wsgi.py`), com aprovação total em `ruff`, `mypy`, `pip check` e `pip-audit`.
4. A compatibilidade retroativa dos contratos REST com a `v2.0.0` foi integralmente preservada.

O parecer é emitido como **`APPROVED WITH OBSERVATIONS`** (e não `APPROVED` incondicional) exclusivamente porque:
- As modificações da `v2.0.1` encontram-se ainda não comitadas na árvore de trabalho local, de modo que a execução real da matriz `Python 3.11 / 3.12 / 3.13` no GitHub Actions (`PEND-01`) precisa ser disparada via commit + push antes da efetivação do merge na branch `main`;
- A análise estatística por bootstrap pareado ($B = 10.000$) demonstra significância estatística a 95% de confiança para `Recall@10`, `Diversity` e `Coverage`, mas intervalos de confiança que ainda cruzam zero para `Precision@10`, `NDCG@10` e `MAP@10` em $Q = 200$; e
- O container apresenta duplicação interna de 5 instâncias de `movies_df` em memória (`568,6 MiB` no boot e `790–920 MiB` após CoW sob carga), comportamento seguro para operação mas documentado para otimização futura (`PEND-03`).
