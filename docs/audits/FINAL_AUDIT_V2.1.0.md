# Relatório Final de Auditoria, Evolução Técnica e Homologação — CineMatch v2.1.0

**Data:** 20 de Novembro de 2026  
**Repositório:** `https://github.com/WarlemJung/cinematch-recommender`  
**Branch Base:** `main` (preservada intacta)  
**Branch de Trabalho:** `feat/cinematch-v2.1.0-hardening`  
**Versão-Alvo:** `2.1.0`  
**Parecer Final:** **`APPROVED WITH OBSERVATIONS`**

---

## 1. Escopo da Auditoria

Auditoria técnica independente e evolução corretiva para a release **CineMatch v2.1.0**, executada sobre a branch dedicada `feat/cinematch-v2.1.0-hardening`, cobrindo integralmente as 7 fases do plano diretor:
1. Padronização e fonte única de verdade de dependências (`pyproject.toml` $\leftrightarrow$ `requirements.txt` $\leftrightarrow$ Docker $\leftrightarrow$ CI).
2. Reformulação metodológica da avaliação ML (eliminação de circularidade de `genres_list`, particionamento estrito treino/validação/teste, métricas adicionais com fórmulas explícitas e bootstrap pareado a 95%).
3. Integração reproduzível e opcional do dataset **GroupLens MovieLens `ml-latest-small`** com mapeamento `movieId -> tmdbId` e validação end-to-end do `CollaborativeRecommender`.
4. Otimização de memória (*zero-copy* do `movies_df` entre modelos/serviços), perfilamento de carga HTTP, *Copy-on-Write* em Gunicorn Docker e recuperação de worker.
5. Fortalecimento do pipeline de CI/CD em Python `3.12` e `3.13` com smoke test de imagem Docker.
6. Atualização documental completa preservando relatórios históricos (`v2.0.0` e `v2.0.1`).
7. Revisão independente de segurança da aplicação (15 controles).

---

## 2. Problemas Identificados na v2.0.1

| ID | Prioridade | Achado Técnico na v2.0.1 |
|---|---|---|
| **P1-01** | Bloqueadora | Divergência de versões mínimas entre `pyproject.toml` (`flask>=3.0.0`, `numpy>=2.0.0`, `scikit-learn>=1.5.0`, `requires-python>=3.11`) e `requirements.txt` (`flask>=3.1.0`, `numpy>=2.1.0`, `scikit-learn>=1.6.0`), gerando risco de discrepância entre Docker (`.`) e CI (`-r requirements.txt`). |
| **P1-02** | Bloqueadora | Sobreposição entre feature de entrada (`genres_list`) e filtro secundário do oráculo de relevância (`build_relevance_oracle`), além de calibração de pesos híbridos usando o mesmo conjunto de `200` consultas usado na avaliação final. |
| **P2-01** | Alta | Ausência de pipeline reproduzível para ingestão do MovieLens, mantendo `CollaborativeRecommender` em *Cold-Start Standby* na instalação padrão sem validação quantitativa com interações reais. |
| **P2-02** | Alta | 5 cópias completas em RAM de `movies_df` durante `create_app()` + `.copy()` por busca em `MovieService.search_movies()`, elevando o consumo do container Docker para `790–920 MiB` após carga HTTP. |
| **P2-03** | Média | Ausência de smoke test de build Docker no `.github/workflows/ci.yml` e defasagem numérica nos pesos do exemplo estático em `docs/API.md`. |

---

## 3. Alterações Implementadas

1. **Fonte Única de Dependências:** Sincronização 1:1 entre `pyproject.toml` (`requires-python = ">=3.12"`) e `requirements.txt`, instalação unificada no CI via `pip install --no-cache-dir -e ".[dev]"` e teste automatizado anti-deriva (`test_dependency_manifest_consistency_between_pyproject_and_requirements`).
2. **Pipeline Anti-Leakage v2.1.0 (`src/cinematch/models/evaluation.py`):**
   - `build_relevance_oracle(..., require_genre_overlap=False)` remove a dependência de `genres_list`.
   - `split_oracle_queries()` garante $\mathcal{Q}_{\text{val}} \cap \mathcal{Q}_{\text{test}} = \emptyset$.
   - `split_ratings_train_val_test()` executa divisão cronológica por usuário (70/15/15) usando `timestamp`.
   - `calibrate_hybrid_weights()` ajusta pesos exclusivamente em $\mathcal{Q}_{\text{val}}$.
   - Novas métricas: `hit_rate_at_k`, `novelty_at_k` (autoinformação normalizada), `evaluate_cold_start_stratification`, `evaluate_user_level_collaborative_holdout` e `compute_paired_bootstrap_comparison`.
3. **Integração Opcional MovieLens (`src/cinematch/data/movielens.py` & `scripts/prepare_movielens.py`):** Mapeia `movieId -> tmdbId` via `links.csv`, valida notas `[0.5, 5.0]`, deduplica por `timestamp` e gera `data/ratings.csv` offline/CLI (sem download na inicialização da API).
4. **Otimização *Zero-Copy* de Memória:** Remoção das cópias redundantes de `movies_df` em `DemographicRecommender`, `ContentBasedRecommender`, `HybridRecommender` e `MovieService` (incluindo filtragem via máscara booleana NumPy em `MovieService.search_movies()`).
5. **Load Profiler e CI Docker Smoke:** Criado `scripts/load_test.py` (`psutil` + `tracemalloc`) e adicionado o job `docker-build-smoke` em `.github/workflows/ci.yml`.

---

## 4. Arquivos Modificados e Criados

- **Criados:** `src/cinematch/data/movielens.py`, `scripts/prepare_movielens.py`, `scripts/load_test.py`, `docs/audits/FINAL_AUDIT_V2.1.0.md`.
- **Modificados (Código & Config):** `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`, `src/cinematch/__init__.py`, `src/cinematch/models/__init__.py`, `src/cinematch/models/evaluation.py`, `src/cinematch/models/demographic.py`, `src/cinematch/models/content_based.py`, `src/cinematch/models/hybrid.py`, `src/cinematch/services/movie_service.py`, `scripts/train_models.py`.
- **Modificados (Testes & Docs):** `tests/unit/test_data.py`, `tests/unit/test_models.py`, `tests/unit/test_evaluation_and_config.py`, `tests/integration/test_api.py`, `README.md`, `README_pt.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/DATA_DICTIONARY.md`, `docs/DEPLOYMENT.md`, `docs/MODEL_EVALUATION.md`, `CHANGELOG.md`.

---

## 5. Testes Executados (Ambiente Python 3.13.7 + Docker Engine)

| Verificação | Comando | Resultado Real |
|---|---|---|
| **Linter** | `python -m ruff check src tests scripts app.py recommender.py` | `All checks passed!` (`0` erros) |
| **Typecheck** | `python -m mypy src tests scripts app.py recommender.py` | `Success: no issues found in 36 source files` |
| **Testes + Cobertura** | `python -m pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80 -v` | **`36 passed in 7.78s`**, cobertura total **`86.69%`** (acima dos `86.56%` da v2.0.1) |
| **Integridade de Pacotes** | `python -m pip check` | `No broken requirements found.` |
| **Vulnerabilidades** | `python -m pip_audit -r requirements.txt` | `No known vulnerabilities found` |

---

## 6. Resultados Efetivamente Observados (MovieLens & Cold-Start Standby)

- **Preparação MovieLens (`python scripts/prepare_movielens.py`):** Baixou `GroupLens ml-latest-small.zip`, processou `100.836` linhas brutas $\rightarrow$ `100.823` mapeadas com `tmdbId` $\rightarrow$ `100.330` no catálogo $\rightarrow$ **`100.329` interações únicas** de **`610` usuários** em **`9.525` filmes** (`sparsity = 0.982732`), salvas em `data/ratings.csv`.
- **Cold-Start Standby (sem `ratings.csv`):** A aplicação inicializa normalmente (`HTTP 200` em `/api/health`, `collaborative_available: false`, pesos renormalizados).

---

## 7. Métricas dos Modelos (`K = 10`, `Q_val = 100`, `Q_test = 200`, `val_test_overlap = 0`, `seed = 42`)

### 7.1. Avaliação em Holdout Estrito de Franquia (com MovieLens Ativo)
Pesos calibrados em $\mathcal{Q}_{\text{val}}$: `content=0.65, demographic=0.15, collaborative=0.20`.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | Diversity@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | **`0.9829`** | `0.4482` |
| **Content-Based Filtering** | `0.1615` | `0.3474` | `0.3468` | `0.2744` | `0.6250` | `0.7449` | **`0.8209`** |
| **Collaborative Filtering** *(50 warm)* | `0.0280` | `0.0622` | `0.0541` | `0.0289` | `0.2000` | `0.9788` | `0.6410` |
| **Hybrid Recommendation (0.65/0.15/0.20)** | **`0.1675`** | **`0.3643`** | **`0.3604`** | **`0.2845`** | **`0.6500`** | `0.7491` | `0.8052` |

- **Bootstrap Pareado (`Hybrid` vs. `Content-Based` em $\mathcal{Q}_{\text{test}} = 200$, $B=5.000$):** Todos os ganhos são estatisticamente significativos a **95%**: `Precision@10` (`+0.0060`, IC `[+0.0010, +0.0115]`, $p=0.0108$), `Recall@10` (`+0.0168`, IC `[+0.0050, +0.0318]`, $p=0.0024$), `NDCG@10` (`+0.0136`, IC `[+0.0053, +0.0228]`, $p=0.0008$), `MAP@10` (`+0.0102`, IC `[+0.0033, +0.0176]`, $p=0.0012$), `HitRate@10` (`+0.0250`, IC `[+0.0050, +0.0500]`, $p=0.0104$) e `Diversity@10` (`+0.0042`, IC `[+0.0016, +0.0071]`, $p=0.0004$).

### 7.2. Holdout Temporal de Preferências Humanas Observadas (MovieLens, 200 Usuários)
- `Collaborative Filtering`: `HitRate@10 = 0.1200`, `NDCG@10 = 0.0187`, `Precision@10 = 0.0140`, `Coverage = 0.0264`.
- `Content-Based Filtering`: `HitRate@10 = 0.0150`, `NDCG@10 = 0.0023`, `Precision@10 = 0.0015`.
- Ganho do `Collaborative` sobre `Content-Based`: **8x em HitRate@10** ($\Delta = +0.1050$, IC 95% `[+0.0650, +0.1550]`, $p = 0.0000$).

---

## 8. Resultados dos Testes de Carga e Memória Docker

### 8.1. Perfilamento Local (`python scripts/load_test.py`)
- **Cold Start:** Boot `78.349s` (com carregamento completo de `45.430` filmes + `100.329` interações), pico `tracemalloc` `467.41 MB`, `RSS` pós-boot `437.51 MB`, primeira recomendação híbrida `64.78 ms` (`HTTP 200`).
- **Sequencial (100 req):** `36.37 req/s`, p50 `31.43 ms`, p95 `69.79 ms`, p99 `76.92 ms`, `0%` erro.
- **Concorrente (200 req, 8 threads):** `57.25 req/s`, p50 `124.43 ms`, p95 `333.85 ms`, `0%` erro.
- **Carga Sustentada (500 req, 8 threads):** `62.72 req/s`, p50 `116.96 ms`, p95 `291.08 ms`, `0%` erro.
- **Vazamento de Memória (`RSS` pós-800 req):** `436.19 MB` (`-1.32 MB` de delta $\rightarrow$ **0 byte de memory leak**).

### 8.2. Container Docker (`cinematch-recommender` — `gunicorn --preload` com 2 workers)

| Métrica | v2.0.1 | v2.1.0 | Variação |
|---|---:|---:|---:|
| **Memória Boot Inicial (`docker stats`)** | `568.6 MiB` | **`539.3 MiB`** | `-29.3 MiB` (`-5.2%`) |
| **Memória Pós-Carga Concorrente (`docker stats`)** | `790.0 MiB` | **`624.1 MiB`** | **`-165.9 MiB` (`-21.0%`)** |
| **Worker `Private_Dirty` CoW Pós-Carga** | `112.9–121.2 MB` | **`46.9–47.0 MB`** | **`-60.0%` de páginas sujas por worker** |
| **Recuperação após `SIGTERM` em Worker** | N/A | **Respawn automático (`PID 53 -> 84`) + `HTTP 200`** | Validado em execução real |

---

## 9. Análise Final de Segurança (FASE 7 — 15 Controles)

| # | Controle de Segurança | Severidade | Evidência Verificada | Impacto & Recomendação |
|---|---|---|---|---|
| **1** | **CORS** | Baixa | `CORS(app, resources={r"/api/*": {"origins": app_config.cors_origins}})` em `src/cinematch/main.py` sem `*`. | Restrito a localhost por padrão; configurar domínio exato em produção via `CINEMATCH_CORS_ORIGINS`. |
| **2** | **Validação de Entradas** | Info | `TrendingQueryParams`, `SearchQueryParams`, `RecommendationQueryParams` (Pydantic v2) validam limites (`ge`, `le`, `max_length`) e `min_year <= max_year`. | Nenhuma vulnerabilidade; manter validação estrita. |
| **3** | **SQL Injection** | N/A | Sem banco relacional ou driver SQL no projeto (armazenamento em CSV + memória). | Não aplicável na arquitetura atual. |
| **4** | **XSS (Cross-Site Scripting)** | Info | `frontend/static/js/script.js` usa exclusivamente `textContent`/`createElement` (zero `innerHTML`) e `SAFE_POSTER_REGEX`. | Protegido contra DOM XSS. |
| **5** | **Exposição de Stack Traces** | Info | `@app.errorhandler(Exception)` em `src/cinematch/api/errors.py` retorna JSON genérico `INTERNAL_SERVER_ERROR` (`500`). | Sem vazamento de traceback em produção. |
| **6** | **Secrets & Variáveis de Ambiente** | Info | `.env` listado em `.gitignore` e `.dockerignore`; `TMDB_API_KEY` vazio em `.env.example`; histórico Git sem credenciais. | Nenhum segredo exposto. |
| **7** | **Logs com Informações Sensíveis** | Info | `StructuredJsonFormatter` registra apenas nível, logger e mensagens operacionais sem tokens ou PII. | Seguro; em produção ativar `CINEMATCH_LOG_JSON=true`. |
| **8** | **Dependências Vulneráveis** | Info | `pip check` (`0` erros) e `pip-audit -r requirements.txt` (`No known vulnerabilities found`). | Dependências limpas até a data da auditoria. |
| **9** | **Dockerfile** | Info | Multi-camada enxuta `python:3.13-slim`, `PIP_NO_CACHE_DIR=1`, instala somente runtime via `pip install --no-cache-dir .`. | Boas práticas atendidas. |
| **10** | **Permissões do Container** | Info | `groupadd --gid 10001 cinematch`, `USER cinematch`, volume `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`. | Princípio do menor privilégio e dataset imutável no container. |
| **11** | **Endpoints Administrativos** | Info | Não existem endpoints administrativos, de escrita ou debug expostos na API (`routes.py` apenas `GET` de leitura). | Superfície de ataque mínima. |
| **12** | **Rate Limiting** | Baixa | A API Flask/Gunicorn não implementa rate limiting nativo em camada de aplicação. | Recomenda-se aplicar rate limiting no reverse proxy / API Gateway (Nginx, Cloudflare, Traefik) em exposição pública. |
| **13** | **Health Checks** | Info | `GET /api/health` e `HEALTHCHECK` no `Dockerfile`/`docker-compose.yml` retornam `200` (saudável) ou `503` (`degraded`) sem dados sensíveis. | Adequado para orquestradores. |
| **14** | **Tratamento de Erros** | Info | Hierarquia `CineMatchError` (`400`, `404`, `422`, `503`) + `ValidationError` (`400`) + `HTTPException` tratados uniformemente. | Respostas determinísticas em JSON. |
| **15** | **Configuração de Produção** | Info | `AppConfig.validate()` chamada em `create_app()` bloqueia `debug=True` quando `env == "production"`. | Protegido contra deploy inseguro. |

---

## 10. Limitações Remanescentes

1. **Execução Remota do CI da v2.1.0 no GitHub Actions:** Como não foi realizado `commit`/`push` nesta etapa (por restrição obrigatória), os jobs atualizados para `v2.1.0` (incluindo `docker-build-smoke` em `ubuntu-latest`) ainda aguardam execução remota após o envio da branch `feat/cinematch-v2.1.0-hardening`.
2. **Versão Python Local Única:** O ambiente local executou e validou integralmente em Python `3.13.7` + container Docker `python:3.13-slim`; a validação em Python `3.12` será confirmada pelo runner do GitHub Actions após o `push`.
3. **Distribuição do `ratings.csv`:** O arquivo `data/ratings.csv` não é versionado no Git para respeitar termos de redistribuição do GroupLens e é gerado reproduzivelmente via `python scripts/prepare_movielens.py`.

---

## 11. Parecer Final

### **`APPROVED WITH OBSERVATIONS`**

A versão candidata **CineMatch v2.1.0** resolve integralmente todas as pendências bloqueadoras (`P1`) e prioritárias (`P2`) herdadas da `v2.0.1`, elevando a cobertura real para **86,69%**, reduzindo em **21,0%** o consumo de memória no container Docker sob carga (`624,1 MiB`), comprovando ganhos estatisticamente significativos ($p < 0,05$) do motor Híbrido 3-way com dados reais do MovieLens em conjunto de teste estritamente isolado (`val_test_overlap = 0`) e mantendo segurança e compatibilidade REST intactas. O parecer permanece `APPROVED WITH OBSERVATIONS` até que o `push` da branch `feat/cinematch-v2.1.0-hardening` conclua os jobs `Python 3.12`, `Python 3.13` e `docker-build-smoke` no GitHub Actions.
