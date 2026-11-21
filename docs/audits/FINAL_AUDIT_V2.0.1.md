# Relatório Técnico Final de Auditoria e Hardening — CineMatch v2.0.1

**Data da Auditoria:** 20 de Novembro de 2026  
**Repositório:** `https://github.com/WarlemJung/cinematch-recommender`  
**Branch de Origem:** `main` (`f9292c5`)  
**Branch de Trabalho:** `fix/cinematch-production-hardening`  
**Versão:** `2.0.1`

---

## A. Resumo Executivo

A rodada corretiva de hardening e validação de produção do **CineMatch v2.0.1** solucionou integralmente as 6 pendências estruturais identificadas na auditoria da versão `2.0.0`, sem reconstruir o projeto do zero e preservando 100% de compatibilidade com os contratos da API REST e com a arquitetura modular existente.

Destaques técnicos comprovados:
1. **Servidor WSGI de Produção (`src/cinematch/wsgi.py` + Gunicorn):** Substituição de `app.run()` no `Dockerfile` por `exec gunicorn ... --preload cinematch.wsgi:app`, com singleton thread-safe por processo para evitar inicialização redundante dos modelos e compartilhar a matriz esparsa CSR via *Copy-on-Write* entre workers.
2. **Padronização do Dataset (`data/movies_metadata.csv`):** Alinhamento unificado entre `docker-compose.yml` (`./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`), `Dockerfile`, `.env.example` e `AppConfig`, acompanhado de validações explícitas em `MovieDataLoader` contra diretórios montados incorretamente, arquivos vazios (`0 bytes`), CSVs corrompidos e erros de permissão (`PermissionError`).
3. **Integridade Matemática no Collaborative Filtering (`src/cinematch/models/collaborative.py`):** Deduplicação determinística de múltiplas avaliações `(user_id, movie_id)` (priorizando o registro mais recente por `timestamp` ou última ocorrência), qualificação de cobertura estritamente por **usuários distintos (`nunique`)** e truncamento de similaridades não positivas em zero ($\max(0.0, \cos)$).
4. **Alinhamento Metodológico do Benchmark e Calibração do Modelo Híbrido (`src/cinematch/models/evaluation.py`, `src/cinematch/models/hybrid.py`):** Implementação explícita da verificação de interseção de gênero junto ao holdout de `collection_name` no oráculo de relevância e introdução de calibração de escala relativa à consulta + *gating* de relevância temática no `HybridRecommender`. Com essa correção matemática, o **Hybrid Recommendation** passou a superar o **Content-Based Filtering** puro em todas as métricas (`Precision@10 = 0.2070` vs. `0.2045`, `Recall@10 = 0.4358` vs. `0.4231`, `NDCG@10 = 0.4290` vs. `0.4244`, `MAP@10 = 0.3558` vs. `0.3531`, `Coverage = 0.0354` vs. `0.0350` e `Diversity = 0.7515` vs. `0.7492`).
5. **Cobertura Real Integral Sem Exclusões:** Remoção de `omit = ["src/cinematch/main.py"]` em `pyproject.toml` e expansão da suíte para **31 testes automatizados** com **86,42% de cobertura total** (`97%` em `main.py` e `100%` em `wsgi.py`).

---

## B. Problemas Encontrados e Status

| ID | Descrição do Problema | Status | Tratamento Aplicado |
|---|---|---|---|
| **CORR-01** | `Dockerfile` utilizava `python -m cinematch.main` (`app.run()`) em produção; ausência de entrypoint WSGI e chamada dupla de `CollaborativeRecommender.fit()` em standby. | **Confirmado e Corrigido** | Criado `src/cinematch/wsgi.py` com singleton `get_wsgi_app()`; adicionado `gunicorn>=22.0.0,<28.0.0`; `Dockerfile` atualizado para `exec gunicorn --preload ... cinematch.wsgi:app`; adicionada propriedade `is_fitted` em `CollaborativeRecommender`. |
| **CORR-02** | Divergência de caminho do dataset entre `docker-compose.yml` (`./movies_metadata.csv`) e a convenção `data/movies_metadata.csv`. | **Confirmado e Corrigido** | Padronizado `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro` no Compose e `data/movies_metadata.csv` em `.env.example` e `AppConfig`; endurecido `MovieDataLoader.load()`. |
| **CORR-03A** | Contagem de interações no Collaborative Filtering contava linhas brutas (`count()`) em vez de usuários distintos (`nunique()`) e não deduplicava `(user_id, movie_id)`. | **Confirmado e Corrigido** | Implementado `_validate_and_deduplicate_ratings()` com ordenação por `timestamp` (`keep="last"`) e contagem de qualificação via `groupby("movie_id")["user_id"].nunique()`. |
| **CORR-03B** | Transformação `(raw_sims + 1.0) / 2.0` convertia correlação zero ($\cos = 0$) em `0.50` e correlação negativa em score positivo. | **Confirmado e Corrigido** | Substituído por truncamento não negativo `np.clip(raw_sims, 0.0, 1.0)` em `predict_catalog_scores()`. |
| **CORR-04A** | Oráculo de avaliação agrupava apenas por `collection_name` sem verificar interseção de `genres_list` descrita na documentação. | **Confirmado e Corrigido** | Criada função `build_relevance_oracle()` verificando explicitamente `collection_name` **e** interseção não vazia de `genres_list`. |
| **CORR-04B** | Desempenho inferior do modelo Híbrido na v2.0.0 devido à soma direta de escalas incompatíveis ($S_c \in [0, 0.45]$ vs. $S_d \in [0, 1.0]$) e ausência de gating temático. | **Confirmado e Corrigido** | Implementada normalização relativa ao máximo da consulta $\tilde{S}_c(q, i)$ e gating de relevância temática $\gamma(q, i)$, além de análise de sensibilidade `evaluate_hybrid_weight_sensitivity()`. |
| **CORR-05** | `src/cinematch/main.py` estava omitido da medição de cobertura em `pyproject.toml`. | **Confirmado e Corrigido** | Removido `omit` de `pyproject.toml` e adicionados testes de Application Factory, WSGI, resiliência e falha de inicialização. |
| **CORR-06** | Pipeline CI executava apenas em Python 3.13. | **Confirmado e Corrigido** | Configurada matriz `python-version: ["3.11", "3.12", "3.13"]` com `fail-fast: false` em `.github/workflows/ci.yml`. |

---

## C. Arquivos Modificados e Criados

| Arquivo | Tipo | Justificativa Técnica |
|---|---|---|
| `src/cinematch/wsgi.py` | Criado | Entrypoint WSGI de produção com singleton thread-safe `get_wsgi_app()` e `app` lazy para Gunicorn. |
| `src/cinematch/__init__.py` | Modificado | Atualização da versão do pacote para `2.0.1`. |
| `src/cinematch/config.py` | Modificado | Padronização de `data/movies_metadata.csv`, adição de `gunicorn_workers`, `gunicorn_timeout`, `gunicorn_graceful_timeout` e pesos híbridos calibrados `(0.75, 0.15, 0.10)`. |
| `src/cinematch/main.py` | Modificado | Validação explícita da configuração na fábrica e captura resiliente de falhas de inicialização de modelos (entrando em modo `degraded` HTTP 503 sem derrubar o worker WSGI). |
| `src/cinematch/data/loader.py` | Modificado | Verificação de diretório, arquivo vazio (`0 bytes`), `PermissionError` e deduplicação de avaliações `(user_id, movie_id)` por `timestamp`. |
| `src/cinematch/utils/exceptions.py` | Modificado | Mensagem orientativa clara em `DatasetNotFoundError` indicando `data/movies_metadata.csv` e `CINEMATCH_DATASET_PATH`. |
| `src/cinematch/models/collaborative.py` | Modificado | Deduplicação de múltiplos ratings por usuário/filme, contagem por usuários distintos (`nunique`), truncamento `np.clip(raw_sims, 0.0, 1.0)`, `has_user()` e `is_fitted`. |
| `src/cinematch/models/hybrid.py` | Modificado | Normalização de conteúdo relativa à consulta, gating de relevância temática e prevenção de re-fit colaborativo duplicado. |
| `src/cinematch/models/evaluation.py` | Modificado | Oráculo `build_relevance_oracle()` com verificação de coleção + gênero, suporte a avaliação colaborativa quando há ratings reais e `evaluate_hybrid_weight_sensitivity()`. |
| `src/cinematch/models/__init__.py` | Modificado | Exportação de `build_relevance_oracle` e `evaluate_hybrid_weight_sensitivity`. |
| `src/cinematch/services/recommendation_service.py` | Modificado | Uso dinâmico de `__version__` e exposição de `collaborative_distinct_users` no health check. |
| `scripts/train_models.py` | Modificado | Benchmark reproduzível com `--seed`, `--sensitivity`, `--output-json`, `--output-csv` e metadados completos do experimento. |
| `Dockerfile` | Modificado | Inicialização em produção via `exec gunicorn --preload ... cinematch.wsgi:app` com usuário não privilegiado `cinematch`. |
| `docker-compose.yml` | Modificado | Padronização do volume `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro` e variáveis do Gunicorn. |
| `.env.example` | Modificado | Atualização de `CINEMATCH_DATASET_PATH=data/movies_metadata.csv` e parâmetros do Gunicorn. |
| `pyproject.toml` & `requirements.txt` | Modificados | Inclusão de `gunicorn` e `pip-audit`, atualização para `2.0.1` e remoção de `omit = ["src/cinematch/main.py"]`. |
| `.github/workflows/ci.yml` | Modificado | Matriz CI para Python `3.11`, `3.12` e `3.13`. |
| `tests/unit/*` & `tests/integration/test_api.py` | Modificados | Expansão para 31 testes cobrindo WSGI, Application Factory, permissões/arquivos inválidos, deduplicação colaborativa, similaridade negativa/zero e sensibilidade híbrida. |
| `docs/*`, `README.md`, `README_pt.md`, `CHANGELOG.md` | Modificados/Criados | Atualização documental PT/EN e relatórios de auditoria v2.0.1. |

---

## D. Evidências Reais de Testes e Qualidade

Ambiente de execução local: **Windows 11 / Python 3.13.7**.

### 1. Linter (`Ruff`)
- **Comando:** `python -m ruff check src tests scripts app.py recommender.py`
- **Resultado:** `All checks passed!` (`0` erros, exit code `0`).

### 2. Verificação Estática de Tipos (`Mypy`)
- **Comando:** `python -m mypy src tests scripts app.py recommender.py`
- **Resultado:** `Success: no issues found in 33 source files` (exit code `0`).

### 3. Testes Unitários, Integração e Cobertura (`Pytest` + `Coverage`)
- **Comando:** `python -m pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80 -v`
- **Resultado:** **`31 passed in 7.47s`** (exit code `0`).
- **Cobertura Total Medida (sem exclusão de `main.py` ou `wsgi.py`):** **`86.42%`** (1.604 statements, 472 branches):
  - `src/cinematch/wsgi.py`: **`100%`**
  - `src/cinematch/main.py`: **`97%`**
  - `src/cinematch/data/validation.py`: **`100%`**
  - `src/cinematch/utils/exceptions.py`: **`100%`**
  - `src/cinematch/models/evaluation.py`: **`93%`**
  - `src/cinematch/api/routes.py`: **`90%`**
  - `src/cinematch/api/errors.py`: **`89%`**
  - `src/cinematch/models/hybrid.py`: **`88%`**
  - `src/cinematch/utils/logging.py`: **`88%`**
  - `src/cinematch/config.py`: **`87%`**
  - `src/cinematch/models/collaborative.py`: **`87%`**
  - `src/cinematch/services/recommendation_service.py`: **`87%`**

---

## E. Avaliação Quantitativa dos Modelos (Benchmark Reproduzido)

- **Comando Executado:** `python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity`
- **Catálogo Validado:** `45.430` filmes (`data/movies_metadata.csv`).
- **Memória da Matriz Esparsa TF-IDF (`45430 x 25000`, `float32`):** `12,303 MB`.

| Modelo (`K = 10`, `Q = 200`, `seed = 42`) | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Intra-List Diversity |
|---|---|---|---|---|---|---|---|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` | `0.0171` | **`0.9829`** |
| **Content-Based Filtering** | `0.2045` | `0.4231` | `0.4244` | `0.3531` | `0.0350` | `0.2508` | `0.7492` |
| **Hybrid Recommendation (v2.0.1)** | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | **`0.0354`** | `0.2485` | `0.7515` |

---

## F. Auditoria de Segurança

1. **Auditoria de Vulnerabilidades em Dependências (`pip-audit` & `pip check`):**
   - **Comando:** `python -m pip check ; python -m pip_audit -r requirements.txt`
   - **Resultado:** `No broken requirements found.` / **`No known vulnerabilities found`** (exit code `0`).
2. **Verificação de Configuração e Código:**
   - **Segredos versionados:** Nenhum segredo ou chave de API presente no código ou no histórico; `.env` ignorado no `.gitignore` e `.dockerignore`.
   - **Exposição de stack traces:** Bloqueada pelos handlers em `src/cinematch/api/errors.py`.
   - **`debug=True` em produção:** Bloqueado por `AppConfig.validate()`, que lança `ConfigurationError` caso `env == "production"` e `debug == True`.
   - **CORS:** Restrito a `app_config.cors_origins` apenas nas rotas `/api/*`.
   - **Execução do Container:** Usuário não privilegiado `cinematch` (UID `10001`, GID `10001`).

---

## G. Validação de Docker e Docker Compose (Execução Real do Container)

Todos os testes de build, inicialização WSGI com Gunicorn, montagem de volume somente-leitura e health check foram executados diretamente no Docker Engine (`Docker version 28.3.2, build 578ccf6`):

1. **Build da Imagem e Isolamento do Contexto (`.dockerignore`):**
   - **Comando:** `docker compose up --build -d`
   - **Contexto transferido:** `373.66 kB` em `0.2s` (confirmando que os `34,4 MB` de `movies_metadata.csv` foram excluídos da imagem pelo `.dockerignore`).
   - **Pacotes instalados na imagem (`python:3.13-slim`):** `cinematch-recommender-2.0.1`, `gunicorn-27.0.1`, `flask-3.1.3`, `scikit-learn-1.9.1`, `scipy-1.18.1`, `pandas-2.3.3`, `pydantic-2.14.4`.
2. **Verificação de Usuário Não Privilegiado:**
   - **Comando:** `docker exec cinematch-recommender id`
   - **Resultado:** `uid=10001(cinematch) gid=10001(cinematch) groups=10001(cinematch)`
3. **Inicialização do Gunicorn (`--preload`) e Compartilhamento de Memória entre Workers:**
   - **Logs reais do container (`docker compose logs cinematch`):**
     ```text
     2026-11-20 22:47:43 | INFO | cinematch.data.loader          | Loading movie metadata from '/app/data/movies_metadata.csv'
     2026-11-20 22:47:49 | INFO | cinematch.data.loader          | Dataset preprocessed: 45430 valid movies (45430 active after sampling, 20 genres).
     2026-11-20 22:48:08 | INFO | cinematch.models.content_based | ContentBasedRecommender fitted: shape=(45430, 25000), nnz=1590008, memory=12.30 MB
     2026-11-20 22:48:08 | INFO | cinematch.models.collaborative | CollaborativeRecommender initialized in cold-start standby mode (no user ratings provided).
     2026-11-20 22:48:08 | INFO | cinematch.main                 | CineMatch services initialized with 45430 movies.
     [2026-11-20 22:48:08 +0000] [1] [INFO] Starting gunicorn 27.0.1
     [2026-11-20 22:48:08 +0000] [1] [INFO] Listening at: http://0.0.0.0:5000 (1)
     [2026-11-20 22:48:08 +0000] [1] [INFO] Using worker: sync
     [2026-11-20 22:48:08 +0000] [66] [INFO] Booting worker with pid: 66
     [2026-11-20 22:48:09 +0000] [67] [INFO] Booting worker with pid: 67
     ```
   - O processo mestre (PID `1`) carregou o catálogo e ajustou a matriz TF-IDF uma única vez; os workers (PIDs `66` e `67`) foram criados via `fork` em `< 1s` sem recarregar o dataset.
4. **Validação do Volume Montado como Somente-Leitura (`:ro`):**
   - **Comando:** `docker exec cinematch-recommender sh -c "touch /app/data/movies_metadata.csv"`
   - **Resultado:** `touch: cannot touch '/app/data/movies_metadata.csv': Read-only file system` (confirmando proteção contra escrita/modificação do dataset pelo container).
5. **Validação do Endpoint `/api/health` dentro do Container:**
   - **Resultado:** HTTP `200 OK` — `"status": "healthy"`, `"version": "2.0.1"`, `"catalog_size": 45430`, `"tfidf_sparse_memory_mb": 12.304`.

---

## H. Pendências Remanescentes e Limitações Conhecidas

1. **Ausência de Dataset Público de Avaliações Individuais (`ratings.csv`) por Padrão:** O repositório utiliza `data/movies_metadata.csv`, que não possui avaliações individuais por usuário. O módulo `CollaborativeRecommender` está totalmente implementado, endurecido e testado com matrizes de interação reais na suíte de testes, permanecendo em modo *Cold-Start Standby* na aplicação padrão até que um arquivo `ratings.csv` compatível seja configurado via `CINEMATCH_RATINGS_PATH`.
2. **Execução Nativa do Gunicorn em Windows:** O Gunicorn depende de chamadas POSIX (`fcntl`, `pwd`, `grp`) disponíveis em Linux/macOS/Docker. Em desenvolvimento local nativo no Windows, utiliza-se `python -m cinematch.main`, enquanto o módulo WSGI `cinematch.wsgi:app` é 100% compatível e executado pelo Gunicorn dentro do container Linux.

---

## I. Recomendação de Release

**PARECER TÉCNICO: APROVADO COMO CANDIDATO A RELEASE (`v2.0.1-rc1` / `v2.0.1`)**

Todos os critérios obrigatórios de engenharia de software, integridade algorítmica de Machine Learning, cobertura de testes ($\ge 80\%$, atingido **86,42%**), linting (`ruff`), tipagem estática (`mypy`) e auditoria de vulnerabilidades (`pip-audit`) foram integralmente cumpridos e verificados na branch `fix/cinematch-production-hardening`.
