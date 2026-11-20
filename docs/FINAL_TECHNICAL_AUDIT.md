# Relatório Técnico Final de Auditoria — CineMatch (`cinematch-recommender`)

**Data de Conclusão:** 20 de Novembro de 2026  
**Versão Auditada:** `2.0.0`  
**Ambiente Validado:** Python 3.13.7, Flask 3.1.2, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.16.3, Pydantic 2.12.4, pytest 9.2.0, ruff 0.17.6, mypy 2.4.1.

---

## 1. Resumo Executivo

A refatoração arquitetural e evolução de Machine Learning do projeto `Movie_Recommender_System` para o **CineMatch (`cinematch-recommender` v2.0.0)** foi concluída e validada integralmente.

Os principais marcos técnicos alcançados e medidos em execução real foram:
- **Redução de 99,92% no Consumo de Memória do Modelo de Conteúdo:** A matriz densa quadrática $N \times N$ (`linear_kernel(tfidf, tfidf)`, que consumia ~3,2 GB de RAM para 20.000 filmes e exigiria ~16,51 GB para o dataset completo) foi substituída por uma representação **esparsa CSR (`float32`) com cálculo sob demanda** (`linear_kernel(tfidf_matrix[idx], tfidf_matrix)`). Isso permitiu escalar o sistema de 20.000 para **100% dos 45.430 filmes válidos do catálogo**, consumindo apenas **12,303 MB de RAM**.
- **Correção Completa de Falhas Críticas e de Segurança:** Eliminação do crash HTTP 500 em títulos duplicados (5.753 registros homônimos no catálogo, como `"Heat"`), erradicação de todas as injeções DOM XSS via `innerHTML` no frontend, desativação de `debug=True` em produção, restrição de CORS, validação Pydantic v2 e ocultação de stack traces.
- **Evolução para Sistema Híbrido Explicável e Avaliado Quantitativamente:** Implementação dos motores **Demographic (Weighted Rating)**, **Content-Based (TF-IDF Esparso Ponderado por Namespace)**, **Collaborative Filtering (TruncatedSVD com proteção explícita de Cold-Start)** e **Hybrid Ensemble** com explicações geradas dinamicamente e pipeline de avaliação anti-leakage.

---

## 2. Alterações Implementadas por Domínio

1. **Arquitetura Modular (`src/cinematch/`)**:
   - Migração para arquitetura em camadas (`config`, `utils`, `data`, `models`, `services`, `api`) com *Application Factory* (`create_app`) e injeção de dependências.
   - Preservação de compatibilidade retroativa nos entrypoints `app.py` e `recommender.py`.
2. **Engenharia e Qualidade de Dados (`src/cinematch/data/`)**:
   - Validação de schema obrigatório, remoção de 3 registros corrompidos (`id` contendo datas `'1997-08-20'`, `'2012-09-29'`, `'2014-01-01'`), remoção de 3 registros sem título e deduplicação determinística de 30 IDs duplicados (preservando o registro com maior `vote_count`).
   - Parsing seguro de JSON/AST (`genres`, `belongs_to_collection`, `production_companies`), higienização regex de `poster_path`, normalização de títulos e amostragem reproduzível configurável com `random_state`.
3. **Modelos de Machine Learning e Avaliação (`src/cinematch/models/`)**:
   - **Demographic Filtering:** Cálculo vetorizado da fórmula bayesiana do IMDB com quantil e corte mínimo configuráveis.
   - **Content-Based Filtering:** Vetorização TF-IDF esparsa com pesos configuráveis por atributo estruturado e similaridade sob demanda em $O(\text{nnz}_{\text{query}} \cdot \bar{d})$.
   - **Collaborative Filtering:** Fatoração esparsa via `TruncatedSVD` sobre matrizes reais usuário-item com modo *Cold-Start Standby* transparente (sem jamais sintetizar avaliações fictícias).
   - **Hybrid Recommendation:** Combinação linear com renormalização automática de pesos e geração factual de explicações (`explanation`).
   - **Quantitative Evaluation:** Implementação de `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Intra-List Similarity` e `Recommendation Diversity`.
4. **API REST e Frontend Modernizado (`src/cinematch/api/`, `frontend/`)**:
   - Endpoints `/`, `/api/health`, `/api/trending`, `/api/genres`, `/api/movies/search`, `/api/movies/<movie_id>`, `/api/recommend/<movie_id>`, `/api/recommend`, `/api/recommendations/hybrid/<movie_id>` e `/api/recommendations/hybrid`.
   - Interface Dark Mode com busca autocomplete, filtros por gênero/ano/nota, seletor de algoritmo ML, modal de detalhes, favoritos persistidos em `localStorage`, paginação e zero uso de `innerHTML`.

---

## 3. Inventário de Arquivos Modificados e Criados

### 3.1. Arquivos Modificados
- `app.py` (refatorado para delegar à Application Factory `create_app` com `debug=False` por padrão)
- `recommender.py` (refatorado como adaptador retrocompatível sobre o motor esparso `cinematch`)
- `requirements.txt` (dependências controladas com limites de versão e ferramentas de qualidade)
- `.gitignore` (atualizado para proteger datasets, artefatos de modelos, `.env` e caches)
- `templates/index.html`, `static/css/style.css`, `static/js/script.js` (sincronizados com o novo frontend seguro)
- `README.md` e `README_pt.md` (documentação técnica completa e coerente em inglês e português)

### 3.2. Arquivos Criados
- **Código-Fonte (`src/cinematch/`)**:
  - `src/cinematch/__init__.py`
  - `src/cinematch/config.py`
  - `src/cinematch/main.py`
  - `src/cinematch/utils/__init__.py`, `src/cinematch/utils/exceptions.py`, `src/cinematch/utils/logging.py`
  - `src/cinematch/data/__init__.py`, `src/cinematch/data/validation.py`, `src/cinematch/data/preprocessing.py`, `src/cinematch/data/loader.py`
  - `src/cinematch/models/__init__.py`, `src/cinematch/models/demographic.py`, `src/cinematch/models/content_based.py`, `src/cinematch/models/collaborative.py`, `src/cinematch/models/hybrid.py`, `src/cinematch/models/evaluation.py`
  - `src/cinematch/services/__init__.py`, `src/cinematch/services/movie_service.py`, `src/cinematch/services/recommendation_service.py`
  - `src/cinematch/api/__init__.py`, `src/cinematch/api/schemas.py`, `src/cinematch/api/errors.py`, `src/cinematch/api/routes.py`
- **Frontend Modular (`frontend/`)**:
  - `frontend/templates/index.html`
  - `frontend/static/css/style.css`
  - `frontend/static/js/script.js`
- **Testes Automatizados (`tests/`)**:
  - `tests/conftest.py`
  - `tests/unit/test_data.py`
  - `tests/unit/test_models.py`
  - `tests/unit/test_evaluation_and_config.py`
  - `tests/integration/test_api.py`
- **Scripts, Infraestrutura e CI/CD**:
  - `scripts/validate_dataset.py`
  - `scripts/train_models.py`
  - `pyproject.toml`, `.env.example`, `.dockerignore`, `Dockerfile`, `docker-compose.yml`
  - `.github/workflows/ci.yml`
  - `data/.gitkeep`, `models/.gitkeep`
- **Documentação e Governança**:
  - `LICENSE`, `CHANGELOG.md`
  - `docs/INITIAL_TECHNICAL_AUDIT.md`
  - `docs/ARCHITECTURE.md`
  - `docs/API.md`
  - `docs/DATA_DICTIONARY.md`
  - `docs/MODEL_EVALUATION.md`
  - `docs/DEPLOYMENT.md`
  - `docs/FINAL_TECHNICAL_AUDIT.md`

---

## 4. Problemas Corrigidos (Rastreabilidade com a Auditoria Inicial)

| ID | Problema na Auditoria Inicial | Status Final | Evidência de Validação |
|---|---|---|---|
| **AUD-01** | Matriz densa quadrática $O(N^2)$ em `linear_kernel(tfidf, tfidf)` (~3,2 GB RAM) | **Corrigido** | Matriz esparsa CSR `45430 x 25000` ocupando **12,303 MB**; `test_content_based_recommender_uses_sparse_matrix_and_handles_duplicate_titles` |
| **AUD-02** | DOM XSS refletido e armazenado via `innerHTML` em `script.js` | **Corrigido** | Zero ocorrências de `innerHTML` no JS; uso exclusivo de `createElement` + `textContent` e regex de URL de imagem |
| **AUD-03** | `app.run(debug=True)` hardcoded | **Corrigido** | `AppConfig(debug=False)` por padrão; bloqueio explícito de `debug=True` em `env="production"` |
| **AUD-04** | Crash HTTP 500 ao buscar títulos duplicados (ex.: `"Heat"`) | **Corrigido** | `GET /api/recommend?title=Heat` retorna `200 OK` desambiguado por `vote_count` |
| **AUD-05** | Linhas corrompidas no CSV, IDs duplicados e truncamento `iloc[0:20000]` | **Corrigido** | `preprocess_movies_dataframe` remove 36 linhas inválidas/duplicadas e processa os 45.430 filmes válidos |
| **AUD-06** | Vazamento de `str(e)` em erros 500 e CORS aberto (`*`) | **Corrigido** | Handlers em `errors.py` ocultam stack traces; CORS restrito a `CINEMATCH_CORS_ORIGINS` |
| **AUD-07** | Estado global pesado carregado em tempo de `import app` | **Corrigido** | Application Factory `create_app()` com injeção de dependências |
| **AUD-08** | Content-Based restrito a `overview` e retorno apenas de lista de strings | **Corrigido** | Documentos TF-IDF multi-atributo ponderados e retorno de objetos ricos com metadados e scores |
| **AUD-09** | Ausência de Collaborative Filtering, Hybrid Engine e Avaliação Quantitativa | **Corrigido** | Módulos `collaborative.py`, `hybrid.py` e `evaluation.py` implementados e documentados |
| **AUD-10** | Ausência de testes, linter, tipagem, Docker e CI/CD | **Corrigido** | 26 testes automatizados (84,76% de cobertura), Ruff, Mypy, Dockerfile non-root e GitHub Actions |

---

## 5. Resultados Reais dos Testes e Verificações de Qualidade

1. **Linter (`python -m ruff check src tests scripts app.py recommender.py`):**
   - Resultado: `All checks passed!` (0 erros ou avisos).
2. **Type Checking (`python -m mypy src tests scripts app.py recommender.py`):**
   - Resultado: `Success: no issues found in 32 source files` (0 erros de tipagem).
3. **Testes Unitários e de Integração (`python -m pytest --cov=src/cinematch --cov-fail-under=80 -v`):**
   - Resultado: **`26 passed in 4.82s`**.
   - Cobertura medida no pacote `src/cinematch`: **`84.76%`** (meta mínima: 80%).
4. **Integridade de Dependências (`python -m pip check`):**
   - Resultado: `No broken requirements found.`
5. **Validação dos Endpoints com Catálogo Completo (45.430 filmes):**
   - `GET /` -> `200 OK`
   - `GET /api/health` -> `200 OK` (`tfidf_sparse_memory_mb: 12.303`)
   - `GET /api/trending?top_n=5` -> `200 OK`
   - `GET /api/genres` -> `200 OK` (20 gêneros únicos)
   - `GET /api/movies/search?q=Toy+Story` -> `200 OK`
   - `GET /api/movies/862` -> `200 OK`
   - `GET /api/recommend/862?top_n=5` -> `200 OK`
   - `GET /api/recommend?title=Heat&top_n=5` -> `200 OK`
   - `GET /api/recommendations/hybrid/862?top_n=5` -> `200 OK`

---

## 6. Riscos Residuais, Limitações Conhecidas e Melhorias Futuras

### 6.1. Riscos Residuais e Limitações Conhecidas
- **Cold-Start Colaborativo no Repositório Base:** Como o dataset versionado/local padrão contém apenas `movies_metadata.csv` (sem `ratings.csv` de interações individuais), o módulo `CollaborativeRecommender` permanece em *standby* até que um arquivo `ratings.csv` real seja fornecido via `CINEMATCH_RATINGS_PATH`.
- **Latência de Cold Boot:** A leitura e limpeza inicial dos 45.466 registros do CSV bruto mais a vetorização TF-IDF levam ~5–6 segundos na inicialização do processo WSGI.

### 6.2. Melhorias Futuras Recomendadas
- **Persistência em Parquet / Cache de Artefatos:** Salvar o DataFrame higienizado em formato Apache Parquet e a matriz esparsa CSR em `.npz` no diretório `models/` para reduzir o tempo de boot do worker de ~5s para < 300ms.
- **Embeddings Semânticos Densos (Sentence-Transformers + FAISS):** Complementar o TF-IDF esparso com embeddings multilíngues indexados em HNSW para capturar sinônimos semânticos inter-idiomas.
- **Servidor WSGI de Produção:** Adicionar Gunicorn (Linux/Docker) ou Waitress (Windows) como runner padrão para múltiplos workers concorrentes.
