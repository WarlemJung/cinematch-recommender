# Relatório de Auditoria Corretiva Inicial — CineMatch v2.0.1

**Data da Auditoria:** 20 de Novembro de 2026  
**Branch de Origem:** `main` (commit `f9292c5`)  
**Branch de Trabalho:** `fix/cinematch-production-hardening`  
**Versão Auditada:** `2.0.0` $\rightarrow$ **Alvo:** `2.0.1`

---

## 1. Estado Inicial do Repositório

- **Estado do Git:** Working tree limpa na branch `main`; criada a branch dedicada `fix/cinematch-production-hardening`.
- **Instruções Locais (`AGENTS.md`):** Verificado que não há arquivo `AGENTS.md` no repositório; toda a documentação técnica reside em `docs/`, `README.md` e `README_pt.md`.
- **Ferramentas do Ambiente Local:** Python 3.13.7, Docker 28.3.2 (`build 578ccf6`), pytest 9.2.0, pytest-cov 7.1.0, ruff 0.17.6, mypy 2.4.1.

---

## 2. Matriz de Problemas Confirmados vs. Não Reproduzidos

### 2.1. Problemas Confirmados no Código Atual (v2.0.0)

| ID | Prioridade | Problema Confirmado | Causa Raiz Identificada no Código | Arquivos Afetados |
|---|---|---|---|---|
| **CORR-01** | **CRÍTICA** | Uso do servidor de desenvolvimento Flask (`app.run()`) no container Docker de produção e ausência de entrypoint WSGI dedicado. | `Dockerfile` executa `CMD ["python", "-m", "cinematch.main"]`, que invoca `app.run()`. `gunicorn` não consta em `pyproject.toml` nem em `requirements.txt`. Adicionalmente, `CollaborativeRecommender.fit()` era chamado duas vezes durante a inicialização em standby porque `HybridRecommender.fit()` testava `not self.collaborative_model.is_available` em vez de `_is_fitted`. | `Dockerfile`, `src/cinematch/main.py`, `src/cinematch/wsgi.py` (novo), `pyproject.toml`, `requirements.txt`, `src/cinematch/models/collaborative.py` |
| **CORR-02** | **CRÍTICA** | Inconsistência na convenção do caminho do dataset entre Docker Compose, `.env.example` e estrutura local `data/`. | `docker-compose.yml` monta `./movies_metadata.csv:/app/data/movies_metadata.csv:ro` (raiz local), enquanto a convenção arquitetural define `data/movies_metadata.csv`. Além disso, `MovieDataLoader.load()` não diferencia explicitamente diretório montado por engano, arquivo vazio (`0 bytes`) ou erro de permissão (`PermissionError`). | `docker-compose.yml`, `.env.example`, `src/cinematch/config.py`, `src/cinematch/data/loader.py`, `README.md`, `README_pt.md`, `docs/DEPLOYMENT.md` |
| **CORR-03A** | **ALTA** | Contagem de interações no Collaborative Filtering baseada em linhas brutas em vez de usuários distintos, sem deduplicação de múltiplas avaliações `(user_id, movie_id)`. | Em `src/cinematch/models/collaborative.py` (L88), `valid_ratings.groupby("movie_id")["user_id"].count()` conta linhas totais; se um mesmo usuário avaliar o mesmo filme múltiplas vezes, infla a qualificação `min_interactions`, distorce `user_means` e soma entradas duplicadas na construção da `csr_matrix`. | `src/cinematch/models/collaborative.py`, `src/cinematch/data/loader.py` |
| **CORR-03B** | **ALTA** | Transformação linear `(raw_sims + 1.0) / 2.0` converte similaridade zero ou negativa em score positivo artificial. | Em `src/cinematch/models/collaborative.py` (L156), itens ortogonais ($\cos = 0$) recebem score `0.50` e itens negativamente correlacionados ($\cos = -0.5$) recebem `0.25`, superando itens sem avaliação (`0.0`). | `src/cinematch/models/collaborative.py` |
| **CORR-04A** | **ALTA** | Divergência entre documentação e código na definição do oráculo de relevância do benchmark. | A docstring de `evaluate_catalog_models` (`src/cinematch/models/evaluation.py`, L143) descreve associação por `collection_name` combinada com verificação de gênero, mas o código (L166-177) agrupava apenas por `collection_name` sem verificar interseção de gênero (`genres_list`). | `src/cinematch/models/evaluation.py`, `docs/MODEL_EVALUATION.md` |
| **CORR-04B** | **ALTA** | Degradação de ranking do modelo Híbrido frente ao Content-Based puro (`NDCG@10` de `0.3292` vs. `0.4108`) por incompatibilidade de escala e viés de popularidade global. | Em `src/cinematch/models/hybrid.py` (L194-197), `content_scores` brutos (cujo topo para documentos distintos varia tipicamente entre `0.20` e `0.45`) eram somados diretamente a `demographic_scores` globais (que atingem `0.95 – 1.00` para blockbusters). Assim, filmes populares com similaridade temática zero ($S_c = 0.0$) obtinham score híbrido $\approx 0.29$, superando sequências reais com similaridade temática moderada. | `src/cinematch/models/hybrid.py`, `src/cinematch/models/evaluation.py`, `scripts/train_models.py`, `docs/MODEL_EVALUATION.md` |
| **CORR-05** | **ALTA** | Exclusão de `src/cinematch/main.py` da medição de cobertura de testes. | `pyproject.toml` (L47-49) contém `omit = ["src/cinematch/main.py"]`. | `pyproject.toml`, `src/cinematch/main.py`, `tests/` |
| **CORR-06** | **MÉDIA** | Workflow de CI executa apenas em Python 3.13 em vez de matriz `3.11`, `3.12` e `3.13`. | `.github/workflows/ci.yml` (L19) fixa `python-version: "3.13"` sem `strategy.matrix`. | `.github/workflows/ci.yml` |

### 2.2. Problemas Não Reproduzidos (Já Mitigados na v2.0.0)

- **Vazamento de memória por matriz quadrática densa:** Confirmado que `ContentBasedRecommender` já utiliza exclusivamente `scipy.sparse.csr_matrix` (`12,30 MB` para 45.430 filmes).
- **Vulnerabilidades DOM XSS via `innerHTML`:** Confirmado que `frontend/static/js/script.js` não possui uso dinâmico de `innerHTML`.
- **Exposição de stack traces na API:** Confirmado que `src/cinematch/api/errors.py` intercepta exceções e retorna payloads padronizados sem traceback.

---

## 3. Riscos Técnicos Identificados

1. **Multiplicação de Memória em Múltiplos Workers Gunicorn:** Se cada worker Gunicorn carregar o CSV de 34 MB e ajustar o `TfidfVectorizer` independentemente após o `fork()`, o consumo de RAM será multiplicado por $W$ workers ($W \times 190\text{ MB}$). Utilizando `--preload` no Gunicorn (ou singleton por processo no módulo `cinematch.wsgi`), o catálogo e a matriz CSR esparsa são inicializados uma única vez no processo mestre e compartilhados via *Copy-on-Write (CoW)* entre os workers no Linux.
2. **Compatibilidade Multiplataforma (Windows Dev vs. Linux Container):** O pacote `gunicorn` depende de módulos POSIX (`fcntl`, `pwd`, `grp`) em tempo de execução do servidor, mas o módulo Python `cinematch.wsgi` deve ser 100% importável e testável tanto em Windows quanto em Linux.

---

## 4. Plano de Execução Incremental

1. **Etapa 1 — Servidor WSGI de Produção (`CORR-01`):**
   - Criar `src/cinematch/wsgi.py` expondo a aplicação WSGI com inicialização única por processo e configuração via variáveis de ambiente (`GUNICORN_WORKERS`, `GUNICORN_TIMEOUT`, `GUNICORN_BIND`).
   - Adicionar `gunicorn>=22.0.0,<24.0.0` em `pyproject.toml` e `requirements.txt`.
   - Atualizar `Dockerfile` para executar `gunicorn` com `--preload`, timeout controlado, graceful shutdown e usuário não privilegiado `cinematch`.
2. **Etapa 2 — Padronização do Dataset em `data/movies_metadata.csv` (`CORR-02`):**
   - Padronizar `data/movies_metadata.csv` localmente (copiando o arquivo original sem sobrescrevê-lo) e configurar `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro` no `docker-compose.yml`.
   - Fortalecer `MovieDataLoader.load()` para tratar arquivo ausente, caminho que aponta para diretório, arquivo vazio (`0 bytes`), CSV malformado e falha de permissão (`PermissionError`).
3. **Etapa 3 — Hardening do Collaborative Filtering (`CORR-03A`, `CORR-03B`, `CORR-03C`):**
   - Implementar deduplicação determinística de `(user_id, movie_id)` (priorizando o registro mais recente por `timestamp` quando disponível, ou a última ocorrência válida), contagem estrita por usuários distintos (`nunique`) e truncamento de similaridades de cosseno não positivas ($\max(0.0, \cos)$).
4. **Etapa 4 — Alinhamento Metodológico da Avaliação e Otimização do Motor Híbrido (`CORR-04A`, `CORR-04B`):**
   - Implementar verificação explícita de interseção de gênero junto ao holdout de `collection_name` no oráculo de avaliação.
   - Corrigir a fusão de scores no `HybridRecommender` (normalização relativa ao máximo da consulta $\tilde{S}_c(q, i)$ e gating temático para impedir que filmes com similaridade de conteúdo zero dominem o ranking apenas por nota demográfica).
   - Atualizar `scripts/train_models.py` para registrar metadados completos do experimento, análise de sensibilidade de pesos e exportar resultados reproduzíveis em JSON/CSV.
5. **Etapa 5 — Cobertura Integral (Incluindo `main.py` e `wsgi.py`) e Testes de Resiliência (`CORR-05`):**
   - Remover `omit = ["src/cinematch/main.py"]` de `pyproject.toml` e expandir a suíte de testes para cobrir todos os cenários de fábrica, WSGI, dataset, collaborative edge cases e avaliação ($\ge 85\%$ de cobertura total).
6. **Etapa 6 — Matriz Python 3.11 / 3.12 / 3.13 no CI, Documentação v2.0.1 e Auditoria Final (`CORR-06`):**
   - Atualizar `.github/workflows/ci.yml`, atualizar documentação PT/EN e gerar `docs/audits/FINAL_AUDIT_V2.0.1.md`.
