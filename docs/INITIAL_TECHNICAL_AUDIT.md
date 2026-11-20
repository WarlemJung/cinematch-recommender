# Relatório Inicial de Auditoria Técnica — CineMatch (`Movie_Recommender_System`)

**Data da Auditoria:** 20 de Novembro de 2026  
**Escopo:** Inspeção integral de arquitetura, performance algorítmica, engenharia de dados, segurança da aplicação (AppSec), API, frontend, testes e infraestrutura.  
**Ambiente Auditado:** Python 3.13.7, Flask 3.1.2, pandas 2.3.3, scikit-learn 1.7.2.

---

## 1. Resumo Executivo

A auditoria técnica do repositório `Movie_Recommender_System` revelou que a aplicação atual funciona como um protótipo funcional básico, mas apresenta **falhas críticas de escalabilidade de memória, vulnerabilidades de segurança (XSS refletido via DOM, debug ativo, CORS irrestrito, vazamento de exceções), bugs funcionais em títulos duplicados (HTTP 500), ausência de validação/higienização de dados e inexistência de testes automatizados**.

Durante os testes dinâmicos da implementação original:
- A inicialização do módulo `app.py` bloqueia o processo calculando uma matriz densa de $20.000 \times 20.000$ (`float64`), consumindo **~3,2 GB de RAM** apenas para metade do catálogo.
- O endpoint `GET /api/recommend?title=Heat` falha com **HTTP 500 (`ValueError: The truth value of an array with more than one element is ambiguous`)** porque o mapeamento reverso de títulos não remove chaves duplicadas.
- Qualquer busca por um título inexistente contendo tags HTML/JS é refletida pela API e injetada via `innerHTML` no frontend, resultando em **Cross-Site Scripting (DOM XSS)** imediato.

---

## 2. Matriz de Problemas Identificados

| ID | Categoria | Problema Identificado | Severidade | Impacto | Arquivos Afetados |
|---|---|---|---|---|---|
| **AUD-01** | Performance / ML | Materialização de matriz densa quadrática $O(N^2)$ em `linear_kernel(tfidf_matrix, tfidf_matrix)` | **Crítica** | Consumo de ~3,2 GB de RAM para 20k filmes (~16,5 GB para o dataset completo de 45.466 filmes), alto tempo de boot (~25s) e impossibilidade de escalar. | `recommender.py` (L77), `app.py` (L17) |
| **AUD-02** | Segurança (AppSec) | DOM XSS refletido e armazenado via uso de `innerHTML` com dados não sanitizados (incluindo mensagem de erro refletindo o input do usuário) | **Crítica** | Execução arbitrária de JavaScript no navegador do usuário ao pesquisar payloads maliciosos ou renderizar metadados externos. | `static/js/script.js` (L48, L70-77, L88-93), `recommender.py` (L95) |
| **AUD-03** | Segurança (AppSec) | Execução com `app.run(debug=True)` hardcoded | **Crítica** | Exposição do debugger interativo do Werkzeug, permitindo execução remota de código (RCE) caso exposto em ambiente acessível. | `app.py` (L59) |
| **AUD-04** | Bug Funcional / Dados | Falha no tratamento de títulos duplicados (`pd.Series.drop_duplicates()` remove valores de índice em vez de rótulos de título) | **Alta** | 3.183 registros com títulos duplicados (ex.: `"Heat"`, `"Hamlet"`, `"Cinderella"`) retornam múltiplos índices e causam crash HTTP 500 ao ordenar similaridades. | `recommender.py` (L80, L93-101), `app.py` (L48-56) |
| **AUD-05** | Qualidade de Dados | Ausência de validação de schema, registros corrompidos (`id` contendo datas), IDs duplicados e truncamento arbitrário (`iloc[0:20000]`) | **Alta** | 3 linhas corrompidas no CSV (`id` = `'1997-08-20'`, `'2012-09-29'`, `'2014-01-01'`), 30 IDs duplicados, 6 títulos nulos, 90 datas inválidas e viés de amostragem posicional. | `recommender.py` (L18-28) |
| **AUD-06** | Segurança (AppSec) | Vazamento de mensagens internas de exceção (`str(e)`) e CORS irrestrito (`CORS(app)`) | **Alta** | Exposição de detalhes internos da stack ao cliente em erros 500 e permissão irrestrita para requisições cross-origin de qualquer domínio (`*`). | `app.py` (L7, L36, L56) |
| **AUD-07** | Arquitetura | Acoplamento monolítico e inicialização pesada em tempo de importação (`import app` executa carga do CSV e cálculo TF-IDF no escopo global) | **Alta** | Inviabiliza testes unitários rápidos, injeção de dependências, múltiplos workers WSGI controlados e configuração dinâmica por ambiente. | `app.py` (L11-20), `recommender.py` (L6-7) |
| **AUD-08** | Algoritmos de ML | Recomendação baseada em conteúdo restrita apenas à coluna `overview` e retorno apenas de strings de títulos | **Média** | Baixa riqueza semântica (ignora gêneros, tagline, franquia/coleção, estúdios) e perda de metadados (ID, ano, nota, score, explicação) na resposta da API. | `recommender.py` (L21, L74, L110), `app.py` (L54) |
| **AUD-09** | Algoritmos de ML | Ausência de Filtragem Colaborativa, Recomendação Híbrida explicável e Pipeline de Avaliação Quantitativa | **Média** | Sistema limitado a dois métodos isolados sem combinação ponderada, sem tratamento documentado de cold-start e sem métricas (Precision@K, NDCG@K, Coverage, Diversity). | `recommender.py` |
| **AUD-10** | Qualidade / DevOps | Ausência total de testes automatizados, linter/type-checking, pinagem de dependências, Dockerfile e CI/CD | **Média** | Alto risco de regressão, builds não reprodutíveis (`requirements.txt` sem versões) e ausência de validação contínua. | `requirements.txt`, repositório raiz |

---

## 3. Análise Detalhada por Domínio

### 3.1. Performance Algorítmica e Complexidade (`recommender.py`)

- **Diagnóstico:**  
  No método `compute_content_similarity()`, a matriz esparsa `tfidf_matrix` de dimensão $N \times V$ (onde $N = 20.000$ e $V \approx 47.000$) é multiplicada por sua transposta via `linear_kernel(tfidf_matrix, tfidf_matrix)`, gerando um `numpy.ndarray` denso de $N \times N$ (`float64`).
  - Memória consumida ($N = 20.000$): $20.000^2 \times 8\text{ bytes} = 3,2\text{ GB}$.
  - Memória necessária para o dataset completo ($N = 45.466$): $45.466^2 \times 8\text{ bytes} \approx 16,54\text{ GB}$.
- **Solução Proposta:**  
  Manter apenas a matriz TF-IDF esparsa em formato CSR (`scipy.sparse.csr_matrix`), que ocupa **< 35 MB** mesmo para todo o catálogo de ~45.430 filmes válidos ($O(\text{nnz})$ em memória). Calcular a similaridade sob demanda apenas para o vetor do filme consultado `linear_kernel(tfidf_matrix[idx], tfidf_matrix)` com complexidade de tempo $O(\text{nnz}_{\text{query}} \cdot \bar{d})$ e espaço auxiliar $O(N)$ (~360 KB por consulta).
- **Critério de Validação:**  
  Consumo de memória do modelo de conteúdo inferior a 100 MB para o dataset completo; tempo de inferência por consulta inferior a 50 ms; testes unitários garantindo ausência de matriz $N \times N$.

### 3.2. Qualidade de Dados (`movies_metadata.csv`)

A inspeção automatizada do arquivo `movies_metadata.csv` (45.466 linhas e 24 colunas) identificou:
- **Registros corrompidos:** 3 linhas possuem valores de data na coluna `id` (`'1997-08-20'`, `'2012-09-29'`, `'2014-01-01'`) devido a deslocamento de colunas no dataset bruto do Kaggle.
- **Identificadores duplicados:** 30 registros possuem `id` numérico duplicado.
- **Valores ausentes / inválidos:**
  - `title`: 6 valores nulos (`NaN`).
  - `vote_average` e `vote_count`: 6 valores nulos/inválidos.
  - `overview`: 954 valores `NaN` e 5 strings contendo apenas espaços (total de 959 sinopses vazias).
  - `release_date`: 87 valores nulos e 3 valores malformados (90 datas inválidas no total).
  - `poster_path`: 386 valores nulos.
  - `tagline`: 25.054 valores nulos.
- **Títulos duplicados:** 3.183 linhas compartilham título idêntico com outro filme (remakes, homônimos de anos diferentes).
- **Solução Proposta:**  
  Implementar pipeline estruturado em `src/cinematch/data/` com validação de colunas obrigatórias, coerção segura de tipos (`pd.to_numeric(..., errors="coerce")`, `pd.to_datetime(..., errors="coerce")`), parsing seguro de estruturas JSON/Python serializadas (`genres`, `belongs_to_collection`, `production_companies`), deduplicação por `movie_id` preservando o registro com maior `vote_count`, geração de relatório estruturado de qualidade de dados (`DataQualityReport`) e amostragem reproduzível opcional com `random_state` configurável (permitindo também processar o catálogo completo graças à representação esparsa).
- **Critério de Validação:**  
  Zero IDs não numéricos ou duplicados após o pré-processamento; relatório de qualidade auditável via script `scripts/validate_dataset.py` e testes unitários cobrindo dados corrompidos.

### 3.3. Segurança da Aplicação (AppSec)

1. **DOM Cross-Site Scripting (XSS) (`static/js/script.js`):**
   - `recommendationsGrid.innerHTML = '<p ...>' + data.error + '</p>'` injeta o texto de erro retornado pela API, que inclui o parâmetro `title` enviado pelo usuário sem escape HTML.
   - `card.innerHTML` injeta `movie.title`, `overview`, `vote_average` e `vote_count` diretamente no DOM.
   - **Correção:** Proibir `innerHTML` para dados dinâmicos; construir elementos com `document.createElement` e `textContent`; validar URLs de imagens (permitindo apenas caminhos válidos ou origens HTTPS confiáveis como `https://image.tmdb.org/t/p/` com fallback seguro em SVG/CSS).
2. **Exposição de Erros e Debug (`app.py`):**
   - `app.run(debug=True)` ativo por padrão e `jsonify({'error': str(e)}), 500` em blocos `except Exception`.
   - **Correção:** Centralizar configuração em `src/cinematch/config.py` com `DEBUG=False` por padrão; implementar handlers padronizados em `src/cinematch/api/errors.py` que registram a stack trace em log estruturado no servidor e retornam mensagens seguras e padronizadas ao cliente.
3. **CORS e Validação de Entrada (`app.py`):**
   - `CORS(app)` aberto para `*` e ausência de limites em parâmetros de consulta.
   - **Correção:** Restringir origens do CORS por variável de ambiente (`CORS_ORIGINS`), validar todos os parâmetros de entrada via schemas Pydantic (`top_n` limitado entre 1 e 50, paginação validada, strings higienizadas).

---

## 4. Plano de Implementação Incremental

A evolução para o **CineMatch (`cinematch-recommender`)** será executada nas seguintes fases verificáveis:

1. **Fase 1 — Auditoria e Diagnóstico (Concluída neste documento):** Mapeamento completo de bugs, vulnerabilidades, perfil do dataset e arquitetura atual.
2. **Fase 2 — Nova Arquitetura Modular (`src/cinematch/`):**
   - Criação da estrutura em camadas (`config`, `utils`, `data`, `models`, `services`, `api`) com Application Factory (`create_app`), injeção de dependências e compatibilidade retroativa.
3. **Fase 3 — Evolução dos Algoritmos de Machine Learning:**
   - **Demographic Filtering:** Fórmula IMDB Weighted Rating vetorizada, parâmetros `quantile` e `min_votes` configuráveis e normalização de score.
   - **Content-Based Filtering:** Matriz TF-IDF esparsa ponderada por características documentadas (`overview`, `genres`, `tagline`, `collection`, `production_companies`, `original_language`), similaridade sob demanda $O(N)$, busca por ID e busca normalizada por título com desambiguação por ano/popularidade.
   - **Collaborative Filtering:** Implementação de filtragem colaborativa Item-Based / Matrix Factorization isolada (SVD / Cosseno esparso sobre matriz usuário-item real quando arquivo de avaliações reais `ratings.csv` estiver presente, ou modo opcional transparente com fallback documentado para cold-start sem jamais inventar interações fictícias).
   - **Hybrid Recommendation Engine:** Combinação linear ponderada com reajuste automático de pesos caso algum componente não esteja aplicável, retornando metadados completos, scores individuais e explicação dinâmica gerada a partir dos atributos reais.
4. **Fase 4 — Pipeline de Avaliação Quantitativa:**
   - Implementação de `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Recommendation Diversity` e `Intra-list Similarity` com split sem data leakage e documentação dos resultados medidos reais em `docs/MODEL_EVALUATION.md`.
5. **Fase 5 — Evolução da API REST:**
   - Endpoints `/`, `/api/health`, `/api/trending`, `/api/movies/search`, `/api/movies/<int:movie_id>`, `/api/recommend/<int:movie_id>`, `/api/recommend` (compatibilidade retroativa por título/ID), `/api/recommendations/hybrid/<int:movie_id>` e `/api/genres`, com schemas Pydantic, paginação e logging estruturado.
6. **Fase 6 — Modernização do Frontend:**
   - Interface Dark Mode SaaS/Streaming com busca autocomplete, filtros por gênero/ano/avaliação, modal de detalhes, favoritos persistentes em `localStorage`, exibição de explicações do modelo híbrido, paginação, suporte opcional a pôsteres TMDB com fallback visual elegante e zero uso inseguro de `innerHTML`.
7. **Fase 7 — Testes Automatizados e Qualidade de Código:**
   - Suíte `pytest` (unitários e integração com fixtures sintéticas), `ruff` e `mypy`, superando 80% de cobertura no código crítico.
8. **Fase 8 & 9 — Segurança, Produção, Docker e CI/CD:**
   - `.env.example`, `Dockerfile` multi-stage / non-root com healthcheck, `docker-compose.yml`, `pyproject.toml` e workflow `.github/workflows/ci.yml`.
9. **Fase 10 & 11 — Documentação Profissional e Auditoria Técnica Final:**
   - Atualização completa de `README.md`, `README_pt.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/DATA_DICTIONARY.md`, `docs/DEPLOYMENT.md` e `docs/FINAL_TECHNICAL_AUDIT.md`.
