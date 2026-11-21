# 🎬 CineMatch — Motor Híbrido de Recomendação de Filmes

![Python 3.12 | 3.13](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Flask 3.1](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Sparse%20TF--IDF%20%26%20SVD-F7931E?logo=scikit-learn&logoColor=white)
![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Ruff & Mypy](https://img.shields.io/badge/Code%20Quality-Ruff%20%2B%20Mypy-261230)
![Licença: MIT](https://img.shields.io/badge/License-MIT-00f2ff)

> 🇺🇸 **[Read the documentation in English (README.md)](README.md)**

O **CineMatch (`cinematch-recommender`)** é um **Motor Híbrido de Recomendação de Filmes** de nível profissional desenvolvido em Python, Scikit-Learn, matrizes esparsas SciPy e Flask. Ele combina **Filtragem Demográfica (IMDB Bayesian Weighted Rating)**, **Filtragem Baseada em Conteúdo Esparsa (TF-IDF Ponderado Multifacetado)** e **Filtragem Colaborativa (Fatores Latentes TruncatedSVD com Fallback de Cold-Start)** em um sistema explicável, servido por uma API REST validada e uma interface moderna inspirada em plataformas de streaming.

---

## 🎯 1. Problema que o Projeto Resolve

Protótipos tradicionais de recomendação de filmes sofrem com três falhas críticas de engenharia:
1. **Explosão Quadrática de Memória ($O(N^2)$):** Materializar uma matriz densa de similaridade (`linear_kernel(tfidf, tfidf)`) para 45.466 filmes consome **~16,5 GB de RAM** (ou ~3,2 GB mesmo truncando o dataset em 20.000 linhas). O CineMatch preserva a matriz TF-IDF em **formato esparso CSR (`float32`)** e calcula vetores de similaridade **sob demanda em $O(\text{nnz}_{\text{query}} \cdot \bar{d})$**, processando todo o catálogo de **45.430 filmes válidos em apenas 12,30 MB de RAM** (**redução de 99,92% no uso de memória**).
2. **Limitação de Sinal Único:** Modelos puramente baseados em texto recomendam sequências mal avaliadas, enquanto rankings globais ignoram o gosto específico do usuário. O CineMatch combina afinidade temática, aclamação crítica ponderada por votos e padrões colaborativos com **explicações em linguagem natural** para cada recomendação.
3. **Dados Corrompidos e Títulos Duplicados:** O CSV bruto do Kaggle possui linhas deslocadas com datas na coluna `id`, IDs duplicados, sinopses vazias e milhares de títulos homônimos (remakes como *"Heat"* ou *"Hamlet"*) que causavam erro HTTP 500 na implementação anterior. O CineMatch aplica um pipeline rigoroso de validação, higienização e desambiguação.

---

## ✨ 2. Funcionalidades Principais

- **Motor Híbrido Explicável:** Gera explicações dinâmicas (ex.: mesma franquia, similaridade temática, gêneros compartilhados, mesmo estúdio e avaliação geral) junto com os scores individuais de cada algoritmo.
- **Content-Based Filtering Esparso Sob Demanda:** Tokens estruturados e ponderados por namespace (`overview`, `genres`, `tagline`, `belongs_to_collection`, `production_companies`, `original_language`) sem materializar matriz $N \times N$.
- **Demographic Filtering (Top Charts):** Cálculo vetorizado do Weighted Rating do IMDB com quantil configurável e filtros combinados por gênero, intervalo de anos e nota mínima.
- **Collaborative Filtering Isolado com Proteção de Cold-Start:** Fatoração matricial via `TruncatedSVD` sobre avaliações reais usuário-item (`ratings.csv`) quando disponível, reajustando automaticamente os pesos do modelo híbrido em cenários de cold-start sem jamais inventar interações fictícias.
- **Pipeline de Avaliação Quantitativa Sem Data Leakage:** Cálculo automatizado de `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Catalog Coverage`, `Intra-List Similarity (ILS)` e `Recommendation Diversity`.
- **Segurança Hardened (AppSec):** Zero uso dinâmico de `innerHTML` (proteção total contra DOM XSS via `textContent`), validação regex para URLs de pôsteres TMDB, validação de parâmetros com Pydantic v2, CORS restrito, `debug=False` obrigatório em produção e ocultação de stack traces.
- **Interface Dark Mode Moderna:** Busca com autocomplete em tempo real, barra de filtros (gênero, ano, avaliação, modo ML), modal de detalhes do filme, lista de Favoritos persistida em `localStorage` e paginação.

---

## 🏗️ 3. Arquitetura

```text
cinematch-recommender/
├── src/
│   └── cinematch/
│       ├── main.py                  # Flask Application Factory (create_app)
│       ├── wsgi.py                  # Entrypoint WSGI singleton de produção (Gunicorn)
│       ├── config.py                # Configuração imutável por ambiente e validação
│       ├── api/                     # Rotas Blueprint, schemas Pydantic v2 e error handlers
│       ├── data/                    # Loader CSV, validação de schema, limpeza e DataQualityReport
│       ├── models/                  # Demographic, Sparse Content-Based, Collaborative, Hybrid e Evaluation
│       ├── services/                # Orquestração em MovieService e RecommendationService
│       └── utils/                   # Hierarquia de exceções e logging estruturado em JSON
├── frontend/
│   ├── templates/index.html         # Interface web dark-mode acessível (ARIA)
│   └── static/                      # Design system CSS e controlador JS seguro contra XSS
├── scripts/
│   ├── validate_dataset.py          # CLI de auditoria de integridade e qualidade do dataset
│   └── train_models.py              # CLI de treinamento, telemetria de memória e benchmark quantitativo
├── tests/
│   ├── unit/                        # Testes unitários (dados, fórmulas, modelos, avaliação, config)
│   └── integration/                 # Testes de integração de todos os endpoints REST e erros
├── docs/                            # Auditorias técnicas, arquitetura, API, dicionário de dados e avaliação
├── Dockerfile                       # Container de produção non-root com healthcheck
├── docker-compose.yml               # Orquestração de container com montagem read-only do dataset
└── pyproject.toml                   # Configuração de pacote, pytest, coverage, ruff e mypy
```

Consulte [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para diagramas detalhados.

---

## 🛠️ 4. Tecnologias Utilizadas

- **Linguagem:** Python 3.11+ / 3.13
- **Machine Learning & Dados:** `scikit-learn`, `scipy` (matrizes esparsas CSR), `pandas`, `numpy`
- **API & Validação:** `Flask`, `flask-cors`, `pydantic` (v2), `python-dotenv`
- **Frontend:** HTML5 Semântico, CSS3 Moderno (Custom Properties, Grid, Glassmorphism), Vanilla ES6+ JavaScript
- **Qualidade & DevOps:** `pytest`, `pytest-cov`, `ruff`, `mypy`, Docker, GitHub Actions CI

---

## 🧠 5. Algoritmos Utilizados

### 5.1. Demographic Filtering (IMDB Weighted Rating)
$$\text{WR} = \left(\frac{v}{v + m} \times R\right) + \left(\frac{m}{v + m} \times C\right)$$
Onde $R$ é a nota média do filme, $v$ é a contagem de votos, $m$ é o limite mínimo de votos (percentil 90 por padrão, $m = 160$ votos no catálogo completo) e $C$ é a nota média global do catálogo ($C = 5,6182$).

### 5.2. Content-Based Filtering Esparso
Constrói um documento ponderado por filme combinando tokens prefixados por namespace (`genre_*`, `collection_*`, `studio_*`, `lang_*`) com o texto livre de `overview` e `tagline`. Vetorizado em uma matriz esparsa CSR `float32` de $45.430 \times 25.000$ (**12,30 MB**). A similaridade é calculada sob demanda apenas para a linha do filme consultado $q$:
$$\mathbf{s}(q) = \mathbf{X}_{q, :} \cdot \mathbf{X}^T$$

### 5.3. Collaborative Filtering
Aplica centralização pela média do usuário e redução de dimensionalidade via `TruncatedSVD` sobre matrizes reais de avaliação usuário-item (`userId, movieId, rating`). Caso nenhum arquivo de avaliações externas esteja configurado ou um filme não possua histórico, o componente opera em *Cold-Start Standby* sem inventar interações.

### 5.4. Hybrid Weighted Recommendation
$$\text{Score}_{\text{hybrid}}(q, i) = \hat{w}_c S_{\text{content}}(q, i) + \hat{w}_d S_{\text{demographic}}(i) + \hat{w}_{cf} S_{\text{collaborative}}(q, i)$$
Os pesos $(\hat{w}_c, \hat{w}_d, \hat{w}_{cf})$ são normalizados automaticamente para somar `1.0` entre os modelos ativos para a consulta $q$.

---

## 📦 6. Instalação e Configuração

### 6.1. Clonar e Instalar Dependências
```bash
git clone https://github.com/WarlemJung/cinematch-recommender.git
cd cinematch-recommender
pip install -r requirements.txt
```

### 6.2. Dataset
Coloque o arquivo `movies_metadata.csv` ([The Movies Dataset no Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)) no caminho padronizado `data/movies_metadata.csv`.
Opcionalmente, adicione `ratings_small.csv` ou `ratings.csv` em `data/` e configure `CINEMATCH_RATINGS_PATH` para habilitar a Filtragem Colaborativa.

### 6.3. Variáveis de Ambiente
Copie `.env.example` para `.env`:
```bash
cp .env.example .env
```

---

## 🚀 7. Execução Local e com Docker

### 7.1. Validar Dataset e Executar Avaliação Quantitativa
```bash
python scripts/validate_dataset.py
python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity
```

### 7.2. Iniciar a Aplicação Web (WSGI Produção vs. Desenvolvimento Local)
```bash
# Servidor WSGI de produção (Linux / macOS / Container):
PYTHONPATH=src gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --preload cinematch.wsgi:app

# Servidor local de desenvolvimento (Windows / macOS / Linux):
python -m cinematch.main
```
Acesse `http://localhost:5000` no navegador.

### 7.3. Executar com Docker Compose
```bash
docker compose up --build -d
```

---

## 🔌 8. Endpoints da API REST

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Interface web interativa |
| `GET` | `/api/health` | Health check, uso de memória da matriz esparsa e relatório de qualidade de dados |
| `GET` | `/api/trending` | Filmes populares por Weighted Rating com paginação e filtros (`genre`, `min_year`, `max_year`, `min_rating`) |
| `GET` | `/api/genres` | Lista de gêneros do catálogo e contagem de filmes |
| `GET` | `/api/movies/search` | Busca paginada por título (`?q=...`) com suporte a múltiplos filtros |
| `GET` | `/api/movies/<movie_id>` | Detalhes completos de um filme por ID |
| `GET` | `/api/recommend/<movie_id>` | Recomendações baseadas em conteúdo por ID (também suporta `/api/recommend?title=...`) |
| `GET` | `/api/recommendations/hybrid/<movie_id>` | Recomendações híbridas explicáveis por ID (também suporta `/api/recommendations/hybrid?title=...`) |

Consulte [docs/API.md](docs/API.md) para exemplos completos de requisição e resposta.

---

## 🧪 9. Testes Automatizados e Qualidade

```bash
# Executar suíte de testes unitários e de integração com verificação de cobertura (>= 80%)
python -m pytest --cov=src/cinematch --cov-report=term-missing --cov-fail-under=80 -v

# Executar linter Ruff e verificação estática de tipos Mypy
python -m ruff check src tests scripts app.py recommender.py
python -m mypy src tests scripts app.py recommender.py
```

---

## 📊 10. Métricas Reais de Avaliação

Medidas no catálogo validado (`N = 45.430` filmes, `K = 10`, separação estrita `Q_val = 100` validação e `Q_test = 200` teste em holdout, `seed = 42`, **sem sobreposição com `genres_list`**) com avaliações reais do **MovieLens `ml-latest-small`** (`data/ratings.csv`, `610` usuários, split temporal 70/15/15):

| Algoritmo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | Diversity@10 | Novelty@10 |
|---|---|---|---|---|---|---|---|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | **`0.9829`** | `0.4482` |
| **Content-Based Filtering** | `0.1615` | `0.3474` | `0.3468` | `0.2744` | `0.6250` | `0.7449` | **`0.8209`** |
| **Collaborative Filtering** *(warm)* | `0.0280` | `0.0622` | `0.0541` | `0.0289` | `0.2000` | `0.9788` | `0.6410` |
| **Hybrid Recommendation (0.65/0.15/0.20)** | **`0.1675`** | **`0.3643`** | **`0.3604`** | **`0.2845`** | **`0.6500`** | `0.7491` | `0.8052` |

Todos os ganhos do Híbrido sobre o Content-Based em `Q_test = 200` têm significância estatística a 95% (IC 95% pareado exclui zero; `NDCG@10` $p = 0,0008$). Prepare o MovieLens opcional via `python scripts/prepare_movielens.py` e execute testes de carga via `python scripts/load_test.py`. Consulte [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) para metodologia, holdout de preferência humana observada e intervalos de confiança.

---

## ⚠️ 11. Limitações Conhecidas

1. **Opcionalidade do Arquivo de Avaliações Individuais:** O arquivo padrão `movies_metadata.csv` contém apenas médias agregadas (`vote_average` e `vote_count`), não possuindo logs individuais de usuários. Sem um arquivo `ratings.csv` compatível com MovieLens configurado em `CINEMATCH_RATINGS_PATH`, o módulo de Filtragem Colaborativa permanece em *Cold-Start Standby* e o motor híbrido redistribui os pesos automaticamente entre Conteúdo e Demográfico.
2. **Predominância do Inglês nas Sinopses:** As sinopses (`overview`) do dataset estão majoritariamente em inglês, embora os tokens estruturados (gêneros, franquias, estúdios e idioma original) operem de forma independente de idioma.

---

## 📄 12. Licença e Créditos

- **Licença:** Distribuído sob a [Licença MIT](LICENSE).
- **Autor:** Desenvolvido por **Warlem Jung**.
- **Atribuição de Dados:** Metadados provenientes do *The Movies Dataset* (TMDB / GroupLens MovieLens).
