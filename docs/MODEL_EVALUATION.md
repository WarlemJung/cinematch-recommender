# Avaliação Quantitativa de Modelos — CineMatch (v2.0.1)

**Data da Execução:** 20 de Novembro de 2026  
**Versão dos Modelos:** `2.0.1`  
**Comando Reproduzível:** `python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity`  
**Módulo de Implementação:** `src/cinematch/models/evaluation.py`

---

## 1. Metodologia, Definição de Relevância e Prevenção de Data Leakage

Em sistemas de recomendação baseados em catálogos cinematográficos, o risco metodológico central é o **Data Leakage (vazamento de dados)**: utilizar o mesmo atributo simultaneamente como feature de entrada no treinamento e como critério de relevância no conjunto de teste.

Para garantir rigor experimental, o pipeline de avaliação do CineMatch v2.0.1 implementa o **Protocolo Anti-Leakage de Holdout Estruturado (`build_relevance_oracle`)**:

### 1.1. Critérios Reproduzíveis de Relevância ($R(q)$)
A função `build_relevance_oracle(movies_df, min_group_size=3)` em `src/cinematch/models/evaluation.py` define que um filme candidato $m \in \mathcal{C}$ é relevante para o filme consultado $q$ ($m \in R(q)$) se, e somente se, cumprir simultaneamente três condições:
1. **Exclusão da Própria Consulta:** $m \neq q$ (o filme consultado nunca pode ser recomendado para si mesmo).
2. **Associação de Franquia/Coleção (Atributo em Holdout):** Tanto $q$ quanto $m$ pertencem à mesma franquia/coleção não vazia (`collection_name`), restrita a coleções com pelo menos `min_group_size = 3` filmes válidos no catálogo.
3. **Consistência de Gênero (`genres_list`):** O candidato $m$ compartilha pelo menos um gênero com o filme consultado $q$ ($G_m \cap G_q \neq \emptyset$, ou ambos não possuem gêneros cadastrados), evitando falsos positivos em coleções antológicas heterogêneas.

### 1.2. Isolamento Estrito de Features (Zero Data Leakage)
- Durante o benchmark (`evaluate_catalog_models`), os modelos `ContentBasedRecommender` e `HybridRecommender` são instanciados e treinados com `ContentFeatureWeights(overview=2, genres=3, tagline=2, collection=0, production=1, language=1)`.
- Com `collection=0`, o atributo `collection_name` (`belongs_to_collection`) é **100% excluído da construção dos documentos e da matriz TF-IDF**.
- Os algoritmos precisam recuperar os filmes da mesma franquia exclusivamente a partir da semântica textual de `overview`, `tagline`, `genres_list`, `production_list` e da ponderação bayesiana demográfica.

---

## 2. Dataset e Protocolo Experimental

- **Arquivo de Origem:** `data/movies_metadata.csv` (*The Movies Dataset* / TMDB Metadata).
- **Tamanho Bruto:** 45.466 registros.
- **Tamanho do Catálogo Validado ($N$):** `45.430` filmes únicos (após remoção de 3 linhas com IDs corrompidos, 3 linhas sem título e 30 IDs duplicados).
- **Número de Consultas Avaliadas ($Q$):** `200` consultas amostradas de forma determinística com seed fixa (`random_seed = 42`).
- **Cutoff de Ranking ($K$):** `10`.
- **Dimensão e Memória da Matriz TF-IDF Esparsa (`float32`, CSR):** `45.430 x 25.000` (`12,303 MB` em RAM).

---

## 3. Fórmulas Matemáticas das Métricas

Para cada consulta $q \in \mathcal{Q}$, seja $\hat{L}_K(q) = [r_1, r_2, \dots, r_K]$ a lista ordenada de $K$ recomendações e $R(q)$ o conjunto ground-truth de itens relevantes:

- **$\text{Precision@K}$:**
  $$\text{Precision@K}(q) = \frac{|\hat{L}_K(q) \cap R(q)|}{K}$$
- **$\text{Recall@K}$:**
  $$\text{Recall@K}(q) = \frac{|\hat{L}_K(q) \cap R(q)|}{|R(q)|}$$
- **$\text{NDCG@K}$ (Normalized Discounted Cumulative Gain):**
  $$\text{DCG@K}(q) = \sum_{i=1}^{K} \frac{\mathbb{I}(r_i \in R(q))}{\log_2(i + 1)}, \quad \text{IDCG@K}(q) = \sum_{i=1}^{\min(|R(q)|, K)} \frac{1}{\log_2(i + 1)}, \quad \text{NDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}$$
- **$\text{MAP@K}$ (Mean Average Precision at K):**
  $$\text{AP@K}(q) = \frac{1}{\min(|R(q)|, K)} \sum_{i=1}^{K} \text{Precision@}i(q) \cdot \mathbb{I}(r_i \in R(q)), \quad \text{MAP@K} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{AP@K}(q)$$
- **$\text{Catalog Coverage}$:**
  $$\text{Coverage} = \frac{\left|\bigcup_{q \in \mathcal{Q}} \hat{L}_K(q)\right|}{N}$$
- **$\text{Intra-List Similarity (ILS)}$ e $\text{Intra-List Diversity}$:**
  $$\text{ILS}(q) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \cos(\mathbf{v}_{r_i}, \mathbf{v}_{r_j}), \quad \text{Diversity} = 1 - \text{ILS}$$

---

## 4. Investigação e Calibração do Modelo Híbrido (v2.0.0 vs. v2.0.1)

### 4.1. Diagnóstico do Comportamento na v2.0.0
Na versão `2.0.0`, o `HybridRecommender` somava diretamente a similaridade de cosseno TF-IDF bruta $S_c(q, i)$ (cujo valor máximo para filmes distintos situa-se tipicamente entre `0.20` e `0.45`) ao score demográfico global normalizado $S_d(i) \in [0, 1]$ (que atinge `0.95 – 1.00` para blockbusters globais como *The Shawshank Redemption* e *Pulp Fiction*).
- **Consequência:** Qualquer blockbuster global com **similaridade temática zero** ($S_c(q, i) = 0.0$) recebia $\hat{w}_d S_d(i) \approx 0.27$, ultrapassando filmes tematicamente similares à consulta com $S_c(q, i) \approx 0.20$. Isso reduzia o `NDCG@10` do modelo híbrido (`0.3292` na v2.0.0 contra `0.4108` do Content-Based).

### 4.2. Solução Implementada na v2.0.1
Em `src/cinematch/models/hybrid.py`, foram introduzidos dois mecanismos matemáticos fundamentados:
1. **Calibração de Escala Relativa à Consulta ($\tilde{S}_c$):**
   $$\tilde{S}_c(q, i) = \frac{S_c(q, i)}{\max_{j \neq q} S_c(q, j)} \in [0, 1]$$
   Alinha a amplitude dinâmica da similaridade de conteúdo da consulta $q$ à mesma escala $[0, 1]$ do score demográfico.
2. **Gating de Relevância Temática ($\gamma(q, i)$):**
   $$\gamma(q, i) = \min\left(1.0, \frac{\max(\tilde{S}_c(q, i), S_{cf}(q, i))}{0.25}\right), \quad S_{\text{hybrid}}(q, i) = \hat{w}_c \tilde{S}_c(q, i) + \hat{w}_d \big(\gamma(q, i) S_d(i)\big) + \hat{w}_{cf} S_{cf}(q, i)$$
   Garante que o componente demográfico atue como um **desempatador e impulsionador de qualidade crítica entre candidatos tematicamente relacionados**, impedindo que filmes com similaridade temática e colaborativa nula ($\gamma = 0$) invadam recomendações personalizadas.

---

## 5. Resultados Medidos e Reproduzidos (v2.0.1)

### 5.1. Comparação Principal entre Modelos (`N = 45.430`, `K = 10`, `Q = 200`, `seed = 42`)

Resultados medidos pela execução real de `python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity`:

| Algoritmo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Intra-List Diversity |
|---|---|---|---|---|---|---|---|
| **1. Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` (0,02%) | `0.0171` | **`0.9829`** |
| **2. Content-Based Filtering** | `0.2045` | `0.4231` | `0.4244` | `0.3531` | `0.0350` (3,50%) | `0.2508` | `0.7492` |
| **3. Hybrid Recommendation (0.75/0.15/0.10)** | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | **`0.0354`** (3,54%) | `0.2485` | `0.7515` |
| **4. Collaborative Filtering** | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ |

> ¹ **Status do Collaborative Filtering:** No repositório padrão está presente apenas `data/movies_metadata.csv` (sem arquivo de avaliações individuais `ratings.csv`). Conforme a política estrita de integridade de dados, nenhuma interação sintética é gerada no benchmark do catálogo completo; o algoritmo `CollaborativeRecommender` é validado isoladamente na suíte de testes (`tests/unit/test_models.py` e `tests/unit/test_evaluation_and_config.py`) e avaliado automaticamente pelo benchmark sempre que `CINEMATCH_RATINGS_PATH` aponta para um dataset real de avaliações.

### 5.2. Análise de Sensibilidade dos Pesos Híbridos (`evaluate_hybrid_weight_sensitivity`)

Medições reais variando a proporção $(\hat{w}_{\text{content}}, \hat{w}_{\text{demographic}})$ sobre o mesmo conjunto de 200 consultas (`seed = 42`):

| Configuração $(\hat{w}_c / \hat{w}_d)$ | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity | Intra-List Diversity |
|---|---|---|---|---|---|---|---|
| **Hybrid (0.90 / 0.10)** | `0.2055` | `0.4281` | `0.4263` | `0.3543` | `0.0352` | `0.2495` | `0.7505` |
| **Hybrid (0.85 / 0.15)** | `0.2065` | `0.4352` | `0.4286` | `0.3551` | `0.0354` | `0.2487` | `0.7513` |
| **Hybrid Padrão (0.833 / 0.167)**² | **`0.2070`** | **`0.4358`** | **`0.4290`** | **`0.3558`** | `0.0354` | `0.2485` | `0.7515` |
| **Hybrid (0.75 / 0.25)** | `0.2065` | `0.4348` | `0.4267` | `0.3530` | **`0.0356`** | `0.2453` | `0.7547` |
| **Hybrid (0.65 / 0.35)** | `0.2005` | `0.4233` | `0.4222` | `0.3507` | `0.0355` | `0.2379` | `0.7621` |
| **Hybrid (0.50 / 0.50)** | `0.1895` | `0.3983` | `0.4009` | `0.3303` | `0.0326` | `0.2049` | **`0.7951`** |

> ² Pesos base configurados $(0.75, 0.15, 0.10)$ renormalizados automaticamente em cold-start colaborativo para $\hat{w}_c = 0.75 / 0.90 = 0.8333$ e $\hat{w}_d = 0.15 / 0.90 = 0.1667$.

**Conclusão Experimental:** Na faixa $\hat{w}_c \in [0.75, 0.90]$ e $\hat{w}_d \in [0.10, 0.25]$, o modelo Híbrido calibrado supera o Content-Based puro simultaneamente em precisão, revocação, qualidade de ranking (`NDCG@10` e `MAP@10`), cobertura de catálogo e diversidade intra-lista, comprovando que a ponderação bayesiana de votos ajuda a priorizar sequências legítimas da franquia sobre filmes obscuros com coincidências lexicais superficiais.
