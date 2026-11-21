# Avaliação Quantitativa de Modelos — CineMatch (v2.1.0)

**Data da Execução:** 20 de Novembro de 2026  
**Versão dos Modelos:** `2.1.0`  
**Comando Reproduzível:** `python scripts/train_models.py --k 10 --val-queries 100 --max-queries 200 --seed 42 --sensitivity`  
**Módulo de Implementação:** `src/cinematch/models/evaluation.py`

---

## 1. Metodologia v2.1.0: Separação Estrita Treino / Validação / Teste e Eliminação de Circularidade

### 1.1. Revisão do Oráculo de Relevância e Fim da Sobreposição de `genres_list`
Na versão `2.0.1`, `build_relevance_oracle()` verificava pertencimento à mesma coleção (`collection_name`, ocultada do treino) **e** interseção não vazia de `genres_list` (que integrava o TF-IDF com peso `3`). Na **v2.1.0**, `build_relevance_oracle(movies_df, min_group_size=3, require_genre_overlap=False)` adota por padrão `require_genre_overlap=False`:

$$m \in R_{\text{meta}}(q) \iff (m \neq q) \land \big(\text{collection}(m) = \text{collection}(q) \neq \emptyset\big) \land \big(|\text{collection}(q)| \ge 3\big)$$

- **Zero Vazamento / Zero Circularidade:** `ContentBasedRecommender` e `HybridRecommender` são ajustados com `ContentFeatureWeights(collection=0)` — a coluna `collection_name` nunca entra nos documentos TF-IDF, e o oráculo de metadados não depende de `genres_list`.
- **Distinção Metodológica Explícita:**
  - *Relevância por Metadados ($R_{\text{meta}}(q)$):* proxy estrutural baseado em franquias (`2.741` consultas elegíveis no catálogo de `45.430` filmes); não equivale a preferência humana observada.
  - *Preferência Humana Observada ($R_{\text{user}}(u)$):* avaliada separadamente sobre interações reais do **MovieLens `ml-latest-small`** com divisão temporal cronológica por usuário (`split_ratings_train_val_test()`).

### 1.2. Particionamento Estrito em Três Conjuntos Disjuntos
1. **Conjunto de Treino:**
   - Metadados do catálogo (`collection=0` na avaliação).
   - Quando `data/ratings.csv` está presente: **primeiros 70% cronológicos por usuário** ($\mathcal{R}_{\text{train}}$, ordenados por `timestamp` ascendente) alimentam `CollaborativeRecommender.fit()`. Avaliações de validação e teste jamais são vistas no treinamento.
2. **Conjunto de Validação ($\mathcal{Q}_{\text{val}} = 100$ consultas, $\mathcal{R}_{\text{val}} \approx 15\%$):**
   - Usado exclusivamente por `calibrate_hybrid_weights()` e `evaluate_hybrid_weight_sensitivity()` para selecionar os pesos híbridos que maximizam `NDCG@10` na validação.
3. **Conjunto de Teste em Holdout ($\mathcal{Q}_{\text{test}} = 200$ consultas, $\mathcal{R}_{\text{test}} \approx 15\%$):**
   - Totalmente isolado (`val_test_overlap = 0`), executado uma única vez ao final para comparar modelos e calcular intervalos de confiança pareados via bootstrap ($B = 5.000$, `seed = 42`).

---

## 2. Fórmulas Matemáticas das Métricas (com Denominadores Explícitos)

Seja $\hat{L}_K(q) = [r_1, \dots, r_K]$ o ranking top-$K$ ($K = 10$) para a consulta $q \in \mathcal{Q}_{\text{test}}$ ($|\mathcal{Q}_{\text{test}}| = 200$) e $R(q)$ o conjunto relevante:

- **$\text{Precision@K}(q) = \dfrac{|\hat{L}_K(q) \cap R(q)|}{K}$** (denominador fixo $K$).
- **$\text{Recall@K}(q) = \dfrac{|\hat{L}_K(q) \cap R(q)|}{|R(q)|}$** (denominador $|R(q)|$).
- **$\text{HitRate@K}(q) = \mathbb{I}\big(|\hat{L}_K(q) \cap R(q)| \ge 1\big) \in \{0, 1\}$** (média sobre $|\mathcal{Q}_{\text{test}}|$).
- **$\text{NDCG@K}(q) = \dfrac{\sum_{i=1}^{K} \frac{\mathbb{I}(r_i \in R(q))}{\log_2(i+1)}}{\sum_{i=1}^{\min(|R(q)|, K)} \frac{1}{\log_2(i+1)}}$** (normalizado por $\text{IDCG@K}$).
- **$\text{MAP@K} = \dfrac{1}{|\mathcal{Q}_{\text{test}}|} \sum_{q} \dfrac{1}{\min(|R(q)|, K)} \sum_{i=1}^{K} \text{Precision@}i(q)\cdot\mathbb{I}(r_i \in R(q))$**.
- **$\text{Catalog Coverage} = \dfrac{\left|\bigcup_{q \in \mathcal{Q}_{\text{test}}} \hat{L}_K(q)\right|}{N}$**, com $N = 45.430$.
- **$\text{Diversity@K} = 1 - \text{ILS@K}$**, onde $\text{ILS}(q) = \dfrac{2}{K(K-1)}\sum_{1 \le i < j \le K} \cos(\mathbf{v}_{r_i}, \mathbf{v}_{r_j})$.
- **$\text{Novelty@K}(q) = \dfrac{1}{K}\sum_{r \in \hat{L}_K(q)} \dfrac{-\log_2 p(r)}{\log_2 Z} \in [0, 1]$**, com $p(r) = \dfrac{\text{vote\_count}_r + 1}{Z}$ e $Z = \sum_{j=1}^{N} (\text{vote\_count}_j + 1)$.

---

## 3. Resultados Reais Medidos (`K = 10`, `Q_val = 100`, `Q_test = 200`, `seed = 42`)

### 3.1. Benchmark de Metadados (Holdout Estrito de Franquia, Sem Sobreposição de Gênero)

#### Cenário A — Com MovieLens Ativo (`data/ratings.csv`: 610 usuários, 6.216 filmes no split de treino)
Pesos calibrados exclusivamente em $\mathcal{Q}_{\text{val}}$: **`content=0.65, demographic=0.15, collaborative=0.20`**.

| Algoritmo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | Coverage | Diversity@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` | **`0.9829`** | `0.4482` |
| **Content-Based Filtering** | `0.1615` | `0.3474` | `0.3468` | `0.2744` | `0.6250` | `0.0360` | `0.7449` | `0.8209` |
| **Collaborative Filtering** *(50 consultas warm)* | `0.0280` | `0.0622` | `0.0541` | `0.0289` | `0.2000` | `0.0103` | `0.9788` | `0.6410` |
| **Hybrid Recommendation (0.65/0.15/0.20)** | **`0.1675`** | **`0.3643`** | **`0.3604`** | **`0.2845`** | **`0.6500`** | **`0.0363`** | `0.7491` | `0.8052` |

**Análise Estatística Pareada no Conjunto de Teste (`Hybrid 3-way` vs. `Content-Based`, $B = 5.000$, $Q_{\text{test}} = 200$):**

| Métrica | $\Delta$ Médio (`Hybrid - CB`) | IC 95% Bootstrap | Vitórias / Empates / Derrotas | $p$-valor Bilateral | Significativo ($\alpha=0.05$)? |
|---|---:|---:|---:|---:|---|
| **Precision@10** | **`+0.0060`** | **`[+0.0010, +0.0115]`** | `13 / 182 / 5` | `0.0108` | **Sim** |
| **Recall@10** | **`+0.0168`** | **`[+0.0050, +0.0318]`** | `13 / 182 / 5` | `0.0024` | **Sim** |
| **NDCG@10** | **`+0.0136`** | **`[+0.0053, +0.0228]`** | `31 / 153 / 16` | `0.0008` | **Sim** |
| **MAP@10** | **`+0.0102`** | **`[+0.0033, +0.0176]`** | `31 / 153 / 16` | `0.0012` | **Sim** |
| **HitRate@10** | **`+0.0250`** | **`[+0.0050, +0.0500]`** | `5 / 195 / 0` | `0.0104` | **Sim** |
| **Diversity@10** | **`+0.0042`** | **`[+0.0016, +0.0071]`** | `75 / 88 / 37` | `0.0004` | **Sim** |

**Estratificação Cold-Start vs. Warm (`collaborative_coverage`):**
- **Consultas Cold-Start ($Q = 150$, sem cobertura colaborativa):** `Hybrid` mantém paridade segura sem degradação (`NDCG@10 = 0.3354` vs. `0.3323` do Content-Based; `HitRate@10 = 0.6133` vs. `0.6000`).
- **Consultas Warm ($Q = 50$, com cobertura no MovieLens):** `Hybrid` obtém salto substancial sobre Content-Based (`Precision@10 = 0.2060` vs. `0.1860`, `Recall@10 = 0.4562` vs. `0.3962`, `NDCG@10 = 0.4355` vs. `0.3902`, `HitRate@10 = 0.7600` vs. `0.7000`).

#### Cenário B — Modo Cold-Start Standby (Sem `ratings.csv`, apenas Metadados)
Pesos calibrados em $\mathcal{Q}_{\text{val}}$: `content=0.65, demographic=0.35, collaborative=0.0`.
- `Content-Based`: `Precision@10 = 0.1615`, `Recall@10 = 0.3474`, `NDCG@10 = 0.3468`, `HitRate@10 = 0.6250`.
- `Hybrid (0.65/0.35/0.0)`: `Precision@10 = 0.1645` (IC 95% `[-0.0025, +0.0085]`), `Recall@10 = 0.3577` (IC 95% `[-0.0041, +0.0270]`), `NDCG@10 = 0.3532` (IC 95% `[-0.0025, +0.0163]`), `HitRate@10 = 0.6450`, `Diversity@10 = 0.7523` (IC 95% `[+0.0042, +0.0108]`, $p < 0.001$). Em standby puro, o ganho de diversidade é estatisticamente significativo, enquanto o ganho médio de ranking não atinge significância de 95% no teste isolado (evidenciando que o sinal colaborativo real é o principal impulsionador estatístico do modelo Híbrido).

---

### 3.2. Avaliação sobre Preferências Humanas Observadas (Holdout Temporal MovieLens)
Avaliação conduzida sobre **`200` usuários** do `ml-latest-small` no conjunto de teste temporal ($\mathcal{R}_{\text{test}}$, últimos 15% cronológicos por usuário com `rating >= 4.0`, semente sendo o item favorito do usuário em $\mathcal{R}_{\text{train}}$):

| Algoritmo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | Coverage | Diversity@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Content-Based Filtering** | `0.0015` | `0.0017` | `0.0023` | `0.0008` | `0.0150` | `0.0280` | `0.7390` | `0.8098` |
| **Collaborative Filtering** | **`0.0140`** | **`0.0159`** | **`0.0187`** | **`0.0074`** | **`0.1200`** | `0.0264` | **`0.9803`** | `0.6053` |
| **Hybrid Recommendation** | `0.0025` | `0.0033` | `0.0036` | `0.0013` | `0.0250` | **`0.0281`** | `0.7505` | `0.7718` |
| **Demographic Filtering** *(Baseline Popularidade)* | `0.0350` | `0.0416` | `0.0567` | `0.0275` | `0.2950` | `0.0002` | `0.9826` | `0.4484` |

**Bootstrap Pareado (`Collaborative` vs. `Content-Based`, $Q = 200$ usuários):**
- **HitRate@10:** $\Delta = +0.1050$ (**8x superior**, IC 95% `[+0.0650, +0.1550]`, $p = 0.0000$, 22 vitórias vs. 1 derrota).
- **NDCG@10:** $\Delta = +0.0164$ (IC 95% `[+0.0089, +0.0252]`, $p = 0.0000$).
- **Limitação Documentada:** No protocolo *item-to-item* a partir de um único filme semente, o baseline `Demographic` apresenta maior taxa de acerto bruto por viés de popularidade clássica nos usuários do MovieLens (`Coverage = 0.0002`, `Novelty = 0.4484`), enquanto `Collaborative Filtering` combina personalização real com cobertura de catálogo 132x maior (`0.0264` vs. `0.0002`) e diversidade máxima (`0.9803`).
