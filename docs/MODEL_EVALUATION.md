# Avaliação Quantitativa de Modelos — CineMatch

**Data da Execução:** 20 de Novembro de 2026  
**Script Reproduzível:** `python scripts/train_models.py --k 10 --max-queries 200`  
**Módulo de Implementação:** `src/cinematch/models/evaluation.py`

---

## 1. Metodologia e Estratégia Anti-Leakage

Em sistemas de recomendação baseados em catálogos de metadados sem histórico de cliques em tempo real, o maior risco metodológico é o **Data Leakage (vazamento de dados)**: utilizar o mesmo atributo (ex.: nome da franquia ou gênero) simultaneamente para treinar o modelo e para definir o conjunto de itens relevantes de teste $R(q)$.

Para garantir rigor científico e ausência total de data leakage, o pipeline de avaliação do CineMatch implementa um **Protocolo de Holdout de Atributo Estruturado (Feature-Holdout Oracle)**:

1. **Oráculo de Relevância Ground-Truth $R(q)$:**
   - Utilizamos o agrupamento real de franquias e coleções cinematográficas (`belongs_to_collection` / `collection_name`) presente no dataset `movies_metadata.csv`, considerando todas as coleções com $\ge 3$ filmes válidos no catálogo.
   - Para um filme consultado $q$ pertencente à coleção $C_q$, o conjunto de itens verdadeiramente relevantes no teste é definido por:
     $$R(q) = \{ m \in C_q \mid m \neq q \}$$
2. **Isolamento Estrito de Features (Zero Data Leakage):**
   - Durante o treinamento dos modelos avaliados (`Content-Based Filtering` e `Hybrid Recommendation`), o peso da feature de coleção é zerado (`ContentFeatureWeights(collection=0)`).
   - Dessa forma, a matriz TF-IDF esparsa **não recebe qualquer informação sobre `belongs_to_collection`**. O algoritmo precisa recuperar as sequências e filmes da mesma franquia exclusivamente a partir da semântica das sinopses (`overview`), `tagline`, gêneros (`genres`) e estúdios (`production_companies`).
3. **Amostragem Reproduzível de Consultas:**
   - Foram avaliadas $Q = 200$ consultas amostradas de forma determinística (`random_state=42`) sobre o catálogo completo validado de $N = 45.430$ filmes, com cutoff $K = 10$.

---

## 2. Dataset Utilizado

- **Fonte:** `movies_metadata.csv` (*The Movies Dataset* / TMDB Metadata).
- **Total bruto de registros:** 45.466 linhas.
- **Registros válidos após limpeza e deduplicação por `movie_id`:** 45.430 filmes únicos.
- **Registros removidos na higienização:** 36 (3 linhas com `id` corrompido contendo datas, 3 linhas sem título e 30 IDs duplicados).
- **Dimensão da matriz TF-IDF esparsa (`float32`, CSR):** $45.430 \times 25.000$ (**12,303 MB em RAM** vs. 16.511 MB que seriam exigidos por uma matriz densa quadrática $N \times N$).

---

## 3. Definição Matemática das Métricas

Para cada consulta $q \in \mathcal{Q}$, seja $\hat{L}_K(q) = [r_1, r_2, \dots, r_K]$ a lista ordenada de $K$ recomendações e $R(q)$ o conjunto ground-truth de itens relevantes:

- **$\text{Precision@K}$:** Fração dos $K$ itens recomendados que são relevantes:
  $$\text{Precision@K}(q) = \frac{|\hat{L}_K(q) \cap R(q)|}{K}$$
- **$\text{Recall@K}$:** Fração do total de itens relevantes recuperados no top-$K$:
  $$\text{Recall@K}(q) = \frac{|\hat{L}_K(q) \cap R(q)|}{|R(q)|}$$
- **$\text{NDCG@K}$ (Normalized Discounted Cumulative Gain):** Avalia a qualidade do ranking penalizando logaritmicamente itens relevantes posicionados mais abaixo na lista:
  $$\text{DCG@K}(q) = \sum_{i=1}^{K} \frac{\mathbb{I}(r_i \in R(q))}{\log_2(i + 1)}, \quad \text{NDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}$$
- **$\text{MAP@K}$ (Mean Average Precision at K):** Média da precisão calculada em cada posição onde ocorre um acerto relevante:
  $$\text{AP@K}(q) = \frac{1}{\min(|R(q)|, K)} \sum_{i=1}^{K} \text{Precision@}i(q) \cdot \mathbb{I}(r_i \in R(q))$$
- **$\text{Catalog Coverage}$:** Proporção do catálogo total $N$ coberta pela união de todas as listas recomendadas:
  $$\text{Coverage} = \frac{\left|\bigcup_{q \in \mathcal{Q}} \hat{L}_K(q)\right|}{N}$$
- **$\text{Intra-List Similarity (ILS)}$:** Similaridade média de cosseno TF-IDF entre todos os pares $(i, j)$ dentro da mesma lista recomendada $\hat{L}_K(q)$:
  $$\text{ILS}(q) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \cos(\mathbf{v}_{r_i}, \mathbf{v}_{r_j})$$
- **$\text{Recommendation Diversity}$:** Complemento da similaridade intra-lista:
  $$\text{Diversity} = 1 - \text{ILS}$$

---

## 4. Resultados Medidos Reais

### 4.1. Benchmark no Catálogo Completo (`N = 45.430`, `K = 10`, `Q = 200` consultas)

Resultados obtidos diretamente pela execução de `python scripts/train_models.py --k 10 --max-queries 200`:

| Algoritmo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage | Intra-List Similarity (ILS) | Recommendation Diversity |
|---|---|---|---|---|---|---|---|
| **1. Demographic Filtering** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0002` (0,02%) | `0.0171` | `0.9829` |
| **2. Content-Based Filtering** | **`0.2050`** | **`0.4156`** | **`0.4108`** | **`0.3332`** | **`0.0359`** (3,59%) | `0.2449` | `0.7551` |
| **3. Hybrid Recommendation** | `0.1510` | `0.2970` | `0.3292` | `0.2705` | `0.0210` (2,10%) | `0.1363` | **`0.8637`** |
| **4. Collaborative Filtering** | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ | *Standby*¹ |

> ¹ **Nota sobre o Collaborative Filtering no Dataset Padrão:** O repositório original inclui apenas `movies_metadata.csv` (que possui médias agregadas `vote_average` e `vote_count`, mas **não contém logs individuais de avaliações por usuário `userId, movieId, rating`**). Seguindo a diretriz estrita de engenharia (*"Não inventar avaliações de usuários; não tratar dados inexistentes como interações reais"*), o módulo `CollaborativeRecommender` entra automaticamente em modo *Cold-Start Standby* quando nenhum arquivo `ratings.csv` real está presente, e o `HybridRecommender` redistribui os pesos entre Conteúdo e Demográfico.

### 4.2. Avaliação Isolada do Módulo Collaborative Filtering (Matriz Usuário-Item)

Para validar matematicamente o algoritmo `CollaborativeRecommender` (TruncatedSVD + similaridade de cosseno centrada na média do usuário), a suíte de testes (`tests/unit/test_models.py`) executa validação com matrizes esparsas de interações usuário-item (`user_id`, `movie_id`, `rating`), verificando:
- Recuperação de itens co-avaliados positivamente pelos mesmos clusters de usuários ($\text{NDCG@5} > 0.85$ em grupos de preferência bem definidos).
- Isolamento de cold-start (`has_item(unrated_movie_id) == False` e retorno `None` sem fabricar scores).

---

## 5. Análise Comparativa entre os Algoritmos

1. **Demographic Filtering (Weighted Rating):**
   - **Pontos Fortes:** Excelente para a página inicial (*Top Charts / Trending*) e para resolver o problema de *cold-start* de novos usuários sem filme de referência. Apresenta altíssima diversidade temática (`0.9829`).
   - **Limitações:** Por ser um ranking global estático (não personalizado por filme consultado), sua cobertura de catálogo é mínima (`0.0002`, recomendando sempre os mesmos top-$K$ clássicos globais) e sua precisão para recuperar filmes semanticamente relacionados a uma busca específica é `0.0000`.
2. **Content-Based Filtering (TF-IDF Esparso Ponderado):**
   - **Pontos Fortes:** Mesmo com o atributo `belongs_to_collection` ocultado durante o treino (anti-leakage), alcançou **`Recall@10 = 41,56%`** e **`NDCG@10 = 0.4108`**, recuperando mais de 40% dos filmes da mesma franquia apenas pela similaridade textual da sinopse, tagline, gêneros e estúdios. Também obteve a maior cobertura de catálogo (`3,59%` do catálogo de 45.430 filmes em apenas 200 consultas).
   - **Limitações:** Pode recomendar sequências ou filmes com sinopses muito parecidas mas de baixa qualidade crítica (baixo `vote_average` ou poucos votos), além de apresentar maior similaridade intra-lista (`ILS = 0.2449`).
3. **Hybrid Recommendation (Combinação Ponderada Conteúdo + Demográfico + Colaborativo):**
   - **Pontos Fortes:** Equilibra relevância semântica (`Recall@10 = 29,70%`, `NDCG@10 = 0.3292`) com aclamação crítica (Weighted Rating) e aumenta significativamente a **Diversidade das Recomendações** (`0.8637` vs. `0.7551` do Content-Based puro), reduzindo a redundância intra-lista (`ILS` cai de `0.2449` para `0.1363`) e fornecendo explicações legíveis item a item.

---

## 6. Limitações Conhecidas

1. **Ausência de `ratings.csv` no repositório base:** A filtragem colaborativa requer um arquivo de interações reais (`userId,movieId,rating`, como o `ratings_small.csv` do dataset público MovieLens/Kaggle configurado via `CINEMATCH_RATINGS_PATH`). Sem esse arquivo, o sistema opera de forma transparente como um híbrido *Content-Based + Demographic*.
2. **Viés de Idioma nas Sinopses:** As sinopses (`overview`) em `movies_metadata.csv` estão predominantemente em inglês (`stop_words="english"`), embora o sistema inclua tokens estruturados de gênero, coleção, estúdio e idioma original que funcionam independentemente do idioma da sinopse.
