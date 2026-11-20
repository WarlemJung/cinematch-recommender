# Dicionário de Dados — CineMatch

## 1. Origem e Licença do Dataset

- **Dataset Principal:** `movies_metadata.csv` (*The Movies Dataset*, compilado a partir de metadados públicos do TMDB / MovieLens).
- **Licença dos Dados:** CC0: Public Domain / TMDB API Terms of Use (atribuição não comercial preservada na interface e documentação).
- **Dataset Opcional de Avaliações (Collaborative Filtering):** `ratings.csv` ou `ratings_small.csv` (formato GroupLens MovieLens: `userId,movieId,rating,timestamp`).

---

## 2. Schema Bruto (`movies_metadata.csv`) vs. Schema Processado (`LoadedDataset.movies_df`)

| Coluna Original | Coluna Processada | Tipo Final | Regra de Validação e Transformação |
|---|---|---|---|
| `id` | `movie_id` | `int` | Coerção via `pd.to_numeric(..., errors="coerce")`. Registros com IDs não numéricos ou $\le 0$ são descartados. Deduplicado mantendo o registro com maior `vote_count`. |
| `title` | `title` | `str` | Remoção de espaços extremos (`strip()`). Registros com título nulo ou vazio são descartados. |
| — | `normalized_title` | `str` | Campo derivado: normalização Unicode NFKD, conversão para minúsculas e remoção de pontuação para busca e desambiguação. |
| `overview` | `overview` | `str` | Valores `NaN` são convertidos para string vazia `""` e contabilizados em `missing_overview_count` (959 registros). |
| `tagline` | `tagline` | `str` | Valores `NaN` convertidos para `""`. Utilizado com peso configurável no modelo TF-IDF. |
| `genres` | `genres_list` | `list[str]` | Parsing seguro de lista de dicionários (`[{'id': 16, 'name': 'Animation'}]`) para lista de nomes (`['Animation']`), limitado a 6 gêneros por filme. |
| `belongs_to_collection` | `collection_name` | `str` | Extração segura do atributo `name` do dicionário da franquia/coleção (ex.: `'Toy Story Collection'`). |
| `production_companies` | `production_list` | `list[str]` | Extração segura dos até 3 principais estúdios produtores. |
| `release_date` | `release_date` | `str` | Validação ISO `YYYY-MM-DD`. Datas inválidas são convertidas para `""` e registradas em `invalid_release_date_count` (84 registros). |
| — | `release_year` | `int` | Ano extraído de `release_date` (`0` quando a data é ausente/inválida). |
| `vote_average` | `vote_average` | `float` | Validado no intervalo $[0.0, 10.0]$, arredondado para 2 casas decimais. |
| `vote_count` | `vote_count` | `int` | Validado como inteiro não negativo ($\ge 0$). |
| `popularity` | `popularity` | `float` | Coerção numérica com fallback `0.0` e truncamento em valores $\ge 0.0$. |
| `runtime` | `runtime` | `float` | Duração em minutos ($\ge 0.0$). |
| `original_language` | `original_language` | `str` | Código ISO 639-1 (ex.: `'en'`, `'fr'`, `'pt'`), fallback `'en'`. |
| `imdb_id` | `imdb_id` | `str` | Identificador IMDB (ex.: `'tt0114709'`). |
| `poster_path` | `poster_path` | `str \| None` | Validado contra regex `^/[A-Za-z0-9_\-]+\.(jpg\|jpeg\|png\|webp)$`. Valores inválidos tornam-se `None`. |

---

## 3. Métricas Reais de Qualidade do Dataset (`DataQualityReport`)

Obtidas via `python scripts/validate_dataset.py` no arquivo `movies_metadata.csv`:

- **`total_raw_rows`**: `45.466`
- **`valid_rows`**: `45.430`
- **`dropped_corrupted_id_rows`**: `3` (linhas `19730`, `29503` e `35587` onde `id` contém datas `'1997-08-20'`, `'2012-09-29'`, `'2014-01-01'`)
- **`dropped_missing_title_rows`**: `3`
- **`dropped_invalid_rating_rows`**: `0` (as 6 linhas com `vote_average` nulo já foram eliminadas nas etapas de ID corrompido e título ausente)
- **`dropped_duplicate_id_rows`**: `30`
- **`duplicate_title_count`**: `5.753` filmes compartilham título normalizado com ao menos outro registro (homônimos e remakes de anos distintos)
- **`missing_overview_count`**: `959`
- **`invalid_release_date_count`**: `84`
- **`unique_genres_count`**: `20`
