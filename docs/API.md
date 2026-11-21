# Documentação da API REST — CineMatch

Todos os endpoints da API retornam respostas JSON padronizadas com codificação UTF-8 e cabeçalhos de segurança HTTP.

---

## 1. Resumo dos Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Interface web interativa do CineMatch |
| `GET` | `/api/health` | Diagnóstico de saúde, memória esparsa TF-IDF e qualidade dos dados |
| `GET` | `/api/trending` | Filmes populares e aclamados (Demographic Weighted Rating) com paginação e filtros |
| `GET` | `/api/genres` | Lista de gêneros disponíveis no catálogo e contagem de filmes |
| `GET` | `/api/movies/search` | Pesquisa de filmes por título com filtros e paginação |
| `GET` | `/api/movies/<movie_id>` | Metadados completos de um filme específico por ID |
| `GET` | `/api/recommend/<movie_id>` | Recomendações baseadas em conteúdo (TF-IDF esparso) por ID |
| `GET` | `/api/recommend` | Recomendações baseadas em conteúdo por query (`?title=...` ou `?movie_id=...`) |
| `GET` | `/api/recommendations/hybrid/<movie_id>` | Recomendações híbridas explicáveis por ID |
| `GET` | `/api/recommendations/hybrid` | Recomendações híbridas explicáveis por query (`?title=...` ou `?movie_id=...`) |

---

## 2. Especificação Detalhada

### 2.1. `GET /api/health`
Retorna o estado operacional da aplicação, métricas de memória da matriz esparsa e o relatório de qualidade do dataset.

**Resposta `200 OK`:**
```json
{
  "status": "healthy",
  "service": "CineMatch Hybrid Recommendation Engine",
  "version": "2.1.0",
  "catalog_size": 45430,
  "models": {
    "demographic_fitted": true,
    "content_based_fitted": true,
    "collaborative_available": false,
    "collaborative_covered_movies": 0,
    "collaborative_distinct_users": 0,
    "hybrid_fitted": true,
    "tfidf_matrix_shape": [45430, 25000],
    "tfidf_sparse_memory_mb": 12.303
  },
  "data_quality": {
    "total_raw_rows": 45466,
    "valid_rows": 45430,
    "sampled_rows": 45430,
    "dropped_corrupted_id_rows": 3,
    "dropped_missing_title_rows": 3,
    "dropped_invalid_rating_rows": 0,
    "dropped_duplicate_id_rows": 30,
    "duplicate_title_count": 5753,
    "missing_overview_count": 959,
    "invalid_release_date_count": 84,
    "unique_genres_count": 20
  }
}
```

---

### 2.2. `GET /api/trending`
Retorna os filmes com maior pontuação segundo a fórmula bayesiana Weighted Rating (IMDB).

**Parâmetros de Query (opcionais):**
- `top_n` (`int`, 1..200, padrão `12`): Limite total de itens do ranking.
- `page` (`int`, >= 1, padrão `1`): Página atual.
- `page_size` (`int`, 1..100, opcional): Quantidade de itens por página.
- `genre` (`str`, opcional): Filtro exato por nome de gênero (ex.: `Animation`, `Science Fiction`).
- `min_year` (`int`, 1880..2100, opcional): Ano mínimo de lançamento.
- `max_year` (`int`, 1880..2100, opcional): Ano máximo de lançamento.
- `min_rating` (`float`, 0..10, opcional): Nota média mínima (`vote_average`).

**Resposta `200 OK`:**
```json
{
  "items": [
    {
      "movie_id": 19404,
      "title": "Dilwale Dulhania Le Jayenge",
      "release_year": 1995,
      "release_date": "1995-10-20",
      "vote_average": 9.1,
      "vote_count": 661,
      "popularity": 34.46,
      "genres": ["Comedy", "Drama", "Romance"],
      "overview": "Raj is a rich, carefree, happy-go-lucky second generation NRI...",
      "poster_path": "/2CAL2433ZeIihfX1Hb2139CX0pW.jpg",
      "poster_url": "https://image.tmdb.org/t/p/w500/2CAL2433ZeIihfX1Hb2139CX0pW.jpg",
      "score": 8.5321
    }
  ],
  "page": 1,
  "page_size": 12,
  "total": 12,
  "total_pages": 1,
  "parameters": {
    "mean_vote_c": 5.6182,
    "min_votes_m": 160.0,
    "quantile": 0.9,
    "qualifying_count": 4555
  }
}
```

---

### 2.3. `GET /api/movies/search`
Pesquisa filmes por título normalizado (com priorização de match exato, prefixo e substring, desambiguando títulos duplicados por volume de votos) e filtros combinados.

**Parâmetros de Query:**
- `q` (`str`, opcional, máx. 200 caracteres): Termo de busca.
- `page` (`int`, padrão `1`), `page_size` (`int`, padrão `20`, máx. `100`).
- `genre`, `min_year`, `max_year`, `min_rating`.

---

### 2.4. `GET /api/movies/<movie_id>`
Retorna os detalhes enriquecidos de um filme específico.

**Resposta `200 OK`:**
```json
{
  "movie_id": 862,
  "title": "Toy Story",
  "release_year": 1995,
  "release_date": "1995-10-30",
  "vote_average": 7.7,
  "vote_count": 5415,
  "popularity": 21.95,
  "genres": ["Animation", "Comedy", "Family"],
  "overview": "Led by Woody, Andy's toys live happily in his room...",
  "tagline": "",
  "collection_name": "Toy Story Collection",
  "production_companies": ["Pixar Animation Studios"],
  "original_language": "en",
  "runtime": 81.0,
  "imdb_id": "tt0114709",
  "poster_path": "/rhIRbceoE9lR4veEXuwCC2wARtG.jpg",
  "poster_url": "https://image.tmdb.org/t/p/w500/rhIRbceoE9lR4veEXuwCC2wARtG.jpg"
}
```

---

### 2.5. `GET /api/recommend/<movie_id>` e `GET /api/recommend?title=...`
Retorna recomendações baseadas em conteúdo (TF-IDF esparso + similaridade de cosseno sob demanda).

**Parâmetros de Query:**
- `top_n` (`int`, 1..50, padrão `12`).

---

### 2.6. `GET /api/recommendations/hybrid/<movie_id>` e `GET /api/recommendations/hybrid?title=...`
Retorna recomendações do motor híbrido com scores individuais, pesos ativos normalizados e explicação gerada dinamicamente.

**Parâmetros de Query:**
- `top_n` (`int`, 1..50, padrão `12`).
- `genre`, `min_year`, `max_year`, `min_rating` (filtros pós-inferência opcionais).

**Resposta `200 OK`:**
```json
{
  "algorithm": "hybrid_weighted_ensemble",
  "collaborative_active": false,
  "active_weights": {
    "content": 0.8333,
    "demographic": 0.1667,
    "collaborative": 0.0
  },
  "source_movie": {
    "movie_id": 862,
    "title": "Toy Story",
    "release_year": 1995
  },
  "total": 2,
  "items": [
    {
      "movie_id": 10193,
      "title": "Toy Story 3",
      "release_year": 2010,
      "release_date": "2010-06-16",
      "vote_average": 7.6,
      "vote_count": 4710,
      "genres": ["Animation", "Family", "Comedy"],
      "overview": "Woody, Buzz, and the rest of Andy's toys...",
      "poster_path": "/mMltbSxwEdNE4Cv8QYLpzkHWTDo.jpg",
      "poster_url": "https://image.tmdb.org/t/p/w500/mMltbSxwEdNE4Cv8QYLpzkHWTDo.jpg",
      "final_score": 0.6512,
      "individual_scores": {
        "content": 0.5621,
        "demographic": 0.8829,
        "collaborative": null
      },
      "active_weights": {
        "content": 0.8333,
        "demographic": 0.1667,
        "collaborative": 0.0
      },
      "explanation": "Recomendado porque faz parte da mesma franquia (Toy Story Collection), apresenta alta similaridade temática e narrativa com o filme selecionado, compartilha os gêneros Animation, Family, Comedy, foi produzido pelo mesmo estúdio (Pixar Animation Studios) e possui excelente avaliação geral (7.6/10 em 4710 votos)."
    }
  ]
}
```

---

## 3. Padrão de Respostas de Erro

A API nunca expõe stack traces ao cliente. Todos os erros seguem o schema `ErrorResponseSchema`:

| HTTP Status | `error_code` | Cenário |
|---|---|---|
| `400 Bad Request` | `VALIDATION_ERROR` / `INVALID_PARAMETER` | Parâmetros fora dos limites (`top_n > 50`, `page < 1`, `min_year > max_year`) |
| `404 Not Found` | `MOVIE_NOT_FOUND` | `movie_id` ou `title` inexistente no catálogo |
| `422 Unprocessable Entity` | `DATA_VALIDATION_ERROR` | Falha de validação de schema do dataset |
| `503 Service Unavailable` | `SYSTEM_NOT_INITIALIZED` / `DATASET_NOT_FOUND` | Dataset ausente no disco |
| `500 Internal Server Error` | `INTERNAL_SERVER_ERROR` | Erro interno inesperado (registrado apenas nos logs estruturados do servidor) |
