# Guia de Implantação e Operação em Produção — CineMatch (v2.1.0)

## 1. Pré-requisitos

- **Python:** `3.12` ou `3.13` (`requires-python = ">=3.12"`, validado em matriz CI Python `3.12` e `3.13`).
- **Memória RAM (Medida em Produção Docker):**
  - Boot inicial com `--preload` e 2 workers Gunicorn: **`539,3 MiB`** (`docker stats`).
  - Sob carga concorrente sustentada (após paginação e CoW): **`624,1 MiB`** (redução de **21%** frente aos `790 MiB` da `v2.0.1` graças ao compartilhamento *zero-copy* do `movies_df`).
  - **Limites recomendados de container:** `memory: 1024M` (mínimo `768M`, recomendado `1536M` quando `data/ratings.csv` estiver ativo) e `cpus: 1.0–2.0`.
- **Docker & Docker Compose:** Recomendado para execução em produção com Gunicorn.

---

## 2. Instalação Reproduzível (Fonte Única de Verdade)

As dependências oficiais residem em `pyproject.toml`, sincronizadas 1:1 com `requirements.txt` (verificadas automaticamente por teste unitário):

```bash
# Produção (somente dependências de runtime, igual ao Dockerfile):
pip install --no-cache-dir .

# Desenvolvimento, QA, Testes e Auditoria de Segurança (igual ao GitHub Actions CI):
pip install --no-cache-dir -e ".[dev]"
# Ou equivalentemente:
pip install -r requirements.txt
```

---

## 3. Datasets (`data/movies_metadata.csv` e MovieLens Opcional)

1. **Catálogo Principal (Obrigatório):** Posicione `movies_metadata.csv` (*The Movies Dataset* / Kaggle) em `data/movies_metadata.csv`.
2. **Avaliações Reais MovieLens (Opcional para Collaborative Filtering):**
   - Origem: GroupLens MovieLens `ml-latest-small` (`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`, ~978 KB, uso não comercial/pesquisa conforme licença GroupLens).
   - **Nunca** há download automático na inicialização da API Flask/Gunicorn. Se `data/ratings.csv` estiver ausente, a aplicação inicia normalmente com `CollaborativeRecommender` em *Cold-Start Standby*.
   - Para preparar localmente de forma reproduzível e mapear `movieId -> tmdbId` via `links.csv`:
     ```bash
     python scripts/prepare_movielens.py
     # Ou offline a partir de arquivo local:
     python scripts/prepare_movielens.py --source-zip /caminho/ml-latest-small.zip
     ```

---

## 4. Variáveis de Ambiente (`.env`)

```bash
cp .env.example .env
```

Variáveis principais de produção:
- `CINEMATCH_ENV=production` (bloqueia `debug=True` obrigatoriamente em `AppConfig.validate()`)
- `CINEMATCH_DEBUG=false`
- `CINEMATCH_DATASET_PATH=data/movies_metadata.csv` (ou `/app/data/movies_metadata.csv` no container)
- `CINEMATCH_RATINGS_PATH=` (opcional; se omitido, detecta automaticamente `data/ratings.csv` se existir)
- `CINEMATCH_CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000`
- `CINEMATCH_LOG_JSON=true`
- `GUNICORN_WORKERS=2`, `GUNICORN_TIMEOUT=120`, `GUNICORN_GRACEFUL_TIMEOUT=30`

---

## 5. Execução, Benchmark e Teste de Carga

```bash
# 1. Validar integridade do catálogo
python scripts/validate_dataset.py

# 2. Executar benchmark ML (split estrito train/val/test + bootstrap 95%)
python scripts/train_models.py --k 10 --val-queries 100 --max-queries 200 --seed 42 --sensitivity

# 3. Executar perfilamento de carga, latência (p50/p95/p99) e memória
python scripts/load_test.py --seq 100 --conc 200 --threads 8 --sustained 500
```

---

## 6. Implantação com Docker e Docker Compose

- Imagem base `python:3.13-slim`, usuário não privilegiado `cinematch` (`UID 10001`, `GID 10001`).
- Servidor WSGI **Gunicorn** com `exec gunicorn --preload ... cinematch.wsgi:app`.
- Montagem somente-leitura (`:ro`) de `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`.
- `HEALTHCHECK` verificando HTTP `200 OK` em `http://127.0.0.1:5000/api/health`.

```bash
docker compose up --build -d
docker compose logs -f cinematch
curl http://localhost:5000/api/health
docker compose down
```
