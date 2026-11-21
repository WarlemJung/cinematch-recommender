# Guia de Implantação e Operação em Produção — CineMatch (v2.0.1)

## 1. Pré-requisitos

- **Python:** 3.11, 3.12 ou 3.13 (validado em matriz CI Python 3.11, 3.12 e 3.13)
- **Memória RAM:** Mínimo 512 MB (a matriz TF-IDF esparsa consome ~12,3 MB para 45.430 filmes; com Gunicorn `--preload` e 2 workers, o processo compartilha páginas de memória via *Copy-on-Write*, mantendo o consumo total em ~210–240 MB)
- **Docker & Docker Compose:** Recomendado para execução em produção com Gunicorn

---

## 2. Convenção Padronizada do Dataset (`data/movies_metadata.csv`)

O CineMatch adota a convenção única `data/movies_metadata.csv` tanto para execução local quanto para montagem de volume no Docker:

1. Obtenha o arquivo `movies_metadata.csv` a partir do dataset público [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv).
2. Posicione o arquivo dentro do diretório `data/` do repositório:
   ```text
   cinematch-recommender/
   └── data/
       └── movies_metadata.csv
   ```
3. *(Opcional)* Para habilitar a Filtragem Colaborativa com avaliações reais de usuários, posicione `ratings.csv` ou `ratings_small.csv` em `data/ratings.csv` e defina `CINEMATCH_RATINGS_PATH=data/ratings.csv`.
4. **Nota de Segurança:** A aplicação nunca faz download automático de datasets externos sem autorização, e arquivos `.csv` dentro de `data/` são ignorados pelo Git e pelo `.dockerignore` para evitar imagens inchadas.

---

## 3. Configuração de Ambiente (`.env`)

1. Copie o arquivo de exemplo:
   ```bash
   cp .env.example .env
   ```
2. Variáveis principais de produção:
   - `CINEMATCH_ENV=production` (bloqueia `debug=True` obrigatoriamente)
   - `CINEMATCH_DEBUG=false`
   - `CINEMATCH_DATASET_PATH=data/movies_metadata.csv` (ou `/app/data/movies_metadata.csv` no container)
   - `CINEMATCH_CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000`
   - `CINEMATCH_LOG_JSON=true` (ativa logs estruturados em JSON)
   - `GUNICORN_WORKERS=2` (número conservador de workers WSGI)
   - `GUNICORN_TIMEOUT=120` (timeout de worker em segundos)
   - `GUNICORN_GRACEFUL_TIMEOUT=30` (tempo para encerramento limpo de requisições em andamento)

---

## 4. Execução Local

### 4.1. Instalação e Validação do Dataset
```bash
pip install -r requirements.txt
python scripts/validate_dataset.py
```

### 4.2. Benchmark Quantitativo Reproduzível e Análise de Sensibilidade
```bash
python scripts/train_models.py --k 10 --max-queries 200 --seed 42 --sensitivity
```

### 4.3. Inicialização do Servidor WSGI (Linux / macOS / Produção)
```bash
PYTHONPATH=src gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --timeout 120 \
  --graceful-timeout 30 \
  --preload \
  cinematch.wsgi:app
```

### 4.4. Inicialização Local de Desenvolvimento (Windows / macOS / Linux)
```bash
python -m cinematch.main
# Ou via entrypoint na raiz:
python app.py
```

---

## 5. Implantação com Docker e Docker Compose

O `Dockerfile` e o `docker-compose.yml` foram configurados para produção:
- Imagem base `python:3.13-slim`.
- Servidor WSGI **Gunicorn** inicializado via `exec gunicorn ... --preload cinematch.wsgi:app` (garantindo entrega direta de sinais `SIGTERM` / `SIGINT` para graceful shutdown dos workers).
- Execução sob usuário não privilegiado (`cinematch`, UID `10001`).
- Montagem somente-leitura (`:ro`) do dataset padronizado `./data/movies_metadata.csv:/app/data/movies_metadata.csv:ro`.
- `HEALTHCHECK` validando HTTP `200 OK` em `http://127.0.0.1:5000/api/health`.

### 5.1. Comandos Reproduzíveis de Build e Execução
```bash
# 1. Verificar que o dataset está posicionado em data/movies_metadata.csv
ls -lh data/movies_metadata.csv

# 2. Construir a imagem e iniciar o container em background
docker compose up --build -d

# 3. Acompanhar os logs de inicialização do Gunicorn
docker compose logs -f cinematch

# 4. Validar o endpoint de saúde e telemetria de memória
curl http://localhost:5000/api/health

# 5. Encerrar os containers com graceful shutdown
docker compose down
```
