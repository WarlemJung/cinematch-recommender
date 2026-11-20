# Guia de Implantação e Operação em Produção — CineMatch

## 1. Pré-requisitos

- **Python:** 3.11, 3.12 ou 3.13 (testado em Python 3.13.7)
- **Memória RAM:** Mínimo 512 MB (o modelo esparso consome ~12,3 MB para 45.430 filmes; o processo completo ocupa ~190 MB)
- **Docker & Docker Compose:** Opcional, recomendado para implantação em containers

---

## 2. Configuração de Ambiente (`.env`)

1. Copie o arquivo de exemplo:
   ```bash
   cp .env.example .env
   ```
2. Ajuste as variáveis essenciais para produção:
   - `CINEMATCH_ENV=production` (garante `debug=False` obrigatoriamente)
   - `CINEMATCH_DEBUG=false`
   - `CINEMATCH_CORS_ORIGINS=https://seu-dominio.com`
   - `CINEMATCH_LOG_JSON=true` (ativa logs estruturados em JSON linha a linha)
   - `CINEMATCH_DATASET_PATH=/app/data/movies_metadata.csv`

---

## 3. Execução Local

### 3.1. Instalação e Validação do Dataset
```bash
pip install -r requirements.txt
python scripts/validate_dataset.py
```

### 3.2. Treinamento e Benchmark Quantitativo
```bash
python scripts/train_models.py --k 10 --max-queries 200
```

### 3.3. Inicialização do Servidor
```bash
python -m cinematch.main
# Ou utilizando o entrypoint compatível:
python app.py
```
A aplicação ficará disponível em `http://localhost:5000` e o health check em `http://localhost:5000/api/health`.

---

## 4. Implantação com Docker e Docker Compose

O `Dockerfile` foi construído seguindo práticas de hardening de segurança:
- Imagem base enxuta `python:3.13-slim`.
- Execução sob usuário não privilegiado (`cinematch`, UID `10001`).
- `.dockerignore` configurado para impedir a inclusão de datasets pesados, `.env` ou caches na imagem.
- `HEALTHCHECK` nativo consultando `/api/health`.

### 4.1. Subindo com Docker Compose
Certifique-se de que o arquivo `movies_metadata.csv` esteja na raiz do projeto (ele será montado como volume somente-leitura `:ro` em `/app/data/movies_metadata.csv`):

```bash
docker compose up --build -d
```

### 4.2. Verificando Saúde e Logs do Container
```bash
docker compose ps
docker compose logs -f cinematch
curl http://localhost:5000/api/health
```
