# 🎬 Sistema de Recomendação de Filmes

Bem-vindo ao **Sistema de Recomendação de Filmes**! Este projeto é uma aplicação Python robusta projetada para sugerir filmes com base em técnicas avançadas de análise de dados. Se você está procurando pelos maiores sucessos de todos os tempos ou uma joia escondida semelhante ao seu filme favorito, este sistema tem o que você precisa.

## 🚀 Funcionalidades

### 1. 📈 Filtragem Demográfica (Top Charts)
Assim como o Top 250 do IMDB, este sistema identifica os melhores filmes calculando uma **Classificação Ponderada (Weighted Rating - WR)**. Ele considera tanto a avaliação média quanto o número de votos para garantir que os filmes recomendados sejam populares e altamente aclamados.

*   **Algoritmo**: Fórmula de Classificação Ponderada do IMDB.
*   **Objetivo**: Recomendar filmes que são universalmente apreciados.

### 2. 🧠 Filtragem Baseada em Conteúdo
Procurando por algo específico? Esta funcionalidade recomenda filmes que são semelhantes a um filme específico que você gosta. Ela analisa os **resumos do enredo (sinopses)** dos filmes para encontrar semelhanças.

*   **Algoritmo**: TF-IDF (Term Frequency-Inverse Document Frequency) & Similaridade de Cosseno.
*   **Objetivo**: Recomendar filmes com histórias e temas semelhantes.

---

## 🛠️ Tecnologias Utilizadas

*   **Python 3.13.7**: A linguagem de programação principal.
*   **Pandas**: Para manipulação e análise eficiente de dados.
*   **Scikit-learn**: Para algoritmos de aprendizado de máquina (Vetorizador TF-IDF, Kernel Linear).

---

## 📦 Instalação e Configuração

1.  **Clone o repositório** (ou baixe os arquivos).
2.  **Instale as bibliotecas necessárias**:
    Abra seu terminal e execute:
    ```bash
    pip install pandas scikit-learn
    ```
3.  **Baixe o Dataset**:
    Certifique-se de ter o arquivo `movies_metadata.csv` no mesmo diretório que o script.
    [Baixe o dataset aqui](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 Como Usar

1.  Abra seu terminal ou prompt de comando.
2.  Navegue até o diretório do projeto.
3.  Execute o script:
    ```bash
    python recommender.py
    ```

### 🌐 Executando a Interface Web (UI Ultramoderna)

1.  **Instale o Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **Execute o App**:
    ```bash
    python app.py
    ```
3.  **Abra no Navegador**:
    Vá para `http://localhost:5000` para ver a nova interface!

### O que Esperar (CLI):

*   **Passo 1**: O sistema carregará o dataset (tamanho da amostra padrão: 20.000 filmes).
*   **Passo 2**: Ele exibirá as **Recomendações Demográficas Principais** (maiores classificações ponderadas).
*   **Passo 3**: Ele calculará a matriz de similaridade de conteúdo.
*   **Passo 4**: Ele exibirá **Recomendações Baseadas em Conteúdo** para um filme de teste específico (ex: "The Lion King").

---

## 🔍 Visão Geral do Código

### Classe `MovieRecommenderSystem`

*   **`__init__`**: Carrega os dados, limpa e prepara para análise.
*   **`get_demographic_recommendations`**: Calcula a pontuação ponderada para cada filme e retorna os melhores resultados.
*   **`compute_content_similarity`**: Constrói a matriz TF-IDF a partir das sinopses dos filmes e calcula o kernel de similaridade de cosseno.
*   **`get_content_recommendations`**: Recebe um título de filme como entrada e retorna uma lista dos filmes mais semelhantes com base nas pontuações calculadas.

---

## 📝 Exemplo de Saída

```text
--- Carregando dados de movies_metadata.csv ---
Dados carregados: 20000 filmes.

--- Calculando Recomendação Demográfica (Top Charts) ---
                  title  vote_count  vote_average     score
10309  Dilwale Dulhania Le Jayenge        2000           9.1  8.615
314             The Shawshank Redemption        8205           8.5  8.446
834                  The Godfather        5893           8.5  8.425
...

--- Filmes similares a 'The Lion King' ---
34682    How the Lion Cub and the Turtle Sang a Song
9353                                The Lion King 1½
9115                  The Lion King 2: Simba's Pride
...
```

---

## 📄 Licença

Este projeto é de código aberto e está disponível para fins educacionais. Sinta-se à vontade para modificar e melhorar!

---

*Criado com ❤️ por Warlem Jung*
