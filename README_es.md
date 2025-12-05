# 🎬 Sistema de Recomendación de Películas

¡Bienvenido al **Sistema de Recomendación de Películas**! Este proyecto es una aplicación Python robusta diseñada para sugerir películas basadas en técnicas avanzadas de análisis de datos. Ya sea que busques los mayores éxitos de todos los tiempos o una joya oculta similar a tu película favorita, este sistema te tiene cubierto.

## 🚀 Características

### 1. 📈 Filtrado Demográfico (Top Charts)
Al igual que el Top 250 de IMDB, este sistema identifica las mejores películas calculando una **Calificación Ponderada (Weighted Rating - WR)**. Considera tanto la calificación promedio como el número de votos para asegurar que las películas recomendadas sean populares y altamente aclamadas.

*   **Algoritmo**: Fórmula de Calificación Ponderada de IMDB.
*   **Objetivo**: Recomendar películas que son universalmente apreciadas.

### 2. 🧠 Filtrado Basado en Contenido
¿Buscas algo específico? Esta función recomienda películas que son similares a una película particular que te gusta. Analiza los **resúmenes de la trama (sinopsis)** de las películas para encontrar similitudes.

*   **Algoritmo**: TF-IDF (Term Frequency-Inverse Document Frequency) & Similitud de Coseno.
*   **Objetivo**: Recomendar películas con historias y temas similares.

---

## 🛠️ Tecnologías Utilizadas

*   **Python 3.13.7**: El lenguaje de programación principal.
*   **Pandas**: Para manipulación y análisis eficiente de datos.
*   **Scikit-learn**: Para algoritmos de aprendizaje automático (Vectorizador TF-IDF, Kernel Lineal).

---

## 📦 Instalación y Configuración

1.  **Clonar el repositorio** (o descargar los archivos).
2.  **Instalar las bibliotecas necesarias**:
    Abre tu terminal y ejecuta:
    ```bash
    pip install pandas scikit-learn
    ```
3.  **Descargar el Dataset**:
    Asegúrate de tener el archivo `movies_metadata.csv` en el mismo directorio que el script.
    [Descarga el dataset aquí](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 Cómo Usar

1.  Abre tu terminal o símbolo del sistema.
2.  Navega al directorio del proyecto.
3.  Ejecuta el script:
    ```bash
    python recommender.py
    ```

### 🌐 Ejecutando la Interfaz Web (UI Ultramoderna)

1.  **Instalar Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **Ejecutar la App**:
    ```bash
    python app.py
    ```
3.  **Abrir en el Navegador**:
    ¡Ve a `http://localhost:5000` para ver la nueva interfaz!

### Qué Esperar (CLI):

*   **Paso 1**: El sistema cargará el dataset (tamaño de muestra predeterminado: 20,000 películas).
*   **Paso 2**: Mostrará las **Recomendaciones Demográficas Principales** (calificaciones ponderadas más altas).
*   **Paso 3**: Calculará la matriz de similitud de contenido.
*   **Paso 4**: Mostrará **Recomendaciones Basadas en Contenido** para una película de prueba específica (ej: "The Lion King").

---

## 🔍 Descripción General del Código

### Clase `MovieRecommenderSystem`

*   **`__init__`**: Carga los datos, los limpia y los prepara para el análisis.
*   **`get_demographic_recommendations`**: Calcula la puntuación ponderada para cada película y devuelve los mejores resultados.
*   **`compute_content_similarity`**: Construye la matriz TF-IDF a partir de las sinopsis de las películas y calcula el kernel de similitud de coseno.
*   **`get_content_recommendations`**: Toma un título de película como entrada y devuelve una lista de las películas más similares basadas en las puntuaciones calculadas.

---

## 📝 Ejemplo de Salida

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

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos. ¡Siéntete libre de modificarlo y mejorarlo!

---

*Creado con ❤️ por Warlem Jung*
