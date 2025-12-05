# 🎬 Movie Recommender System

Welcome to the **Movie Recommender System**! This project is a robust Python application designed to suggest movies based on advanced data analysis techniques. Whether you're looking for the all-time greatest hits or a hidden gem similar to your favorite film, this system has you covered.

## 🚀 Features

### 1. 📈 Demographic Filtering (Top Charts)
Just like IMDB's Top 250, this system identifies the best movies by calculating a **Weighted Rating (WR)**. It considers both the average rating and the number of votes to ensure that the recommended movies are both popular and highly acclaimed.

*   **Algorithm**: IMDB Weighted Rating Formula.
*   **Goal**: Recommend movies that are universally liked.

### 2. 🧠 Content-Based Filtering
Looking for something specific? This feature recommends movies that are similar to a particular movie you like. It analyzes the **plot summaries (overviews)** of movies to find similarities.

*   **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency) & Cosine Similarity.
*   **Goal**: Recommend movies with similar storylines and themes.

---

## 🛠️ Technologies Used

*   **Python 3.13.7**: The core programming language.
*   **Pandas**: For efficient data manipulation and analysis.
*   **Scikit-learn**: For machine learning algorithms (TF-IDF Vectorizer, Linear Kernel).

---

## 📦 Installation & Setup

1.  **Clone the repository** (or download the files).
2.  **Install the required libraries**:
    Open your terminal and run:
    ```bash
    pip install pandas scikit-learn
    ```
3.  **Download the Dataset**:
    Ensure you have the `movies_metadata.csv` file in the same directory as the script.
    [Download the dataset here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 How to Use

1.  Open your terminal or command prompt.
2.  Navigate to the project directory.
3.  Run the script:
    ```bash
    python recommender.py
    ```

### 🌐 Running the Web Interface (Ultramodern UI)

1.  **Install Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **Run the App**:
    ```bash
    python app.py
    ```
3.  **Open in Browser**:
    Go to `http://localhost:5000` to see the new interface!

### What to Expect (CLI):

### What to Expect:

*   **Step 1**: The system will load the dataset (default sample size: 20,000 movies).
*   **Step 2**: It will display the **Top Demographic Recommendations** (highest weighted ratings).
*   **Step 3**: It will calculate the content similarity matrix.
*   **Step 4**: It will display **Content-Based Recommendations** for a specific test movie (e.g., "The Lion King").

---

## 🔍 Code Overview

### `MovieRecommenderSystem` Class

*   **`__init__`**: Loads the data, cleans it, and prepares it for analysis.
*   **`get_demographic_recommendations`**: Calculates the weighted score for each movie and returns the top results.
*   **`compute_content_similarity`**: Constructs the TF-IDF matrix from movie overviews and computes the cosine similarity kernel.
*   **`get_content_recommendations`**: Takes a movie title as input and returns a list of the most similar movies based on the computed scores.

---

## 📝 Example Output

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

## 📄 License

This project is open-source and available for educational purposes. Feel free to modify and improve it!

---

*Created with ❤️ by Warlem Jung*
