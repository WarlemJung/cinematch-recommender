# 🎬 Système de Recommandation de Films

Bienvenue dans le **Système de Recommandation de Films** ! Ce projet est une application Python robuste conçue pour suggérer des films en se basant sur des techniques avancées d'analyse de données. Que vous recherchiez les plus grands succès de tous les temps ou une perle rare similaire à votre film préféré, ce système est là pour vous.

## 🚀 Fonctionnalités

### 1. 📈 Filtrage Démographique (Top Charts)
Tout comme le Top 250 d'IMDB, ce système identifie les meilleurs films en calculant une **Note Pondérée (Weighted Rating - WR)**. Il prend en compte à la fois la note moyenne et le nombre de votes pour s'assurer que les films recommandés sont à la fois populaires et très acclamés.

*   **Algorithme**: Formule de Note Pondérée IMDB.
*   **Objectif**: Recommander des films universellement appréciés.

### 2. 🧠 Filtrage Basé sur le Contenu
Vous cherchez quelque chose de spécifique ? Cette fonctionnalité recommande des films similaires à un film particulier que vous aimez. Elle analyse les **résumés de l'intrigue (synopsis)** des films pour trouver des similitudes.

*   **Algorithme**: TF-IDF (Term Frequency-Inverse Document Frequency) & Similarité Cosinus.
*   **Objectif**: Recommander des films avec des histoires et des thèmes similaires.

---

## 🛠️ Technologies Utilisées

*   **Python 3.13.7**: Le langage de programmation principal.
*   **Pandas**: Pour la manipulation et l'analyse efficaces des données.
*   **Scikit-learn**: Pour les algorithmes d'apprentissage automatique (Vectoriseur TF-IDF, Noyau Linéaire).
*   **Flask**: Pour créer le backend Web.

---

## 📦 Installation et Configuration

1.  **Cloner le dépôt** (ou télécharger les fichiers).
2.  **Installer les bibliothèques requises**:
    Ouvrez votre terminal et exécutez :
    ```bash
    pip install pandas scikit-learn flask flask-cors
    ```
3.  **Télécharger le Jeu de Données**:
    Assurez-vous d'avoir le fichier `movies_metadata.csv` dans le même répertoire que le script.
    [Téléchargez le jeu de données ici](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 Comment Utiliser

### 🖥️ Interface en Ligne de Commande (CLI)

1.  Ouvrez votre terminal ou invite de commande.
2.  Naviguez vers le répertoire du projet.
3.  Exécutez le script :
    ```bash
    python recommender.py
    ```

### 🌐 Exécuter l'Interface Web (UI Ultramoderne)

1.  **Installer Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **Lancer l'App**:
    ```bash
    python app.py
    ```
3.  **Ouvrir dans le Navigateur**:
    Allez sur `http://localhost:5000` pour voir la nouvelle interface !

---

## 🔍 Aperçu du Code

### Classe `MovieRecommenderSystem`

*   **`__init__`**: Charge les données, les nettoie et les prépare pour l'analyse.
*   **`get_demographic_recommendations`**: Calcule le score pondéré pour chaque film et renvoie les meilleurs résultats.
*   **`compute_content_similarity`**: Construit la matrice TF-IDF à partir des synopsis de films et calcule le noyau de similarité cosinus.
*   **`get_content_recommendations`**: Prend un titre de film en entrée et renvoie une liste des films les plus similaires en fonction des scores calculés.

---

## 📄 Licence

Ce projet est open-source et disponible à des fins éducatives. N'hésitez pas à le modifier et à l'améliorer !

---

*Créé avec ❤️ par Warlem Jung*
