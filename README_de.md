# 🎬 Filmempfehlungssystem

Willkommen beim **Filmempfehlungssystem**! Dieses Projekt ist eine robuste Python-Anwendung, die entwickelt wurde, um Filme basierend auf fortschrittlichen Datenanalysetechniken vorzuschlagen. Egal, ob Sie nach den größten Hits aller Zeiten oder einem versteckten Juwel suchen, das Ihrem Lieblingsfilm ähnelt, dieses System hat alles, was Sie brauchen.

## 🚀 Funktionen

### 1. 📈 Demografische Filterung (Top-Charts)
Genau wie die Top 250 von IMDB identifiziert dieses System die besten Filme durch Berechnung einer **Gewichteten Bewertung (Weighted Rating - WR)**. Es berücksichtigt sowohl die Durchschnittsbewertung als auch die Anzahl der Stimmen, um sicherzustellen, dass die empfohlenen Filme sowohl beliebt als auch hoch gelobt sind.

*   **Algorithmus**: IMDB-Formel für gewichtete Bewertung.
*   **Ziel**: Filme empfehlen, die allgemein beliebt sind.

### 2. 🧠 Inhaltsbasierte Filterung
Suchen Sie etwas Bestimmtes? Diese Funktion empfiehlt Filme, die einem bestimmten Film ähneln, den Sie mögen. Sie analysiert die **Handlungszusammenfassungen (overviews)** von Filmen, um Ähnlichkeiten zu finden.

*   **Algorithmus**: TF-IDF (Term Frequency-Inverse Document Frequency) & Kosinus-Ähnlichkeit.
*   **Ziel**: Filme mit ähnlichen Handlungssträngen und Themen empfehlen.

---

## 🛠️ Verwendete Technologien

*   **Python 3.13.7**: Die Kernprogrammiersprache.
*   **Pandas**: Für effiziente Datenmanipulation und -analyse.
*   **Scikit-learn**: Für maschinelle Lernalgorithmen (TF-IDF Vectorizer, Linear Kernel).
*   **Flask**: Zum Erstellen des Web-Backends.

---

## 📦 Installation & Einrichtung

1.  **Repository klonen** (oder Dateien herunterladen).
2.  **Erforderliche Bibliotheken installieren**:
    Öffnen Sie Ihr Terminal und führen Sie aus:
    ```bash
    pip install pandas scikit-learn flask flask-cors
    ```
3.  **Datensatz herunterladen**:
    Stellen Sie sicher, dass sich die Datei `movies_metadata.csv` im selben Verzeichnis wie das Skript befindet.
    [Datensatz hier herunterladen](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 Wie man es benutzt

### 🖥️ Befehlszeilenschnittstelle (CLI)

1.  Öffnen Sie Ihr Terminal oder Ihre Eingabeaufforderung.
2.  Navigieren Sie zum Projektverzeichnis.
3.  Führen Sie das Skript aus:
    ```bash
    python recommender.py
    ```

### 🌐 Ausführen der Weboberfläche (Ultramoderne UI)

1.  **Flask installieren**:
    ```bash
    pip install flask flask-cors
    ```
2.  **App ausführen**:
    ```bash
    python app.py
    ```
3.  **Im Browser öffnen**:
    Gehen Sie zu `http://localhost:5000`, um die neue Oberfläche zu sehen!

---

## 🔍 Code-Übersicht

### Klasse `MovieRecommenderSystem`

*   **`__init__`**: Lädt die Daten, bereinigt sie und bereitet sie für die Analyse vor.
*   **`get_demographic_recommendations`**: Berechnet die gewichtete Punktzahl für jeden Film und gibt die besten Ergebnisse zurück.
*   **`compute_content_similarity`**: Erstellt die TF-IDF-Matrix aus Filmzusammenfassungen und berechnet den Kosinus-Ähnlichkeits-Kernel.
*   **`get_content_recommendations`**: Nimmt einen Filmtitel als Eingabe und gibt eine Liste der ähnlichsten Filme basierend auf den berechneten Punktzahlen zurück.

---

## 📄 Lizenz

Dieses Projekt ist Open Source und für Bildungszwecke verfügbar. Fühlen Sie sich frei, es zu ändern und zu verbessern!

---

*Erstellt mit ❤️ von Warlem Jung*
