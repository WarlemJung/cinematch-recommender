# 🎬 电影推荐系统

欢迎使用 **电影推荐系统**！这是一个强大的 Python 应用程序，旨在基于先进的数据分析技术推荐电影。无论您是在寻找历史上的经典佳作，还是与您最喜欢的电影相似的隐藏宝石，本系统都能满足您的需求。

## 🚀 功能特点

### 1. 📈 人口统计过滤 (热门排行榜)
就像 IMDB 的 Top 250 一样，本系统通过计算 **加权评分 (Weighted Rating - WR)** 来识别最佳电影。它同时考虑平均评分和投票数量，以确保推荐的电影既受欢迎又备受好评。

*   **算法**: IMDB 加权评分公式。
*   **目标**: 推荐广受好评的电影。

### 2. 🧠 基于内容的过滤
正在寻找特定的东西？此功能推荐与您喜欢的特定电影相似的电影。它分析电影的 **剧情简介 (overviews)** 以寻找相似之处。

*   **算法**: TF-IDF (词频-逆文档频率) & 余弦相似度。
*   **目标**: 推荐具有相似故事情节和主题的电影。

---

## 🛠️ 使用技术

*   **Python 3.13.7**: 核心编程语言。
*   **Pandas**: 用于高效的数据处理和分析。
*   **Scikit-learn**: 用于机器学习算法 (TF-IDF 向量化器, 线性核)。
*   **Flask**: 用于构建 Web 后端。

---

## 📦 安装与设置

1.  **克隆仓库** (或下载文件)。
2.  **安装所需的库**:
    打开终端并运行:
    ```bash
    pip install pandas scikit-learn flask flask-cors
    ```
3.  **下载数据集**:
    确保 `movies_metadata.csv` 文件与脚本位于同一目录中。
    [在此处下载数据集](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 如何使用

### 🖥️ 命令行界面 (CLI)

1.  打开终端或命令提示符。
2.  导航到项目目录。
3.  运行脚本:
    ```bash
    python recommender.py
    ```

### 🌐 运行 Web 界面 (超现代 UI)

1.  **安装 Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **运行应用**:
    ```bash
    python app.py
    ```
3.  **在浏览器中打开**:
    前往 `http://localhost:5000` 查看全新界面！

---

## 🔍 代码概览

### `MovieRecommenderSystem` 类

*   **`__init__`**: 加载数据，进行清洗并准备分析。
*   **`get_demographic_recommendations`**: 计算每部电影的加权分数并返回排名靠前的结果。
*   **`compute_content_similarity`**: 根据电影简介构建 TF-IDF 矩阵并计算余弦相似度核。
*   **`get_content_recommendations`**:以此电影标题作为输入，并根据计算出的分数返回最相似的电影列表。

---

## 📄 许可证

本项目开源并可用于教育目的。欢迎修改和改进！

---

*由 Warlem Jung 制作 ❤️*
