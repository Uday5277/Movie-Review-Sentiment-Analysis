# IMDb Sentiment Analysis: Machine Learning Classification

## 1. Project Overview

This project implements various machine learning and deep learning models for "document-level sentiment analysis" on a dataset of 50,000 IMDb movie reviews. The core objective is to classify reviews as either `positive (1)` or `negative (0)`.

The project follows a standard NLP pipeline:
1.  Data Loading and Preprocessing: Loading raw IMDb text data, cleaning, and normalizing it (lowercasing, removing HTML, punctuation, stop words, and lemmatization).
2.  Vectorization: Transforming cleaned text into a numerical feature matrix using techniques like **Binary Count Vectorization** and TF-IDF Vectorization with N-grams.
3.  Model Training and Evaluation: Training and comparing six different classification models: Multinomial Naive Bayes, Logistic Regression, Support Vector Machine (SVM), Random Forest, Gradient Boosting, and a Deep Neural Network (DNN).

## 2. Project Structure

The repository is structured as follows:
imdb-sentiment/
├── data/
│   ├── processed/
│   ├── raw/
├── models/
│   ├── nb_pipeline.joblib
│   ├── logreg_tfidf_pipeline.joblib
│   ├── svm_tfidf_pipeline.joblib
│   ├── rf_tfidf_pipeline.joblib
│   ├── gb_tfidf_pipeline.joblib
├── notebooks/
│   └── 01_explore.ipynb (Main development notebook)
├── reports/
│   └── ML_miniproject.pdf (Final Project Write-up)
├── src/
│   ├── init.py
│   └── data.py (Data loading utility)
├── .gitignore
├── predict.py (Example script for loading and using a saved model)
├── README.md
├── requirements.txt


## 3. Setup and Installation Instructions

### Prerequisites

You need Python installed on your system.

### Steps

1.  Clone the Repository:
    ```bash
    git clone https://github.com/Uday5277/Movie-Review-Sentiment-Analysis.git
    cd imdb-sentiment
    ```

2.  Create and Activate Virtual Environment:
    * Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  Install Dependencies:
    All necessary packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The notebook also downloads required NLTK data (`stopwords`, `punkt`, `wordnet`, `omw-1.4`).*

4.  Data Acquisition:
    The original raw data (`aclImdb_v1.tar.gz`) is assumed to be in the `data/raw/` directory.

## 4. How to Run the Project

### A. Training and Exploration (Reproduce Results)

The entire data cleaning, feature extraction, model training, and evaluation process is contained within the notebook.

1.  Start a Jupyter Notebook/Lab session:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
2.  Navigate to the `notebooks` folder and open **`01_explore.ipynb`**.
3.  Run all cells in the notebook sequentially.
    * This will:
        * Load the raw data.
        * Clean the text and save the processed CSVs (`train_clean.csv`, `test_clean.csv`).
        * Train and evaluate the six models (Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting, DNN).
        * Save the best-performing pipeline models to the `models` folder (`nb_pipeline.joblib`, `logreg_tfidf_pipeline.joblib`, `svm_tfidf_pipeline.joblib`, etc.).

### B. Making Predictions

You can use the saved `logreg_tfidf_pipeline.joblib` model for quick predictions on new text, as demonstrated in the notebook and the `predict.py` script.

Using the provided example:
```bash
python predict.py
