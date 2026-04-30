"""
CFPB Complaint Categorizer - Data Preprocessing & Feature Engineering
=====================================================================

This module provides the necessary pre-processing steps for training the ML models.
It handles:
1. Data Cleaning: Basic text cleaning (lowercasing, special character removal).
2. Lemmatization: Advanced spaCy-based lemmatization for improved semantic representation.
3. TF-IDF Vectorization: Building a feature representation (v2) for the prediction pipeline.
4. Data Sampling: Efficient subsetting for rapid experimentation while maintaining performance.

Inputs: Post-feature engineering CSV data.
Outputs: Pre-processed training data and a fitted TF-IDF vectorizer (v2).
"""

import pandas as pd
import pickle
import os
import sys

# ==============================
# 0. Environment Setup
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", ".."))

from src.utils.text_utils import lemmatize_pipe
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # Setup paths relative to the project root
    ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "post_f_engg_data.csv")
    SAVE_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer_v2.pkl")
    TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "train_data_v2.csv")
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Use a subset of 20,000 for timely completion while showing performance gains
    if len(df) > 20000:
        print("Using a subset of 20,000 rows for faster processing...")
        df = df.sample(n=20000, random_state=42)

    # Basic cleaning first
    print("Performing basic cleaning...")
    df['text'] = df['text'].fillna("")
    
    # Advanced Lemmatization
    print("Performing advanced lemmatization (spaCy)...")
    df['clean_text'] = lemmatize_pipe(df['text'].tolist())
    
    print("Fitting TF-IDF Vectorizer with improved parameters...")
    # ngram_range=(1, 2), max_features=20000, min_df=5, max_df=0.8
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=5,
        max_df=0.8
    )
    
    vectorizer.fit(df['clean_text'])
    
    print(f"Saving new vectorizer to {SAVE_PATH}...")
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
    
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"Saving processed data to {TRAIN_DATA_PATH}...")
    df.to_csv(TRAIN_DATA_PATH, index=False)
    print("✅ Feature Engineering & Pre-processing complete!")

if __name__ == "__main__":
    main()
