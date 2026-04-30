"""
CFPB Complaint Categorizer - Sub-product Model Training Pipeline (v2)
=====================================================================

This script handles the full training lifecycle for the 'Sub-product' classification model.
It addresses the high cardinality and class imbalance of sub-products through:
1. Data Loading: Imports pre-processed and feature-engineered data (v2).
2. Sub-product Normalization: Standardizes granular sub-product labels into consistent categories (e.g., checking account, savings account, crypto, etc.).
3. Majority Class Downsampling: Reduces the dominance of 'credit reporting' related entries to ensure the model learns diverse patterns.
4. Feature Extraction: Loads a pre-fitted TF-IDF vectorizer (v2) for text-to-numerical conversion.
5. Hyperparameter Tuning: Optimized LinearSVC parameters via GridSearchCV.
6. Ensemble Learning: Combines LinearSVC, LogisticRegression, and MultinomialNB using a VotingClassifier.
7. Advanced Imbalance Handling: Uses SMOTE with adjusted neighbors (k=1) to handle small minority classes.
8. Model Persistence: Saves the high-fidelity ensemble model as a pickle file (v2).

Goal: Provide deep granularity in complaint classification by identifying specific financial instruments.
"""

import pandas as pd
import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==============================
# 0. Path Setup
# ==============================
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ==============================
# 1. Load data (Processed Subset)
# ==============================
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "train_data_v2.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "post_f_engg_data.csv")

print(f"Loading improved training data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

TEXT_COL = "clean_text"
TARGET_COL = "sub_product"

# ==============================
# 2. Sub-product Normalization
# ==============================
def normalize_subproduct(p):
    p = str(p).lower().strip()
    if "credit card" in p: return "credit card"
    elif "prepaid" in p: return "prepaid card"
    elif "checking" in p: return "checking account"
    elif "savings" in p: return "savings account"
    elif "mortgage" in p: return "mortgage"
    elif "student loan" in p: return "student loan"
    elif "vehicle" in p or "auto loan" in p: return "auto loan"
    elif "payday" in p: return "payday loan"
    elif "loan" in p: return "personal loan"
    elif "money transfer" in p: return "money transfer"
    elif "virtual currency" in p: return "crypto"
    elif "debt collection" in p: return "debt collection"
    elif "credit reporting" in p: return "credit reporting"
    elif "overdraft" in p: return "overdraft"
    elif "fees" in p: return "fees"
    else: return "other"

df[TARGET_COL] = df[TARGET_COL].apply(normalize_subproduct)

# Reduce dominant class
df_major = df[df[TARGET_COL] == "credit reporting"]
df_rest = df[df[TARGET_COL] != "credit reporting"]
if len(df_major) > 10000:
    df_major = df_major.sample(n=10000, random_state=42)
df = pd.concat([df_major, df_rest])

print("\nSub-product Distribution:\n", df[TARGET_COL].value_counts())

# ==============================
# 3. Load vectorizer (Improved)
# ==============================
VECT_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer_v2.pkl")
with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# 4. Transform text
# ==============================
X = vectorizer.transform(df[TEXT_COL].fillna(""))
y = df[TARGET_COL]

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. Hyperparameter Tuning
# ==============================
print("\nTuning LinearSVC...")
svc_params = {'C': [0.1, 1, 5, 10, 20]}
svc_grid = GridSearchCV(LinearSVC(class_weight="balanced", dual=False, max_iter=2000), svc_params, cv=3, scoring='f1_weighted')
svc_grid.fit(X_train, y_train)
best_svc = svc_grid.best_estimator_
print(f"Best Sub-product LinearSVC C: {svc_grid.best_params_['C']}")

# ==============================
# 7. Ensemble & SMOTE
# ==============================
ensemble = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('lr', LogisticRegression(class_weight="balanced", max_iter=2000, C=2.0)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
        ('nb', MultinomialNB(alpha=0.1))
    ],
    voting='hard'
)

model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=1)), # k_neighbors=1 to handle very small classes
    ('ensemble', ensemble)
])

# ==============================
# 8. Train & Evaluate
# ==============================
print("Training Sub-product Ensemble model...")
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ==============================
# 9. Save model (v2)
# ==============================
SAVE_PATH = os.path.join(ROOT_DIR, "models", "sub_product_model_v2.pkl")
with open(SAVE_PATH, "wb") as f:
    pickle.dump(model_pipeline, f)

print(f"\n✅ Sub-product Model saved successfully to {SAVE_PATH}!")

# ==============================
# 10. Generate Reports
# ==============================
print("\nGenerating evaluation reports...")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(os.path.join(REPORTS_DIR, "confusion_matrices"), exist_ok=True)

# 1. Metrics (JSON format)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
_, _, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "macro_f1": macro_f1,
    "weighted_f1": f1,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

METRICS_PATH = os.path.join(REPORTS_DIR, "sub_product_metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {METRICS_PATH}")

# 2. Confusion Matrix (PNG)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model_pipeline.classes_, 
            yticklabels=model_pipeline.classes_)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Sub-Product Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
CM_PATH = os.path.join(REPORTS_DIR, "confusion_matrices", "sub_product_cm.png")
plt.savefig(CM_PATH)
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# 3. Training Logs
LOG_PATH = os.path.join(REPORTS_DIR, "training_logs.txt")
with open(LOG_PATH, "a") as f:
    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING SESSION: SUB-PRODUCT MODEL\n")
    f.write(f"Model used: Ensemble (LinearSVC, LogisticRegression, MultinomialNB)\n")
    f.write(f"Hyperparameters (LinearSVC): {svc_grid.best_params_}\n")
    f.write(f"Dataset: train_data_v2.csv (Sub-product subset)\n")
    f.write(f"Performance: Accuracy {metrics['accuracy']:.4f}, Macro-F1 {metrics['macro_f1']:.4f}, Weighted-F1 {metrics['weighted_f1']:.4f}\n")
    f.write("-" * 50 + "\n")
print(f"Logs appended to {LOG_PATH}")
