"""
CFPB Complaint Categorizer - Product Model Training Pipeline (v2)
=================================================================

This script handles the full training lifecycle for the 'Product' classification model.
It implements a robust machine learning pipeline including:
1. Data Loading: Imports pre-processed and feature-engineered data (v2).
2. Product Normalization: Groups similar financial products into broader categories (e.g., credit card & prepaid -> card) to handle class imbalance.
3. Feature Extraction: Loads a pre-fitted TF-IDF vectorizer (v2) for text-to-numerical conversion.
4. Hyperparameter Tuning: Optimized LinearSVC parameters via GridSearchCV.
5. Ensemble Learning: Combines LinearSVC, LogisticRegression, and MultinomialNB using a VotingClassifier for maximum predictive power.
6. Imbalance Handling: Uses SMOTE (Synthetic Minority Over-sampling Technique) within an imbalanced-learn pipeline.
7. Model Persistence: Saves the high-fidelity ensemble model as a pickle file (v2).

Goal: Achieve high precision and recall for product-level categorization of complaints.
"""

import pandas as pd
import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
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
# 1. Load Data (Improved Subset)
# ==============================
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "train_data_v2.csv")
if not os.path.exists(DATA_PATH):
    # Fallback home directory relative to project
    DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "post_f_engg_data.csv")

print(f"Loading improved training data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

TEXT_COL = "clean_text"   
TARGET_COL = "product"    

# ==============================
# 2. Product Normalization
# ==============================
def normalize_product(p):
    p = str(p).lower().strip()
    if "credit card" in p or "prepaid" in p: return "card"
    elif "loan" in p or "mortgage" in p: return "loan"
    elif "bank" in p or "checking" in p or "savings" in p: return "bank account"
    elif "money transfer" in p or "virtual currency" in p: return "transfer"
    elif "debt" in p: return "debt"
    elif "credit reporting" in p: return "credit reporting"
    else: return "other"

df[TARGET_COL] = df[TARGET_COL].apply(normalize_product)

print("Total samples:", len(df))
print("Classes:", df[TARGET_COL].nunique())

# ==============================
# 3. Load Vectorizer (Improved)
# ==============================
VECT_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer_v2.pkl")
print(f"Loading improved vectorizer from {VECT_PATH}...")
with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# 4. Transform Text
# ==============================
X = vectorizer.transform(df[TEXT_COL].fillna(""))
y = df[TARGET_COL]

print("Feature shape:", X.shape)

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. Hyperparameter Tuning (GridSearchCV for LinearSVC)
# ==============================
print("\nTuning LinearSVC hyperparameters...")
# We use a small grid for speed in this turn
svc_params = {'C': [0.1, 1, 3, 5, 10]}
svc_grid = GridSearchCV(LinearSVC(class_weight="balanced", dual=False), svc_params, cv=3, scoring='accuracy')
svc_grid.fit(X_train, y_train)
best_svc = svc_grid.best_estimator_
print(f"Best LinearSVC C: {svc_grid.best_params_['C']}")

# ==============================
# 7. Ensemble Models (VotingClassifier)
# ==============================
print("\nBuilding Ensemble (VotingClassifier)...")
ensemble = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('lr', LogisticRegression(class_weight="balanced", max_iter=1000)),
        ('nb', MultinomialNB())
    ],
    voting='hard'
)

# ==============================
# 8. Pipeline with SMOTE
# ==============================
# SMOTE is applied only to the training data to handle imbalance
model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('ensemble', ensemble)
])

# ==============================
# 9. Train Model
# ==============================
print("Training ensemble model with SMOTE...")
model_pipeline.fit(X_train, y_train)

# ==============================
# 10. Evaluation
# ==============================
y_pred = model_pipeline.predict(X_test)
print("\nFinal Accuracy (Ensemble):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================
# 11. Save Model (v2)
# ==============================
SAVE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "product_model_v2.pkl")
with open(SAVE_MODEL_PATH, "wb") as f:
    pickle.dump(model_pipeline, f)

print(f"\n✅ Optimized Product Model saved successfully to {SAVE_MODEL_PATH}!")

# ==============================
# 12. Generate Reports
# ==============================
print("\nGenerating evaluation reports...")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(os.path.join(REPORTS_DIR, "confusion_matrices"), exist_ok=True)

# 1. Metrics (JSON format)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
_, _, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "macro_f1": macro_f1,
    "weighted_f1": f1, # weighted f1 is already calculated
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

METRICS_PATH = os.path.join(REPORTS_DIR, "product_metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {METRICS_PATH}")

# 2. Confusion Matrix (PNG)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model_pipeline.classes_, 
            yticklabels=model_pipeline.classes_)
plt.title("Product Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
CM_PATH = os.path.join(REPORTS_DIR, "confusion_matrices", "product_cm.png")
plt.savefig(CM_PATH)
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# 3. Training Logs
LOG_PATH = os.path.join(REPORTS_DIR, "training_logs.txt")
with open(LOG_PATH, "a") as f:
    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING SESSION: PRODUCT MODEL\n")
    f.write(f"Model used: Ensemble (LinearSVC, LogisticRegression, MultinomialNB)\n")
    f.write(f"Hyperparameters (LinearSVC): {svc_grid.best_params_}\n")
    f.write(f"Dataset: train_data_v2.csv (Size: {len(df)})\n")
    f.write(f"Performance: Accuracy {metrics['accuracy']:.4f}, Macro-F1 {metrics['macro_f1']:.4f}, Weighted-F1 {metrics['weighted_f1']:.4f}\n")
    f.write("-" * 50 + "\n")
print(f"Logs appended to {LOG_PATH}")
