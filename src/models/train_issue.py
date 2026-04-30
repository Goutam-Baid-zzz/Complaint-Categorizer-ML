"""
CFPB Complaint Categorizer - Issue Model Training Pipeline (v2)
===============================================================

This script handles the full training lifecycle for the 'Issue' classification model.
It implements a robust machine learning pipeline including:
1. Data Loading: Imports pre-processed and feature-engineered data (v2).
2. Issue Grouping: Maps complex, granular CFPB issue descriptions into broader, more balanced categories (e.g., Fraud/Scam, Payment Issues, etc.) to improve classification stability.
3. Feature Extraction: Loads a pre-fitted TF-IDF vectorizer (v2) for text-to-numerical conversion.
4. Hyperparameter Tuning: Optimized LinearSVC parameters via GridSearchCV.
5. Ensemble Learning: Combines LinearSVC, LogisticRegression, and MultinomialNB using a VotingClassifier for maximum predictive power.
6. Imbalance Handling: Uses SMOTE (Synthetic Minority Over-sampling Technique) within an imbalanced-learn pipeline.
7. Model Persistence: Saves the high-fidelity ensemble model as a pickle file (v2).

Goal: Achieve stable and meaningful categorization of complex complaint issues.
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
ISSUE_COL = "issue"

# ==============================
# 2. Improved grouping logic
# ==============================
def map_issue_to_group(issue):
    issue = str(issue).lower()
    if any(word in issue for word in ["fraud", "scam", "unauthorized", "identity theft"]): return "Fraud / Scam"
    elif any(word in issue for word in ["credit report", "credit score", "report"]): return "Credit Report Issues"
    elif any(word in issue for word in ["payment", "transaction", "transfer", "balance"]): return "Payment Issues"
    elif any(word in issue for word in ["loan", "mortgage"]): return "Loan Issues"
    elif any(word in issue for word in ["account", "login", "access", "closed"]): return "Account Issues"
    elif any(word in issue for word in ["service", "support", "response", "delay"]): return "Customer Service"
    elif any(word in issue for word in ["error", "technical", "bug", "website"]): return "Technical Issues"
    else: return "Other"

df["Issue_Grouped"] = df[ISSUE_COL].apply(map_issue_to_group)

# Reduce "Other" dominance if needed (already handled by SMOTE mostly, but keeping for data quality)
df_other = df[df["Issue_Grouped"] == "Other"]
df_rest = df[df["Issue_Grouped"] != "Other"]
if len(df_other) > 10000:
    df_other = df_other.sample(n=10000, random_state=42)
df = pd.concat([df_rest, df_other])

print("\nGrouped Distribution:\n", df["Issue_Grouped"].value_counts())

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
y = df["Issue_Grouped"]

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
svc_params = {'C': [0.1, 1, 3, 5, 10]}
svc_grid = GridSearchCV(LinearSVC(class_weight="balanced", dual=False, max_iter=2000), svc_params, cv=3, scoring='f1_weighted')
svc_grid.fit(X_train, y_train)
best_svc = svc_grid.best_estimator_
print(f"Best Issue LinearSVC C: {svc_grid.best_params_['C']}")

# ==============================
# 7. Ensemble & SMOTE
# ==============================
ensemble = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('lr', LogisticRegression(class_weight="balanced", max_iter=2000, C=1.5)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
        ('nb', MultinomialNB(alpha=0.5))
    ],
    voting='hard'
)

model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('ensemble', ensemble)
])

# ==============================
# 8. Train & Evaluate
# ==============================
print("Training Issue Ensemble model...")
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ==============================
# 9. Save model (v2)
# ==============================
SAVE_PATH = os.path.join(ROOT_DIR, "models", "issue_model_v2.pkl")
with open(SAVE_PATH, "wb") as f:
    pickle.dump(model_pipeline, f)

print(f"\n✅ Issue Model saved successfully to {SAVE_PATH}!")

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

METRICS_PATH = os.path.join(REPORTS_DIR, "issue_metrics.json")
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
plt.title("Issue Group Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
CM_PATH = os.path.join(REPORTS_DIR, "confusion_matrices", "issue_cm.png")
plt.savefig(CM_PATH)
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# 3. Training Logs
LOG_PATH = os.path.join(REPORTS_DIR, "training_logs.txt")
with open(LOG_PATH, "a") as f:
    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING SESSION: ISSUE MODEL\n")
    f.write(f"Model used: Ensemble (LinearSVC, LogisticRegression, MultinomialNB)\n")
    f.write(f"Hyperparameters (LinearSVC): {svc_grid.best_params_}\n")
    f.write(f"Dataset: train_data_v2.csv (Issue grouping subset)\n")
    f.write(f"Performance: Accuracy {metrics['accuracy']:.4f}, Macro-F1 {metrics['macro_f1']:.4f}, Weighted-F1 {metrics['weighted_f1']:.4f}\n")
    f.write("-" * 50 + "\n")
print(f"Logs appended to {LOG_PATH}")
