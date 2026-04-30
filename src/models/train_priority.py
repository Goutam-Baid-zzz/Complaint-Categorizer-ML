"""
CFPB Complaint Categorizer - Priority Model Training Pipeline (v2)
===================================================================

This script handles the full training lifecycle for the 'Priority' classification model.
Since the CFPB dataset doesn't have an explicit 'Priority' label, this script:
1. Data Loading: Imports pre-processed and feature-engineered data (v2).
2. Label Engineering: Derives a 'Priority' label (High, Medium, Low) based on the 'Issue Group' and specific keywords in the complaint text.
3. Feature Extraction: Loads a pre-fitted TF-IDF vectorizer (v2) for text-to-numerical conversion.
4. Hyperparameter Tuning: Optimized LinearSVC parameters via GridSearchCV.
5. Ensemble Learning: Combines LinearSVC, LogisticRegression, and MultinomialNB using a VotingClassifier for maximum predictive power.
6. Imbalance Handling: Uses SMOTE (Synthetic Minority Over-sampling Technique) within an imbalanced-learn pipeline.
7. Model Persistence: Saves the high-fidelity ensemble model as a pickle file (v2).

Goal: Provide a strategic monitoring layer by identifying high-risk and urgent complaints.
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
# 2. Recreate Issue Grouping (for Priority calculation)
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

# ==============================
# 3. Create Priority (Derived Target)
# ==============================
def assign_priority(issue_group, text):
    text = str(text).lower()
    if issue_group in ["Fraud / Scam"] or any(word in text for word in ["fraud", "unauthorized", "scam", "identity theft"]):
        return "High"
    if issue_group in ["Payment Issues", "Loan Issues"] or any(word in text for word in ["delay", "charged", "payment issue"]):
        return "Medium"
    return "Low"

df["Priority"] = df.apply(lambda row: assign_priority(row["Issue_Grouped"], row[TEXT_COL]), axis=1)

print("\nPriority Distribution:\n", df["Priority"].value_counts())

# ==============================
# 4. Load vectorizer (Improved)
# ==============================
VECT_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer_v2.pkl")
with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# 5. Transform text
# ==============================
X = vectorizer.transform(df[TEXT_COL].fillna(""))
y = df["Priority"]

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 7. Hyperparameter Tuning
# ==============================
print("\nTuning LinearSVC...")
svc_params = {'C': [0.1, 1, 3, 5]}
svc_grid = GridSearchCV(LinearSVC(class_weight="balanced", dual=False), svc_params, cv=3, scoring='accuracy')
svc_grid.fit(X_train, y_train)
best_svc = svc_grid.best_estimator_

# ==============================
# 8. Ensemble & SMOTE
# ==============================
ensemble = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('lr', LogisticRegression(class_weight="balanced", max_iter=1000)),
        ('nb', MultinomialNB())
    ],
    voting='hard'
)

model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('ensemble', ensemble)
])

# ==============================
# 9. Train & Evaluate
# ==============================
print("Training Priority Ensemble model...")
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ==============================
# 10. Save model (v2)
# ==============================
SAVE_PATH = os.path.join(ROOT_DIR, "models", "priority_model_v2.pkl")
with open(SAVE_PATH, "wb") as f:
    pickle.dump(model_pipeline, f)

print(f"\n✅ Priority Model saved successfully to {SAVE_PATH}!")

# ==============================
# 11. Generate Reports
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

METRICS_PATH = os.path.join(REPORTS_DIR, "priority_metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {METRICS_PATH}")

# 2. Confusion Matrix (PNG)
plt.figure(figsize=(10, 8))
# Sort labels if possible for better visualization (Low, Medium, High)
labels = ["Low", "Medium", "High"]
# Intersection of our labels and classes actually present
labels = [l for l in labels if l in list(model_pipeline.classes_)]
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels)
plt.title("Priority Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
CM_PATH = os.path.join(REPORTS_DIR, "confusion_matrices", "priority_cm.png")
plt.savefig(CM_PATH)
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# 3. Training Logs
LOG_PATH = os.path.join(REPORTS_DIR, "training_logs.txt")
with open(LOG_PATH, "a") as f:
    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING SESSION: PRIORITY MODEL\n")
    f.write(f"Model used: Ensemble (LinearSVC, LogisticRegression, MultinomialNB)\n")
    f.write(f"Hyperparameters (LinearSVC): {svc_grid.best_params_}\n")
    f.write(f"Dataset: train_data_v2.csv (Derived Priority)\n")
    f.write(f"Performance: Accuracy {metrics['accuracy']:.4f}, Macro-F1 {metrics['macro_f1']:.4f}, Weighted-F1 {metrics['weighted_f1']:.4f}\n")
    f.write("-" * 50 + "\n")
print(f"Logs appended to {LOG_PATH}")
