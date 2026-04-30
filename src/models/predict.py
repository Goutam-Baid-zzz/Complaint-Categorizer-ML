"""
CFPB Complaint Categorizer - Inference Engine (v2)
===================================================

This module handles the core prediction logic for incoming complaints.
It utilizes a multi-layered approach to categorization:
1. Advanced Pre-processing: Clean and lemmatize text using spaCy.
2. Vectorization: Convert text into numerical forms using a pre-trained TF-IDF vectorizer.
3. Machine Learning Ensemble: Use high-fidelity v2 models for predicting Product, Sub-product, Issue, and Priority.
4. Rule-based Layer: Apply a custom rule engine to detect specific high-priority keywords (fraud, scam, etc.).
5. Domain Correction: Refine predictions based on financial context clues.

Goal: Deliver a highly accurate and stable prediction system for the Gradio UI.
"""

import pickle
import os
import sys

# ==============================
# 1. Environment & Path Setup
# ==============================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.text_utils import clean_and_lemmatize

def load_models_robust():
    """Search recursively for models to handle any zip extraction structure."""
    target_file = "product_model_v2.pkl"
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    selected_path = None
    for root, dirs, files in os.walk(root_dir):
        if target_file in files:
            selected_path = root
            break

    if not selected_path:
        print("Models not found!")
        return None, None, None, None, None

    try:
        p = pickle.load(open(os.path.join(selected_path, "product_model_v2.pkl"), "rb"))
        sp = pickle.load(open(os.path.join(selected_path, "sub_product_model_v2.pkl"), "rb"))
        i = pickle.load(open(os.path.join(selected_path, "issue_model_v2.pkl"), "rb"))
        pr = pickle.load(open(os.path.join(selected_path, "priority_model_v2.pkl"), "rb"))
        vec = pickle.load(open(os.path.join(selected_path, "tfidf_vectorizer_v2.pkl"), "rb"))
        return p, sp, i, pr, vec
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None

# Load upgraded v2 models
product_model, sub_product_model, issue_model, priority_model, vectorizer = load_models_robust()

# ==============================
# 2. Rule Engine (Additional Layer)
# ==============================
def detect_rules(text):
    """
    Apply domain-specific rule-based overrides to handle edge cases and high-priority keywords.
    Ensures that critical issues (fraud, scams) are flagged immediately, bypassing ML inaccuracies.
    """
    text = text.lower()
    # 🔴 FRAUD (highest priority)
    if any(word in text for word in ["fraud", "scam", "unauthorized", "identity theft"]):
        return "Fraud / Scam", "High"
    # 🟡 PAYMENT
    if any(word in text for word in ["charged", "payment", "transaction", "deducted"]):
        return "Payment Issues", "Medium"
    # 🟡 TECHNICAL
    if any(word in text for word in ["crash", "error", "bug", "not working"]):
        return "Technical Issues", "Medium"
    # 🟢 CUSTOMER SERVICE
    if any(word in text for word in ["not responding", "no response", "customer support"]):
        return "Customer Service", "Low"
    return None, None

def correct_product(text, product, sub_product):
    """
    Apply hard-coded corrections based on high-confidence textual patterns.
    This step handles classification shifts by ensuring that specific mentions (e.g., 'card', 'account') point to their respective domains.
    """
    text = text.lower()
    if "card" in text: return "card", "credit card"
    if "account" in text: return "bank account", "checking account"
    if "loan" in text: return "loan", "personal loan"
    return product, sub_product

# ==============================
# 4. Final Prediction Function
# ==============================
def predict_complaint(text):
    """
    Orchestrate the holistic prediction workflow for a single consumer complaint.
    It combines advanced NLP (spaCy), TF-IDF vectorization, ML ensemble models, 
    and custom rule-based heuristics to return a comprehensive categorization.
    """
    try:
        # 1. Deep Pre-processing
        text_clean = clean_and_lemmatize(text)
        
        # 2. Vectorization
        X = vectorizer.transform([text_clean])
        
        # 3. ML Ensemble Predictions (v2)
        product = product_model.predict(X)[0]
        sub_product = sub_product_model.predict(X)[0]
        issue = issue_model.predict(X)[0]
        priority = priority_model.predict(X)[0]
        
        # 4. Rule Engine Layer (Override/Enhance)
        rule_issue, rule_priority = detect_rules(text)
        if rule_issue: issue = rule_issue
        if rule_priority: priority = rule_priority
        
        # 5. Domain Correction Layer
        product, sub_product = correct_product(text, product, sub_product)
        
        return {
            "Product": product,
            "Sub-product": sub_product,
            "Issue": issue,
            "Priority": priority,
            "Cleaned Text": text_clean  # Optional for debugging
        }
    except Exception as e:
        return {"Error": str(e)}

# ==============================
# 5. Interactive Test
# ==============================
if __name__ == "__main__":
    print("\n--- Optimized CFPB Complaint Classifier (v2) ---")
    sample = input("Enter financial complaint: ")
    if not sample.strip():
        print("Empty input. Exiting.")
    else:
        result = predict_complaint(sample)
        print("\nPrediction Results:")
        print("-" * 20)
        for key, value in result.items():
            print(f"{key:15}: {value}")
