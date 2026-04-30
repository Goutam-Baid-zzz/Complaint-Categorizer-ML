---
title: CFPB Complaint Intelligence
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🤖 CFPB Complaint Intelligence

## 📌 Overview
An expert-level Machine Learning system designed to categorize financial consumer complaints for the **Consumer Financial Protection Bureau (CFPB)**. This system automates the classification of unstructured text into four key dimensions: **Product**, **Sub-product**, **Issue**, and **Priority**.

## 🧠 Model Details
- **Architecture**: Voting Ensemble (LinearSVC, Logistic Regression, MultinomialNB, Random Forest)
- **NLP**: spaCy Lemmatization + TF-IDF Vectorization
- **Optimization**: SMOTE for class balancing and custom domain-specific rule logic.
- **Accuracy**: Optimized for high precision in fraud and identity theft detection.

## 🚀 How it Works
1. **Reception**: User submits a complaint description.
2. **Analysis**: Text is normalized and vectorized against financial domain dictionaries.
3. **Inference**: Multi-class ensemble models predict categories.
4. **Outcome**: Instant visualization of complaint categorization and urgency level.

## 📂 Data & Resources
The model weights are securely stored and retrieved dynamically to maintain a lightweight repository foot print while ensuring high performance on free-tier infrastructure.

## 🛠️ Tech Stack
- **Engine**: Python 3.12, Scikit-learn
- **NLP**: spaCy (en_core_web_sm)
- **Frontend**: Gradio (Glassmorphism UI)
- **Deployment**: Hugging Face Spaces

## 🏁 Live Demo
Visit the [Hugging Face Space](https://huggingface.co/spaces/Goutam-Baid-zzz/Complaint-Categorizer) to see it in action.

