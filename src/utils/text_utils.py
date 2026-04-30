"""
CFPB Complaint Categorizer - Text Utilities
==============================================

Shared utility functions for text cleaning and lemmatization used across:
1. Gradio Web App (app/app.py) for real-time inference.
2. Inference Engine (src/models/predict.py) for batch processing.
3. Training Pipelines (src/data/preprocess.py) during feature engineering.

It ensures consistency in pre-processing by:
- Normalizing character cases and whitespace.
- Filtering out domain-specific stop words (dear, cfpb, sir, madam, etc.).
- Using spaCy for high-fidelity lemmatization.

Dependencies: Spacy, Re
"""

import spacy
import re
import os

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

DOMAIN_STOP_WORDS = {
    "dear", "cfpb", "xxxx", "to whom it may concern", "concern", "regarding", 
    "complaint", "consumer", "financial", "protection", "bureau", "dear cfpb", "sir", "madam"
}

def clean_text_basic(text):
    """Normalized text normalization helper."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Basic Cleaning
    text = clean_text_basic(text)
    
    # 2. Extract Lemmatized Tokens
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and token.lemma_ not in DOMAIN_STOP_WORDS and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

def lemmatize_pipe(texts, batch_size=500, n_process=1):
    """Optimized batch processing for large datasets using nlp.pipe."""
    from tqdm import tqdm
    processed_texts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts), desc="Lemmatizing"):
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and token.lemma_ not in DOMAIN_STOP_WORDS and len(token.lemma_) > 2
        ]
        processed_texts.append(" ".join(tokens))
    return processed_texts
