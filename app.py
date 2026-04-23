import gradio as gr
import pandas as pd
import joblib
import os
import spacy
import random
import pickle
from src.utils.text_utils import clean_and_lemmatize

# 1. SAMPLE DATA FOR THE "USE SAMPLE" FEATURE
sample_inputs = {
    "Loan Issues": [
        "I applied for a personal loan and got approval, but the amount has not been credited to my account even after several days.",
        "My loan EMI is being deducted regularly, but the outstanding balance is not reducing accordingly.",
        "The interest rate applied to my loan is higher than what was promised during the application process.",
        "There has been an unusual delay in loan disbursement despite completing all required formalities.",
        "My loan application was rejected without any proper explanation even though I meet all eligibility criteria."
    ],
    "Credit Card Issues": [
        "My credit card was charged twice for a single transaction, and the extra amount has not been refunded.",
        "There is a billing error in my credit card statement, and I am being overcharged.",
        "My credit card was blocked without any prior notification, causing inconvenience during payments.",
        "Unauthorized transactions have appeared on my credit card, and I did not receive any OTP alerts.",
        "The credit limit shown in my account is incorrect and lower than expected."
    ],
    "Fraud & Security": [
        "There was an unauthorized transaction from my account, and I did not receive any verification alert.",
        "I suspect my account has been hacked as there are multiple unknown transactions.",
        "I received alerts for login attempts from unknown devices, which is concerning.",
        "The bank failed to detect fraudulent transactions in my account.",
        "My debit card details seem to have been misused without my consent."
    ],
    "Bank Account Issues": [
        "My bank account was frozen without any prior notice, and I am unable to access my funds.",
        "There are errors in my bank statement showing incorrect transactions.",
        "My account was closed without informing me, and I cannot access my balance.",
        "There are duplicate entries in my account statement causing confusion.",
        "The balance shown in my account does not match my actual transactions."
    ]
}

def use_sample():
    category = random.choice(list(sample_inputs.keys()))
    return random.choice(sample_inputs[category])

# 2. MODEL REGISTRY (SINGLETON) - Updated for v2 names
class ModelRegistry:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.vectorizer = None
            cls._instance.nlp = None
        return cls._instance

    def load_all(self):
        if not self.models:
            print("🧠 Loading models into RAM...")
            # Ensure spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

            # Load upgraded v2 models (using pickle as in predict.py)
            base = "models"
            try:
                self.models['product'] = joblib.load(f"{base}/product_model_v2.pkl")
                self.models['sub_product'] = joblib.load(f"{base}/sub_product_model_v2.pkl")
                self.models['issue'] = joblib.load(f"{base}/issue_model_v2.pkl")
                self.models['priority'] = joblib.load(f"{base}/priority_model_v2.pkl")
                self.vectorizer = joblib.load(f"{base}/tfidf_vectorizer_v2.pkl")
                print("✅ All v2 models loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                # Fallback to local search if 'models/' isn't at root
                print("🔍 Searching for models in subdirectories...")
                return self._search_and_load()
        return self.models, self.vectorizer, self.nlp

    def _search_and_load(self):
        root_dir = "."
        for root, dirs, files in os.walk(root_dir):
            if "product_model_v2.pkl" in files:
                try:
                    self.models['product'] = joblib.load(os.path.join(root, "product_model_v2.pkl"))
                    self.models['sub_product'] = joblib.load(os.path.join(root, "sub_product_model_v2.pkl"))
                    self.models['issue'] = joblib.load(os.path.join(root, "issue_model_v2.pkl"))
                    self.models['priority'] = joblib.load(os.path.join(root, "priority_model_v2.pkl"))
                    self.vectorizer = joblib.load(os.path.join(root, "tfidf_vectorizer_v2.pkl"))
                    return self.models, self.vectorizer, self.nlp
                except: pass
        return {}, None, self.nlp

registry = ModelRegistry()

# 3. RULE ENGINE LOGIC
def detect_rules(text):
    text = text.lower()
    if any(word in text for word in ["fraud", "scam", "unauthorized", "identity theft"]):
        return "Fraud / Scam", "High"
    if any(word in text for word in ["charged", "payment", "transaction", "deducted"]):
        return "Payment Issues", "Medium"
    return None, None

def correct_product(text, product, sub_product):
    text = text.lower()
    if "card" in text: return "Credit Card", "Checking Card"
    if "loan" in text: return "Loan", "Personal Loan"
    return product, sub_product

# 4. CYBERPUNK NEON CSS
custom_css = """
body, .gradio-container {
    background: radial-gradient(circle at 20% 30%, #1a1f3a, transparent 40%),
                radial-gradient(circle at 80% 70%, #2a1f4a, transparent 40%),
                linear-gradient(135deg, #0b0f2a, #0d1335, #0a0e2a) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: auto !important; }
.card {
    background: #111735 !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 0 25px rgba(124, 58, 237, 0.15) !important;
    margin-bottom: 20px !important;
}
.title {
    text-align: center !important; font-size: 38px !important; font-weight: 700 !important;
    background: linear-gradient(90deg, #a855f7, #7c3aed);
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
}
.subtitle { text-align: center !important; color: #cbd5e1 !important; margin-bottom: 30px !important; font-size: 18px !important; }
textarea { background: #0f1a3a !important; color: #ffffff !important; border: 1px solid rgba(139, 92, 246, 0.3) !important; }
.main-button {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    height: 50px !important; box-shadow: 0 0 20px rgba(124, 58, 237, 0.4) !important;
}
.sample-btn { background: #5b21b6 !important; font-size: 14px !important; height: 40px !important; border: none !important; color: white !important; }
.result-box { background: #0f1a3a !important; border: 1px solid rgba(139, 92, 246, 0.3) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
.priority-low { background: rgba(34,197,94,0.1) !important; color: #22c55e !important; border-left: 5px solid #22c55e !important; }
.priority-high { background: rgba(239,68,68,0.1) !important; color: #ef4444 !important; border-left: 5px solid #ef4444 !important; }
footer { display: none !important; }
"""

# 5. INFERENCE ENGINE (v2 Logic)
def analyze_complaint(text):
    if not text or len(text.strip()) < 5:
        return [gr.update(visible=False)] * 7

    models, vectorizer, nlp = registry.load_all()
    if not models or not vectorizer:
        return gr.update(visible=True), "⚠️ Model Error", "Not Loaded", "Check Paths", "Error", f"<div style='color:red;'>Failed to load .pkl files. Check your 'models/' folder.</div>", "Error"

    # 1. Vectorization
    text_clean = clean_and_lemmatize(text)
    X = vectorizer.transform([text_clean])
    
    # 2. ML Ensemble Predictions
    product = models['product'].predict(X)[0]
    sub_product = models['sub_product'].predict(X)[0]
    issue = models['issue'].predict(X)[0]
    priority = models['priority'].predict(X)[0]
    
    # 3. Rule Engine Layer
    rule_issue, rule_priority = detect_rules(text)
    if rule_issue: issue = rule_issue
    if rule_priority: priority = rule_priority
    
    # 4. Domain Correction
    product, sub_product = correct_product(text, product, sub_product)
    
    # UI Elements
    status_class = "priority-high" if priority == "High" else "priority-low"
    status_icon = "🚨" if priority == "High" else "✅"
    
    html_status = f"""
    <div class='{status_class}' style='padding: 15px; border-radius: 10px;'>
        <h3 style='margin:0;'>{status_icon} {priority} Urgency Detected</h3>
        <p style='margin:5px 0 0 0; opacity: 0.8;'>AI flagged this complaint as {priority.lower()} priority.</p>
    </div>
    """
    
    explanation = f"Analysis complete. The system identified this as a {product} issue. Rule engine is active."
    
    return (gr.update(visible=True), product, sub_product, issue, priority, html_status, explanation)

# 6. UI COMPOSITION
with gr.Blocks(title="CFPB AI Intelligence", css=custom_css) as demo:
    gr.HTML("<div class='title'>🤖 CFPB Complaint Intelligence</div><div class='subtitle'>Strategic ML Monitoring System for Smarter Analysis</div>")
    
    with gr.Column(elem_classes="card"):
        with gr.Row():
            gr.Markdown("### 🚀 Enter Complaint")
            sample_btn = gr.Button("✨ Use Sample Input", elem_classes="sample-btn", scale=0)
        inp = gr.Textbox(placeholder="Describe the complaint...", show_label=False, lines=4)
        btn = gr.Button("🔍 Analyze Now", elem_classes="main-button")

    with gr.Column(visible=False) as out_sec:
        with gr.Row():
            p_out = gr.Label(label="Product", elem_classes="result-box")
            s_out = gr.Label(label="Sub-product", elem_classes="result-box")
        with gr.Row():
            i_out = gr.Label(label="Issue", elem_classes="result-box")
            pr_out = gr.Label(label="Priority", elem_classes="result-box")
        stat_out = gr.HTML()
        expl_out = gr.Textbox(label="System Explanation", lines=2, elem_classes="result-box")

    sample_btn.click(use_sample, outputs=inp)
    btn.click(analyze_complaint, inp, [out_sec, p_out, s_out, i_out, pr_out, stat_out, expl_out])

if __name__ == "__main__":
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('HF_HUB_HTTP_ENDPOINT')
    server_name = "0.0.0.0" if is_docker else "127.0.0.1"
    demo.launch(server_name=server_name, server_port=7860)
