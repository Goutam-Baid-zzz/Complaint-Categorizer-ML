import gradio as gr
import pandas as pd
import joblib
import os
import spacy
import random
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
    ],
    "Payments & Transactions": [
        "A transaction failed, but the amount was deducted from my account and not refunded.",
        "The payment was successful from my side but not received by the recipient.",
        "The payment gateway failed during checkout, but money was still deducted.",
        "There is a delay in transaction processing, and payments are not reflecting immediately.",
        "My refund has not been processed even after several days."
    ],
    "Customer Service Issues": [
        "Customer support has been unresponsive despite multiple attempts to contact them.",
        "My complaint has not been resolved even after several follow-ups.",
        "The bank representatives are not providing clear information regarding my issue.",
        "I am not receiving proper updates regarding my raised complaint.",
        "The support team keeps redirecting me without resolving my issue."
    ],
    "Charges & Fees": [
        "I have been charged hidden fees that were not disclosed at the time of account opening.",
        "An incorrect penalty has been applied to my account without justification.",
        "The bank has deducted extra charges without providing any explanation.",
        "There are unexplained service charges in my account statement.",
        "Late payment fees have been applied incorrectly."
    ],
    "KYC & Verification": [
        "My KYC verification failed even though I submitted all valid documents.",
        "There is an issue with document verification delaying my account activation.",
        "My account is restricted due to incomplete KYC despite submitting documents.",
        "The system is not accepting my valid identification proof.",
        "KYC update requests are not being processed on time."
    ]
}

def use_sample():
    category = random.choice(list(sample_inputs.keys()))
    return random.choice(sample_inputs[category])

# 2. MODEL REGISTRY (SINGLETON)
class ModelRegistry:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.nlp = None
        return cls._instance

    def load_all(self):
        if not self.models:
            print("🧠 Loading models into RAM...")
            # Ensure spaCy model is available
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except:
                import os
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

            # Load joblib models
            base = "models"
            try:
                self.models['product'] = joblib.load(f"{base}/product_classifier.pkl")
                self.models['sub_product'] = joblib.load(f"{base}/sub_product_classifier.pkl")
                self.models['issue'] = joblib.load(f"{base}/issue_classifier.pkl")
                self.models['priority'] = joblib.load(f"{base}/priority_classifier.pkl")
                print("✅ All models loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
        return self.models, self.nlp

registry = ModelRegistry()

# 3. CYBERPUNK NEON CSS
custom_css = """
body, .gradio-container {
    background: radial-gradient(circle at 20% 30%, #1a1f3a, transparent 40%),
                radial-gradient(circle at 80% 70%, #2a1f4a, transparent 40%),
                linear-gradient(135deg, #0b0f2a, #0d1335, #0a0e2a) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}

/* Main container */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

/* Cards */
.card {
    background: #111735 !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 0 25px rgba(124, 58, 237, 0.15) !important;
    margin-bottom: 20px !important;
}

/* Title */
.title {
    text-align: center !important;
    font-size: 38px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a855f7, #7c3aed);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-top: 20px !important;
}

/* Subtitle */
.subtitle {
    text-align: center !important;
    color: #cbd5e1 !important;
    margin-bottom: 30px !important;
    font-size: 18px !important;
}

/* Input */
textarea, input {
    background: #0f1a3a !important;
    color: #ffffff !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
}

/* Button */
.main-button {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    height: 50px !important;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.3s ease !important;
}
.main-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.6) !important;
}

/* Secondary button (sample input) */
.sample-btn {
    background: #6d28d9 !important;
    font-size: 14px !important;
    height: 40px !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Result boxes */
.result-box {
    background: #0f1a3a !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    padding: 15px !important;
    color: #e2e8f0 !important;
}

/* Status Cards */
.priority-low {
    background: rgba(34,197,94,0.1) !important;
    color: #22c55e !important;
    border-left: 5px solid #22c55e !important;
}
.priority-high {
    background: rgba(239,68,68,0.1) !important;
    color: #ef4444 !important;
    border-left: 5px solid #ef4444 !important;
}

footer { display: none !important; }
"""

# 4. INFERENCE ENGINE
def analyze_complaint(text):
    if not text or len(text.strip()) < 10:
        return [gr.update(visible=False)] * 7

    models, nlp = registry.load_all()
    cleaned = clean_and_lemmatize(text)
    
    # 1. Product
    prod_pred = models['product'].predict([cleaned])[0]
    prod_prob = models['product'].predict_proba([cleaned]).max() * 100
    
    # 2. Sub-product
    sp_pred = models['sub_product'].predict([cleaned])[0]
    
    # 3. Issue
    issue_pred = models['issue'].predict([cleaned])[0]
    
    # 4. Priority
    priority_pred = models['priority'].predict([cleaned])[0]
    
    # UI Elements
    status_class = "priority-high" if priority_pred == "High" else "priority-low"
    status_icon = "🚨" if priority_pred == "High" else "✅"
    
    html_status = f"""
    <div class='{status_class}' style='padding: 15px; border-radius: 10px;'>
        <h3 style='margin:0;'>{status_icon} {priority_pred} Urgency Detected</h3>
        <p style='margin:5px 0 0 0; opacity: 0.8;'>Confidence Score: {prod_prob:.1f}%</p>
    </div>
    """
    
    explanation = f"Analysis complete. The complaint mainly concerns {prod_pred} ({sp_pred}), specifically relating to {issue_pred}. The system has flagged this as {priority_pred} priority based on linguistic sentiment and categorized keywords."
    
    return (
        gr.update(visible=True),
        f"📦 {prod_pred}",
        f"🏷️ {sp_pred}",
        f"⚠️ {issue_pred}",
        f"🚩 {priority_pred}",
        html_status,
        explanation
    )

# 5. UI COMPOSITION
with gr.Blocks(title="CFPB AI Intelligence") as demo:
    gr.HTML("""
    <div class="title">🤖 CFPB Complaint Intelligence</div>
    <div class="subtitle">Strategic ML Monitoring System for Smarter Complaint Analysis</div>
    """)
    
    with gr.Column(elem_classes="card"):
        with gr.Row():
            gr.Markdown("### 🚀 Enter Complaint Details")
            sample_btn = gr.Button("✨ Use Sample Input", elem_classes="sample-btn", scale=0)
            
        inp = gr.Textbox(
            placeholder="Describe the complaint in detail...",
            show_label=False,
            lines=4
        )
        btn = gr.Button("🔍 Analyze Now", elem_classes="main-button")

    with gr.Column(visible=False) as out_sec:
        with gr.Row():
            p_out = gr.Label(label="Product", elem_classes="result-box")
            s_out = gr.Label(label="Sub-product", elem_classes="result-box")
        
        with gr.Row():
            i_out = gr.Label(label="Issue", elem_classes="result-box")
            pr_out = gr.Label(label="Priority", elem_classes="result-box")
            
        stat_out = gr.HTML()
        expl_out = gr.Textbox(label="System Explanation", lines=3, elem_classes="result-box")

    # Event Handlers
    sample_btn.click(use_sample, outputs=inp)
    btn.click(analyze_complaint, inp, [out_sec, p_out, s_out, i_out, pr_out, stat_out, expl_out])

if __name__ == "__main__":
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('HF_HUB_HTTP_ENDPOINT')
    server_name = "0.0.0.0" if is_docker else "127.0.0.1"
    
    demo.launch(
        server_name=server_name,
        server_port=7860,
        css=custom_css
    )
