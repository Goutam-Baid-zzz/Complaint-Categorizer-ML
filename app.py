"""
CFPB Complaint Intelligence - Hugging Face Deployment
========================================================
Expertly configured for high-performance inference on HF Spaces.
"""

import os
import sys
import zipfile
import pickle
import base64
import numpy as np
import pandas as pd
import gradio as gr
import gdown

# 1. Environment & Path Setup
# Ensure consistency with local run and HF root exposure
sys.path.append(os.path.dirname(__file__))
from src.utils.text_utils import clean_and_lemmatize

# 2. Automated Model Retrieval
def download_models():
    """Download and extract models from Google Drive if not present."""
    model_dir = "models"
    zip_path = "trained_models.zip"
    # Shared Link ID for pre-trained weights
    file_id = "1Bd6d2cnyLH5AQ7gSngn6H1DK3dgzlTrb"
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Check if a sentinel file exists to avoid redownloading
    sentinel = os.path.join(model_dir, "product_model_v2.pkl")
    if not os.path.exists(sentinel):
        print("🚀 Downloading models from Google Drive...")
        gdown.download(url, zip_path, quiet=False)
        print("📦 Extracting models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_path)
        print("✅ Models ready!")
    else:
        print("ℹ️ Models already present.")

# 3. Model Loading (Efficient Singleton Pattern)
class ModelRegistry:
    def __init__(self):
        self.p_m = self.i_m = self.s_m = self.pr_m = self.v = None
        self.load()

    def load(self):
        download_models()
        # Search recursively for models to handle zip artifacts
        target = "product_model_v2.pkl"
        selected_path = None
        for root, _, files in os.walk("models"):
            if target in files:
                selected_path = root
                break
        
        if not selected_path:
            raise FileNotFoundError("Model verification failed after download.")

        print(f"🧠 Loading models from {selected_path}...")
        self.p_m = pickle.load(open(os.path.join(selected_path, "product_model_v2.pkl"), "rb"))
        self.i_m = pickle.load(open(os.path.join(selected_path, "issue_model_v2.pkl"), "rb"))
        self.s_m = pickle.load(open(os.path.join(selected_path, "sub_product_model_v2.pkl"), "rb"))
        self.pr_m = pickle.load(open(os.path.join(selected_path, "priority_model_v2.pkl"), "rb"))
        self.v = pickle.load(open(os.path.join(selected_path, "tfidf_vectorizer_v2.pkl"), "rb"))

registry = ModelRegistry()

# 4. UI Assets & Styling
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_path = os.path.join("static", "background.png")
bin_str = get_base64_of_bin_file(bg_path) if os.path.exists(bg_path) else ""

custom_css = f"""
    body, .gradio-container {{
        background-image: url("data:image/png;base64,{bin_str}") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }}
    .gradio-container::before {{
        content: ""; position: fixed; inset: 0;
        background: rgba(255, 255, 255, 0.4); z-index: -1; backdrop-filter: blur(8px);
    }}
    .card {{
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }}
    /* ... (rest of shared styles) */
    .result-box {{
        background: rgba(245, 245, 220, 0.5) !important;
        padding: 15px; border-radius: 12px;
        border: 1px solid #4F46E5; margin-top: 10px;
    }}
    .priority-high {{ background: #FEE2E2; color: #991B1B; border-left: 6px solid #DC2626; padding: 12px; border-radius: 8px; }}
    footer {{ visibility: hidden; }}
"""

# 5. Inference Logic
def analyze_complaint(text):
    if not text.strip():
        return gr.update(visible=False), "", "", "", "", "", "", ""
    
    cln = clean_and_lemmatize(text)
    X = registry.v.transform([cln])
    
    rp = registry.p_m.predict(X)[0]
    ri = registry.i_m.predict(X)[0]
    rs = registry.s_m.predict(X)[0]
    rpr = registry.pr_m.predict(X)[0]
    
    p_html = f'<div class="result-box"><b>Product:</b> {rp}</div>'
    i_html = f'<div class="result-box"><b>Issue:</b> {ri}</div>'
    sp_html = f'<div class="result-box"><b>Sub-product:</b> {rs}</div>'
    pr_html = f'<div class="result-box"><b>Priority:</b> {rpr}</div>'
    
    status_class = f"priority-{rpr.lower()}"
    status_html = f'<div class="{status_class}">Prediction Status: {rpr} Urgency Detected</div>'
    
    expl_text = f"Confidence analysis based on tokens: `{cln}`"
    return gr.update(visible=True), p_html, i_html, sp_html, pr_html, status_html, expl_text, ""

# 6. UI Composition
with gr.Blocks(title="CFPB AI Intelligence") as demo:
    gr.HTML("<div style='text-align: center;'><h1>🤖 CFPB Complaint Intelligence</h1><p>Strategic ML Monitoring System</p></div>")
    
    with gr.Column(elem_classes="card"):
        inp = gr.Textbox(lines=5, label="Input Complaint", placeholder="Describe the financial issue...")
        btn = gr.Button("🚀 Analyze Now", variant="primary")
        
    with gr.Column(elem_classes="card", visible=False) as out_sec:
        with gr.Row():
            p_out = gr.HTML()
            s_out = gr.HTML()
        with gr.Row():
            i_out = gr.HTML()
            pr_out = gr.HTML()
        stat_out = gr.HTML()
        expl_out = gr.Markdown()

    btn.click(analyze_complaint, inp, [out_sec, p_out, i_out, s_out, pr_out, stat_out, expl_out])

if __name__ == "__main__":
    # If running in Docker (HF), it needs 0.0.0.0. Otherwise, defaults to localhost.
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('HF_HUB_HTTP_ENDPOINT')
    server_name = "0.0.0.0" if is_docker else "127.0.0.1"
    
    demo.launch(
        server_name=server_name,
        server_port=7860,
        css=custom_css
    )
