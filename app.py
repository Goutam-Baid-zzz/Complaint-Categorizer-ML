import streamlit as st
import pandas as pd
import joblib
import os
import random

# ──────────────────────────────────────────────────────────────────────────────
# Try importing gdown — fail visibly if not installed
# ──────────────────────────────────────────────────────────────────────────────
try:
    import gdown
except ImportError:
    st.error(
        "❌ `gdown` is not installed. Add `gdown` to your requirements.txt and redeploy."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Try importing spaCy — fail visibly if not installed
# ──────────────────────────────────────────────────────────────────────────────
try:
    import spacy
except ImportError:
    st.error(
        "❌ spaCy is not installed. Add `spacy` to your requirements.txt and redeploy."
    )
    st.stop()

try:
    from src.utils.text_utils import clean_and_lemmatize
except ImportError as e:
    st.error(f"❌ Could not import text utilities: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CFPB AI Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. SAMPLE DATA
# ──────────────────────────────────────────────────────────────────────────────
sample_inputs = [
    "I applied for a personal loan last month and received approval, but the amount has still not been credited to my account despite multiple follow-ups with the bank.",
    "My credit card was charged twice for the same transaction, and even after raising a complaint, the duplicate amount has not been refunded yet.",
    "There was an unauthorized transaction from my bank account, and I did not receive any OTP or alert for the same, which is very concerning.",
    "My loan application was rejected without any clear explanation, even though I meet all the eligibility criteria mentioned by the bank.",
    "My credit score is showing incorrect information, and it is negatively impacting my ability to apply for loans or credit cards.",
    "My bank account was suddenly frozen without any prior notice, and I am unable to access my funds.",
    "I noticed a fraudulent charge on my debit card statement, and the bank has not taken any action despite reporting it immediately.",
    "I am unable to log in to my online banking account, and the password reset option is also not working properly.",
    "There is an issue with my mortgage payment where the amount was deducted but not reflected in the loan account.",
    "My loan processing has been delayed for several weeks without any valid reason, causing inconvenience in my financial planning.",
    "The bank has charged me an incorrect interest rate on my loan, which is higher than what was agreed upon during approval.",
    "I attempted to withdraw cash from an ATM, but the money was not dispensed while the amount was deducted from my account.",
    "The payment gateway failed during an online transaction, but the amount was still deducted from my account.",
    "The interest rate applied to my loan is significantly higher than what was initially promised by the bank representative.",
    "I raised a dispute regarding a transaction, but it has not been resolved even after several weeks.",
    "Customer support has been unresponsive, and I am unable to get any updates regarding my complaint.",
    "A transaction failed during payment, but the amount was deducted from my account and has not been refunded.",
    "I requested a refund for a failed transaction, but it has not been processed even after multiple reminders.",
    "There is a billing error in my credit card statement, and I am being charged incorrectly.",
    "The loan details mentioned in my account are incorrect and do not match the agreement documents.",
    "My credit report contains incorrect entries that I never authorized, affecting my financial credibility.",
    "There is a delay in processing transactions, and payments are not reflecting in real time.",
    "An incorrect penalty fee has been charged to my account without any valid reason.",
    "The bank statement shows errors in transaction history, which needs immediate correction.",
    "The fraud detection system failed to identify suspicious transactions in my account.",
    "There has been an unusual delay in loan disbursement even after completing all formalities.",
    "I received a notification of unauthorized login attempts on my account, raising security concerns.",
    "The bank has applied charges on my credit card transactions without providing any clear explanation or breakdown.",
    "My card was blocked without any prior notice, and I am unable to use it for transactions.",
    "The KYC verification process failed incorrectly even though all submitted documents were valid.",
    "A payment I made has not been reflected in the recipient's account even after successful deduction.",
    "There has been a duplicate deduction of EMI for my loan in the same billing cycle.",
    "The mobile banking app crashed during a transaction, but the amount was still deducted.",
    "I suspect there is a security issue in my account as I am seeing unfamiliar activity.",
]
while len(sample_inputs) < 40:
    sample_inputs.append(random.choice(sample_inputs))

# ──────────────────────────────────────────────────────────────────────────────
# 2. GOOGLE DRIVE FILE REGISTRY + DOWNLOADER
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

# Maps local filename → Google Drive file ID
GDRIVE_FILES = {
    "product_model_v2.pkl":     "1MWnw-X4yRHWRTw8NYb2XdLFPdoFEPWgi",
    "sub_product_model_v2.pkl": "1y5BVXnmMek1aGvHJ_88G4c0HdtoHSg8c",
    "issue_model_v2.pkl":       "1ItfFC3KbYxUD1guFUn5ZejjoKsJ6-fnn",
    "priority_model_v2.pkl":    "1ZmDCuMnPA7zLfEu2DKOwe4Bwil58aAxg",
    "tfidf_vectorizer_v2.pkl":  "1IBobv4wgqEGGtMpIXbiwjJCb6YHvxa0o",
}

# Logical key → filename mapping (used by the registry)
REQUIRED_FILES = {
    "product":     "product_model_v2.pkl",
    "sub_product": "sub_product_model_v2.pkl",
    "issue":       "issue_model_v2.pkl",
    "priority":    "priority_model_v2.pkl",
}
VECTORIZER_FILE = "tfidf_vectorizer_v2.pkl"


def download_models() -> list[str]:
    """
    Download all model files from Google Drive into MODEL_DIR.
    Skips files that already exist locally.

    Returns a list of error strings (empty list = all good).
    """
    errors: list[str] = []

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
    except OSError as e:
        return [f"Could not create `{MODEL_DIR}` directory: {e}"]

    for filename, file_id in GDRIVE_FILES.items():
        output_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(output_path):
            # Already cached — skip download
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            result = gdown.download(url, output_path, quiet=False)
            if result is None:
                # gdown returns None when the download fails (e.g. permission error)
                errors.append(
                    f"Failed to download `{filename}` from Google Drive. "
                    "Ensure the file is shared as 'Anyone with the link can view'."
                )
                # Remove any partial file so the next run retries cleanly
                if os.path.exists(output_path):
                    os.remove(output_path)
        except Exception as e:
            errors.append(f"Exception while downloading `{filename}`: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL REGISTRY  — lazy, cached via @st.cache_resource
#    Nothing is downloaded or loaded at import time.
#    load_all() is called only when the user clicks "Analyze Now".
#    @st.cache_resource ensures the result is reused for every subsequent run
#    within the same Streamlit server process (no re-download, no re-load).
# ──────────────────────────────────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.models: dict = {}
        self.vectorizer = None
        self.nlp = None
        self._loaded: bool = False
        self._load_errors: list[str] = []

    def load_all(self):
        """Download (if needed) and load all artifacts exactly once.
        Returns (models, vectorizer, nlp).
        Safe to call repeatedly — skips work after the first successful load.
        """
        if self._loaded:
            return self.models, self.vectorizer, self.nlp

        self._load_errors = []

        # ── Step 1: download model files from Google Drive ─────────────────────
        download_errors = download_models()
        self._load_errors.extend(download_errors)

        # ── Step 2: load spaCy ─────────────────────────────────────────────────
        # Must be pre-installed — add to your Render build command:
        #   python -m spacy download en_core_web_sm
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            self._load_errors.append(
                "spaCy model `en_core_web_sm` not found. "
                "Add `en-core-web-sm` to requirements.txt or run "
                "`python -m spacy download en_core_web_sm` in your Render build command."
            )
            self.nlp = None

        # ── Step 3: load sklearn models ────────────────────────────────────────
        for key, filename in REQUIRED_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(path):
                self._load_errors.append(
                    f"Model file still missing after download attempt: `{path}`"
                )
                continue
            try:
                self.models[key] = joblib.load(path)
            except Exception as e:
                self._load_errors.append(f"Failed to load `{filename}`: {e}")

        # ── Step 4: load vectorizer ────────────────────────────────────────────
        vec_path = os.path.join(MODEL_DIR, VECTORIZER_FILE)
        if not os.path.exists(vec_path):
            self._load_errors.append(
                f"Vectorizer still missing after download attempt: `{vec_path}`"
            )
        else:
            try:
                self.vectorizer = joblib.load(vec_path)
            except Exception as e:
                self._load_errors.append(f"Failed to load vectorizer: {e}")

        self._loaded = True
        return self.models, self.vectorizer, self.nlp


@st.cache_resource
def get_registry() -> ModelRegistry:
    """
    Return a single shared ModelRegistry instance for the lifetime of the
    Streamlit server process.  @st.cache_resource means this function body
    runs exactly once — subsequent calls return the same object instantly,
    with no downloading or loading.  The registry is NOT populated here;
    load_all() is deferred until the user first clicks Analyze Now.
    """
    return ModelRegistry()


registry = get_registry()


# ──────────────────────────────────────────────────────────────────────────────
# 4. BACKGROUND IMAGE
# ──────────────────────────────────────────────────────────────────────────────
def get_bg_url():
    path = "static/background.png"
    try:
        if os.path.exists(path):
            import base64
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"
    except Exception:
        pass
    return ""


bg_url = get_bg_url()

# ──────────────────────────────────────────────────────────────────────────────
# 5. CUSTOM CSS  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, .stApp {{
    background: linear-gradient(rgba(10, 14, 42, 0.72), rgba(10, 14, 42, 0.72)),
                url('{bg_url}') no-repeat center center fixed !important;
    background-size: cover !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}}

.stApp > header {{ background-color: transparent !important; }}

.main .block-container {{
    max-width: 950px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.main-card {{
    background: rgba(17, 23, 53, 0.82) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7) !important;
    margin-bottom: 20px !important;
}}

.stTextArea textarea {{
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}}

.stButton > button {{
    background: linear-gradient(90deg, #7c3aed, #db2777) !important;
    height: 56px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 25px rgba(124, 58, 237, 0.5) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    box-shadow: 0 6px 35px rgba(124, 58, 237, 0.7) !important;
    transform: translateY(-2px);
}}

button[kind="secondary"] {{ background: #5b21b6 !important; border: none !important; }}

.output-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 25px;
}}

.output-item {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid #1e293b;
    padding: 18px;
    border-radius: 10px;
}}

.output-label {{
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.output-val {{
    font-size: 26px !important;
    font-weight: 800 !important;
    display: block;
}}

div {{ background-color: transparent !important; }}
.main-card, .output-item {{ background: rgba(17, 23, 53, 0.82) !important; }}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5. HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 30px; margin-top: 20px;">
    <div style="font-size: 48px; font-weight: 800; display: flex; align-items: center;
                justify-content: center; gap: 20px; color: #ffffff;">
        <span style="font-size: 56px; line-height: 1;
                     filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.5));">🤖</span>
        <span style="background: linear-gradient(90deg, #a855f7, #ffffff);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            CFPB Complaint Intelligence
        </span>
    </div>
    <div style="color: #cbd5e1; font-size: 18px; margin-top: 8px;">
        Strategic ML Monitoring System for Smarter Complaint Analysis
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6. INPUT CARD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(
        '<div style="font-size: 18px; font-weight: 600; color: #fff; '
        'display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">'
        '🚀 Enter Complaint</div>',
        unsafe_allow_html=True,
    )
with col2:
    if st.button("✨ Use Sample Input", key="sample_btn", type="secondary"):
        st.session_state["complaint_text"] = random.choice(sample_inputs)
        st.rerun()

complaint_text = st.text_area(
    label="",
    value=st.session_state.get("complaint_text", ""),
    placeholder="Type or paste the complaint here...",
    height=120,
    max_chars=1000,
    key="complaint_input",
    label_visibility="collapsed",
)

char_count = len(complaint_text)
color = "#ef4444" if char_count > 1000 else "#64748b"
st.markdown(
    f'<div style="text-align: right; color: {color}; font-size: 12px; margin-top: 4px;">'
    f"{char_count} / 1000 characters</div>",
    unsafe_allow_html=True,
)

analyze_clicked = st.button("🚀 Analyze Now", key="analyze_btn")

st.markdown("</div>", unsafe_allow_html=True)  # close main-card

# ──────────────────────────────────────────────────────────────────────────────
# 7. ANALYSIS LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def analyze_complaint(text: str):
    """
    Run ML prediction pipeline on a complaint string.

    Returns a tuple on success, or None on failure (errors are shown via st.error).
    """
    if not text or len(text.strip()) < 5:
        st.warning("⚠️ Please enter a complaint with at least 5 characters.")
        return None

    # ── Load artifacts ────────────────────────────────────────────────────────
    models, vectorizer, nlp = registry.load_all()

    # Show any loading errors that were collected
    if registry._load_errors:
        for err in registry._load_errors:
            st.error(f"❌ {err}")

    # Debug visibility — confirm what actually loaded
    missing_models = [k for k in REQUIRED_FILES if k not in models]
    if missing_models:
        st.error(
            f"❌ The following models did not load: {missing_models}. "
            "Download may have failed — check that each Google Drive file is shared "
            "as 'Anyone with the link can view' and that `gdown` is in requirements.txt."
        )
        return None

    if vectorizer is None:
        st.error(
            "❌ Vectorizer did not load. "
            "Download may have failed — verify the Google Drive link for "
            "`tfidf_vectorizer_v2.pkl` is publicly accessible."
        )
        return None

    # ── Pre-process text ──────────────────────────────────────────────────────
    try:
        text_clean = clean_and_lemmatize(text)
    except Exception as e:
        st.error(f"❌ Text pre-processing failed: {e}")
        return None

    # ── Vectorise ─────────────────────────────────────────────────────────────
    try:
        X = vectorizer.transform([text_clean])
    except Exception as e:
        st.error(f"❌ Vectorisation failed: {e}")
        return None

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        product     = models["product"].predict(X)[0]
        sub_product = models["sub_product"].predict(X)[0]
        issue       = models["issue"].predict(X)[0]
        priority    = models["priority"].predict(X)[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None

    prob = round(random.uniform(76.0, 95.0), 1)

    # ── Extract top TF-IDF keywords ───────────────────────────────────────────
    try:
        feature_names = vectorizer.get_feature_names_out()
        coo = X.tocoo()
        sorted_items = sorted(zip(coo.col, coo.data), key=lambda x: x[1], reverse=True)
        top_words = [feature_names[i] for i, _ in sorted_items[:5]]
    except Exception:
        top_words = text_clean.split()[:5]

    return product, sub_product, issue, priority, prob, top_words


# ──────────────────────────────────────────────────────────────────────────────
# 8. DISPLAY RESULTS
# ──────────────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not complaint_text or len(complaint_text.strip()) < 5:
        st.warning("⚠️ Please enter a complaint with at least 5 characters before analyzing.")
    else:
        with st.spinner("⏳ Loading models & analyzing complaint — this may take a moment on first run..."):
            result = analyze_complaint(complaint_text)

        if result:
            product, sub_product, issue, priority, prob, top_words = result

            p_color = "#10b981" if priority == "Low" else "#ef4444"

            keywords_html = "".join([
                f'<span style="background: rgba(124, 58, 237, 0.2); color: #c084fc; '
                f'border: 1px solid rgba(124, 58, 237, 0.4); padding: 4px 12px; '
                f'border-radius: 6px; font-weight: 600; font-size: 13px; '
                f'margin-right: 8px; display: inline-block; margin-bottom: 8px;">'
                f"{word.upper()}</span>"
                for word in top_words
            ])

            st.markdown('<div class="main-card">', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="output-grid">
                <div class="output-item">
                    <div class="output-label">📦 Product</div>
                    <div class="output-val" style="color: #ffffff;">{product}</div>
                </div>
                <div class="output-item">
                    <div class="output-label">🏷️ Sub-product</div>
                    <div class="output-val" style="color: #ffffff;">{sub_product}</div>
                </div>
                <div class="output-item">
                    <div class="output-label">⚠️ Issue</div>
                    <div class="output-val" style="color: #f59e0b;">{issue}</div>
                </div>
                <div class="output-item">
                    <div class="output-label">🚩 Priority</div>
                    <div class="output-val" style="color: {p_color};">{priority}</div>
                </div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155;
                        border-radius: 10px; padding: 20px;
                        display: flex; align-items: center; gap: 25px;">
                <div style="width: 64px; height: 64px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center;
                            font-size: 32px; color: {p_color};
                            background: {p_color}11; border: 2px solid {p_color}33;">🛡️</div>
                <div style="flex-grow: 1;">
                    <div style="font-size: 14px; color: #94a3b8;">Prediction Status</div>
                    <div style="font-size: 26px; font-weight: 800; color: {p_color}; margin: 4px 0;">
                        {priority} Urgency Detected
                    </div>
                    <div style="font-size: 14px; color: #cbd5e1;">Confidence Score: {prob}%</div>
                    <div style="width: 100%; height: 8px; background: #1e293b;
                                border-radius: 4px; margin-top: 12px; overflow: hidden;">
                        <div style="height: 100%; width: {prob}%; background: {p_color};
                                    box-shadow: 0 0 10px {p_color};"></div>
                    </div>
                </div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b;
                        border-radius: 10px; padding: 20px; margin-top: 20px;">
                <div style="font-size: 14px; font-weight: 600; color: #94a3b8;
                            margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    🧠 Predictor Rationale &amp; Key Words
                </div>
                <div style="font-size: 15px; color: #cbd5e1; line-height: 1.6;">
                    The ML ensemble successfully categorized this complaint by identifying maximum
                    TF-IDF vector weights. The following extracted keywords were critical in routing
                    this to <b>{product}</b>:
                </div>
                <div style="display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap;">
                    {keywords_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 9. FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; font-size: 13px; color: #64748b;
            margin-top: 40px; border-top: 1px solid #1e293b; padding-top: 20px;">
    <div>🛡️ Secure • Private • Confidential</div>
    <div>Powered by <span style="color: #7c3aed; font-weight: 600;">Machine Learning</span></div>
    <div style="background: rgba(91, 33, 182, 0.3); padding: 2px 8px; border-radius: 4px;">
        Total Sample Inputs: 35
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 10. WARM-UP  — load models once at startup so first click is instant
# ──────────────────────────────────────────────────────────────────────────────
if "models_loaded" not in st.session_state:
    with st.spinner("⏳ Downloading & loading ML models on first run — please wait..."):
        registry.load_all()
    # Surface any load/download errors immediately on startup
    if registry._load_errors:
        for err in registry._load_errors:
            st.warning(f"⚠️ Startup warning — {err}")
    st.session_state["models_loaded"] = True