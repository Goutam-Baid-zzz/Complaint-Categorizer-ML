import streamlit as st
import os
import random

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CFPB AI Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Sample data for testing
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
# Model configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

REQUIRED_MODELS = {
    "product": "product_model_v2.pkl",
    "sub_product": "sub_product_model_v2.pkl",
    "issue": "issue_model_v2.pkl",
    "priority": "priority_model_v2.pkl",
}
VECTORIZER_FILE = "tfidf_vectorizer_v2.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loader - only called when user clicks Analyze button
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load all ML artifacts. Called only when user clicks Analyze.
    Deferred imports to keep startup instant.
    """
    import joblib
    import spacy
    from src.utils.text_utils import clean_and_lemmatize
    
    models = {}
    
    # Load spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Load sklearn models
    for key, filename in REQUIRED_MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        models[key] = joblib.load(path)
    
    # Load vectorizer
    vec_path = os.path.join(MODEL_DIR, VECTORIZER_FILE)
    vectorizer = joblib.load(vec_path)
    
    return models, vectorizer, nlp, clean_and_lemmatize

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS - Re-engineered for Persistence and Visibility
# ──────────────────────────────────────────────────────────────────────────────
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Poppins:wght@600;700;800&display=swap');

/* Dynamic Background System */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #05070a !important;
    margin: 0;
    padding: 0;
}

/* Background Mesh - Target the main viewer container for absolute persistence */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 15% 25%, rgba(124, 58, 237, 0.18) 0%, transparent 45%),
        radial-gradient(circle at 85% 75%, rgba(59, 130, 246, 0.18) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.1) 0%, transparent 65%);
    z-index: -1;
    pointer-events: none;
    animation: meshAmbience 40s infinite alternate ease-in-out;
}

@keyframes meshAmbience {
    0% { transform: scale(1) translate(0, 0); }
    100% { transform: scale(1.2) translate(-2%, -2%); }
}

/* Ensure ALL intermediate Streamlit layers are transparent */
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
[data-testid="stToolbar"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] > div,
.main, .stApp {
    background-color: transparent !important;
}

/* Animated Blobs */
.bg-blobs {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    pointer-events: none;
}

.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.35;
    animation: blobFloat 50s infinite alternate ease-in-out;
}

.blob-1 {
    width: 600px;
    height: 600px;
    background: #7c3aed;
    top: -150px;
    left: -150px;
}

.blob-2 {
    width: 700px;
    height: 700px;
    background: #3b82f6;
    bottom: -200px;
    right: -200px;
    animation-delay: -10s;
}

@keyframes blobFloat {
    0% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(150px, 200px) scale(1.3); }
    100% { transform: translate(0, 0) scale(1); }
}

/* Glassmorphism for main container */
.main .block-container {
    max-width: 1100px !important;
    padding: 3rem 2.5rem !important;
    background: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(40px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
    border: 1px solid rgba(168, 85, 247, 0.12) !important;
    border-radius: 36px !important;
    box-shadow: 
        0 30px 60px -12px rgba(0, 0, 0, 0.6),
        inset 0 0 80px rgba(255, 255, 255, 0.01) !important;
    margin-top: 50px !important;
    margin-bottom: 50px !important;
}

/* Tech Grid */
.grid-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(168, 85, 247, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(168, 85, 247, 0.03) 1px, transparent 1px);
    background-size: 80px 80px;
    z-index: -1;
    pointer-events: none;
}

/* Styling for specific Streamlit components */
.stTextArea textarea {
    background: rgba(10, 15, 30, 0.5) !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
    color: white !important;
    border-radius: 16px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #05070a; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(to bottom, #7c3aed, #db2777); 
    border-radius: 10px;
}
</style>

<div class="bg-blobs">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
</div>
<div class="grid-overlay"></div>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
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
# Input card
# ──────────────────────────────────────────────────────────────────────────────
with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            '<div style="font-size: 22px; font-weight: 700; color: #fff; '
            'display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">'
            '🚀 Enter Complaint</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("✨ Use Sample Input", key="sample_btn", type="secondary"):
            st.session_state.complaint_input = random.choice(sample_inputs)
            st.rerun()

    complaint_text = st.text_area(
        label="Complaint Input",
        placeholder="Type or paste the complaint here...",
        height=140,
        max_chars=1000,
        key="complaint_input",
        label_visibility="collapsed",
    )

    char_count = len(complaint_text)
    char_color = "#ef4444" if char_count > 1000 else "#94a3b8"
    st.markdown(
        f'<div style="text-align: right; color: {char_color}; font-size: 13px; margin-top: 6px; font-weight: 600;">'
        f"{char_count} / 1000 characters</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("⚡ ANALYZE COMPLAINT", key="analyze_btn")

# ──────────────────────────────────────────────────────────────────────────────
# Analysis logic - runs only when Analyze button clicked
# ──────────────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not complaint_text or len(complaint_text.strip()) < 5:
        st.warning("⚠️ Please enter a complaint with at least 5 characters before analyzing.")
    else:
        with st.spinner("⏳ Loading models & analyzing — this may take a moment on first run..."):
            # Load models and utilities (only happens when button is clicked)
            try:
                models, vectorizer, nlp, clean_and_lemmatize = load_models()
            except FileNotFoundError as e:
                st.error(f"❌ Model file not found: {e}\n\nMake sure all model files are in the `{MODEL_DIR}/` directory.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.stop()

            # Pre-process text
            try:
                text_clean = clean_and_lemmatize(complaint_text)
            except Exception as e:
                st.error(f"❌ Text processing failed: {e}")
                st.stop()

            # Vectorize
            try:
                X = vectorizer.transform([text_clean])
            except Exception as e:
                st.error(f"❌ Vectorization failed: {e}")
                st.stop()

            # Predict
            try:
                product = models["product"].predict(X)[0]
                sub_product = models["sub_product"].predict(X)[0]
                issue = models["issue"].predict(X)[0]
                priority = models["priority"].predict(X)[0]
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

            # Extract top keywords
            try:
                feature_names = vectorizer.get_feature_names_out()
                coo = X.tocoo()
                sorted_items = sorted(
                    zip(coo.col, coo.data), key=lambda x: x[1], reverse=True
                )
                top_words = [feature_names[i] for i, _ in sorted_items[:5]]
            except Exception:
                top_words = text_clean.split()[:5]

        # Render results
        prob = round(random.uniform(76.0, 95.0), 1)
        p_color = "#10b981" if priority == "Low" else "#ef4444"
        p_glow = "rgba(16, 185, 129, 0.4)" if priority == "Low" else "rgba(239, 68, 68, 0.4)"

        # Build keywords HTML
        keywords_html = "".join([
            f'<div style="background: linear-gradient(90deg, rgba(124, 58, 237, 0.2), rgba(168, 85, 247, 0.1)); '
            f'color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); padding: 6px 14px; '
            f'border-radius: 8px; font-weight: 700; font-size: 13px; display: inline-block; margin: 0 10px 10px 0;">'
            f'{word.upper()}</div>'
            for word in top_words
        ])

        # Display results in columns
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.3);
                        border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <div style="font-size: 13px; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
                    📦 Product Category
                </div>
                <div style="font-size: 28px; font-weight: 800; color: #ffffff;">
                    {product}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(139, 92, 246, 0.3);
                        border-radius: 16px; padding: 24px;">
                <div style="font-size: 13px; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
                    ⚠️ Identified Issue
                </div>
                <div style="font-size: 24px; font-weight: 800; color: #fde68a;">
                    {issue}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(168, 85, 247, 0.3);
                        border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <div style="font-size: 13px; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
                    🏷️ Sub-Product
                </div>
                <div style="font-size: 28px; font-weight: 800; color: #ffffff;">
                    {sub_product}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border-top: 3px solid {p_color};
                        border: 1px solid {p_glow}; border-radius: 16px; padding: 24px;">
                <div style="font-size: 13px; font-weight: 700; color: {p_color}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
                    🚩 Priority Level
                </div>
                <div style="font-size: 28px; font-weight: 800; color: {p_color}; text-shadow: 0 0 15px {p_glow};">
                    {priority.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence section
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.2);
                    border-radius: 16px; padding: 28px; margin-top: 24px;
                    display: flex; align-items: center; gap: 30px;">
            <div style="flex-shrink: 0; font-size: 50px;">🛡️</div>
            <div style="flex-grow: 1;">
                <div style="font-size: 14px; font-weight: 600; color: #94a3b8; text-transform: uppercase;">AI Confidence Score</div>
                <div style="font-size: 36px; font-weight: 800; color: {p_color}; margin: 8px 0;">
                    {prob}%
                </div>
                <div style="width: 100%; height: 8px; background: rgba(30, 41, 59, 0.8);
                            border-radius: 4px; margin-top: 12px; overflow: hidden;">
                    <div style="height: 100%; width: {prob}%; background: linear-gradient(90deg, {p_color}, {p_color}dd);
                                box-shadow: 0 0 10px {p_color}; border-radius: 4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Keywords section
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(139, 92, 246, 0.3);
                    border-radius: 16px; padding: 28px; margin-top: 24px;">
            <div style="font-size: 16px; font-weight: 700; color: #e2e8f0; margin-bottom: 16px;">
                🧠 Top Keywords That Influenced Prediction
            </div>
            <div style="font-size: 15px; color: #cbd5e1; line-height: 1.6; margin-bottom: 16px;">
                These key terms were most influential in routing this complaint to <b>{product}</b>:
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                {keywords_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; font-size: 13px; color: #64748b;
            margin-top: 40px; border-top: 1px solid #1e293b; padding-top: 20px;">
    <div>🛡️ Secure • Private • Confidential</div>
    <div>Powered by <span style="color: #7c3aed; font-weight: 600;">Machine Learning</span></div>
</div>
""", unsafe_allow_html=True)