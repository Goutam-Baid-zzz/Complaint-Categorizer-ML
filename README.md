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

The dataset used in this project is sourced from the public database of the Consumer Financial Protection Bureau (CFPB), a U.S. government agency that collects and maintains consumer complaints related to financial products and services such as credit cards, loans, and banking.

**🔗 Official Dataset Source**:
https://www.consumerfinance.gov/data-research/consumer-complaints/

**Note**: The complete dataset provided by CFPB is quite large and contains extensive records. For the purpose of this project, a filtered and preprocessed subset of the dataset has been used to ensure efficient training and faster performance.

---

## 🛠️ Tech Stack
- **Engine**: Python 3.12, Scikit-learn
- **NLP**: spaCy (en_core_web_sm)
- **Frontend**: Streamlit (Custom Glassmorphism UI)
- **Deployment**: Streamlit Cloud / Docker

---

## 🏃‍♂️ Execution Walkthrough (Run Locally)

If you've cloned this repository from GitHub and want to run it on your own machine from scratch, follow these instructions carefully.

### 1. Download the Dataset
The machine learning models require data to learn from. The pipeline expects pre-processed data to start feature engineering.

1. Download or prepare your dataset to be processed.
   - 📊 **Raw Dataset (`complaints.csv`)**: [Download from Google Drive](https://drive.google.com/file/d/1yPyi5YCKh1wbtJxauktORltpZxALejh7/view?usp=drive_link)
   - 🧠 **Pre-trained Models (`trained_models.zip`)**: [Download from Google Drive](https://drive.google.com/file/d/1Bd6d2cnyLH5AQ7gSngn6H1DK3dgzlTrb/view?usp=drive_link) *(Optional, to skip training)*

2. Create the necessary directory structure if it doesn't exist: `data/processed/`

3. Place your preliminary text data file and name it exactly `post_f_engg_data.csv` inside `data/processed/`. (Ensure it has columns named `text` for the complaint descriptions).

### 2. Setup your Environment
Initialize a fresh environment and install the exact dependencies needed.

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate
# Note: If on Mac/Linux, use: source .venv/bin/activate

# Install strictly pinned dependencies
pip install -r requirements.txt

# Download the spaCy English NLP language model
python -m spacy download en_core_web_sm
```

### 3. Data Preprocessing
Process the raw text and extract features (`TF-IDF`). This command generates `train_data_v2.csv` and compiles your vectorizer model in the `models/` folder.

```bash
python src/data/preprocess.py
```

### 4. Model Training
Run all the individual training scripts to generate your `.pkl` model weight files.

```bash
# 1. Product Classifier
python src/models/train_product.py

# 2. Sub-product Classifier
python src/models/train_subproduct.py

# 3. Issue Classifier
python src/models/train_issue.py

# 4. Priority Level Classifier
python src/models/train_priority.py
```

### 5. Launch the Intelligence Platform
Once the models are generated into the `models/` folder, you can launch the local Streamlit server.

```bash
streamlit run app.py
```

*Look for the local URL in the console (e.g., `http://localhost:8501/`) and open it in your browser!*

---

## 📁 Project Structure
```bash
.
├── app.py                    # Main Streamlit application
├── models/                   # Trained model files (.pkl)
├── src/
│   ├── data/
│   │   └── preprocess.py    # Data preprocessing pipeline
│   ├── models/              # Training scripts
│   │   ├── train_product.py
│   │   ├── train_subproduct.py
│   │   ├── train_issue.py
│   │   └── train_priority.py
│   └── utils/
│       └── text_utils.py    # Text cleaning utilities
├── static/
│   └── background.png       # UI background image
├── data/
│   └── processed/           # Processed datasets
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```
---
