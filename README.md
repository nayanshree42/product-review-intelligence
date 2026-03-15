# 🧠 Product Review Intelligence Engine
### Aspect-Based Sentiment Analysis (ABSA) for E-Commerce Reviews

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Standard sentiment analysis tells you **positive or negative** — that's not enough for business decisions.

This project builds a **multi-component NLP intelligence engine** that goes deeper:

> *"The battery is excellent but the camera is disappointing and delivery was very late"*

| Aspect | Sentiment | Confidence |
|--------|-----------|------------|
| Battery | ✅ Positive | 94.2% |
| Camera | ❌ Negative | 91.7% |
| Delivery | ❌ Negative | 88.3% |

This is **Aspect-Based Sentiment Analysis (ABSA)** — extracting *what* customers feel about *specific product attributes*, not just the overall review.

---

## 🎯 Business Problem

E-commerce platforms receive millions of product reviews. Aggregate star ratings hide critical signals:

- A product rated **3.5★** could mean *"great battery, terrible camera"* — not just "mediocre overall"
- Product teams need to know **which specific aspects** to fix
- Operations teams need early signals on **delivery and packaging issues**
- Manual review reading does not scale at 80,000+ reviews

**This engine automates that intelligence extraction end-to-end.**

---

## 🏗️ Project Architecture

```
Raw Reviews
    │
    ▼
┌─────────────────────────────────────┐
│  Component 1 — Aspect Detection     │  spaCy keyword matching + rule engine
│  (Which aspects are mentioned?)     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Component 2 — ABSA Model           │  Fine-tuned RoBERTa
│  (What sentiment per aspect?)       │  (vs TF-IDF baseline)
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Component 3 — Review Summarization │  BART (facebook/bart-large-cnn)
│  (Business-ready insight reports)   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Component 4 — BI Dashboard         │  Matplotlib / Seaborn
│  (Visual insight for stakeholders)  │
└─────────────────────────────────────┘
```

---

## 📊 Results

| Model | Weighted F1 | Precision | Recall |
|-------|-------------|-----------|--------|
| TF-IDF + Logistic Regression (Baseline) | 0.79 | 0.81 | 0.79 |
| **RoBERTa Fine-tuned (Ours)** | **0.88** | **0.89** | **0.88** |

**RoBERTa outperforms the baseline by ~11% on weighted F1.**

### Aspect-Level Performance
| Aspect | F1 | Notes |
|--------|----|-------|
| Battery | 0.91 | High signal, clear vocabulary |
| Camera | 0.89 | Good coverage |
| Price | 0.88 | Strong positive/negative contrast |
| Delivery | 0.87 | Temporal language handled well |
| Quality | 0.85 | Most ambiguous aspect |
| Packaging | 0.90 | Clear damage/intact signals |
| Display | 0.88 | Technical vocabulary helps |
| Customer Service | 0.86 | Sentiment-heavy language |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| NLP / Modelling | HuggingFace Transformers, RoBERTa, BART, spaCy |
| Baseline | Scikit-learn, TF-IDF, Logistic Regression |
| Training | PyTorch, HuggingFace Trainer API |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
product-review-intelligence/
│
├── absa_pipeline.py        # Main end-to-end training pipeline
├── inference.py            # Inference script for new reviews
├── requirements.txt        # Dependencies
│
├── data/
│   └── simulated_reviews.csv   # Generated on first run
│
├── models/
│   └── roberta-absa/           # Saved fine-tuned model
│
├── outputs/
│   ├── eda_analysis.png        # EDA visualizations
│   ├── evaluation_results.png  # Model comparison plots
│   ├── bi_dashboard.png        # Business Intelligence dashboard
│   └── aspect_summaries.json  # BART-generated summaries
│
└── notebooks/
    └── exploration.ipynb       # Step-by-step walkthrough
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/nayanshree42/product-review-intelligence.git
cd product-review-intelligence
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python absa_pipeline.py
```
This will:
- Simulate/load 800 product reviews
- Run EDA and save plots
- Train TF-IDF baseline
- Fine-tune RoBERTa (ABSA)
- Evaluate and compare models
- Generate BI dashboard
- Generate aspect summaries using BART

### 3. Run Inference on New Reviews
```bash
# Single review
python inference.py --review "battery is amazing but camera quality is disappointing"

# Batch inference
python inference.py --batch
```

### 4. View Experiments in MLflow
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 💡 Key Design Decisions

**Why RoBERTa over plain BERT?**
RoBERTa removes the Next Sentence Prediction objective and uses dynamic masking — better suited for short, noisy review text.

**Why aspect-level instead of review-level sentiment?**
Review-level sentiment loses critical granularity. A 3-star review with excellent battery but terrible camera needs different action than a uniformly poor review.

**Why TF-IDF as baseline?**
Provides a strong, interpretable benchmark. Shows the real uplift from using transformer-based models — important for stakeholder communication.

**Why MLflow?**
Reproducibility. All hyperparameters, metrics, and model artifacts are logged — making it easy to compare runs and roll back if needed.

---

## 📈 Business Impact (Simulated Scenario)

- Product teams can identify **top 3 aspects driving negative reviews** per category
- Operations can detect **delivery/packaging spikes** within hours instead of days
- Reduces manual review reading effort by **~80%**
- Enables **per-SKU insight reports** at scale

---

## 🔮 Future Improvements

- [ ] Real Amazon Reviews dataset integration (via Kaggle API)
- [ ] Fine-grained aspect extraction using spaCy dependency parsing
- [ ] Multilingual support (Hindi, regional Indian languages)
- [ ] REST API wrapper using FastAPI
- [ ] Streamlit dashboard for interactive exploration

---

## 👩‍💻 Author

**Nayanshree Menpale**
Data Scientist | NLP | MLOps
[LinkedIn](https://linkedin.com/in/nayanshree-ml) | [Kaggle](https://kaggle.com/nayanshreemenpale)

---

## 📄 License
MIT License
