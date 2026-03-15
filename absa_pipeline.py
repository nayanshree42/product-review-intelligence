"""
Product Review Intelligence Engine
====================================
Aspect-Based Sentiment Analysis (ABSA) Pipeline
Author: Nayanshree Menpale
"""

import pandas as pd
import numpy as np
import re
import json
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ASPECTS = ["battery", "camera", "price", "quality", "delivery",
           "packaging", "display", "customer_service"]

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = "models/roberta-absa"


# ─────────────────────────────────────────────
# STEP 1 — DATA LOADING & SIMULATION
# ─────────────────────────────────────────────

def load_or_simulate_data(csv_path=None, n_samples=800):
    """
    Load real Amazon reviews CSV if available,
    otherwise simulate realistic review data for development.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[INFO] Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    print(f"[INFO] Simulating {n_samples} realistic product reviews...")

    review_templates = [
        # Battery
        ("battery life is amazing, lasts 2 days easily", "battery", "positive"),
        ("battery drains super fast, very disappointed", "battery", "negative"),
        ("battery performance is average, nothing special", "battery", "neutral"),
        ("incredible battery, best I have seen in this range", "battery", "positive"),
        ("battery dies within 4 hours of normal use", "battery", "negative"),
        ("battery is okay for the price", "battery", "neutral"),

        # Camera
        ("camera quality is outstanding, photos are crisp", "camera", "positive"),
        ("camera is terrible in low light conditions", "camera", "negative"),
        ("camera is decent for casual photography", "camera", "neutral"),
        ("front camera is great for video calls", "camera", "positive"),
        ("camera produces very blurry images", "camera", "negative"),
        ("camera quality is acceptable but not impressive", "camera", "neutral"),

        # Price
        ("great value for money, totally worth it", "price", "positive"),
        ("way too expensive for what you get", "price", "negative"),
        ("price is reasonable compared to competitors", "price", "neutral"),
        ("best budget option in this segment", "price", "positive"),
        ("overpriced product, not worth the cost", "price", "negative"),
        ("price seems fair for the features offered", "price", "neutral"),

        # Quality
        ("build quality is excellent, feels premium", "quality", "positive"),
        ("quality is very poor, broke within a month", "quality", "negative"),
        ("quality is average, expected more at this price", "quality", "neutral"),
        ("solid build, high quality materials used", "quality", "positive"),
        ("cheap plastic body, quality is disappointing", "quality", "negative"),
        ("quality is acceptable for everyday use", "quality", "neutral"),

        # Delivery
        ("delivered in 1 day, super fast shipping", "delivery", "positive"),
        ("delivery took 2 weeks, very frustrating", "delivery", "negative"),
        ("delivery was on time as expected", "delivery", "neutral"),
        ("excellent packaging and quick delivery", "delivery", "positive"),
        ("product arrived damaged due to poor delivery", "delivery", "negative"),
        ("delivery was slightly delayed but manageable", "delivery", "neutral"),

        # Packaging
        ("product was very well packed, no damage", "packaging", "positive"),
        ("packaging was terrible, product was scratched", "packaging", "negative"),
        ("packaging is simple but adequate", "packaging", "neutral"),
        ("excellent protective packaging, arrived perfectly", "packaging", "positive"),
        ("minimal packaging, product got dented", "packaging", "negative"),
        ("packaging is standard, nothing special", "packaging", "neutral"),

        # Display
        ("display is vibrant and bright, love it", "display", "positive"),
        ("display has too much glare, hard to see outside", "display", "negative"),
        ("display quality is okay for casual use", "display", "neutral"),
        ("stunning AMOLED display, colours are vivid", "display", "positive"),
        ("display scratches very easily", "display", "negative"),
        ("display resolution is decent but not great", "display", "neutral"),

        # Customer Service
        ("customer service was very helpful and quick", "customer_service", "positive"),
        ("customer service ignored my complaint completely", "customer_service", "negative"),
        ("customer service response was average", "customer_service", "neutral"),
        ("support team resolved my issue in minutes", "customer_service", "positive"),
        ("very rude customer service experience", "customer_service", "negative"),
        ("customer service was polite but slow to resolve", "customer_service", "neutral"),
    ]

    np.random.seed(42)
    records = []
    for i in range(n_samples):
        template = review_templates[i % len(review_templates)]
        # Add slight variation
        noise_words = ["honestly", "overall", "actually", "really", "tbh", ""]
        noise = np.random.choice(noise_words)
        text = f"{noise} {template[0]}".strip()
        records.append({
            "review_id": i + 1,
            "review_text": text,
            "aspect": template[1],
            "sentiment": template[2],
            "product_category": np.random.choice(
                ["Electronics", "Mobile", "Laptop", "Headphones", "Tablet"]
            ),
            "rating": {"positive": np.random.randint(4, 6),
                       "neutral": 3,
                       "negative": np.random.randint(1, 3)}[template[2]]
        })

    df = pd.DataFrame(records)
    df.to_csv("data/simulated_reviews.csv", index=False)
    print(f"[INFO] Simulated data saved to data/simulated_reviews.csv")
    return df


# ─────────────────────────────────────────────
# STEP 2 — EDA
# ─────────────────────────────────────────────

def run_eda(df):
    """Exploratory Data Analysis with visualizations."""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    print(f"Total Reviews       : {len(df)}")
    print(f"Unique Aspects      : {df['aspect'].nunique()}")
    print(f"Sentiment Distribution:\n{df['sentiment'].value_counts()}")
    print(f"Aspect Distribution:\n{df['aspect'].value_counts()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Product Review Intelligence — EDA", fontsize=14, fontweight='bold')

    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    axes[0].bar(sentiment_counts.index, sentiment_counts.values,
                color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title("Sentiment Distribution", fontweight='bold')
    axes[0].set_ylabel("Count")
    for i, v in enumerate(sentiment_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # Aspect distribution
    aspect_counts = df['aspect'].value_counts()
    axes[1].barh(aspect_counts.index, aspect_counts.values, color='#3498db', edgecolor='white')
    axes[1].set_title("Aspect Distribution", fontweight='bold')
    axes[1].set_xlabel("Count")

    # Aspect vs Sentiment heatmap
    pivot = df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='RdYlGn',
                ax=axes[2], linewidths=0.5)
    axes[2].set_title("Aspect × Sentiment Heatmap", fontweight='bold')

    plt.tight_layout()
    plt.savefig("outputs/eda_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] EDA plots saved to outputs/eda_analysis.png")
    return df


# ─────────────────────────────────────────────
# STEP 3 — BASELINE MODEL (TF-IDF + LogReg)
# ─────────────────────────────────────────────

def train_baseline(df):
    """TF-IDF + Logistic Regression baseline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    print("\n" + "="*50)
    print("BASELINE MODEL — TF-IDF + Logistic Regression")
    print("="*50)

    X = df['review_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ])
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nBaseline Weighted F1  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        mlflow.log_param("model", "TF-IDF + LogisticRegression")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("weighted_f1", f1)
        mlflow.log_metric("test_size", len(X_test))

    return baseline, f1, X_test, y_test


# ─────────────────────────────────────────────
# STEP 4 — ROBERTA FINE-TUNING
# ─────────────────────────────────────────────

def prepare_hf_dataset(df):
    """Prepare HuggingFace dataset for RoBERTa fine-tuning."""
    df = df.copy()
    df['label'] = df['sentiment'].map(LABEL2ID)
    df['text'] = df['review_text']

    train_df, test_df = train_test_split(
        df[['text', 'label', 'aspect']],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label']
    )

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    return {"f1": f1, "precision": precision, "recall": recall}


def train_roberta(train_dataset, val_dataset):
    """Fine-tune RoBERTa for aspect-level sentiment classification."""
    print("\n" + "="*50)
    print("FINE-TUNING RoBERTa FOR ABSA")
    print("="*50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    train_tok = tokenize_dataset(train_dataset, tokenizer)
    val_tok = tokenize_dataset(val_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="outputs/logs",
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="roberta_absa_finetune"):
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("max_length", MAX_LENGTH)

        print("[INFO] Starting RoBERTa fine-tuning...")
        trainer.train()

        eval_results = trainer.evaluate()
        mlflow.log_metric("val_f1", eval_results.get("eval_f1", 0))
        mlflow.log_metric("val_precision", eval_results.get("eval_precision", 0))
        mlflow.log_metric("val_recall", eval_results.get("eval_recall", 0))

        print(f"\nValidation Results: {eval_results}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Model saved to {OUTPUT_DIR}")

    return trainer, tokenizer, model


# ─────────────────────────────────────────────
# STEP 5 — EVALUATION & VISUALIZATIONS
# ─────────────────────────────────────────────

def evaluate_roberta(trainer, test_dataset, tokenizer, baseline_f1):
    """Evaluate fine-tuned RoBERTa on test set."""
    print("\n" + "="*50)
    print("MODEL EVALUATION — RoBERTa vs Baseline")
    print("="*50)

    test_tok = tokenize_dataset(test_dataset, tokenizer)
    predictions = trainer.predict(test_tok)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    roberta_f1 = f1_score(labels, preds, average='weighted')
    print(f"\nBaseline F1  : {baseline_f1:.4f}")
    print(f"RoBERTa F1   : {roberta_f1:.4f}")
    print(f"Improvement  : +{(roberta_f1 - baseline_f1):.4f} ({((roberta_f1-baseline_f1)/baseline_f1)*100:.1f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds,
                                target_names=["negative", "neutral", "positive"]))

    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RoBERTa ABSA — Evaluation Results", fontsize=13, fontweight='bold')

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"],
                ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontweight='bold')
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Model comparison
    models = ['TF-IDF\nBaseline', 'RoBERTa\nFine-tuned']
    f1_scores = [baseline_f1, roberta_f1]
    bars = axes[1].bar(models, f1_scores,
                       color=['#e67e22', '#2980b9'],
                       edgecolor='white', linewidth=2, width=0.5)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Model Comparison — Weighted F1", fontweight='bold')
    axes[1].set_ylabel("F1 Score")
    for bar, score in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{score:.3f}", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("outputs/evaluation_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Evaluation plots saved to outputs/evaluation_results.png")

    return roberta_f1, preds, labels


# ─────────────────────────────────────────────
# STEP 6 — BUSINESS INTELLIGENCE DASHBOARD
# ─────────────────────────────────────────────

def build_bi_dashboard(df, preds=None):
    """
    Build business intelligence dashboard showing
    aspect-level sentiment trends per product category.
    """
    print("\n[INFO] Building Business Intelligence Dashboard...")

    if preds is not None:
        df = df.copy().reset_index(drop=True)
        sample_size = min(len(preds), len(df))
        df_eval = df.iloc[:sample_size].copy()
        df_eval['predicted_sentiment'] = [ID2LABEL[p] for p in preds[:sample_size]]
    else:
        df_eval = df.copy()
        df_eval['predicted_sentiment'] = df_eval['sentiment']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Product Review Intelligence Dashboard",
                 fontsize=16, fontweight='bold', y=1.01)

    # 1. Aspect-level Sentiment Breakdown
    aspect_sentiment = df_eval.groupby(
        ['aspect', 'predicted_sentiment']
    ).size().unstack(fill_value=0)

    aspect_sentiment.plot(
        kind='bar', ax=axes[0, 0],
        color={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
        edgecolor='white'
    )
    axes[0, 0].set_title("Aspect-Level Sentiment Breakdown", fontweight='bold')
    axes[0, 0].set_xlabel("Aspect")
    axes[0, 0].set_ylabel("Review Count")
    axes[0, 0].legend(title="Sentiment")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Negative Review Drivers (Top Pain Points)
    neg_counts = df_eval[df_eval['predicted_sentiment'] == 'negative'][
        'aspect'].value_counts()
    colors_neg = ['#c0392b' if v == neg_counts.max() else '#e74c3c'
                  for v in neg_counts.values]
    axes[0, 1].barh(neg_counts.index, neg_counts.values,
                    color=colors_neg, edgecolor='white')
    axes[0, 1].set_title("Top Drivers of Negative Reviews\n(Business Pain Points)",
                         fontweight='bold')
    axes[0, 1].set_xlabel("Negative Review Count")
    for i, v in enumerate(neg_counts.values):
        axes[0, 1].text(v + 0.3, i, str(v), va='center', fontweight='bold')

    # 3. Satisfaction Score per Aspect
    sentiment_score = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_eval['score'] = df_eval['predicted_sentiment'].map(sentiment_score)
    aspect_score = df_eval.groupby('aspect')['score'].mean().sort_values()

    colors_score = ['#e74c3c' if s < 0 else '#2ecc71' for s in aspect_score.values]
    axes[1, 0].barh(aspect_score.index, aspect_score.values,
                    color=colors_score, edgecolor='white')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title("Aspect Satisfaction Score\n(-1 = Poor | 0 = Neutral | +1 = Great)",
                         fontweight='bold')
    axes[1, 0].set_xlabel("Satisfaction Score")

    # 4. Category-level Sentiment Distribution
    if 'product_category' in df_eval.columns:
        cat_sentiment = df_eval.groupby(
            ['product_category', 'predicted_sentiment']
        ).size().unstack(fill_value=0)
        cat_sentiment_pct = cat_sentiment.div(cat_sentiment.sum(axis=1), axis=0) * 100
        cat_sentiment_pct.plot(
            kind='bar', ax=axes[1, 1], stacked=True,
            color={'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'},
            edgecolor='white'
        )
        axes[1, 1].set_title("Sentiment Distribution by Product Category",
                             fontweight='bold')
        axes[1, 1].set_ylabel("Percentage (%)")
        axes[1, 1].legend(title="Sentiment", loc='upper right')
        axes[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig("outputs/bi_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] BI Dashboard saved to outputs/bi_dashboard.png")


# ─────────────────────────────────────────────
# STEP 7 — REVIEW SUMMARIZATION (BART)
# ─────────────────────────────────────────────

def generate_aspect_summaries(df, top_n_per_aspect=10):
    """
    Use BART summarization to generate business-ready
    insight summaries per product aspect.
    """
    print("\n[INFO] Generating Aspect-Level Summaries using BART...")

    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        summaries = {}
        for aspect in ASPECTS:
            aspect_reviews = df[df['aspect'] == aspect]['review_text'].tolist()
            combined = " ".join(aspect_reviews[:top_n_per_aspect])
            if len(combined.split()) < 30:
                summaries[aspect] = combined
                continue
            result = summarizer(
                combined,
                max_length=80,
                min_length=20,
                do_sample=False
            )
            summaries[aspect] = result[0]['summary_text']
            print(f"  {aspect.upper()}: {summaries[aspect]}")

        with open("outputs/aspect_summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)
        print("[INFO] Summaries saved to outputs/aspect_summaries.json")

    except Exception as e:
        print(f"[WARNING] Summarization skipped: {e}")
        summaries = {a: "Summary not available" for a in ASPECTS}

    return summaries


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    mlflow.set_experiment("product-review-intelligence")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n" + "="*60)
    print("   PRODUCT REVIEW INTELLIGENCE ENGINE")
    print("   Aspect-Based Sentiment Analysis Pipeline")
    print("="*60)

    # Step 1 — Load data
    df = load_or_simulate_data(n_samples=800)

    # Step 2 — EDA
    df = run_eda(df)

    # Step 3 — Baseline
    baseline_model, baseline_f1, X_test, y_test = train_baseline(df)

    # Step 4 — Prepare HuggingFace datasets
    train_dataset, val_dataset, test_dataset = prepare_hf_dataset(df)

    # Step 5 — Fine-tune RoBERTa
    trainer, tokenizer, model = train_roberta(train_dataset, val_dataset)

    # Step 6 — Evaluate
    roberta_f1, preds, labels = evaluate_roberta(
        trainer, test_dataset, tokenizer, baseline_f1
    )

    # Step 7 — BI Dashboard
    build_bi_dashboard(df, preds)

    # Step 8 — Aspect summaries
    summaries = generate_aspect_summaries(df)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE — RESULTS SUMMARY")
    print("="*60)
    print(f"  Baseline F1 (TF-IDF)     : {baseline_f1:.4f}")
    print(f"  RoBERTa Fine-tuned F1    : {roberta_f1:.4f}")
    print(f"  Improvement              : +{(roberta_f1-baseline_f1)*100:.1f}%")
    print(f"  Outputs saved to         : ./outputs/")
    print(f"  Model saved to           : ./{OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
