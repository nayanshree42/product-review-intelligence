"""
Product Review Intelligence Engine — Inference
================================================
Run ABSA predictions on new product reviews.
Usage: python inference.py --review "battery life is amazing but camera is bad"
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = "models/roberta-absa"

ASPECTS = ["battery", "camera", "price", "quality",
           "delivery", "packaging", "display", "customer_service"]

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "power", "drain", "backup"],
    "camera": ["camera", "photo", "picture", "image", "lens", "selfie", "video"],
    "price": ["price", "cost", "value", "expensive", "cheap", "worth", "money", "budget"],
    "quality": ["quality", "build", "material", "durability", "finish", "feel", "solid"],
    "delivery": ["delivery", "shipping", "arrived", "dispatch", "courier", "late", "fast"],
    "packaging": ["packaging", "packed", "box", "wrap", "damaged", "scratch"],
    "display": ["display", "screen", "resolution", "brightness", "amoled", "lcd", "glare"],
    "customer_service": ["service", "support", "helpline", "refund", "return", "response", "complaint"]
}


def detect_aspects(review_text):
    """Detect which aspects are mentioned in a review."""
    review_lower = review_text.lower()
    detected = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in review_lower for kw in keywords):
            detected.append(aspect)
    return detected if detected else ["quality"]  # default fallback


def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    return ID2LABEL[pred], round(confidence, 4)


def analyze_review(review_text, model=None, tokenizer=None):
    """
    Full ABSA inference on a single review.
    Detects aspects and predicts sentiment for each.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        print(f"[INFO] Loading model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()

    detected_aspects = detect_aspects(review_text)
    results = []

    print(f"\n{'='*55}")
    print(f"REVIEW: {review_text}")
    print(f"{'='*55}")
    print(f"{'ASPECT':<20} {'SENTIMENT':<12} {'CONFIDENCE':<10}")
    print(f"{'-'*55}")

    for aspect in detected_aspects:
        aspect_text = f"{aspect}: {review_text}"
        sentiment, confidence = predict_sentiment(
            aspect_text, model, tokenizer, device
        )
        emoji = {"positive": "✅", "neutral": "⚪", "negative": "❌"}[sentiment]
        print(f"{aspect:<20} {emoji} {sentiment:<10} {confidence:.2%}")
        results.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "confidence": confidence
        })

    print(f"{'='*55}")
    return results


def batch_analyze(reviews, output_path="outputs/batch_results.json"):
    """Analyze a batch of reviews."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    all_results = []
    for i, review in enumerate(reviews):
        print(f"\n[{i+1}/{len(reviews)}] Processing...")
        result = analyze_review(review, model, tokenizer)
        all_results.append({"review": review, "analysis": result})

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Batch results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Product Review Intelligence — ABSA Inference"
    )
    parser.add_argument(
        "--review",
        type=str,
        default="battery life is excellent but camera quality is very disappointing and delivery was late",
        help="Product review text to analyze"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch inference on sample reviews"
    )
    args = parser.parse_args()

    if args.batch:
        sample_reviews = [
            "The battery is great but camera is average for the price",
            "Terrible delivery experience, packaging was damaged",
            "Excellent display quality, very happy with the purchase",
            "Customer service was unhelpful, product quality is poor",
            "Great value for money, build quality is solid and durable"
        ]
        batch_analyze(sample_reviews)
    else:
        analyze_review(args.review)
