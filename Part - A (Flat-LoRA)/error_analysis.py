import os
from transformers import pipeline


def analyze_errors(model_dir="results/financial_sentiment/flat_lora_seed42"):
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist. Run experiments first.")
        return

    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir)

    difficult_examples = [
        {
            "text": "The company beat earnings estimates but missed revenue projections",
            "true_label": "neutral",
            "explanation": "Mixed performance - beat on earnings but miss on revenue",
        },
        {
            "text": "Restructuring charges will impact Q3 results before expected recovery",
            "true_label": "neutral",
            "explanation": "Short-term negative, long-term positive",
        },
        {
            "text": "Inventory write-downs reflect challenging market conditions",
            "true_label": "negative",
            "explanation": "Clear negative signal from write-downs",
        },
    ]

    print("Error Analysis - Difficult Financial Texts")
    print("=" * 60)
    for example in difficult_examples:
        pred = classifier(example["text"])[0]
        print(f"\nText: {example['text']}")
        print(f"True Label: {example['true_label']}")
        print(f"Predicted: {pred['label']} (score: {pred['score']:.3f})")
        print(f"Explanation: {example['explanation']}")
        print("-" * 40)


if __name__ == "__main__":
    analyze_errors()
