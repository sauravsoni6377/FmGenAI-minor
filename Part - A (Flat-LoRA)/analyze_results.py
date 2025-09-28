import json
import numpy as np
from scipy import stats


def analyze_results():
    with open("financial_sentiment_results.json", "r") as f:
        results = json.load(f)

    summary = {}
    metrics = ["accuracy", "f1"]

    for method, runs in results.items():
        summary[method] = {}
        for metric in metrics:
            values = [r["metrics"].get(metric, 0.0) for r in runs]
            summary[method][f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
            summary[method][f"{metric}_std"] = float(np.std(values)) if values else 0.0
            summary[method][f"{metric}_values"] = values

    methods = [m for m in results.keys()]

    print("Financial Sentiment Analysis Results")
    print("=" * 50)
    print(f"{'Method':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 50)
    for method in methods:
        acc_mean = summary[method]["accuracy_mean"]
        acc_std = summary[method]["accuracy_std"]
        f1_mean = summary[method]["f1_mean"]
        f1_std = summary[method]["f1_std"]
        print(f"{method:<20} {acc_mean:.3f} ± {acc_std:.3f}  {f1_mean:.3f} ± {f1_std:.3f}")

    # If both lora and flat_lora exist, perform t-test
    if "lora" in summary and "flat_lora" in summary:
        lora_acc = summary["lora"]["accuracy_values"]
        flat_acc = summary["flat_lora"]["accuracy_values"]
        if lora_acc and flat_acc:
            t_stat, p_value = stats.ttest_ind(flat_acc, lora_acc)
            print("\nStatistical Significance (Flat-LoRA vs LoRA):")
            print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
            if p_value < 0.05:
                print("Difference is statistically significant (p < 0.05)")
            else:
                print("Difference is not statistically significant")


if __name__ == "__main__":
    analyze_results()
