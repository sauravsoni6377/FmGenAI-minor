import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

from logTrainer import FlatLoraTrainer
import torch


def find_all_linear_modules(model) -> list:
    """Return a list of candidate module names (last attribute) that are torch.nn.Linear modules.
    This mirrors the helper in run_exp.py and ensures LoRA targets only supported modules."""
    linear_cls = torch.nn.Linear
    output_layer_names = ["classifier", "lm_head", "embed_tokens"]
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(out in name for out in output_layer_names):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def load_financial_dataset():
    # Try known HF dataset ids; if not available, fallback to a synthetic dataset
    candidate_ids = [
        ("kalvez/financial_phrasebank", "sentences_50agree"),
        ("financial_phrasebank", "sentences_50agree"),
        ("financial_phrasebank", None),
    ]

    for ds_id, config in candidate_ids:
        try:
            if config is None:
                dataset = load_dataset(ds_id)
            else:
                dataset = load_dataset(ds_id, config)
            label_map = {"positive": 0, "negative": 1, "neutral": 2}

            def preprocess_function(example):
                return {"text": example.get("sentence", example.get("text", "")), "label": label_map[example.get("label")]}

            dataset = dataset.map(preprocess_function)
            return dataset
        except Exception:
            continue

    # Fallback synthetic dataset (small but representative) if HF dataset not accessible
    print("Warning: FinancialPhraseBank not available — using synthetic fallback dataset.")
    texts = [
        "The company reported strong quarterly earnings growth of 15%",
        "Sales declined due to market conditions",
        "Management expects EBITDA improvements next quarter",
        "The merger will likely improve market share",
        "Inventory write-downs reflect challenging market conditions",
        "Restructuring charges will impact Q3 results before expected recovery",
        "Revenue increased thanks to higher demand for core products",
        "Guidance downgraded due to supply chain disruptions",
        "Patent approval boosts long-term outlook",
        "Layoffs announced as part of cost-cutting measures",
    ]
    labels = [0, 1, 0, 0, 1, 2, 0, 1, 0, 1]

    # Repeat to create a larger sample (e.g., 200 examples)
    texts = (texts * 20)[:200]
    labels = (labels * 20)[:200]

    from datasets import Dataset as HFDataset

    return HFDataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    # Lightweight numpy implementation to avoid scikit-learn dependency
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels = np.array(labels)
    preds = np.array(preds)

    accuracy = float((preds == labels).sum() / len(labels))

    # Compute weighted precision, recall, f1
    classes, counts = np.unique(labels, return_counts=True)
    tp = np.array([int(((preds == c) & (labels == c)).sum()) for c in classes])
    pred_pos = np.array([int((preds == c).sum()) for c in classes])
    true_pos = np.array([int((labels == c).sum()) for c in classes])

    precision_per_class = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos != 0)
    recall_per_class = np.divide(tp, true_pos, out=np.zeros_like(tp, dtype=float), where=true_pos != 0)
    f1_per_class = np.divide(2 * precision_per_class * recall_per_class, precision_per_class + recall_per_class, out=np.zeros_like(tp, dtype=float), where=(precision_per_class + recall_per_class) != 0)

    weights = true_pos / true_pos.sum()
    weighted_precision = float((precision_per_class * weights).sum())
    weighted_recall = float((recall_per_class * weights).sum())
    weighted_f1 = float((f1_per_class * weights).sum())

    return {
        "accuracy": accuracy,
        "f1": weighted_f1,
        "precision": weighted_precision,
        "recall": weighted_recall,
    }


def run_experiment(method="lora", rho=0.0, seed=42, epochs=3, per_device_batch_size=8, model_name="prajjwal1/bert-tiny"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = load_financial_dataset()
    # dataset may be a DatasetDict (with 'train' key) or a single Dataset (fallback)
    if hasattr(dataset, "keys") and "train" in dataset:
        train_ds = dataset["train"]
        # prefer validation split if present, otherwise try test, otherwise split train
        if "validation" in dataset:
            val_ds = dataset["validation"]
        elif "test" in dataset:
            val_ds = dataset["test"]
        else:
            split = dataset["train"].train_test_split(test_size=0.2, seed=seed)
            train_ds = split["train"]
            val_ds = split["test"]
    else:
        # single Dataset -> split into train/val
        split = dataset.train_test_split(test_size=0.2, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]

    print(f"Using model: {model_name}  (CPU-friendly defaults)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    if method != "full":
        # Auto-detect candidate linear module names for this model to improve LoRA compatibility
        candidate_modules = find_all_linear_modules(model)
        if not candidate_modules:
            # Fallback conservative list
            candidate_modules = ["dense", "output", "query", "value", "key"]
        print(f"Applying LoRA to target modules: {candidate_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=candidate_modules,
            bias="none",
        )
        try:
            model = get_peft_model(model, peft_config)
            peft_applied = True
        except Exception as e:
            print("Warning: applying LoRA failed on chosen model. Error:", e)
            # Try fallback models known to be PEFT-friendly
            fallback_models = ["bert-base-uncased", "roberta-base"]
            peft_applied = False
            for fb in fallback_models:
                try:
                    print(f"Attempting to reload model '{fb}' and apply LoRA...")
                    tokenizer = AutoTokenizer.from_pretrained(fb)
                    model = AutoModelForSequenceClassification.from_pretrained(fb, num_labels=3)
                    candidate_modules = find_all_linear_modules(model)
                    if not candidate_modules:
                        candidate_modules = ["dense", "output", "query", "value", "key"]
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        r=8,
                        lora_alpha=16,
                        target_modules=candidate_modules,
                        bias="none",
                    )
                    model = get_peft_model(model, peft_config)
                    peft_applied = True
                    print(f"Successfully applied LoRA using fallback model: {fb}")
                    break
                except Exception as e2:
                    print(f"Fallback model {fb} failed: {e2}")
                    continue
    else:
        peft_applied = False

    # Tokenize datasets
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    output_dir = f"results/financial_sentiment/{method}_seed{seed}"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        logging_steps=50,
        seed=seed,
        load_best_model_at_end=False,
    )

    trainer_cls = FlatLoraTrainer if rho > 0 else Trainer

    # HF Trainer does not accept rho; only pass rho to FlatLoraTrainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if trainer_cls is FlatLoraTrainer:
        trainer = trainer_cls(**trainer_kwargs, rho=rho)
    else:
        trainer = trainer_cls(**trainer_kwargs)

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()

    # Save metrics and model
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=["full", "lora", "flat_lora"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny", help="Hugging Face model name (use a tiny model for CPU)")
    args = parser.parse_args()

    methods_cfg = {
        "full": ("full", 0.0),
        "lora": ("lora", 0.0),
        "flat_lora": ("flat_lora", 0.05),
    }

    results = {}
    for m in args.methods:
        method_name, rho = methods_cfg[m]
        results[method_name] = []
        for seed in args.seeds:
            print(f"Running {method_name} seed={seed}")
            metrics = run_experiment(
                method=method_name if method_name != 'flat_lora' else 'lora',
                rho=rho,
                seed=seed,
                epochs=args.epochs,
                per_device_batch_size=args.batch,
                model_name=args.model_name,
            )
            results[method_name].append({"seed": seed, "metrics": metrics})

    with open("financial_sentiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("All experiments finished. Results saved to financial_sentiment_results.json")


if __name__ == "__main__":
    main()
