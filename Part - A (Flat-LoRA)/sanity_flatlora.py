import os
import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from utils import initialize_text_to_text_model, train_text_to_text_model, model_inference


def main():
    # Tiny dataset (toy mapping) that is easy to overfit
    examples = {
        "x": [
            "repeat: hello",
            "repeat: world",
            "repeat: foo",
            "repeat: bar",
        ],
        "y": [
            "hello hello",
            "world world",
            "foo foo",
            "bar bar",
        ],
    }
    ds = Dataset.from_dict(examples)
    # split into train/val (use same small set for both to check overfitting)
    train_ds = ds
    val_ds = ds

    model_name = "t5-small"
    model_type = "ConditionalGeneration"

    print("Loading base model and tokenizer...")
    model, tokenizer = initialize_text_to_text_model(model_name, model_type, bf16=False, use_peft=False)

    # Create a small LoRA config and wrap model
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q", "v", "k", "o", "wi", "wo", "dense"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    print("Wrapping model with LoRA adapters...")
    model = get_peft_model(model, peft_config)

    # Run a tiny training with rho>0 to enable Flat-LoRA trainer
    print("Starting tiny Flat-LoRA training (rho=0.1)...")
    model = train_text_to_text_model(
        "sanity_test/small_run",
        train_ds,
        val_ds,
        model,
        tokenizer,
        model_type,
        num_train_epochs=3,
        per_device_batch_size=1,
        real_batch_size=1,
        max_length=64,
        bf16=False,
        seed=42,
        rho=0.1,
        logging_steps=1,
        eval_epochs=1,
        early_stopping_patience=1,
    )

    # Do a before/after style inference: use the trained model to predict
    print("Running inference on toy inputs:")
    for inp in examples["x"]:
        out = model_inference(model, tokenizer, inp, model_type, max_source_length=64, max_target_length=16)
        print(f"Input: {inp} -> Pred: {out}")


if __name__ == "__main__":
    main()
