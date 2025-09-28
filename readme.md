
# Foundation Models and GenAI Minor Examination 

### **Saurav Soni - (B22AI035)**


Final Report: https://github.com/sauravsoni6377/FmGenAI-minor/blob/main/Minor-Saurav%20Soni-Flat-LoRA%3A%20Low-Rank%20Adaptation%20over%20a%20Flat%20Loss%20Landscape.pdf

---
## Flat-LoRA - Part A

## Overview

This repository contains the implementation and evaluation of Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape for the Foundational Models and Generative AI course mid-semester examination. The submission includes comprehensive experiments, sanity checks, and domain-shift evaluation on financial sentiment analysis.

## code Files created for Part - A 

```
├── run_financial_sentiment.py     # Main experiment script for Q4
├── analyze_results.py            # Results analysis and statistics
├── error_analysis.py             # Qualitative error analysis
├── sanity_flatlora.py           # Q3 sanity check script
├── financial_sentiment_results.json  # Experimental results (generated)
└── README.md                     # This file
```

## Requirements

```bash
# Core dependencies
pip install torch transformers datasets peft accelerate

# For statistical analysis
pip install scipy numpy

# For wandb logging (optional)
pip install wandb

# Or

# Create Virtual Environment
python -m venv my_project_venv
# macOs/linux
source my_project_venv/bin/activate

# install packacges in requirements.txt
pip install -r requirements.txt #(present in PART - A)
```

## Quick Start

### 1. Sanity Check (Q3)

Run the minimal reproduction to verify Flat-LoRA functionality:

```bash
WANDB_MODE=offline python sanity_flatlora.py
```

**Expected Output:**
- Flat-LoRA trainer activation with rho=0.1
- Training completion with loss curves
- Basic inference on toy examples

### 2. Financial Sentiment Experiment (Q4)

Run the main experiment comparing Full FT, LoRA, and Flat-LoRA:

```bash
# Quick test with small model (CPU-friendly)
WANDB_MODE=offline python run_financial_sentiment.py \
  --methods full lora flat_lora \
  --seeds 42 \
  --epochs 3 \
  --batch 8 \
  --model_name prajjwal1/bert-tiny

# Full experiment with proper model
WANDB_MODE=offline python run_financial_sentiment.py \
  --methods full lora flat_lora \
  --seeds 42 43 44 \
  --epochs 5 \
  --batch 8 \
  --model_name bert-base-uncased
```

### 3. Results Analysis

Generate performance tables and statistical tests:

```bash
python analyze_results.py
```

### 4. Error Analysis

Perform qualitative analysis on difficult examples:

```bash
python error_analysis.py
```

## Key Findings

### Q1: Executive Summary

- **Problem:** LoRA optimizes in low-dimensional subspace but may find sharp minima in full parameter space
- **Solution:** Flat-LoRA uses Bayesian expected loss with random perturbations to find flat minima
- **Contribution:** Memory-efficient flatness optimization without SAM's computational overhead
- **Limitation:** Introduces perturbation hyperparameter (rho) requiring tuning

### Q2: Method Deep Dive

- **Architecture:** Training methodology built on standard LoRA framework
- **Objective:** min_{A,B} 𝔼_ε[L(W + BA + ε_W)]
- **Evidence:** Ablation shows full-space optimization outperforms LoRA-space optimization
- **Compute:** Scales from T5-Base (~7.5h on RTX 4090) to SDXL models

### Q3: Sanity Check Results

- ✅ **Success:** Flat-LoRA code path exercised with rho=0.1
- ✅ **Training:** End-to-end execution without errors
- ⚠️ **Limitation:** Toy dataset showed input echoing rather than target transformation
- 📊 **Metrics:** Evaluation accuracy reached 1.0 on validation set

### Q4: Financial Sentiment Results

| Method    | Accuracy      | F1-Score      |
|-----------|---------------|---------------|
| Full FT   | 0.718 ± 0.012 | 0.712 ± 0.014 |
| LoRA      | 0.705 ± 0.015 | 0.697 ± 0.013 |
| Flat-LoRA | 0.742 ± 0.010 | 0.736 ± 0.011 |

**Statistical Significance:** Flat-LoRA vs LoRA (p < 0.05)

## Error Analysis Findings

### Success Cases

Flat-LoRA better handles nuanced financial language:

> "The merger announcement failed to impress investors despite projected synergies"
> - **LoRA:** Neutral (incorrect)
> - **Flat-LoRA:** Negative (correct)

### Failure Modes

#### 1. Numerical Context Misinterpretation
- **Example:** "Revenue grew 2% but missed expectations of 5%"
- **Error:** Models overweight "grew" keyword

#### 2. Financial Jargon Ambiguity
- **Example:** "The company took a write-down on assets"
- **Error:** Inconsistent negative/neutral classification

#### 3. Forward-Looking Statement Complexity
- **Example:** "Guidance suggests potential recovery in H2 despite current challenges"
- **Error:** Struggle with mixed temporal sentiment

## Configuration Details

### Model Settings

- **Base Model:** bert-base-uncased (recommended) or prajjwal1/bert-tiny (CPU-testing)
- **LoRA Config:** rank=8, alpha=16, target modules auto-detected
- **Flat-LoRA:** rho=0.05 (perturbation strength)

### Training Parameters

- **Learning Rate:** 5e-5
- **Batch Size:** 8 (per device)
- **Epochs:** 5 (recommended), 3 (quick test)
- **Max Length:** 256 tokens
- **Evaluation:** Accuracy, weighted F1, precision, recall

## Dataset Information

### Financial PhraseBank

- **Source:** Hugging Face kalvez/financial_phrasebank
- **License:** MIT
- **Task:** 3-class sentiment (positive/negative/neutral)
- **Samples:** ~5,000 financial news sentences
- **Domain Shift:** Specialized financial vocabulary and business context

## Reproducibility

All experiments use fixed seeds (42, 43, 44) for deterministic results. The code includes:

- Automatic dataset fallback (synthetic data if HF unavailable)
- Module auto-detection for LoRA compatibility
- Comprehensive metrics without external dependencies



## LLM Evaluation - Part - B

## Files Included

### 1. Main Report
- **`Report.pdf`** - Complete report with all findings, tables, and analysis for both Part A and B.

### 2. Data Files (CSVs)
- **`Q1_multilingual_results.csv`** - Multilingual & code-switching test results (80 rows)
- **`Q2_robustness_results.csv`** - Robustness to messy inputs results (50 rows)

### 3. Code
- **`run_multilingual_test.py`** - Python script to reproduce Q1 experiments.
- **`run_robustness_test.py`** - Python script to reproduce Q2 experiments.
- **`requirements.txt`** - Python dependencies

### 4. Documentation
- **`README.md`** - This file

## How to Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set API key:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

3. **Run experiments:**
```bash
python3 run_multilingual_test.py
python3 run_robustness_test.py
```

## Requirements File Content
```txt
anthropic>=0.25.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Key Findings Summary

### Q1: Multilingual & Code-Switch
- **Best Performance**: English (92.5% accuracy)
- **Worst Performance**: Code-switching (63.0% accuracy)
- **Mitigation**: Language pinning improved accuracy by 16.6%

### Q2: Robustness to Messy Inputs
- **Most Robust**: Emoji noise (91% with light noise)
- **Least Robust**: Unicode confusables (71% with heavy noise)
- **Mitigation**: Robustness prompt template improved heavy noise handling by 15%


## Methodology Notes

- **Model**: Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
- **Temperature**: 0.2 (consistent across all experiments)
- **Max Tokens**: 512
- **Evaluation**: Automatic metrics with manual fluency ratings on subset
- **Ethics**: All prompts culturally neutral, no stereotypes or sensitive content

## Statistical Notes

- 95% confidence intervals calculated using bootstrap method (1000 iterations)
- Each stochastic experiment run 3 times and averaged
- Error bars included in main report tables


## Declaration

I declare that this work is my own, except where explicitly stated. LLM assistance was used for code formatting and CSV generation as documented in the methods section.
