# LLM Evaluation Assignment - Submission Package

## Author
Saurav Soni
B22AI035
Date: September 28, 2025

## Files Included

### 1. Main Report
- **`Report.md`** - Complete report with all findings, tables, and analysis for both Part A and B.

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