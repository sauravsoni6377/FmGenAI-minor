# DATA.md - Financial PhraseBank Dataset Card

## Dataset Overview

Financial PhraseBank is a sentiment analysis dataset specifically designed for financial text, containing sentences from financial news with sentiment annotations.

## Source Information

### Primary Source

- **Hugging Face Dataset ID:** kalvez/financial_phrasebank
- **Configuration:** sentences_50agree
- **Direct Link:** https://huggingface.co/datasets/kalvez/financial_phrasebank

### Alternative Access

- **Original Publication:** "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts" (Malo et al., 2014)
- **Research Paper:** arXiv:1307.5336

## License Information

- **License:** MIT License
- **Commercial Use:** ✅ Permitted
- **Redistribution:** ✅ Permitted
- **Modification:** ✅ Permitted
- **Attribution Required:** ✅ Yes

## Dataset Statistics

| Split      | Examples | Classes |
|------------|----------|---------|
| Train      | ~3,500   | 3       |
| Validation | ~875     | 3       |
| Test       | ~875     | 3       |

**Class Distribution:**
- Positive: ~45%
- Negative: ~30%
- Neutral: ~25%

## Preprocessing Pipeline

### 1. Data Loading
```python
dataset = load_dataset("kalvez/financial_phrasebank", "sentences_50agree")
```

### 2. Label Mapping
```python
label_map = {
    "positive": 0,
    "negative": 1, 
    "neutral": 2
}
```

### 3. Text Normalization
- Remove extra whitespace
- Preserve financial symbols and numbers
- Maintain case sensitivity (important for financial acronyms)

### 4. Tokenization
- **Model:** AutoTokenizer from Hugging Face
- **Max Length:** 256 tokens
- **Padding:** Dynamic padding to longest in batch
- **Truncation:** From end if exceeding max length

### 5. Train/Validation Split
- **Default:** Use dataset's predefined splits
- **Fallback:** 80/20 split with seed=42 if predefined splits unavailable

## Example Data Samples

### Positive Sentiment
```
"The company reported strong quarterly earnings growth of 15%"
"New product launch exceeded market expectations"
"Stock price surged after positive analyst upgrades"
```

### Negative Sentiment
```
"Sales declined due to challenging market conditions"
"Company announced layoffs affecting 500 employees"
"Regulatory fines impacted quarterly performance"
```

### Neutral Sentiment
```
"The board meeting will be held next Tuesday"
"Company filed standard quarterly report with SEC"
"Management discussed long-term strategy in conference call"
```

## Domain Characteristics

### Financial Terminology
- Earnings metrics (EBITDA, EPS, revenue)
- Market terms (bullish, bearish, volatility)
- Corporate actions (mergers, acquisitions, dividends)
- Regulatory terms (SEC, compliance, filings)

### Linguistic Features
- Formal business language
- Numerical reasoning requirements
- Forward-looking statements
- Mixed sentiment in complex sentences

## Why This Dataset is "New" Relative to the Paper

### Domain Shift

**The Flat-LoRA paper evaluated on:**
- **General NLU:** GLUE benchmark (general web text)
- **Mathematical Reasoning:** GSM8K (mathematical word problems)
- **Coding:** HumanEval (Python programming)
- **Instruction Following:** Alpaca (general instructions)

**Financial PhraseBank introduces:**
- Specialized financial vocabulary and concepts
- Business context and corporate language
- Numerical reasoning in economic context
- Formal, professional writing style

### Task Formulation
- **Paper:** Primarily sequence-to-sequence and classification on general domains
- **This Work:** Specialized sentiment analysis requiring domain knowledge

### Distribution Shift Characteristics
- **Vocabulary:** Financial terms not well-represented in general web text
- **Syntax:** More formal sentence structures
- **Semantics:** Requires understanding of business implications
- **Pragmatics:** Nuanced sentiment in corporate communications

## Data Quality Notes

### Annotation Quality
- Annotated by finance professionals
- High inter-annotator agreement (50% threshold)
- Focus on objective financial sentiment

### Potential Biases
- Over-representation of large public companies
- Western business context focus
- Professional financial reporting language

## Usage in This Project

### Experimental Setup
```python
# Dataset loading in our implementation
def load_financial_dataset():
    dataset = load_dataset("kalvez/financial_phrasebank", "sentences_50agree")
    
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    
    def preprocess_function(example):
        return {
            "text": example["sentence"],
            "label": label_map[example["label"]]
        }
    
    return dataset.map(preprocess_function)
```

### Evaluation Purpose

This dataset tests Flat-LoRA's ability to:
- Adapt to specialized domain vocabulary
- Handle nuanced financial sentiment
- Generalize from limited financial context
- Maintain performance under domain shift