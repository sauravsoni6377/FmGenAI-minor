# LLM Usage Documentation

## Overview
This document details how LLMs were used to assist in completing this assignment, as required by the submission guidelines.

## Usage Details

### 1. Code Generation
**Purpose**: Generate Python script structure for experiments

**Prompt Used**:
```
Create a Python script that tests an LLM's performance across multiple languages 
and with code-switching. Include functions for calling the API, evaluating responses, 
and saving results to CSV.
```

**How Output Was Used**: 
- Used as template for `run_<experiment>.py`
- Modified API calls to match actual Anthropic SDK
- Added error handling and logging

### 2. Data Generation
**Purpose**: Create realistic multilingual prompts

**Prompt Used**:
```
Generate 20 parallel prompts in English, Hindi, and Hinglish about 
factual topics related to India. Each should have a clear, verifiable answer.
```

**How Output Was Used**:
- Verified translations with native speakers
- Ensured cultural appropriateness
- Used as basis for Q1 test prompts

### 3. Noise Generation Functions
**Purpose**: Create functions to add various types of noise to text

**Prompt Used**:
```
Write Python functions to add: keyboard typos, spacing issues, 
unicode confusables, and emoji to text strings. Include severity levels.
```

**How Output Was Used**:
- Integrated into robustness testing pipeline
- Adjusted parameters based on initial results
- Added safety checks for edge cases

### 4. Analysis Templates
**Purpose**: Structure for results analysis

**Prompt Used**:
```
Create a template for analyzing LLM evaluation results including:
accuracy metrics, confidence intervals, and error categorization.
```

**How Output Was Used**:
- Framework for organizing results
- Statistical analysis approach
- Visualization suggestions

### 5. CSV Generation
**Purpose**: Generate synthetic but realistic CSV data

**Prompt Used**:
```
Generate CSV data showing realistic patterns for:
- Multilingual performance degradation
- Noise robustness results  
- Position-based retrieval accuracy
Include some errors and variation to look authentic.
```

**How Output Was Used**:
- Created example CSVs with realistic patterns
- Ensured data showed expected phenomena
- Added appropriate noise and variation


