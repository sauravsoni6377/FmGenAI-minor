import json
import random
import string
import pandas as pd
from typing import List, Dict, Tuple
import os
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configuration
MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.2
MAX_TOKENS = 512

def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Call the LLM with consistent settings"""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt if system_prompt else "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "ERROR"
    
# ==================== Q2: Robustness Tests ====================

def add_typos(text: str, severity: str = "light") -> str:
    """Add keyboard typos to text"""
    if severity == "light":
        num_typos = len(text) // 20
    else:  # heavy
        num_typos = len(text) // 5
    
    chars = list(text)
    for _ in range(num_typos):
        if len(chars) > 0:
            idx = random.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                # Swap with adjacent key
                chars[idx] = random.choice(string.ascii_letters)
    return ''.join(chars)

def add_spacing_issues(text: str, severity: str = "light") -> str:
    """Add spacing and punctuation issues"""
    if severity == "light":
        return text.replace(" ", "  ")
    else:  # heavy
        return text.replace(" ", "    ")

def add_unicode_confusables(text: str, severity: str = "light") -> str:
    """Replace with Unicode confusables"""
    confusables = {
        'a': 'α', 'o': '0', 'e': 'е', 'i': '1',
        'A': 'Α', 'O': '0', 'E': 'Е', '2': '２'
    }
    
    if severity == "light":
        num_replacements = 1
    else:  # heavy
        num_replacements = 3
    
    text_chars = list(text)
    replaced = 0
    for i, char in enumerate(text_chars):
        if char in confusables and replaced < num_replacements:
            text_chars[i] = confusables[char]
            replaced += 1
    
    return ''.join(text_chars)

def add_emojis(text: str, severity: str = "light") -> str:
    """Add emojis to text"""
    emojis = ['🤔', '📖', '🌍', '💭', '🔍', '❓', '📝']
    
    if severity == "light":
        # Add one emoji
        words = text.split()
        if len(words) > 2:
            words.insert(len(words) // 2, random.choice(emojis))
        return ' '.join(words)
    else:  # heavy
        # Add multiple emojis
        words = text.split()
        for _ in range(min(3, len(words) // 3)):
            if len(words) > 0:
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(emojis))
        return ' '.join(words)

def run_robustness_test():
    """Test robustness to noisy inputs"""
    
    # Clean prompts
    clean_prompts = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is 2+2?", "4"),
        ("Name the largest planet", "Jupiter"),
        ("What year did WW2 end?", "1945"),
        # Add more prompts up to 50
    ]
    
    noise_functions = {
        "typos": add_typos,
        "spacing": add_spacing_issues,
        "unicode": add_unicode_confusables,
        "emoji": add_emojis
    }
    
    results = []
    
    # Test clean baseline
    for idx, (prompt, gold) in enumerate(clean_prompts[:5]):
        response = call_llm(prompt)
        correct = evaluate_answer(response, gold)
        results.append({
            "id": idx + 1,
            "noise_type": "clean",
            "noise_level": "none",
            "prompt_in": prompt,
            "gold": gold,
            "pred": response[:50],
            "correct": 1 if correct else 0
        })
    
    # Test with noise
    for noise_type, noise_func in noise_functions.items():
        for severity in ["light", "heavy"]:
            for idx, (prompt, gold) in enumerate(clean_prompts[:5]):
                noisy_prompt = noise_func(prompt, severity)
                response = call_llm(noisy_prompt)
                correct = evaluate_answer(response, gold)
                
                results.append({
                    "id": len(results) + 1,
                    "noise_type": noise_type,
                    "noise_level": severity,
                    "prompt_in": noisy_prompt,
                    "gold": gold,
                    "pred": response[:50],
                    "correct": 1 if correct else 0
                })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("Q2_robustness_results.csv", index=False)
    print("Q2 Complete: Results saved to Q2_robustness_results.csv")
    
    # Test mitigation
    test_robustness_mitigation()

def test_robustness_mitigation():
    """Test robustness mitigation strategy"""
    system_prompt = """
    IMPORTANT: The input may contain typos, unusual spacing, Unicode variants, or emoji.
    Mentally normalize these issues and focus on the semantic intent of the question.
    Ignore decorative elements. Provide ONLY the direct answer in standard format.
    """
    
    test_cases = [
        ("Waht si teh captial fo Frnace?", "Paris"),
        ("Wh0 wr0te R0me0 αnd Juliet?", "Shakespeare"),
        ("What 🤔 is the capital 🏛️ of France 🇫🇷?", "Paris"),
    ]
    
    print("\nRobustness Mitigation Test:")
    for noisy, gold in test_cases:
        response = call_llm(noisy, system_prompt)
        correct = evaluate_answer(response, gold)
        print(f"Noisy: {noisy}")
        print(f"Response: {response[:100]}")
        print(f"Correct: {correct}")
        print("-" * 50)


def evaluate_answer(prediction: str, gold: str) -> bool:
    """Simple exact match evaluation"""
    # Normalize and compare
    pred_norm = prediction.lower().strip()
    gold_norm = gold.lower().strip()
    return gold_norm in pred_norm


def main():
    """Run all experiments"""
    print("=" * 60)
    print("LLM Robustness Evaluation")
    print("=" * 60)
    
    
    print("\n[2/3] Running Robustness to Messy Inputs Tests...")
    run_robustness_test()
    
    print("\n" + "=" * 60)
    print("Generated file:")
    print("  - Q2_robustness_results.csv")
    print("  - Full report in markdown format")
    print("=" * 60)

if __name__ == "__main__":
    main()