#!/usr/bin/env python3
"""
ICE Evaluation for Large Language Models

This script evaluates explanation faithfulness for LLMs like Llama, Mistral, etc.
on text classification tasks.

Usage:
    python scripts/run_ice_llm.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset sst2 \
        --extractors llm_attention llm_gradient \
        --max_examples 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)

from ice import ICEEvaluator, ICEConfig, get_extractor
from data import get_eraser_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Evaluation for LLMs")
    
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",  # Small LLM for testing
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal",
        choices=["causal", "seq_cls"],
        help="Model type: causal (LLM) or seq_cls (classifier)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=[
            # English
            "sst2", "imdb", "esnli", "agnews",
            # Multilingual
            "chinese",    # ChnSentiCorp - Chinese sentiment
            "german",     # GermEval - German sentiment  
            "french",     # Allocine - French sentiment
            "hindi",      # IndicSentiment - Hindi sentiment
        ],
        help="Dataset to evaluate (supports English + Chinese/German/French/Hindi)"
    )
    parser.add_argument(
        "--extractors",
        type=str,
        nargs="+",
        default=["llm_attention", "llm_gradient"],
        help="Explanation methods to evaluate"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50,
        help="Maximum examples to evaluate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.2,
        help="Rationale budget"
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=50,
        help="Permutations for randomization test"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for large models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional HF dataset revision/commit hash for pinning (primary dataset repo)"
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Optional HF model revision/commit hash for reproducibility"
    )
    
    return parser.parse_args()


def format_classification_prompt(text: str, dataset: str) -> str:
    """Format text for classification with an LLM."""
    
    # English sentiment
    if dataset in ["sst2", "imdb"]:
        prompt = f"""Classify the sentiment of the following text as positive or negative.

Text: {text}

Sentiment:"""
    
    # English NLI
    elif dataset == "esnli":
        if "[SEP]" in text:
            premise, hypothesis = text.split("[SEP]")
            prompt = f"""Determine the relationship between the premise and hypothesis.
Options: entailment, neutral, contradiction

Premise: {premise.strip()}
Hypothesis: {hypothesis.strip()}

Relationship:"""
        else:
            prompt = f"""Classify this text.\n\nText: {text}\n\nLabel:"""
    
    # AG News (Topic Classification)
    elif dataset == "agnews":
        prompt = f"""Classify the topic of the following news article.
Options: World, Sports, Business, Technology

Article: {text}

Topic:"""
    
    # Chinese sentiment
    elif dataset == "chinese":
        prompt = f"""请判断以下文本的情感是正面还是负面。

文本: {text}

情感:"""
    
    # German sentiment
    elif dataset == "german":
        prompt = f"""Klassifizieren Sie die Stimmung des folgenden Textes als positiv oder negativ.

Text: {text}

Stimmung:"""
    
    # French sentiment
    elif dataset == "french":
        prompt = f"""Classifiez le sentiment du texte suivant comme positif ou négatif.

Texte: {text}

Sentiment:"""
    
    # Hindi sentiment
    elif dataset == "hindi":
        prompt = f"""निम्नलिखित पाठ की भावना को सकारात्मक या नकारात्मक के रूप में वर्गीकृत करें।

पाठ: {text}

भावना:"""
    
    else:
        prompt = f"""Classify the following text.\n\nText: {text}\n\nLabel:"""
    
    return prompt


def get_target_tokens(dataset: str, tokenizer) -> dict:
    """Get token IDs for target labels."""
    
    # English sentiment
    if dataset in ["sst2", "imdb"]:
        return {
            0: tokenizer.encode(" negative", add_special_tokens=False)[0],
            1: tokenizer.encode(" positive", add_special_tokens=False)[0],
        }
    # English NLI
    elif dataset == "esnli":
        return {
            0: tokenizer.encode(" entailment", add_special_tokens=False)[0],
            1: tokenizer.encode(" neutral", add_special_tokens=False)[0],
            2: tokenizer.encode(" contradiction", add_special_tokens=False)[0],
        }
    # AG News (Topic Classification)
    elif dataset == "agnews":
        return {
            0: tokenizer.encode(" World", add_special_tokens=False)[0],
            1: tokenizer.encode(" Sports", add_special_tokens=False)[0],
            2: tokenizer.encode(" Business", add_special_tokens=False)[0],
            3: tokenizer.encode(" Technology", add_special_tokens=False)[0],
        }
    # Chinese sentiment (negative=负面, positive=正面)
    elif dataset == "chinese":
        return {
            0: tokenizer.encode("负面", add_special_tokens=False)[0],
            1: tokenizer.encode("正面", add_special_tokens=False)[0],
        }
    # German sentiment
    elif dataset == "german":
        return {
            0: tokenizer.encode(" negativ", add_special_tokens=False)[0],
            1: tokenizer.encode(" positiv", add_special_tokens=False)[0],
        }
    # French sentiment
    elif dataset == "french":
        return {
            0: tokenizer.encode(" négatif", add_special_tokens=False)[0],
            1: tokenizer.encode(" positif", add_special_tokens=False)[0],
        }
    # Hindi sentiment (negative=नकारात्मक, positive=सकारात्मक)
    elif dataset == "hindi":
        return {
            0: tokenizer.encode("नकारात्मक", add_special_tokens=False)[0],
            1: tokenizer.encode("सकारात्मक", add_special_tokens=False)[0],
        }
    else:
        return {}


def evaluate_llm_example(
    model,
    tokenizer,
    text: str,
    true_label: int,
    extractor,
    scorer,
    operators,
    dataset: str,
    k: float,
    n_permutations: int,
    device: str
):
    """
    Evaluate a single example with an LLM.
    
    For causal LLMs, we use DELETION instead of masking:
    - Masking with pad tokens corrupts the sequence and produces constant outputs
    - Deletion keeps only the selected tokens, preserving sequence integrity
    """
    
    # Format as classification prompt
    prompt = format_classification_prompt(text, dataset)
    
    # Tokenize
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True
    )
    
    # Get the actual device from model (important for multi-GPU)
    if hasattr(model, 'device'):
        model_device = model.device
    else:
        # For models with device_map="auto", get device of first parameter
        model_device = next(model.parameters()).device
    
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)
    
    # Get target tokens for classification
    target_tokens = get_target_tokens(dataset, tokenizer)
    if not target_tokens:
        return None
    
    # Helper function to get prediction score
    def get_prediction_score(ids, mask):
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Check for NaN
            if torch.isnan(logits).any():
                return None
            
            target_logits = torch.tensor([logits[0, tid].item() for tid in target_tokens.values()])
            probs = torch.softmax(target_logits, dim=0)
            return probs
    
    # Get model prediction on original
    probs = get_prediction_score(input_ids, attention_mask)
    if probs is None:
        return None  # Skip if NaN
    
    predicted_class = probs.argmax().item()
    confidence = probs[predicted_class].item()
    
    # Skip if NaN or model is very uncertain
    if np.isnan(confidence) or confidence < 0.4:
        return None
    
    original_score = probs[predicted_class].item()
    
    # Get importance scores
    importance = extractor.get_importance_scores(
        input_ids.squeeze(0),
        attention_mask.squeeze(0),
        target_class=predicted_class
    )
    
    # Get valid positions (non-padding)
    valid_positions = [i for i in range(len(attention_mask[0])) if attention_mask[0, i] == 1]
    n_tokens = max(1, int(k * len(valid_positions)))
    
    # Get top-k indices by importance (sorted for deletion)
    valid_importance = [(i, importance[i].item()) for i in valid_positions]
    valid_importance.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = sorted([idx for idx, _ in valid_importance[:n_tokens]])
    
    # DELETION approach: Create new sequence with only rationale tokens
    rationale_ids = input_ids[0, top_k_indices].unsqueeze(0).to(model_device)
    rationale_mask = torch.ones(1, len(top_k_indices), device=model_device)
    
    # Get score with rationale only
    rationale_probs = get_prediction_score(rationale_ids, rationale_mask)
    if rationale_probs is None:
        return None  # Skip if NaN
    rationale_score = rationale_probs[predicted_class].item()
    
    # Compute random baseline scores using deletion
    random_scores = []
    
    for _ in range(n_permutations):
        # Random selection of same number of tokens
        random_indices = sorted(np.random.choice(valid_positions, size=n_tokens, replace=False))
        
        # Create sequence with only random tokens
        random_ids = input_ids[0, random_indices].unsqueeze(0).to(model_device)
        random_mask = torch.ones(1, len(random_indices), device=model_device)
        
        random_probs = get_prediction_score(random_ids, random_mask)
        if random_probs is None:
            continue  # Skip this permutation if NaN
        random_score = random_probs[predicted_class].item()
        if not np.isnan(random_score):
            random_scores.append(random_score)
    
    # Need at least 10 valid random scores
    if len(random_scores) < 10:
        return None
    
    random_scores = np.array(random_scores)
    
    # Compute metrics
    win_rate = np.mean(rationale_score > random_scores)
    
    # Handle edge case where all random scores are identical
    std_random = np.std(random_scores)
    if std_random < 1e-8:
        effect_size = 0.0 if abs(rationale_score - np.mean(random_scores)) < 1e-8 else np.sign(rationale_score - np.mean(random_scores)) * 10.0
    else:
        effect_size = (rationale_score - np.mean(random_scores)) / std_random
    
    p_value = (1 + np.sum(random_scores >= rationale_score)) / (n_permutations + 1)
    
    return {
        "rationale_score": rationale_score,
        "original_score": original_score,
        "win_rate": win_rate,
        "effect_size": effect_size,
        "p_value": p_value,
        "predicted_class": predicted_class,
        "true_label": true_label,
        "n_tokens": n_tokens,
        "random_score_std": std_random,
        "random_score_mean": np.mean(random_scores)
    }


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")
    
    # Determine if we need eager attention (for llm_attention extractor)
    needs_eager_attention = "llm_attention" in args.extractors
    skip_attention_extractor = False
    
    # Load model - try bfloat16 first (more stable), then float16, then float32
    model = None
    tokenizer = None
    
    dtype_attempts = [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ]
    
    # Try with eager attention first if needed, then fallback to without
    attention_configs = [True, False] if needs_eager_attention else [False]
    
    for use_eager in attention_configs:
        if model is not None:
            break
            
        eager_label = "with eager attention" if use_eager else "without eager attention"
        print(f"\n--- Attempting load {eager_label} ---")
        
        for dtype_name, dtype in dtype_attempts:
            print(f"Trying {dtype_name}...", end=" ")
            try:
                load_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                
                # Only set attn_implementation if using eager
                if use_eager:
                    load_kwargs["attn_implementation"] = "eager"
                
                if args.use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype
                    )
                    load_kwargs["quantization_config"] = quantization_config
                else:
                    load_kwargs["torch_dtype"] = dtype
                    if device != "cuda":
                        load_kwargs["device_map"] = None
                
                test_model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    **load_kwargs
                )
                
                test_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                
                # Quick test for NaN
                test_input = test_tokenizer("Hello world", return_tensors="pt")
                test_input = {k: v.to(test_model.device) for k, v in test_input.items()}
                
                with torch.no_grad():
                    outputs = test_model(**test_input)
                    if torch.isnan(outputs.logits).any():
                        print(f"NaN detected, trying next...")
                        del test_model
                        torch.cuda.empty_cache()
                        continue
                
                print(f"✓ Works!")
                model = test_model
                tokenizer = test_tokenizer
                
                # If we loaded without eager but needed it, skip attention extractor
                if needs_eager_attention and not use_eager:
                    skip_attention_extractor = True
                    print("⚠ Loaded without eager attention - llm_attention will be skipped")
                    args.extractors = [e for e in args.extractors if e != "llm_attention"]
                break
                
            except Exception as e:
                print(f"Failed: {e}")
                if 'test_model' in dir():
                    try:
                        del test_model
                    except:
                        pass
                torch.cuda.empty_cache()
                continue
    
    if model is None:
        print("ERROR: Could not load model with any dtype configuration!")
        print("Try a different model like 'gpt2' or 'facebook/opt-125m'")
        sys.exit(1)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"Model loaded: {model.__class__.__name__}")
    if needs_eager_attention:
        print("Using eager attention implementation for attention extraction")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    
    # Load raw dataset for text access
    from datasets import load_dataset
    ds_kwargs = {}
    if args.dataset_revision:
        ds_kwargs["revision"] = args.dataset_revision
    
    if args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2", split="validation", **ds_kwargs)
        texts = [ex["sentence"] for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb", split="test", **ds_kwargs)
        texts = [ex["text"][:500] for ex in dataset]  # Truncate long reviews
        labels = [ex["label"] for ex in dataset]
    elif args.dataset == "esnli":
        dataset = load_dataset("stanfordnlp/snli", split="test", **ds_kwargs)
        dataset = dataset.filter(lambda x: x["label"] != -1)
        texts = [f"{ex['premise']} [SEP] {ex['hypothesis']}" for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    
    # AG News (Topic Classification)
    elif args.dataset == "agnews":
        dataset = load_dataset("ag_news", split="test", **ds_kwargs)
        texts = [ex["text"][:400] for ex in dataset]  # Truncate
        labels = [ex["label"] for ex in dataset]  # 0=World, 1=Sports, 2=Business, 3=Tech
        print(f"  Task: Topic Classification (4-class)")
    
    # === MULTILINGUAL DATASETS ===
    
    # Chinese sentiment (using multilingual-sentiments)
    elif args.dataset == "chinese":
        try:
            dataset = load_dataset("tyqiangz/multilingual-sentiments", "chinese", split="test")
            texts = [ex["text"][:500] for ex in dataset]
            labels = [ex["label"] for ex in dataset]  # 0=neg, 1=neutral, 2=pos
            # Convert to binary (skip neutral)
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 1]
            texts = [t for t, _ in filtered]
            labels = [0 if l == 0 else 1 for _, l in filtered]
        except Exception as e:
            print(f"Error loading Chinese dataset: {e}")
            print("Trying alternative: c-s-ale/amazon_reviews_multi_zh...")
            dataset = load_dataset("amazon_reviews_multi", "zh", split="test")
            texts = [ex["review_body"][:500] for ex in dataset]
            labels = [1 if ex["stars"] >= 4 else 0 for ex in dataset]  # Binary sentiment
        print(f"  Language: Chinese (中文)")
    
    # German sentiment (using amazon reviews as GermEval may need special handling)
    elif args.dataset == "german":
        try:
            dataset = load_dataset("tyqiangz/multilingual-sentiments", "german", split="test")
            texts = [ex["text"][:500] for ex in dataset]
            labels = [ex["label"] for ex in dataset]  # 0=negative, 1=neutral, 2=positive
            # Convert to binary (0=negative, 1=positive, skip neutral)
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 1]
            texts = [t for t, _ in filtered]
            labels = [0 if l == 0 else 1 for _, l in filtered]
        except:
            # Fallback to smaller dataset
            dataset = load_dataset("SetFit/amazon_reviews_multi_de", split="test")
            texts = [ex["text"][:500] for ex in dataset]
            labels = [1 if ex["label"] >= 3 else 0 for ex in dataset]
        print(f"  Language: German (Deutsch)")
    
    # French sentiment (Allocine movie reviews)
    elif args.dataset == "french":
        try:
            dataset = load_dataset("tyqiangz/multilingual-sentiments", "french", split="test")
            texts = [ex["text"][:500] for ex in dataset]
            labels = [ex["label"] for ex in dataset]
            # Convert to binary
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 1]
            texts = [t for t, _ in filtered]
            labels = [0 if l == 0 else 1 for _, l in filtered]
        except:
            dataset = load_dataset("allocine", split="test")
            texts = [ex["review"][:500] for ex in dataset]
            labels = [ex["label"] for ex in dataset]
        print(f"  Language: French (Français)")
    
    # Hindi sentiment
    elif args.dataset == "hindi":
        try:
            dataset = load_dataset("tyqiangz/multilingual-sentiments", "hindi", split="test")
            texts = [ex["text"][:500] for ex in dataset]
            labels = [ex["label"] for ex in dataset]
            # Convert to binary
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 1]
            texts = [t for t, _ in filtered]
            labels = [0 if l == 0 else 1 for _, l in filtered]
        except Exception as e:
            print(f"Hindi dataset failed: {e}")
            print("Falling back to IIITH Hindi corpus...")
            # Fallback - you may need a different dataset
            texts = []
            labels = []
        print(f"  Language: Hindi (हिन्दी)")
    
    # Limit examples
    if args.max_examples:
        texts = texts[:args.max_examples]
        labels = labels[:args.max_examples]
    
    print(f"Loaded {len(texts)} examples")
    
    # Evaluate each extractor
    results = {}
    
    for extractor_name in args.extractors:
        print(f"\n{'='*60}")
        print(f"Evaluating: {extractor_name}")
        print('='*60)
        
        try:
            extractor = get_extractor(extractor_name, model, tokenizer, device)
        except Exception as e:
            print(f"Failed to create extractor {extractor_name}: {e}")
            continue
        
        example_results = []
        
        for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="ICE Evaluation")):
            try:
                result = evaluate_llm_example(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    true_label=label,
                    extractor=extractor,
                    scorer=None,
                    operators=None,
                    dataset=args.dataset,
                    k=args.k,
                    n_permutations=args.n_permutations,
                    device=device
                )
                
                if result is not None:
                    example_results.append(result)
                    
            except Exception as e:
                print(f"Error on example {i}: {e}")
                continue
        
        if not example_results:
            print(f"No valid results for {extractor_name}")
            continue
        
        # Aggregate results
        win_rates = [r["win_rate"] for r in example_results]
        effect_sizes = [r["effect_size"] for r in example_results]
        p_values = [r["p_value"] for r in example_results]
        random_stds = [r.get("random_score_std", 0) for r in example_results]
        
        mean_win_rate = np.mean(win_rates)
        mean_effect_size = np.mean([e for e in effect_sizes if not np.isnan(e) and abs(e) < 100])
        n_significant = np.sum(np.array(p_values) < 0.1)
        mean_random_std = np.mean(random_stds)
        
        print(f"\n{extractor_name} Results:")
        print(f"  Examples: {len(example_results)}")
        print(f"  Win Rate: {mean_win_rate*100:.1f}% (>50% = better than random)")
        print(f"  Effect Size: {mean_effect_size:.2f} (Cohen's d)")
        print(f"  Significant (p<0.1): {n_significant}/{len(example_results)} ({n_significant/len(example_results)*100:.1f}%)")
        print(f"  Random score std (diagnostic): {mean_random_std:.4f}")
        
        results[extractor_name] = {
            "n_examples": len(example_results),
            "win_rate": mean_win_rate,
            "effect_size": mean_effect_size,
            "n_significant": int(n_significant),
            "sig_rate": n_significant / len(example_results),
            "random_score_std": mean_random_std,
            "example_results": example_results
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("LLM FAITHFULNESS COMPARISON")
    print("="*80)
    print()
    
    header = "{:<20} {:>12} {:>12} {:>12}".format(
        "Method", "Win Rate", "Effect Size", "Sig. Rate"
    )
    print(header)
    print("-" * 60)
    
    for name, res in results.items():
        print("{:<20} {:>11.1f}% {:>12.2f} {:>11.1f}%".format(
            name,
            res["win_rate"] * 100,
            res["effect_size"],
            res["sig_rate"] * 100
        ))
    
    # Save results
    Path(args.output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"ice_llm_{args.dataset}_{timestamp}.json"
    
    # Convert to serializable format
    save_results = {
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "n_examples": len(texts),
            "k": args.k,
            "n_permutations": args.n_permutations,
            "timestamp": timestamp
        },
        "results": {
            name: {
                "n_examples": res["n_examples"],
                "win_rate": res["win_rate"],
                "effect_size": res["effect_size"],
                "sig_rate": res["sig_rate"]
            }
            for name, res in results.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
