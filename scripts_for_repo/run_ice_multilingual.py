#!/usr/bin/env python3
"""
ICE Evaluation for Multilingual LLMs

Dedicated script for multilingual experiments with:
- Gradient checkpointing for 7B+ models (memory optimization)
- Multi-token label support
- Dual GPU parallel execution (--dual_gpu flag)
- Focus on French, German, Hindi, Chinese

Usage:
    # Single GPU
    python scripts/run_ice_multilingual.py --model gpt2 --languages french german

    # Dual GPU (parallel)
    python scripts/run_ice_multilingual.py --model gpt2 --languages french german --dual_gpu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import torch
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configurations
LANGUAGES = ["french", "german", "hindi", "chinese", "turkish", "arabic", "de_native", "fr_native", "hi_native", "cn_native"]

# Dataset revisions for reproducibility
# Dataset revisions for reproducibility
DATASET_REVISIONS = {
    "multilingual-sentiments": "a3080a58e5631380b388dc572d",  # tyqiangz/multilingual-sentiments
    "germeval2017": "99da66e994364c565ff980960c83fc9039f81266",  # uhhlt/GermEval2017
    "allocine": "a4654f4896408912913a62ace89614879a549287",  # allocine
    "chnsenticorp": "b0c4c119c3fb33b8e735969202ef9ad13d7177e5a",  # lansinuote/ChnSentiCorp (parquet version)
    "indicsentiment": "dc8f3f66886531c6897fedffcae938a68fc5013",  # ai4bharat/IndicSentiment
}

# Native language datasets (non-translated, native text)
NATIVE_DATASETS = {
    "de_native": {
        "name": "uhhlt/GermEval2017",
        "split": "test_syn",
        "text_field": "Text",
        "label_field": "Sentiment",
        "revision_key": "germeval2017",
    },
    "fr_native": {
        "name": "allocine",
        "split": "test",
        "text_field": "review",
        "label_field": "label",
        "revision_key": "allocine",
    },
    "cn_native": {
        "name": "lansinuote/ChnSentiCorp",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "revision_key": "chnsenticorp",
    },
    "hi_native": {
        "name": "ai4bharat/IndicSentiment",
        "config": "translation-hi",
        "split": "test",
        "text_field": "INDIC REVIEW",  # Correct capitalization
        "label_field": "LABEL",        # Correct capitalization
        "revision_key": "indicsentiment",
    },
}

PROMPTS = {
    "french": """Classifiez le sentiment du texte suivant comme positif ou négatif.

Texte: {text}

Sentiment:""",
    "fr_native": """Classifiez le sentiment du texte suivant comme positif ou négatif.

Texte: {text}

Sentiment:""",
    "german": """Klassifizieren Sie die Stimmung des folgenden Textes als positiv oder negativ.

Text: {text}

Stimmung:""",
    "de_native": """Klassifizieren Sie die Stimmung des folgenden Textes als positiv oder negativ.

Text: {text}

Stimmung:""",
    "hindi": """निम्नलिखित पाठ की भावना को सकारात्मक या नकारात्मक के रूप में वर्गीकृत करें।

पाठ: {text}

भावना:""",
    "hi_native": """निम्नलिखित पाठ की भावना को सकारात्मक या नकारात्मक के रूप में वर्गीकृत करें।

पाठ: {text}

भावना:""",
    "chinese": """请判断以下文本的情感是正面还是负面。

文本: {text}

情感:""",
    "cn_native": """请判断以下文本的情感是正面还是负面。

文本: {text}

情感:""",
    "turkish": """Aşağıdaki metnin duygusunu pozitif veya negatif olarak sınıflandırın.

Metin: {text}

Duygu:""",
    "arabic": """صنف مشاعر النص التالي على أنه إيجابي أو سلبي.

النص: {text}

المشاعر:""",
}

LABELS = {
    "french": {0: " négatif", 1: " positif"},
    "fr_native": {0: " négatif", 1: " positif"},
    "german": {0: " negativ", 1: " positiv"},
    "de_native": {0: " negativ", 1: " positiv"},
    "hindi": {0: "नकारात्मक", 1: "सकारात्मक"},
    "hi_native": {0: "नकारात्मक", 1: "सकारात्मक"},
    "chinese": {0: "负面", 1: "正面"},
    "cn_native": {0: "负面", 1: "正面"},
    "turkish": {0: " negatif", 1: " pozitif"},
    "arabic": {0: " سلبي", 1: " إيجابي"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Multilingual Evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model")
    parser.add_argument("--languages", nargs="+", default=["french", "german"], 
                        choices=LANGUAGES, help="Languages to evaluate")
    parser.add_argument("--extractor", type=str, default="gradient", 
                        choices=["gradient", "attention"], help="Extraction method")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--k", type=float, default=0.2, help="Rationale budget")
    parser.add_argument("--n_permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_revision", type=str, default=None,
                        help="Optional HF dataset revision/commit hash for pinning (multilingual-sentiments)")
    parser.add_argument("--model_revision", type=str, default=None,
                        help="Optional HF model revision/commit hash for reproducibility")
    # Dual GPU flags
    parser.add_argument("--dual_gpu", action="store_true", help="Enable dual GPU parallel execution (different languages on different GPUs)")
    parser.add_argument("--model_parallel", action="store_true", help="Enable model parallelism (single model across 2 GPUs for 48GB VRAM)")
    parser.add_argument("--gpu_ids", nargs=2, type=int, default=[0, 1], help="GPU IDs to use (default: 0 1)")
    return parser.parse_args()


def load_germeval_dataset(max_examples: int = 100, dataset_revision: str = None):
    """Load GermEval 2017 dataset for native German sentiment.
    
    GermEval 2017 has sentiment labels: 'positive', 'negative', 'neutral'
    We filter to binary (positive/negative only).
    """
    try:
        ds_kwargs = {"split": "test_syn"}  # Use synchronic test set
        if dataset_revision:
            ds_kwargs["revision"] = dataset_revision
        else:
            ds_kwargs["revision"] = DATASET_REVISIONS["germeval2017"]
        
        ds = load_dataset("uhhlt/GermEval2017", **ds_kwargs)
        
        # Map sentiment labels: positive=1, negative=0, skip neutral
        sentiment_map = {"positive": 1, "negative": 0}
        filtered = []
        for ex in ds:
            sentiment = ex.get("Sentiment", "").lower()
            if sentiment in sentiment_map:
                text = ex.get("Text", "")[:400]
                if text:
                    filtered.append((text, sentiment_map[sentiment]))
        
        texts = [t for t, _ in filtered][:max_examples]
        labels = [l for _, l in filtered][:max_examples]
        
        print(f"Loaded {len(texts)} GermEval examples (native German)")
        return texts, labels
    except Exception as e:
        print(f"Failed to load GermEval: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def load_multilingual_dataset(lang: str, max_examples: int = 100, dataset_revision: str = None):
    """Load dataset for specified language."""
    # Handle native datasets (use NATIVE_DATASETS config)
    if lang in NATIVE_DATASETS:
        return load_native_dataset(lang, max_examples, dataset_revision)
    
    try:
        ds_kwargs = {"split": "test"}
        if dataset_revision:
            ds_kwargs["revision"] = dataset_revision
        ds = load_dataset("tyqiangz/multilingual-sentiments", lang, **ds_kwargs)
        # Filter to binary (skip neutral=1)
        filtered = [(ex["text"][:400], ex["label"]) for ex in ds if ex["label"] != 1]
        texts = [t for t, _ in filtered][:max_examples]
        labels = [0 if l == 0 else 1 for _, l in filtered][:max_examples]
        return texts, labels
    except Exception as e:
        print(f"Failed to load {lang}: {e}")
        return [], []


def load_native_dataset(lang: str, max_examples: int = 100, dataset_revision: str = None):
    """Load native language dataset using NATIVE_DATASETS config."""
    if lang not in NATIVE_DATASETS:
        print(f"Unknown native dataset: {lang}")
        return [], []
    
    config = NATIVE_DATASETS[lang]
    try:
        ds_kwargs = {"split": config["split"]}
        
        # Use pinned revision if available
        revision_key = config.get("revision_key")
        if dataset_revision:
            ds_kwargs["revision"] = dataset_revision
        elif revision_key and DATASET_REVISIONS.get(revision_key):
            ds_kwargs["revision"] = DATASET_REVISIONS[revision_key]
        
        # Load dataset with optional config
        if "config" in config:
            ds = load_dataset(config["name"], config["config"], **ds_kwargs)
        else:
            ds = load_dataset(config["name"], **ds_kwargs)
        
        text_field = config["text_field"]
        label_field = config["label_field"]
        
        # Process based on dataset type
        if lang == "de_native":
            # GermEval has text labels (positive/negative/neutral)
            sentiment_map = {"positive": 1, "negative": 0}
            filtered = []
            for ex in ds:
                sentiment = str(ex.get(label_field, "")).lower()
                if sentiment in sentiment_map:
                    text = str(ex.get(text_field, ""))[:400]
                    if text:
                        filtered.append((text, sentiment_map[sentiment]))
        elif lang == "hi_native":
            # IndicSentiment often has 0=Negative, 1=Neutral, 2=Positive
            # We map 2->1 (Pos), 0->0 (Neg), and skip 1 (Neutral)
            filtered = []
            for ex in ds:
                text = str(ex.get(text_field, ""))[:400]
                label = ex.get(label_field)
                
                # Robust checking (handle string or int)
                try:
                    l_str = str(label).strip()
                    if l_str in ["2", "positive", "Positive"]:  # Positive
                        filtered.append((text, 1))
                    elif l_str in ["0", "negative", "Negative"]: # Negative
                        filtered.append((text, 0))
                except:
                    pass
        else:
            # Other datasets have numeric labels (0=negative, 1=positive)
            filtered = []
            for ex in ds:
                text = str(ex.get(text_field, ""))[:400]
                label = ex.get(label_field)
                if text and label in [0, 1]:
                    filtered.append((text, label))
        
        texts = [t for t, _ in filtered][:max_examples]
        labels = [l for _, l in filtered][:max_examples]
        
        print(f"Loaded {len(texts)} {lang} examples (native)")
        return texts, labels
        
    except Exception as e:
        print(f"Failed to load {lang}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def get_label_token_ids(lang: str, tokenizer):
    """Get token IDs for labels (handles multi-token)."""
    label_map = LABELS[lang]
    return {k: tokenizer.encode(v, add_special_tokens=False) for k, v in label_map.items()}


def compute_label_prob(logits, label_token_ids, label_key):
    """Compute average probability across all tokens for a label."""
    token_ids = label_token_ids[label_key]
    avg_logit = sum(logits[tid].item() for tid in token_ids) / len(token_ids)
    return avg_logit


def get_prediction(logits, label_token_ids):
    """Get predicted label and probability."""
    sorted_keys = sorted(label_token_ids.keys())
    scores = [compute_label_prob(logits, label_token_ids, k) for k in sorted_keys]
    probs = torch.softmax(torch.tensor(scores), dim=0)
    pred_idx = probs.argmax().item()
    return sorted_keys[pred_idx], probs[pred_idx].item(), probs


def evaluate_example_gradient(model, tokenizer, text, lang, k, n_permutations, device):
    """Evaluate using gradient-based extraction."""
    prompt = PROMPTS[lang].format(text=text)
    label_token_ids = get_label_token_ids(lang, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
    
    predicted_key, confidence, probs = get_prediction(logits, label_token_ids)
    
    if confidence < 0.4:
        return None
    
    # Get importance scores via gradient
    with torch.no_grad():
        embeds = model.get_input_embeddings()(input_ids)
    
    embeds = embeds.detach().clone()
    embeds.requires_grad = True
    
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    target_token_id = label_token_ids[predicted_key][0]
    target_logit = outputs.logits[0, -1, target_token_id]
    target_logit.backward()
    
    importance = embeds.grad.abs().sum(dim=-1).squeeze()
    
    # Get top-k tokens
    valid_pos = list(range(len(importance)))
    n_tokens = max(1, int(k * len(valid_pos)))
    top_k = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:n_tokens]
    top_k = sorted(top_k)
    
    # Rationale score
    rationale_ids = input_ids[0, top_k].unsqueeze(0)
    rationale_mask = torch.ones(1, len(top_k), device=device)
    
    with torch.no_grad():
        rat_outputs = model(input_ids=rationale_ids, attention_mask=rationale_mask)
        rat_logits = rat_outputs.logits[0, -1, :]
    
    _, _, rat_probs = get_prediction(rat_logits, label_token_ids)
    rationale_score = rat_probs[predicted_key].item()
    
    # Random baselines
    random_scores = []
    for _ in range(n_permutations):
        rand_idx = sorted(np.random.choice(valid_pos, size=n_tokens, replace=False))
        rand_ids = input_ids[0, rand_idx].unsqueeze(0)
        rand_mask = torch.ones(1, len(rand_idx), device=device)
        
        with torch.no_grad():
            rand_outputs = model(input_ids=rand_ids, attention_mask=rand_mask)
            rand_logits = rand_outputs.logits[0, -1, :]
        
        _, _, rand_probs = get_prediction(rand_logits, label_token_ids)
        random_scores.append(rand_probs[predicted_key].item())
    
    win_rate = np.mean(rationale_score > np.array(random_scores))
    effect_size = (rationale_score - np.mean(random_scores)) / (np.std(random_scores) + 1e-8)
    
    return {"win_rate": win_rate, "effect_size": effect_size}


def evaluate_example_attention(model, tokenizer, text, lang, k, n_permutations, device):
    """Evaluate using attention-based extraction."""
    prompt = PROMPTS[lang].format(text=text)
    label_token_ids = get_label_token_ids(lang, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get prediction with attention outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits[0, -1, :]
    
    predicted_key, confidence, probs = get_prediction(logits, label_token_ids)
    
    if confidence < 0.4:
        return None
    
    # Get importance from attention (last layer, last token, average over heads)
    attentions = outputs.attentions[-1]  # Last layer
    attn_weights = attentions[0, :, -1, :].mean(dim=0)  # Average over heads, last token query
    importance = attn_weights.cpu()
    
    # Get top-k tokens
    valid_pos = list(range(len(importance)))
    n_tokens = max(1, int(k * len(valid_pos)))
    top_k = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:n_tokens]
    top_k = sorted(top_k)
    
    # Rationale score
    rationale_ids = input_ids[0, top_k].unsqueeze(0)
    rationale_mask = torch.ones(1, len(top_k), device=device)
    
    with torch.no_grad():
        rat_outputs = model(input_ids=rationale_ids, attention_mask=rationale_mask)
        rat_logits = rat_outputs.logits[0, -1, :]
    
    _, _, rat_probs = get_prediction(rat_logits, label_token_ids)
    rationale_score = rat_probs[predicted_key].item()
    
    # Random baselines
    random_scores = []
    for _ in range(n_permutations):
        rand_idx = sorted(np.random.choice(valid_pos, size=n_tokens, replace=False))
        rand_ids = input_ids[0, rand_idx].unsqueeze(0)
        rand_mask = torch.ones(1, len(rand_idx), device=device)
        
        with torch.no_grad():
            rand_outputs = model(input_ids=rand_ids, attention_mask=rand_mask)
            rand_logits = rand_outputs.logits[0, -1, :]
        
        _, _, rand_probs = get_prediction(rand_logits, label_token_ids)
        random_scores.append(rand_probs[predicted_key].item())
    
    win_rate = np.mean(rationale_score > np.array(random_scores))
    effect_size = (rationale_score - np.mean(random_scores)) / (np.std(random_scores) + 1e-8)
    
    return {"win_rate": win_rate, "effect_size": effect_size}


def run_language_on_gpu(gpu_id: int, model_name: str, lang: str, extractor: str,
                        max_examples: int, k: float, n_permutations: int, 
                        results_queue):
    """Run evaluation for a single language on a specific GPU."""
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"[GPU {gpu_id}] Loading {model_name} for {lang}...")
        
        # Load model on specific GPU
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": gpu_id},
            "trust_remote_code": True,
        }
        
        if extractor == "attention":
            model_kwargs["attn_implementation"] = "eager"
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Enable gradient checkpointing for 7B+ models
        if "7b" in model_name.lower() or "8b" in model_name.lower():
            print(f"[GPU {gpu_id}] Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
        
        model.eval()
        
        # Load dataset
        texts, labels = load_multilingual_dataset(lang, max_examples)
        if not texts:
            print(f"[GPU {gpu_id}] No data for {lang}")
            results_queue.put((lang, {"error": "no_data"}))
            return
        
        print(f"[GPU {gpu_id}] Evaluating {len(texts)} examples for {lang}...")
        
        evaluate_fn = evaluate_example_gradient if extractor == "gradient" else evaluate_example_attention
        
        win_rates = []
        effect_sizes = []
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc=f"GPU{gpu_id}:{lang}"):
            try:
                result = evaluate_fn(model, tokenizer, text, lang, k, n_permutations, device)
                if result:
                    win_rates.append(result["win_rate"])
                    effect_sizes.append(result["effect_size"])
            except Exception as e:
                continue
        
        if win_rates:
            result = {
                "win_rate": float(np.mean(win_rates)),
                "effect_size": float(np.mean(effect_sizes)),
                "n_examples": len(win_rates),
            }
            print(f"[GPU {gpu_id}] {lang}: Win Rate = {result['win_rate']*100:.1f}%")
        else:
            result = {"error": "no_valid_results"}
            print(f"[GPU {gpu_id}] {lang}: No valid results")
        
        results_queue.put((lang, result))
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        results_queue.put((lang, {"error": str(e)}))


def run_dual_gpu(args):
    """Run evaluation on dual GPUs in parallel."""
    print(f"\n=== DUAL GPU MODE ===")
    print(f"Using GPUs: {args.gpu_ids}")
    
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()
    
    all_results = {}
    
    # Process languages in pairs
    for i in range(0, len(args.languages), 2):
        processes = []
        
        # GPU 0
        if i < len(args.languages):
            lang0 = args.languages[i]
            p0 = mp.Process(
                target=run_language_on_gpu,
                args=(args.gpu_ids[0], args.model, lang0, args.extractor,
                      args.max_examples, args.k, args.n_permutations, results_queue)
            )
            processes.append(p0)
        
        # GPU 1
        if i + 1 < len(args.languages):
            lang1 = args.languages[i + 1]
            p1 = mp.Process(
                target=run_language_on_gpu,
                args=(args.gpu_ids[1], args.model, lang1, args.extractor,
                      args.max_examples, args.k, args.n_permutations, results_queue)
            )
            processes.append(p1)
        
        # Start and wait
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        # Collect results
        while not results_queue.empty():
            lang, result = results_queue.get()
            all_results[lang] = result
    
    return all_results


def run_single_gpu(args):
    """Run evaluation on single GPU sequentially."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"\nLoading model: {args.model}...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    # Model parallelism: spread model across multiple GPUs
    if args.model_parallel:
        print(f"=== MODEL PARALLEL MODE ===")
        print(f"Spreading model across GPUs: {args.gpu_ids}")
        # Set visible GPUs and use auto device mapping
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
    
    if args.extractor == "attention":
        model_kwargs["attn_implementation"] = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for 7B+ models
    if "7b" in args.model.lower() or "8b" in args.model.lower():
        print("Enabling gradient checkpointing for 7B+ model")
        model.gradient_checkpointing_enable()
    
    model.eval()
    model_device = next(model.parameters()).device
    
    all_results = {}
    
    for lang in args.languages:
        print(f"\n{'='*40}")
        print(f"Evaluating: {lang.upper()}")
        print(f"{'='*40}")
        
        texts, labels = load_multilingual_dataset(lang, args.max_examples)
        if not texts:
            print(f"No data for {lang}, skipping...")
            all_results[lang] = {"error": "no_data"}
            continue
        
        print(f"Loaded {len(texts)} examples")
        
        evaluate_fn = evaluate_example_gradient if args.extractor == "gradient" else evaluate_example_attention
        
        win_rates = []
        effect_sizes = []
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc=lang):
            try:
                result = evaluate_fn(model, tokenizer, text, lang, args.k, args.n_permutations, model_device)
                if result:
                    win_rates.append(result["win_rate"])
                    effect_sizes.append(result["effect_size"])
            except Exception as e:
                continue
        
        if win_rates:
            all_results[lang] = {
                "win_rate": float(np.mean(win_rates)),
                "effect_size": float(np.mean(effect_sizes)),
                "n_examples": len(win_rates),
            }
            print(f"\n{lang.upper()} Results:")
            print(f"  Win Rate: {all_results[lang]['win_rate']*100:.1f}%")
            print(f"  Effect Size: {all_results[lang]['effect_size']:.3f}")
        else:
            print(f"No valid results for {lang}")
            all_results[lang] = {"error": "no_valid_results"}
    
    return all_results


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("ICE MULTILINGUAL EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Languages: {args.languages}")
    print(f"Extractor: {args.extractor}")
    print(f"Dual GPU: {args.dual_gpu}")
    
    # Run evaluation
    if args.dual_gpu:
        all_results = run_dual_gpu(args)
    else:
        all_results = run_single_gpu(args)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "languages": args.languages,
            "dataset_revision": args.dataset_revision,
            "extractor": args.extractor,
            "k": args.k,
            "n_permutations": args.n_permutations,
            "max_examples": args.max_examples,
            "dual_gpu": args.dual_gpu,
            "timestamp": timestamp,
        },
        "results": all_results,
    }
    
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    langs_str = "_".join(args.languages)
    output_file = Path("results") / f"ice_multilingual_{model_short}_{langs_str}_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n| Language | Win Rate | Effect Size |")
    print(f"|----------|----------|-------------|")
    for lang in args.languages:
        r = all_results.get(lang, {})
        if "win_rate" in r:
            print(f"| {lang:8} | {r['win_rate']*100:6.1f}% | {r['effect_size']:+.3f}      |")
        else:
            print(f"| {lang:8} | FAILED   | -           |")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

