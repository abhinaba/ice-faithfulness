#!/usr/bin/env python3
"""
ICE Evaluation with Retrieval Infill Operator
==============================================

NEW VERSION: Adds Retrieval Infill as an alternative to Deletion.

This runner extends run_ice_llm_nsr.py with:
1. Retrieval Infill operator (distribution-matched intervention)
2. Operator comparison mode: deletion vs retrieval
3. Sentiment scrub option for ablation

Usage:
    python scripts/run_ice_llm_retrieval.py \
        --model microsoft/phi-2 \
        --dataset sst2 \
        --extractors llm_attention llm_gradient \
        --operators deletion retrieval \
        --max_examples 100

For the original behavior (deletion only), use run_ice_llm_nsr.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import retrieval operator
from ice.retrieval_operator import RetrievalPool


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Evaluation with Retrieval Infill")
    
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["sst2", "imdb", "esnli", "agnews"])
    parser.add_argument("--extractors", type=str, nargs="+", 
                       default=["llm_attention", "llm_gradient"])
    parser.add_argument("--operators", type=str, nargs="+",
                       default=["deletion", "retrieval"],
                       help="Operators to compare: deletion, retrieval, or both")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--k", type=float, default=0.2)
    parser.add_argument("--n_permutations", type=int, default=50)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/retrieval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    
    # Retrieval-specific options
    parser.add_argument("--sentiment_scrub", action="store_true",
                       help="Also filter sentiment-bearing words (not just labels)")
    
    return parser.parse_args()


def format_classification_prompt(text: str, dataset: str) -> str:
    """Format text for classification with an LLM."""
    if dataset in ["sst2", "imdb"]:
        return f"""Classify the sentiment of the following text as positive or negative.

Text: {text}

Sentiment:"""
    elif dataset == "esnli":
        if "[SEP]" in text:
            premise, hypothesis = text.split("[SEP]")
            return f"""Determine the relationship between the premise and hypothesis.
Options: entailment, neutral, contradiction

Premise: {premise.strip()}
Hypothesis: {hypothesis.strip()}

Relationship:"""
        return f"Classify this text.\n\nText: {text}\n\nLabel:"
    elif dataset == "agnews":
        return f"""Classify the topic of the following news article.
Options: World, Sports, Business, Technology

Article: {text}

Topic:"""
    return f"Classify the following text.\n\nText: {text}\n\nLabel:"


def get_target_tokens(dataset: str, tokenizer) -> dict:
    """Get token IDs for target labels."""
    if dataset in ["sst2", "imdb"]:
        return {
            0: tokenizer.encode(" negative", add_special_tokens=False)[0],
            1: tokenizer.encode(" positive", add_special_tokens=False)[0],
        }
    elif dataset == "esnli":
        return {
            0: tokenizer.encode(" entailment", add_special_tokens=False)[0],
            1: tokenizer.encode(" neutral", add_special_tokens=False)[0],
            2: tokenizer.encode(" contradiction", add_special_tokens=False)[0],
        }
    elif dataset == "agnews":
        return {
            0: tokenizer.encode(" World", add_special_tokens=False)[0],
            1: tokenizer.encode(" Sports", add_special_tokens=False)[0],
            2: tokenizer.encode(" Business", add_special_tokens=False)[0],
            3: tokenizer.encode(" Technology", add_special_tokens=False)[0],
        }
    return {}


def evaluate_with_deletion(
    input_ids, attention_mask, model, tokenizer,
    importance, k, n_permutations, predicted_class, target_tokens, device
):
    """Original deletion-based evaluation."""
    valid_positions = [i for i in range(len(attention_mask)) if attention_mask[i] == 1]
    n_tokens = max(1, int(k * len(valid_positions)))
    
    # Get top-k by importance
    valid_importance = [(i, importance[i].item()) for i in valid_positions]
    valid_importance.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = sorted([idx for idx, _ in valid_importance[:n_tokens]])
    
    # Helper to get score
    def get_score(ids, mask):
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits[:, -1, :]
            if torch.isnan(logits).any():
                return None
            target_logits = torch.tensor([logits[0, tid].item() for tid in target_tokens.values()])
            probs = torch.softmax(target_logits, dim=0)
            return probs[predicted_class].item()
    
    # Rationale score (deletion = keep only rationale tokens)
    rationale_ids = input_ids[top_k_indices].unsqueeze(0).to(device)
    rationale_mask = torch.ones(1, len(top_k_indices), device=device)
    rationale_score = get_score(rationale_ids, rationale_mask)
    
    if rationale_score is None:
        return None
    
    # Random baseline scores
    random_scores = []
    for _ in range(n_permutations):
        random_indices = sorted(np.random.choice(valid_positions, size=n_tokens, replace=False))
        random_ids = input_ids[random_indices].unsqueeze(0).to(device)
        random_mask = torch.ones(1, len(random_indices), device=device)
        score = get_score(random_ids, random_mask)
        if score is not None and not np.isnan(score):
            random_scores.append(score)
    
    if len(random_scores) < 10:
        return None
    
    random_scores = np.array(random_scores)
    win_rate = np.mean(rationale_score > random_scores)
    
    return {
        "operator": "deletion",
        "rationale_score": rationale_score,
        "win_rate": win_rate,
        "random_score_mean": np.mean(random_scores),
    }


def evaluate_with_retrieval(
    input_ids, attention_mask, model, tokenizer,
    importance, k, n_permutations, predicted_class, target_tokens, device,
    retrieval_pool, example_id
):
    """Retrieval-based evaluation (NEW)."""
    valid_positions = [i for i in range(len(attention_mask)) if attention_mask[i] == 1]
    n_tokens = max(1, int(k * len(valid_positions)))
    
    # Get top-k by importance
    valid_importance = [(i, importance[i].item()) for i in valid_positions]
    valid_importance.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = set([idx for idx, _ in valid_importance[:n_tokens]])

    # Helper to get score
    def get_score(ids, mask):
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits[:, -1, :]
            if torch.isnan(logits).any():
                return None
            target_logits = torch.tensor([logits[0, tid].item() for tid in target_tokens.values()])
            probs = torch.softmax(target_logits, dim=0)
            return probs[predicted_class].item()
    
    # Rationale score: replace NON-rationale with retrieved tokens
    retrieved_ids = input_ids.clone()
    non_rationale_positions = [i for i in valid_positions if i not in top_k_indices]
    
    if len(non_rationale_positions) > 0:
        replacement_tokens = retrieval_pool.sample_tokens(
            len(non_rationale_positions), 
            exclude_example_id=example_id
        )
        for i, pos in enumerate(non_rationale_positions):
            retrieved_ids[pos] = replacement_tokens[i]
    
    rationale_score = get_score(retrieved_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    
    if rationale_score is None:
        return None
    
    # Random baseline: replace rationale with retrieved tokens
    random_scores = []
    for _ in range(n_permutations):
        random_indices = set(np.random.choice(valid_positions, size=n_tokens, replace=False))
        random_ids = input_ids.clone()
        
        # Replace non-random positions with retrieved tokens
        positions_to_replace = [i for i in valid_positions if i not in random_indices]
        if len(positions_to_replace) > 0:
            replacement_tokens = retrieval_pool.sample_tokens(
                len(positions_to_replace), 
                exclude_example_id=example_id
            )
            for i, pos in enumerate(positions_to_replace):
                random_ids[pos] = replacement_tokens[i]
        
        score = get_score(random_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        if score is not None and not np.isnan(score):
            random_scores.append(score)
    
    if len(random_scores) < 10:
        return None
    
    random_scores = np.array(random_scores)
    win_rate = np.mean(rationale_score > random_scores)
    
    return {
        "operator": "retrieval",
        "rationale_score": rationale_score,
        "win_rate": win_rate,
        "random_score_mean": np.mean(random_scores),
    }


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("ICE Evaluation with Retrieval Infill")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Operators: {args.operators}")
    print(f"Sentiment Scrub: {args.sentiment_scrub}")
    print(f"k={args.k}, n_permutations={args.n_permutations}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.gpu},
        trust_remote_code=True,
        attn_implementation="eager"  # For attention extraction
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    from datasets import load_dataset
    
    if args.dataset == "sst2":
        dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
        texts = [ex["sentence"] for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    elif args.dataset == "imdb":
        dataset = load_dataset("stanfordnlp/imdb", split="test")
        texts = [ex["text"][:500] for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    elif args.dataset == "esnli":
        # Use stanfordnlp/snli: the legacy "esnli" script-based dataset is
        # unsupported by datasets>=3.0
        dataset = load_dataset("stanfordnlp/snli", split="test")
        dataset = dataset.filter(lambda x: x["label"] != -1)
        texts = [f"{ex['premise']} [SEP] {ex['hypothesis']}" for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    elif args.dataset == "agnews":
        dataset = load_dataset("fancyzhx/ag_news", split="test")
        texts = [ex["text"] for ex in dataset]
        labels = [ex["label"] for ex in dataset]
    
    # Limit examples
    texts = texts[:args.max_examples]
    labels = labels[:args.max_examples]
    print(f"Loaded {len(texts)} examples")
    
    # Build retrieval pool if needed
    retrieval_pool = None
    if "retrieval" in args.operators:
        print("\nBuilding retrieval pool...")
        
        # Tokenize all examples for pool
        all_input_ids = []
        all_rationale_masks = []
        
        for text in texts:
            prompt = format_classification_prompt(text, args.dataset)
            encoded = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            input_ids = encoded["input_ids"].squeeze()
            # Use all tokens as potential rationale for pool building
            rationale_mask = torch.ones_like(input_ids)
            all_input_ids.append(input_ids)
            all_rationale_masks.append(rationale_mask)
        
        target_tokens = get_target_tokens(args.dataset, tokenizer)
        extra_bl = set(target_tokens.values())
        retrieval_pool = RetrievalPool(
            tokenizer,
            seed=args.seed,
            use_sentiment_scrub=args.sentiment_scrub,
            extra_blacklist_ids=extra_bl
        )
        retrieval_pool.build_pool(all_input_ids, all_rationale_masks)
    else:
        target_tokens = get_target_tokens(args.dataset, tokenizer)

    # Get extractor
    from ice import get_extractor
    
    all_results = {}
    
    for extractor_name in args.extractors:
        print(f"\n{'='*60}")
        print(f"Evaluating with {extractor_name}")
        print('='*60)
        
        extractor = get_extractor(extractor_name, model, tokenizer, device)
        
        operator_results = {op: [] for op in args.operators}
        
        for example_id, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts))):
            try:
                prompt = format_classification_prompt(text, args.dataset)
                encoded = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                input_ids = encoded["input_ids"].to(device).squeeze()
                attention_mask = encoded["attention_mask"].to(device).squeeze()
                
                # Get model prediction
                with torch.no_grad():
                    outputs = model(input_ids=input_ids.unsqueeze(0), 
                                   attention_mask=attention_mask.unsqueeze(0))
                    logits = outputs.logits[:, -1, :]
                    target_logits = torch.tensor([logits[0, tid].item() for tid in target_tokens.values()])
                    probs = torch.softmax(target_logits, dim=0)
                    predicted_class = probs.argmax().item()
                
                if probs[predicted_class].item() < 0.4:
                    continue  # Skip uncertain predictions
                
                # Get importance scores
                target_token_id = target_tokens[predicted_class]
                importance = extractor.get_importance_scores(
                    input_ids, attention_mask, target_class=target_token_id
                )
                
                # Evaluate with each operator
                for op_name in args.operators:
                    if op_name == "deletion":
                        result = evaluate_with_deletion(
                            input_ids, attention_mask, model, tokenizer,
                            importance, args.k, args.n_permutations,
                            predicted_class, target_tokens, device
                        )
                    elif op_name == "retrieval":
                        result = evaluate_with_retrieval(
                            input_ids, attention_mask, model, tokenizer,
                            importance, args.k, args.n_permutations,
                            predicted_class, target_tokens, device,
                            retrieval_pool, example_id
                        )
                    else:
                        continue
                    
                    if result is not None:
                        operator_results[op_name].append(result)
            
            except Exception:
                continue
        
        # Aggregate results per operator
        extractor_summary = {}
        for op_name, results in operator_results.items():
            if results:
                win_rates = [r["win_rate"] for r in results]
                extractor_summary[op_name] = {
                    "n_examples": len(results),
                    "win_rate": float(np.mean(win_rates)),
                    "win_rate_std": float(np.std(win_rates)),
                }
                print(f"  {op_name}: Win Rate = {np.mean(win_rates)*100:.1f}% ± {np.std(win_rates)*100:.1f}%")
        
        all_results[extractor_name] = extractor_summary
    
    # Compute operator agreement
    if "deletion" in args.operators and "retrieval" in args.operators:
        print("\n" + "="*60)
        print("OPERATOR AGREEMENT ANALYSIS")
        print("="*60)
        
        for extractor_name in args.extractors:
            if extractor_name in all_results:
                del_wr = all_results[extractor_name].get("deletion", {}).get("win_rate", 0)
                ret_wr = all_results[extractor_name].get("retrieval", {}).get("win_rate", 0)
                print(f"{extractor_name}:")
                print(f"  Deletion Win Rate:  {del_wr*100:.1f}%")
                print(f"  Retrieval Win Rate: {ret_wr*100:.1f}%")
                print(f"  Difference: {abs(del_wr - ret_wr)*100:.1f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    output_file = output_dir / f"ice_retrieval_{model_short}_{args.dataset}_{timestamp}.json"
    
    final_results = {
        "model": args.model,
        "dataset": args.dataset,
        "operators": args.operators,
        "sentiment_scrub": args.sentiment_scrub,
        "k": args.k,
        "n_permutations": args.n_permutations,
        "n_examples": args.max_examples,
        "seed": args.seed,
        "results": all_results,
        "timestamp": timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
