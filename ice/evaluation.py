"""
ICE Evaluation Pipeline

Main module that orchestrates ICE faithfulness evaluation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from pathlib import Path

from .operators import BaseOperator, create_ice_lite_operators, create_default_operators
from .metrics import ICEScorer, compute_auc_over_k, aggregate_across_operators
from .stats import ICEStatisticalEvaluator, bootstrap_ci, benjamini_hochberg
from .extractors import BaseRationaleExtractor, get_extractor


@dataclass
class ICEConfig:
    """Configuration for ICE evaluation"""
    # Operator settings
    operators: str = "lite"  # "lite" or "full"
    
    # Statistical settings
    n_permutations: int = 100  # M for randomization test (increased for granularity)
    n_bootstrap: int = 200   # B for bootstrap CI
    alpha: float = 0.10      # Significance level (relaxed to 0.10)
    confidence_level: float = 0.95
    
    # Evaluation settings
    k_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    score_type: str = "prob"  # "prob", "logit", or "margin"
    
    # Runtime settings
    batch_size: int = 1
    device: str = None
    seed: int = 42


@dataclass
class ICEResult:
    """Result container for ICE evaluation"""
    # Per-example results
    example_results: List[Dict]
    
    # Aggregated statistics
    sufficiency_stats: Dict
    comprehensiveness_stats: Dict
    
    # Per-operator results (for robustness analysis)
    operator_results: Dict
    
    # AUC results
    auc_sufficiency: Dict
    auc_comprehensiveness: Dict
    
    # Metadata
    config: ICEConfig
    extractor_name: str
    n_examples: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "sufficiency": self.sufficiency_stats,
            "comprehensiveness": self.comprehensiveness_stats,
            "auc_sufficiency": self.auc_sufficiency,
            "auc_comprehensiveness": self.auc_comprehensiveness,
            "operator_results": self.operator_results,
            "n_examples": self.n_examples,
            "extractor": self.extractor_name,
            "config": {
                "n_permutations": self.config.n_permutations,
                "n_bootstrap": self.config.n_bootstrap,
                "alpha": self.config.alpha,
                "k_values": self.config.k_values
            }
        }
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        # Compute mean win rate and effect size
        win_rates = [r.get("win_rate", 0) for r in self.example_results]
        effect_sizes = [r.get("effect_size", 0) for r in self.example_results]
        mean_win_rate = np.mean(win_rates) if win_rates else 0
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0
        
        lines = [
            f"ICE Evaluation Results ({self.extractor_name})",
            f"{'='*50}",
            f"Examples: {self.n_examples}",
            "",
            "Sufficiency (NSD):",
            f"  Mean: {self.sufficiency_stats['mean']:.3f} ± {self.sufficiency_stats['mean_se']:.3f}",
            f"  95% CI: [{self.sufficiency_stats['mean_ci_lower']:.3f}, {self.sufficiency_stats['mean_ci_upper']:.3f}]",
            "",
            "Comprehensiveness (NSD):",
            f"  Mean: {self.comprehensiveness_stats['mean']:.3f} ± {self.comprehensiveness_stats['mean_se']:.3f}",
            f"  95% CI: [{self.comprehensiveness_stats['mean_ci_lower']:.3f}, {self.comprehensiveness_stats['mean_ci_upper']:.3f}]",
            "",
            f"AUC-Sufficiency: {self.auc_sufficiency.get('mean', 'N/A'):.3f}",
            f"AUC-Comprehensiveness: {self.auc_comprehensiveness.get('mean', 'N/A'):.3f}",
            "",
            "Faithfulness vs Random:",
            f"  Win Rate: {mean_win_rate*100:.1f}% (>50% = better than random)",
            f"  Effect Size: {mean_effect_size:.2f} (Cohen's d)",
        ]
        
        if 'n_significant_fdr' in self.sufficiency_stats:
            lines.extend([
                "",
                "Statistical Significance:",
                f"  Sufficiency significant (FDR): {self.sufficiency_stats['n_significant_fdr']}/{self.n_examples}",
            ])
        
        return "\n".join(lines)


class ICEEvaluator:
    """
    Main ICE evaluation class.
    
    Computes faithfulness scores with statistical rigor.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: ICEConfig = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ICEConfig()
        
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        import random
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Initialize components
        self.scorer = ICEScorer(
            model, tokenizer,
            score_type=self.config.score_type,
            device=self.config.device
        )
        
        self.stat_evaluator = ICEStatisticalEvaluator(
            n_permutations=self.config.n_permutations,
            n_bootstrap=self.config.n_bootstrap,
            alpha=self.config.alpha,
            confidence_level=self.config.confidence_level
        )
        
        # Initialize operators
        if self.config.operators == "lite":
            self.operators = create_ice_lite_operators(tokenizer)
        else:
            families = create_default_operators(tokenizer)
            self.operators = []
            for family in families.values():
                self.operators.extend(family.operators)
        
        np.random.seed(self.config.seed)
    
    def evaluate_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        importance_scores: torch.Tensor,
        target_class: int,
        k: float = 0.2
    ) -> Dict:
        """
        Evaluate a single example.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            importance_scores: Per-token importance from explanation method
            target_class: Target class for scoring
            k: Rationale budget (fraction of tokens)
            
        Returns:
            Dictionary with all scores and statistics
        """
        # Ensure proper dimensions
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.squeeze(0)
        if importance_scores.dim() == 2:
            importance_scores = importance_scores.squeeze(0)
        
        # Build special token set for fair comparison with random rationales
        special_token_ids = set()
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special_token_ids.add(token_id)
        
        # Get valid (non-special) positions - same as randomization test
        valid_positions = []
        for i in range(len(input_ids)):
            if attention_mask[i] == 1:
                if input_ids[i].item() not in special_token_ids:
                    valid_positions.append(i)
        
        # Create rationale mask from importance scores (only considering non-special tokens)
        n_tokens = max(1, int(k * len(valid_positions)))
        
        # Get importance scores only for valid positions
        valid_importance = [(i, importance_scores[i].item()) for i in valid_positions]
        valid_importance.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in valid_importance[:n_tokens]]
        
        rationale_mask = torch.zeros_like(attention_mask)
        for idx in top_k_indices:
            rationale_mask[idx] = 1
        
        # Compute scores for each operator
        suf_results = []
        comp_results = []
        
        for operator in self.operators:
            suf = self.scorer.compute_sufficiency(
                input_ids, attention_mask, rationale_mask, target_class, operator
            )
            comp = self.scorer.compute_comprehensiveness(
                input_ids, attention_mask, rationale_mask, target_class, operator
            )
            suf_results.append(suf)
            comp_results.append(comp)
        
        # Aggregate across operators
        suf_agg = aggregate_across_operators(suf_results)
        comp_agg = aggregate_across_operators(comp_results)
        
        def _compute_auc_statistic(self, input_ids, attention_mask, importance_scores, 
                            target_class, operator):
            """Compute AUC as single test statistic."""
            auc_result = compute_auc_over_k(
                self.scorer, input_ids, attention_mask, importance_scores,
                target_class, operator, self.config.k_values, "sufficiency"
            )
            return auc_result["auc"]
            
        # Randomization test for sufficiency
        def suf_score_fn(mask):
            results = []
            for op in self.operators:
                r = self.scorer.compute_sufficiency(
                    input_ids, attention_mask, mask, target_class, op
                )
                results.append(r["sufficiency_nsd"])
            return np.median(results)
        
        suf_test = self._run_randomization_test(
            observed_score=suf_agg.get("sufficiency_nsd_robust", 0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            rationale_length=n_tokens,
            score_function=suf_score_fn
        )
        
        return {
            "sufficiency_nsd": suf_agg.get("sufficiency_nsd_robust", 0),
            "sufficiency_nsd_std": suf_agg.get("sufficiency_nsd_std", 0),
            "comprehensiveness_nsd": comp_agg.get("comprehensiveness_nsd_robust", 0),
            "comprehensiveness_nsd_std": comp_agg.get("comprehensiveness_nsd_std", 0),
            "randomization_p_value": suf_test["p_value"],
            "randomization_significant": suf_test["significant"],
            "effect_size": suf_test["effect_size"],
            "win_rate": suf_test["win_rate"],  # NEW: fraction of random beaten
            "percentile": suf_test["percentile"],  # NEW: percentile rank
            "null_mean": suf_test["null_mean"],  # NEW: for reference
            "null_std": suf_test["null_std"],  # NEW: for reference
            "rationale_length": n_tokens,
            "rationale_fraction": k,
            "per_operator_suf": suf_results,
            "per_operator_comp": comp_results
        }
    
    def _run_randomization_test(
        self,
        observed_score: float,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_length: int,
        score_function
    ) -> Dict:
        """Run randomization test excluding special tokens"""
        # Build special token set
        special_token_ids = set()
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special_token_ids.add(token_id)
        
        # Build candidate positions (exclude special tokens)
        valid_positions = []
        for i in range(len(input_ids)):
            if attention_mask[i] == 1:
                if input_ids[i].item() not in special_token_ids:
                    valid_positions.append(i)
        
        valid_positions = np.array(valid_positions)
        null_scores = []
        
        for _ in range(self.config.n_permutations):
            if len(valid_positions) >= rationale_length:
                random_indices = np.random.choice(
                    valid_positions,
                    size=rationale_length,
                    replace=False
                )
            else:
                random_indices = valid_positions
            
            random_mask = torch.zeros_like(attention_mask)
            random_mask[random_indices] = 1
            null_scores.append(score_function(random_mask))
        
        null_scores = np.array(null_scores)
        
        # Win rate: fraction of random samples that observed beats
        n_wins = np.sum(observed_score > null_scores)
        win_rate = n_wins / len(null_scores)
        
        # Percentile: where observed falls in null distribution
        percentile = win_rate * 100
        
        # P-value: fraction of random >= observed (one-tailed)
        n_greater_equal = np.sum(null_scores >= observed_score)
        p_value = (1 + n_greater_equal) / (self.config.n_permutations + 1)
        
        # Effect size (Cohen's d)
        null_mean = np.mean(null_scores)
        null_std = np.std(null_scores)
        effect_size = (observed_score - null_mean) / null_std if null_std > 0 else 0
        
        return {
            "p_value": p_value,
            "significant": p_value < self.config.alpha,
            "effect_size": effect_size,
            "win_rate": win_rate,
            "percentile": percentile,
            "null_mean": null_mean,
            "null_std": null_std
        }
    
    def evaluate_dataset(
        self,
        dataset,
        extractor: BaseRationaleExtractor,
        k: float = 0.2,
        max_examples: int = None,
        show_progress: bool = True
    ) -> ICEResult:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: Iterable of (input_ids, attention_mask, label) tuples or dicts
            extractor: Rationale extraction method
            k: Rationale budget
            max_examples: Limit evaluation to this many examples
            show_progress: Show progress bar
            
        Returns:
            ICEResult with all statistics
        """
        example_results = []
        auc_suf_scores = []
        auc_comp_scores = []
        
        iterator = dataset
        if max_examples:
            iterator = list(dataset)[:max_examples]
        if show_progress:
            iterator = tqdm(iterator, desc="ICE Evaluation")
        
        for item in iterator:
            # Handle both dict and tuple formats
            if isinstance(item, dict):
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]
                label = item.get("label", item.get("labels"))
            else:
                input_ids, attention_mask, label = item
            
            # Convert to tensors if needed
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            
            # Get model prediction as target class
            with torch.no_grad():
                ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
                mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
                outputs = self.model(
                    input_ids=ids.to(self.config.device),
                    attention_mask=mask.to(self.config.device)
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = probs.max().item()
                target_class = outputs.logits.argmax(dim=-1).item()
            
            # Skip low-confidence examples where NSD is unstable
            if confidence < 0.5:
                continue
            
            # Get importance scores from extractor
            importance = extractor.get_importance_scores(
                input_ids, attention_mask, target_class
            )
            
            # Evaluate single example
            result = self.evaluate_single(
                input_ids, attention_mask, importance, target_class, k
            )
            example_results.append(result)
            
            # Compute AUC over k
            for operator in self.operators[:1]:  # Use first operator for AUC
                auc_suf = compute_auc_over_k(
                    self.scorer, input_ids, attention_mask, importance,
                    target_class, operator, self.config.k_values, "sufficiency"
                )
                auc_comp = compute_auc_over_k(
                    self.scorer, input_ids, attention_mask, importance,
                    target_class, operator, self.config.k_values, "comprehensiveness"
                )
                auc_suf_scores.append(auc_suf["auc"])
                auc_comp_scores.append(auc_comp["auc"])
        
        # Aggregate dataset statistics
        suf_stats = self.stat_evaluator.evaluate_dataset(
            example_results, "sufficiency_nsd"
        )
        comp_stats = self.stat_evaluator.evaluate_dataset(
            example_results, "comprehensiveness_nsd"
        )
        
        # FDR correction for p-values
        p_values = np.array([r["randomization_p_value"] for r in example_results])
        adjusted_p, rejected = benjamini_hochberg(p_values, self.config.alpha)
        
        suf_stats["n_significant_fdr"] = int(np.sum(rejected))
        suf_stats["fdr_rejection_rate"] = float(np.mean(rejected))
        
        # AUC statistics
        auc_suf_ci = bootstrap_ci(np.array(auc_suf_scores), "mean", self.config.confidence_level, self.config.n_bootstrap)
        auc_comp_ci = bootstrap_ci(np.array(auc_comp_scores), "mean", self.config.confidence_level, self.config.n_bootstrap)
        
        # Operator-level results
        operator_results = {}
        for i, op in enumerate(self.operators):
            op_suf = [r["per_operator_suf"][i]["sufficiency_nsd"] for r in example_results]
            op_comp = [r["per_operator_comp"][i]["comprehensiveness_nsd"] for r in example_results]
            operator_results[op.name] = {
                "sufficiency_mean": np.mean(op_suf),
                "sufficiency_std": np.std(op_suf),
                "comprehensiveness_mean": np.mean(op_comp),
                "comprehensiveness_std": np.std(op_comp)
            }
        
        return ICEResult(
            example_results=example_results,
            sufficiency_stats=suf_stats,
            comprehensiveness_stats=comp_stats,
            operator_results=operator_results,
            auc_sufficiency={
                "mean": auc_suf_ci.point_estimate,
                "ci_lower": auc_suf_ci.ci_lower,
                "ci_upper": auc_suf_ci.ci_upper,
                "std": np.std(auc_suf_scores)
            },
            auc_comprehensiveness={
                "mean": auc_comp_ci.point_estimate,
                "ci_lower": auc_comp_ci.ci_lower,
                "ci_upper": auc_comp_ci.ci_upper,
                "std": np.std(auc_comp_scores)
            },
            config=self.config,
            extractor_name=extractor.__class__.__name__,
            n_examples=len(example_results)
        )
    
    def compare_extractors(
        self,
        dataset,
        extractor_names: List[str] = ["attention", "gradient", "lime"],
        k: float = 0.2,
        max_examples: int = None,
        show_progress: bool = True
    ) -> Dict[str, ICEResult]:
        """
        Compare multiple rationale extraction methods.
        
        Args:
            dataset: Dataset to evaluate
            extractor_names: List of extractor names
            k: Rationale budget
            max_examples: Limit examples
            show_progress: Show progress
            
        Returns:
            Dictionary mapping extractor name to ICEResult
        """
        results = {}
        
        for name in extractor_names:
            print(f"\nEvaluating {name}...")
            extractor = get_extractor(name, self.model, self.tokenizer, self.config.device)
            
            # Cache dataset as list for reuse
            if not isinstance(dataset, list):
                dataset = list(dataset)
            
            result = self.evaluate_dataset(
                dataset, extractor, k, max_examples, show_progress
            )
            results[name] = result
        
        return results


def compare_with_eraser(
    ice_result: ICEResult,
    eraser_scores: Dict[str, float]
) -> Dict:
    """
    Compare ICE results with ERASER baseline scores.
    
    Args:
        ice_result: ICE evaluation result
        eraser_scores: Dictionary with ERASER sufficiency/comprehensiveness
        
    Returns:
        Comparison analysis
    """
    return {
        "ice_sufficiency": ice_result.sufficiency_stats["mean"],
        "ice_sufficiency_ci": (
            ice_result.sufficiency_stats["mean_ci_lower"],
            ice_result.sufficiency_stats["mean_ci_upper"]
        ),
        "eraser_sufficiency": eraser_scores.get("sufficiency"),
        "ice_comprehensiveness": ice_result.comprehensiveness_stats["mean"],
        "ice_comprehensiveness_ci": (
            ice_result.comprehensiveness_stats["mean_ci_lower"],
            ice_result.comprehensiveness_stats["mean_ci_upper"]
        ),
        "eraser_comprehensiveness": eraser_scores.get("comprehensiveness"),
        "significance_rate": ice_result.sufficiency_stats.get("fdr_rejection_rate", 0),
        "operator_variance": {
            op: res["sufficiency_std"]
            for op, res in ice_result.operator_results.items()
        }
    }
