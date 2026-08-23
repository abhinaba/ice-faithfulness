"""
Statistical Tests for ICE Framework

Implements:
- Randomization test (is rationale better than random?)
- Bootstrap confidence intervals
- Benjamini-Hochberg FDR correction
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from tqdm import tqdm


@dataclass
class RandomizationTestResult:
    """Result of randomization test"""
    observed_score: float
    null_scores: np.ndarray
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool
    alpha: float
    n_permutations: int


@dataclass 
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation"""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    bootstrap_distribution: np.ndarray
    std_error: float


def randomization_test(
    observed_score: float,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rationale_length: int,
    score_function: Callable,
    n_permutations: int = 20,
    alpha: float = 0.05,
    seed: int = None,
    special_token_ids: set = None  
) -> RandomizationTestResult:
    """
    Test if rationale is significantly better than random rationales of same length.
    
    Null hypothesis: The observed rationale is no better than random token selections
    of the same length.
    
    Args:
        observed_score: Score (NSD) for the actual rationale
        input_ids: Original token IDs
        attention_mask: Attention mask
        rationale_length: Number of tokens in rationale
        score_function: Function(rationale_mask) -> score
        n_permutations: Number of random rationales to sample (M)
        alpha: Significance level
        seed: Random seed for reproducibility
        
    Returns:
        RandomizationTestResult with p-value and effect size
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get valid positions (non-special, non-padding tokens)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.squeeze(0)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    
    
    # Build candidate positions (exclude special tokens)
    valid_positions = []
    for i in range(len(input_ids)):
        if attention_mask[i] == 1:
            if special_token_ids and input_ids[i].item() in special_token_ids:
                continue  # Skip special tokens
            valid_positions.append(i)
    
    valid_positions = np.array(valid_positions)
    
    # Generate null distribution
    null_scores = []
    
    
    null_scores = []
    for _ in range(n_permutations):
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
    
    # Compute p-value (proportion of null scores >= observed)
    # Using (1 + count) / (M + 1) for proper permutation test p-value
    n_greater_equal = np.sum(null_scores >= observed_score)
    p_value = (1 + n_greater_equal) / (n_permutations + 1)
    
    # Compute effect size (Cohen's d)
    null_mean = np.mean(null_scores)
    null_std = np.std(null_scores)
    if null_std > 0:
        effect_size = (observed_score - null_mean) / null_std
    else:
        effect_size = float('inf') if observed_score > null_mean else 0.0
    
    return RandomizationTestResult(
        observed_score=observed_score,
        null_scores=null_scores,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=(p_value < alpha),
        alpha=alpha,
        n_permutations=n_permutations
    )


def bootstrap_ci(
    scores: np.ndarray,
    statistic: str = "mean",
    confidence_level: float = 0.95,
    n_bootstrap: int = 200,
    seed: int = None
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        scores: Array of scores to bootstrap
        statistic: "mean" or "median"
        confidence_level: CI level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (B)
        seed: Random seed
        
    Returns:
        BootstrapCIResult with CI bounds
    """
    if seed is not None:
        np.random.seed(seed)
    
    scores = np.array(scores)
    n = len(scores)
    
    # Compute point estimate
    if statistic == "mean":
        point_estimate = np.mean(scores)
        stat_func = np.mean
    elif statistic == "median":
        point_estimate = np.median(scores)
        stat_func = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Generate bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(scores, size=n, replace=True)
        bootstrap_stats.append(stat_func(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return BootstrapCIResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        bootstrap_distribution=bootstrap_stats,
        std_error=np.std(bootstrap_stats)
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: Array of p-values
        alpha: Target FDR level
        
    Returns:
        Tuple of (adjusted_p_values, rejected_mask)
    """
    n = len(p_values)
    
    # Sort p-values and get ranks
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Compute BH critical values
    ranks = np.arange(1, n + 1)
    critical_values = (ranks / n) * alpha
    
    # Find largest k where p_(k) <= (k/n) * alpha
    significant = sorted_p <= critical_values
    
    if np.any(significant):
        # All p-values up to the largest significant one are rejected
        max_significant_idx = np.max(np.where(significant)[0])
        rejected_sorted = np.zeros(n, dtype=bool)
        rejected_sorted[:max_significant_idx + 1] = True
    else:
        rejected_sorted = np.zeros(n, dtype=bool)
    
    # Compute adjusted p-values
    adjusted_sorted = np.minimum.accumulate(
        (sorted_p * n / ranks)[::-1]
    )[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    
    # Restore original order
    rejected = np.empty(n, dtype=bool)
    adjusted = np.empty(n)
    rejected[sorted_indices] = rejected_sorted
    adjusted[sorted_indices] = adjusted_sorted
    
    return adjusted, rejected


class ICEStatisticalEvaluator:
    """
    Combines all statistical tests for ICE evaluation.
    """
    
    def __init__(
        self,
        n_permutations: int = 20,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        confidence_level: float = 0.95
    ):
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.confidence_level = confidence_level
    
    def evaluate_single_example(
        self,
        observed_suf_score: float,
        observed_comp_score: float,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_length: int,
        suf_score_function: Callable,
        comp_score_function: Callable,
        seed: int = None
    ) -> Dict:
        """
        Run statistical tests for a single example.
        """
        # Sufficiency randomization test
        suf_test = randomization_test(
            observed_suf_score,
            input_ids,
            attention_mask,
            rationale_length,
            suf_score_function,
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            seed=seed
        )
        
        # Comprehensiveness randomization test
        comp_test = randomization_test(
            observed_comp_score,
            input_ids,
            attention_mask,
            rationale_length,
            comp_score_function,
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            seed=seed + 1 if seed else None
        )
        
        return {
            "sufficiency": {
                "observed": suf_test.observed_score,
                "p_value": suf_test.p_value,
                "effect_size": suf_test.effect_size,
                "is_significant": suf_test.is_significant,
                "null_mean": np.mean(suf_test.null_scores),
                "null_std": np.std(suf_test.null_scores)
            },
            "comprehensiveness": {
                "observed": comp_test.observed_score,
                "p_value": comp_test.p_value,
                "effect_size": comp_test.effect_size,
                "is_significant": comp_test.is_significant,
                "null_mean": np.mean(comp_test.null_scores),
                "null_std": np.std(comp_test.null_scores)
            }
        }
    
    def evaluate_dataset(
        self,
        example_results: List[Dict],
        metric_key: str = "sufficiency_nsd"
    ) -> Dict:
        """
        Compute dataset-level statistics with bootstrap CIs.
        
        Args:
            example_results: List of per-example result dicts
            metric_key: Key to aggregate
            
        Returns:
            Dataset-level statistics
        """
        # Extract scores
        scores = np.array([r[metric_key] for r in example_results if metric_key in r])
        
        if len(scores) == 0:
            return {"error": "No valid scores found"}
        
        # Bootstrap CI for mean
        mean_ci = bootstrap_ci(
            scores,
            statistic="mean",
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap
        )
        
        # Bootstrap CI for median
        median_ci = bootstrap_ci(
            scores,
            statistic="median",
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap
        )
        
        # Collect p-values for FDR correction
        p_values = []
        for r in example_results:
            if "randomization_p_value" in r:
                p_values.append(r["randomization_p_value"])
        
        fdr_results = {}
        if p_values:
            adjusted_p, rejected = benjamini_hochberg(
                np.array(p_values), 
                alpha=self.alpha
            )
            fdr_results = {
                "n_significant_raw": np.sum(np.array(p_values) < self.alpha),
                "n_significant_fdr": np.sum(rejected),
                "fdr_rejection_rate": np.mean(rejected)
            }
        
        return {
            "n_examples": len(scores),
            "mean": mean_ci.point_estimate,
            "mean_ci_lower": mean_ci.ci_lower,
            "mean_ci_upper": mean_ci.ci_upper,
            "mean_se": mean_ci.std_error,
            "median": median_ci.point_estimate,
            "median_ci_lower": median_ci.ci_lower,
            "median_ci_upper": median_ci.ci_upper,
            "std": np.std(scores),
            "confidence_level": self.confidence_level,
            **fdr_results
        }
    
    def format_result(self, stats: Dict, metric_name: str = "ICE") -> str:
        """Format results for reporting"""
        if "error" in stats:
            return f"{metric_name}: Error - {stats['error']}"
        
        ci_pct = int(stats['confidence_level'] * 100)
        return (
            f"{metric_name}: {stats['mean']:.3f} ± {stats['mean_se']:.3f} "
            f"({ci_pct}% CI: [{stats['mean_ci_lower']:.3f}, {stats['mean_ci_upper']:.3f}])"
        )
