"""
ICE Metrics Module

Implements:
- Normalized Score Drop (NSD) 
- Sufficiency and Comprehensiveness scores
- AUC computation over rationale budgets
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from .operators import BaseOperator, InterventionResult


@dataclass
class ScoreResult:
    """Container for model score outputs"""
    logits: torch.Tensor
    predicted_class: int
    predicted_prob: float
    margin: float  # Difference between top-1 and top-2 logits
    
    @classmethod
    def from_logits(cls, logits: torch.Tensor, target_class: Optional[int] = None):
        """Create ScoreResult from model logits"""
        probs = F.softmax(logits, dim=-1)
        
        if target_class is None:
            predicted_class = logits.argmax(dim=-1).item()
        else:
            predicted_class = target_class
            
        predicted_prob = probs[0, predicted_class].item()
        
        # Compute margin (top-1 minus top-2)
        sorted_logits, _ = logits[0].sort(descending=True)
        margin = (sorted_logits[0] - sorted_logits[1]).item()
        
        return cls(
            logits=logits,
            predicted_class=predicted_class,
            predicted_prob=predicted_prob,
            margin=margin
        )


class ICEScorer:
    """
    Computes ICE faithfulness scores.
    
    Uses Normalized Score Drop (NSD) for stable, bounded metrics.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        score_type: str = "prob",  # "prob", "logit", or "margin"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.score_type = score_type
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Cache for baseline (empty) score
        self._baseline_cache = {}
    
    def get_model_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int
    ) -> float:
        """
        Get model score for given input.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            target_class: Class to get score for
            
        Returns:
            Score (probability, logit, or margin depending on score_type)
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        if self.score_type == "prob":
            probs = F.softmax(logits, dim=-1)
            return probs[0, target_class].item()
        elif self.score_type == "logit":
            return logits[0, target_class].item()
        elif self.score_type == "margin":
            # Margin between target class and highest other class
            other_max = logits[0].clone()
            other_max[target_class] = float('-inf')
            return (logits[0, target_class] - other_max.max()).item()
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")
            
            


    def get_baseline_score(self, target_class: int, operator: BaseOperator = None) -> float:
        """Get operator-aware baseline score."""
        op_name = operator.name if operator else "default"
        cache_key = (target_class, self.score_type, op_name)
        
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]
        
        if operator and hasattr(operator, 'get_baseline_input'):
            # Operator-specific baseline
            baseline_ids, baseline_mask = operator.get_baseline_input(self.tokenizer)
        else:
            # Default: empty string
            encoded = self.tokenizer("", return_tensors="pt", padding=True)
            baseline_ids = encoded['input_ids']
            baseline_mask = encoded['attention_mask']
        
        score = self.get_model_score(baseline_ids, baseline_mask, target_class)
        self._baseline_cache[cache_key] = score
        return score
    
    
    
    def compute_nsd(
        self,
        original_score: float,
        intervened_score: float,
        baseline_score: float,
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute Normalized Score Drop.
        
        NSD(r) = (s(r) - s(∅)) / (s(x) - s(∅))
        
        Bounded approximately in [0, 1] for well-behaved interventions.
        Values > 1 indicate intervention preserved more signal than original.
        Values < 0 indicate intervention is worse than baseline.
        
        Args:
            original_score: s(x) - score on original input
            intervened_score: s(r) - score on intervened input
            baseline_score: s(∅) - score on empty/baseline input
            epsilon: Small value to prevent division by zero
            
        Returns:
            NSD score (clipped to [-2, 2] for stability)
        """
        denominator = original_score - baseline_score
        
        # Stability check: if denominator is too small, NSD is undefined
        # This happens when model confidence ≈ baseline (random guessing)
        if abs(denominator) < 0.05:
            return 0.0
        
        nsd = (intervened_score - baseline_score) / denominator
        
        # Clip extreme values to prevent outliers from dominating statistics
        return float(np.clip(nsd, -2.0, 2.0))
    
    def compute_sufficiency(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor,
        target_class: int,
        operator: BaseOperator
    ) -> Dict[str, float]:
        """
        Compute sufficiency score: how much signal does rationale preserve?
        
        High sufficiency = rationale alone is sufficient for prediction
        """
        # Original score
        original_score = self.get_model_score(
            input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask,
            target_class
        )
        
        # Intervened score (keep only rationale)
        intervention = operator.apply_sufficiency(
            input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids,
            attention_mask.squeeze(0) if attention_mask.dim() == 2 else attention_mask,
            rationale_mask.squeeze(0) if rationale_mask.dim() == 2 else rationale_mask
        )
        
        intervened_score = self.get_model_score(
            intervention.input_ids,
            intervention.attention_mask,
            target_class
        )
        
        # Baseline score (operator-aware)
        baseline_score = self.get_baseline_score(target_class, operator)
        
        # Raw delta (ERASER-style)
        delta = intervened_score - original_score
        
        # NSD (our improvement)
        nsd = self.compute_nsd(original_score, intervened_score, baseline_score)
        
        return {
            "sufficiency_delta": delta,
            "sufficiency_nsd": nsd,
            "original_score": original_score,
            "intervened_score": intervened_score,
            "baseline_score": baseline_score,
            "operator": operator.name
        }
    
    def compute_comprehensiveness(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor,
        target_class: int,
        operator: BaseOperator
    ) -> Dict[str, float]:
        """
        Compute comprehensiveness score: how much signal does rationale contain?
        
        High comprehensiveness = removing rationale destroys prediction
        """
        # Original score
        original_score = self.get_model_score(
            input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask,
            target_class
        )
        
        # Intervened score (remove rationale)
        intervention = operator.apply_comprehensiveness(
            input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids,
            attention_mask.squeeze(0) if attention_mask.dim() == 2 else attention_mask,
            rationale_mask.squeeze(0) if rationale_mask.dim() == 2 else rationale_mask
        )
        
        intervened_score = self.get_model_score(
            intervention.input_ids,
            intervention.attention_mask,
            target_class
        )
        
        # Baseline score (operator-aware)
        baseline_score = self.get_baseline_score(target_class, operator)
        
        # Raw delta (ERASER-style): how much score dropped
        delta = original_score - intervened_score
        
        # NSD for comprehensiveness: 1 - NSD(complement)
        # High value = removing rationale causes big drop
        nsd_complement = self.compute_nsd(original_score, intervened_score, baseline_score)
        comp_nsd = 1.0 - nsd_complement
        
        return {
            "comprehensiveness_delta": delta,
            "comprehensiveness_nsd": comp_nsd,
            "original_score": original_score,
            "intervened_score": intervened_score,
            "baseline_score": baseline_score,
            "operator": operator.name
        }


def compute_auc_over_k(
    scorer: ICEScorer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    importance_scores: torch.Tensor,
    target_class: int,
    operator: BaseOperator,
    k_values: List[float] = None,
    metric: str = "sufficiency"
) -> Dict[str, float]:
    """
    Compute AUC over different rationale budgets k.
    
    Instead of reporting one top-k, compute curve over k% tokens.
    
    Args:
        scorer: ICEScorer instance
        input_ids: Token IDs
        attention_mask: Attention mask  
        importance_scores: Per-token importance scores from explanation method
        target_class: Target class for scoring
        operator: Intervention operator
        k_values: List of k percentages (0 to 1), default [0.1, 0.2, ..., 1.0]
        metric: "sufficiency" or "comprehensiveness"
        
    Returns:
        Dict with AUC and per-k scores
    """
    if k_values is None:
        k_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Ensure tensors are 1D
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.squeeze(0)
    if importance_scores.dim() == 2:
        importance_scores = importance_scores.squeeze(0)
    
    # Get valid token count (excluding padding)
    valid_length = attention_mask.sum().item()
    
    # Sort tokens by importance
    sorted_indices = importance_scores.argsort(descending=True)
    
    scores_by_k = {}
    nsd_values = []
    
    for k in k_values:
        # Number of tokens in rationale
        n_tokens = max(1, int(k * valid_length))
        
        # Create rationale mask
        rationale_mask = torch.zeros_like(attention_mask)
        top_k_indices = sorted_indices[:n_tokens]
        rationale_mask[top_k_indices] = 1
        
        # Compute metric
        if metric == "sufficiency":
            result = scorer.compute_sufficiency(
                input_ids, attention_mask, rationale_mask, target_class, operator
            )
            nsd = result["sufficiency_nsd"]
        else:
            result = scorer.compute_comprehensiveness(
                input_ids, attention_mask, rationale_mask, target_class, operator
            )
            nsd = result["comprehensiveness_nsd"]
        
        scores_by_k[k] = nsd
        nsd_values.append(nsd)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(nsd_values, k_values)
    
    return {
        "auc": auc,
        "scores_by_k": scores_by_k,
        "k_values": k_values,
        "metric": metric,
        "operator": operator.name
    }


def aggregate_across_operators(
    results_by_operator: List[Dict],
    aggregation: str = "median"
) -> Dict[str, float]:
    """
    Aggregate scores across multiple operators for robustness.
    
    Args:
        results_by_operator: List of result dicts from different operators
        aggregation: "median" or "mean"
        
    Returns:
        Aggregated scores
    """
    if not results_by_operator:
        return {}
    
    # Collect all numeric values
    all_keys = set()
    for r in results_by_operator:
        all_keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    
    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results_by_operator if key in r]
        if values:
            if aggregation == "median":
                aggregated[f"{key}_robust"] = np.median(values)
            else:
                aggregated[f"{key}_robust"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
    
    return aggregated
