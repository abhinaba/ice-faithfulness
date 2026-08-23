"""
Retrieval Infill Operator for Original ICE Framework
=====================================================

This operator integrates the reviewer-proof Retrieval Infill v2 
with the original ICE framework's BaseOperator interface.

For ACL paper: Can be used alongside deletion and mask operators.
"""

import torch
import numpy as np
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass

# Import from original ICE framework
try:
    from ice.operators import BaseOperator, InterventionResult
except ImportError:
    # Fallback: define minimal interface
    @dataclass
    class InterventionResult:
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        original_length: int
        intervened_length: int
        operator_name: str
    
    class BaseOperator:
        def __init__(self, tokenizer, name: str):
            self.tokenizer = tokenizer
            self.name = name


class RetrievalPool:
    """Token-level retrieval pool for ICE framework integration."""
    
    LABEL_BLACKLIST = {
        "positive", "negative", "Positive", "Negative",
        "entailment", "neutral", "contradiction",
        "true", "false", "yes", "no"
    }
    
    SENTIMENT_SCRUB = {
        "amazing", "terrible", "great", "awful", "excellent", "horrible",
        "wonderful", "bad", "good", "best", "worst", "love", "hate"
    }
    
    def __init__(self, tokenizer, seed: int = 42, use_sentiment_scrub: bool = False):
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        self.use_sentiment_scrub = use_sentiment_scrub
        
        # Pool: list of (token_id, example_id)
        self.token_pool: List[Tuple[int, int]] = []
        self.blacklist_ids: Set[int] = set()
        self._built = False
        
        self._build_blacklist_ids()
    
    def _build_blacklist_ids(self):
        """Convert blacklisted words to token IDs."""
        blacklist_words = self.LABEL_BLACKLIST
        if self.use_sentiment_scrub:
            blacklist_words = blacklist_words | self.SENTIMENT_SCRUB
        
        for word in blacklist_words:
            # Try different tokenization variants
            for variant in [word, f" {word}", word.lower(), f" {word.lower()}"]:
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                self.blacklist_ids.update(ids)
    
    def build_pool(self, input_ids_list: List[torch.Tensor], 
                   rationale_masks: List[torch.Tensor]):
        """
        Build pool from rationale tokens across examples.
        
        Args:
            input_ids_list: List of input_ids tensors (one per example)
            rationale_masks: List of rationale masks (1=rationale, 0=context)
        """
        self.token_pool = []
        
        for ex_id, (input_ids, rationale_mask) in enumerate(zip(input_ids_list, rationale_masks)):
            # Flatten if needed
            if len(input_ids.shape) > 1:
                input_ids = input_ids.squeeze()
            if len(rationale_mask.shape) > 1:
                rationale_mask = rationale_mask.squeeze()
            
            # Extract rationale tokens
            rationale_positions = (rationale_mask == 1).nonzero(as_tuple=True)[0]
            
            for pos in rationale_positions:
                token_id = input_ids[pos].item()
                # Skip blacklisted and special tokens
                if token_id not in self.blacklist_ids:
                    self.token_pool.append((token_id, ex_id))
        
        self._built = True
        print(f"✓ Retrieval pool built: {len(self.token_pool)} tokens from {len(input_ids_list)} examples")
    
    def sample_tokens(self, n: int, exclude_example_id: int = -1) -> List[int]:
        """Sample n tokens, optionally excluding a specific example (leave-one-out)."""
        if not self._built:
            raise RuntimeError("Pool not built")
        
        valid_tokens = [t for t, eid in self.token_pool if eid != exclude_example_id]
        
        if len(valid_tokens) < n:
            # Sample with replacement if not enough
            return list(self.rng.choice(valid_tokens, size=n, replace=True))
        
        return list(self.rng.choice(valid_tokens, size=n, replace=False))


class RetrievalInfillOperator(BaseOperator):
    """
    Retrieval Infill operator for ICE framework.
    
    Replaces tokens with retrieved tokens from OTHER examples,
    providing distribution-matched interventions without OOD artifacts.
    
    For ACL paper: Add this alongside deletion and mask operators.
    """
    
    def __init__(self, tokenizer, pool: RetrievalPool = None, 
                 use_sentiment_scrub: bool = False, seed: int = 42):
        super().__init__(tokenizer, "retrieval_infill")
        self.pool = pool
        self.use_sentiment_scrub = use_sentiment_scrub
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._current_example_id = -1
    
    def set_pool(self, pool: RetrievalPool):
        self.pool = pool
    
    def set_current_example(self, example_id: int):
        """Set current example ID for leave-one-out sampling."""
        self._current_example_id = example_id
    
    def apply_sufficiency(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """
        Replace non-rationale tokens with retrieved tokens.
        
        This tests if the rationale is SUFFICIENT by surrounding it
        with in-distribution (but unrelated) tokens.
        """
        if self.pool is None or not self.pool._built:
            raise RuntimeError("Pool must be built before use")
        
        device = input_ids.device
        original_length = attention_mask.sum().item()
        
        # Flatten if batched
        if len(input_ids.shape) > 1:
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            rationale_mask = rationale_mask.squeeze()
        
        new_input_ids = input_ids.clone()
        
        # Find positions to replace (non-rationale, within attention)
        replace_mask = ~rationale_mask.bool() & attention_mask.bool()
        replace_positions = replace_mask.nonzero(as_tuple=True)[0]
        
        if len(replace_positions) > 0:
            # Get replacement tokens from pool (leave-one-out)
            replacement_tokens = self.pool.sample_tokens(
                len(replace_positions), 
                exclude_example_id=self._current_example_id
            )
            
            # Apply replacements
            for pos, new_token in zip(replace_positions, replacement_tokens):
                new_input_ids[pos] = new_token
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
            original_length=original_length,
            intervened_length=original_length,
            operator_name=self.name
        )
    
    def apply_comprehensiveness(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """
        Replace rationale tokens with retrieved tokens.
        
        This tests if the rationale is NECESSARY by replacing it
        with in-distribution alternatives.
        """
        if self.pool is None or not self.pool._built:
            raise RuntimeError("Pool must be built before use")
        
        device = input_ids.device
        original_length = attention_mask.sum().item()
        
        # Flatten if batched
        if len(input_ids.shape) > 1:
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            rationale_mask = rationale_mask.squeeze()
        
        new_input_ids = input_ids.clone()
        
        # Find positions to replace (rationale tokens, within attention)
        replace_mask = rationale_mask.bool() & attention_mask.bool()
        replace_positions = replace_mask.nonzero(as_tuple=True)[0]
        
        if len(replace_positions) > 0:
            # Get replacement tokens from pool (leave-one-out)
            replacement_tokens = self.pool.sample_tokens(
                len(replace_positions), 
                exclude_example_id=self._current_example_id
            )
            
            # Apply replacements
            for pos, new_token in zip(replace_positions, replacement_tokens):
                new_input_ids[pos] = new_token
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
            original_length=original_length,
            intervened_length=original_length,
            operator_name=self.name
        )
    
    def get_baseline_input(self, tokenizer):
        """Return a baseline input for this operator."""
        # Use pool tokens for baseline
        if self.pool and self.pool._built:
            tokens = self.pool.sample_tokens(5, exclude_example_id=-1)
            text = tokenizer.decode(tokens)
            encoded = tokenizer(text, return_tensors="pt")
            return encoded['input_ids'], encoded['attention_mask']
        else:
            encoded = tokenizer("", return_tensors="pt")
            return encoded['input_ids'], encoded['attention_mask']


def create_retrieval_operators(tokenizer, use_sentiment_scrub: bool = False) -> List[BaseOperator]:
    """
    Create retrieval infill operator for ICE evaluation.
    
    Note: Pool must be built separately after loading data.
    """
    return [
        RetrievalInfillOperator(tokenizer, use_sentiment_scrub=use_sentiment_scrub)
    ]


def add_retrieval_to_ice(tokenizer, existing_operators: list, 
                         use_sentiment_scrub: bool = False) -> list:
    """
    Add retrieval infill to existing ICE operators.
    
    Usage:
        operators = create_ice_lite_operators(tokenizer)
        operators = add_retrieval_to_ice(tokenizer, operators)
    """
    retrieval_op = RetrievalInfillOperator(tokenizer, use_sentiment_scrub=use_sentiment_scrub)
    return existing_operators + [retrieval_op]


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create operator
    op = RetrievalInfillOperator(tokenizer)
    
    # Simulate data
    texts = ["This movie was great!", "Terrible film, waste of time."]
    encodings = [tokenizer(t, return_tensors="pt") for t in texts]
    
    input_ids_list = [e['input_ids'].squeeze() for e in encodings]
    # Simulate rationale masks (e.g., from attention/gradient attribution)
    rationale_masks = [torch.ones_like(ids) for ids in input_ids_list]
    
    # Build pool
    pool = RetrievalPool(tokenizer)
    pool.build_pool(input_ids_list, rationale_masks)
    
    # Set pool on operator
    op.set_pool(pool)
    op.set_current_example(0)
    
    # Apply intervention
    result = op.apply_comprehensiveness(
        input_ids_list[0].unsqueeze(0),
        torch.ones_like(input_ids_list[0]).unsqueeze(0),
        (torch.rand_like(input_ids_list[0].float()) > 0.5).long().unsqueeze(0)
    )
    
    print(f"Original: {tokenizer.decode(input_ids_list[0])}")
    print(f"Intervened: {tokenizer.decode(result.input_ids.squeeze())}")
