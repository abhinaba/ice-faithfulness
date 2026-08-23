"""
Intervention Operators for ICE Framework

Defines operator families for faithfulness evaluation:
- Deletion: Remove tokens/spans entirely
- Mask: Replace with [MASK], [UNK], or padding tokens
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
from dataclasses import dataclass


@dataclass
class InterventionResult:
    """Result of applying an intervention operator"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    original_length: int
    intervened_length: int
    operator_name: str


class BaseOperator(ABC):
    """Base class for intervention operators"""
    
    def __init__(self, tokenizer, name: str):
        self.tokenizer = tokenizer
        self.name = name
    
    @abstractmethod
    def apply_sufficiency(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """Keep only rationale tokens (sufficiency test)"""
        pass
    
    @abstractmethod
    def apply_comprehensiveness(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """Remove rationale tokens (comprehensiveness test)"""
        pass


class DeletionOperator(BaseOperator):
    """
    Deletion operator: physically remove tokens from sequence.
    
    For sufficiency: keep only rationale tokens
    For comprehensiveness: remove rationale tokens
    """
    
    def __init__(self, tokenizer, keep_special_tokens: bool = True):
        super().__init__(tokenizer, "deletion")
        self.keep_special_tokens = keep_special_tokens
        self._init_special_token_ids()
    
    def _init_special_token_ids(self):
        """Initialize special token IDs to preserve"""
        self.special_ids = set()
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                self.special_ids.add(token_id)
    
    def _get_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for special tokens (1 = special, 0 = regular)"""
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in self.special_ids:
            special_mask |= (input_ids == special_id)
        return special_mask
    
    def apply_sufficiency(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """Keep only rationale tokens (and optionally special tokens)"""
        device = input_ids.device
        original_length = attention_mask.sum().item()
        
        if self.keep_special_tokens:
            special_mask = self._get_special_token_mask(input_ids)
            keep_mask = (rationale_mask.bool() | special_mask) & attention_mask.bool()
        else:
            keep_mask = rationale_mask.bool() & attention_mask.bool()
        
        # Extract kept tokens
        new_input_ids = input_ids[keep_mask]
        new_attention_mask = torch.ones_like(new_input_ids)
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0),
            attention_mask=new_attention_mask.unsqueeze(0),
            original_length=original_length,
            intervened_length=new_input_ids.shape[0],
            operator_name=self.name
        )
    
    def apply_comprehensiveness(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """Remove rationale tokens (keep non-rationale and special tokens)"""
        device = input_ids.device
        original_length = attention_mask.sum().item()
        
        if self.keep_special_tokens:
            special_mask = self._get_special_token_mask(input_ids)
            keep_mask = (~rationale_mask.bool() | special_mask) & attention_mask.bool()
        else:
            keep_mask = ~rationale_mask.bool() & attention_mask.bool()
        
        new_input_ids = input_ids[keep_mask]
        new_attention_mask = torch.ones_like(new_input_ids)
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0),
            attention_mask=new_attention_mask.unsqueeze(0),
            original_length=original_length,
            intervened_length=new_input_ids.shape[0],
            operator_name=self.name
        )
        

    def get_baseline_input(self, tokenizer):
        """Return minimal valid input for this operator family."""
        encoded = tokenizer("", return_tensors="pt")
        return encoded['input_ids'], encoded['attention_mask']

    


class MaskOperator(BaseOperator):
    """
    Mask operator: replace tokens with mask/unk token.
    
    Preserves sequence length but replaces content.
    """
    
    def __init__(self, tokenizer, mask_token: str = "mask"):
        super().__init__(tokenizer, f"mask_{mask_token}")
        self.mask_token = mask_token
        self._init_replacement_id()
    
    def _init_replacement_id(self):
        """Determine which token ID to use for masking"""
        if self.mask_token == "mask" and hasattr(self.tokenizer, 'mask_token_id'):
            self.replacement_id = self.tokenizer.mask_token_id
        elif self.mask_token == "unk" and hasattr(self.tokenizer, 'unk_token_id'):
            self.replacement_id = self.tokenizer.unk_token_id
        elif self.mask_token == "pad" and hasattr(self.tokenizer, 'pad_token_id'):
            self.replacement_id = self.tokenizer.pad_token_id
        else:
            # Fallback to UNK or a zero token
            self.replacement_id = getattr(self.tokenizer, 'unk_token_id', 0)
    
    def _get_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for special tokens"""
        special_ids = set()
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(token_id)
        
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in special_ids:
            special_mask |= (input_ids == special_id)
        return special_mask
    
    def apply_sufficiency(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_mask: torch.Tensor
    ) -> InterventionResult:
        """Mask non-rationale tokens"""
        original_length = attention_mask.sum().item()
        special_mask = self._get_special_token_mask(input_ids)
        
        # Tokens to mask: not rationale, not special, and within attention
        mask_positions = ~rationale_mask.bool() & ~special_mask & attention_mask.bool()
        
        new_input_ids = input_ids.clone()
        new_input_ids[mask_positions] = self.replacement_id
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
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
        """Mask rationale tokens"""
        original_length = attention_mask.sum().item()
        special_mask = self._get_special_token_mask(input_ids)
        
        # Tokens to mask: rationale, not special, and within attention
        mask_positions = rationale_mask.bool() & ~special_mask & attention_mask.bool()
        
        new_input_ids = input_ids.clone()
        new_input_ids[mask_positions] = self.replacement_id
        
        return InterventionResult(
            input_ids=new_input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            original_length=original_length,
            intervened_length=original_length,
            operator_name=self.name
        )
    
    # In MaskOperator:
    def get_baseline_input(self, tokenizer):
        """Return fully-masked input for this operator family."""
        # Create a short sequence of just mask tokens
        mask_seq = " ".join([tokenizer.mask_token] * 5)
        encoded = tokenizer(mask_seq, return_tensors="pt")
        return encoded['input_ids'], encoded['attention_mask']


class OperatorFamily:
    """Collection of operators in a family for robust evaluation"""
    
    def __init__(self, name: str, operators: List[BaseOperator]):
        self.name = name
        self.operators = operators
    
    def __iter__(self):
        return iter(self.operators)
    
    def __len__(self):
        return len(self.operators)


def create_default_operators(tokenizer) -> dict:
    """
    Create default operator families for ICE evaluation.
    
    Returns:
        dict mapping family name to OperatorFamily
    """
    families = {
        "deletion": OperatorFamily(
            "deletion",
            [DeletionOperator(tokenizer, keep_special_tokens=True)]
        ),
        "mask": OperatorFamily(
            "mask",
            [
                MaskOperator(tokenizer, mask_token="mask"),
                MaskOperator(tokenizer, mask_token="unk"),
            ]
        )
    }
    return families


def create_ice_lite_operators(tokenizer) -> List[BaseOperator]:
    """
    Create ICE-lite operator set (2 operators for efficiency).
    
    Returns:
        List of operators: [deletion, mask]
    """
    return [
        DeletionOperator(tokenizer, keep_special_tokens=True),
        MaskOperator(tokenizer, mask_token="mask")
    ]
