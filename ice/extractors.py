"""
Rationale Extraction Methods

Implements common explanation methods for extracting token-level importance:
- Attention-based
- Gradient × Input
- Integrated Gradients
- LIME
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseRationaleExtractor(ABC):
    """Base class for rationale extraction methods"""
    
    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        
        # If model has device_map (multi-gpu), don't force move it
        if getattr(model, "hf_device_map", None):
            self.device = model.device # This might be the first device
            # Do NOT call model.to(device)
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
    
    @abstractmethod
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract token importance scores.
        
        Returns:
            Tensor of shape [seq_len] with importance score per token
        """
        pass
    
    def get_top_k_rationale(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: float = 0.2,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get binary rationale mask for top-k% important tokens.
        
        Args:
            k: Fraction of tokens to include (0-1)
            
        Returns:
            Binary mask [seq_len] where 1 = rationale token
        """
        scores = self.get_importance_scores(input_ids, attention_mask, target_class)
        
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.squeeze(0)
        
        # Only consider valid tokens
        valid_length = attention_mask.sum().item()
        n_tokens = max(1, int(k * valid_length))
        
        # Get top-k indices
        top_k_indices = scores.argsort(descending=True)[:n_tokens]
        
        # Create mask
        rationale_mask = torch.zeros_like(attention_mask)
        rationale_mask[top_k_indices] = 1
        
        return rationale_mask


class AttentionExtractor(BaseRationaleExtractor):
    """
    Extract rationale based on attention weights.
    
    Aggregates attention across heads and layers.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer,
        layer: int = -1,  # Which layer to use (-1 = last)
        head_aggregation: str = "mean",  # "mean", "max", or specific head index
        device: str = None
    ):
        super().__init__(model, tokenizer, device)
        self.layer = layer
        self.head_aggregation = head_aggregation
    
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Extract attention-based importance"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Check if attentions are available
        if outputs.attentions is None:
            raise RuntimeError(
                "Model did not return attention weights. "
                "Ensure model config has output_attentions=True or use a different extractor."
            )
        
        # Get attention from specified layer
        # attentions shape: (batch, heads, seq, seq)
        attentions = outputs.attentions[self.layer]
        
        # Aggregate across heads
        if self.head_aggregation == "mean":
            attention = attentions.mean(dim=1)  # (batch, seq, seq)
        elif self.head_aggregation == "max":
            attention = attentions.max(dim=1)[0]
        elif isinstance(self.head_aggregation, int):
            attention = attentions[:, self.head_aggregation]
        else:
            attention = attentions.mean(dim=1)
        
        # Get attention to CLS token (or first token) as importance
        # This represents how much each token attends to the classification token
        # Alternatively, use attention FROM CLS to each token
        cls_attention = attention[:, 0, :]  # Attention from CLS to all tokens
        
        # Normalize
        importance = cls_attention / (cls_attention.sum() + 1e-8)
        
        return importance.squeeze(0).cpu()


class GradientExtractor(BaseRationaleExtractor):
    """
    Extract rationale based on gradient × input.
    
    Computes gradient of predicted class w.r.t. input embeddings.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        multiply_by_input: bool = True,
        absolute: bool = True,
        device: str = None
    ):
        super().__init__(model, tokenizer, device)
        self.multiply_by_input = multiply_by_input
        self.absolute = absolute
    
    

    def get_importance_scores(self, input_ids, attention_mask, target_class=None):
        """Extract gradient-based importance using generic approach."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Try generic approach first
        try:
            # Get embeddings
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(input_ids)
            embeds.requires_grad_(True)
            embeds.retain_grad()
            
            # Forward with inputs_embeds if supported
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            logits = outputs.logits
            
        except TypeError:
            # Model doesn't support inputs_embeds, fall back to input gradients
            input_ids.requires_grad = False  # Can't differentiate through embeddings
            raise NotImplementedError(
                f"Model {type(self.model).__name__} doesn't support inputs_embeds. "
                "Use attention-based extraction instead."
            )
        
        # Determine target
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward
        logits[0, target_class].backward()
        grads = embeds.grad
        
        # Compute importance
        if self.multiply_by_input:
            importance = (grads * embeds.detach()).abs().sum(dim=-1)
        else:
            importance = grads.abs().sum(dim=-1)
        
        importance = importance / (importance.sum() + 1e-8)
        return importance.squeeze(0).detach().cpu()
    
    '''
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Extract gradient-based importance"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Get embeddings
        self.model.eval()
        
        # Enable gradient computation for embeddings
        embeddings = self.model.get_input_embeddings()
        embed_out = embeddings(input_ids)
        embed_out.requires_grad_(True)
        embed_out.retain_grad()
        
        # Forward pass with embeddings
        # Need to handle different model architectures
        if hasattr(self.model, 'bert'):
            # BERT-style model
            outputs = self.model.bert(
                inputs_embeds=embed_out,
                attention_mask=attention_mask
            )
            pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
            logits = self.model.classifier(self.model.dropout(pooled))
        elif hasattr(self.model, 'roberta'):
            # RoBERTa-style model
            outputs = self.model.roberta(
                inputs_embeds=embed_out,
                attention_mask=attention_mask
            )
            logits = self.model.classifier(outputs.last_hidden_state)
        elif hasattr(self.model, 'distilbert'):
            # DistilBERT-style
            outputs = self.model.distilbert(
                inputs_embeds=embed_out,
                attention_mask=attention_mask
            )
            hidden = outputs.last_hidden_state[:, 0]
            logits = self.model.classifier(self.model.pre_classifier(hidden))
        else:
            # Generic fallback - may not work for all models
            raise NotImplementedError(
                f"Gradient extraction not implemented for {type(self.model)}"
            )
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward pass
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        # Get gradients
        gradients = embed_out.grad  # (batch, seq, hidden)
        
        # Compute importance
        if self.multiply_by_input:
            importance = gradients * embed_out.detach()
        else:
            importance = gradients
        
        # Reduce over hidden dimension
        if self.absolute:
            importance = importance.abs().sum(dim=-1)
        else:
            importance = importance.sum(dim=-1)
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        return importance.squeeze(0).detach().cpu()
    '''

class IntegratedGradientsExtractor(BaseRationaleExtractor):
    """
    Extract rationale using Integrated Gradients.
    
    More principled than simple gradient × input.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_steps: int = 50,
        device: str = None
    ):
        super().__init__(model, tokenizer, device)
        self.n_steps = n_steps
        self.gradient_extractor = GradientExtractor(
            model, tokenizer, 
            multiply_by_input=False, 
            absolute=False,
            device=device
        )
    
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Extract importance using Integrated Gradients"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()
        input_embed = embeddings(input_ids).detach()
        
        # Baseline: zero embedding or pad token embedding
        if self.tokenizer.pad_token_id is not None:
            baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            baseline_embed = embeddings(baseline_ids).detach()
        else:
            baseline_embed = torch.zeros_like(input_embed)
        
        # Determine target class
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                target_class = outputs.logits.argmax(dim=-1).item()
        
        # Integrate gradients
        integrated_grads = torch.zeros_like(input_embed)
        
        for step in range(self.n_steps):
            alpha = step / self.n_steps
            interpolated = baseline_embed + alpha * (input_embed - baseline_embed)
            interpolated.requires_grad_(True)
            
            # Forward pass (simplified - for full implementation use _forward_with_embeds)
            outputs = self._forward_with_embeds(interpolated, attention_mask)
            logits = outputs
            
            # Backward
            target_logit = logits[0, target_class]
            grads = torch.autograd.grad(target_logit, interpolated, retain_graph=False)[0]
            
            integrated_grads += grads / self.n_steps
        
        # Multiply by input difference
        attributions = (input_embed - baseline_embed) * integrated_grads
        
        # Reduce to per-token importance
        importance = attributions.abs().sum(dim=-1)
        importance = importance / (importance.sum() + 1e-8)
        
        return importance.squeeze(0).detach().cpu()
    
    def _forward_with_embeds(self, embeds, attention_mask):
        """Forward pass with embeddings instead of input_ids"""
        if hasattr(self.model, 'bert'):
            outputs = self.model.bert(inputs_embeds=embeds, attention_mask=attention_mask)
            pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
            return self.model.classifier(self.model.dropout(pooled))
        elif hasattr(self.model, 'roberta'):
            outputs = self.model.roberta(inputs_embeds=embeds, attention_mask=attention_mask)
            return self.model.classifier(outputs.last_hidden_state)
        else:
            raise NotImplementedError(f"Model type {type(self.model)} not supported")


class LIMEExtractor(BaseRationaleExtractor):
    """
    Extract rationale using LIME (Local Interpretable Model-agnostic Explanations).
    
    Fits a local linear model to explain predictions.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 500,
        device: str = None
    ):
        super().__init__(model, tokenizer, device)
        self.n_samples = n_samples
    
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Extract LIME-based importance"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                target_class = outputs.logits.argmax(dim=-1).item()
        
        seq_len = attention_mask.sum().item()
        
        # Generate perturbed samples
        # Each sample is a binary mask indicating which tokens are kept
        perturbations = np.random.binomial(1, 0.5, size=(self.n_samples, seq_len))
        
        # Ensure at least one token is kept in each sample
        perturbations[perturbations.sum(axis=1) == 0, 0] = 1
        
        # Get model predictions for perturbations
        predictions = []
        distances = []
        
        for mask in perturbations:
            # Create perturbed input
            perturbed_ids = input_ids.clone()
            perturbed_mask = attention_mask.clone()
            
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
            
            # Replace masked tokens with pad token
            full_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=self.device)
            full_mask[:seq_len] = ~mask_tensor
            
            if self.tokenizer.pad_token_id is not None:
                perturbed_ids[0, full_mask] = self.tokenizer.pad_token_id
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=perturbed_ids,
                    attention_mask=perturbed_mask
                )
                probs = F.softmax(outputs.logits, dim=-1)
                predictions.append(probs[0, target_class].item())
            
            # Compute distance (cosine similarity to original in perturbation space)
            original = np.ones(seq_len)
            distance = 1 - np.dot(mask, original) / (np.linalg.norm(mask) * np.linalg.norm(original) + 1e-8)
            distances.append(distance)
        
        # Fit weighted linear regression
        predictions = np.array(predictions)
        distances = np.array(distances)
        
        # Kernel weights (exponential kernel)
        kernel_width = 0.25 * seq_len
        weights = np.exp(-distances ** 2 / kernel_width ** 2)
        
        # Weighted least squares
        X = perturbations  # (n_samples, seq_len)
        y = predictions    # (n_samples,)
        
        # Add intercept
        X_with_intercept = np.hstack([np.ones((self.n_samples, 1)), X])
        
        # Solve weighted least squares
        W = np.diag(weights)
        try:
            coefficients = np.linalg.lstsq(
                X_with_intercept.T @ W @ X_with_intercept,
                X_with_intercept.T @ W @ y,
                rcond=None
            )[0]
            
            # Token importance (excluding intercept)
            importance = coefficients[1:]
        except np.linalg.LinAlgError:
            # Fallback to uniform importance
            importance = np.ones(seq_len) / seq_len
        
        # Normalize (absolute values for importance magnitude)
        importance = np.abs(importance)
        importance = importance / (importance.sum() + 1e-8)
        
        # Pad to full sequence length
        full_importance = torch.zeros(input_ids.shape[1])
        full_importance[:seq_len] = torch.tensor(importance, dtype=torch.float32)
        
        return full_importance


def get_extractor(
    name: str,
    model,
    tokenizer,
    device: str = None,
    **kwargs
) -> BaseRationaleExtractor:
    """
    Factory function to get rationale extractor by name.
    
    Args:
        name: "attention", "gradient", "integrated_gradients", "lime", 
              "llm_attention", "llm_gradient", or "prompting"
        model: The model to explain
        tokenizer: Associated tokenizer
        device: Device to run on
        **kwargs: Additional arguments for specific extractors
        
    Returns:
        Configured rationale extractor
    """
    extractors = {
        "attention": AttentionExtractor,
        "gradient": GradientExtractor,
        "integrated_gradients": IntegratedGradientsExtractor,
        "lime": LIMEExtractor,
        # LLM-specific extractors
        "llm_attention": LLMAttentionExtractor,
        "llm_gradient": LLMGradientExtractor,
    }
    
    if name not in extractors:
        raise ValueError(f"Unknown extractor: {name}. Choose from {list(extractors.keys())}")
    
    return extractors[name](model, tokenizer, device=device, **kwargs)


# =============================================================================
# LLM-SPECIFIC EXTRACTORS
# =============================================================================

class LLMAttentionExtractor(BaseRationaleExtractor):
    """
    Attention-based rationale extraction for causal LLMs (Llama, Mistral, etc.)
    
    Works with HuggingFace causal LMs that support output_attentions=True.
    Aggregates attention from the last token (prediction position) to all input tokens.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = None,
        layer_aggregation: str = "mean",  # "mean", "last", "max"
        head_aggregation: str = "mean"    # "mean", "max"
    ):
        super().__init__(model, tokenizer, device)
        self.layer_aggregation = layer_aggregation
        self.head_aggregation = head_aggregation
        
        # Try to enable attention output
        if hasattr(self.model, 'config'):
            self.model.config.output_attentions = True
            # Force eager attention for models using SDPA
            if hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = "eager"
    
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract attention-based importance for causal LLMs.
        
        For causal LMs, we look at attention FROM the last token TO all previous tokens,
        since the last token position is where the prediction is made.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
        except Exception as e:
            if "sdpa" in str(e).lower() or "output_attentions" in str(e).lower():
                raise RuntimeError(
                    f"Model does not support output_attentions with current attention implementation. "
                    f"Try loading the model with: model = AutoModelForCausalLM.from_pretrained('{self.model.name_or_path}', attn_implementation='eager')"
                ) from e
            raise
        
        # Get attention weights: tuple of (batch, heads, seq_len, seq_len) per layer
        attentions = outputs.attentions
        
        if attentions is None:
            raise RuntimeError("Model did not return attention weights. "
                             "Use a model that supports output_attentions=True")
        
        # Stack all layers: (n_layers, batch, heads, seq_len, seq_len)
        # Move to CPU first to handle multi-GPU (layers on different devices)
        attentions_cpu = [a.cpu() for a in attentions]
        stacked = torch.stack(attentions_cpu)
        
        # Get attention FROM last token TO all tokens
        # Shape: (n_layers, batch, heads, seq_len)
        seq_len = stacked.shape[-1]
        last_token_attention = stacked[:, :, :, -1, :]  # Attention from last position
        
        # Aggregate across heads
        if self.head_aggregation == "mean":
            head_agg = last_token_attention.mean(dim=2)  # (n_layers, batch, seq_len)
        else:  # max
            head_agg = last_token_attention.max(dim=2)[0]
        
        # Aggregate across layers
        if self.layer_aggregation == "mean":
            layer_agg = head_agg.mean(dim=0)  # (batch, seq_len)
        elif self.layer_aggregation == "last":
            layer_agg = head_agg[-1]  # Last layer only
        else:  # max
            layer_agg = head_agg.max(dim=0)[0]
        
        # Return importance scores for first example
        importance = layer_agg[0]  # (seq_len,)
        
        # Mask padding (ensure same device - importance is already on CPU from attention stacking)
        importance = importance * attention_mask[0].cpu().float()
        
        return importance.cpu()


class LLMGradientExtractor(BaseRationaleExtractor):
    """
    Gradient-based rationale extraction for causal LLMs.
    
    Computes gradient of the predicted token's logit with respect to input embeddings,
    then aggregates to get per-token importance.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = None,
        aggregation: str = "l2"  # "l2", "mean", "max"
    ):
        super().__init__(model, tokenizer, device)
        self.aggregation = aggregation
    
    def get_importance_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute gradient-based importance for causal LLMs.
        
        For causal LMs, we compute gradient of the most likely next token
        with respect to input embeddings.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get embeddings
        if hasattr(self.model, 'get_input_embeddings'):
            embedding_layer = self.model.get_input_embeddings()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embedding_layer = self.model.model.embed_tokens
        else:
            raise RuntimeError("Cannot find embedding layer in model")
        
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()
        
        # Forward pass with embeddings
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        
        # Get logits at last position
        logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
        
        # Use the most likely token or target class
        if target_class is not None:
            target_logit = logits[:, target_class]
        else:
            target_logit = logits.max(dim=-1)[0]
        
        # Backward pass
        target_logit.sum().backward()
        
        # Get gradients
        grads = embeddings.grad  # (batch, seq_len, hidden_dim)
        
        # Aggregate gradient magnitudes
        if self.aggregation == "l2":
            importance = torch.norm(grads[0], dim=-1)  # L2 norm per token
        elif self.aggregation == "mean":
            importance = grads[0].abs().mean(dim=-1)
        else:  # max
            importance = grads[0].abs().max(dim=-1)[0]
        
        # Mask padding
        importance = importance * attention_mask[0].float()
        
        return importance.cpu().detach()
