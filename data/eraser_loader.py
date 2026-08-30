"""
ERASER Dataset Loaders

Loads datasets from the ERASER benchmark:
- e-SNLI (Natural Language Inference with explanations)
- BoolQ (Boolean Questions)
- MultiRC (Multi-sentence Reading Comprehension)
- FEVER (Fact Extraction and Verification)
- Movies (Sentiment with rationales)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def _load_dataset_pinned(path, *args, revision: Optional[str] = None, **kwargs):
    """
    Wrapper to optionally pin HuggingFace dataset revisions.
    If revision is None, behaves exactly like load_dataset().
    """
    if revision:
        kwargs["revision"] = revision
    return load_dataset(path, *args, **kwargs)


class ERASERDataset(Dataset):
    """Base class for ERASER benchmark datasets"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "test",
        revision: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.revision = revision
        self.examples = []
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict:
        return self.examples[idx]
    
    def _tokenize(self, text: str, text_pair: str = None) -> Dict:
        """Tokenize text with proper formatting"""
        encoded = self.tokenizer(
            text,
            text_pair,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


class ESNLIDataset(ERASERDataset):
    """
    e-SNLI: SNLI with human-annotated explanations.
    
    Labels: entailment (0), neutral (1), contradiction (2)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        split: str = "test",
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load SNLI from HuggingFace datasets.

        The legacy script-based "esnli" repo is unsupported by datasets>=3.0.
        We only need premise/hypothesis/label (not the human explanations),
        which SNLI provides with identical splits and label encoding.
        """
        dataset = _load_dataset_pinned("stanfordnlp/snli", split=self.split, revision=self.revision)
        
        # Filter out examples with -1 label (unlabeled)
        dataset = dataset.filter(lambda x: x["label"] != -1)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = item["label"]  # Already 0=entailment, 1=neutral, 2=contradiction
            
            encoded = self._tokenize(premise, hypothesis)
            
            self.examples.append({
                **encoded,
                "label": label,
                "premise": premise,
                "hypothesis": hypothesis,
            })


class BoolQDataset(ERASERDataset):
    """
    BoolQ: Boolean questions about passages.
    
    Labels: False (0), True (1)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "validation",  # BoolQ test set doesn't have labels
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load BoolQ from HuggingFace datasets"""
        dataset = _load_dataset_pinned("google/boolq", split=self.split, revision=self.revision)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            question = item["question"]
            passage = item["passage"]
            label = int(item["answer"])  # True=1, False=0
            
            # Format: [CLS] question [SEP] passage [SEP]
            encoded = self._tokenize(question, passage)
            
            self.examples.append({
                **encoded,
                "label": label,
                "question": question,
                "passage": passage
            })


class MultiRCDataset(ERASERDataset):
    """
    MultiRC: Multi-sentence Reading Comprehension.
    
    Labels: False (0), True (1) - whether answer is correct
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "validation",
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load MultiRC from HuggingFace datasets.

        Uses the parquet mirror aps/super_glue: the legacy script-based
        "super_glue" repo is unsupported by datasets>=3.0.
        """
        dataset = _load_dataset_pinned("aps/super_glue", "multirc", split=self.split, revision=self.revision)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            paragraph = item["paragraph"]
            question = item["question"]
            answer = item["answer"]
            label = item["label"]
            
            # Format: question + answer [SEP] paragraph
            qa_text = f"{question} {answer}"
            encoded = self._tokenize(qa_text, paragraph)
            
            self.examples.append({
                **encoded,
                "label": label,
                "paragraph": paragraph,
                "question": question,
                "answer": answer
            })


class MovieReviewDataset(ERASERDataset):
    """
    Movie Reviews with sentiment rationales.
    
    Uses SST-2 as proxy (original ERASER movies dataset requires manual download).
    Labels: negative (0), positive (1)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        split: str = "validation",
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load SST-2 from HuggingFace datasets"""
        dataset = _load_dataset_pinned("nyu-mll/glue", "sst2", split=self.split, revision=self.revision)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            text = item["sentence"]
            label = item["label"]
            
            encoded = self._tokenize(text)
            
            self.examples.append({
                **encoded,
                "label": label,
                "text": text
            })


class IMDBDataset(ERASERDataset):
    """
    IMDB Movie Reviews - longer sequences for better faithfulness evaluation.
    
    Labels: negative (0), positive (1)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "test",
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load IMDB from HuggingFace datasets"""
        dataset = _load_dataset_pinned("stanfordnlp/imdb", split=self.split, revision=self.revision)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            text = item["text"]
            label = item["label"]  # 0=negative, 1=positive
            
            encoded = self._tokenize(text)
            
            self.examples.append({
                **encoded,
                "label": label,
                "text": text[:200] + "..." if len(text) > 200 else text  # Truncate for display
            })


class FEVERDataset(ERASERDataset):
    """
    FEVER: Fact Extraction and VERification.
    
    Labels: SUPPORTS (0), REFUTES (1), NOT ENOUGH INFO (2)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "validation",  # "labelled_dev" in original
        max_examples: int = None,
        revision: Optional[str] = None
    ):
        super().__init__(tokenizer, max_length, split, revision=revision)
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        self._load_data(max_examples)
    
    def _load_data(self, max_examples: int = None):
        """Load FEVER from HuggingFace datasets"""
        try:
            # Try loading FEVER dataset
            if self.split == "validation":
                hf_split = "labelled_dev"
            else:
                hf_split = self.split
            dataset = _load_dataset_pinned("fever", "v1.0", split=hf_split, revision=self.revision)
        except Exception:
            # Fallback: create empty dataset
            print("Warning: Could not load FEVER dataset. Using empty placeholder.")
            return
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        for item in dataset:
            claim = item["claim"]
            label_str = item["label"]
            label = self.label_map.get(label_str, 2)
            
            # For simplicity, just use claim (full FEVER requires evidence retrieval)
            encoded = self._tokenize(claim)
            
            self.examples.append({
                **encoded,
                "label": label,
                "claim": claim
            })


def get_eraser_dataset(
    name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "test",
    max_length: int = 512,
    max_examples: int = None,
    revision: Optional[str] = None
) -> ERASERDataset:
    """
    Factory function to get ERASER dataset by name.
    
    Args:
        name: Dataset name ("esnli", "boolq", "multirc", "movies", "fever")
        tokenizer: HuggingFace tokenizer
        split: Data split
        max_length: Maximum sequence length
        max_examples: Limit number of examples
        revision: Optional HF dataset revision/commit hash for pinning
        
    Returns:
        ERASERDataset instance
    """
    datasets = {
        "esnli": ESNLIDataset,
        "boolq": BoolQDataset,
        "multirc": MultiRCDataset,
        "movies": MovieReviewDataset,
        "fever": FEVERDataset,
        "sst2": MovieReviewDataset,  # Alias - SST-2 used as movie review proxy
        "imdb": IMDBDataset  # Long movie reviews
    }
    
    if name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")
    
    return datasets[name.lower()](
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        max_examples=max_examples,
        revision=revision
    )


def create_dataloader(
    dataset: ERASERDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader from ERASER dataset"""
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.tensor([item["label"] for item in batch])
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
