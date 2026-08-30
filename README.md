# ICE: Intervention-Consistent Explanation Evaluation

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026%20Findings-blue)](https://arxiv.org/abs/2603.18579)
[![PyPI](https://img.shields.io/pypi/v/ice-faithfulness)](https://pypi.org/project/ice-faithfulness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs"** (EMNLP 2026 Findings).

## Key Finding

Faithfulness is **operator-dependent**: switching the intervention operator crosses the positive-evidence threshold in **18% of configurations** (5/28), with gaps up to 44 percentage points. One-third of configurations are anti-faithful (worse than random), invisible without randomized baselines.

## What ICE Does

ICE evaluates whether model explanations (attention, gradient) identify tokens that are genuinely more important than random tokens, under multiple intervention operators.

```
Input: "a gorgeous, witty, seductive movie"   (positive sentiment)
Rationale: {gorgeous, seductive}

               Deletion    Retrieval Infill
NSR:           0.79        0.39
Win Rate:      92%         68%
Verdict:       Faithful    Weakly Faithful   <-- operator changes conclusion
```

## Installation

```bash
pip install ice-faithfulness
```

## Quick Start

```python
from ice import ICEEvaluator, ICEConfig

config = ICEConfig(
    k=0.2,                # top 20% tokens as rationale
    n_permutations=50,    # random baselines
    operators=["delete", "retrieval"],
)

evaluator = ICEEvaluator(model, tokenizer, config)
result = evaluator.evaluate(dataset, extractor="attention")

print(f"Win Rate: {result.win_rate:.1%}")
print(f"Effect Size: {result.effect_size:.2f}")
print(f"Operators agree: {result.operators_agree}")
```

## Core Modules

| Module | Purpose |
|--------|---------|
| `ice.evaluation` | Main evaluator (`ICEEvaluator`, `ICEConfig`) |
| `ice.extractors` | Attention, gradient, IG, LIME extractors |
| `ice.operators` | Deletion, mask, retrieval infill operators |
| `ice.metrics` | NSR scoring, AUC-over-k |
| `ice.stats` | Randomization tests, bootstrap CI, BH correction |
| `ice.retrieval_operator` | Leave-one-out retrieval infill |

## Evaluation Scripts

```bash
# English benchmarks (7 models x 4 tasks)
python run_ice_eval.py --model llama-3.2-3b --dataset sst2

# Multilingual (6 languages)
python run_multilingual.py --model qwen-2.5-7b --lang french

# Operator comparison
python run_retrieval.py --model mistral-7b --dataset esnli
```

## Key Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| Win Rate | 0-100% | % of random baselines beaten by the rationale |
| Effect Size | Cohen's d | Magnitude (>0.8 = large, <0 = anti-faithful) |
| Operator Agreement | bool | Both operators give same verdict |

## Results Summary

Evaluated on **7 LLMs** (1.5B-8B), **4 English tasks**, **6 non-English languages**, **2 attribution methods**:

- Operator gaps reach **44 pp** (e.g., Llama-3.2 e-SNLI: 86% deletion vs 43% retrieval)
- **Anti-faithfulness** in 1/3 of configurations (gradient selects function words)
- No correlation between faithfulness and human plausibility (|r| < 0.04)
- Cross-lingual faithfulness spans 16%-83%, not predicted by tokenization alone

## Citation

```bibtex
@inproceedings{basu2026ice,
    title={ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs},
    author={Basu, Abhinaba and Chakraborty, Pavan},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
    year={2026}
}
```

## License

MIT
