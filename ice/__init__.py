"""
ICE: Intervention-Consistent Explanation Framework

A statistically grounded framework for evaluating faithfulness of explanations.
"""

from .operators import (
    BaseOperator,
    DeletionOperator,
    MaskOperator,
    OperatorFamily,
    create_default_operators,
    create_ice_lite_operators
)

from .metrics import (
    ICEScorer,
    ScoreResult,
    compute_auc_over_k,
    aggregate_across_operators
)

from .stats import (
    RandomizationTestResult,
    BootstrapCIResult,
    randomization_test,
    bootstrap_ci,
    benjamini_hochberg,
    ICEStatisticalEvaluator
)

from .extractors import (
    BaseRationaleExtractor,
    AttentionExtractor,
    GradientExtractor,
    IntegratedGradientsExtractor,
    LIMEExtractor,
    LLMAttentionExtractor,
    LLMGradientExtractor,
    get_extractor
)

from .evaluation import (
    ICEConfig,
    ICEResult,
    ICEEvaluator,
    compare_with_eraser
)

__version__ = "1.0.1"
__all__ = [
    # Operators
    "BaseOperator",
    "DeletionOperator", 
    "MaskOperator",
    "OperatorFamily",
    "create_default_operators",
    "create_ice_lite_operators",
    
    # Metrics
    "ICEScorer",
    "ScoreResult",
    "compute_auc_over_k",
    "aggregate_across_operators",
    
    # Statistics
    "RandomizationTestResult",
    "BootstrapCIResult",
    "randomization_test",
    "bootstrap_ci",
    "benjamini_hochberg",
    "ICEStatisticalEvaluator",
    
    # Extractors
    "BaseRationaleExtractor",
    "AttentionExtractor",
    "GradientExtractor",
    "IntegratedGradientsExtractor",
    "LIMEExtractor",
    "LLMAttentionExtractor",
    "LLMGradientExtractor",
    "get_extractor",
    
    # Evaluation
    "ICEConfig",
    "ICEResult",
    "ICEEvaluator",
    "compare_with_eraser"
]
