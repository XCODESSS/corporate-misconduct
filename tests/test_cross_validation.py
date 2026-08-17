"""Cross-validation policy tests."""

import numpy as np
import pytest
from src.evaluation.cross_validation import WalkForwardCV


def test_in_sample_threshold_optimization_is_rejected() -> None:
    cv = WalkForwardCV(min_fraud_per_fold=1)
    with pytest.raises(ValueError, match="independent threshold-validation scores"):
        cv._select_threshold(
            y_train=np.array([0, 1]),
            train_score=np.array([0.1, 0.9]),
            default_threshold=0.5,
            should_optimize=True,
        )
