import numpy as np

from src.features import build_feature_vector
from src.registry import Language
from tests.fixtures.synthetic import synthetic_hand


def _lang():
    return Language(
        name="ASL",
        code="asl",
        input_hands=1,
        classes=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        triton_model_name="asl_classifier",
        references_dir=None,
        notes={},
    )


def test_one_handed_language_dispatches_to_normalise_one_hand():
    detections = [("Right", synthetic_hand(seed=4))]
    feat = build_feature_vector(_lang(), detections)
    assert feat is not None
    assert feat.shape == (63,)
    assert np.allclose(feat[:3], 0.0)  # wrist at origin -> normalised


def test_no_detections_returns_none_for_one_handed():
    assert build_feature_vector(_lang(), []) is None
