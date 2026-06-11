import numpy as np

from src.features.normalise import normalise_one_hand
from tests.fixtures.synthetic import synthetic_hand


def test_output_shape_one_hand():
    out = normalise_one_hand(synthetic_hand())
    assert out.shape == (63,)
    assert out.dtype == np.float32


def test_translation_invariance():
    a = normalise_one_hand(synthetic_hand(seed=1))
    b = normalise_one_hand(synthetic_hand(seed=1, translate=(5, -3, 2)))
    assert np.allclose(a, b, atol=1e-6)


def test_scale_invariance():
    a = normalise_one_hand(synthetic_hand(seed=2))
    b = normalise_one_hand(synthetic_hand(seed=2, scale=2.0))
    assert np.allclose(a, b, atol=1e-5)


def test_wrist_at_origin():
    out = normalise_one_hand(synthetic_hand(seed=3))
    assert np.allclose(out[:3], 0.0)


def test_zero_scale_handled():
    lm = np.zeros((21, 3), dtype=np.float32)  # wrist and MCP coincide
    out = normalise_one_hand(lm)
    assert np.all(np.isfinite(out))
