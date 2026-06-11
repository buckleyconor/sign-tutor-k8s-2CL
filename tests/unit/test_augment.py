import numpy as np

from training.augment import augment_one_hand


def _vec(seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, 63).astype(np.float32)


def test_output_shape_and_dtype():
    out = augment_one_hand(_vec(), np.random.default_rng(0))
    assert out.shape == (63,)
    assert out.dtype == np.float32


def test_deterministic_for_same_seed():
    v = _vec(1)
    a = augment_one_hand(v, np.random.default_rng(5))
    b = augment_one_hand(v, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_mirror_defaults_off():
    v = _vec(2)
    # default call must equal explicit mirror=False (noise+rotation only)
    default = augment_one_hand(v, np.random.default_rng(3))
    explicit = augment_one_hand(v, np.random.default_rng(3), mirror=False)
    assert np.array_equal(default, explicit)


def test_mirror_flag_is_reachable():
    # Across seeds, enabling mirror must change at least one output vs. off
    # (proving the flip path exists and is gated by the flag).
    v = _vec(4)
    differs = any(
        not np.array_equal(
            augment_one_hand(v, np.random.default_rng(s), mirror=False),
            augment_one_hand(v, np.random.default_rng(s), mirror=True),
        )
        for s in range(20)
    )
    assert differs
