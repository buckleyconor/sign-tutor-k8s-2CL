"""Integration tests — require a running Triton serving the ASL model.

Run with:  pytest -m integration   (skipped by default via -m "not integration")

These feed the committed real ASL landmark vectors (languages/asl/landmarks.csv)
through the deployed Triton and check predictions — no MediaPipe / fixture images
needed, so the test runs anywhere Triton is reachable (e.g. inside the tutor-app
pod, or with TRITON_URL pointing at a port-forward).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.triton_client import TritonClassifier
from src.registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")


@pytest.fixture(scope="module")
def asl():
    return load_registry(ROOT / "languages")["asl"]


@pytest.mark.integration
def test_inference_returns_correct_shape(asl):
    clf = TritonClassifier(url=TRITON_URL, model_name=asl.triton_model_name)
    out = clf.infer(np.zeros(63, dtype=np.float32))
    assert out.shape == (len(asl.classes),)


@pytest.mark.integration
def test_e2e_accuracy_on_real_landmarks(asl):
    clf = TritonClassifier(url=TRITON_URL, model_name=asl.triton_model_name)
    df = pd.read_csv(ROOT / "languages/asl/landmarks.csv", header=None)
    idx = np.random.default_rng(0).choice(len(df), 50, replace=False)
    correct = 0
    for i in idx:
        feat = df.iloc[i, 3:].to_numpy(dtype=np.float32)
        pred = asl.classes[int(np.argmax(clf.infer(feat)))]
        correct += pred == str(df.iloc[i, 0])
    assert correct / len(idx) >= 0.90, f"top-1 accuracy {correct}/{len(idx)} < 90%"
