"""Integration tests — require a running Triton with at least one model loaded.

Run with:  pytest -m integration
Skipped by default via:  pytest -m "not integration"
"""
import cv2
import pytest

from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.inference.triton_client import TritonClassifier
from src.registry import load_registry


@pytest.mark.integration
@pytest.mark.parametrize("letter", ["A", "B", "C", "L", "Y"])
def test_known_image_predicts_correctly(letter):
    langs = load_registry()
    asl = langs["asl"]
    tracker = HandTracker(static_image_mode=True)
    classifier = TritonClassifier(model_name=asl.triton_model_name)

    img = cv2.imread(f"tests/fixtures/asl_reference/{letter}.png")
    assert img is not None, f"missing fixture image for {letter}"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = tracker.process(rgb)
    feat = build_feature_vector(asl, detections)
    assert feat is not None, "MediaPipe failed to detect hand in reference image"

    logits = classifier.infer(feat)
    pred = asl.classes[int(logits.argmax())]
    assert pred == letter
