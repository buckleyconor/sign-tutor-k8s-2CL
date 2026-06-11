"""Triton Inference Server client wrapper.

Thin HTTP client around a single classifier model. The same interface scales
from a single-GPU node to a fleet — only the URL changes.
"""
import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "tritonclient[http] is required. Install it inside the tutor-app "
        "container (see Dockerfile)."
    ) from exc


class TritonClassifier:
    def __init__(self, url: str = "triton:8000", model_name: str = "asl_classifier"):
        self._client = httpclient.InferenceServerClient(url=url)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def infer(self, x: np.ndarray) -> np.ndarray:
        """``x``: (D,) or (B, D). Returns (num_classes,) softmax-able logits."""
        if x.ndim == 1:
            x = x[None, :]
        inp = httpclient.InferInput("input", x.shape, "FP32")
        inp.set_data_from_numpy(x.astype(np.float32))
        resp = self._client.infer(self._model_name, [inp])
        return resp.as_numpy("output")[0]
