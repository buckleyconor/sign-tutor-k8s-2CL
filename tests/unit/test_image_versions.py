"""Guardrail: the tutor-app base image and the Triton image must come from the
same NGC monthly release, so they ship the same TensorRT version.

A mismatch lets engines build in the tutor-app pod but fail to deserialize in
Triton ("Serialization ... Version tag does not match") — which silently breaks
the Module-2 engine-deploy flow. This caught us once; never again.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# NGC tags look like "26.04-py3" -> release "26.04".
_TAG = re.compile(r":(\d{2}\.\d{2})(?:-\w+)?")


def _release(image: str) -> str:
    m = _TAG.search(image)
    assert m, f"could not parse NGC release from image ref: {image!r}"
    return m.group(1)


def test_pytorch_and_triton_share_ngc_release():
    dockerfile = (ROOT / "Dockerfile").read_text()
    fm = re.search(r"^FROM\s+(nvcr\.io/nvidia/pytorch:\S+)", dockerfile, re.M)
    assert fm, "no NVIDIA pytorch FROM line in Dockerfile"
    pytorch_release = _release(fm.group(1))

    values = yaml.safe_load((ROOT / "k8s/chart/values.yaml").read_text())
    triton_release = _release(values["images"]["triton"])

    assert pytorch_release == triton_release, (
        f"TensorRT version drift: tutor-app base pytorch:{pytorch_release} vs "
        f"tritonserver:{triton_release}. They must share an NGC release so "
        f"engines built in tutor-app load in Triton."
    )
