# tutor-app image — NVIDIA PyTorch base (Blackwell-optimised). Bundles trtexec,
# so the lab builds TensorRT engines locally in this pod.
#
# IMPORTANT: this tag MUST match the Triton image's TensorRT version. TRT engines
# are serialization-version-specific, so an engine built here is loaded by Triton
# only if both use the same TRT. pytorch:26.04 ships TRT 10.16.1, matching
# tritonserver:26.04 (TRT 10.16.01). (pytorch:25.03 has TRT 10.9 — its engines
# fail to deserialize in a 10.16 Triton: "Version tag does not match".)
FROM nvcr.io/nvidia/pytorch:26.04-py3

# The NVIDIA base pins protobuf==4.24.4 via PIP_CONSTRAINT, which conflicts with
# mediapipe 0.10.18 (needs protobuf>=4.25.3). Clear the constraint for our deps
# and pin a compatible protobuf — the base components we rely on (torch, onnx)
# work fine with protobuf 4.25.
RUN PIP_CONSTRAINT= pip install --no-cache-dir \
    "protobuf>=4.25.3,<5" \
    "mediapipe==0.10.18" \
    opencv-python-headless \
    "gradio>=5,<6" \
    tritonclient[http,grpc] \
    pyyaml \
    pandas \
    onnx onnxscript \
    pytest pytest-cov

# mediapipe depends on full opencv (opencv-contrib-python), which needs libGL +
# glib at runtime. The slim 26.04 base lacks them (25.03 happened to have them).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# opencv-python-headless avoids GUI/X11 deps; Gradio renders in the browser. The
# webcam is browser-side (HTTPS via the Ingress), not /dev/video0.
CMD ["python", "-m", "src.ui.app"]
