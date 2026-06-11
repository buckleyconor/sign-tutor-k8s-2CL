# tutor-app image — NVIDIA PyTorch base (Blackwell-optimised). Bundles trtexec,
# so the lab builds TensorRT engines locally in this pod.
FROM nvcr.io/nvidia/pytorch:25.03-py3

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

WORKDIR /app
COPY . /app

# opencv-python-headless avoids GUI/X11 deps; Gradio renders in the browser. The
# webcam is browser-side (HTTPS via the Ingress), not /dev/video0.
CMD ["python", "-m", "src.ui.app"]
