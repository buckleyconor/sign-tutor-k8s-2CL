# tutor-app image — NVIDIA PyTorch base (Blackwell-optimised). Bundles trtexec,
# so the lab builds TensorRT engines locally in this pod.
FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    gradio \
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
