"""Export a trained checkpoint to ONNX (the portable artefact).

    python training/export_onnx.py \
        --checkpoint checkpoints/isl/best.pt \
        --output languages/isl/model.onnx
"""
import argparse
import sys
from pathlib import Path

# Allow `python training/export_onnx.py ...` (script form) to resolve packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
import torch

from training.model_one_hand import OneHandClassifier


def _consolidate(output: Path) -> None:
    """Force a single self-contained ONNX (no external-data sidecar).

    Recent torch exporters externalise weights into `<model>.onnx.data`, but our
    delivery path (ConfigMap -> Triton init container) ships only the .onnx, so
    the weights must be embedded. Re-save inline and remove any sidecar.
    """
    model = onnx.load(str(output), load_external_data=True)
    onnx.save_model(model, str(output), save_as_external_data=False)
    for sidecar in output.parent.glob(output.name + "*.data"):
        sidecar.unlink()


def export(checkpoint: Path, output: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    num_classes = ckpt.get("num_classes", 26)
    model = OneHandClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 63)
    torch.onnx.export(
        model, dummy, str(output),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    _consolidate(output)
    print(f"Exported {checkpoint} -> {output} ({num_classes} classes, self-contained)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
