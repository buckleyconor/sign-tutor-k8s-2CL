"""Language registry — the keystone of the multi-language design.

Every other component reads from the registry. Adding a new sign language is a
data-only operation: drop a `languages/<code>/config.yaml`, a model, and a
reference image set; no code changes required.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_FIELDS = ("name", "code", "input_hands", "classes", "triton_model_name")


@dataclass
class Language:
    name: str
    code: str
    input_hands: int
    classes: list[str]
    triton_model_name: str
    references_dir: Path
    notes: dict


def load_registry(root: Path = Path("languages")) -> dict[str, Language]:
    """Load every `languages/*/config.yaml` into a `{code: Language}` map.

    An empty/missing directory yields an empty registry rather than raising. A
    config missing a required field raises a clear ``KeyError`` naming the field.
    """
    registry: dict[str, Language] = {}
    root = Path(root)
    for cfg_path in sorted(root.glob("*/config.yaml")):
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        missing = [k for k in _REQUIRED_FIELDS if k not in data]
        if missing:
            raise KeyError(
                f"{cfg_path}: missing required field(s): {', '.join(missing)}"
            )
        lang_dir = cfg_path.parent
        registry[data["code"]] = Language(
            name=data["name"],
            code=data["code"],
            input_hands=int(data["input_hands"]),
            classes=list(data["classes"]),
            triton_model_name=data["triton_model_name"],
            references_dir=lang_dir / data.get("references_dir", "references"),
            notes=data.get("notes", {}) or {},
        )
    return registry
