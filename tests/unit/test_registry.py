from pathlib import Path

import pytest
import yaml

from src.registry import load_registry

FIXTURE = Path(__file__).parent.parent / "fixtures" / "languages"


def test_loads_two_languages():
    reg = load_registry(FIXTURE)
    assert set(reg) == {"asl", "isl"}


def test_asl_isl_one_handed():
    reg = load_registry(FIXTURE)
    assert reg["asl"].input_hands == 1
    assert reg["isl"].input_hands == 1


def test_classes_have_26_letters():
    reg = load_registry(FIXTURE)
    for code in ("asl", "isl"):
        assert len(reg[code].classes) == 26
        assert reg[code].classes[0] == "A"
        assert reg[code].classes[-1] == "Z"


def test_missing_config_yields_empty_registry(tmp_path):
    assert load_registry(tmp_path) == {}


def test_malformed_yaml_fails_clearly(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.yaml").write_text(yaml.safe_dump({"name": "X"}))  # missing fields
    with pytest.raises((KeyError, ValueError)):
        load_registry(tmp_path)
