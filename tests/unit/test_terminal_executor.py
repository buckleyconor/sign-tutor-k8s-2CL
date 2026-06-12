"""Tests for the embedded-terminal executor (Spec 3 §2.7).

run_command is a generator that streams the buffer; we drain it and inspect the
final terminal text.
"""

from src.terminal import config, executor
from src.terminal.executor import run_command


def _final(cmd, history=""):
    out = (history, "")
    for out in run_command(cmd, history):
        pass
    return out[0]


def test_allowlisted_command_runs():
    assert "hi" in _final("echo hi")


def test_disallowed_command_rejected():
    out = _final("scp a b")
    assert "is not permitted" in out


def test_blocked_pattern_rejected():
    out = _final("kubectl delete pod x")
    assert "disallowed pattern" in out
    assert "delete" in out


def test_kubectl_readonly_verbs_only():
    # get passes the gate (kubectl may be absent -> runtime error, but NOT the
    # inspection-only rejection)
    assert "inspection-only" not in _final("kubectl get pods")
    assert "inspection-only" in _final("kubectl apply -f x.yaml")


def test_timeout_kills_command(monkeypatch):
    monkeypatch.setattr(config, "COMMAND_TIMEOUT_SECONDS", 1)
    out = _final("python3 -c 'import time; time.sleep(5)'")
    assert "[Killed after 1s timeout]" in out


def test_output_truncated(monkeypatch):
    monkeypatch.setattr(config, "MAX_OUTPUT_LINES", 5)
    out = _final("python3 -c 'for i in range(50): print(i)'")
    assert "[Output truncated" in out


def test_namespace_injected(monkeypatch):
    monkeypatch.setenv("NAMESPACE", "lab-p7")
    assert "lab-p7" in _final("echo $NAMESPACE")


def test_namespace_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("NAMESPACE", raising=False)
    assert "lab-p1" in _final("echo $NAMESPACE")


def test_empty_command_is_noop():
    assert _final("", history="prev") == "prev"


def test_unbalanced_quotes_handled():
    assert "Could not parse" in _final("echo 'unbalanced")


def test_reject_does_not_spawn(monkeypatch):
    called = {"n": 0}
    real_popen = executor.subprocess.Popen

    def spy(*a, **k):
        called["n"] += 1
        return real_popen(*a, **k)

    monkeypatch.setattr(executor.subprocess, "Popen", spy)
    _final("rm -rf /")
    assert called["n"] == 0
