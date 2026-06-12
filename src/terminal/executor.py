"""Server-side command executor for the embedded terminal.

Each submitted line runs as a subprocess inside the tutor-app pod, operating on
the volumes mounted there (including the shared ``triton-models`` PVC). Output is
streamed back line-by-line so long training runs show progress live.
"""

import os
import shlex
import subprocess
import threading

from src.terminal import config


def _reject(cmd: str) -> str | None:
    """Return an error string if the command is not permitted, else None."""
    lowered = cmd.lower()
    for pat in config.BLOCKED_PATTERNS:
        if pat in lowered:
            return f"Blocked: command contains disallowed pattern '{pat.strip()}'."
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return "Could not parse command (unbalanced quotes?)."
    if not tokens:
        return None
    prog = os.path.basename(tokens[0])
    if prog not in config.ALLOWLIST:
        allowed = ", ".join(sorted(config.ALLOWLIST))
        return f"Command '{prog}' is not permitted. Allowed commands: {allowed}"
    if prog == "kubectl" and (
        len(tokens) < 2 or tokens[1] not in config.KUBECTL_READONLY_VERBS
    ):
        return "kubectl is inspection-only here. Allowed verbs: " + ", ".join(
            sorted(config.KUBECTL_READONLY_VERBS)
        )
    return None


def run_command(cmd: str, history: str = ""):
    """Generator — streams the terminal buffer as the command runs.

    Yields ``(terminal_text, cleared_input)``. ``NAMESPACE`` is injected so
    kubectl/scripts default to the participant's namespace.
    """
    cmd = (cmd or "").strip()
    prior = (history or "").rstrip()

    def render(body_lines):
        return (prior + ("\n" if prior else "") + "\n".join(body_lines)).strip()

    if not cmd:
        yield prior, ""
        return

    err = _reject(cmd)
    if err is not None:
        yield render([f"$ {cmd}", err]), ""
        return

    env = {**os.environ, "NAMESPACE": os.environ.get("NAMESPACE", "lab-p1")}
    # In the pod the working dir is /app; off-cluster (tests) fall back to cwd.
    workdir = config.WORKDIR if os.path.isdir(config.WORKDIR) else None
    lines = [f"$ {cmd}"]
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # Watchdog kills the process at the deadline even if it produces no output.
    timer = threading.Timer(config.COMMAND_TIMEOUT_SECONDS, proc.kill)
    timer.start()
    truncated = False
    try:
        for line in proc.stdout:  # streams line-by-line
            lines.append(line.rstrip("\n"))
            if len(lines) > config.MAX_OUTPUT_LINES:
                lines = lines[: config.MAX_OUTPUT_LINES]
                lines.append("[Output truncated — re-run with a narrower filter]")
                proc.kill()
                truncated = True
                break
            yield render(lines), ""  # live update to the UI
        proc.wait()
    finally:
        timer.cancel()

    if not truncated and proc.returncode and proc.returncode < 0:
        lines.append(f"[Killed after {config.COMMAND_TIMEOUT_SECONDS}s timeout]")
    yield render(lines), ""
