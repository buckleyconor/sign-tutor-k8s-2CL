"""Embedded-terminal executor configuration.

Security model is specified in Spec 1 §6.5. The allowlist is UX (clear errors,
fewer foot-guns), not the security perimeter — the pod boundary is.
"""
COMMAND_TIMEOUT_SECONDS = 300   # 5 min — long enough to run training in the foreground
MAX_OUTPUT_LINES = 500
WORKDIR = "/app"

# First token of the command (basename) must be in this set.
ALLOWLIST = {
    "kubectl", "curl", "ls", "cat", "echo", "python", "python3",
    "trtexec", "nvidia-smi", "mkdir", "cp", "head", "tail", "grep", "wc",
    "pwd", "date", "env", "whoami", "uptime", "df", "top", "ps", "free", "dmesg",
}
# mkdir/cp are needed for the Module-2 deploy step (create the model-version
# dir and place config.pbtxt on the shared triton-models PVC). They can only
# touch the pod's own mounts; BLOCKED_PATTERNS still rejects rm -rf, /etc, etc.

# Reject if any of these substrings appear anywhere in the command.
BLOCKED_PATTERNS = (
    "delete", "rm -rf", "shred", "dd ", "mkfs",
    "> /etc", ">/etc", "/proc/sys", "docker",
)

# kubectl is inspection-only: the verb (2nd token) must be in this set.
KUBECTL_READONLY_VERBS = {"get", "describe", "logs", "version", "api-resources"}
