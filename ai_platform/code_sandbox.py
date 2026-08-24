"""Code AI - sandboxed execution (spec Part 11).

Honest scoping: this is PROCESS isolation with a hard wall-clock
timeout, restricted builtins, and no filesystem/network access from
within the executed code - not container- or VM-level security
isolation (no Docker/gVisor/seccomp available in this environment). It
stops a runaway loop and blocks the obvious escape hatches (import os,
open(), __import__, exec of further code), which is a real, meaningful
safety improvement over "no sandbox at all" - and is documented as
exactly that in the registry, not oversold as production-grade.

Never executes generated code directly in this process - always a
fresh `python -I -S` subprocess with a timeout, matching spec Part 11's
"never execute generated code directly on the production host" +
Part 27's "never execute arbitrary model-generated commands" by
constraining what that subprocess can do, not by trusting the input.
"""

import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 5.0

# Names that, if present as a standalone identifier, indicate an escape
# attempt (filesystem, network, process, or dynamic-import access). This
# is a blocklist, not a proof of safety - real isolation comes from the
# restricted __builtins__ injected into the executed code below, this is
# a fast first-pass rejection.
_FORBIDDEN_TOKENS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "__import__", "open(", "eval(", "exec(", "compile(",
    "os.system", "os.popen", "globals()", "locals()", "__builtins__",
]

_ALLOWED_BUILTINS = [
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sum", "min", "max",
    "sorted", "reversed", "abs", "round", "int", "float", "str", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "StopIteration", "True", "False", "None",
]

# Runs the USER code inside exec() with a purpose-built globals dict, so the
# frame exec() creates for it derives f_builtins from that dict at creation
# time - this is what actually restricts builtin name resolution in CPython.
# (Reassigning __builtins__ in the *same* already-running frame, tried
# first, does NOT work - f_builtins is cached when a frame is created, not
# re-read on each assignment. Caught this with a `dir()` test before
# shipping it; see the code_sandbox tests.)
_RUNNER_TEMPLATE = """
import builtins as _b
_allowed = {allowed!r}
_safe_builtins = {{k: getattr(_b, k) for k in _allowed if hasattr(_b, k)}}
_user_code = {user_code!r}
_restricted_globals = {{'__builtins__': _safe_builtins}}
exec(compile(_user_code, '<sandboxed>', 'exec'), _restricted_globals)
"""


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    rejected_reason: str = None

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.rejected_reason


def _static_check(code: str) -> str:
    """Cheap pre-flight rejection of obvious escape attempts. Returns a
    reason string if rejected, None if it passes this check (still runs
    inside the restricted-builtins subprocess regardless)."""
    lowered = code.lower()
    for token in _FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            return f"Rejected: contains forbidden token '{token}'"
    return None


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ExecutionResult:
    reason = _static_check(code)
    if reason:
        return ExecutionResult(stdout="", stderr="", returncode=-1,
                                timed_out=False, rejected_reason=reason)

    full_source = _RUNNER_TEMPLATE.format(allowed=_ALLOWED_BUILTINS, user_code=code)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_source)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", script_path],  # isolated + no site packages
            capture_output=True, text=True, timeout=timeout,
        )
        return ExecutionResult(
            stdout=proc.stdout[-4000:], stderr=proc.stderr[-2000:],
            returncode=proc.returncode, timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            stdout=(e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=f"Execution exceeded {timeout}s timeout - terminated.",
            returncode=-1, timed_out=True,
        )
    finally:
        import os
        try:
            os.unlink(script_path)
        except OSError:
            pass
