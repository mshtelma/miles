#!/usr/bin/env python3
"""The OfficeQA ``compute`` tool: run the agent's Python in a bubblewrap sandbox.

The agent writes Python to parse the tables it retrieved and compute answers (sums over months,
ratios, percent changes); numpy and pandas are available. The boundary is the kernel, not an
in-process filter (an AST or audit-hook filter can be escaped):

  * every call is a fresh interpreter in new user, mount, network, PID, IPC and UTS namespaces, so
    it has no network at all;
  * the root is read-only and the directories holding the corpus, questions, weights and credentials
    (/Volumes, /dbfs, /home, /root, /local_disk0, /mnt, /media, /srv, /databricks) are masked by empty
    tmpfs; the agent sees retrieved text only through the tool outputs;
  * the environment is cleared except for interpreter discovery (never PYTHONPATH or credentials);
  * the parent kills the sandbox after ``OQ_COMPUTE_TIMEOUT_S`` seconds (``--die-with-parent`` takes
    the child with it), and the child sets CPU, memory (``OQ_COMPUTE_MEM_MB``) and file-size limits.

Inside, the code runs ordinary Python. Without a usable ``bwrap`` (the AIR image installs it) the tool
runs nothing and returns an error.

Public API: ``run_code(code: str) -> str``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

_MAX_CODE = 20_000
_MAX_OUTPUT = 8_000
_TIMEOUT_S = float(os.environ.get("OQ_COMPUTE_TIMEOUT_S", "15"))
_MEM_MB = int(os.environ.get("OQ_COMPUTE_MEM_MB", "0"))  # 0 = rely on cgroup; else RLIMIT_AS
# Directories to MASK (empty tmpfs) inside the sandbox: data, secrets, scratch, homes.
# User code receives the numerals from search/grep/read through tool outputs. Those
# parent-side paths must be invisible INSIDE compute so model code cannot open the raw
# corpus, benchmark CSVs, answer keys, model weights, or the reward implementation.
_MASK_CANDIDATES = (
    "/Volumes",
    "/dbfs",
    "/home",
    "/root",
    "/local_disk0",
    "/mnt",
    "/media",
    "/srv",
    "/databricks",
)


# =============================== CHILD SIDE ===================================
def _run_child() -> None:
    """Executed in the fresh child interpreter INSIDE bwrap (``py_compute.py --child``).
    Reads code from stdin, runs it as ordinary Python (the kernel namespace is the
    boundary -- see the module docstring), writes one JSON line to stdout.

    Resource bounds are set here (bwrap isolates FS/net but not CPU/mem/write)."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (int(_TIMEOUT_S) + 5, int(_TIMEOUT_S) + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if _MEM_MB > 0:
            b = _MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
    except Exception:  # noqa: BLE001
        pass

    code = sys.stdin.read()

    g: dict[str, object] = {"__name__": "__compute__", "__builtins__": __builtins__}
    for mod, names in (("numpy", ("numpy", "np")), ("pandas", ("pandas", "pd"))):
        try:
            m = __import__(mod)
            for n in names:
                g[n] = m
        except Exception:  # noqa: BLE001
            pass

    result: dict[str, object] = {"ok": False, "stdout": "", "result": None, "err": None}
    buf = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):  # capture trailing expr
            last = tree.body.pop()
            tree.body.append(ast.Assign(targets=[ast.Name(id="__result__", ctx=ast.Store())], value=last.value))
            ast.fix_missing_locations(tree)
        compiled = compile(tree, "<compute>", "exec")
    except SyntaxError as e:
        result["err"] = f"SyntaxError: {e}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return
    except Exception as e:  # noqa: BLE001
        result["err"] = f"{type(e).__name__}: {e}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return

    try:
        with contextlib.redirect_stdout(buf):
            exec(compiled, g)  # noqa: S102 - isolated child under bwrap
        result["ok"] = True
        rv = g.get("__result__")
        result["result"] = None if rv is None else repr(rv)
    except Exception as e:  # noqa: BLE001
        result["err"] = f"{type(e).__name__}: {e}"
    result["stdout"] = buf.getvalue()
    # Write to the real stdout fd (already open -> no file-open needed).
    os.write(1, json.dumps(result).encode("utf-8", "replace"))


# =============================== PARENT SIDE ==================================
_SANDBOX_OK: bool | None = None


def _sandbox_ok() -> bool:
    """True iff a REAL bwrap sandbox is usable here: the binary exists AND the runtime
    actually permits the unshare (present-but-blocked userns -> False). Cached."""
    global _SANDBOX_OK
    if _SANDBOX_OK is not None:
        return _SANDBOX_OK
    bw = shutil.which("bwrap")
    if not bw:
        _SANDBOX_OK = False
        return False
    try:
        probe = [
            bw,
            "--unshare-user",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "true",
        ]
        r = subprocess.run(probe, capture_output=True, timeout=15)
        _SANDBOX_OK = r.returncode == 0
    except Exception:  # noqa: BLE001
        _SANDBOX_OK = False
    return _SANDBOX_OK


def _mask_dirs() -> list[str]:
    """Data/secret/scratch dirs to hide with an empty tmpfs -- but never one that
    contains the interpreter or its venv (that would break Python inside the sandbox)."""
    py = {
        os.path.realpath(sys.prefix),
        os.path.realpath(sys.base_prefix),
        os.path.dirname(os.path.realpath(sys.executable)),
    }
    out = []
    for d in _MASK_CANDIDATES:
        if not os.path.isdir(d):
            continue
        rd = os.path.realpath(d)
        if any(p == rd or p.startswith(rd + os.sep) for p in py):
            continue  # interpreter lives here -> keep it
        out.append(d)
    return out


def _child_setenv() -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "MPLCONFIGDIR": "/tmp",
        "XDG_CACHE_HOME": "/tmp",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Pass the resource knobs THROUGH -- the child re-reads them at import (the old
        # _child_env dropped OQ_COMPUTE_MEM_MB, so RLIMIT_AS was never set in the child).
        "OQ_COMPUTE_MEM_MB": str(_MEM_MB),
        "OQ_COMPUTE_TIMEOUT_S": str(_TIMEOUT_S),
    }
    # Interpreter/runtime discovery only, never app secrets or PYTHONPATH. The masked
    # /home and /local_disk0 paths would otherwise expose the repository/corpus inside
    # the child; pandas/numpy/scipy are imported from the interpreter's site-packages.
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH"):
        v = os.environ.get(k)
        if v:
            env[k] = v
    return env


_CHILD_IN_SANDBOX = "/tmp/_oq_compute_child.py"


def _bwrap_cmd() -> list[str]:
    cmd = [
        shutil.which("bwrap"),
        "--unshare-user",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",  # everything readable...
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for d in _mask_dirs():  # ...except data/secret/scratch, masked empty
        cmd += ["--tmpfs", d]
    # Expose THIS script at a stable, UNMASKED path. Its real directory may itself be
    # masked (the snapshot lands under /local_disk0 or /home on the runner), so running
    # `python <real_path> --child` would hit "No such file". bwrap resolves the bind
    # SOURCE in the OUTER namespace, so the masked view does not block the bind.
    cmd += ["--ro-bind", os.path.abspath(__file__), _CHILD_IN_SANDBOX]
    cmd += ["--chdir", "/tmp", "--clearenv"]
    for k, v in _child_setenv().items():
        cmd += ["--setenv", k, v]
    return cmd + [sys.executable, _CHILD_IN_SANDBOX, "--child"]


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + "\n... (output truncated)"


def run_code(code: str) -> str:
    """Execute one self-contained Python snippet in an isolated bwrap child; return text."""
    code = str(code or "").strip()
    if not code:
        return "Error: 'code' must be a non-empty string"
    if len(code) > _MAX_CODE:
        return f"Error: code exceeds {_MAX_CODE} chars"

    if not _sandbox_ok():
        return (
            "Error: compute sandbox unavailable -- bubblewrap (bwrap) is not installed or "
            "user namespaces are blocked here. Refusing to execute code without an isolation boundary."
        )

    try:
        proc = subprocess.Popen(
            _bwrap_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            close_fds=True,
        )
    except Exception as e:  # noqa: BLE001 - FAIL CLOSED, never run in-process
        return f"Error: compute sandbox failed to launch ({type(e).__name__}: {e})"

    try:
        out, err = proc.communicate(code.encode("utf-8"), timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()  # SIGKILL bwrap -> --die-with-parent kills child
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return _clip(f"Error: TimeoutError: code execution exceeded {_TIMEOUT_S:.0f}s")

    if not out:
        tail = (err or b"").decode("utf-8", "replace").strip()[-500:]
        rc = proc.returncode
        if rc and rc < 0:
            return _clip(f"Error: compute child terminated by signal {-rc} (killed: OOM / resource limit).")
        return _clip(f"Error: compute produced no result (rc={rc}).{(' stderr: ' + tail) if tail else ''}")
    try:
        data = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return "Error: compute returned an unparseable result."

    # Uniform output cap on EVERY path -- the error branch previously bypassed it, so
    # `raise ValueError('x'*10000)` returned ~10k chars past the advertised cap.
    if data.get("err"):
        pre = str(data.get("stdout") or "").strip()
        msg = f"Error: {str(data['err'])[:2000]}"
        return _clip(f"{msg}\nOutput before error:\n{pre[:1500]}" if pre else msg)

    parts = []
    if data.get("result") is not None:
        parts.append(f"Result: {data['result']}")
    so = str(data.get("stdout") or "").rstrip()
    if so:
        parts.append(f"Output:\n{so}")
    return _clip("\n\n".join(parts) if parts else "(code executed successfully, no output)")


if __name__ == "__main__":
    if "--child" in sys.argv:
        _run_child()
