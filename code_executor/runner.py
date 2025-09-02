import os
import signal
import subprocess
import sys
import tempfile
from typing import Optional


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def execute_code(code: str, timeout: float) -> str:
    """
    Execute `code` as a brand-new Python script in a subprocess.

    - stderr is redirected to stdout
    - on success: return stdout
    - on error: return stdout (which includes traceback)
    - on timeout: return f"Execution timed out after {timeout} seconds."

    This implementation follows the repository's existing pattern of running
    untrusted code in a separate process (no extra sandboxing is added).
    """
    tmp_file = None
    proc: Optional[subprocess.Popen] = None
    try:
        # Create a temp directory to hold the guard module and the user script
        tempdir_obj = None
        temp_dir = None
        tempdir_obj = tempfile.TemporaryDirectory()
        temp_dir = tempdir_obj.name
        guard_path = os.path.join(temp_dir, "__reliability_guard__.py")
        # copy guard module content from the repository file
        repo_guard = os.path.join(os.path.dirname(__file__), "__reliability_guard__.py")
        try:
            with open(repo_guard, "r") as rg, open(guard_path, "w") as gw:
                gw.write(rg.read())
        except Exception:
            # fallback: write a minimal no-op guard if copy fails
            with open(guard_path, "w") as gw:
                gw.write("def reliability_guard(maximum_memory_bytes=None):\n    pass\n")

        # write the user script, but only prepend minimal import + call lines
        user_script_path = os.path.join(temp_dir, "user_script.py")
        with open(user_script_path, "w") as f:
            f.write("from __reliability_guard__ import reliability_guard\n")
            f.write("reliability_guard()\n")
            f.write("# begin user code\n")
            f.write(code)
            f.flush()
            tmp_file = user_script_path

        # Start process in its own process group so we can kill the whole group on timeout
        proc = subprocess.Popen(
            [sys.executable, tmp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=temp_dir,
        )

        try:
            out_bytes, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # try to kill the process group first
            _kill_process_group(proc)
            return f"Execution timed out after {timeout} seconds."

        # decode output
        try:
            out = out_bytes.decode("utf-8", errors="replace") if out_bytes is not None else ""
        except Exception:
            out = str(out_bytes)

        return out
        
    finally:
        # cleanup TemporaryDirectory if created
        try:
            if tempdir_obj is not None:
                tempdir_obj.cleanup()
        except Exception:
            pass
        # cleanup temp file if it was created outside of TemporaryDirectory
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
