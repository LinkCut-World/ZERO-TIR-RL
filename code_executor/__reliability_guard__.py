"""
Light-weight reliability guard extracted from runner preamble.
This module is intended to be copied alongside the temporary user script
so the script can `from __reliability_guard__ import reliability_guard`.
"""
import faulthandler
import os
import platform
import sys


def reliability_guard(maximum_memory_bytes=None):
    try:
        if maximum_memory_bytes is not None:
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
            )
            if not platform.uname().system == "Darwin":
                resource.setrlimit(
                    resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
                )
    except Exception:
        pass

    try:
        faulthandler.disable()
    except Exception:
        pass

    try:
        import builtins

        builtins.quit = None
    except Exception:
        pass

    try:
        if os.putenv:
            os.environ["OMP_NUM_THREADS"] = "20"
    except Exception:
        pass

    # disable many dangerous os / shutil / subprocess functions by replacing
    # them with a wrapper that raises a clear error at call time. This avoids
    # breaking legitimate imports (e.g., numpy) while preventing destructive
    # operations.
    def _disabled(*args, **kwargs):
        raise PermissionError("disabled by reliability_guard")

    try:
        os.kill = _disabled
    except Exception:
        pass
    for name in (
        "system",
        "remove",
        "removedirs",
        "rmdir",
    # do not disable fchdir/getcwd/chdir as imports (e.g. numpy) may call them
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
    "lchmod",
    "lchown",
    ):
        try:
            setattr(os, name, _disabled)
        except Exception:
            pass

    try:
        import shutil

        shutil.rmtree = _disabled
        shutil.move = _disabled
        shutil.chown = _disabled
    except Exception:
        pass

    # Do not disable subprocess.Popen globally: some legitimate imports (e.g. numpy
    # or other packages) may call subprocess during import. Disabling Popen can
    # therefore break imports. Leave subprocess alone.

    # Do not set sys.modules[...] = None here; that can break legitimate imports
    # like numpy. Only block known interactive/debug modules if desired.


__all__ = ["reliability_guard"]
