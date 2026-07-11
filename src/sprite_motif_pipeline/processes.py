from __future__ import annotations

import os
import subprocess


def terminate_process_tree(process: subprocess.Popen[bytes], *, timeout_s: int = 15) -> None:
    """Terminate only the process tree represented by a pipeline-owned handle."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
            creationflags=creationflags,
        )
        if result.returncode != 0 and process.poll() is None:
            process.terminate()
    else:
        process.terminate()

    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)
