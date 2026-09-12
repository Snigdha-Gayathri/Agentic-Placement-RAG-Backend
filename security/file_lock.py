"""Cross-process filesystem lock for shared resource synchronization."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CrossProcessLock:
    """Non-blocking cross-process file lock using atomic file creation.
    
    Compatible with Windows and POSIX without requiring external C extensions.
    Includes stale lock recovery based on PID liveness and expiration timeout.
    """

    def __init__(self, lock_path: str | Path, stale_timeout_seconds: float = 300.0) -> None:
        self.lock_path = Path(lock_path)
        self.stale_timeout_seconds = stale_timeout_seconds
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Attempt to acquire the lock atomically. Returns True if acquired, False if locked."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()

        # Check if an existing lock file is stale
        if self.lock_path.exists():
            try:
                mtime = self.lock_path.stat().st_mtime
                if now - mtime > self.stale_timeout_seconds:
                    logger.warning("Overriding stale cross-process lock at %s (age %.1fs)", self.lock_path, now - mtime)
                    self._force_remove()
                else:
                    # Check if locking process is dead
                    content = self.lock_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if content.isdigit():
                        pid = int(content)
                        if not self._is_pid_running(pid):
                            logger.warning("Overriding dead-process lock at %s (PID %d)", self.lock_path, pid)
                            self._force_remove()
                        else:
                            return False
                    else:
                        return False
            except Exception:
                return False

        try:
            # Atomic creation: fails with FileExistsError if another process created it concurrently
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            self._fd = os.open(str(self.lock_path), flags)
            pid_str = f"{os.getpid()}\n"
            os.write(self._fd, pid_str.encode("utf-8"))
            return True
        except FileExistsError:
            return False
        except Exception as exc:
            logger.error("Failed to acquire cross-process lock %s: %s", self.lock_path, exc)
            return False

    def release(self) -> None:
        """Release and clean up the lock file."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        self._force_remove()

    def _force_remove(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Could not remove lock file %s: %s", self.lock_path, exc)

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        """Check if process with given PID is currently active on the OS."""
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def __enter__(self) -> bool:
        if not self.acquire():
            raise RuntimeError("Cross-process lock is already held by another worker process.")
        return True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
