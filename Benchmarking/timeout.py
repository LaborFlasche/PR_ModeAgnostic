import signal
from contextlib import contextmanager


class BackendTimeout(Exception):
    """Raised when a backend call exceeds its allotted wall-clock budget."""


@contextmanager
def time_limit(seconds: float | None):
    """Abort the block with BackendTimeout after ``seconds`` (no-op if None/0).

    Uses SIGALRM: main-thread and Unix only (fine here — macOS + SLURM/Linux,
    no Windows in scope). This is best-effort, not a hard kill — a backend stuck
    in a single long C/Cython call only sees the signal once that call returns
    control to Python, so pure-native hangs may overrun the budget.
    """
    if not seconds:
        yield
        return

    def _on_alarm(signum, frame):
        raise BackendTimeout(f"exceeded {seconds}s timeout")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
