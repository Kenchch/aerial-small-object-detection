"""Logging shared by every script in src/.

stdout keeps what a script reports: the result tables and the `metrics -> path`
lines, so `python src/evaluate.py ... > results.txt` still captures the results
and nothing else. Everything about the run itself goes through the "aerial"
logger, and each script logs as aerial.<script>:

    console (stderr)   INFO and above, without tracebacks (Python prints those)
    runs/aerial.log    every command, appended, with tracebacks

Each command logs the line it was started with, how it ended (done, stopped
with a message, interrupted, or failed) and how long it took. Records carry
their level, so `grep -E "WARNING|ERROR" runs/aerial.log` finds every warning
and every run that did not finish. Python warnings (torch, numpy) are logged
too. Times are UTC, like the timestamps in the reports.

Nothing is configured on import, so pytest keeps its own handlers; only a
script's main() calls command(). AERIAL_LOG_FILE moves the run log, which the
container needs: its working directory is not writable for `docker run --user`.
A run log that cannot be opened is said once and the run carries on.

Standard library only, like the top of every script here: `--help` has to work
before the requirements are installed.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ENV = "AERIAL_LOG_FILE"

log = logging.getLogger("aerial")
WARNINGS = logging.getLogger("py.warnings")  # where logging.captureWarnings sends them
LINE, DATE = "%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%dT%H:%M:%SZ"


def formatter(cls: type[logging.Formatter] = logging.Formatter) -> logging.Formatter:
    """LINE with UTC times."""
    f = cls(LINE, DATE)
    f.converter = time.gmtime
    return f


FORMAT = formatter()


def log_path() -> Path:
    """Where the run log goes: $AERIAL_LOG_FILE, or runs/aerial.log."""
    return Path(os.environ.get(LOG_ENV) or ROOT / "runs" / "aerial.log")


def capture_warnings() -> None:
    """Send Python warnings through logging, so they reach the run log.

    Off, then on: logging only installs its hook when it has none saved, and a
    test framework that restores warnings.showwarning leaves a stale one.
    """
    logging.captureWarnings(False)
    logging.captureWarnings(True)


class _Brief(logging.Formatter):
    """Console lines without the traceback: Python prints it when the error is re-raised."""

    def format(self, record: logging.LogRecord) -> str:
        saved = record.exc_info, record.exc_text
        record.exc_info = record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text = saved


class _Console(logging.StreamHandler):
    """Writes to whatever sys.stderr is now, so it survives pytest swapping it per test."""

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def console() -> None:
    """Show the scripts' records on the console, once however often it is called."""
    if not any(isinstance(h, _Console) for h in log.handlers):
        h = _Console()
        h.setFormatter(formatter(_Brief))
        log.addHandler(h)
        WARNINGS.addHandler(h)


@contextmanager
def to_file(path: str | Path):
    """Append every record (and Python's warnings) to path for the length of the block.

    A path that cannot be opened is reported and the block runs without it: a
    read-only mount must not stop a benchmark that has somewhere else to write
    its results.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(path, encoding="utf-8")
    except OSError as e:
        log.warning(
            f"no run log: cannot write {path} ({type(e).__name__}: {e}); "
            f"set {LOG_ENV} to a writable file"
        )
        yield
        return
    h.setFormatter(FORMAT)
    loggers = [log, WARNINGS]
    for lg in loggers:
        lg.addHandler(h)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeHandler(h)
        h.close()


@contextmanager
def step(name: str):
    """Log a step's start, end and time, or the error it stopped with."""
    t0 = time.perf_counter()
    log.info(f"{name}: start")
    try:
        yield
    except Exception as e:
        log.error(
            f"{name}: failed after {time.perf_counter() - t0:,.1f} s: "
            f"{type(e).__name__}: {e}"
        )
        raise
    log.info(f"{name}: done in {time.perf_counter() - t0:,.1f} s")


@contextmanager
def command(name: str):
    """Run a script's main() with logging set up and appended to the run log.

    The console shows INFO and above; the run log gets the same lines plus a
    crash's traceback. A SystemExit with an error status or a message is logged
    as the run stopping rather than crashing; with status 0 it is done. Ctrl-C
    on a training run is logged as interrupted, with how far it got in time.
    """
    console()
    capture_warnings()
    log.setLevel(logging.INFO)
    with to_file(log_path()):
        t0 = time.perf_counter()

        def took() -> str:
            return f"{time.perf_counter() - t0:,.1f} s"

        log.info(f"{name}: start: {' '.join([Path(sys.argv[0]).name, *sys.argv[1:]])}")
        try:
            yield
        except SystemExit as e:
            if e.code in (None, 0):
                log.info(f"{name}: done in {took()}")
            else:
                detail = f"exit status {e.code}" if isinstance(e.code, int) else e.code
                log.error(f"{name}: stopped after {took()}: {detail}")
            raise
        except KeyboardInterrupt:
            log.error(f"{name}: interrupted after {took()}")
            raise
        except Exception as e:
            # The error goes on the line itself: the console drops the traceback.
            what = f"{type(e).__name__}: {e}"
            log.exception(f"{name}: failed after {took()}: {what}")
            raise
        log.info(f"{name}: done in {took()}")


def rel(path: str | Path) -> str:
    """path relative to the repository when inside it, so logs don't carry this machine's layout."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
