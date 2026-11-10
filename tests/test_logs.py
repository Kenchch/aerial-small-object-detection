"""What a script leaves in the run log, and that nothing is configured on import."""

import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import logs
import make_demo_gif
import train

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
UTC = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ  "


def read(path):
    return path.read_text(encoding="utf-8")


@pytest.fixture
def run_log(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "aerial.log"
    monkeypatch.setenv(logs.LOG_ENV, str(path))
    return path


def test_log_times_are_utc():
    record = logging.LogRecord("aerial", logging.INFO, __file__, 1, "x", None, None)
    record.created = 0.0  # 1970-01-01 00:00:00 UTC
    assert logs.FORMAT.format(record).startswith("1970-01-01T00:00:00Z  INFO     x")


def test_paths_in_logs_are_relative_to_the_repository():
    assert logs.rel(logs.ROOT / "reports" / "tracking.json") == "reports/tracking.json"
    outside = logs.ROOT.parent / "elsewhere" / "x"
    assert logs.rel(outside) == str(outside)


def test_the_run_log_defaults_to_runs_and_can_be_moved(monkeypatch, tmp_path):
    monkeypatch.delenv(logs.LOG_ENV)
    assert logs.log_path() == logs.ROOT / "runs" / "aerial.log"
    monkeypatch.setenv(logs.LOG_ENV, str(tmp_path / "x.log"))
    assert logs.log_path() == tmp_path / "x.log"


def test_console_is_added_once():
    logs.console()
    logs.console()
    assert sum(isinstance(h, logs._Console) for h in logs.log.handlers) == 1


def test_importing_the_scripts_configures_nothing():
    code = (
        "import logging, sys, warnings\n"
        "before = warnings.showwarning\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import logs, train, evaluate, benchmark, track, make_demo_clip, make_demo_gif\n"
        "package = logging.getLogger('aerial')\n"
        "assert package.handlers == [] and package.level == logging.NOTSET\n"
        "assert logging.getLogger().handlers == []\n"
        "assert logging.getLogger('py.warnings').handlers == []\n"
        "assert warnings.showwarning is before\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_a_step_logs_its_start_end_and_time(caplog):
    caplog.set_level(logging.INFO, logger="aerial")
    with logs.step("label statistics (val)"):
        pass
    with pytest.raises(ZeroDivisionError), logs.step("export"):
        raise ZeroDivisionError("division by zero")
    got = [r.getMessage() for r in caplog.records]
    assert got[0] == "label statistics (val): start"
    assert re.fullmatch(r"label statistics \(val\): done in \d+\.\d s", got[1])
    assert re.fullmatch(
        r"export: failed after \d+\.\d s: ZeroDivisionError: division by zero", got[3]
    )


def test_a_command_logs_the_line_it_was_started_with_and_how_it_ended(
    run_log, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["src/benchmark.py", "--weights", "w.pt"])
    with logs.command("benchmark"):
        pass
    with pytest.raises(SystemExit), logs.command("train"):
        raise SystemExit("runs/n_1024 already exists.")
    with pytest.raises(SystemExit), logs.command("track"):
        raise SystemExit(2)
    with pytest.raises(KeyboardInterrupt), logs.command("train"):
        raise KeyboardInterrupt

    lines = read(run_log).splitlines()
    assert re.fullmatch(
        UTC + r"INFO     benchmark: start: benchmark.py --weights w.pt", lines[0]
    )
    assert re.fullmatch(UTC + r"INFO     benchmark: done in \d+\.\d s", lines[1])
    text = "\n".join(lines)
    assert re.search(
        r"ERROR    train: stopped after \d+\.\d s: runs/n_1024 already exists\.", text
    )
    assert re.search(r"ERROR    track: stopped after \d+\.\d s: exit status 2", text)
    assert re.search(r"ERROR    train: interrupted after \d+\.\d s", text)


def test_a_crash_is_in_the_run_log_with_its_traceback_and_on_the_console_without(
    run_log, capsys
):
    with pytest.raises(ZeroDivisionError), logs.command("evaluate"):
        raise ZeroDivisionError("division by zero")
    text = read(run_log)
    assert re.search(
        r"ERROR    evaluate: failed after \d+\.\d s: ZeroDivisionError: division by zero\n"
        r"Traceback \(most recent call last\):",
        text,
    )
    err = capsys.readouterr().err
    assert "evaluate: failed after" in err
    assert "Traceback" not in err, "Python prints it once when the error is re-raised"


def test_python_warnings_reach_the_run_log(run_log):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with logs.command("train"):
            warnings.warn(
                "torch says something is deprecated", UserWarning, stacklevel=1
            )
    assert re.search(
        UTC + r"WARNING  .*torch says something is deprecated", read(run_log)
    )


def test_a_run_log_that_cannot_be_written_does_not_stop_the_run(
    tmp_path, monkeypatch, capsys
):
    """`docker run --user` cannot write the working directory; the benchmark
    writes its report elsewhere and must still run."""
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    monkeypatch.setenv(logs.LOG_ENV, str(blocker / "aerial.log"))
    ran = False
    with logs.command("benchmark"):
        ran = True
    assert ran
    err = capsys.readouterr().err
    assert "WARNING  no run log: cannot write" in err
    assert "benchmark: done in" in err


def test_the_console_goes_to_stderr_so_redirected_results_stay_clean(run_log, capsys):
    with logs.command("evaluate"):
        logs.log.warning("no label files under x")
        print("metrics -> reports/evaluation.json")
    out, err = capsys.readouterr()
    assert out == "metrics -> reports/evaluation.json\n"
    assert "WARNING  no label files under x" in err


def test_a_refused_training_run_is_recorded(run_log, tmp_path, monkeypatch):
    """The refusal used to exist only on the terminal that ran it."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path)
    (tmp_path / "taken").mkdir()
    monkeypatch.setattr(sys, "argv", ["src/train.py", "--name", "taken"])
    with pytest.raises(SystemExit, match="already exists"):
        train.main()
    text = read(run_log)
    assert "INFO     train: start: train.py --name taken" in text
    assert re.search(r"ERROR    train: stopped after \d+\.\d s: .*already exists", text)


def test_bad_arguments_are_refused_before_a_run_is_logged(run_log, monkeypatch, capsys):
    """--help and argparse errors are not runs, and leave no run log behind."""
    monkeypatch.setattr(sys, "argv", ["src/make_demo_gif.py", "--fps", "0"])
    with pytest.raises(SystemExit):
        make_demo_gif.main()
    assert not run_log.exists()
    assert "start:" not in capsys.readouterr().err
