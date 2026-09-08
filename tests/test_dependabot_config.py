"""The ignore list has to name dependencies that exist.

A misspelled `dependency-name` in dependabot.yml is not an error: Dependabot
reads the file, matches nothing, and keeps opening the pull requests the entry
was written to stop. The config looks correct in review and does nothing, which
is how this repository ended up with four open bumps against a runtime that was
supposed to be pinned.

So: every ignored name must be a package this project actually pins, and every
pin must either be ignored or be one of the two the runtime does not depend on.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Pillow reads image headers and PyYAML parses dataset specs; neither can move
# a published mAP, latency or placement figure, so both stay upgradable.
NOT_MEASURED = {"Pillow", "PyYAML"}


def _pip_ignores() -> list[str]:
    config = yaml.safe_load(
        (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    )
    pip = [u for u in config["updates"] if u["package-ecosystem"] == "pip"]
    assert len(pip) == 1, "expected exactly one pip ecosystem entry"
    return [entry["dependency-name"] for entry in pip[0]["ignore"]]


def _pinned() -> list[str]:
    names = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[=<>!~]", line)[0].strip())
    return names


def test_every_ignored_name_is_actually_pinned():
    """A name that matches nothing silently does nothing."""
    unmatched = [name for name in _pip_ignores() if name not in _pinned()]
    assert not unmatched, (
        f"dependabot.yml ignores {unmatched}, which requirements.txt does not "
        f"pin; the entry matches no dependency and has no effect"
    )


def test_every_measured_pin_is_ignored():
    """A pin added later without an ignore reopens the decision by default."""
    missing = [
        p for p in _pinned() if p not in _pip_ignores() and p not in NOT_MEASURED
    ]
    assert not missing, (
        f"{missing} is pinned but not ignored: Dependabot will propose upgrading "
        f"it, and merging that would replace a measured version. Either add it to "
        f"the ignore list or add it to NOT_MEASURED with the reason it cannot "
        f"move a published number."
    )


def test_the_upgradable_pins_are_still_upgradable():
    """The exceptions are exceptions, not an empty set that drifted shut."""
    ignored = set(_pip_ignores())
    assert not (NOT_MEASURED & ignored), (
        f"{NOT_MEASURED & ignored} is listed as not affecting any published "
        f"number, yet its upgrades are ignored"
    )
