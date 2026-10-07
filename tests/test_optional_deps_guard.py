"""
tests/test_optional_deps_guard.py
The optional-dependency guard in conftest.py.

This exists because CI was green while proving nothing about the headline
results: five modules are gated on GalSim with `pytest.importorskip`, CI never
installed GalSim, and all five skipped on every push. A skip and a pass look the
same in the summary line unless you read the counts.

The guard turns that into a hard failure when FEMMI_REQUIRE_OPTIONAL is set, so
these tests check the switch itself -- if a future change makes it permissive
again, the silent-skip failure mode comes straight back.

Run:
    python -m pytest tests/test_optional_deps_guard.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from conftest import OPTIONAL_MODULES, _required


@pytest.fixture
def env(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("FEMMI_REQUIRE_OPTIONAL", raising=False)
        else:
            monkeypatch.setenv("FEMMI_REQUIRE_OPTIONAL", value)
    return _set


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "  "])
def test_unset_or_falsey_requires_nothing(env, value):
    """The default must stay permissive -- a contributor without GalSim should
    still be able to run the suite."""
    env(value)
    assert _required() == set()


@pytest.mark.parametrize("value", ["1", "true", "yes", "all", "ALL", "True"])
def test_truthy_requires_every_extra(env, value):
    env(value)
    assert _required() == set(OPTIONAL_MODULES)


def test_a_subset_can_be_named(env):
    """CI may not always be able to build every extra, so naming a subset has to
    work -- otherwise the only options are all-or-nothing and it reverts to
    nothing."""
    env("galsim,neural")
    assert _required() == {"galsim", "neural"}
    env(" galsim , io ")
    assert _required() == {"galsim", "io"}


def test_an_unknown_extra_is_an_error_not_a_silent_no_op(env):
    """A typo'd extra name must not quietly require nothing -- that would
    recreate the exact failure this guard exists to prevent."""
    env("galsimm")
    with pytest.raises(pytest.UsageError, match="unknown extras"):
        _required()


def test_galsim_is_covered_because_the_headline_results_depend_on_it():
    """test_truth and test_density both importorskip galsim, and they guard the
    independent-truth machinery and the source-density claim respectively."""
    assert "galsim" in OPTIONAL_MODULES
    assert OPTIONAL_MODULES["galsim"] == ["galsim"]


def test_every_listed_extra_is_declared_in_pyproject():
    """The guard names extras by their pyproject key so the error message can
    tell you what to install. If they drift apart the message sends you to a
    pip command that does not work."""
    import pathlib
    import re
    text = pathlib.Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    declared = set(re.findall(r"^\s*([A-Za-z0-9_-]+)\s*=", block, re.M))
    assert set(OPTIONAL_MODULES) <= declared, (
        f"extras named in conftest but not declared in pyproject: "
        f"{set(OPTIONAL_MODULES) - declared}")


def test_the_guard_does_not_list_extras_nothing_imports():
    """`mesh` (triangle) is declared in pyproject but imported nowhere, so
    requiring it would make CI depend on a package the suite never exercises."""
    assert "mesh" not in OPTIONAL_MODULES


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
