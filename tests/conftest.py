"""Optional-dependency guards for scientific test environments.

FEMMI_REQUIRE_OPTIONAL=1 requires every listed optional test dependency.
A comma-separated list requires selected extras; unset permits pytest skips.
"""

import importlib
import os

import pytest


# extra name in pyproject -> modules that must import when it is required.
# Only extras that some test actually skips on belong here. The `mesh` extra
# (triangle) is deliberately absent: nothing imports it, so requiring it would
# make CI depend on a package the suite never exercises.
OPTIONAL_MODULES = {
    "galsim": ["galsim"],     # test_truth, test_density, test_benchmark,
                              # test_lambda_selection, test_experiments
    "io": ["astropy"],        # test_catalog_pipeline (FITS catalogs)
    "neural": ["flax", "optax"],   # test_neural_prior
}


def _required():
    v = os.environ.get("FEMMI_REQUIRE_OPTIONAL", "").strip().lower()
    if v in ("", "0", "false", "no"):
        return set()
    if v in ("1", "true", "yes", "all"):
        return set(OPTIONAL_MODULES)
    unknown = {e for e in (s.strip() for s in v.split(",")) if e
               and e not in OPTIONAL_MODULES}
    if unknown:
        raise pytest.UsageError(
            f"FEMMI_REQUIRE_OPTIONAL names unknown extras {sorted(unknown)}; "
            f"known extras are {sorted(OPTIONAL_MODULES)}")
    return {s.strip() for s in v.split(",") if s.strip()}


def pytest_configure(config):
    """Fail the whole run up front if a required extra is missing.

    Checked at configure time rather than per test: the point is to catch a
    misconfigured CI job, and one clear error beats N identical ones. Reporting
    it as a collection-time skip would be exactly the failure mode being fixed.
    """
    missing = []
    for extra in sorted(_required()):
        for mod in OPTIONAL_MODULES[extra]:
            try:
                importlib.import_module(mod)
            except ImportError:
                missing.append(f"{mod} (extra '{extra}')")
    if missing:
        raise pytest.UsageError(
            "FEMMI_REQUIRE_OPTIONAL is set but these optional dependencies are "
            "missing, so the tests that need them would have SKIPPED silently: "
            + ", ".join(missing)
            + ". Install them (pip install -e '.[dev,neural,galsim,io]') or "
              "unset FEMMI_REQUIRE_OPTIONAL to allow skipping.")
