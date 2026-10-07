"""Catalog-density experiments with shared observations and paired comparisons.

Published comparisons require a held-out calibration artifact; see
:mod:`femmi.calibration` and docs/calibration.md. Historical defaults are retained
for API compatibility and must not be treated as newly calibrated settings.
"""
from .sampling import (SURVEY_NEFF, mesh_quality, n_gal_for_density,
                       density_for_n_gal, sample_catalog, _truth_at, _score)
from .runners import (KS_CALIBRATION, ks_params_for_density, c1_catalog_run,
                      argyris_catalog_run, hct_catalog_run, p3_catalog_run, ks_catalog_run)
from .sweep import density_sweep
from .stats import average_over_seeds, paired_comparison, paired_table, to_table
