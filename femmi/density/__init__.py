"""Catalogue sampling, mesh diagnostics and paired statistics.

Use femmi.calibration for held-out experiments.
"""
from .sampling import (SURVEY_NEFF, mesh_quality, n_gal_for_density,
                       density_for_n_gal, sample_catalog, _truth_at, _score)
from .stats import average_over_seeds, paired_comparison, paired_table, to_table
