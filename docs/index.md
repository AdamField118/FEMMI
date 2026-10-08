# FEMMI

FEMMI maps weak-lensing shear catalogues to convergence fields with finite and
boundary elements. It offers three element families, a reusable quadratic
estimator, and a FITS workflow modeled on SMPy's survey interface.

Use the [quickstart](quickstart.md) for a complete synthetic example. For real
catalogues, read the [survey guide](survey-io.md) before choosing columns,
calibration, and shear signs. The [API reference](api.md) lists the callable
interfaces and output fields.

The [observation model](observation-model.md) explains the likelihood, boundary
assumption, and interpretation of B maps. The [research configuration](configuration.md)
exposes additional priors and approximate sampling methods.

To compare methods, follow the [benchmark protocol](smpy-benchmarks.md).
To measure cost, use the [profiling guide](production-performance.md).
Generated results belong in the chosen run directory; this documentation describes
how to produce and interpret them.
