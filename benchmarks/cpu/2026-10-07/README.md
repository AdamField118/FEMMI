# CPU measurements, 7 October 2026

See `docs/cpu-performance.md` for methods, environments, timing scope and limitations.
`final-*` are the three independent timing processes per workload/backend;
`comparison-*` check each updated backend against the corresponding baseline.
`jit-*` are separate first-compilation/cache-load measurements.
Full NPZ outputs and cProfile data are in the evidence bundle supplied with
this patch, or can be regenerated with the documented drivers.
