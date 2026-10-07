# CPU acceleration regression fixture

`bem_fa079a9.npz` was generated with the unmodified FEMMI commit
`fa079a9468a19cb1713f814304270723f2025e44` (the correctness baseline).
It contains straight-boundary single/double layer matrices on the five-vertex
irregular polygon stored in `points`, with degree 3/5, quadrature 7/25,
and coordinate scales 0.3/2.0. Keys `dD_qQ_sS_V/K` use `bem_hp`; degree 3
also has `P3V/P3K` from the independent historical P3 implementation.

Generate with `python tests/fixtures_cpu/generate_bem_reference.py --source-root
/path/to/unmodified/fa079a9 --output tests/fixtures_cpu/bem_fa079a9.npz` (one line).
The generator checks the supplied git revision and refuses modified source.
