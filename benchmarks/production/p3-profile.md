# p3 cumulative profile

Profiled timings include instrumentation overhead.

```text
Wed Oct  7 23:32:50 2026    benchmarks/production/p3-profile.prof

         3834530 function calls (3766553 primitive calls) in 3.341 seconds

   Ordered by: cumulative time
   List reduced from 10261 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      146    0.004    0.000    3.545    0.024 __init__.py:1(<module>)
   1728/1    0.040    0.000    3.350    3.350 {built-in method builtins.exec}
        1    0.000    0.000    3.350    3.350 <string>:1(<module>)
        1    0.001    0.001    3.350    3.350 profile_cpu.py:33(run)
  1500/58    0.006    0.000    1.584    0.027 <frozen importlib._bootstrap>:1349(_find_and_load)
  1478/50    0.004    0.000    1.583    0.032 <frozen importlib._bootstrap>:1304(_find_and_load_unlocked)
  3433/64    0.005    0.000    1.581    0.025 <frozen importlib._bootstrap>:480(_call_with_frames_removed)
  1311/52    0.003    0.000    1.580    0.030 <frozen importlib._bootstrap>:911(_load_unlocked)
  1183/52    0.002    0.000    1.579    0.030 <frozen importlib._bootstrap_external>:993(exec_module)
        1    0.000    0.000    1.447    1.447 mapping.py:114(__init__)
        1    0.002    0.002    1.446    1.446 operators.py:518(build_operators_catalog)
        1    0.002    0.002    1.436    1.436 operators.py:250(_assemble_operators_from_mesh)
  561/398    0.001    0.000    1.396    0.004 traceback_util.py:190(reraise_with_filtered_traceback)
  926/197    0.001    0.000    1.210    0.006 {built-in method builtins.__import__}
8829/6878    0.006    0.000    1.169    0.000 <frozen importlib._bootstrap>:1390(_handle_fromlist)
        1    0.000    0.000    1.122    1.122 operators.py:43(_build_ref_hessians)
       10    0.000    0.000    1.112    0.111 api.py:746(jacfun)
 3362/522    0.015    0.000    1.087    0.002 core.py:686(bind)
   128/69    0.001    0.000    1.045    0.015 pjit.py:257(cache_miss)
 3582/521    0.005    0.000    1.042    0.002 core.py:735(bind_with_trace)
       11    0.000    0.000    0.993    0.090 api.py:834(jacfun)
   137/82    0.000    0.000    0.988    0.012 linear_util.py:210(call_wrapped)
    21/11    0.001    0.000    0.980    0.089 api.py:1189(vmap_f)
  570/527    0.002    0.000    0.976    0.002 profiler.py:417(wrapper)
    21/11    0.000    0.000    0.972    0.088 batching.py:323(_batch_outer)


```
