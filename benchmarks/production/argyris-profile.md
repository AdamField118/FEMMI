# argyris cumulative profile

Profiled timings include instrumentation overhead.

```text
Wed Oct  7 23:32:18 2026    benchmarks/production/argyris-profile.prof

         3924280 function calls (3875128 primitive calls) in 3.097 seconds

   Ordered by: cumulative time
   List reduced from 9341 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      146    0.004    0.000    3.597    0.025 __init__.py:1(<module>)
   1730/1    0.040    0.000    3.104    3.104 {built-in method builtins.exec}
        1    0.001    0.001    3.104    3.104 <string>:1(<module>)
        1    0.001    0.001    3.102    3.102 profile_cpu.py:33(run)
  1502/60    0.006    0.000    1.626    0.027 <frozen importlib._bootstrap>:1349(_find_and_load)
  1480/52    0.005    0.000    1.625    0.031 <frozen importlib._bootstrap>:1304(_find_and_load_unlocked)
  3438/68    0.005    0.000    1.622    0.024 <frozen importlib._bootstrap>:480(_call_with_frames_removed)
  1313/54    0.003    0.000    1.622    0.030 <frozen importlib._bootstrap>:911(_load_unlocked)
  1185/54    0.002    0.000    1.621    0.030 <frozen importlib._bootstrap_external>:993(exec_module)
  926/197    0.001    0.000    1.229    0.006 {built-in method builtins.__import__}
7594/5643    0.006    0.000    1.123    0.000 <frozen importlib._bootstrap>:1390(_handle_fromlist)
        1    0.003    0.003    1.043    1.043 mapping.py:114(__init__)
        6    0.001    0.000    1.038    0.173 profile_cpu.py:24(timed)
        1    0.000    0.000    1.035    1.035 c1_coupling.py:139(__init__)
        1    0.009    0.009    0.945    0.945 c1_assembly.py:68(assemble_c1)
      458    0.004    0.000    0.723    0.002 elements.py:469(element)
      428    0.002    0.000    0.716    0.002 elements.py:183(__init__)
      428    0.022    0.000    0.698    0.002 elements.py:252(_build)
    11706    0.614    0.000    0.692    0.000 elements.py:87(_mono)
        2    0.000    0.000    0.513    0.256 mesh.py:1(<module>)
        1    0.000    0.000    0.509    0.509 operators.py:1(<module>)
4579/4480    0.039    0.000    0.372    0.000 {built-in method builtins.__build_class__}
     1185    0.006    0.000    0.355    0.000 <frozen importlib._bootstrap_external>:1066(get_code)
        1    0.000    0.000    0.354    0.354 pyplot.py:1(<module>)
     1179    0.137    0.000    0.266    0.000 <frozen importlib._bootstrap_external>:755(_compile_bytecode)


```
