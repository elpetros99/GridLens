# GridLens tutorials

Run these from the repository root with the bundled environment:

```powershell
& 'C:\Users\elpetros\.conda\envs\jax\python.exe' examples\tutorial_batched_scenarios.py
& 'C:\Users\elpetros\.conda\envs\jax\python.exe' examples\tutorial_n1_security.py
& 'C:\Users\elpetros\.conda\envs\jax\python.exe' examples\tutorial_n2_tree.py
& 'C:\Users\elpetros\.conda\envs\jax\python.exe' examples\tutorial_exhaustive_n2_verification.py
& 'C:\Users\elpetros\.conda\envs\jax\python.exe' examples\tutorial_diagnostics.py
```

The scenario tutorial demonstrates the common `(scenario, bus)` batch interface on both CPU/SuperLU and JAX. The N-1 tutorial performs complete DC screening plus exhaustive selected AC validation. The N-2 tutorial demonstrates the parent N-1 warm-start tree; it is a triage example with an explicitly limited risk pool. The exhaustive N-2 tutorial enumerates and attempts every case14 line pair and prints the audit certificate. The diagnostics tutorial shows causal reports for missing references and zero-impedance data.

For a reproducible size sweep, run `scripts\benchmark_contingency_batching.py`. The JAX topology path uses the same aligned topology/scenario arrays on CPU or GPU; select a CUDA-enabled JAX installation and pass `--include-jax` to benchmark the accelerated backend.
