# Scalability Report

Environment:

- Python: `C:\Users\elpetros\.conda\envs\jax\python.exe`
- JAX backend observed: `cpu`
- JAX devices observed: `[CpuDevice(id=0)]`
- These timings are CPU-only; the code is JAX/GPU-ready but this environment does not expose a GPU to JAX.

## Base-Case Scaling

Warm solve excludes JAX compilation. First solve includes compilation.

| case | buses | state | preconditioner | first solve s | warm solve s | converged | mismatch |
|---|---:|---:|---|---:|---:|---|---:|
| `case118` | 118 | 181 | dense fast-decoupled | 2.854 | 0.038 | yes | 2.51e-14 |
| `case300` | 300 | 530 | dense fast-decoupled | 3.792 | 0.145 | yes | 4.25e-13 |
| `case1354pegase` | 1354 | 2447 | dense fast-decoupled | 3.657 | 0.788 | yes | 6.80e-12 |
| `case2848rte` | 2848 | 5325 | dense fast-decoupled | 9.035 | 6.023 | yes | 9.49e-12 |
| `case2869pegase` | 2869 | 5227 | dense fast-decoupled | 4.807 | 2.545 | yes | 3.79e-12 |
| `case3120sp` | 3120 | 5991 | dense fast-decoupled | 5.502 | 3.253 | yes | 1.18e-11 |

Large cases with memory-safe diagonal preconditioning built and ran, but did not converge under the tested CPU budget:

| case | buses | state | warm solve s | converged | mismatch |
|---|---:|---:|---:|---|---:|
| `case6470rte` | 6470 | 12486 | 74.300 | no | 2.60 |
| `case6515rte` | 6515 | 12536 | 72.196 | no | 2.86 |
| `case9241pegase` | 9241 | 17036 | 104.732 | no | 0.522 |

Interpretation: the current dense fast-decoupled preconditioner is robust through 3k-bus cases but is not viable for 6k-9k bus systems. The unpreconditioned/diagonal JAX GMRES path is memory-safe but not sufficiently strong yet.

## Batch Scaling On `case118`

| batch | warm solve s | s / power flow | convergence | max mismatch |
|---:|---:|---:|---:|---:|
| 1 | 0.030 | 0.0302 | 1.0 | 2.90e-14 |
| 2 | 0.056 | 0.0282 | 1.0 | 6.34e-14 |
| 4 | 0.099 | 0.0248 | 1.0 | 6.34e-14 |
| 8 | 0.185 | 0.0231 | 1.0 | 1.02e-13 |
| 16 | 0.338 | 0.0211 | 1.0 | 1.14e-13 |
| 32 | 0.729 | 0.0228 | 1.0 | 1.14e-13 |
| 64 | 1.472 | 0.0230 | 1.0 | 1.14e-13 |
| 128 | 3.112 | 0.0243 | 1.0 | 1.26e-13 |
| 256 | 5.567 | 0.0217 | 1.0 | 1.41e-13 |

On CPU, batching mainly amortizes Python overhead but does not provide the massive throughput expected on GPU. GPU scaling should be retested after installing CUDA-enabled JAX.

## Contingency Scaling On `case118`

Full AC topology-batched subset:

| k | contingencies | runtime s | convergence |
|---:|---:|---:|---:|
| 1 | 32 | 17.411 | 0.9375 |

Hierarchical screening plus AC validation:

| k | screened | selected for AC | DC screening s | AC validation s | total s | AC convergence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 173 | 32 | 0.457 | 10.749 | 11.207 | 1.0 |
| 2 | 1024 | 32 | 2.282 | 9.047 | 11.329 | 1.0 |

The hierarchical path follows the same structure as fast security-analysis tools such as LightSim2Grid: preprocess/register contingencies, screen cheaply, then retrieve/validate detailed AC results only for selected contingencies.

## Implemented Speed Improvements

- Removed full `Ybus.toarray()` densification during pandapower conversion.
- Added `BatchedPowerFlowSolver` to reuse compiled JAX functions.
- Vectorized random load `Sbus` batch generation instead of deep-copying and reconverting a pandapower net per scenario.
- Added `auto`, dense fast-decoupled, diagonal fast-decoupled, and no-preconditioner modes.
- Added lazy/chunkable N-k enumeration.
- Added DC contingency screening and hierarchical AC validation.

## Next Required Work

- Add a sparse preconditioner/factorization path for large systems, e.g. ILU on CPU and cuDSS/custom-call/Pallas-backed sparse solve on GPU.
- Add chunked topology batching for full N-k so incompatible/islanded contingencies are reported as failures without aborting a whole batch.
- Add GPU benchmarks after CUDA-enabled `jaxlib` is available.

## Update: CPU Sparse Backend And Whole-Workflow Timings

Added a separate CPU sparse Newton backend with explicit sparse Jacobian assembly and SciPy sparse direct/ILU linear solves. This is intentionally separate from the JAX backend so the GPU-ready JAX path does not contain host callbacks.

CPU sparse direct base-case results:

| case | buses | state | runtime s | converged | mismatch |
|---|---:|---:|---:|---|---:|
| `case118` | 118 | 181 | 0.010 | yes | 1.09e-10 |
| `case300` | 300 | 530 | 0.019 | yes | 1.89e-12 |
| `case1354pegase` | 1354 | 2447 | 0.052 | yes | 2.56e-12 |
| `case1888rte` | 1888 | 3503 | 0.223 | no | 4.75e8 |
| `case2848rte` | 2848 | 5325 | 0.153 | yes | 7.40e-9 |
| `case2869pegase` | 2869 | 5227 | 0.083 | yes | 4.48e-11 |
| `case3120sp` | 3120 | 5991 | 0.134 | yes | 8.06e-12 |
| `case6470rte` | 6470 | 12486 | 0.994 | no | 8.60e12 |
| `case6515rte` | 6515 | 12536 | 0.839 | no | 9.03e12 |
| `case9241pegase` | 9241 | 17036 | 0.344 | yes | 5.55e-12 |

Whole contingency workflow timing on `case118`:

| workflow | k | screened | selected AC | DC screen s | AC build s | first/compile AC s | warm AC s | diagnostics s | total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full AC subset | 1 | 0 | 32 | 0.000 | 5.676 | 3.714 | 1.167 | 5.569 | 16.203 |
| hierarchical | 1 | 173 | 32 | 0.386 | 2.284 | 4.535 | 1.029 | 3.192 | 11.514 |
| hierarchical | 2 | 1024 | 32 | 2.441 | 2.146 | 3.885 | 0.810 | 3.140 | 12.452 |

Current compatibility policy:

- Controllers and tap-control-style actions: supported by freezing through pandapower control preprocessing on a copied net.
- Static line temperature: supported when pandapower-required line temperature columns are present.
- Voltage-dependent loads: default structured rejection; optional freeze policy materializes a static solved load point.
- Dynamic TDPF and distributed slack: structured rejection, because they require changing the JAX residual/state equations.

Resolved convergence gap:

- The previous RTE failures were caused by missing pandapower initialization parity. Pandapower's `init="auto"` uses a DC power-flow angle initialization for these grids; the adapter was only reading the raw `_pd2ppc` bus VM/VA values.
- The adapter now defaults to `init="auto"`, applies pandapower's internal DC initialization before extracting `Ybus`, `Sbus`, `V0`, PV/PQ/ref sets, and uses pandapower's `_get_pf_variables_from_ppci` / `_get_Sbus` helpers for extraction.
- With CPU sparse direct Newton at tolerance `1e-10`, all tested cases from `case118` through `case9241pegase` converge. The largest final mismatch in the run below is `1.73e-11`.

Updated CPU sparse direct base-case results:

| case | buses | state | build s | solve s | converged | mismatch |
|---|---:|---:|---:|---:|---|---:|
| `case118` | 118 | 181 | 0.054 | 0.019 | yes | 7.39e-14 |
| `case300` | 300 | 530 | 0.065 | 0.023 | yes | 8.24e-13 |
| `case1354pegase` | 1354 | 2447 | 0.068 | 0.038 | yes | 3.80e-12 |
| `case1888rte` | 1888 | 3503 | 0.062 | 0.129 | yes | 8.18e-12 |
| `case2848rte` | 2848 | 5325 | 0.071 | 0.137 | yes | 1.45e-11 |
| `case2869pegase` | 2869 | 5227 | 0.059 | 0.124 | yes | 6.10e-12 |
| `case3120sp` | 3120 | 5991 | 0.070 | 0.122 | yes | 8.48e-12 |
| `case6470rte` | 6470 | 12486 | 0.098 | 0.362 | yes | 1.73e-11 |
| `case6495rte` | 6495 | 12491 | 0.110 | 0.271 | yes | 1.00e-11 |
| `case6515rte` | 6515 | 12536 | 0.118 | 0.283 | yes | 1.05e-11 |
| `case9241pegase` | 9241 | 17036 | 0.143 | 0.403 | yes | 5.20e-12 |

## Update: Complete Large-Grid N-1 DC Screening

Added an LODF/PTDF screening path for complete N-1 screening. This is much more realistic for large security analysis than solving one DC power flow per outage.

Complete all active `line + trafo` N-1 DC screening with top-20 AC validation:

| case | N-1 candidates | DC failed/islanded | DC screen s | max base loading % | top DC score | AC validation s | AC converged top20 | AC failed top20 | max AC mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `case6470rte` | 9005 | 2084 | 12.816 | 618.038 | 7067.924 | 14.260 | 15 | 5 | 2.09e-11 |
| `case6495rte` | 9019 | 2117 | 12.238 | 601.422 | 3430.743 | 12.552 | 15 | 5 | 1.51e-11 |
| `case6515rte` | 9037 | 2133 | 12.718 | 632.261 | 6128.207 | 10.732 | 17 | 3 | 1.62e-11 |
| `case9241pegase` | 16049 | 1402 | 42.265 | 1946.867 | 3169.100 | 13.644 | 18 | 2 | 1.21e-11 |

Interpretation:

- The DC screening itself is complete for the listed N-1 candidate sets.
- Top-20 AC validation now converges for most selected contingencies on all four large cases; converged cases have residuals around `1e-11`.
- Remaining AC failures are singular/islanded or otherwise non-solvable selected contingencies and must be classified as insecure for guarantee purposes, not as acceptable residual mismatches.

## Update: Smart N-k Screening And Guarantee Semantics

Guarantee semantics:

- A security guarantee is only valid for the exact model, contingency universe, constraints, and solver used.
- Exhaustive DC N-k screening can certify security only in the DC approximation and only if every enumerated N-k combination is evaluated and all islanded/singular cases are treated as insecure.
- AC N-k security cannot be guaranteed from a ranked subset. It requires exhaustive AC validation of every N-k contingency in the defined universe, with convergence and limit checks.
- If any N-1 contingency is insecure, the network is already not N-k secure for any `k >= 1`.

Implemented a smart N-k LODF screen:

- complete N-1 over all active `line + trafo` outages
- select top N-1 risk pool
- exact DC multi-outage evaluation over that pool using `post_flow = base_flow + LODF[:, K] @ solve(I - offdiag(LODF[K, K]), base_flow[K])`
- singular multi-outage systems are classified as insecure

Smart N-k run with AC validation on the top 10 screened N-k cases:

| case | k | N-1 total | N-1 failed | N-1 screen s | candidate pool | N-k combos | N-k failed | N-k screen s | exhaustive over pool | AC converged top10 | AC max mismatch | top contingency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `case6470rte` | 2 | 9005 | 2084 | 12.551 | 80 | 3160 | 3 | 11.963 | true | 2 | 1.26e-11 | `line:7408+trafo:1046` |
| `case6470rte` | 3 | 9005 | 2084 | 12.551 | 80 | 20000 | 14 | 31.657 | false | 1 | 1.45e-11 | `trafo:134+line:7084+line:7204` |
| `case9241pegase` | 2 | 16049 | 1402 | 41.220 | 80 | 3160 | 1 | 32.681 | true | 1 | 1.25e-11 | `line:2559+line:2571` |
| `case9241pegase` | 3 | 16049 | 1402 | 41.220 | 80 | 20000 | 7 | 76.417 | false | 6 | 7.71e-12 | `trafo:2068+line:2559+line:2571` |

Conclusion: these tested large networks cannot be certified secure under this DC N-k screen because complete N-1 already contains many failed/islanded contingencies and the selected N-2/N-3 pools contain singular/insecure combinations.
