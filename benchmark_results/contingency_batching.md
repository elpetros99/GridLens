# Contingency batching benchmark

CPU rows use SciPy SuperLU (`splu`) and explicit sparse Newton. `jacobian2` reuses a numerical Jacobian/factorization for two Newton steps. JAX rows run on the detected JAX device and are included only with `--include-jax`.

| case | buses | contingencies | mode | batch_size | build_s | solve_s | total_s | converged | batch_fallbacks | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|
| case14 | 14 | 20 | cpu_sequential_cold | 1 | 0.969194 | 0.16285 | 1.13204 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_sequential_warm | 1 | 1.19683 | 0.163442 | 1.36028 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu | 1 | 1.02964 | 0.172159 | 1.2018 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu_jacobian2 | 1 | 0.902866 | 0.15058 | 1.05345 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu | 8 | 1.17515 | 0.128624 | 1.30377 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu_jacobian2 | 8 | 1.03632 | 0.109127 | 1.14544 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu | 32 | 1.1591 | 0.136792 | 1.2959 | 19 | 0 | inf |
| case14 | 14 | 20 | cpu_batch_warm_superlu_jacobian2 | 32 | 0.995508 | 0.0920828 | 1.08759 | 19 | 0 | inf |
| case30 | 30 | 32 | cpu_sequential_cold | 1 | 1.72746 | 0.242203 | 1.96966 | 31 | 0 | inf |
| case30 | 30 | 32 | cpu_sequential_warm | 1 | 1.521 | 0.238354 | 1.75935 | 30 | 0 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu | 1 | 1.66065 | 0.231394 | 1.89204 | 30 | 0 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu_jacobian2 | 1 | 1.5669 | 0.192177 | 1.75908 | 30 | 0 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu | 8 | 2.28247 | 0.218107 | 2.50058 | 30 | 8 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu_jacobian2 | 8 | 2.23961 | 0.145432 | 2.38504 | 30 | 8 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu | 32 | 3.33742 | 0.210047 | 3.54747 | 30 | 32 | inf |
| case30 | 30 | 32 | cpu_batch_warm_superlu_jacobian2 | 32 | 3.34027 | 0.161202 | 3.50148 | 30 | 32 | inf |
| case57 | 57 | 32 | cpu_sequential_cold | 1 | 1.52617 | 0.385244 | 1.91142 | 32 | 0 | 3.69496e-14 |
| case57 | 57 | 32 | cpu_sequential_warm | 1 | 1.77102 | 0.26202 | 2.03304 | 32 | 0 | 5.66063e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu | 1 | 1.81864 | 0.305114 | 2.12375 | 32 | 0 | 5.66063e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu_jacobian2 | 1 | 1.58395 | 0.229231 | 1.81318 | 32 | 0 | 9.01934e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu | 8 | 1.77657 | 0.238248 | 2.01481 | 32 | 0 | 5.66063e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu_jacobian2 | 8 | 1.74609 | 0.174139 | 1.92023 | 32 | 0 | 9.01934e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu | 32 | 1.4902 | 0.237847 | 1.72805 | 32 | 0 | 5.66063e-11 |
| case57 | 57 | 32 | cpu_batch_warm_superlu_jacobian2 | 32 | 1.83911 | 0.203107 | 2.04222 | 32 | 0 | 9.01934e-11 |
| case118 | 118 | 32 | cpu_sequential_cold | 1 | 1.75593 | 0.331807 | 2.08774 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_sequential_warm | 1 | 1.74257 | 0.262306 | 2.00487 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu | 1 | 1.59484 | 0.257944 | 1.85278 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu_jacobian2 | 1 | 1.71585 | 0.243997 | 1.95984 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu | 8 | 1.93209 | 0.288218 | 2.2203 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu_jacobian2 | 8 | 1.59041 | 0.175112 | 1.76552 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu | 32 | 2.00763 | 0.245393 | 2.25303 | 30 | 0 | inf |
| case118 | 118 | 32 | cpu_batch_warm_superlu_jacobian2 | 32 | 1.72554 | 0.185387 | 1.91092 | 30 | 0 | inf |
| case300 | 300 | 32 | cpu_sequential_cold | 1 | 1.61744 | 0.938585 | 2.55603 | 27 | 0 | inf |
| case300 | 300 | 32 | cpu_sequential_warm | 1 | 2.0867 | 0.921206 | 3.0079 | 26 | 0 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu | 1 | 1.88067 | 0.880167 | 2.76084 | 26 | 0 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu_jacobian2 | 1 | 1.6166 | 0.517 | 2.1336 | 26 | 0 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu | 8 | 2.28777 | 0.872678 | 3.16045 | 26 | 8 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu_jacobian2 | 8 | 2.22248 | 0.490556 | 2.71303 | 26 | 8 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu | 32 | 3.78605 | 0.760364 | 4.54641 | 26 | 32 | inf |
| case300 | 300 | 32 | cpu_batch_warm_superlu_jacobian2 | 32 | 3.86518 | 0.57197 | 4.43715 | 26 | 32 | inf |
