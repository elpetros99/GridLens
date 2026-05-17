# Base AC Backend Comparison

|backend|build_s_excluded|case|converged|max_s|mean_s|median_s|min_s|mismatch|n_bus|n_line|n_state|n_trafo|scope|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batched_powerflow_cpu_sparse|0.04619609998189844|case118|True|0.011085799982538447|0.009892949994537048|0.009892949994537048|0.00870010000653565|7.385330560303211e-14|118|173|181|13|base_ac_solve_only|
|pandapower_runpp||case118|True|0.039100299996789545|0.03728745000262279|0.03728745000262279|0.035474600008456036|nan|118|173||13|base_ac_runpp_wall|
|pandapower_lightsim2grid_runpp||case118|True|0.02714429999468848|0.026090249986737035|0.026090249986737035|0.02503619997878559|nan|118|173||13|base_ac_runpp_wall|
