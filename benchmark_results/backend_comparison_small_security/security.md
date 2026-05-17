# N-1 Security Backend Comparison

|backend|case|converged_count|error|failed_or_islanded|ls2g_amps_computation_time_s|ls2g_compute_wall_s|ls2g_flows_wall_s|ls2g_internal_total_time_s|ls2g_modif_ybus_time_s|ls2g_preprocessing_time_s|ls2g_solver_time_s|ls2g_total_wall_s|max_finite_mismatch|n_bus|n_contingencies|n_line|n_trafo|nb_solved|s_per_contingency|scope|solver_type|top_n|top_score|total_wall_s|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batched_powerflow_dc_lodf|case118|||8||||||||||118|186|173|13||0.0002636564515283211|complete_n1_dc_screening||20|4.775931248092701|0.04904009998426773|
|batched_powerflow_cpu_sparse|case118|20|||||||||||1.387633913796152e-13|118|20|173|13||0.011751925000862685|selected_top_n_ac_solve_only||||0.23503850001725368|
|lightsim2grid_run_contingency_ls2g|case118||UserWarning("lightsim2grid package not installed. Install lightsim2grid e.g. by running 'pip install lightsim2grid' in command prompt.")|||||||||||118|186|173|13|||complete_n1_ac_wall|||||
|lightsim2grid_cpp_direct|case118||||0.0016767|0.1672233999997843|0.001707900024484843|0.1671777|0.0024508999999999937|0.0012493|0.16219130000000004|0.16893130002426915||118|186|173|13|177|0.000908232795829404|complete_n1_ac_compute_only|SolverType.SparseLUSingleSlack||||
|pandapower_run_contingency|case118|||||||||||||118|186|173|13||0.04064681720437973|complete_n1_ac_wall||||7.560308000014629|
|batched_powerflow_dc_lodf|case300|||74||||||||||300|411|283|128||0.00013511727495773846|complete_n1_dc_screening||20|14.98989898989899|0.05553320000763051|
|batched_powerflow_cpu_sparse|case300|16|||||||||||269763.99570630933|300|20|283|128||0.029522149999684187|selected_top_n_ac_solve_only||||0.5904429999936838|
|lightsim2grid_run_contingency_ls2g|case300||UserWarning("lightsim2grid package not installed. Install lightsim2grid e.g. by running 'pip install lightsim2grid' in command prompt.")|||||||||||300|411|283|128|||complete_n1_ac_wall|||||
|lightsim2grid_cpp_direct|case300||||0.0050816|1.2046098000137135|0.005114800005685538|1.2045688|0.00665100000000001|0.0042656|1.1906738000000012|1.209724600019399||300|411|283|128|322|0.002943368856494888|complete_n1_ac_compute_only|SolverType.SparseLUSingleSlack||||
|pandapower_run_contingency|case300|||||||||||||300|411|283|128||0.05879989878347862|complete_n1_ac_wall||||24.166758400009712|
