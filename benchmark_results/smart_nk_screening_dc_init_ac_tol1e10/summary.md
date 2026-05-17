# Smart N-k Screening

N-1 is complete over all active selected elements. N-k is exact DC over the selected N-1 risk pool.

|case|k|N-1 total|N-1 failed|N-1 screen s|candidate pool|N-k combos|N-k failed|N-k screen s|exhaustive over pool|top score|AC converged|AC max mismatch|top contingency|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
|case6470rte|2|9005|2084|12.551|80|3160|3|11.963|True|inf|2|1.255795467614007e-11|line:7408+trafo:1046|
|case6470rte|3|9005|2084|12.551|80|20000|14|31.657|False|inf|1|1.4470258324905672e-11|trafo:134+line:7084+line:7204|
|case9241pegase|2|16049|1402|41.220|80|3160|1|32.681|True|inf|1|1.248467995651481e-11|line:2559+line:2571|
|case9241pegase|3|16049|1402|41.220|80|20000|7|76.417|False|inf|6|7.710165839114325e-12|trafo:2068+line:2559+line:2571|
