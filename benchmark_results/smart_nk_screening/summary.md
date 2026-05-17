# Smart N-k Screening

N-1 is complete over all active line+trafo outages. N-2/N-3 is exact DC over the selected top-80 N-1 risk pool; N-3 is capped at 20,000 combinations.

|case|k|N-1 total|N-1 failed|N-1 screen s|candidate pool|N-k combos|N-k failed|N-k screen s|exhaustive over pool|top score|top contingency|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
|case9241pegase|2|16049|1402|29.881|80|3160|1|25.249|True|inf|line:2559+line:2571|
|case9241pegase|3|16049|1402|29.881|80|20000|7|42.402|False|inf|trafo:2068+line:2559+line:2571|
|case6470rte|2|9005|2084|7.386|80|3160|3|6.627|True|inf|line:7408+trafo:1046|
|case6470rte|3|9005|2084|7.386|80|20000|14|15.541|False|inf|trafo:134+line:7084+line:7204|
