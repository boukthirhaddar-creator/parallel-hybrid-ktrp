# Exact optima of the circuit variant

Computed by `ktrp_exact.c`, a dynamic programme over subsets, with distances equal
to the integer part of the Euclidean distance plus one, the objective of
Equation (2) of the paper, and K non-empty routes. Every solution was re-checked
independently by the `ktrp_worker.py` kernel, both for feasibility and for the
value of its objective. The optimal tours are given in `circuit_optimum_*.txt`.

Two of the four published reference values shown as not optimal rest on instances
reconstructed from coordinates, since the original files were unavailable; they are
marked as conditional below. The other two rest on the original instance files.

| Instance | Published reference | Exact optimum | Status of the reference |
|---|---|---|---|
| P16K5 | 785 | 778 | not optimal (+0.9 %) |
| P16K7 | 847 | 824 | not optimal (+2.8 %) |
| P19K2 | 1110 | 1110 | optimal |
| P19K3 | 933 | 933 | optimal |
| P20K2 | 1211 | 1211 | optimal |
| P20K3 | 1015 | 1015 | optimal |
| P21K3 | 1049 | 1049 | optimal |
| E22K4 | 1269 | 1269 | optimal |
| P22K7 | 1105 | 1066 | not optimal (+3.7 %), conditional on the reconstruction |
| P23K7 | 1098 | 1054 | not optimal (+4.2 %), conditional on the reconstruction |
| P16K8 | not published | 858 |  |
| P21K2 | not published | 1262 |  |
| P22K2 | not published | 1339 |  |
