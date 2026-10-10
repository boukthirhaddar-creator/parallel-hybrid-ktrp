# Replication package

Code, instances, protocols and complete run logs for the paper

> **Parallel Multi-Agent Tabu Search for the K-Traveling Repairman Problem:
> A Pre-specified Assessment of Learning, Cooperation and Transfer**
> S. Jmal, B. Haddar, H. Chabchoub. *The Journal of Supercomputing*.

The paper studies a parallel framework in which several tabu search agents share a
collective memory and adapt their tabu tenure, and it measures what each component
contributes under protocols fixed before the runs. Varying the tabu tenure improves
solution quality substantially. Learning which tenure to play within a run does
not improve upon varying it at random: the reward observable within a run ranks
the tenures nearly in reverse order of their contribution to the final objective,
and thirty decisions per agent are too few to overcome the bias. A policy frozen
on a restricted range and transferred across families performs significantly
worse than random variation. A fourth pre-specified study moves the learning to
a stream of repeated solves on one family of instances and transfers knowledge as
a soft, full-support prior that keeps adapting. That study was executed twice, on
two machines, and both executions are deposited and reported. Neither establishes
a benefit of learning or of transfer. The contribution is a bounded absence rather
than a finding: pooled over both executions, the intervals place any median effect
of accumulated or inherited knowledge within five hundredths of a percentage point
of the control. On that median scale the pooled interval also excludes the deficit
of the frozen transfer, whose median was 0.138 percentage points. On the scale of
the mean it does not: the pooled bootstrap interval reaches 0.128 and so contains
the mean deficit of 0.120.

This repository contains everything needed to verify those claims without running
the solver, and to reproduce them from scratch if desired.

---

## Quick verification, under one minute

```bash
python reproduce_tables.py --check
```

Nothing needs to be installed first. The reconstruction of the tables uses only
the standard library, and every statistic it prints, the Wilcoxon signed-rank
$p$-values included, is computed in the deposited sources. The output therefore
does not depend on which version of any third-party package happens to be
present, and a reader who installs nothing obtains the same numbers as one who
installs everything in `requirements.txt`.

The check recomputes twenty-seven headline values from the deposited logs, covering all
four studies and both executions of the fourth, and compares them with the values
printed in the paper. It exits with status 0 when all agree. It opens by stating
which null distribution each published comparison was tested under, the exact
enumeration of the signed ranks or the normal approximation that shared ranks
require, so that the rule declared in Section 4.7 of the paper can be read off
the run itself.

To print the tables of the paper from the logs:

```bash
python reproduce_tables.py             # every table derived from the logs
python reproduce_tables.py 5 14 C1     # selected tables, by their label in the paper
```

Tables are named as the paper names them: a number for the body, a letter and a
number for the appendices. Tables 1, 2, A1 and A2 are not derived from the run
logs; the first two position the work in the literature, and the last two are
produced by `build_instances.py`, which verifies the distance convention and the
objective.

Two of the figures are drawn from the same logs:

```bash
python reproduce_tables.py --figures
```

This writes `fig_instance_level_circuit.pdf`, the per-instance difference between
RANDOM and FIX-3 ordered by mean route length, which is Figure 3 of the paper, and
`fig_stream_curves.pdf`, the mean relative gap by round for the four arms of
Study 4, which is Figure 4. The remaining figures are drawings of the architecture
and of the neighbourhood moves, and a distribution of the time to reach a target.

The output of the latter command, as produced on our machine, is stored in
`results/tables_from_logs.txt` for comparison.

---

## Layout

```
.
├── ktrp_worker.py          tabu search kernel, executed in each worker process
├── ktrp_coordinator.py     epoch coordinator, collective memory, tenure policies
├── build_instances.py      instance reconstruction and verification
├── selfcheck.py            self-checks on the kernel, run before any campaign
├── run_campaign.py         Studies 1 and 2, the main campaign
├── analyse_campaign.py     decision rule and statistics for the main campaign
├── run_transfer_study.py   Study 3, cross-instance transfer
├── run_stream_study.py     Study 4, learning across a stream of repeated solves
├── analyse_stream_study.py analysis of Study 4
├── run_replication.py      Study 4b, confirmatory replication on fresh seeds
├── analyse_replication.py  analysis of Study 4b
├── run_parallel_study.py   parallel measurements
├── reproduce_tables.py     regenerates from the logs every table derived from them,
│                          and, with --figures, the two figures drawn from them
├── instances_*.txt         instance lists used by the study scripts
├── bks_circuit.json        reference values, circuit variant, with provenance
├── bks_path.json           reference values, path variant, with provenance
├── cvrplib/                source coordinates, 52 files
├── instances/
│   ├── circuit/            23 circuit instances plus 3 validation instances
│   └── path/               36 path instances, 3 validation, 11 transfer test
├── exact/
│   ├── ktrp_exact.c        exact solver, dynamic programming over subsets
│   ├── circuit_optimum_*.txt   optimal value and tours, 13 instances
│   └── CIRCUIT_OPTIMA.md   summary table
├── logs/
│   ├── campaign/           1,500 runs of Studies 1 and 2, plus calibration
│   ├── transfer/           424 runs of Study 3, plus the frozen policy
│   ├── study4_run1/        first execution of Study 4, with its hardware record
│   ├── study4_run2/        second execution, identical to logs/study4
│   ├── study4/             the copy the analysis scripts read
│   └── parallel/           throughput, time-to-target, recorded hardware
├── fig_instance_level_circuit.pdf   Figure 3, written by --figures
├── fig_stream_curves.pdf            Figure 4, written by --figures
├── results/                output of reproduce_tables.py
├── PROTOCOLS.md            the protocols, frozen before the corresponding runs
├── PROTOCOL_STUDY4.md      full protocol of Study 4
├── PROTOCOL_STUDY4_AMENDMENT.md    amendment covering the replication
├── PROTOCOL_STUDY4_AMENDMENT2.md   amendment covering the re-execution
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

The Python files sit at the repository root because that is the working directory
in which they were executed; they resolve their paths relative to their own
location, so moving them into a subdirectory would require editing them.

---

## Reproducing from scratch

The four studies are independent and can be run in any order. Each writes to a
log directory and resumes where it stopped if interrupted.

### Preliminaries

Running the campaigns, unlike reconstructing the tables, needs the two packages
listed in `requirements.txt`: `psutil`, for the hardware configuration that the
parallel study records, and `matplotlib`, for the figures.

```bash
pip install -r requirements.txt
python build_instances.py     # regenerate the instances and verify both objectives
python selfcheck.py          # self-checks on the kernel
```

`build_instances.py` rebuilds every instance from the coordinates in
`cvrplib/` and checks the two objective functions against published
reference values. All ten of its checks must print `OK` before anything else is
run.

### Study 1 and Study 2, the main campaign

Edit `TOPOLOGIE` at the top of `run_campaign.py`, then:

```bash
python run_campaign.py     # TOPOLOGIE = "circuit", about 11 hours
python analyse_campaign.py     # same value of TOPOLOGIE
python run_campaign.py     # TOPOLOGIE = "path", about 29 hours
python analyse_campaign.py
```

### Study 3, cross-instance transfer

```bash
python run_transfer_study.py    # about 15 hours
```

The script trains, freezes the policy into a timestamped file, then tests it. If
no tenure window beats random variation on the training instances by at least
0.02 percentage points, it stops before the test phase, as the protocol requires.

### Study 4, stream learning and the transfer of priors, and its replication

```bash
python run_stream_study.py    # pilot, then the four arms
python analyse_stream_study.py  # pre-specified analysis of the stream
python run_replication.py             # confirmatory replication on fresh seeds
python analyse_replication.py           # pre-specified analysis of the replication
```

The first script runs the pilot campaign, freezes the priors into a
timestamped file, then runs the four arms. The second analyses the stream under
the rules fixed in `PROTOCOL_STUDY4.md`. The third runs the replication of
`PROTOCOL_STUDY4_AMENDMENT.md` on fresh seeds, and the fourth analyses it under its
rules. All scripts resume where they stopped.

The two analysis scripts carry the statistic fixed in advance for the first
execution, the normal approximation with continuity correction, and they keep it
whichever logs they are pointed at. Tables 14 and 15 of the paper, which report
the second execution, follow instead the rule stated in Section 4.7 and
implemented in `reproduce_tables.py`. Run on the logs of the second execution,
the two conventions differ by less than one hundredth and agree on every
comparison, so the figures these scripts print there are close to the published
ones without being identical to them. Table C1, which reports the first
execution, is printed by `reproduce_tables.py` under the pre-specified statistic
that produced it, and matches the paper exactly. The quick verification above is
therefore the path to compare against the published tables figure for figure.

### Parallel measurements

Run these alone on the machine, with no other computation in progress:

```bash
python run_parallel_study.py       # about 11 hours
```

### Exact optima

```bash
gcc -O3 -o ktrp_exact exact/ktrp_exact.c
./ktrp_exact instances/circuit/KTRP_test_P16_5.txt circuit
```

The solver is exact up to about 22 customers. Beyond that its memory requirement,
which grows as 2^n times n, exceeds a standard workstation.

---

## Conventions that matter

Two conventions determine every gap reported in the paper, and both were
established empirically rather than assumed.

**Circuit variant.** Travel times are integers obtained by truncating the
Euclidean distance and adding one unit. This rule reproduces the original instance
files exactly, entry by entry; rounding to the nearest integer differs on 152 to
292 entries depending on the instance, and rounding upwards on 26 to 38.

**Path variant.** Travel times are Euclidean distances in double precision, and
the latency of a route is the sum of the arrival times at the customers, with
weights decreasing from m to 1 and no return arc. With this definition the kernel
reproduces exactly the reference values published by independent groups on P16k8,
P19k2, E22k4, E23k3, P20k2, P21k2, P22k2 and P23k8.

`build_instances.py` verifies both conventions.

---

## Naming

The configuration names used in the code are in French and map to the paper as
follows.

| Code | Paper | Description |
|---|---|---|
| `IND-3` | FIX-3 | default fixed tenure |
| `IND-OPT` | FIX-CAL | fixed tenure calibrated on the validation suite |
| `HASARD` | RANDOM | tenure drawn uniformly from the grid at each epoch |
| `PRIVEE` | PRIVATE | each agent learns its own policy |
| `PARTAGEE` | SHARED | all agents share one policy |
| `TRANSFERT:Wk` | TRANSFER | frozen window policy, Study 3 |
| `melange` | stream arms | tenure drawn from an explicit mixture, Study 4 |

Instance names follow the same mapping: `KTRP_test_E76_10` is E76k10 of the
circuit variant, `KTRP_path_CMT11` is CMT11 of the path variant.

File names, directories, comments and printed output are in English. The keys
inside the JSON logs, and the identifiers that mirror them in the code, are not:
they are the schema of data already recorded, and renaming them would mean
rewriting the deposited logs, that is altering the record rather than describing
it. The glossary below covers every key a reader will meet. Instance names are
left unchanged for the same reason, since they appear as values inside the logs.

`logs/study4_run1/console_output.txt` is in the same position. It is the console
trace recorded while the first execution ran, so it is in French and names the
script under the name it then carried, `analyse_apprentissage.py`, which is now
`analyse_stream_study.py`. It is left exactly as it was written: it is a record of
what happened, and editing it to match the current names would make it a
reconstruction rather than a trace.

| Key or identifier | Meaning |
|---|---|
| `latence` | latency, the objective value |
| `graine` | random seed |
| `methode`, `etiquette` | configuration label |
| `instance`, `n_clients`, `K` | instance name, customer count, number of repairmen |
| `duree_tabou`, `duree` | tabu tenure |
| `epoque`, `epoques` | epoch, epochs |
| `limite_s`, `limite_temps` | wall-clock limit, in seconds |
| `temps_meilleure_s` | time at which the best solution was obtained |
| `evaluations` | number of neighbour evaluations |
| `releves` | quality recorded at fixed instants |
| `valide` | feasibility flag, checked independently |
| `ameliorations` | dated improvements of the best solution |
| `choix_durees` | tenures actually played during the run |
| `debit`, `mode`, `noyau`, `cadre` | throughput, mode, kernel mode, framework mode |
| `reference` | reference value used for the gap |

---

## Hardware

All results in the paper were produced on a single machine, whose configuration
was recorded automatically at run time in `logs/parallel/hardware.json`: an AMD
EPYC 7443 processor of the Zen 3 generation, nominal clock 2.85 GHz, 64 GB of
memory, Windows 10 Professional, Python 3.11.5. The operating system reports a
dual-socket configuration; the execution environment made sixteen logical
processors available to the Python runtime, and it is that figure which bounds the
parallel experiments.

Run times given above are those observed on this machine.

---

## Scope and limitations

The conclusions concern tenure control for a tabu search kernel on the K-TRP, with
three agents and the budgets stated in the paper. They do not establish that
learned control is ineffective for other parameters, other metaheuristics or other
problems. The absence of a measurable difference between the learned
configurations and random variation is a failure to reject rather than a proof of
equality, and the confidence intervals reported in the paper bound what could have
escaped detection.

On the circuit suite, the advantage of tenure variation over the default setting
is significant at the level of runs but concentrated on the instances whose routes
are longest; it does not reach significance at the level of instances. On the path
suite it holds at both levels. The paper states this distinction wherever the
claim is made, and `reproduce_tables.py` prints both analyses.

---

## Availability of this repository

This copy is served from an anonymised mirror at
<https://anonymous.4open.science/r/parallel-hybrid-ktrp-CCDE/>, which remains
accessible until 30 October 2027. It will be replaced by a permanent public
repository with a citable identifier once the paper is accepted, and the present
section will then carry that identifier.

## Citation

See `CITATION.cff`. Please cite the paper rather than this repository alone.

## License

The code is released under the MIT License, see `LICENSE`. The benchmark
coordinates in `cvrplib/` come from the public library at
<https://vrp.galgos.inf.puc-rio.br/> and remain subject to its terms.
