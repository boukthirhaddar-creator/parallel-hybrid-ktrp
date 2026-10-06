# The protocols

This document states, in English, the protocols of the four studies reported in
the paper, together with those of the confirmatory replication and of the parallel
measurements. Each was written down and frozen before the corresponding runs were
executed, and this file is the record deposited alongside the logs it governs.

We describe the protocols as **pre-specified** rather than pre-registered. They
were fixed before the runs and are deposited here with the logs, but they were not
lodged with a public registry ahead of data collection. The one component for which
an independent trace exists is the transfer policy of Study 3, which the script
writes to a timestamped file at the moment it is frozen, before any test run
begins; that file is `logs/transfer/transfer_policy.json` and its `date_gel`
field records the instant.

---

## Common ground

**Configurations compared.** All share the same kernel, the same deterministic
starting solution, the same wall-clock budget, the same number of agents and the
same seeds. They differ only in how the tabu tenure is set at each epoch.

| Name in the paper | Name in the code | Tenure at each epoch |
|---|---|---|
| FIX-3 | `IND-3` | fixed at the smallest value of the grid |
| FIX-CAL | `IND-OPT` | fixed, calibrated on the validation suite |
| RANDOM | `HASARD` | drawn uniformly from the grid |
| PRIVATE | `PRIVEE` | sampled from a policy private to each agent |
| SHARED | `PARTAGEE` | sampled from a single policy shared by all agents |

**Tenure grid.** Five values, at 3 %, 6 %, 12 %, 20 % and 30 % of the number of
customers, rounded to the nearest integer and bounded below by 2.

**Budget.** Three agents, 1.2 seconds per customer, bounded below by 60 seconds and
above by 600, divided into 30 epochs.

**Seeds.** Ten distinct seeds per instance, the same ten for every configuration so
that runs are paired: 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061.

**Validation instances.** A-n45-k6, A-n60-k9 and A-n80-k10, used only to calibrate
FIX-CAL. They appear in no result table and are disjoint from every test instance.
The script verifies this disjointness at run time and refuses to proceed otherwise.

**Reported quantities.** The latency obtained, the instant at which the best
solution was obtained, and the quality reached at 10 %, 25 %, 50 % and 100 % of the
limit. Every returned solution is checked for feasibility and its objective is
recomputed independently before it enters any statistic.

**Statistics.** Comparisons are paired by instance and seed. The paired quantity is
the relative gap of a run to the best value observed on its instance. Tests are
two-sided Wilcoxon signed-rank tests. Effect sizes are Hodges-Lehmann estimates of
the median paired difference with distribution-free 95 % confidence intervals.

---

## Studies 1 and 2: the main campaign

**Question.** Does learning which tenure to play improve the search, and does
pooling the experience of the agents improve it further?

**Instances.** Those of the suite with at least 49 customers, a threshold fixed in
advance. Eight circuit instances, twenty-two path instances. Smaller instances are
excluded because they are solved to the same value by every configuration and
therefore carry no discriminating information.

**Calibration of FIX-CAL.** The fraction minimising the mean relative gap over the
five candidate fractions on the validation instances, with ten seeds, ties broken by
the shorter mean time to best. The selected fraction is written to a file before the
test runs begin.

**Decision rule, fixed before any test run.**

1. **Learning** is established if PRIVATE and SHARED both outperform FIX-3 and
   RANDOM significantly, and are not significantly outperformed by FIX-CAL.
2. **Cooperation** is established if SHARED outperforms PRIVATE significantly.

The seven comparisons form one family and the 5 % level is corrected by the method
of Holm. No other formulation of the claim is retained from these data.

**Declared as exploratory.** The rule does not compare RANDOM with the two fixed
settings, because RANDOM was introduced as a control condition rather than as a
candidate method. Since the results make it the configuration of interest, that
comparison is reported as well, labelled exploratory, on the same paired data and
with no correction for the additional inspection.

**Sensitivity analysis.** The rule operates on runs. Because the conclusions concern
instances, an analysis in which each instance contributes the mean of its ten runs is
also reported. It is a robustness check, not a revision of the rule, which remains
the primary analysis.

**Outcome.** Learning not established, cooperation not established, on both
variants.

---

## Study 3: cross-instance transfer

**Question.** Does a tenure window policy learned across instances outperform random
variation on instances never seen?

**Instance signature.** The mean route length `L = n / K`. Three classes, with
thresholds fixed in advance: short for `L < 6`, medium for `6 <= L < 10`, long for
`L >= 10`.

**Tenure windows.** `W0` is the full grid and coincides with RANDOM. `W1`, `W2` and
`W3` are the three contiguous triples {3, 6, 12} %, {6, 12, 20} % and
{12, 20, 30} %. Within a window the tenure is drawn uniformly at each epoch.

**Phase A, training.** The four windows on seventeen training instances, with three
seeds. The training instances are those of the path campaign, less the five on which
every run of that campaign returned the same value.

**Phase B, learning.** Each class receives the window of lowest mean relative gap on
the training instances of that class. `W0` is retained unless another window improves
upon it by at least 0.02 percentage points. The resulting policy is written to a
timestamped file and never recomputed.

**Stopping rule, fixed in advance.** If `W0` is retained for all three classes, the
hypothesis is not supported and the test phase does not take place.

**Phase C, test.** The frozen policy against RANDOM on eleven instances of the
X-series, between 100 and 147 customers, with ten new seeds: 3001, 3011, 3019, 3023,
3037, 3041, 3049, 3061, 3067, 3079. These instances were used neither for design nor
for training.

**Phase D, decision.** A single two-sided Wilcoxon signed-rank test at the 5 % level,
requiring no correction. The transfer is established if TRANSFER outperforms RANDOM.

**Outcome.** Not established. The frozen policy performs significantly worse than
random variation.

---

## Study 4: learning across a stream of repeated solves, and the transfer of priors

The full protocol is `PROTOCOL_STUDY4.md`, frozen before any run of the study.
This section is its faithful summary.

**Question.** Does the learning work once it operates where observations
accumulate, namely a stream of repeated solves on one family of instances, and
does knowledge transferred as a soft, full-support prior that keeps adapting
improve upon random variation?

**Setting.** The eleven X-series instances of Study 3, in a new operational
role: the deployment family of the stream. Budget 0.6 seconds per customer,
bounded by 60 and 90 seconds, thirty epochs, three agents, two solves running
concurrently on six worker processes, on a machine distinct from that of
Studies 1 to 3 and recorded automatically. The comparison is internal to the
study.

**Candidate mixtures.** Six distributions over the tenure grid: the uniform `U`,
and `G3`, `G6`, `G12`, `G20`, `G30`, each boosting one grid tenure to
probability 1/3 while the other four keep 1/6. No candidate restricts the range.

**Learner.** A Thompson sampler over the six candidates, Gaussian posteriors of
the mean reward, common standard deviation 0.003, no forgetting. The reward of a
solve is minus its final latency over the deterministic construction latency of
its instance, divided by the pilot mean of that ratio, so rewards are
comparable across instances.

**Arms.** A plays `U` throughout and carries no learner. B, C and D carry the
sampler and differ only in their priors: B flat, C estimated on the legacy
corpus of the training family (the 204 runs of `logs/transfer/training.jsonl`
and the 170 campaign HASARD runs on the same instances) by a ridge regression of
the final gap on the mixture played, D estimated the same way on the pilot
campaign. The informative priors of C and D receive twelve pseudo-counts.

**Pilot campaign.** Four uniform solves per instance, seeds disjoint from every
other, providing the corpus of D and the per-instance normalisation of the
reward.

**Seeds.** Stream solve of arm m, instance i, round r: 4001 + 40 i + r, the
same for every arm. Pilot: 5001 + 40 i + p. Eight rounds per arm.

**Decision rules, fixed before any stream run.** Two-sided Wilcoxon
signed-rank tests at the 5 % level. Learning is established if B outperforms A
over rounds 6 to 8. Transfer within the family is established if D outperforms
B over rounds 1 to 3 and D outperforms A over all rounds. Transfer across
families is examined by C against B over rounds 1 to 3 and C against A over
rounds 6 to 8, with no direction fixed in advance.

**Executions.** The study was executed twice, and both are reported. The first,
on 2 October 2026, ran on a machine exposing two logical processors, where the six
worker processes of the study contended for them; its logs are in
`logs/study4_run1`. It was re-executed on 4 October 2026 on the workstation used
by Studies 1 to 3, with the protocol unchanged apart from the machine; its logs
are in `logs/study4_run2`, and `logs/study4` holds the copy the scripts read. The
re-execution was decided after the first execution had been analysed, and no record
of that decision predates it, so its independence from the first outcome cannot be
demonstrated; both executions are reported for that reason.
`PROTOCOL_STUDY4_AMENDMENT2.md`, written after both, records this chronology.

**Outcome.** Neither execution establishes learning, transfer within the family or
transfer across families. In the second, no comparison reaches the 5 % level, the
smallest adjusted p-value being 0.433. In the first, one comparison reaches the
nominal level, C against A over rounds 6 to 8 at p = 0.013, which does not survive
the Holm correction over the five comparisons (0.065) and was not confirmed by the
replication of that same execution.

## Study 4b: confirmatory replication of the transfer of the learned state

The full protocol is `PROTOCOL_STUDY4_AMENDMENT.md`, written after the primary
stream and before any replication run. Because the decision to run it was taken
after the primary stream, the replication is not part of the pre-specified family;
its rules were nevertheless fixed before its runs.

**Question.** Do the states learned during the primary stream transfer to
fresh solves?

**Arms.** A2, a fresh uniform control. B2, starting from the final posterior of
the primary arm B. C2, starting from the final posterior of the primary arm C.
Six rounds, fresh seeds 4501 + 40 i + r, same instances, same budget and
conditions.

**Decision rules.** Two-sided Wilcoxon signed-rank tests on all paired solves,
corrected for the two comparisons by the method of Holm.

**Outcome.** In neither execution is either comparison significant, so no transfer
to fresh solves is established. In the second execution, B2 against A2 ends at a
Hodges-Lehmann median of -0.005 percentage points, interval [-0.047, +0.036]; C2
against A2 at -0.019, interval [-0.069, +0.021]. In the first, the same two end at
+0.023, interval [-0.026, +0.122], and +0.028, interval [-0.021, +0.140].

The two executions differ on the upper end of the C2 interval, which the first
leaves above the deficit of Study 3 and the second does not. Since pairing the
arms by instance and by seed protects the comparison from contention, the two
executions estimate the same quantity and the paper pools them, the correlation
between the paired differences of the two executions being r = -0.18 over the 66
common pairs. Pooled over the 132 paired runs, C2 against A2 sits at +0.001,
interval [-0.030, +0.035], p = 0.938, and B2 against A2 at +0.006, interval
[-0.024, +0.045]. The pooling was chosen after both outcomes were known and is a
secondary analysis; the two executions reported separately remain the primary
result. On the median scale the pooled interval excludes the deficit of the frozen
transfer of Study 3, whose Hodges-Lehmann median was +0.138 with interval
[+0.087, +0.179]. On the scale of the mean it does not: the pooled mean is +0.041
with a bootstrap interval reaching +0.128, which contains the mean deficit of
+0.120.

---

## Parallel measurements

**Question.** How does the framework scale, and what does parallelism deliver to the
user?

Three quantities are measured separately, because they answer different questions.

**Throughput.** Neighbour evaluations per second as a function of the number of
processes, from 1 to 8, in two modes. The *kernel* mode runs fully independent
searches with no coordination and therefore measures the capacity of the machine.
The *framework* mode runs the complete architecture on the same processes. The
difference isolates the cost of coordination. Three repetitions of thirty seconds
each, on E76k10 and CMT11. From these are derived the speedup, the efficiency, the
experimental sequential fraction of Karp and Flatt, and the corresponding Amdahl
bound.

**Time to reach a quality target.** For 1 to 4 processes, on four instances, with ten
seeds and a limit of 240 seconds. Three targets fixed in advance, at 1 %, 0.5 % and
0.2 % above a stated reference whose provenance is recorded in the script. Median
times are computed by the Kaplan-Meier estimator, which treats runs stopped before
the target as censored rather than discarding them.

**Quality at fixed wall-clock time.** The gap to the reference at 15, 30, 60, 120 and
240 seconds, for each number of processes.

**Conditions.** The machine runs no other computation, is connected to mains power,
and has sleep disabled. The hardware configuration is recorded automatically at run
time into `logs/parallel/hardware.json`.

**Outcome.** Throughput scales linearly to eight processes with a coordination cost
below two percent, while the time-to-target speedup reaches 1.65 at four processes.
The principal benefit of additional agents is the reliability with which a target is
reached rather than the speed of reaching it.
