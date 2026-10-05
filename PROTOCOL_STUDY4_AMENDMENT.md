# Amendment 1 to the protocol of Study 4: replication of the transfer of the
# learned state to fresh solves

This amendment was written after the completion of the primary stream of
Study 4 and before any run of the replication described here. Because the
decision to replicate was taken after observing the primary outcome, the
replication is not part of the pre-specified family of five tests; it is a
confirmatory experiment, labelled as such, whose outcome will be reported
whether it confirms the primary signal or not.

## Motivation

The primary stream produced one nominally significant result among its five
pre-specified comparisons: arm C, which entered the stream with the prior
inherited from the legacy corpus of the training family and kept adapting,
outperformed the uniform control A over the last three rounds, with a
Hodges-Lehmann median difference of -0.097 percentage points, twenty-three
wins and ten losses over thirty-three pairs, and nine instances out of
eleven with a negative mean difference, p = 0.013. The four other
comparisons did not reach the 5 % level. Under a Holm correction over the
five comparisons of the study the smallest adjusted p-value is 0.065, and
under a correction within question 3 alone, which comprises two tests, it
is 0.026. The signal therefore warrants an independent confirmation before
any claim is retained from it.

## Design

The replication, denoted Study 4b, applies the states learned during the
primary stream to fresh solves, and compares them with a fresh control.

**Arms.** Three arms.
- A2, the control, plays the uniform mixture at every run and carries no
  learner. Its runs are fresh: they use seeds that no earlier run of this
  study has used.
- B2 starts from the final posterior state of arm B of the primary stream,
  reconstructed from the deposited journal, and continues to adapt under the
  same rule as arm B. It carries the experience accumulated by the stream
  learner alone.
- C2 starts from the final posterior state of arm C of the primary stream,
  likewise reconstructed from the journal, and continues to adapt under the
  same rule. It carries the inherited prior together with the experience
  accumulated during the primary stream.

**Instances, budget, kernel.** Unchanged: the eleven instances of the
X-series, the budget of 0.6 seconds per customer bounded by 60 and 90
seconds, thirty epochs, three agents, two runs executing concurrently on
six worker processes, instance-major order, decisions taken in fixed pairs
of different arms before either outcome is observed.

**Rounds and seeds.** Six rounds per arm. The run of arm m on instance i at
round r uses the seed 4501 + 40 i + r, the same for every arm, so that the
arms remain paired. These seeds are disjoint from every seed used by the
pilot and the primary stream.

**Decision rules, fixed before any replication run.** Two comparisons,
two-sided Wilcoxon signed-rank tests on all paired runs, at the 5 % level,
corrected for the two comparisons by the method of Holm, with
Hodges-Lehmann estimates and distribution-free 95 % confidence intervals,
and the sensitivity analysis at the level of instances.

1. The state learned across the stream transfers to fresh solves if B2
   outperforms A2 significantly.
2. The state that combines the inherited prior with the stream experience
   transfers to fresh solves if C2 outperforms A2 significantly.

The replication confirms the primary signal if rule 2 is satisfied and rule
1 is not, since the primary signal concerned arm C and not arm B. If neither
rule is satisfied, the primary signal is not confirmed and the conclusion
of the study remains the one drawn from the primary stream alone. If both
rules are satisfied, the stream learner alone is credited with the
transfer. No other formulation of the claim is retained from these data.

## Outcomes

To be filled after the runs, from the logs only.
