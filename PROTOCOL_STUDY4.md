# Protocol of Study 4: learning across a stream of repeated solves, and the transfer of priors

This protocol was written down and frozen before any run of Study 4 was
executed. The file that carries the frozen priors, `logs/study4/priors.json`,
is written with a timestamp by the pilot phase, before any stream run begins,
and is never recomputed. This document follows the conventions of
`PROTOCOLS.md`, of which it is the continuation.

---

## Scenario

The three deposited studies established that the tenure must vary, that
learning which tenure to play cannot operate within a single run for want of
observations, and that a policy frozen on a restricted tenure window and
transferred across families performs worse than random variation. Study 4
moves the learning to the level at which observations accumulate in
deployment: a solver that processes, repeatedly and indefinitely, instances
drawn from one and the same family.

Three questions, each with a decision rule fixed in advance.

1. **Learning.** Does a coordinator that accumulates experience across a
   stream of repeated solves learn a tenure mixture that outperforms random
   variation?
2. **Transfer within the family.** Does a prior estimated from a small pilot
   campaign on the deployment family accelerate this learning?
3. **Transfer across families.** Does a prior estimated on the legacy corpus
   of the training family, which Study 3 showed to be non-transferable when
   frozen, mislead the learner, and does continued adaptation correct it?

## Common ground

**Kernel, agents, epochs.** The kernel is unchanged. Three agents, thirty
epochs, the starting solution and the residual tabu lists carried across
epochs exactly as in Studies 1 to 3.

**Instances.** The eleven instances of the X-series of the path variant,
between 100 and 147 customers, used by Study 3 as pure test instances and
never used for any design decision of that study. Study 4 uses them in a new
operational role, stated here for the first time: they constitute the
deployment family of the stream scenario.

**Budget.** This study runs on a machine other than that of Studies 1 to 3,
recorded automatically in `logs/study4/hardware.json`. The wall-clock budget
is 0.6 seconds per customer, bounded below by 60 seconds and above by 90
seconds, divided into thirty epochs. All arms run under identical conditions,
two runs executing concurrently on six worker processes. The comparison is
internal to the study; the absolute quality is not comparable with that of
Studies 1 to 3 and no such comparison is made.

**Tenure candidates.** Six mixtures over the five-tenure grid of
Studies 1 to 3. The uniform mixture, denoted U, assigns probability 1/5 to
each tenure. The five boosted mixtures, denoted G3, G6, G12, G20 and G30,
assign weight 2 to one tenure of the grid and weight 1 to each of the four
others, then normalise by 6: the boosted tenure is played with probability
1/3, the others 1/6. Every candidate keeps the full support of the grid; no
candidate restricts the range, which is the design error diagnosed in Study 3.
At each epoch each agent draws its tenure independently from the mixture in
force.

**Learner.** A Thompson sampler over the six candidates, one per arm,
modelled by Gaussian posteriors of the mean reward with common known
standard deviation sigma = 0.003 and no forgetting. The reward of a run is
minus its final latency divided by the latency of the deterministic
construction of its instance, then divided by the mean of that ratio over
the pilot runs of the instance, so that rewards are comparable across
instances and centred near -1. The prior of each arm is stated below. After
each run, the posterior of the candidate played is updated with the reward
observed. Within each instance block the runs are dispatched in fixed pairs
of different arms, both mixtures being sampled before either outcome is
observed, so that the sequence of decisions is deterministic given the
sequence of outcomes.

## Arms

Four arms, each an independent stream of eight rounds over the eleven
instances, in the fixed alphabetical order of the instances, one run per
instance and per round. The run of arm m on instance i at round r always uses
the seed 4001 + 40 i + r, the same for every arm, so that arms are paired.

**A, RANDOM.** The uniform mixture U at every run. No learning. This is the
control of every comparison.

**B, LEARN.** The Thompson sampler starts from a flat prior: every candidate
with mean -1 and pseudo-count 1.

**C, LEARN with a cross-family prior.** The prior means come from the legacy
corpus of the training family of Study 3, namely the 204 logged runs of
`logs/transfer/training.jsonl` and the 170 logged runs of the campaign
on the training instances, all executed under the budget of Studies 1 to 3.
A ridge regression of the final relative gap on the mixture played, with
instance effects and ridge parameter 0.25, is fitted on that corpus; the
predicted advantage of each candidate over U, in percentage points of final
latency, is converted to the reward scale and subtracted from the common
mean -1. Every candidate receives the pseudo-count 12.

**D, LEARN with a same-family prior.** Identical to C except that the corpus
is the pilot campaign of this study, described below, so that the prior
comes from the deployment family itself, under the deployment budget.

## Pilot campaign

Before the streams, four uniform runs per instance, seeds 5001 + 40 i + p,
p from 0 to 3, with epoch detail logged, under the budget and the conditions
of the streams. The pilot serves three purposes: it provides the per-instance
normalisation of the reward, it provides the corpus from which the prior of
arm D is estimated, and its regression is reported as the measure of what a
small pilot campaign on the deployment family reveals. The pilot uses no run
of the streams and the streams use no run of the pilot.

## Decision rules, fixed before any stream run

All tests are two-sided Wilcoxon signed-rank tests on paired runs, the pair
being the instance and the round, at the 5 % level, with Hodges-Lehmann
estimates and distribution-free 95 % confidence intervals. A sensitivity
analysis at the level of instances, in which each instance contributes the
mean of its paired differences, accompanies every test.

1. **Learning is established** if the mean relative gap of B over rounds 6
   to 8 is lower than that of A over the same rounds, significantly.
2. **Transfer within the family is established** if the mean relative gap of
   D over rounds 1 to 3 is lower than that of B over the same rounds,
   significantly, and if D is lower than A over all rounds, significantly.
3. **Transfer across families** is examined by C against B over rounds 1 to
   3 and C against A over rounds 6 to 8. The direction is not fixed in
   advance: the prior may mislead, be neutral, or help, and the outcome is
   reported whatever it is.

The gap of a run is its final latency relative to the best final latency
observed on its instance among the runs of the four streams of this study.
No other formulation of the claims is retained from these data. Exploratory
analyses, when reported, are labelled as such.

## Descriptive material

The pilot regression, the learning curves of the four arms round by round,
the composition of the candidates played by round, and the trajectories of
the posteriors are reported. None of these enters a decision rule.

## Outcomes

To be filled after the runs, from the logs only.
