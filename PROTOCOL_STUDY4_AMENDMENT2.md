# Amendment 2 to the protocol of Study 4: re-execution on the main workstation

This amendment was written after both executions of Study 4 had been completed.
It is therefore not a pre-specification, and we do not present it as one. It
records, after the fact, the reason for which the study was re-executed and the
decision to report both executions.

## Reason for the re-execution

The first execution of Study 4, carried out on 2 October 2026, ran on a machine
that no other study of this paper uses: an Intel Xeon processor under Linux
exposing two logical processors to the Python runtime. The study launches two
solves concurrently over six worker processes, so six processes contended for
two logical processors throughout. Two consequences follow. The absolute
wall-clock quantities of that execution are not comparable with those of
Studies 1 to 3, and a reader of a parallel computing journal is entitled to ask
why one study of the paper was run under contention while the others were not.

We therefore re-execute the study on the workstation used by Studies 1 to 3,
where six worker processes occupy six of sixteen logical processors and the
searches do not contend for them.

The re-execution was decided after the first execution had been completed and
analysed. We cannot demonstrate that the decision was independent of what that
first execution found, since no record of it predates the re-execution, which
began at 09:53 on 4 October 2026 according to the hardware file deposited with
its logs. That is precisely why both executions are reported: where the
independence of a decision cannot be established, the remedy is to withhold
nothing rather than to assert an independence that no document supports.

## What changes and what does not

The protocol of `PROTOCOL_STUDY4.md` and the replication protocol of
`PROTOCOL_STUDY4_AMENDMENT.md` are applied unchanged: the same eleven instances,
the same arms, the same number of rounds, the same seed schemes, the same budget
of 0.6 seconds per customer bounded by 60 and 90 seconds, the same thirty epochs
and three agents, the same decision rules. Only the machine changes.

The priors of arms C and D are re-estimated by the same procedure on the same
corpora, since the pilot phase is part of the study and is re-run with it. The
re-estimated priors are frozen into a new timestamped file before the stream
begins.

## Reporting

Both executions are reported. The logs of the first are deposited in
`logs/study4_run1` and those of the second in `logs/study4_run2`, each with its
own hardware record and its own timestamped priors file. The paper reports the
outcome of both, and states explicitly how the divergence between them is read.

The reason for reporting both is stated plainly. An experiment executed twice and
reported once, with the published execution chosen after both outcomes are known,
is the practice that pre-specification exists to prevent. Since the decision to
re-execute was taken after the first outcome was known, the only defensible course
is to publish both and let the reader weigh them.

## Outcomes

Both executions were completed. Neither establishes any benefit of learning or of
transfer, and on that point they agree. They differ on one quantity, the arm of
the replication that carries the inherited prior, where the first execution
leaves a deficit of the magnitude measured in Study 3 inside its confidence
interval and the second excludes it. Section 5.5 of the paper reports both and
states the reading adopted.
