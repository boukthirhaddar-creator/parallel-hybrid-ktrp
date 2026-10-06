#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REGENERATE EVERY TABLE OF THE PAPER FROM THE RUN LOGS.

This script is the one referred to in the Data and Code availability statements of
the manuscript. It reads only the files under ``logs/`` and ``exact/`` and prints
the content of every numbered table of the paper, together with the figures quoted
in the running text. Nothing is recomputed by re-running the solver, so the script
completes in a few seconds and verifies that the published numbers follow from the
deposited data.

Usage, from the repository root::

    python reproduce_tables.py              # all tables
    python reproduce_tables.py 5 7 13       # selected tables only
    python reproduce_tables.py --check      # compare against the published values

The ``--check`` mode compares a set of values hard-coded from the published tables
against the values recomputed here, and exits with a non-zero status if any of them
disagree. It is the quickest way for a reader to confirm that the paper and the
deposit are consistent.

Only the standard library and SciPy are required.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Sequence, Tuple

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:                                    # pragma: no cover
    _scipy_wilcoxon = None

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")

METHODS = ["IND-3", "IND-OPT", "HASARD", "PRIVEE", "PARTAGEE"]
LABEL = {"IND-3": "FIX-3", "IND-OPT": "FIX-CAL", "HASARD": "RANDOM",
         "PRIVEE": "PRIVATE", "PARTAGEE": "SHARED"}
DECISION_RULE = [("PRIVEE", "IND-3"), ("PRIVEE", "HASARD"),
                 ("PARTAGEE", "IND-3"), ("PARTAGEE", "HASARD"),
                 ("IND-OPT", "PRIVEE"), ("IND-OPT", "PARTAGEE"),
                 ("PARTAGEE", "PRIVEE")]
EXPLORATORY = [("HASARD", "IND-3"), ("HASARD", "IND-OPT"),
               ("HASARD", "PRIVEE"), ("HASARD", "PARTAGEE")]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the non-zero differences."""
    d = [x for x in diffs if abs(x) > 1e-12]
    if not d:
        return 1.0
    if _scipy_wilcoxon is not None:
        return float(_scipy_wilcoxon(d).pvalue)
    n = len(d)                                       # normal approximation
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    corr = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        t = j - i + 1
        corr += t ** 3 - t
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - corr / 48)
    if sigma == 0:
        return 1.0
    return math.erfc(abs((w - mu) / sigma) / math.sqrt(2))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjustment."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    cur = 0.0
    for rank, i in enumerate(order):
        cur = max(cur, min(1.0, (m - rank) * pvals[i]))
        out[i] = cur
    return out


def hodges_lehmann(d: Sequence[float]) -> float:
    """Median of the Walsh averages."""
    n = len(d)
    walsh = [(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)]
    return st.median(walsh)


def hl_interval(d: Sequence[float]) -> Tuple[float, float]:
    """Distribution-free 95% interval for the Hodges-Lehmann estimate."""
    n = len(d)
    walsh = sorted((d[i] + d[j]) / 2 for i in range(n) for j in range(i, n))
    N = len(walsh)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    k = max(0, min(int(math.floor(mu - 1.959963985 * sigma)), N - 1))
    return walsh[k], walsh[N - 1 - k]


def kaplan_meier_median(obs: Sequence[Tuple[float, bool]]) -> float | None:
    """Median time from right-censored observations; None if not reached."""
    survival = 1.0
    for t in sorted({t for t, event in obs if event}):
        at_risk = sum(1 for u, _ in obs if u >= t)
        events = sum(1 for u, e in obs if e and u == t)
        if at_risk:
            survival *= 1 - events / at_risk
        if survival <= 0.5:
            return t
    return None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_campaign(variant: str) -> Tuple[List[dict], Dict[str, float]]:
    """Runs of one campaign, each annotated with its gap to the best value seen."""
    name = "runs_circuit.jsonl" if variant == "circuit" else "runs_path.jsonl"
    runs = [r for r in read_jsonl(os.path.join(LOGS, "campaign", name)) if r["valide"]]
    best: Dict[str, float] = {}
    for r in runs:
        best[r["instance"]] = min(best.get(r["instance"], math.inf), r["latence"])
    for r in runs:
        r["gap"] = (r["latence"] - best[r["instance"]]) / best[r["instance"]] * 100
    return runs, best


def paired(runs: Sequence[dict], a: str, b: str) -> List[float]:
    """Paired differences of relative gaps, by (instance, seed)."""
    idx = {(r["instance"], r["graine"], r["methode"]): r["gap"] for r in runs}
    keys = sorted({(i, s) for (i, s, m) in idx if m == a} &
                  {(i, s) for (i, s, m) in idx if m == b})
    return [idx[(i, s, a)] - idx[(i, s, b)] for i, s in keys]


def instance_means(runs: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[Tuple[str, str], List[float]] = {}
    for r in runs:
        acc.setdefault((r["instance"], r["methode"]), []).append(r["latence"])
    out: Dict[str, Dict[str, float]] = {}
    for (inst, meth), values in acc.items():
        out.setdefault(inst, {})[meth] = st.fmean(values)
    return out


def short(instance: str) -> str:
    base = instance.replace("KTRP_test_", "").replace("KTRP_path_", "")
    if "_" in base:
        head, k = base.rsplit("_", 1)
        return f"{head}k{k}"
    return base


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def header(number: int, title: str) -> None:
    print(f"\n{'=' * 78}\nTABLE {number}. {title}\n{'=' * 78}")


def table_3() -> Dict[str, int]:
    header(3, "Exact optima of the circuit variant")
    published = {k: v for k, v in json.load(
        open(os.path.join(ROOT, "bks_circuit.json"), encoding="utf-8")).items()
        if not k.startswith("_")}
    optima: Dict[str, int] = {}
    folder = os.path.join(ROOT, "exact")
    prefix = "circuit_optimum_"
    for fname in sorted(os.listdir(folder)):
        if not fname.startswith(prefix) or not fname.endswith(".txt"):
            continue
        text = open(os.path.join(folder, fname), encoding="utf-8").read().split("\n")
        name = fname[len(prefix):-len(".txt")].replace("_", "K").upper()
        optima[name] = int(float(text[0].split()[1]))
    if not optima:
        raise SystemExit(f"no file matching {prefix}*.txt found in {folder}")
    print(f"{'Instance':9s} {'Published':>10s} {'Optimum':>9s}  Status")
    for name in sorted(optima, key=lambda x: (len(x), x)):
        ref = published.get(name)
        if ref is None:
            status = "not published"
        elif abs(ref - optima[name]) < 0.5:
            status = "optimal"
        else:
            status = f"not optimal (+{(ref - optima[name]) / optima[name] * 100:.1f} %)"
        print(f"{name:9s} {str(ref) if ref else '-':>10s} {optima[name]:9d}  {status}")
    return optima


def table_summary(number: int, variant: str) -> None:
    label = "Circuit" if variant == "circuit" else "Path"
    header(number, f"{label} variant, mean relative gap at fractions of the limit")
    runs, best = load_campaign(variant)
    fractions = sorted({round(float(k) / r["limite_s"], 2)
                        for r in runs for k in r["releves"]})
    print(f"{'Config':9s} " + "".join(f"{int(f * 100):>8d}%" for f in fractions))
    for m in METHODS:
        cells = []
        for f in fractions:
            vals = [(v - best[r["instance"]]) / best[r["instance"]] * 100
                    for r in runs if r["methode"] == m
                    for k, v in r["releves"].items()
                    if abs(float(k) / r["limite_s"] - f) < 0.01]
            cells.append(st.fmean(vals) if vals else float("nan"))
        print(f"{LABEL[m]:9s} " + "".join(f"{c:9.3f}" for c in cells))


def table_rule(number: int, variant: str) -> None:
    label = "Circuit" if variant == "circuit" else "Path"
    header(number, f"{label} variant, pre-specified decision rule (Holm corrected)")
    runs, _ = load_campaign(variant)
    rows = [(a, b, paired(runs, a, b)) for a, b in DECISION_RULE]
    adjusted = holm([wilcoxon_p(d) for _, _, d in rows])
    print(f"{'Comparison':26s} {'W/T/L':>12s} {'Mean':>8s} {'HL [95% CI]':>26s} {'p (Holm)':>11s}")
    for (a, b, d), p in zip(rows, adjusted):
        win = sum(1 for x in d if x < -1e-12)
        loss = sum(1 for x in d if x > 1e-12)
        lo, hi = hl_interval(d)
        wtl = f"{win}/{len(d) - win - loss}/{loss}"
        print(f"{LABEL[a] + ' vs. ' + LABEL[b]:26s} {wtl:>12s} {st.fmean(d):+8.3f} "
              f"{hodges_lehmann(d):+8.3f} [{lo:+7.3f},{hi:+7.3f}] {p:11.3g}")


def table_exploratory(number: int, variant: str) -> None:
    label = "Circuit" if variant == "circuit" else "Path"
    header(number, f"{label} variant, exploratory comparisons (no correction)")
    runs, _ = load_campaign(variant)
    print(f"{'Comparison':26s} {'W/T/L':>12s} {'Mean':>8s} {'HL [95% CI]':>26s} {'p':>11s}")
    for a, b in EXPLORATORY:
        d = paired(runs, a, b)
        win = sum(1 for x in d if x < -1e-12)
        loss = sum(1 for x in d if x > 1e-12)
        lo, hi = hl_interval(d)
        wtl = f"{win}/{len(d) - win - loss}/{loss}"
        print(f"{LABEL[a] + ' vs. ' + LABEL[b]:26s} {wtl:>12s} {st.fmean(d):+8.3f} "
              f"{hodges_lehmann(d):+8.3f} [{lo:+7.3f},{hi:+7.3f}] {wilcoxon_p(d):11.3g}")


def table_7(verbose: bool = True) -> None:
    header(7, "Circuit variant, sensitivity analysis at the instance level")
    runs, _ = load_campaign("circuit")
    means = instance_means(runs)
    insts = sorted(means)
    print(f"{'Comparison':26s} {'Better/tied/worse':>18s} {'Non-tied':>9s} {'p':>9s}")
    for a, b in [("HASARD", "IND-3"), ("HASARD", "IND-OPT"),
                 ("PRIVEE", "IND-3"), ("PRIVEE", "IND-OPT"),
                 ("PARTAGEE", "IND-3"), ("PARTAGEE", "IND-OPT")]:
        d = [(means[i][a] - means[i][b]) / means[i][b] * 100 for i in insts]
        better = sum(1 for x in d if x < -1e-9)
        worse = sum(1 for x in d if x > 1e-9)
        nz = [x for x in d if abs(x) > 1e-9]
        btw = f"{better}/{len(d) - better - worse}/{worse}"
        print(f"{LABEL[a] + ' vs. ' + LABEL[b]:26s} {btw:>18s} {len(nz):9d} "
              f"{wilcoxon_p(nz):9.3f}")
    if verbose:
        print("\nPer-instance mean route length and RANDOM minus FIX-3 (Section 5.2):")
        info = {r["instance"]: (r["n_clients"], r["K"]) for r in runs}
        for i in sorted(insts, key=lambda x: info[x][0] / info[x][1]):
            n, k = info[i]
            delta = (means[i]["HASARD"] - means[i]["IND-3"]) / means[i]["IND-3"] * 100
            print(f"  {short(i):9s} L = {n / k:5.2f}   {delta:+7.3f} %")


def path_instance_level() -> None:
    # Reported in the text of Section 5.3 rather than as a numbered table.
    print(f"\n{'=' * 78}\nPath variant, instance-level sensitivity analysis "
          f"(Section 5.3 text)\n{'=' * 78}")
    runs, _ = load_campaign("path")
    means = instance_means(runs)
    insts = sorted(means)
    for a, b in [("HASARD", "IND-3"), ("HASARD", "IND-OPT")]:
        d = [(means[i][a] - means[i][b]) / means[i][b] * 100 for i in insts]
        better = sum(1 for x in d if x < -1e-9)
        worse = sum(1 for x in d if x > 1e-9)
        nz = [x for x in d if abs(x) > 1e-9]
        print(f"  {LABEL[a]} vs. {LABEL[b]}: {better} better / "
              f"{len(d) - better - worse} tied / {worse} worse, "
              f"p = {wilcoxon_p(nz):.2e}, median improvement {abs(st.median(nz)):.2f} %")


def table_11() -> None:
    header(11, "Transfer study, mean relative gap on the eleven unseen instances")
    runs = [r for r in read_jsonl(os.path.join(LOGS, "transfer", "test.jsonl"))
            if r["valide"]]
    best: Dict[str, float] = {}
    for r in runs:
        best[r["instance"]] = min(best.get(r["instance"], math.inf), r["latence"])
    insts = sorted({r["instance"] for r in runs})
    print(f"{'Instance':12s} {'Cust.':>6s} {'L':>7s} {'TRANSFER':>10s} {'RANDOM':>9s}")
    agg = {"TRANSFER": [], "RANDOM": []}
    for i in insts:
        rows = [r for r in runs if r["instance"] == i]
        n, k = rows[0]["n_clients"], rows[0]["K"]
        cells = {}
        for tag, prefix in (("TRANSFER", "TRANSFERT"), ("RANDOM", "HASARD")):
            vals = [(r["latence"] - best[i]) / best[i] * 100 for r in rows
                    if r["etiquette"].startswith(prefix)]
            cells[tag] = st.fmean(vals) if vals else float("nan")
            agg[tag].extend(vals)
        print(f"{short(i):12s} {n:6d} {n / k:7.2f} {cells['TRANSFER']:10.3f} "
              f"{cells['RANDOM']:9.3f}")
    print(f"{'All':12s} {'':6s} {'':7s} {st.fmean(agg['TRANSFER']):10.3f} "
          f"{st.fmean(agg['RANDOM']):9.3f}")
    verdict = json.load(open(os.path.join(LOGS, "transfer", "verdict.json"),
                             encoding="utf-8"))
    print(f"\n  Pre-specified test: {verdict['paires']} pairs, "
          f"{verdict['victoires']} wins / {verdict['defaites']} losses, "
          f"mean difference {verdict['ecart_moyen']:+.3f} pt, p = {verdict['p']:.3g}, "
          f"transfer established: {'yes' if verdict['etabli'] else 'no'}")


def table_16() -> None:
    header(16, "Path variant, largest instances")
    runs, _ = load_campaign("path")
    refs = {k: v for k, v in json.load(
        open(os.path.join(ROOT, "bks_path.json"), encoding="utf-8")).items()
        if not k.startswith("_")}
    means = instance_means(runs)
    print(f"{'Instance':9s} {'Cust.':>6s} {'Reference':>10s} {'Best':>10s} "
          f"{'Gap best':>9s} {'RANDOM mean gap':>16s}")
    for i in sorted(means, key=lambda x: -max(r["n_clients"] for r in runs
                                              if r["instance"] == x)):
        name = short(i)
        if name not in refs:
            continue
        rows = [r for r in runs if r["instance"] == i]
        if rows[0]["n_clients"] < 100:
            continue
        ref = refs[name]
        best = min(r["latence"] for r in rows)
        print(f"{name:9s} {rows[0]['n_clients']:6d} {ref:10.2f} {best:10.2f} "
              f"{(best - ref) / ref * 100:+8.2f}% "
              f"{(means[i]['HASARD'] - ref) / ref * 100:+15.2f}%")


def table_17() -> None:
    header(17, "Throughput as a function of the number of processes")
    runs = read_jsonl(os.path.join(LOGS, "parallel", "throughput.jsonl"))
    for inst in sorted({r["instance"] for r in runs}):
        print(f"\n  {short(inst)}")
        print(f"    {'P':>3s} {'Kernel':>12s} {'Framework':>12s} {'Speedup':>8s} "
              f"{'Effic.':>7s} {'Overhead':>9s} {'Memory':>8s}")
        base = None
        kf = []
        for P in sorted({r["P"] for r in runs if r["instance"] == inst}):
            sel = [r for r in runs if r["instance"] == inst and r["P"] == P]
            kernel = st.fmean([r["debit"] for r in sel if r["mode"] == "noyau"])
            frame = st.fmean([r["debit"] for r in sel if r["mode"] == "cadre"])
            mem = max(r["memoire_mo"] for r in sel if r["memoire_mo"] is not None)
            base = base or frame
            speed = frame / base
            if P > 1:
                kf.append((1 / speed - 1 / P) / (1 - 1 / P))
            print(f"    {P:3d} {kernel:12,.0f} {frame:12,.0f} {speed:8.2f} "
                  f"{speed / P:7.3f} {(1 - frame / kernel) * 100:8.2f}% {mem:7.1f}M")
        if kf:
            f = st.fmean(kf)
            bound = f"{1 / f:,.0f}" if f > 0 else "not defined (f <= 0)"
            print(f"    Karp-Flatt sequential fraction f = {f:.4f}; Amdahl bound {bound}")


def table_18() -> None:
    header(18, "Time required to reach a quality target")
    runs = [r for r in read_jsonl(os.path.join(LOGS, "parallel", "targets.jsonl"))
            if r["valide"]]
    limit = max(r["limite_s"] if "limite_s" in r else 240.0 for r in runs)

    def reach(r: dict, target: float) -> Tuple[float, bool]:
        if r["latence_depart"] <= target:
            return 0.0, True
        for t, lat in r["ameliorations"]:
            if lat <= target + 1e-9:
                return float(t), True
        return limit, False

    speedups: Dict[int, List[float]] = {2: [], 3: [], 4: []}
    for inst in sorted({r["instance"] for r in runs}):
        ref = next(r["reference"] for r in runs if r["instance"] == inst)
        print(f"\n  {short(inst)} (reference {ref})")
        for delta in (0.01, 0.005, 0.002):
            target = ref * (1 + delta)
            cells = []
            base = None
            for P in (1, 2, 3, 4):
                obs = [reach(r, target) for r in runs
                       if r["instance"] == inst and r["P"] == P]
                if not obs:
                    continue
                rate = sum(1 for _, e in obs if e) / len(obs)
                med = kaplan_meier_median(obs)
                if P == 1:
                    base = med
                acc = (base / med) if (base and med) else None
                if acc and P > 1:
                    speedups[P].append(acc)
                cells.append(f"P{P}: {rate * 100:3.0f}% " +
                             (f"{med:6.1f}s" if med else "    --") +
                             (f" x{acc:.2f}" if acc and P > 1 else ""))
            print(f"    +{delta * 100:.1f}% | " + " | ".join(cells))
    print("\n  Geometric mean of the time-to-target speedup:")
    for P in (2, 3, 4):
        v = speedups[P]
        if v:
            print(f"    P = {P}: x{math.exp(st.fmean([math.log(a) for a in v])):.2f} "
                  f"({len(v)} instance-target pairs)")


def table_19() -> None:
    header(19, "Mean gap to the reference at fixed wall-clock instants")
    runs = [r for r in read_jsonl(os.path.join(LOGS, "parallel", "targets.jsonl"))
            if r["valide"]]
    instants = sorted({int(float(k)) for r in runs for k in r["releves"]})
    print(f"{'Processes':10s} " + "".join(f"{s:>9d}s" for s in instants))
    for P in (1, 2, 3, 4):
        cells = []
        for s in instants:
            vals = [(r["releves"][str(s)] - r["reference"]) / r["reference"] * 100
                    for r in runs if r["P"] == P and str(s) in r["releves"]]
            cells.append(st.fmean(vals) if vals else float("nan"))
        print(f"P = {P:<6d} " + "".join(f"{c:+10.2f}" for c in cells))


def table_c1() -> None:
    """The first execution of Study 4, reported in Appendix C of the paper.

    This table was produced by the pre-specified analysis scripts rather than by
    the present one, and those scripts use the normal approximation with
    continuity correction, dropping tied pairs before estimating the median. The
    two conventions agree to within a thousandth on every comparison, but the
    table is regenerated here with the statistic that produced it, so that the
    printed figures match the paper exactly rather than approximately.
    """
    import analyse_stream_study as pre

    header("C1", "Study 4, first execution")
    stream, _ = load_stream("stream.jsonl", "study4_run1")
    repli, _ = load_stream("replication.jsonl", "study4_run1")
    rows = [("Stream", a, b, w, stream_paired(stream, a, b, r)) for a, b, r, w in (
        ("B", "A", STREAM_ROUNDS_LATE, "rounds 6-8"),
        ("D", "B", STREAM_ROUNDS_EARLY, "rounds 1-3"),
        ("D", "A", None, "all rounds"),
        ("C", "B", STREAM_ROUNDS_EARLY, "rounds 1-3"),
        ("C", "A", STREAM_ROUNDS_LATE, "rounds 6-8"))]
    repli_rows = [("Replication", a, "A2", "", stream_paired(repli, a, "A2"))
                  for a in ("B2", "C2")]

    def stats(d: Sequence[float]):
        p, hl = pre.wilcoxon(list(d), [0.0] * len(d))
        lo, hi = pre.intervalle_hl([x for x in d if abs(x) > 1e-12])
        return p, hl, lo, hi

    # Holm covers the five comparisons of the stream and, separately, the two of
    # the replication, exactly as the protocol fixes it.
    adjusted = holm([stats(d)[0] for *_, d in rows])
    repli_adjusted = holm([stats(d)[0] for *_, d in repli_rows])
    print(f"{'Phase':12s} {'Comparison':20s} {'Pairs':>6s} {'W/T/L':>10s} "
          f"{'Mean':>8s} {'HL [95% CI]':>26s} {'p':>8s} {'p (Holm)':>9s}")
    for (phase, a, b, window, d), adj in zip(rows + repli_rows,
                                             list(adjusted) + list(repli_adjusted)):
        win = sum(1 for x in d if x < -1e-12)
        loss = sum(1 for x in d if x > 1e-12)
        p, hl, lo, hi = stats(d)
        label = f"{a} vs {b}" + (f", {window}" if window else "")
        print(f"{phase:12s} {label:20s} {len(d):6d} "
              f"{f'{win}/{len(d) - win - loss}/{loss}':>10s} {st.fmean(d):+8.3f} "
              f"{hl:+8.3f} [{lo:+7.3f},{hi:+7.3f}] {p:8.3f} {adj:9.3f}")


def table_c2() -> None:
    header("C2", "Circuit variant, per-instance results")
    runs, _ = load_campaign("circuit")
    acc: Dict[Tuple[str, str], List[float]] = {}
    for r in runs:
        acc.setdefault((r["instance"], r["methode"]), []).append(r["latence"])
    print(f"{'Instance':9s} {'Stat':5s} " + "".join(f"{LABEL[m]:>11s}" for m in METHODS))
    for inst in sorted({i for i, _ in acc}):
        for stat in ("mean", "s.d.", "best"):
            cells = []
            for m in METHODS:
                v = acc[(inst, m)]
                cells.append(st.fmean(v) if stat == "mean" else
                             (st.pstdev(v) if stat == "s.d." else min(v)))
            print(f"{short(inst) if stat == 'mean' else '':9s} {stat:5s} " +
                  "".join(f"{c:11.2f}" for c in cells))


def table_c3() -> None:
    header("C3", "Path variant, per-instance summary")
    runs, _ = load_campaign("path")
    refs = {k: v for k, v in json.load(
        open(os.path.join(ROOT, "bks_path.json"), encoding="utf-8")).items()
        if not k.startswith("_")}
    means = instance_means(runs)
    info = {r["instance"]: (r["n_clients"], r["K"]) for r in runs}
    print(f"{'Instance':9s} {'Cust.':>6s} {'K':>3s} {'Reference':>10s} {'FIX-3':>10s} "
          f"{'FIX-CAL':>10s} {'RANDOM':>10s} {'Best':>10s}")
    for inst in sorted(means, key=lambda x: (info[x][0], info[x][1])):
        name = short(inst)
        n, k = info[inst]
        best = min(r["latence"] for r in runs if r["instance"] == inst)
        print(f"{name:9s} {n:6d} {k:3d} {refs.get(name, float('nan')):10.2f} "
              f"{means[inst]['IND-3']:10.2f} {means[inst]['IND-OPT']:10.2f} "
              f"{means[inst]['HASARD']:10.2f} {best:10.2f}")


# ---------------------------------------------------------------------------
# Study 4: learning across a stream of repeated solves, and transfer of priors
# ---------------------------------------------------------------------------
STREAM_ROUNDS_LATE = {5, 6, 7}
STREAM_ROUNDS_EARLY = {0, 1, 2}


def load_stream(name: str, run: str = "study4") -> Tuple[List[dict], Dict[str, float]]:
    """Runs of one phase of Study 4, annotated with their relative gap.

    ``run`` selects the execution: "study4" is the second, reported in the body of
    the paper; "study4_run1" is the first, reported in the appendix.
    """
    runs = [r for r in read_jsonl(os.path.join(LOGS, run, name)) if r["valide"]]
    best: Dict[str, float] = {}
    for r in runs:
        best[r["instance"]] = min(best.get(r["instance"], math.inf), r["latence"])
    for r in runs:
        r["gap"] = (r["latence"] - best[r["instance"]]) / best[r["instance"]] * 100
    return runs, best


def stream_paired(runs: Sequence[dict], a: str, b: str,
                  rounds: set | None = None) -> List[float]:
    """Paired differences of relative gaps, by (instance, round)."""
    idx = {(r["instance"], r["tour"], r["bras"]): r["gap"] for r in runs
           if rounds is None or r["tour"] in rounds}
    keys = sorted({(i, t) for (i, t, arm) in idx if arm == a} &
                  {(i, t) for (i, t, arm) in idx if arm == b})
    return [idx[(i, t, a)] - idx[(i, t, b)] for i, t in keys]


def table_12() -> None:
    header(12, "Study 4, priors estimated on each corpus")
    priors = json.load(open(os.path.join(LOGS, "study4", "priors.json"),
                            encoding="utf-8"))
    candidates = ["G3", "G6", "G12", "G20", "G30"]
    print(f"{'Corpus':22s} {'Runs':>5s} " + "".join(f"{c:>9s}" for c in candidates))
    for arm, label in (("C", "Legacy corpus (arm C)"), ("D", "Pilot campaign (arm D)")):
        block = priors[arm]
        values = block["valeurs_predites_pp"]
        print(f"{label:22s} {block['corpus']:5d} " +
              "".join(f"{values[c]:+9.3f}" for c in candidates))
    print(f"\n  Policy frozen on {priors['date_gel']}, before any run of the stream.")


def table_13() -> None:
    header(13, "Study 4, learning curves")
    runs, _ = load_stream("stream.jsonl")
    arms = sorted({r["bras"] for r in runs})
    print(f"{'Round':7s} " + "".join(f"{a:>10s}" for a in arms))
    for t in sorted({r["tour"] for r in runs}):
        cells = [st.fmean([r["gap"] for r in runs if r["tour"] == t and r["bras"] == a])
                 for a in arms]
        print(f"{t + 1:<7d} " + "".join(f"{c:10.3f}" for c in cells))
    cells = [st.fmean([r["gap"] for r in runs if r["bras"] == a]) for a in arms]
    print(f"{'All':7s} " + "".join(f"{c:10.3f}" for c in cells))


def table_14() -> None:
    header(14, "Study 4, pre-specified decision rules")
    runs, _ = load_stream("stream.jsonl")
    comparisons = [
        ("Learning", "B", "A", STREAM_ROUNDS_LATE, "rounds 6-8"),
        ("Within family", "D", "B", STREAM_ROUNDS_EARLY, "rounds 1-3"),
        ("Within family", "D", "A", None, "all rounds"),
        ("Across families", "C", "B", STREAM_ROUNDS_EARLY, "rounds 1-3"),
        ("Across families", "C", "A", STREAM_ROUNDS_LATE, "rounds 6-8"),
    ]
    rows = [(q, a, b, w, stream_paired(runs, a, b, r)) for q, a, b, r, w in comparisons]
    adjusted = holm([wilcoxon_p(d) for *_, d in rows])
    print(f"{'Question':17s} {'Comparison':20s} {'Pairs':>6s} {'W/T/L':>10s} "
          f"{'Mean':>8s} {'HL [95% CI]':>26s} {'p':>8s} {'p (Holm)':>9s}")
    for (q, a, b, window, d), adj in zip(rows, adjusted):
        win = sum(1 for x in d if x < -1e-12)
        loss = sum(1 for x in d if x > 1e-12)
        lo, hi = hl_interval(d)
        print(f"{q:17s} {a + ' vs ' + b + ', ' + window:20s} {len(d):6d} "
              f"{f'{win}/{len(d) - win - loss}/{loss}':>10s} {st.fmean(d):+8.3f} "
              f"{hodges_lehmann(d):+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{wilcoxon_p(d):8.3f} {adj:9.3f}")
    late = [("C", "A"), ("B", "A")]
    print("\n  Per-instance mean difference, arm C against the control, all rounds:")
    for inst in sorted({r["instance"] for r in runs}):
        c = st.fmean([r["gap"] for r in runs if r["instance"] == inst and r["bras"] == "C"])
        a = st.fmean([r["gap"] for r in runs if r["instance"] == inst and r["bras"] == "A"])
        print(f"    {short(inst):9s} C {c:6.3f}   A {a:6.3f}   difference {c - a:+6.3f}")
    negatives = sum(1 for inst in {r["instance"] for r in runs}
                    if st.fmean([r["gap"] for r in runs if r["instance"] == inst
                                 and r["bras"] == "C" and r["tour"] in STREAM_ROUNDS_LATE])
                    < st.fmean([r["gap"] for r in runs if r["instance"] == inst
                                and r["bras"] == "A" and r["tour"] in STREAM_ROUNDS_LATE]))
    print(f"    Instances with a negative difference on rounds 6-8: {negatives}/11")
    del late


def table_15() -> None:
    header(15, "Study 4b, confirmatory replication on fresh seeds")
    runs, _ = load_stream("replication.jsonl")
    rows = [(a, "A2", stream_paired(runs, a, "A2")) for a in ("B2", "C2")]
    adjusted = holm([wilcoxon_p(d) for *_, d in rows])
    print(f"{'Comparison':16s} {'Pairs':>6s} {'W/T/L':>10s} {'Mean':>8s} "
          f"{'HL [95% CI]':>26s} {'p':>8s} {'p (Holm)':>9s}")
    for (a, b, d), adj in zip(rows, adjusted):
        win = sum(1 for x in d if x < -1e-12)
        loss = sum(1 for x in d if x > 1e-12)
        lo, hi = hl_interval(d)
        print(f"{a + ' vs ' + b:16s} {len(d):6d} "
              f"{f'{win}/{len(d) - win - loss}/{loss}':>10s} {st.fmean(d):+8.3f} "
              f"{hodges_lehmann(d):+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{wilcoxon_p(d):8.3f} {adj:9.3f}")
    print("\n  Holm enforces monotonicity, so the larger raw p-value cannot receive an")
    print("  adjusted value below that of the smaller one; both are therefore equal here.")


# ---------------------------------------------------------------------------
# consistency check against the published values
# ---------------------------------------------------------------------------
PUBLISHED = {
    "circuit SHARED vs FIX-3 wins": 32,
    "circuit SHARED vs FIX-3 losses": 11,
    "circuit RANDOM vs FIX-3 instance p": 0.469,
    "circuit RANDOM vs FIX-CAL instance p": 0.031,
    "path SHARED vs FIX-3 wins": 130,
    "path SHARED vs FIX-3 losses": 9,
    "transfer wins": 18,
    "transfer losses": 87,
    "stream C vs A late wins": 10,
    "stream C vs A late losses": 21,
    "replication B2 vs A2 wins": 33,
    "replication C2 vs A2 losses": 27,
    "run 1 stream C vs A late wins": 23,
    "run 1 stream C vs A late losses": 10,
    "run 1 replication C2 vs A2 losses": 37,
    "pooled replication C2 vs A2 pairs": 132,
    "run 1 stream C vs A late p": 0.013,
    "run 1 stream C vs A late p (Holm)": 0.065,
}


def run_check() -> int:
    circuit, _ = load_campaign("circuit")
    path, _ = load_campaign("path")
    d = paired(circuit, "PARTAGEE", "IND-3")
    got = {"circuit SHARED vs FIX-3 wins": sum(1 for x in d if x < -1e-12),
           "circuit SHARED vs FIX-3 losses": sum(1 for x in d if x > 1e-12)}
    dp = paired(path, "PARTAGEE", "IND-3")
    got["path SHARED vs FIX-3 wins"] = sum(1 for x in dp if x < -1e-12)
    got["path SHARED vs FIX-3 losses"] = sum(1 for x in dp if x > 1e-12)
    cm = instance_means(circuit)
    insts = sorted(cm)
    for key, b in (("circuit RANDOM vs FIX-3 instance p", "IND-3"),
                   ("circuit RANDOM vs FIX-CAL instance p", "IND-OPT")):
        diffs = [(cm[i]["HASARD"] - cm[i][b]) / cm[i][b] * 100 for i in insts]
        got[key] = round(wilcoxon_p([x for x in diffs if abs(x) > 1e-9]), 3)
    verdict = json.load(open(os.path.join(LOGS, "transfer", "verdict.json"),
                             encoding="utf-8"))
    got["transfer wins"] = verdict["victoires"]
    got["transfer losses"] = verdict["defaites"]
    stream, _ = load_stream("stream.jsonl")
    ds = stream_paired(stream, "C", "A", STREAM_ROUNDS_LATE)
    got["stream C vs A late wins"] = sum(1 for x in ds if x < -1e-12)
    got["stream C vs A late losses"] = sum(1 for x in ds if x > 1e-12)
    repli, _ = load_stream("replication.jsonl")
    db = stream_paired(repli, "B2", "A2")
    dc = stream_paired(repli, "C2", "A2")
    got["replication B2 vs A2 wins"] = sum(1 for x in db if x < -1e-12)
    got["replication C2 vs A2 losses"] = sum(1 for x in dc if x > 1e-12)
    s1, _ = load_stream("stream.jsonl", "study4_run1")
    d1 = stream_paired(s1, "C", "A", STREAM_ROUNDS_LATE)
    got["run 1 stream C vs A late wins"] = sum(1 for x in d1 if x < -1e-12)
    got["run 1 stream C vs A late losses"] = sum(1 for x in d1 if x > 1e-12)
    r1, _ = load_stream("replication.jsonl", "study4_run1")
    dc1 = stream_paired(r1, "C2", "A2")
    got["run 1 replication C2 vs A2 losses"] = sum(1 for x in dc1 if x > 1e-12)
    got["pooled replication C2 vs A2 pairs"] = len(dc) + len(dc1)
    # Appendix table C1 is produced by the pre-specified analysis statistic, so
    # the check applies that statistic rather than the one used for the body.
    import analyse_stream_study as pre
    run1_rows = [stream_paired(s1, a, b, r) for a, b, r in (
        ("B", "A", STREAM_ROUNDS_LATE), ("D", "B", STREAM_ROUNDS_EARLY),
        ("D", "A", None), ("C", "B", STREAM_ROUNDS_EARLY),
        ("C", "A", STREAM_ROUNDS_LATE))]
    run1_p = [pre.wilcoxon(list(x), [0.0] * len(x))[0] for x in run1_rows]
    got["run 1 stream C vs A late p"] = run1_p[4]
    got["run 1 stream C vs A late p (Holm)"] = holm(run1_p)[4]

    print(f"\n{'=' * 78}\nCONSISTENCY CHECK AGAINST THE PUBLISHED VALUES\n{'=' * 78}")
    failures = 0
    for key, expected in PUBLISHED.items():
        actual = got[key]
        same = (abs(actual - expected) < 1e-9 if isinstance(expected, int)
                else abs(actual - expected) < 0.0015)
        failures += 0 if same else 1
        print(f"  {'OK  ' if same else 'FAIL'}  {key:42s} published {expected}, "
              f"recomputed {actual}")
    print(f"\n  {len(PUBLISHED) - failures}/{len(PUBLISHED)} values agree.")
    return 0 if failures == 0 else 1


def main(argv: Sequence[str]) -> int:
    if "--check" in argv:
        return run_check()
    # Tables are selected by the label they carry in the paper: a number for the
    # body, a letter and a number for the appendices, as in "C1".
    wanted = {a.upper() for a in argv if not a.startswith("-")}

    def want(n) -> bool:
        return not wanted or str(n).upper() in wanted

    if want(3):
        table_3()
    if want(4):
        table_summary(4, "circuit")
    if want(5):
        table_rule(5, "circuit")
    if want(6):
        table_exploratory(6, "circuit")
    if want(7):
        table_7()
    if want(8):
        table_summary(8, "path")
    if want(9):
        table_rule(9, "path")
    if want(10):
        table_exploratory(10, "path")
    if not wanted:
        path_instance_level()
    if want(11):
        table_11()
    if want(12):
        table_12()
    if want(13):
        table_13()
    if want(14):
        table_14()
    if want(15):
        table_15()
    if want(16):
        table_16()
    if want(17):
        table_17()
    if want(18):
        table_18()
    if want(19):
        table_19()
    if want("C1"):
        table_c1()
    if want("C2"):
        table_c2()
    if want("C3"):
        table_c3()
    print("\nTables 1, 2, A1 and A2 are not derived from the run logs: the first two "
          "position the work in the literature, and the last two document the\n"
          "distance convention and the objective, which are verified by "
          "build_instances.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
