#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS OF THE PRE-SPECIFIED CAMPAIGN.
Run with: python analyse_campaign.py

Applies the decision rule fixed in run_campaign.py, without modifying it:

    Learning     PRIVEE and PARTAGEE significantly outperform IND-3 and HASARD,
                 and are not significantly outperformed by IND-OPT.
    Cooperation  PARTAGEE significantly outperforms PRIVEE.

Tests are two-sided paired Wilcoxon signed-rank tests, paired by (instance, seed)
on the relative gap to the best value observed on the instance, with a Holm
correction over the family of the seven comparisons of the rule.

Also produced: a per-instance table (mean, standard deviation, best value, gap to
the reference, median time to best), the quality recorded at several fractions of
the time limit, and a CSV file.
"""
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Tuple

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# =============================================================================
TOPOLOGIE = "circuit"                         # "circuit" ou "path"
JOURNAL = os.path.join("logs", "campaign", f"runs_{TOPOLOGIE}.jsonl")
FICHIER_REFERENCES = "bks_circuit.json" if TOPOLOGIE == "circuit" else "bks_path.json"
SEUIL = 0.05
# =============================================================================

REGLE = [  # (method, other): the method is required to outperform the other
    ('PRIVEE', 'IND-3'), ('PRIVEE', 'HASARD'), ('PARTAGEE', 'IND-3'), ('PARTAGEE', 'HASARD'),
    ('IND-OPT', 'PRIVEE'), ('IND-OPT', 'PARTAGEE'), ('PARTAGEE', 'PRIVEE'),
]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ICI, p)


def wilcoxon(x: List[float], y: List[float]) -> Tuple[float, int]:
    """Two-sided Wilcoxon signed-rank test; tied pairs are discarded.

    The test itself is the one defined in reproduce_tables.py, so that this
    script and the reconstruction of the published tables cannot drift apart,
    and so that neither of them depends on a third-party library for the choice
    between the exact null distribution and the normal approximation.
    """
    from reproduce_tables import wilcoxon_p
    diffs = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    if not diffs:
        return 1.0, 0
    return wilcoxon_p(diffs), len(diffs)


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    ordre = sorted(range(m), key=lambda i: pvals[i])
    ajust = [0.0] * m
    courant = 0.0
    for rang, i in enumerate(ordre):
        courant = max(courant, min(1.0, (m - rang) * pvals[i]))
        ajust[i] = courant
    return ajust


def main() -> int:
    chemin = _abs(JOURNAL)
    if not os.path.exists(chemin):
        print(f"Journal introuvable : {chemin}")
        return 1
    enr = [json.loads(l) for l in open(chemin, encoding='utf-8') if l.strip()]
    invalides = [r for r in enr if not r.get('valide')]
    if invalides:
        print(f"  WARNING: {len(invalides)} infeasible solution(s); they are excluded.")
    enr = [r for r in enr if r.get('valide')]
    bks = {}
    if os.path.exists(_abs(FICHIER_REFERENCES)):
        bks = {k: v for k, v in json.load(open(_abs(FICHIER_REFERENCES))).items()
               if not k.startswith('_')}

    methodes = [m for m in ['IND-3', 'IND-OPT', 'HASARD', 'PRIVEE', 'PARTAGEE', 'PORTEFEUILLE']
                if any(r['methode'] == m for r in enr)]
    instances = sorted({r['instance'] for r in enr})
    meilleure = {i: min(r['latence'] for r in enr if r['instance'] == i) for i in instances}
    for r in enr:
        r['ecart_rel'] = (r['latence'] - meilleure[r['instance']]) / meilleure[r['instance']] * 100

    def cle_bks(nom):
        # KTRP_test_E76_10 -> E76K10 ; KTRP_path_CMT3 -> CMT3
        base = nom.replace('KTRP_test_', '').replace('KTRP_path_', '')
        if '_' in base:
            a, b = base.rsplit('_', 1)
            return f"{a}K{b}"
        return base

    # --- Per-instance table -----------------------------------------------------
    print("=" * 100)
    print("PER-INSTANCE RESULTS")
    print("=" * 100)
    lignes_csv = []
    for i in instances:
        ref = bks.get(cle_bks(i))
        print(f"\n  {i}  (meilleure observee {meilleure[i]:.2f}"
              + (f", reference {ref}" if ref is not None else "") + ")")
        for m in methodes:
            v = [r['latence'] for r in enr if r['instance'] == i and r['methode'] == m]
            t = [r['temps_meilleure_s'] for r in enr if r['instance'] == i and r['methode'] == m]
            if not v:
                continue
            ecart = (st.fmean(v) - ref) / ref * 100 if ref else None
            print(f"    {m:12s} moy {st.fmean(v):11.2f}  ec.type {st.pstdev(v):8.2f}  "
                  f"meilleure {min(v):11.2f}  "
                  + (f"ecart moy. {ecart:+6.2f} %  " if ecart is not None else "")
                  + f"t. obtention med. {st.median(t):7.1f} s  (n={len(v)})")
            lignes_csv.append({'instance': i, 'methode': m, 'n': len(v), 'moyenne': st.fmean(v),
                               'ecart_type': st.pstdev(v), 'meilleure': min(v), 'pire': max(v),
                               'reference': ref, 'ecart_moyen_ref_pct': ecart,
                               'temps_obtention_median_s': st.median(t)})

    # --- Quality at fractions of the time limit --------------------------------
    print("\n" + "=" * 100)
    print("MEAN RELATIVE GAP TO THE BEST VALUE, BY FRACTION OF THE TIME LIMIT")
    print("=" * 100)
    fractions = sorted({float(k) / r['limite_s'] for r in enr for k in r['releves']})
    for m in methodes:
        ligne = []
        for r_ in sorted({round(f, 2) for f in fractions}):
            vals = []
            for r in enr:
                if r['methode'] != m:
                    continue
                for k, v in r['releves'].items():
                    if abs(float(k) / r['limite_s'] - r_) < 0.01:
                        vals.append((v - meilleure[r['instance']]) / meilleure[r['instance']] * 100)
            if vals:
                ligne.append(f"{int(r_ * 100):3d} % : {st.fmean(vals):6.3f}")
        print(f"  {m:12s} " + " | ".join(ligne))

    # --- Regle de decision ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("PRE-SPECIFIED DECISION RULE")
    print("=" * 100)
    index = {(r['instance'], r['graine'], r['methode']): r['ecart_rel'] for r in enr}
    resultats = []
    for a, b in REGLE:
        if a not in methodes or b not in methodes:
            resultats.append((a, b, None, None, None, 0, (0, 0, 0)))
            continue
        cles = sorted({(i, g) for (i, g, m) in index if m == a} & {(i, g) for (i, g, m) in index if m == b})
        x = [index[(i, g, a)] for i, g in cles]
        y = [index[(i, g, b)] for i, g in cles]
        p, n_eff = wilcoxon(x, y)
        diff = st.fmean([u - v for u, v in zip(x, y)]) if x else None
        vic = sum(1 for u, v in zip(x, y) if u < v - 1e-12)
        def_ = sum(1 for u, v in zip(x, y) if u > v + 1e-12)
        resultats.append((a, b, p, None, diff, n_eff, (vic, len(x) - vic - def_, def_)))
    ps = [r[2] for r in resultats if r[2] is not None]
    ajust = holm(ps)
    k = 0
    final = []
    for a, b, p, _, diff, n_eff, (v, e, d) in resultats:
        if p is None:
            final.append((a, b, None, None, None))
            print(f"  {a:9s} devance {b:9s} ? comparaison impossible (methode absente)")
            continue
        pa = ajust[k]; k += 1
        devance = diff < 0 and pa < SEUIL
        final.append((a, b, pa, diff, devance))
        print(f"  {a:9s} outperforms {b:9s} ? mean diff {diff:+7.3f} pt, wins/ties/losses "
              f"{v}/{e}/{d}, p ajuste (Holm) {pa:.4g} -> {'OUI' if devance else 'non'}")

    def res(a, b):
        for x in final:
            if x[0] == a and x[1] == b:
                return x
        return None

    apprentissage = all(res(a, b) and res(a, b)[4] for a, b in REGLE[:4]) and \
        not any(res('IND-OPT', m) and res('IND-OPT', m)[4] for m in ('PRIVEE', 'PARTAGEE'))
    cooperation = bool(res('PARTAGEE', 'PRIVEE') and res('PARTAGEE', 'PRIVEE')[4])
    print("\n  VERDICT")
    print(f"    Apprentissage etabli : {'OUI' if apprentissage else 'NON'}")
    print(f"    Cooperation etablie  : {'OUI' if cooperation else 'NON'}")
    if not apprentissage:
        print("    The claim cannot be stated as a contribution of learning.")
    if apprentissage and not cooperation:
        print("    The claim is limited to learned control; no contribution of sharing is established.")

    sortie = os.path.join(os.path.dirname(chemin), f'results_{TOPOLOGIE}.csv')
    with open(sortie, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(lignes_csv[0].keys()))
        w.writeheader()
        w.writerows(lignes_csv)
    print(f"\n  Per-instance table: {sortie}")
    return 0


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
