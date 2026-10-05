#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-specified analysis of replication 4b: A2, B2, C2 on fresh seeds."""
import json
import math
import os
import statistics as st
import importlib.util as iu

ICI = os.path.dirname(os.path.abspath(__file__))
spec = iu.spec_from_file_location("aa", os.path.join(ICI, 'analyse_stream_study.py'))
aa = iu.module_from_spec(spec)
spec.loader.exec_module(aa)

DOSSIER = os.path.join(ICI, 'logs', 'study4')


def charger(nom):
    chemin = os.path.join(DOSSIER, nom)
    if not os.path.exists(chemin):
        return []
    out = []
    for l in open(chemin, encoding='utf-8'):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def main():
    repli = [r for r in charger('replication.jsonl') if r['valide']]
    print(f"valid runs: {len(repli)}")
    best = {}
    for r in repli:
        best[r['instance']] = min(best.get(r['instance'], math.inf), r['latence'])
    for r in repli:
        r['ecart'] = (r['latence'] - best[r['instance']]) / best[r['instance']] * 100
        r['tour'] = r['tour']

    tours = sorted({r['tour'] for r in repli})
    instances = sorted({r['instance'] for r in repli})
    print(f"instances: {len(instances)}, rounds: {tours}")

    print("\n--- Mean gap (pp) by round and by arm ---")
    print("  tour     A2       B2       C2")
    for t in tours:
        ligne = f"  {t:4d} "
        for b in ('A2', 'B2', 'C2'):
            v = [r['ecart'] for r in repli if r['tour'] == t and r['bras'] == b]
            ligne += f"{(st.fmean(v) if v else float('nan')):8.4f}"
        print(ligne)
    for b in ('A2', 'B2', 'C2'):
        v = [r['ecart'] for r in repli if r['bras'] == b]
        print(f"  {b} global : {st.fmean(v):8.4f}")

    print("\n--- Candidates played ---")
    for b in ('B2', 'C2'):
        comp = {}
        for r in repli:
            if r['bras'] == b:
                comp[r['candidat']] = comp.get(r['candidat'], 0) + 1
        print(f"  bras {b} : {comp}")

    # Pre-specified tests, Holm correction over the two comparisons
    idx = {}
    for r in repli:
        idx.setdefault((r['instance'], r['tour']), {})[r['bras']] = r['ecart']
    cles = sorted(idx)
    resultats = {}
    for b1, b2 in (('B2', 'A2'), ('C2', 'A2')):
        x = [idx[k][b1] for k in cles]
        y = [idx[k][b2] for k in cles]
        d = [a - bb for a, bb in zip(x, y)]
        p, hl = aa.wilcoxon(x, y)
        bas, haut = aa.intervalle_hl(d)
        v = sum(1 for dd in d if dd < -1e-12)
        e = sum(1 for dd in d if dd > 1e-12)
        resultats[b1] = p
        par_inst = {}
        for (inst, _t), dd in zip(cles, d):
            par_inst.setdefault(inst, []).append(dd)
        moy = {k: st.fmean(vv) for k, vv in par_inst.items()}
        neg = sum(1 for m in moy.values() if m < 0)
        print(f"\n--- {b1} versus {b2} (pre-specified) ---")
        print(f"  {len(x)} pairs, wins/ties/losses {v}/{len(x)-v-e}/{e}")
        print(f"  moyenne {st.fmean(d):+.4f} pp, HL {hl:+.4f} pp, IC 95 % [{bas:+.4f}, {haut:+.4f}]")
        print(f"  p = {p:.4f}")
        print(f"  instance-level sensitivity: {neg}/{len(moy)} negative means, "
              f"moyenne {st.fmean(list(moy.values())):+.4f} pp")

    # Holm step-down. The monotonicity step matters: the adjusted value of a
    # comparison can never fall below that of a comparison with a smaller raw
    # p-value, so the running maximum is carried forward.
    print("\n--- Holm correction over the two comparisons ---")
    ordre = sorted(resultats.items(), key=lambda kv: kv[1])
    m = len(ordre)
    courant = 0.0
    for rang, (nom, p) in enumerate(ordre):
        courant = max(courant, min(1.0, (m - rang) * p))
        print(f"  {nom} versus A2: p = {p:.4f}, adjusted p (Holm) = {courant:.4f} "
              f"{'SIGNIFICANT' if courant < 0.05 else 'not significant'}")

    print("\n--- Mean gap by instance and by arm (pp) ---")
    print(f"  {'instance':22s}   A2       B2       C2    C2-A2   B2-A2")
    for inst in instances:
        ligne = f"  {inst:22s} "
        mvals = {}
        for b in ('A2', 'B2', 'C2'):
            v = [r['ecart'] for r in repli if r['instance'] == inst and r['bras'] == b]
            mvals[b] = st.fmean(v) if v else float('nan')
            ligne += f"{mvals[b]:8.4f}"
        ligne += f"  {mvals['C2']-mvals['A2']:+7.4f} {mvals['B2']-mvals['A2']:+7.4f}"
        print(ligne)


if __name__ == '__main__':
    main()
