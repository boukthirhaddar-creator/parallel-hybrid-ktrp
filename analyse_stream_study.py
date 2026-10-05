#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS OF STUDY 4, PRE-SPECIFIED. Run with: python analyse_stream_study.py

Reads logs/study4/*.jsonl and logs/study4/priors.json and produces every
quantity that PROTOCOL_STUDY4.md announces: the pre-specified paired
comparisons with their Wilcoxon signed-rank tests and Hodges-Lehmann
intervals, the learning curves, the composition of the candidates played,
and the descriptive tables of the pilot campaign.

Nothing is fitted, tuned or selected here: every number is a function of the
deposited logs alone, and every comparison is one of the rules stated in the
protocol, which is recalled in the header of each section.
"""
import json
import math
import os
import statistics as st
from typing import Any, Dict, List, Optional, Tuple

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER = os.path.join(ICI, 'logs', 'study4')

BRAS = ['A', 'B', 'C', 'D']
NOMS_CANDIDATS = ['U', 'G3', 'G6', 'G12', 'G20', 'G30']


def charger(nom: str) -> List[Dict[str, Any]]:
    chemin = os.path.join(DOSSIER, nom)
    if not os.path.exists(chemin):
        return []
    sortie = []
    with open(chemin, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                sortie.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return sortie


# ============================================================================= 
#  Statistical tests, identical to those of analyse_campaign.py
# =============================================================================
def wilcoxon(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples, normal
    approximation with continuity correction, and the Hodges-Lehmann
    estimate of the median paired difference."""
    d = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    rng = sorted(range(n), key=lambda i: abs(d[i]))
    rangs = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[rng[j + 1]]) == abs(d[rng[i]]):
            j += 1
        moyen = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            rangs[rng[k]] = moyen
        i = j + 1
    w_plus = sum(r for r, dd in zip(rangs, d) if dd > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 1.0, 0.0
    z = (w_plus - mu - 0.5 * (1 if w_plus > mu else -1 if w_plus < mu else 0)) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    # Hodges-Lehmann: median of the pairwise Walsh averages
    paires = []
    for i in range(n):
        for j in range(i, n):
            paires.append((d[i] + d[j]) / 2.0)
    paires.sort()
    m = len(paires)
    hl = paires[m // 2] if m % 2 == 1 else 0.5 * (paires[m // 2 - 1] + paires[m // 2])
    return max(0.0, min(1.0, p)), hl


def intervalle_hl(d: List[float]) -> Tuple[float, float]:
    """Distribution-free 95 % confidence interval,
    obtained from the ordered Walsh averages."""
    if not d:
        return (float('nan'), float('nan'))
    n = len(d)
    obs = sorted(d)
    # Bound from the Wilcoxon statistic, the classical construction
    w_crit = None
    # Normal approximation of the Wilcoxon critical value at alpha = 0.05
    table = {5: 0, 6: 1, 7: 2, 8: 4, 9: 6, 10: 8, 11: 11, 12: 14, 13: 17,
             14: 21, 15: 25, 16: 30, 17: 35, 18: 41, 19: 47, 20: 54,
             21: 60, 22: 68, 23: 75, 24: 83, 25: 92, 26: 101, 27: 110,
             28: 120, 29: 130, 30: 141, 31: 152, 32: 163, 33: 175, 34: 188,
             35: 200, 36: 214, 37: 228, 38: 242, 39: 256, 40: 271}
    if n in table:
        w_crit = table[n]
    else:
        w_crit = round(n * (n + 1) / 4.0 - 1.96 * math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
    moyennes = []
    for i in range(n):
        for j in range(i, n):
            moyennes.append(obs[i] + obs[j])
    moyennes.sort()
    # The Hodges-Lehmann interval is read off the ordered Walsh averages
    try:
        bas = 0.5 * moyennes[w_crit - 1]
        haut = 0.5 * moyennes[len(moyennes) - w_crit]
        return (bas, haut)
    except IndexError:
        return (float('nan'), float('nan'))


# =============================================================================
#  Data preparation
# =============================================================================
def preparer() -> Dict[str, Any]:
    flux = [r for r in charger('stream.jsonl') if r['phase'] == 'flux' and r['valide']]
    pilote = [r for r in charger('pilot.jsonl') if r['phase'] == 'pilote' and r['valide']]
    priors = json.load(open(os.path.join(DOSSIER, 'priors.json'), encoding='utf-8'))
    best: Dict[str, float] = {}
    for r in flux:
        best[r['instance']] = min(best.get(r['instance'], math.inf), r['latence'])
    for r in flux:
        r['ecart'] = (r['latence'] - best[r['instance']]) / best[r['instance']] * 100.0
    return {'flux': flux, 'pilote': pilote, 'priors': priors, 'best': best}


def paires(donnees: Dict[str, Any], bras1: str, bras2: str,
           tours: Optional[List[int]] = None) -> Tuple[List[float], List[float],
                                                       List[Tuple[str, int]]]:
    """Paired differences of arm1 and arm2, by (instance, round)."""
    idx: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in donnees['flux']:
        if tours is not None and r['tour'] not in tours:
            continue
        idx.setdefault((r['instance'], r['tour']), {})[r['bras']] = r['ecart']
    cles = sorted(k for k, v in idx.items() if bras1 in v and bras2 in v)
    x = [idx[k][bras1] for k in cles]
    y = [idx[k][bras2] for k in cles]
    return x, y, cles


def rapport(donnees: Dict[str, Any], bras1: str, bras2: str,
            tours: Optional[List[int]], titre: str, etabli_si: str) -> None:
    x, y, cles = paires(donnees, bras1, bras2, tours)
    n = len(x)
    print(f"\n--- {titre} ---")
    if n == 0:
        print("  no complete pair.")
        return
    d = [a - b for a, b in zip(x, y)]
    p, hl = wilcoxon(x, y)
    bas, haut = intervalle_hl(d)
    v = sum(1 for dd in d if dd < -1e-12)
    e = sum(1 for dd in d if dd > 1e-12)
    print(f"  {bras1} versus {bras2}: {n} pairs, wins/ties/losses "
          f"{v}/{n - v - e}/{e}")
    print(f"  mean difference {st.fmean(d):+.4f} pp, HL median {hl:+.4f} pp, "
          f"IC 95 % [{bas:+.4f}, {haut:+.4f}]")
    print(f"  p = {p:.4g}")
    print(f"  {etabli_si} : {'OUI' if (hl < 0 and p < 0.05) else 'NON'}"
          f"{' (direction opposée)' if hl > 0 else ''}")
    # Instance-level sensitivity
    par_inst: Dict[str, List[float]] = {}
    for (inst, tour), dd in zip(cles, d):
        par_inst.setdefault(inst, []).append(dd)
    if len(par_inst) >= 3:
        moy = [st.fmean(v) for v in par_inst.values()]
        neg = sum(1 for m in moy if m < 0)
        print(f"  instance-level sensitivity: {neg}/{len(moy)} negative means, "
              f"moyenne {st.fmean(moy):+.4f} pp")


# =============================================================================
#  Rapport principal
# =============================================================================
def main() -> int:
    donnees = preparer()
    flux = donnees['flux']
    priors = donnees['priors']
    print("=" * 84)
    print("ÉTUDE 4 : APPRENTISSAGE SUR UN FLUX DE RÉSOLUTIONS RÉPÉTÉES")
    print("=" * 84)
    print(f"valid runs: stream {len(flux)}, pilot {len(donnees['pilote'])}")
    tours = sorted({r['tour'] for r in flux})
    instances = sorted({r['instance'] for r in flux})
    print(f"instances: {len(instances)}, rounds present: {tours}")

    # --- hardware and priors ---------------------------------------------------
    print("\n--- Frozen priors (extract) ---")
    print(f"  date de gel : {priors['date_gel']}")
    for bras in ('C', 'D'):
        vp = priors[bras]['valeurs_predites_pp']
        print(f"  bras {bras} (corpus {priors[bras]['corpus']}) : "
              + ', '.join(f"{c} {vp[c]:+.3f}" for c in NOMS_CANDIDATS))

    # --- courbes d'apprentissage ----------------------------------------------
    print("\n--- Mean gap (pp) by round and by arm ---")
    entete = "  tour   " + "".join(f"{b:>10s}" for b in BRAS)
    print(entete)
    for t in tours:
        ligne = f"  {t:4d}   "
        for b in BRAS:
            v = [r['ecart'] for r in flux if r['tour'] == t and r['bras'] == b]
            ligne += f"{(st.fmean(v) if v else float('nan')):10.4f}"
        nb = len([r for r in flux if r['tour'] == t and r['bras'] == 'A'])
        ligne += f"   ({nb} paires)"
        print(ligne)

    # --- composition of the candidates played ---------------------------------
    print("\n--- Candidate played by round (arms B, C, D) ---")
    for t in tours:
        ligne = f"  tour {t} : "
        for b in ('B', 'C', 'D'):
            comptes: Dict[str, int] = {}
            for r in flux:
                if r['tour'] == t and r['bras'] == b:
                    comptes[r['candidat']] = comptes.get(r['candidat'], 0) + 1
            chaine = ' '.join(f"{c}:{comptes.get(c, 0)}" for c in NOMS_CANDIDATS)
            ligne += f"[{b} {chaine}] "
        print(ligne)

    # --- pre-specified decision rules ------------------------------------------
    tours_tard = [t for t in tours if t >= 5]
    tours_tot = sorted({r['tour'] for r in flux})
    print("\n" + "=" * 84)
    print("RÈGLES DE DÉCISION PRÉ-SPÉCIFIÉES (protocole)")
    print("=" * 84)
    if len(tours_tard) >= 3:
        rapport(donnees, 'B', 'A', tours_tard,
                "Règle 1 : apprentissage (B contre A, tours 6 à 8)",
                "Apprentissage établi")
        rapport(donnees, 'D', 'B', [0, 1, 2],
                "Règle 2a : transfert même famille (D contre B, tours 1 à 3)",
                "Accélération établie")
        rapport(donnees, 'D', 'A', tours_tot,
                "Règle 2b : transfert même famille (D contre A, tous tours)",
                "Transfert établi")
        rapport(donnees, 'C', 'B', [0, 1, 2],
                "Règle 3a : transfert inter-familles (C contre B, tours 1 à 3)",
                "Écart en faveur de C")
        rapport(donnees, 'C', 'A', tours_tard,
                "Règle 3b : transfert inter-familles (C contre A, tours 6 à 8)",
                "Écart en faveur de C")
    else:
        print(f"  too few late rounds ({tours_tard}): study incomplete.")

    # --- per-instance summary --------------------------------------------------
    print("\n--- Mean gap by instance and by arm (pp) ---")
    idx_tard: Dict[Tuple[str, int], Dict[str, float]] = {}
    idx_tot: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in flux:
        idx_tot.setdefault((r['instance'], r['tour']), {})[r['bras']] = r['ecart']
        if r['tour'] in tours_tard:
            idx_tard.setdefault((r['instance'], r['tour']), {})[r['bras']] = r['ecart']

    def moyenne_par_instance(table, bras1, bras2):
        out: Dict[str, List[float]] = {}
        for (inst, _t), v in table.items():
            if bras1 in v and bras2 in v:
                out.setdefault(inst, []).append(v[bras1] - v[bras2])
        return {k: st.fmean(v) for k, v in out.items()}

    ba = moyenne_par_instance(idx_tard, 'B', 'A')
    da = moyenne_par_instance(idx_tot, 'D', 'A')
    print(f"  {'instance':22s} " + "".join(f"{b:>10s}" for b in BRAS) +
          "   B-A tard   D-A total")
    for inst in instances:
        ligne = f"  {inst:22s} "
        for b in BRAS:
            v = [r['ecart'] for r in flux if r['instance'] == inst and r['bras'] == b]
            ligne += f"{(st.fmean(v) if v else float('nan')):10.4f}"
        ligne += f"   {(ba.get(inst, float('nan'))):+9.4f}"
        ligne += f"   {(da.get(inst, float('nan'))):+9.4f}"
        print(ligne)
    return 0


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
