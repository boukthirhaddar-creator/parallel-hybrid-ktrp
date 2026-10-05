#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRE-SPECIFIED CAMPAIGN: DOES LEARNED COOPERATION IMPROVE THE SEARCH?
Run with: python run_campaign.py

Question
--------
Do parallel tabu agents that learn together how to steer their search obtain
better solutions than independent agents?

Configurations compared, all on P processes, at equal wall-clock time, from the
same initial solution and with the same ktrp_worker kernel:

    IND-3      independent search, tenure fixed at the smallest value of the
               grid: the default baseline;
    IND-OPT    independent search, fixed tenure calibrated on the validation
               instances, never on the test instances;
    HASARD     tenure drawn at random at each epoch: control without learning;
    PRIVEE     each agent learns alone: learning without cooperation;
    PARTAGEE   the agents learn together: learned cooperation.

Decision rule, fixed before any run
-----------------------------------
    Learning is established if PRIVEE and PARTAGEE both significantly outperform
    IND-3 and HASARD (paired Wilcoxon, Holm correction, 5 % level) without being
    significantly outperformed by IND-OPT.
    Cooperation is established if PARTAGEE significantly outperforms PRIVEE.
    No other formulation of the claim is retained from these data.

Stages
------
    Phase 0  calibration of IND-OPT on the validation instances;
    Phase 1  test runs, resumed where they stopped if interrupted;
    Phase 2  analysis, by analyse_campaign.py.
"""
import json
import math
import multiprocessing as mp
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if ICI not in sys.path:
    sys.path.insert(0, ICI)

import ktrp_worker as W            # noqa: E402
import ktrp_coordinator as C        # noqa: E402

# =============================================================================
#                              CONFIGURATION
# =============================================================================

TOPOLOGIE = "circuit"                         # "circuit" ou "path"
FICHIER_INSTANCES_TEST = f"instances_main_{TOPOLOGIE}.txt"
FICHIER_INSTANCES_VALIDATION = f"instances_validation_{TOPOLOGIE}.txt"

METHODES = ['IND-3', 'IND-OPT', 'HASARD', 'PRIVEE', 'PARTAGEE']

# Ten distinct seeds, one per run and per instance. All configurations share the
# same seeds, which pairs the runs.
GRAINES = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]
GRAINES_CALIBRATION = [2003, 2011, 2017]

PROCESSUS = 3

# Time limit proportional to instance size: SECONDES_PAR_CLIENT seconds per
# customer, bounded. Quality is additionally recorded at several fractions of that
# limit, so that results can be reported for several budgets from the same runs.
SECONDES_PAR_CLIENT = 1.2
DUREE_MIN_S = 60
DUREE_MAX_S = 600
FRACTIONS_RELEVES = (0.1, 0.25, 0.5, 1.0)
N_EPOQUES = 30

DOSSIER = "run_campaign"

# =============================================================================


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ICI, p)


def lire_instance(chemin: str) -> Tuple[int, int, List[List[float]]]:
    lignes = [l.strip() for l in open(chemin, encoding='utf-8', errors='replace') if l.strip()]
    K, n = int(lignes[0]), int(lignes[1])
    d = [[float(x) for x in l.split()] for l in lignes[2:2 + n]]
    return K, n, d


def plus_proche_voisin(d: List[List[float]], K: int) -> List[List[int]]:
    """Initial construction, identical for every configuration."""
    n = len(d)
    restants = set(range(1, n))
    routes = [[0] for _ in range(K)]
    k = 0
    while restants:
        c = min(restants, key=lambda x: (d[routes[k][-1]][x], x))
        routes[k].append(c)
        restants.remove(c)
        k = (k + 1) % K
    return [r + [0] for r in routes]


def charger_liste(fichier: str) -> List[str]:
    chemin = _abs(fichier)
    if not os.path.exists(chemin):
        return []
    base = os.path.dirname(chemin)
    trouves = []
    for l in open(chemin, encoding='utf-8', errors='replace'):
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        for cand in (l, os.path.join(base, l)):
            if os.path.exists(cand):
                trouves.append(os.path.abspath(cand))
                break
        else:
            print(f"  instance introuvable, ignoree : {l}")
    return trouves


def duree_limite(n_clients: int) -> float:
    return float(min(DUREE_MAX_S, max(DUREE_MIN_S, SECONDES_PAR_CLIENT * n_clients)))


def executer(pool, chemin: str, methode: str, graine: int, fraction_opt: float = None
             ) -> Dict[str, Any]:
    K, n, d = lire_instance(chemin)
    circuit = TOPOLOGIE == 'circuit'
    n_clients = n - 1
    T = duree_limite(n_clients)
    depart = plus_proche_voisin(d, K)
    options = C.options_duree_instance(n_clients)
    kw = dict(pool=pool, P=PROCESSUS, T=T, depart=depart, distances=d, circuit=circuit,
              graine=graine, n_epoques=N_EPOQUES, options_duree=options,
              instants_releves=[round(f * T, 3) for f in FRACTIONS_RELEVES])
    if methode == 'IND-3':
        r = C.recherche_cooperative(politique='independante', duree_tabou=3, **kw)
    elif methode == 'IND-OPT':
        duree = max(2, int(round(fraction_opt * n_clients)))
        r = C.recherche_cooperative(politique='independante', duree_tabou=duree, **kw)
    elif methode.startswith('IND-F'):          # calibration : IND-F0.12, etc.
        duree = max(2, int(round(float(methode[5:]) * n_clients)))
        r = C.recherche_cooperative(politique='independante', duree_tabou=duree, **kw)
    elif methode == 'HASARD':
        r = C.recherche_cooperative(politique='hasard', **kw)
    elif methode == 'PRIVEE':
        r = C.recherche_cooperative(politique='privee', **kw)
    elif methode == 'PARTAGEE':
        r = C.recherche_cooperative(politique='partagee', **kw)
    elif methode == 'PORTEFEUILLE':
        r = C.recherche_cooperative(politique='portefeuille', **kw)
    else:
        raise ValueError(methode)
    # Independent check of the returned solution.
    valide = W.est_valide(r['routes'], K, n, True)
    recalcul = W.latence(r['routes'], d, circuit)
    return {
        'instance': os.path.splitext(os.path.basename(chemin))[0],
        'n_clients': n_clients, 'K': K, 'topologie': TOPOLOGIE,
        'methode': methode, 'graine': graine, 'limite_s': T,
        'latence': r['latence'], 'latence_recalculee': recalcul,
        'valide': bool(valide and abs(recalcul - r['latence']) < 1e-6),
        'latence_depart': r['latence_depart'],
        'temps_meilleure_s': r['temps_meilleure_s'],
        'evaluations': r['evaluations'], 'epoques': r['epoques'],
        'options_duree': r['options_duree'], 'choix_durees': r['choix_durees'],
        'releves': r['releves'], 'politique_apprise': r.get('politique_apprise'),
        'duree_mur_s': r['duree_s'], 'routes': r['routes'],
    }


def journal_charger(chemin: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chemin):
        return []
    return [json.loads(l) for l in open(chemin, encoding='utf-8') if l.strip()]


def campagne(instances: List[str], methodes: List[str], graines: List[int], journal: str,
             fraction_opt: float = None) -> List[Dict[str, Any]]:
    faits = {(r['instance'], r['methode'], r['graine']) for r in journal_charger(journal)}
    ctx = mp.get_context('spawn')
    for chemin in instances:
        nom = os.path.splitext(os.path.basename(chemin))[0]
        a_faire = [(m, g) for g in graines for m in methodes if (nom, m, g) not in faits]
        if not a_faire:
            continue
        K, n, d = lire_instance(chemin)
        print(f"\n  {nom} : {n - 1} clients, K = {K}, limite {duree_limite(n - 1):.0f} s, "
              f"{len(a_faire)} execution(s)")
        with ProcessPoolExecutor(PROCESSUS, mp_context=ctx, initializer=W.initialiser,
                                 initargs=(d, K, n, TOPOLOGIE == 'circuit', True)) as pool:
            pool.submit(W.latence, [[0, 0]], d, True).result()   # demarrage hors mesure
            for m, g in a_faire:
                enr = executer(pool, chemin, m, g, fraction_opt)
                with open(journal, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(enr, ensure_ascii=False) + '\n')
                print(f"    {m:12s} seed {g}  latency {enr['latence']:11.2f}  reached at "
                      f"{enr['temps_meilleure_s']:7.1f} s  {'' if enr['valide'] else 'INFEASIBLE'}",
                      flush=True)
    return journal_charger(journal)


def calibrer(journal_calib: str) -> float:
    """Select the tenure fraction minimising the mean gap to the best value
    observed on each validation instance."""
    instances = charger_liste(FICHIER_INSTANCES_VALIDATION)
    if not instances:
        raise SystemExit(f"Aucune instance de validation : {FICHIER_INSTANCES_VALIDATION}")
    methodes = [f'IND-F{f}' for f in C.FRACTIONS_DUREE]
    enr = campagne(instances, methodes, GRAINES_CALIBRATION, journal_calib)
    meilleures: Dict[str, float] = {}
    for r in enr:
        meilleures[r['instance']] = min(meilleures.get(r['instance'], math.inf), r['latence'])
    score: Dict[str, List[float]] = {}
    for r in enr:
        score.setdefault(r['methode'], []).append(
            (r['latence'] - meilleures[r['instance']]) / meilleures[r['instance']] * 100)
    print("\n  Mean gap to the best observed value, by tenure fraction:")
    for m in methodes:
        print(f"    {m:12s} {st.fmean(score[m]):7.3f} %  ({len(score[m])} runs)")
    # Criterion fixed in advance: lowest mean gap; ties broken by the shorter
    # mean time to best.
    temps = {m: st.fmean([r['temps_meilleure_s'] for r in enr if r['methode'] == m]) for m in methodes}
    meilleure = min(methodes, key=lambda m: (round(st.fmean(score[m]), 6), temps[m]))
    return float(meilleure[5:])


def main() -> int:
    os.makedirs(_abs(DOSSIER), exist_ok=True)
    print("=" * 84)
    print("PRE-SPECIFIED CAMPAIGN: LEARNED COOPERATION VERSUS INDEPENDENT SEARCH")
    print("=" * 84)
    print(f"  topology {TOPOLOGIE}, {PROCESSUS} processes, {len(GRAINES)} seeds, "
          f"methodes {', '.join(METHODES)}")

    # --- Phase 0 : calibration -------------------------------------------------
    fichier_calib = os.path.join(_abs(DOSSIER), f'calibration_{TOPOLOGIE}.json')
    if 'IND-OPT' in METHODES:
        if os.path.exists(fichier_calib):
            fraction = json.load(open(fichier_calib))['fraction']
            print(f"\n  Existing calibration: tenure fraction {fraction}")
        else:
            print("\nPHASE 0: CALIBRATION OF IND-OPT ON THE VALIDATION INSTANCES")
            fraction = calibrer(os.path.join(_abs(DOSSIER), f'calibration_{TOPOLOGIE}.jsonl'))
            json.dump({'fraction': fraction, 'fractions_testees': list(C.FRACTIONS_DUREE),
                       'instances': charger_liste(FICHIER_INSTANCES_VALIDATION)},
                      open(fichier_calib, 'w'), indent=1)
            print(f"  Fraction retenue : {fraction}")
    else:
        fraction = None

    # --- Phase 1: test runs ------------------------------------------------------
    instances = charger_liste(FICHIER_INSTANCES_TEST)
    if not instances:
        print(f"No test instance found: {FICHIER_INSTANCES_TEST}")
        return 1
    communes = set(map(os.path.basename, instances)) & set(
        map(os.path.basename, charger_liste(FICHIER_INSTANCES_VALIDATION)))
    if communes:
        print(f"  ERROR: instances common to validation and test: {sorted(communes)}")
        return 1
    total = sum(duree_limite(lire_instance(c)[1] - 1) for c in instances) * len(GRAINES) * len(METHODES)
    print(f"\nPHASE 1: {len(instances)} instance(s), total duration at most {total / 3600:.0f} h")
    journal = os.path.join(_abs(DOSSIER), f'runs_{TOPOLOGIE}.jsonl')
    enr = campagne(instances, METHODES, GRAINES, journal, fraction)
    invalides = [r for r in enr if not r['valide']]
    print(f"\n  {len(enr)} run(s) logged, {len(invalides)} infeasible solution(s).")
    print(f"  Analysis: run analyse_campaign.py on {journal}")
    return 0 if not invalides else 1


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
