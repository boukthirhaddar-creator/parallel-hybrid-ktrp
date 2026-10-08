#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRE-SPECIFIED STUDY: CROSS-INSTANCE TRANSFER LEARNING.
Run with: python run_transfer_study.py

Question
--------
Does a tenure control policy, learned from the collective experience of the agents
accumulated over training instances and then transferred to unseen instances
according to their signature, outperform random tenure variation (HASARD)?

Motivation
----------
The main campaign established that online learning, limited to about thirty
decisions per agent and per run, lacks data and behaves like random variation.
Learning is therefore moved to the level at which data accumulate, namely across
runs and across instances. This is the knowledge transfer the paper examines.

Protocol, fixed before any run
------------------------------
Instance signature: mean route length L = number of customers / K.
Classes, fixed in advance: L < 6 (short), 6 <= L < 10 (medium), L >= 10 (long).
Tenure windows, as indices into the grid 3, 6, 12, 20 and 30 % of the customers:
    W0 = all tenures (equivalent to HASARD), W1 = 3-6-12 %, W2 = 6-12-20 %,
    W3 = 12-20-30 %. At each epoch every agent draws its tenure uniformly from
    the window.

Phase A  Training. The four windows on the training instances, with three seeds.
         These are the instances of the path campaign, less those on which all
         fifty runs of that campaign returned the same value, since they carry no
         information.
Phase B  Learning. For each class, the window of lowest mean relative gap on the
         training instances of that class. W0 is retained unless another window
         improves upon it by at least 0.02 percentage points. The policy is
         frozen into transfer_policy.json, timestamped, and never recomputed.
         If W0 is retained for all three classes, the study stops: the hypothesis
         is not supported.
Phase C  Test. TRANSFER against HASARD, ten new seeds, on the X-series instances,
         path variant, which were never used before.
Phase D  Decision rule. Two-sided paired Wilcoxon signed-rank test on the relative
         gap to the best value observed on the instance, at the 5 % level.
         Transfer learning is established if TRANSFER outperforms HASARD. The
         outcome is reported whatever it is.
"""
import datetime
import json
import math
import multiprocessing as mp
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if ICI not in sys.path:
    sys.path.insert(0, ICI)

import ktrp_worker as W            # noqa: E402
import ktrp_coordinator as C        # noqa: E402
from analyse_campaign import wilcoxon  # noqa: E402

# =============================================================================
#                              CONFIGURATION
# =============================================================================
TOPOLOGIE = "path"
FICHIER_ENTRAINEMENT = "instances_training_transfer.txt"
FICHIER_TEST = "instances_test_transfer.txt"
GRAINES_ENTRAINEMENT = [1009, 1013, 1019]
GRAINES_TEST = [3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079]
PROCESSUS = 3
SECONDES_PAR_CLIENT = 1.2
DUREE_MIN_S = 60
DUREE_MAX_S = 600
N_EPOQUES = 30
FRACTIONS_RELEVES = (0.1, 0.25, 0.5, 1.0)
DOSSIER = "run_transfer_study"
# =============================================================================

FENETRES = {'W0': [0, 1, 2, 3, 4], 'W1': [0, 1, 2], 'W2': [1, 2, 3], 'W3': [2, 3, 4]}
BORNES_CLASSES = (6.0, 10.0)
NOMS_CLASSES = ('courtes', 'moyennes', 'longues')
MARGE = 0.02


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ICI, p)


def classe(L: float) -> int:
    return 0 if L < BORNES_CLASSES[0] else (1 if L < BORNES_CLASSES[1] else 2)


def lire_instance(chemin: str):
    lignes = [l.strip() for l in open(chemin, encoding='utf-8', errors='replace') if l.strip()]
    K, n = int(lignes[0]), int(lignes[1])
    return K, n, [[float(x) for x in l.split()] for l in lignes[2:2 + n]]


def plus_proche_voisin(d, K):
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
    base = os.path.dirname(chemin)
    sortie = []
    for l in open(chemin, encoding='utf-8'):
        l = l.strip()
        if l and not l.startswith('#'):
            p = l if os.path.isabs(l) else os.path.join(base, l)
            if not os.path.exists(p):
                raise SystemExit(f"Instance introuvable : {l}")
            sortie.append(p)
    return sortie


def duree_limite(n_clients: int) -> float:
    return float(min(DUREE_MAX_S, max(DUREE_MIN_S, SECONDES_PAR_CLIENT * n_clients)))


def journal(chemin: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chemin):
        return []
    return [json.loads(l) for l in open(chemin, encoding='utf-8') if l.strip()]


def executer_liste(instances: List[str], plans: Dict[str, List[str]], graines: List[int],
                   fichier_journal: str) -> List[Dict[str, Any]]:
    """plans: for each instance, the list of labels to run (W0..W3, HASARD, TRANSFERT:Wk)."""
    faits = {(r['instance'], r['etiquette'], r['graine']) for r in journal(fichier_journal)}
    ctx = mp.get_context('spawn')
    circuit = TOPOLOGIE == 'circuit'
    for chemin in instances:
        nom = os.path.splitext(os.path.basename(chemin))[0]
        a_faire = [(e, g) for g in graines for e in plans.get(chemin, []) if (nom, e, g) not in faits]
        if not a_faire:
            continue
        K, n, d = lire_instance(chemin)
        T = duree_limite(n - 1)
        L = (n - 1) / K
        print(f"\n  {nom} : {n - 1} clients, K = {K}, L = {L:.2f} ({NOMS_CLASSES[classe(L)]}), "
              f"limite {T:.0f} s, {len(a_faire)} execution(s)", flush=True)
        with ProcessPoolExecutor(PROCESSUS, mp_context=ctx, initializer=W.initialiser,
                                 initargs=(d, K, n, circuit, True)) as pool:
            pool.submit(W.latence, [[0, 0]], d, True).result()
            depart = plus_proche_voisin(d, K)
            for etiquette, g in a_faire:
                if etiquette == 'HASARD':
                    pol, fen = 'hasard', None
                else:
                    pol, fen = 'fenetre', FENETRES[etiquette.split(':')[-1]]
                r = C.recherche_cooperative(pool, PROCESSUS, T, depart, d, circuit, politique=pol,
                                            fenetre=fen, graine=g, n_epoques=N_EPOQUES,
                                            options_duree=C.options_duree_instance(n - 1),
                                            instants_releves=[round(f * T, 3) for f in FRACTIONS_RELEVES],
                                            journaliser_epoques=True)
                valide = W.est_valide(r['routes'], K, n, True)
                recalcul = W.latence(r['routes'], d, circuit)
                enr = {'instance': nom, 'n_clients': n - 1, 'K': K, 'L': L, 'classe': classe(L),
                       'etiquette': etiquette, 'graine': g, 'limite_s': T,
                       'latence': r['latence'],
                       'valide': bool(valide and abs(recalcul - r['latence']) < 1e-6),
                       'temps_meilleure_s': r['temps_meilleure_s'], 'evaluations': r['evaluations'],
                       'releves': r['releves'], 'choix_durees': r['choix_durees'],
                       'detail_epoques': r['detail_epoques'], 'routes': r['routes']}
                with open(fichier_journal, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(enr, ensure_ascii=False) + '\n')
                print(f"    {etiquette:12s} seed {g}  latency {enr['latence']:11.2f}  reached at "
                      f"{enr['temps_meilleure_s']:7.1f} s  {'' if enr['valide'] else 'INFEASIBLE'}",
                      flush=True)
    return journal(fichier_journal)


def apprendre(enr: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase B: one window per class, from the mean relative gap on the training set."""
    meilleure = {}
    for r in enr:
        meilleure[r['instance']] = min(meilleure.get(r['instance'], math.inf), r['latence'])
    tableau = {}
    regle = {}
    for c, nom_c in enumerate(NOMS_CLASSES):
        insts = sorted({r['instance'] for r in enr if r['classe'] == c})
        moyennes = {}
        for w in FENETRES:
            v = [(r['latence'] - meilleure[r['instance']]) / meilleure[r['instance']] * 100
                 for r in enr if r['classe'] == c and r['etiquette'] == w]
            moyennes[w] = st.fmean(v) if v else math.inf
        meilleure_w = min(FENETRES, key=lambda w: (moyennes[w], w))
        choix = meilleure_w if moyennes['W0'] - moyennes[meilleure_w] >= MARGE else 'W0'
        regle[nom_c] = choix
        tableau[nom_c] = {'instances': insts, 'ecart_moyen': {w: round(m, 4) for w, m in moyennes.items()}}
    return {'regle': regle, 'tableau': tableau}


def main() -> int:
    os.makedirs(_abs(DOSSIER), exist_ok=True)
    print("=" * 84)
    print("PRE-SPECIFIED STUDY: CROSS-INSTANCE TRANSFER LEARNING")
    print("=" * 84)
    entr = charger_liste(FICHIER_ENTRAINEMENT)
    test = charger_liste(FICHIER_TEST)
    communes = set(map(os.path.basename, entr)) & set(map(os.path.basename, test))
    if communes:
        print(f"  ERROR: instances common to training and test: {sorted(communes)}")
        return 1
    f_pol = os.path.join(_abs(DOSSIER), 'transfer_policy.json')

    # --- Phases A et B -----------------------------------------------------------
    if os.path.exists(f_pol):
        politique = json.load(open(f_pol, encoding='utf-8'))
        print(f"\n  Policy frozen on {politique['date_gel']}: {politique['regle']}")
    else:
        print(f"\nPHASE A : ENTRAINEMENT, {len(entr)} instance(s), fenetres {list(FENETRES)}, "
              f"graines {GRAINES_ENTRAINEMENT}")
        enr = executer_liste(entr, {c: list(FENETRES) for c in entr}, GRAINES_ENTRAINEMENT,
                             os.path.join(_abs(DOSSIER), 'training.jsonl'))
        attendus = len(entr) * len(FENETRES) * len(GRAINES_ENTRAINEMENT)
        if len(enr) < attendus or not all(r['valide'] for r in enr):
            print(f"  Training incomplete or invalid ({len(enr)}/{attendus}): run again.")
            return 1
        print("\nPHASE B : APPRENTISSAGE")
        appris = apprendre(enr)
        politique = {'date_gel': datetime.datetime.now().isoformat(timespec='seconds'),
                     'bornes_classes': list(BORNES_CLASSES), 'fenetres': FENETRES,
                     'marge': MARGE, 'graines_entrainement': GRAINES_ENTRAINEMENT, **appris}
        json.dump(politique, open(f_pol, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        for c in NOMS_CLASSES:
            print(f"  {c:9s} {politique['tableau'][c]['ecart_moyen']}  ->  {politique['regle'][c]}")
        print(f"  Policy frozen into {f_pol}")

    if all(w == 'W0' for w in politique['regle'].values()):
        print("\n  Window W0 is retained for all three classes: no policy separates itself")
        print("  from random variation on the training instances. The hypothesis is not")
        print("  supported; the study stops here, as the protocol requires.")
        return 0

    # --- Phase C : test --------------------------------------------------------------
    plans = {}
    for chemin in test:
        K, n, _ = lire_instance(chemin)
        w = politique['regle'][NOMS_CLASSES[classe((n - 1) / K)]]
        if w != 'W0':
            plans[chemin] = ['HASARD', f'TRANSFERT:{w}']
    print(f"\nPHASE C: TEST, {len(plans)} of {len(test)} instance(s) concerned, "
          f"{len(GRAINES_TEST)} seeds")
    f_test = os.path.join(_abs(DOSSIER), 'test.jsonl')
    enr = executer_liste(test, plans, GRAINES_TEST, f_test)

    # --- Phase D : regle de decision ----------------------------------------------
    enr = [r for r in enr if r['valide']]
    meilleure = {}
    for r in enr:
        meilleure[r['instance']] = min(meilleure.get(r['instance'], math.inf), r['latence'])
    idx = {}
    for r in enr:
        m = 'TRANSFERT' if r['etiquette'].startswith('TRANSFERT') else 'HASARD'
        idx[(r['instance'], r['graine'], m)] = (r['latence'] - meilleure[r['instance']]) / meilleure[r['instance']] * 100
    cles = sorted({(i, g) for (i, g, m) in idx if m == 'TRANSFERT'} & {(i, g) for (i, g, m) in idx if m == 'HASARD'})
    x = [idx[(i, g, 'TRANSFERT')] for i, g in cles]
    y = [idx[(i, g, 'HASARD')] for i, g in cles]
    print("\n" + "=" * 84)
    print("PHASE D: PRE-SPECIFIED DECISION RULE")
    print("=" * 84)
    for inst in sorted({i for i, _ in cles}):
        a = [idx[(inst, g, 'TRANSFERT')] for i, g in cles if i == inst]
        b = [idx[(inst, g, 'HASARD')] for i, g in cles if i == inst]
        print(f"  {inst:22s} TRANSFERT {st.fmean(a):6.3f} %   HASARD {st.fmean(b):6.3f} %   (n = {len(a)})")
    if not x:
        print("  Aucune paire complete.")
        return 1
    p, _ = wilcoxon(x, y)
    diff = st.fmean([u - v for u, v in zip(x, y)])
    v = sum(u < w - 1e-12 for u, w in zip(x, y))
    d = sum(u > w + 1e-12 for u, w in zip(x, y))
    etabli = diff < 0 and p < 0.05
    print(f"\n  TRANSFER versus RANDOM: {len(x)} pairs, wins/ties/losses {v}/{len(x) - v - d}/{d}, "
          f"mean difference {diff:+.3f} pt, p = {p:.4g}")
    print(f"  Transfer learning established: {'YES' if etabli else 'NO'}")
    json.dump({'paires': len(x), 'victoires': v, 'defaites': d, 'ecart_moyen': diff, 'p': p,
               'etabli': etabli}, open(os.path.join(_abs(DOSSIER), 'verdict.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
