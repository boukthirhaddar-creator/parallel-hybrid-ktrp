#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONFIRMATORY REPLICATION, STUDY 4b: TRANSFER OF THE LEARNED STATE TO FRESH
SOLVES. Run with: python run_replication.py

Implements Amendment 1 to the protocol of Study 4, written after the
primary stream and before any replication run. Three arms: A2, a fresh
uniform control; B2, the final state of the primary arm B; C2, the final
state of the primary arm C. Six rounds, fresh seeds 4501 + 40 i + r.
Resumes where it stopped.
"""
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if ICI not in sys.path:
    sys.path.insert(0, ICI)

import run_stream_study as E      # noqa: E402

# =============================================================================
#                              CONFIGURATION (amendement 1)
# =============================================================================
TOURS_REPLI = 6
BRAS_REPLI = ['A2', 'B2', 'C2']
GRAINES_REPLI = {'repli': lambda i, r: 4501 + 40 * i + r}
FICHIER_REPLI = os.path.join('logs', 'study4', 'replication.jsonl')


def phase_replication() -> None:
    print("=" * 84, flush=True)
    print("STUDY 4b (amendment 1): TRANSFER OF THE LEARNED STATE TO FRESH SEEDS",
          flush=True)
    print("=" * 84, flush=True)
    ctx = mp.get_context('spawn')
    instances = E.charger_liste(E.FICHIER_INSTANCES)
    fichier_priors = E._abs(os.path.join(E.DOSSIER, 'priors.json'))
    priors = json.load(open(fichier_priors, encoding='utf-8'))

    # Final states of arms B and C, rebuilt from the stream log
    etats_finaux = E.restaurer_banditis(E._abs(os.path.join(E.DOSSIER, 'stream.jsonl')),
                                        priors, instances)
    banditis: Dict[str, E.BanditiGaussien] = {
        'B2': etats_finaux['B'], 'C2': etats_finaux['C']}
    for b, bi in banditis.items():
        bi.rng = random.Random({'B2': 220, 'C2': 330}[b])

    fichier = E._abs(FICHIER_REPLI)
    noms = [os.path.splitext(os.path.basename(c))[0] for c in instances]
    for i, (chemin, nom) in enumerate(zip(instances, noms)):
        faits = {(r['bras'], r['tour']) for r in E.journal(fichier)
                 if r['instance'] == nom}
        a_faire = [(r, b) for r in range(TOURS_REPLI) for b in BRAS_REPLI
                   if (b, r) not in faits]
        if not a_faire:
            continue
        K, n, d = E.lire_instance(chemin)
        T = E.duree_limite(n - 1)
        print(f"\n  {nom} : {n - 1} clients, K = {K}, limite {T:.0f} s, "
              f"{len(a_faire)} exécution(s) restante(s)", flush=True)
        with ProcessPoolExecutor(E.PROCESSUS * 2, mp_context=ctx,
                                 initializer=E.W.initialiser,
                                 initargs=(d, K, n, False, True)) as pool:
            E.W.latence([[0, 0]], d, True) if False else \
                pool.submit(E.W.latence, [[0, 0]], d, True).result()
            depart = E.plus_proche_voisin(d, K)
            for idx in range(0, len(a_faire), 2):
                lot = a_faire[idx:idx + 2]
                decisions = []
                for (r, b) in lot:
                    if b == 'A2':
                        cand = 'U'
                    else:
                        cand = E.NOMS_CANDIDATS[banditis[b].choisir()]
                    decisions.append((r, b, cand))
                resultats = []

                def une_seule(decision):
                    (r, b, cand) = decision
                    probas = E.CANDIDATS[cand] if cand != 'U' else None
                    res = E.executer_recherche(pool, K, n, d, False, depart, T,
                                               probas, GRAINES_REPLI['repli'](i, r))
                    return E.preparer_enr(res, nom, K, n, d, False, T,
                                           GRAINES_REPLI['repli'](i, r), 'repli',
                                           b, r, cand)
                with ThreadPoolExecutor(len(decisions)) as tex:
                    resultats = list(tex.map(une_seule, decisions))
                E.enregistrer(fichier, resultats)
                for (r, b, cand), enr in zip(decisions, resultats):
                    if b != 'A2' and enr['valide']:
                        banditis[b].maj(
                            E.NOMS_CANDIDATS.index(cand),
                            E.recompense_de(enr, chemin, priors['normalisation']))
    print("\n  Final bandit states of the replication:", flush=True)
    for b in sorted(banditis):
        et = banditis[b].etat()
        nom_meilleur = E.NOMS_CANDIDATS[max(range(6), key=lambda k: et['mu'][k])]
        print(f"    arm {b}: highest-scoring candidate {nom_meilleur} "
              f"(mu = {[round(x, 5) for x in et['mu']]}, n = {et['n']})", flush=True)


if __name__ == '__main__':
    os.makedirs(E._abs(E.DOSSIER), exist_ok=True)
    phase_replication()
    print("\nReplication complete. Analysis: python analyse_replication.py", flush=True)
