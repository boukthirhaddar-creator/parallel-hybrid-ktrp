#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRE-SPECIFIED STUDY 4: LEARNING ACROSS A STREAM OF REPEATED SOLVES, AND THE
TRANSFER OF PRIORS. Run with: python run_stream_study.py

The protocol is stated in PROTOCOL_STUDY4.md, frozen before any run of this
study. This script implements it and nothing else. It resumes where it
stopped, every completed run being appended to the journals and the states of
the learners being reconstructed from the journals on start-up.

Phases, in order.
  Phase 0  Pilot. Four uniform runs per instance, seeds 5001 + 40 i + p. The
           pilot normalises the reward and provides the same-family prior.
  Phase 1  Freezing of the priors into logs/study4/priors.json, timestamped.
           Arm B: flat prior. Arm C: legacy corpus of the training family.
           Arm D: pilot campaign. Arm A plays the uniform mixture and carries
           no learner.
  Phase 2  Streams. Four arms, eight rounds, eleven instances, instance-major
           order, arms dispatched in fixed pairs of different arms, both
           decisions taken before either outcome is observed, so that the
           sequence of decisions is deterministic given the sequence of
           outcomes.
  Phase 3  Analysis: analyse_stream_study.py, a separate script.

Every returned solution is checked for feasibility and its objective
recomputed independently before it enters any journal.
"""
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if ICI not in sys.path:
    sys.path.insert(0, ICI)

import ktrp_worker as W            # noqa: E402
import ktrp_coordinator as C        # noqa: E402

# =============================================================================
#                              CONFIGURATION
# Fixed by PROTOCOL_STUDY4.md before any run of this study.
# =============================================================================
FICHIER_INSTANCES = "instances_test_transfer.txt"
PROCESSUS = 3
SECONDES_PAR_CLIENT = 0.6
DUREE_MIN_S = 60.0
DUREE_MAX_S = 90.0
N_EPOQUES = 30
GRAINES_PILOTE = 4                    # P = 4 pilot runs per instance
TOURS = 8                             # R = 8 rounds per arm
FRACTIONS_RELEVES = (0.1, 0.25, 0.5, 1.0)
DOSSIER = "logs/study4"

SIGMA = 0.003                         # known reward scale, protocol
KAPPA_PRIOR = 12                      # pseudo-counts of arms C and D
MU0 = -1.0                            # common prior mean
RIDGE = 0.25                          # ridge of the prior regressions
GRAINES = {                           # seed schemes, protocol
    'pilote': lambda i, p: 5001 + 40 * i + p,
    'flux': lambda i, r: 4001 + 40 * i + r,
}
BRAS = ['A', 'B', 'C', 'D']

# Candidate mixtures: U uniform, Gk boosts grid tenure k by a factor of two.
NOMS_CANDIDATS = ['U', 'G3', 'G6', 'G12', 'G20', 'G30']
CANDIDATS: Dict[str, List[float]] = {'U': [1.0 / 5.0] * 5}
for _k, _frac in enumerate(['3', '6', '12', '20', '30']):
    _p = [1.0 / 6.0] * 5
    _p[_k] = 2.0 / 6.0
    CANDIDATS['G' + _frac] = _p
assert set(NOMS_CANDIDATS) == set(CANDIDATS)


# =============================================================================
#                              MACHINERIE COMMUNE
# =============================================================================
def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ICI, p)


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


def duree_limite(n_clients: int) -> float:
    return float(min(DUREE_MAX_S, max(DUREE_MIN_S, SECONDES_PAR_CLIENT * n_clients)))


def journal(chemin: str) -> List[Dict[str, Any]]:
    """Valid JSON lines of a log; a line truncated by an
    interruption est ignorée, l'exécution correspondante étant absente du
    log, and the corresponding solve is redone on resumption."""
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


_DEPARTS: Dict[str, float] = {}


def depart_instance(chemin: str, nom: str) -> float:
    """Latency of the deterministic construction, computed once."""
    if nom not in _DEPARTS:
        K, n, d = lire_instance(chemin)
        _DEPARTS[nom] = W.latence(plus_proche_voisin(d, K), d, False)
    return _DEPARTS[nom]


def materiel() -> Dict[str, Any]:
    try:
        import cpuinfo  # type: ignore
        proc = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
    except Exception:
        proc = platform.processor() or 'inconnu'
    return {'horodatage': datetime.datetime.now().isoformat(timespec='seconds'),
            'machine': platform.node(), 'systeme': platform.system(),
            'version': platform.version(), 'processeur': proc,
            'logique': os.cpu_count(), 'python': platform.python_version(),
            'remarque': "cœurs mis à disposition du processus Python"}


class BanditiGaussien:
    """Thompson sampler over the candidates, Gaussian posteriors, no forgetting."""

    def __init__(self, mu: List[float], n: List[float], sigma: float = SIGMA,
                 rng: Optional[random.Random] = None):
        self.mu, self.n, self.sigma = list(mu), list(n), sigma
        self.rng = rng or random.Random(0)

    def choisir(self) -> int:
        tirages = [self.rng.gauss(self.mu[k], self.sigma / math.sqrt(self.n[k]))
                   for k in range(len(self.mu))]
        return max(range(len(tirages)), key=tirages.__getitem__)

    def maj(self, k: int, recompense: float) -> None:
        self.mu[k] = (self.n[k] * self.mu[k] + recompense) / (self.n[k] + 1.0)
        self.n[k] += 1.0

    def etat(self) -> Dict[str, Any]:
        return {'mu': [round(x, 8) for x in self.mu],
                'n': [round(x, 3) for x in self.n]}


def mixture_de(run: Dict[str, Any]) -> Optional[List[float]]:
    """Mixture actually played by a logged run, from its epoch detail."""
    opts = C.options_duree_instance(run['n_clients'])
    frac = [0.0] * 5
    for e in run.get('detail_epoques') or []:
        for j, v in enumerate(opts):
            if v == e['duree']:
                frac[j] += 1
                break
    t = sum(frac)
    return [f / t for f in frac] if t else None


def resoudre(A: List[List[float]], b: List[float]) -> List[float]:
    """Gauss-Jordan with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            continue
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and abs(M[r][c]) > 1e-15:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def regression_mixture(runs: List[Dict[str, Any]], ridge: float = RIDGE
                        ) -> Optional[List[float]]:
    """gap_pp = instance effect + w . mixture, fitted by ridge OLS.

    The gap of each run is taken relative to the best final latency observed
    on its instance within the corpus given. Returns the five weights w.
    """
    best: Dict[str, float] = {}
    for r in runs:
        best[r['instance']] = min(best.get(r['instance'], math.inf), r['latence'])
    obs = []
    for r in runs:
        m = mixture_de(r)
        if m is None:
            continue
        gap = (r['latence'] - best[r['instance']]) / best[r['instance']] * 100.0
        obs.append((r['instance'], m, gap))
    if len(obs) < 20:
        return None
    instances = sorted({i for i, _, _ in obs})
    idx = {inst: j for j, inst in enumerate(instances)}
    p = len(instances) + 5
    if len(obs) <= p:
        return None
    A = [[0.0] * p for _ in range(p)]
    bvec = [0.0] * p
    for inst, m, gap in obs:
        ligne = [0.0] * p
        ligne[idx[inst]] = 1.0
        for k in range(5):
            ligne[len(instances) + k] = m[k]
        for a in range(p):
            bvec[a] += ligne[a] * gap
            for c in range(p):
                A[a][c] += ligne[a] * ligne[c]
    for a in range(p):
        A[a][a] += ridge
    return resoudre(A, bvec)[len(instances):]


def valeurs_candidats(w: Optional[Sequence[float]]) -> Dict[str, float]:
    """Predicted advantage of each candidate over U, in percentage points.

    Positive means the candidate is predicted to end with a lower gap than
    the uniform mixture, hence a higher reward.
    """
    if w is None:
        return {c: 0.0 for c in NOMS_CANDIDATS}
    vu = sum(w[k] * CANDIDATS['U'][k] for k in range(5))
    out = {}
    for nom in NOMS_CANDIDATS:
        pi = CANDIDATS[nom]
        out[nom] = vu - sum(w[k] * pi[k] for k in range(5))
    return out


def recompense_de(enr: Dict[str, Any], chemin: str, normalisation: Dict[str, float]
                  ) -> float:
    """Reward of a stream run, protocol: minus final latency over construction
    latency, divided by the pilot mean ratio of the instance."""
    nom = enr['instance']
    return -(enr['latence'] / depart_instance(chemin, nom)) / normalisation[nom]


def candidat_de(enr: Dict[str, Any]) -> str:
    """Candidate played by a logged stream run, from its probability vector."""
    probas = enr.get('probabilites')
    if not probas:
        return 'U'
    for nom in NOMS_CANDIDATS:
        if all(abs(a - b) < 1e-9 for a, b in zip(probas, CANDIDATS[nom])):
            return nom
    return '?'


# =============================================================================
#                              EXÉCUTION D'UNE RECHERCHE
# =============================================================================
def executer_recherche(pool, K: int, n: int, d, circuit: bool, depart, T: float,
                       probabilites: Optional[List[float]], graine: int
                       ) -> Dict[str, Any]:
    return C.recherche_cooperative(
        pool, PROCESSUS, T, depart, d, circuit, politique='melange',
        probabilites=probabilites, graine=graine, n_epoques=N_EPOQUES,
        options_duree=C.options_duree_instance(n - 1),
        instants_releves=[round(f * T, 3) for f in FRACTIONS_RELEVES],
        journaliser_epoques=True)


def preparer_enr(r: Dict[str, Any], nom: str, K: int, n: int, d, circuit: bool,
                 T: float, graine: int, phase: str, bras: Optional[str],
                 tour: Optional[int], candidat: Optional[str]) -> Dict[str, Any]:
    valide = W.est_valide(r['routes'], K, n, True)
    recalcul = W.latence(r['routes'], d, circuit)
    return {'phase': phase, 'instance': nom, 'n_clients': n - 1, 'K': K,
            'bras': bras, 'tour': tour, 'candidat': candidat, 'graine': graine,
            'limite_s': T, 'latence': r['latence'],
            'valide': bool(valide and abs(recalcul - r['latence']) < 1e-6),
            'temps_meilleure_s': r['temps_meilleure_s'],
            'evaluations': r['evaluations'], 'releves': r['releves'],
            'choix_durees': r['choix_durees'], 'detail_epoques': r['detail_epoques'],
            'probabilites': r['probabilites'], 'routes': r['routes']}


def enregistrer(fichier: str, enrs: List[Dict[str, Any]]) -> None:
    with open(fichier, 'a', encoding='utf-8') as f:
        for enr in enrs:
            f.write(json.dumps(enr, ensure_ascii=False) + '\n')
    for enr in enrs:
        print(f"    {enr['phase']:6s} {enr['instance']:22s} "
              f"tour {enr.get('tour') if enr.get('tour') is not None else '-'} "
              f"bras {enr.get('bras') or '-'} cand {enr.get('candidat') or '-'} "
              f"latence {enr['latence']:11.2f}  "
              f"{'OK' if enr['valide'] else 'INFEASIBLE'}", flush=True)


# =============================================================================
#                              PHASE 0 : PILOTE
# =============================================================================
def phase_pilote(instances: List[str], fichier: str) -> List[Dict[str, Any]]:
    print("=" * 84, flush=True)
    print("PHASE 0: PILOT CAMPAIGN (uniform mixture, 4 seeds per instance)",
          flush=True)
    print("=" * 84, flush=True)
    ctx = mp.get_context('spawn')
    for i, chemin in enumerate(instances):
        nom = os.path.splitext(os.path.basename(chemin))[0]
        faits = {(r['instance'], r['graine']) for r in journal(fichier)
                 if r['phase'] == 'pilote'}
        a_faire = [p for p in range(GRAINES_PILOTE)
                   if (nom, GRAINES['pilote'](i, p)) not in faits]
        if not a_faire:
            continue
        K, n, d = lire_instance(chemin)
        T = duree_limite(n - 1)
        print(f"\n  {nom} : {n - 1} clients, K = {K}, limite {T:.0f} s, "
              f"{len(a_faire)} exécution(s)", flush=True)
        with ProcessPoolExecutor(PROCESSUS * 2, mp_context=ctx, initializer=W.initialiser,
                                 initargs=(d, K, n, False, True)) as pool:
            pool.submit(W.latence, [[0, 0]], d, True).result()
            depart = plus_proche_voisin(d, K)
            for idx in range(0, len(a_faire), 2):
                lot = a_faire[idx:idx + 2]

                def une_seule(p):
                    r = executer_recherche(pool, K, n, d, False, depart, T, None,
                                           GRAINES['pilote'](i, p))
                    return preparer_enr(r, nom, K, n, d, False, T,
                                       GRAINES['pilote'](i, p), 'pilote', None, None, None)
                with ThreadPoolExecutor(len(lot)) as tex:
                    enrs = list(tex.map(une_seule, lot))
                enregistrer(fichier, enrs)
    return journal(fichier)


# =============================================================================
#                              PHASE 1 : PRIORS GELÉS
# =============================================================================
def phase_priors(instances: List[str], pilote: List[Dict[str, Any]],
                 fichier_priors: str) -> Dict[str, Any]:
    print("\n" + "=" * 84, flush=True)
    print("PHASE 1: FREEZING THE PRIORS", flush=True)
    print("=" * 84, flush=True)
    if os.path.exists(fichier_priors):
        priors = json.load(open(fichier_priors, encoding='utf-8'))
        print(f"  Priors already frozen on {priors['date_gel']}", flush=True)
        return priors

    noms = [os.path.splitext(os.path.basename(c))[0] for c in instances]
    # Per-instance normalisation: pilot mean of the latency-to-start ratio
    normalisation: Dict[str, float] = {}
    for chemin, nom in zip(instances, noms):
        rs = [r['latence'] / depart_instance(chemin, nom)
              for r in pilote
              if r['phase'] == 'pilote' and r['instance'] == nom and r['valide']]
        normalisation[nom] = st.fmean(rs) if rs else 1.0

    # Prior C: corpus inherited from the training family (Study 3)
    corpus_herite: List[Dict[str, Any]] = []
    for r in journal(_abs(os.path.join('logs', 'transfer', 'training.jsonl'))):
        if r.get('valide'):
            corpus_herite.append(r)
    inst_heritees = {r['instance'] for r in corpus_herite}
    for r in journal(_abs(os.path.join('logs', 'campaign', 'runs_path.jsonl'))):
        if r.get('methode') == 'HASARD' and r['instance'] in inst_heritees \
                and r.get('valide'):
            corpus_herite.append(r)
    w_xf = regression_mixture(corpus_herite)

    # Prior D: pilot campaign, deployment family
    pilote_valide = [r for r in pilote if r['phase'] == 'pilote' and r['valide']]
    w_sf = regression_mixture(pilote_valide)

    valeurs_xf = valeurs_candidats(w_xf)
    valeurs_sf = valeurs_candidats(w_sf)
    # A predicted advantage of x percentage points raises the reward by x/100,
    # the reward being centred near -1
    priors = {
        'date_gel': datetime.datetime.now().isoformat(timespec='seconds'),
        'sigma': SIGMA, 'kappa': KAPPA_PRIOR, 'mu0': MU0, 'ridge': RIDGE,
        'normalisation': {k: round(v, 6) for k, v in normalisation.items()},
        'C': {'mu': [round(MU0 + valeurs_xf[c] / 100.0, 8) for c in NOMS_CANDIDATS],
              'n': [float(KAPPA_PRIOR)] * 6,
              'valeurs_predites_pp': {c: round(valeurs_xf[c], 4) for c in NOMS_CANDIDATS},
              'corpus': len(corpus_herite)},
        'D': {'mu': [round(MU0 + valeurs_sf[c] / 100.0, 8) for c in NOMS_CANDIDATS],
              'n': [float(KAPPA_PRIOR)] * 6,
              'valeurs_predites_pp': {c: round(valeurs_sf[c], 4) for c in NOMS_CANDIDATS},
              'corpus': len(pilote_valide)},
        'B': {'mu': [MU0] * 6, 'n': [1.0] * 6},
    }
    json.dump(priors, open(fichier_priors, 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(f"  Legacy corpus (C): {len(corpus_herite)} runs, "
          f"poids w = {[round(x, 3) for x in (w_xf or [])]}", flush=True)
    print(f"  Pilot (D): {len(pilote_valide)} runs, "
          f"poids w = {[round(x, 3) for x in (w_sf or [])]}", flush=True)
    print(f"  Predicted advantages C (pp): "
          f"{ {c: round(v, 3) for c, v in valeurs_xf.items()} }", flush=True)
    print(f"  Predicted advantages D (pp): "
          f"{ {c: round(v, 3) for c, v in valeurs_sf.items()} }", flush=True)
    print(f"  Priors frozen into {fichier_priors}", flush=True)
    return priors


# =============================================================================
#                              PHASE 2 : FLUX
# =============================================================================
def restaurer_banditis(fichier_flux: str, priors: Dict[str, Any],
                       instances: List[str]) -> Dict[str, BanditiGaussien]:
    """Replay the updates from the log, in the order in which they were written."""
    banditis = {b: BanditiGaussien(priors[b]['mu'], priors[b]['n'], SIGMA,
                                   random.Random({'B': 22, 'C': 33,
                                                  'D': 44}[b]))
                for b in BRAS if b in priors}
    chemins = {os.path.splitext(os.path.basename(c))[0]: c for c in instances}
    for enr in journal(fichier_flux):
        if enr['phase'] != 'flux' or enr['bras'] == 'A' or not enr['valide']:
            continue
        cand = candidat_de(enr)
        if cand in NOMS_CANDIDATS:
            banditis[enr['bras']].maj(
                NOMS_CANDIDATS.index(cand),
                recompense_de(enr, chemins[enr['instance']],
                              priors['normalisation']))
    return banditis


def phase_flux(instances: List[str], priors: Dict[str, Any], fichier: str) -> None:
    print("\n" + "=" * 84, flush=True)
    print("PHASE 2: STREAM (4 arms, 8 rounds, 11 instances)", flush=True)
    print("=" * 84, flush=True)
    ctx = mp.get_context('spawn')
    banditis = restaurer_banditis(fichier, priors, instances)
    deja = sum(1 for r in journal(fichier) if r['phase'] == 'flux')
    if deja:
        print(f"  {deja} runs already logged; states replayed.", flush=True)

    f_etats = _abs(os.path.join(DOSSIER, 'bandit_states.json'))
    noms = [os.path.splitext(os.path.basename(c))[0] for c in instances]
    for i, (chemin, nom) in enumerate(zip(instances, noms)):
        faits = {(r['bras'], r['tour']) for r in journal(fichier)
                 if r['phase'] == 'flux' and r['instance'] == nom}
        a_faire = [(r, b) for r in range(TOURS) for b in BRAS if (b, r) not in faits]
        if not a_faire:
            continue
        K, n, d = lire_instance(chemin)
        T = duree_limite(n - 1)
        print(f"\n  {nom} : {n - 1} clients, K = {K}, limite {T:.0f} s, "
              f"{len(a_faire)} exécution(s) restante(s)", flush=True)
        with ProcessPoolExecutor(PROCESSUS * 2, mp_context=ctx, initializer=W.initialiser,
                                 initargs=(d, K, n, False, True)) as pool:
            pool.submit(W.latence, [[0, 0]], d, True).result()
            depart = plus_proche_voisin(d, K)
            for idx in range(0, len(a_faire), 2):
                lot = a_faire[idx:idx + 2]
                # Decisions taken before any outcome of the batch is observed
                decisions = []
                for (r, b) in lot:
                    if b == 'A':
                        cand = 'U'
                    else:
                        cand = NOMS_CANDIDATS[banditis[b].choisir()]
                    decisions.append((r, b, cand))
                resultats = []

                def une_seule(decision):
                    (r, b, cand) = decision
                    probas = CANDIDATS[cand] if cand != 'U' else None
                    res = executer_recherche(pool, K, n, d, False, depart, T, probas,
                                             GRAINES['flux'](i, r))
                    return preparer_enr(res, nom, K, n, d, False, T,
                                       GRAINES['flux'](i, r), 'flux', b, r, cand)
                with ThreadPoolExecutor(len(decisions)) as tex:
                    resultats = list(tex.map(une_seule, decisions))
                # Write, then update, in batch order
                enregistrer(fichier, resultats)
                for (r, b, cand), enr in zip(decisions, resultats):
                    if b != 'A' and enr['valide']:
                        banditis[b].maj(
                            NOMS_CANDIDATS.index(cand),
                            recompense_de(enr, chemin, priors['normalisation']))
                json.dump({b: banditis[b].etat() for b in banditis},
                          open(f_etats, 'w', encoding='utf-8'), indent=1)
    print("\n  Final bandit states:", flush=True)
    for b in sorted(banditis):
        et = banditis[b].etat()
        nom_meilleur = NOMS_CANDIDATS[max(range(6), key=lambda k: et['mu'][k])]
        print(f"    arm {b}: highest-scoring candidate {nom_meilleur} "
              f"(mu = {[round(x, 5) for x in et['mu']]}, n = {et['n']})", flush=True)


# =============================================================================
#                              MAIN
# =============================================================================
def main() -> int:
    os.makedirs(_abs(DOSSIER), exist_ok=True)
    f_materiel = _abs(os.path.join(DOSSIER, 'hardware.json'))
    if not os.path.exists(f_materiel):
        json.dump(materiel(), open(f_materiel, 'w', encoding='utf-8'), indent=1)
    instances = charger_liste(FICHIER_INSTANCES)
    print(f"ÉTUDE 4 PRÉ-SPÉCIFIÉE : {len(instances)} instances X, "
          f"budget {SECONDES_PAR_CLIENT} s/client ({DUREE_MIN_S:.0f}-{DUREE_MAX_S:.0f} s), "
          f"{N_EPOQUES} époques, {PROCESSUS} agents", flush=True)

    fichier_pilote = _abs(os.path.join(DOSSIER, 'pilot.jsonl'))
    fichier_flux = _abs(os.path.join(DOSSIER, 'stream.jsonl'))
    fichier_priors = _abs(os.path.join(DOSSIER, 'priors.json'))

    pilote = phase_pilote(instances, fichier_pilote)
    attendus = len(instances) * GRAINES_PILOTE
    faits = len([r for r in pilote if r['phase'] == 'pilote'])
    if faits < attendus or not all(r['valide'] for r in pilote if r['phase'] == 'pilote'):
        print(f"  Pilote incomplet ou invalide ({faits}/{attendus}) : relancer.",
              flush=True)
        return 1

    priors = phase_priors(instances, pilote, fichier_priors)
    phase_flux(instances, priors, fichier_flux)
    print("\nStudy 4 complete. Analysis: python analyse_stream_study.py", flush=True)
    return 0


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
