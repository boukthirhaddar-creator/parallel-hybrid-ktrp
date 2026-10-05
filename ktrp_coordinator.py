# -*- coding: utf-8 -*-
"""
COOPERATIVE PARALLEL TABU SEARCH.

Principle
---------
Each agent runs one long tabu trajectory inside its own process. The total budget
is divided into epochs. At the end of each epoch the coordinator decides how every
agent continues. Two families of decision are available:

    trajectory decisions   continue, restart from the best solution held in
                           common, restart from a random elite solution;
    control decisions      tabu tenure applied during the next epoch.

Only control decisions are exercised in the configurations compared in the paper.
Trajectory restarts were measured and found to degrade search quality, because an
interrupted trajectory loses the basin it was exploring and because restarting
several agents from a common solution concentrates the search in a single region.
An agent therefore always continues its own trajectory, and only its tenure varies.
The restart machinery is retained so that this finding can be reproduced.

Policies
--------
    'independante'  fixed tenure, never a restart: independent search;
    'portefeuille'  a different fixed tenure per agent, no learning;
    'hasard'        tenure drawn uniformly at each epoch, no learning;
    'fenetre'       tenure drawn uniformly from a subset of the grid (parameter
                    fenetre); this is the support of the policy transferred from
                    one instance to another in run_transfer_study.py;
    'privee'        each agent learns its own control policy;
    'partagee'      the agents learn one common policy from their experience,
                    pooled in the collective memory;
    'regle'         restart from the best solution after stagnation;
    'apprise'       decision delegated to an external decision maker;
    'melange'       tenure drawn at each epoch from an explicit probability
                    vector over the grid (parameter probabilites); it is the
                    support of the mixture candidates of the stream study.

The learned policy is a contextual Thompson sampler with geometric forgetting. The
context is the number of epochs without personal improvement, truncated at three.
The reward is binary: did the agent improve its own best solution during the epoch?
Forgetting lets the policy follow the change of regime between the early phase of
the search, in which short tenures descend quickly, and the stagnation phase, in
which only longer tenures escape the current basin.

The contrast between 'privee' and 'partagee' isolates the contribution of
cooperation: same learner, same agents, but experience held privately or pooled.
The contrast between 'hasard' and 'privee' isolates the contribution of learning:
same set of tenures, chosen at random or chosen by a policy.

This module depends only on ktrp_worker.
"""
import random
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ktrp_worker as W

POURSUIVRE, REPARTIR, DIVERSIFIER = 0, 1, 2

# Grid of tabu tenures, as a fraction of the number of customers. It is fixed
# before any campaign, running from the default setting of about 3 on instances of
# one hundred customers up to 30 per cent of the customers, in geometric steps.
FRACTIONS_DUREE: Tuple[float, ...] = (0.03, 0.06, 0.12, 0.20, 0.30)

POLITIQUES_PILOTAGE = ('independante', 'portefeuille', 'hasard', 'privee', 'partagee')


def options_duree_instance(n_clients: int, fractions: Sequence[float] = FRACTIONS_DUREE
                           ) -> Tuple[int, ...]:
    """Candidate tabu tenures for an instance with n_clients customers.

    The number of options is kept constant, duplicates included, so that the
    policy keeps the same action dimension from one instance to another.
    """
    return tuple(max(2, int(round(f * n_clients))) for f in fractions)


class PolitiqueThompson:
    """Thompson sampler with forgetting, conditioned on stagnation.

    For each context c and each option k, the probability of success is modelled
    by a Beta(a, b) distribution. On every update of a context, the counts of all
    its options are contracted towards the uniform prior by a factor gamma, and
    the count of the option played is then incremented.
    """

    def __init__(self, n_options: int, n_contextes: int = 4, gamma: float = 0.95,
                 rng: Optional[random.Random] = None):
        self.n_options, self.n_contextes, self.gamma = n_options, n_contextes, gamma
        self.a = [[1.0] * n_options for _ in range(n_contextes)]
        self.b = [[1.0] * n_options for _ in range(n_contextes)]
        self.rng = rng or random.Random(0)
        self.historique: List[Tuple[int, int, int]] = []   # (contexte, option, succes)

    def contexte(self, stagnation: int) -> int:
        return min(max(0, int(stagnation)), self.n_contextes - 1)

    def choisir(self, ctx: int) -> int:
        tirages = [self.rng.betavariate(self.a[ctx][k], self.b[ctx][k])
                   for k in range(self.n_options)]
        return max(range(self.n_options), key=tirages.__getitem__)

    def mettre_a_jour(self, ctx: int, option: int, succes: bool) -> None:
        g = self.gamma
        for k in range(self.n_options):
            self.a[ctx][k] = 1.0 + g * (self.a[ctx][k] - 1.0)
            self.b[ctx][k] = 1.0 + g * (self.b[ctx][k] - 1.0)
        if succes:
            self.a[ctx][option] += 1.0
        else:
            self.b[ctx][option] += 1.0
        self.historique.append((ctx, option, int(bool(succes))))

    def moyennes(self) -> List[List[float]]:
        """Posterior expectation of success, by context and by option."""
        return [[round(self.a[c][k] / (self.a[c][k] + self.b[c][k]), 3)
                 for k in range(self.n_options)] for c in range(self.n_contextes)]


def _aretes_elite(elite: List[Tuple[float, List[List[int]]]]) -> List[Tuple[int, int, int]]:
    """Undirected edges of the elite solutions, with their frequency."""
    compte: Dict[Tuple[int, int], int] = {}
    for _lat, routes in elite:
        for r in routes:
            for i in range(len(r) - 1):
                u, v = r[i], r[i + 1]
                cle = (u, v) if u <= v else (v, u)
                compte[cle] = compte.get(cle, 0) + 1
    return [(u, v, f) for (u, v), f in compte.items()]


def _inserer_elite(elite: List[Tuple[float, List[List[int]]]], lat: float,
                   routes: List[List[int]], k: int) -> None:
    """Insert a solution into the elite pool if it belongs there, without duplicates."""
    cle = tuple(tuple(r) for r in routes)
    for _l, rr in elite:
        if tuple(tuple(r) for r in rr) == cle:
            return
    elite.append((lat, [list(r) for r in routes]))
    elite.sort(key=lambda e: e[0])
    del elite[k:]


def recherche_cooperative(pool, P: int, T: float, depart: List[List[int]],
                          distances: List[List[float]], circuit: bool = True,
                          politique: str = 'independante',
                          decideur: Optional[Callable[[Dict[str, Any]], Tuple[int, int]]] = None,
                          graine: int = 0, n_epoques: int = 30, duree_tabou: int = 3,
                          options_duree: Optional[Tuple[int, ...]] = None,
                          patience_epoques: int = 3, taille_elite: int = 5,
                          facteur_elite: float = 0.0, max_voisins: int = 500,
                          apprendre: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                          gamma_oubli: float = 0.95, n_contextes: int = 4,
                          instants_releves: Sequence[float] = (),
                          fenetre: Optional[Sequence[int]] = None,
                          probabilites: Optional[Sequence[float]] = None,
                          journaliser_epoques: bool = False
                          ) -> Dict[str, Any]:
    """Cooperative search on P processes for T seconds.

    duree_tabou       tenure used by 'independante' and 'regle';
    options_duree     candidate tenures of the control policies; defaults to
                      options_duree_instance(number of customers);
    instants_releves  instants, in seconds, at which the best collective latency
                      is recorded, so that quality against time can be traced
                      from a single run;
    fenetre           indices of the options allowed to the 'fenetre' policy;
                      all options when None, which reproduces 'hasard';
    probabilites      weights of the grid options for the 'melange' policy;
                      uniform when None, which also reproduces 'hasard';
    journaliser_epoques  when true, the output carries the detail of every epoch
                      and every agent: stagnation context, tenure played, success
                      and relative gain. This is the material of the offline
                      learning study and of the descriptive analyses.
    """
    t0 = time.time()
    rng = random.Random(graine)
    delta = T / max(1, n_epoques)
    n_clients = len(distances) - 1
    if options_duree is None:
        options_duree = options_duree_instance(n_clients)
    options_duree = tuple(int(x) for x in options_duree)

    lat_depart = W.latence(depart, distances, circuit)
    courantes = [[list(r) for r in depart] for _ in range(P)]
    lat_courantes = [lat_depart] * P
    lat_meilleures = [lat_depart] * P
    stagnation = [0] * P
    durees = [duree_tabou] * P
    tabous: List[Dict[int, int]] = [{} for _ in range(P)]
    meilleures_agents: List[List[List[int]]] = [[list(r) for r in depart] for _ in range(P)]
    elite: List[Tuple[float, List[List[int]]]] = [(lat_depart, [list(r) for r in depart])]
    gbest_routes, gbest_lat = [list(r) for r in depart], lat_depart
    t_meilleure = 0.0

    # Control policies. The shared policy is a single object that every agent
    # consults and updates, and is the collective memory of the experience. The
    # private policy is its isolated counterpart, one per agent.
    politiques: Optional[List[PolitiqueThompson]] = None
    if politique == 'partagee':
        commune = PolitiqueThompson(len(options_duree), n_contextes, gamma_oubli,
                                    random.Random(graine * 7 + 1))
        politiques = [commune] * P
    elif politique == 'privee':
        politiques = [PolitiqueThompson(len(options_duree), n_contextes, gamma_oubli,
                                        random.Random(graine * 7 + 1 + i)) for i in range(P)]

    epoque = evaluations = 0
    compte_actions = {POURSUIVRE: 0, REPARTIR: 0, DIVERSIFIER: 0}
    choix: Counter = Counter()
    trajectoire: List[Tuple[float, float]] = []
    ameliorations_globales: List[Tuple[float, float]] = []
    detail_epoques: List[Dict[str, Any]] = []

    while True:
        ecoule = time.time() - t0
        reste = T - ecoule
        if reste < 0.25:
            break
        duree = min(delta, reste)
        epoque += 1
        debut_epoque = ecoule

        # --- 1. decisions, taken in the main process and in a fixed order ----
        departs, actions, etats, contextes = [], [], [], []
        for i in range(P):
            etat = {
                'agent': i, 'epoque': epoque, 'P': P,
                'fraction_ecoulee': ecoule / T,
                'stagnation': stagnation[i],
                'ecart_courant': (lat_courantes[i] - gbest_lat) / gbest_lat,
                'ecart_meilleur': (lat_meilleures[i] - gbest_lat) / gbest_lat,
                'detient_meilleure': lat_meilleures[i] <= gbest_lat + 1e-9,
                'taille_elite': len(elite),
            }
            ctx = None
            action = POURSUIVRE
            if politique == 'independante':
                d_i = duree_tabou
            elif politique == 'portefeuille':
                d_i = options_duree[(i * (len(options_duree) - 1)) // max(1, P - 1)]
            elif politique == 'hasard':
                d_i = options_duree[rng.randrange(len(options_duree))]
            elif politique == 'fenetre':
                permis = list(fenetre) if fenetre else list(range(len(options_duree)))
                d_i = options_duree[permis[rng.randrange(len(permis))]]
            elif politique == 'melange':
                poids = list(probabilites) if probabilites else \
                    [1.0 / len(options_duree)] * len(options_duree)
                if len(poids) != len(options_duree):
                    raise ValueError("melange : une probabilité par option")
                tirage = rng.random() * sum(poids)
                cumul, indice = 0.0, len(poids) - 1
                for j, pj in enumerate(poids):
                    cumul += pj
                    if tirage < cumul:
                        indice = j
                        break
                d_i = options_duree[indice]
            elif politique in ('privee', 'partagee'):
                c = politiques[i].contexte(stagnation[i])
                k = politiques[i].choisir(c)
                d_i = options_duree[k]
                ctx = (c, k)
            elif politique == 'regle':
                if stagnation[i] >= patience_epoques and not etat['detient_meilleure']:
                    action = REPARTIR
                d_i = duree_tabou
            elif politique == 'apprise':
                action, idx = decideur(etat)
                d_i = options_duree[idx]
            else:
                raise ValueError(f"politique inconnue : {politique}")
            durees[i] = int(d_i)
            choix[int(d_i)] += 1

            if action == REPARTIR:
                depart_i = [list(r) for r in gbest_routes]
                stagnation[i] = 0
                tabous[i] = {}
                meilleures_agents[i] = [list(r) for r in gbest_routes]
                lat_meilleures[i] = gbest_lat
            elif action == DIVERSIFIER:
                _lat_el, _rt_el = rng.choice(elite)
                depart_i = [list(r) for r in _rt_el]
                stagnation[i] = 0
                tabous[i] = {}
                meilleures_agents[i] = [list(r) for r in _rt_el]
                lat_meilleures[i] = _lat_el
            else:
                depart_i = courantes[i]
            compte_actions[action] += 1
            departs.append(depart_i)
            actions.append((action, durees[i]))
            etats.append(etat)
            contextes.append(ctx)

        # --- 2. submission of the searches of this epoch ----------------------
        aretes = _aretes_elite(elite) if facteur_elite > 0 else []
        taches = [{'routes': departs[i], 'duree_tabou': int(durees[i]),
                   'tabou': tabous[i],
                   'meilleur_routes': meilleures_agents[i],
                   'meilleur_latence': lat_meilleures[i],
                   'max_voisins': max_voisins, 'limite_temps': duree,
                   'aretes_elite': aretes, 'facteur_elite': facteur_elite,
                   'graine': graine * 1000003 + epoque * 131 + i} for i in range(P)]
        futurs = [pool.submit(W.recherche_tabou, t) for t in taches]
        resultats = [f.result() for f in futurs]

        # --- 3. updates, in the order of submission ---------------------------
        transitions = []
        for i, r in enumerate(resultats):
            evaluations += r['evaluations']
            courantes[i] = r['routes_courantes']
            lat_courantes[i] = r['latence_courante']
            tabous[i] = r.get('tabou_restant', {})
            ancien_perso = lat_meilleures[i]
            succes = r['latence'] < ancien_perso - 1e-9
            if succes:
                lat_meilleures[i] = r['latence']
                meilleures_agents[i] = [list(x) for x in r['routes']]
                stagnation[i] = 0
            else:
                stagnation[i] += 1
            if politiques is not None and contextes[i] is not None:
                c, k = contextes[i]
                politiques[i].mettre_a_jour(c, k, succes)
            if journaliser_epoques:
                detail_epoques.append({
                    'epoque': epoque, 'agent': i,
                    'stagnation': int(etats[i]['stagnation']),
                    'duree': int(durees[i]), 'succes': bool(succes),
                    'gain_relatif': max(0.0, (ancien_perso - r['latence']) / ancien_perso)})
            _inserer_elite(elite, r['latence'], r['routes'], taille_elite)
            ancien_global = gbest_lat
            if r['latence'] < gbest_lat - 1e-9:
                gbest_lat, gbest_routes = r['latence'], [list(x) for x in r['routes']]
                t_rel = r.get('t_meilleure_s')
                t_meilleure = debut_epoque + float(t_rel if t_rel is not None else duree)
                ameliorations_globales.append((round(t_meilleure, 3), gbest_lat))
            transitions.append({
                'etat': etats[i], 'action': actions[i],
                'gain_personnel': max(0.0, ancien_perso - r['latence']),
                'gain_global': max(0.0, ancien_global - gbest_lat),
                'latence_depart': lat_depart})
        trajectoire.append((round(time.time() - t0, 2), gbest_lat))
        if apprendre is not None:
            apprendre(transitions)

    # Quality recorded at the requested instants, from the dated improvements.
    releves = {}
    for s in instants_releves:
        v = lat_depart
        for t_am, lat in ameliorations_globales:
            if t_am <= s:
                v = lat
        releves[str(s)] = v

    sortie = {
        'routes': gbest_routes, 'latence': gbest_lat,
        'latence_depart': lat_depart,
        'temps_meilleure_s': round(t_meilleure, 3),
        'evaluations': evaluations, 'epoques': epoque,
        'actions': {'poursuivre': compte_actions[POURSUIVRE],
                    'repartir': compte_actions[REPARTIR],
                    'diversifier': compte_actions[DIVERSIFIER]},
        'options_duree': list(options_duree),
        'choix_durees': {str(k): v for k, v in sorted(choix.items())},
        'trajectoire': trajectoire,
        'ameliorations': ameliorations_globales,
        'releves': releves,
        'duree_s': round(time.time() - t0, 2),
        'politique': politique,
        'fenetre': list(fenetre) if fenetre else None,
        'probabilites': [float(p) for p in probabilites] if probabilites else None,
    }
    if journaliser_epoques:
        sortie['detail_epoques'] = detail_epoques
    if politiques is not None:
        uniques = list({id(p): p for p in politiques}.values())
        sortie['politique_apprise'] = [p.moyennes() for p in uniques]
    return sortie
