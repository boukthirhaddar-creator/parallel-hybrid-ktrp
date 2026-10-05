# -*- coding: utf-8 -*-
"""
TABU SEARCH KERNEL, EXECUTED INSIDE THE WORKER PROCESSES.

This module is deliberately self-contained and light: it imports neither NumPy nor
any scientific package. Under Windows, process-based parallelism runs in spawn
mode, in which every child process re-imports the module of the function it
executes. A heavy module would impose several seconds of start-up on each process,
and a function defined in the main script would not be importable from a child
process at all. This module avoids both difficulties.

It provides the single kernel used identically by every configuration compared in
the paper. That is what makes the comparison fair: all configurations share the
same operators, the same tabu formulation and the same evaluation. What
distinguishes them lies outside the kernel, namely the tabu tenure applied and the
way the agents are coordinated between epochs.

All exchanged data are built-in types and therefore serialisable.
"""
import random
import time
from typing import Any, Dict, List, Optional, Tuple


def cout_tournee(route: List[int], d: List[List[float]], circuit: bool) -> float:
    """Cost of a single route.

    Circuit variant, Equation (2) of the paper: the sum of the arrival times at
    the m customers, plus the instant at which the repairman returns to the depot.
    Arc weights decrease from m+1 on the first arc to 1 on the return arc.

    Path variant, Equation (1) of the paper: the sum of the arrival times at the
    customers only, with no return. Arc weights decrease from m to 1.

    The distinction between the two weight sequences is not cosmetic. Applying the
    circuit weights to the path variant while omitting the return arc yields a
    quantity that exceeds the latency by the duration of the route, which on these
    instances amounts to between 19 and 63 per cent. With the definition below,
    the kernel reproduces exactly the reference values published by independent
    groups, among them P16k8 at 382.90, P19k2 at 812.15 and E22k4 at 819.39.
    The check is performed by build_instances.py.
    """
    clients = [c for c in route if c != 0]
    m = len(clients)
    if m == 0:
        return 0.0
    if circuit:
        total = (m + 1) * d[0][clients[0]]
        for i in range(m - 1):
            total += (m - i) * d[clients[i]][clients[i + 1]]
        return total + d[clients[-1]][0]
    total = m * d[0][clients[0]]
    for i in range(m - 1):
        total += (m - 1 - i) * d[clients[i]][clients[i + 1]]
    return total


def bonus_tournee(route: List[int], elite: Dict[Tuple[int, int], int], facteur: float) -> float:
    """Elite-edge bonus of a route.

    Every undirected edge of the route that also occurs among the edges of the
    elite solutions held in the collective memory contributes its frequency times
    a factor. This term is the channel through which a shared pool of solutions
    could bias the local search of an agent. It is disabled in every configuration
    compared in the paper, the factor being zero, and is retained because the
    paper reports that enabling it brings no improvement.
    """
    if not elite:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        cle = (u, v) if u <= v else (v, u)
        f = elite.get(cle)
        if f:
            total += f * facteur
    return total


def latence(routes: List[List[int]], d: List[List[float]], circuit: bool) -> float:
    return sum(cout_tournee(r, d, circuit) for r in routes)


def est_valide(routes: List[List[int]], K: int, n_noeuds: int, exiger_non_vide: bool) -> bool:
    """Feasibility predicate: every customer exactly once, no empty route."""
    if len(routes) != K:
        return False
    vus = [False] * n_noeuds
    total = 0
    for r in routes:
        if len(r) < 2 or r[0] != 0 or r[-1] != 0:
            return False
        compte = 0
        for c in r:
            if c == 0:
                continue
            if c < 1 or c >= n_noeuds or vus[c]:
                return False
            vus[c] = True
            compte += 1
        if exiger_non_vide and compte == 0:
            return False
        total += compte
    return total == n_noeuds - 1


def _tirer_mouvement(routes: List[List[int]], rng: random.Random, exiger_non_vide: bool
                     ) -> Optional[Tuple[str, Tuple[int, ...], frozenset]]:
    """Draw one move among the five operators of the neighbourhood.

    Returns the operator, its parameters, and the set of displaced customers,
    which is what carries the tabu status. The operators are intra-route swap,
    relocate and segment reversal, and inter-route swap and relocate.
    """
    op = rng.choice(('swap', 'relocate', 'reverse', 'inter_swap', 'inter_relocate'))
    if op in ('swap', 'relocate', 'reverse'):
        candidats = [i for i, r in enumerate(routes) if len(r) > 3]
        if not candidats:
            return None
        ri = rng.choice(candidats)
        n = len(routes[ri]) - 2
        if op == 'swap':
            p1, p2 = rng.sample(range(1, n + 1), 2)
            return ('swap', (ri, p1, p2), frozenset((routes[ri][p1], routes[ri][p2])))
        if op == 'relocate':
            pd = rng.randint(1, n)
            pa = rng.randint(1, n)
            if pa == pd:
                return None
            return ('relocate', (ri, pd, pa), frozenset((routes[ri][pd],)))
        i, j = sorted(rng.sample(range(1, n + 1), 2))
        return ('reverse', (ri, i, j), frozenset((routes[ri][i], routes[ri][j])))
    if len(routes) < 2:
        return None
    r1, r2 = rng.sample(range(len(routes)), 2)
    if op == 'inter_swap':
        if len(routes[r1]) < 3 or len(routes[r2]) < 3:
            return None
        p1 = rng.randint(1, len(routes[r1]) - 2)
        p2 = rng.randint(1, len(routes[r2]) - 2)
        return ('inter_swap', (r1, p1, r2, p2),
                frozenset((routes[r1][p1], routes[r2][p2])))
    if len(routes[r1]) < 3:
        return None
    if exiger_non_vide and len(routes[r1]) == 3:
        return None          # the relocation would empty the source route
    pd = rng.randint(1, len(routes[r1]) - 2)
    pa = rng.randint(1, len(routes[r2]) - 1)
    return ('inter_relocate', (r1, pd, r2, pa), frozenset((routes[r1][pd],)))


def _appliquer(routes: List[List[int]], mvt: Tuple[str, Tuple[int, ...]]
               ) -> Tuple[List[List[int]], Tuple[int, ...]]:
    """Apply a move and return the new routes together with the modified indices."""
    op, p = mvt
    nouv = list(routes)
    if op == 'swap':
        ri, a, b = p
        r = list(routes[ri]); r[a], r[b] = r[b], r[a]
        nouv[ri] = r
        return nouv, (ri,)
    if op == 'relocate':
        ri, pd, pa = p
        r = list(routes[ri]); c = r.pop(pd); r.insert(pa, c)
        nouv[ri] = r
        return nouv, (ri,)
    if op == 'reverse':
        ri, i, j = p
        r = list(routes[ri]); r[i:j + 1] = r[i:j + 1][::-1]
        nouv[ri] = r
        return nouv, (ri,)
    if op == 'inter_swap':
        r1, p1, r2, p2 = p
        a, b = list(routes[r1]), list(routes[r2])
        a[p1], b[p2] = b[p2], a[p1]
        nouv[r1], nouv[r2] = a, b
        return nouv, (r1, r2)
    r1, pd, r2, pa = p
    a, b = list(routes[r1]), list(routes[r2])
    c = a.pop(pd); b.insert(pa, c)
    nouv[r1], nouv[r2] = a, b
    return nouv, (r1, r2)


def _perturbation_forte(routes: List[List[int]], rng: random.Random, exiger_non_vide: bool
                        ) -> List[List[int]]:
    """Strong perturbation: random relocations between routes.

    The number of relocations equals 15 per cent of the customers, with a minimum
    of three. Each transfers one customer from one route to another, at a position
    drawn at random.
    """
    nouv = [list(r) for r in routes]
    n_clients = sum(len(r) - 2 for r in nouv)
    for _ in range(max(3, int(n_clients * 0.15))):
        if len(nouv) < 2:
            break
        r1, r2 = rng.sample(range(len(nouv)), 2)
        if len(nouv[r1]) <= 2 or (exiger_non_vide and len(nouv[r1]) == 3):
            continue
        pd = rng.randint(1, len(nouv[r1]) - 2)
        pa = rng.randint(1, len(nouv[r2]) - 1)
        c = nouv[r1].pop(pd)
        nouv[r2].insert(pa, c)
    return nouv


# Instance data, transmitted once to each process by the pool initialiser. Passing
# them with every task would serialise a matrix of several tens of thousands of
# values on each call.
_INSTANCE: Dict[str, Any] = {}


def initialiser(distances: List[List[float]], K: int, n_noeuds: int,
                circuit: bool, exiger_non_vide: bool) -> None:
    """Pool initialiser: store the instance once per worker process."""
    _INSTANCE.clear()
    _INSTANCE.update({'distances': distances, 'K': K, 'n_noeuds': n_noeuds,
                      'circuit': circuit, 'exiger_non_vide': exiger_non_vide})


def recherche_tabou(tache: Dict[str, Any]) -> Dict[str, Any]:
    """Attribute-based tabu search, executable inside a child process.

    Input: a dictionary of built-in types.
        distances        distance matrix, as a list of lists
        K, n_noeuds      number of repairmen and of nodes, depot included
        circuit          True for the circuit variant
        exiger_non_vide  constraint that all K repairmen be used
        routes           starting solution
        duree_tabou      tabu tenure, in iterations
        max_voisins      size of the sampled neighbourhood at each iteration
        max_iterations   iteration limit, or None
        patience         iterations without improvement before stopping, or None
        budget           limit on the number of evaluations, or None
        limite_temps     wall-clock limit in seconds, or None
        graine           seed of the generator private to this search

    Output: best routes, latency, evaluation count, residual tabu list.

    Evaluating a neighbour recomputes only the modified routes. The total cost
    being the sum of independent route costs, the cost of a neighbour equals the
    current cost, minus the cost of the modified routes before the move, plus
    their cost after it. This computation is exact rather than approximate, a
    property verified on twenty thousand randomly generated moves.
    """
    t0 = time.time()
    # Instance data come from the task when it carries them, and from the process
    # initialiser otherwise.
    source = tache if 'distances' in tache else _INSTANCE
    d = source['distances']
    circuit = source.get('circuit', True)
    K, n_noeuds = source['K'], source['n_noeuds']
    exiger = source.get('exiger_non_vide', True)
    duree = int(tache.get('duree_tabou', 3))
    max_voisins = int(tache.get('max_voisins', 500))
    max_iter = tache.get('max_iterations')
    patience = tache.get('patience')
    budget = tache.get('budget')
    limite = tache.get('limite_temps')
    rng = random.Random(tache.get('graine', 0))
    # Elite edges are passed as a list of (u, v, frequency) triples, the only form
    # that inter-process serialisation guarantees.
    elite: Dict[Tuple[int, int], int] = {(int(u), int(v)): int(f)
                                          for u, v, f in tache.get('aretes_elite', [])}
    facteur = float(tache.get('facteur_elite', 0.1))
    # Diverse solutions drawn from the collective memory, used by the restart
    # strategy. It is disabled in every configuration compared in the paper.
    relances: List[List[List[int]]] = [[list(r) for r in sol]
                                       for sol in tache.get('solutions_memoire', [])]
    # Diversification threshold. It is tied to the tenure itself: the search
    # diversifies once it has stagnated for as many iterations as the tenure
    # lasts, that is once a full tabu list has failed to make it progress. Tying
    # the two quantities matters, because a threshold of one would empty the tabu
    # list after every unproductive iteration and the tenure would then have no
    # effect at all. The threshold is bounded so that diversification can occur
    # before the stopping criterion is met.
    diversifier = tache.get('diversifier', False)
    p_relance = float(tache.get('p_relance_memoire', 0.3))
    seuil_div = max(1, min(duree, (patience - 1) if patience else duree))
    diversifications = 0

    courante = [list(r) for r in tache['routes']]
    couts = [cout_tournee(r, d, circuit) for r in courante]
    bonus = [bonus_tournee(r, elite, facteur) for r in courante]
    total = sum(couts)
    total_bonus = sum(bonus)
    meilleure, meilleur_total = [list(r) for r in courante], total
    total_initial = total
    # Reference best solution, inherited from the previous epoch when the agent
    # continues its trajectory. The aspiration criterion compares against this
    # reference. Were it reset to the current solution at each epoch, the tabu
    # status would be waived for any move improving on the current solution, a far
    # weaker requirement than improving on the best of the trajectory, and the
    # search would start cycling at the beginning of every epoch.
    if tache.get('meilleur_latence') is not None and tache['meilleur_latence'] < meilleur_total:
        meilleur_total = float(tache['meilleur_latence'])
        meilleure = [list(r) for r in tache['meilleur_routes']]

    # Tabu memory inherited from a previous epoch, passed as the number of
    # iterations remaining on each prohibition. Carrying it across epochs lets an
    # agent resume its trajectory exactly, without the perturbation that resetting
    # the list would introduce at every epoch boundary.
    tabou: Dict[int, int] = {int(c): int(reste) for c, reste in tache.get('tabou', {}).items()
                             if int(reste) > 0}
    iteration = evaluations = interdits = ameliorations = 0
    # Instant and evaluation count at which the best solution of this call was
    # obtained. They remain None when the call does not improve on the reference
    # received, in which case the time to best is read from the preceding call.
    t_meilleure: Optional[float] = None
    eval_meilleure: Optional[int] = None
    sans_progres = 0
    sans_progres_div = 0

    while True:
        if max_iter is not None and iteration >= max_iter:
            break
        if budget is not None and evaluations >= budget:
            break
        if limite is not None and (time.time() - t0) >= limite:
            break
        if patience is not None and sans_progres >= patience:
            break
        iteration += 1

        # Move selection. The score of a neighbour is its quality, that is the
        # opposite of its latency, plus the elite-edge bonus. Among the moves that
        # are not forbidden, the best score wins. A forbidden move is admitted only
        # if its quality, and not its score, exceeds the best quality reached so
        # far; it is then retained if its score exceeds that of the best
        # non-forbidden move.
        meilleur_libre, score_libre = None, -float('inf')
        meilleur_aspire, score_aspire = None, -float('inf')
        essais = 0
        vus = 0
        while vus < max_voisins and essais < max_voisins * 20:
            essais += 1
            mvt = _tirer_mouvement(courante, rng, exiger)
            if mvt is None:
                continue
            op, params, attributs = mvt
            nouv, modifiees = _appliquer(courante, (op, params))
            nouveau_total, nouveau_bonus = total, total_bonus
            for idx in modifiees:
                nouveau_total += cout_tournee(nouv[idx], d, circuit) - couts[idx]
                if elite:
                    nouveau_bonus += bonus_tournee(nouv[idx], elite, facteur) - bonus[idx]
            vus += 1
            evaluations += 1
            score = -nouveau_total + nouveau_bonus
            candidat = (nouv, modifiees, attributs, nouveau_total, nouveau_bonus)
            if any(tabou.get(c, 0) > iteration for c in attributs):
                interdits += 1
                if -nouveau_total > -meilleur_total + 1e-9 and score > score_aspire:
                    meilleur_aspire, score_aspire = candidat, score
                continue
            if score > score_libre:
                meilleur_libre, score_libre = candidat, score

        if meilleur_aspire is not None and (meilleur_libre is None or score_aspire > score_libre):
            retenu = meilleur_aspire
        else:
            retenu = meilleur_libre
        if retenu is None:
            sans_progres += 1
            sans_progres_div += 1
            continue
        nouv, modifiees, attributs, retenu_total, retenu_bonus = retenu
        courante = nouv
        for idx in modifiees:
            couts[idx] = cout_tournee(courante[idx], d, circuit)
            bonus[idx] = bonus_tournee(courante[idx], elite, facteur)
        total, total_bonus = retenu_total, retenu_bonus
        for c in attributs:
            tabou[c] = iteration + duree
        if total < meilleur_total - 1e-9:
            meilleure, meilleur_total = [list(r) for r in courante], total
            ameliorations += 1
            t_meilleure, eval_meilleure = time.time() - t0, evaluations
            sans_progres = 0
            sans_progres_div = 0
        else:
            sans_progres += 1
            sans_progres_div += 1

        # Diversification: a strong perturbation most of the time, and a restart
        # from a diverse solution of the memory otherwise. The tabu list is cleared
        # in both cases.
        if diversifier and sans_progres_div >= seuil_div:
            if relances and rng.random() < p_relance:
                courante = [list(r) for r in rng.choice(relances)]
            else:
                courante = _perturbation_forte(courante, rng, exiger)
            couts = [cout_tournee(r, d, circuit) for r in courante]
            bonus = [bonus_tournee(r, elite, facteur) for r in courante]
            total, total_bonus = sum(couts), sum(bonus)
            if total < meilleur_total - 1e-9:
                meilleure, meilleur_total = [list(r) for r in courante], total
            tabou.clear()
            sans_progres_div = 0
            diversifications += 1

    return {
        'routes': meilleure,
        # Current solution at the end of the search, distinct from the best one. It
        # allows the trajectory to resume where it stopped, which is what makes a
        # search split into successive epochs behave like a continuous one.
        'routes_courantes': [list(r) for r in courante],
        # Prohibitions still active, expressed as remaining iterations, so that a
        # subsequent epoch can resume them.
        'tabou_restant': {c: e - iteration for c, e in tabou.items() if e > iteration},
        'latence_courante': total,
        'latence': meilleur_total,
        'latence_initiale': total_initial,
        'evaluations': evaluations,
        'iterations': iteration,
        'interdits': interdits,
        'ameliorations': ameliorations,
        'diversifications': diversifications,
        'seuil_diversification': seuil_div,
        'duree_s': time.time() - t0,
        't_meilleure_s': t_meilleure,
        'evaluations_meilleure': eval_meilleure,
    }
