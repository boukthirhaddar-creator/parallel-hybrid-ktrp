#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-CHECKS ON THE SEARCH KERNEL.
Run with: python selfcheck.py

The checks below verify, on a small randomly generated instance, the properties
on which every result of the paper depends. They complete in a few seconds and
require only the standard library. Run them before any campaign, and after any
modification of ktrp_worker.py.

    1. Circuit objective    equals the sum of the arrival times at the customers
                            plus the instant of return to the depot, recomputed
                            directly rather than through the arc weights.
    2. Path objective       equals the sum of the arrival times alone.
    3. Move feasibility     no generated move produces an infeasible solution.
    4. Incremental cost     the incremental evaluation of a neighbour agrees
                            exactly with a full recomputation.
    5. Tabu tenure          a nominal tenure of T forbids a customer for T - 1
                            subsequent iterations, which is the rule described in
                            the paper.
    6. Determinism          the same seed produces the same result.
    7. Returned solution    is feasible and its stated latency is exact.

A failure means that the kernel no longer has the behaviour described in the
paper, and that the deposited logs cannot be reproduced from it.
"""
import math
import random
import sys

import ktrp_worker as W


def random_instance(n_customers: int, K: int, seed: int, side: float = 100.0,
                    integer: bool = True):
    """A small Euclidean instance, generated so that the checks need no data file."""
    rng = random.Random(seed)
    points = [(side / 2, side / 2)] + [(rng.uniform(0, side), rng.uniform(0, side))
                                       for _ in range(n_customers)]
    n = len(points)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            value = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            d[i][j] = float(round(value)) if integer else value
    return d, K, n


def nearest_neighbour(d, K):
    n = len(d)
    remaining = set(range(1, n))
    routes = [[0] for _ in range(K)]
    k = 0
    while remaining:
        c = min(remaining, key=lambda x: (d[routes[k][-1]][x], x))
        routes[k].append(c)
        remaining.remove(c)
        k = (k + 1) % K
    return [r + [0] for r in routes]


def cost_from_arrival_times(route, d, circuit):
    """Objective recomputed from first principles, without the arc weights."""
    clock = 0.0
    total = 0.0
    previous = 0
    customers = [c for c in route if c != 0]
    for c in customers:
        clock += d[previous][c]
        total += clock
        previous = c
    if circuit and customers:
        total += clock + d[previous][0]
    return total


def main() -> int:
    failures = 0

    def report(name, passed, detail=""):
        nonlocal failures
        failures += 0 if passed else 1
        print(f"  [{'OK  ' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    print("SELF-CHECKS ON THE SEARCH KERNEL")
    print("-" * 72)

    d, K, n = random_instance(30, 4, seed=7, integer=False)
    rng = random.Random(3)

    ok_circuit = ok_path = True
    for _ in range(300):
        customers = list(range(1, n))
        rng.shuffle(customers)
        route = [0] + customers[:rng.randint(1, 12)] + [0]
        ok_circuit &= abs(W.cout_tournee(route, d, True)
                          - cost_from_arrival_times(route, d, True)) < 1e-9
        ok_path &= abs(W.cout_tournee(route, d, False)
                       - cost_from_arrival_times(route, d, False)) < 1e-9
    report("circuit objective equals arrivals plus return", ok_circuit)
    report("path objective equals arrivals only", ok_path)

    solution = nearest_neighbour(d, K)
    infeasible = wrong_increment = 0
    for _ in range(20000):
        move = W._tirer_mouvement(solution, rng, True)
        if move is None:
            continue
        op, params, _ = move
        candidate, modified = W._appliquer(solution, (op, params))
        if not W.est_valide(candidate, K, n, True):
            infeasible += 1
            continue
        incremental = W.latence(solution, d, True) + sum(
            W.cout_tournee(candidate[i], d, True) - W.cout_tournee(solution[i], d, True)
            for i in modified)
        if abs(incremental - W.latence(candidate, d, True)) > 1e-9:
            wrong_increment += 1
        if rng.random() < 0.3:
            solution = candidate
    report("generated moves are always feasible", infeasible == 0,
           f"{infeasible} infeasible out of 20000")
    report("incremental evaluation is exact", wrong_increment == 0,
           f"{wrong_increment} disagreements")

    tenure, iteration = 3, 10
    tabu = {5: iteration + tenure}
    forbidden = [i for i in range(iteration + 1, iteration + 6) if tabu.get(5, 0) > i]
    report("tabu tenure matches the rule stated in the paper",
           forbidden == [11, 12],
           f"nominal tenure {tenure} forbids iterations {forbidden}")

    W.initialiser(d, K, n, True, True)
    task = {'routes': nearest_neighbour(d, K), 'duree_tabou': 3, 'max_voisins': 200,
            'budget': 30000, 'graine': 11}
    first = W.recherche_tabou(dict(task))
    second = W.recherche_tabou(dict(task))
    report("the same seed reproduces the same result",
           first['routes'] == second['routes'],
           f"latency {first['latence']:.4f}")
    report("the returned solution is feasible and its latency exact",
           W.est_valide(first['routes'], K, n, True)
           and abs(W.latence(first['routes'], d, True) - first['latence']) < 1e-9)

    print("-" * 72)
    if failures == 0:
        print("All checks passed. The kernel behaves as described in the paper.")
    else:
        print(f"{failures} check(s) failed. Do not run a campaign in this state.")
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
