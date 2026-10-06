#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECONSTRUCTION AND VERIFICATION OF THE K-TRP INSTANCES.
Run with: python build_instances.py

Instances are rebuilt from the CVRPLIB coordinates supplied in the cvrplib folder
(.vrp files, depot at node 1).

Distance conventions
--------------------
Circuit variant: d_ij is the integer part of the Euclidean distance, plus one,
    for i different from j. This rule reproduces the original instance files
    exactly, entry by entry, on the seven files that were available (P16K5,
    P16K7, P16K8, P19K2, P20K2, P21K2, P22K2). Rounding to the nearest integer
    differs on 152 to 292 entries depending on the instance, and rounding upwards
    on 26 to 38. The convention is therefore a truncation increased by one unit
    rather than a rounding. Under this rule the exact resolution recovers the
    proven optima published in the literature (P19K3 933, P21K3 1049, E22K4 1269).

Path variant: Euclidean distances in double precision. With the objective of
    Equation (1), the kernel reproduces exactly the reference values of
    Nucamendi-Guillen et al. (P16K8 382.90, P19K2 812.15, E22K4 819.39, E23K3
    1555.87, P20K2 905.19, P21K2 937.10, P22K2 993.10, P23K8 561.33).

Circuit instances whose source is not immediate
-----------------------------------------------
    P22K7: coordinates of P-n22-k2. With those of P-n22-k8 the exact optimum
           (1195) exceeds the published reference (1105), which is impossible.
    P23K7: coordinates of P-n23-k8, which contain those of P-n22-k2.

The controler() function below checks both conventions against published values.
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DOSSIER_VRP = os.path.join(ICI, 'cvrplib')
SORTIE = os.path.join(ICI, 'instances')

# nom -> (fichier CVRPLIB, K)
CIRCUIT = {
    'P16K5': ('P-n16-k8', 5), 'P16K7': ('P-n16-k8', 7), 'P19K2': ('P-n19-k2', 2),
    'P19K3': ('P-n19-k2', 3), 'P20K2': ('P-n20-k2', 2), 'P20K3': ('P-n20-k2', 3),
    'P21K3': ('P-n21-k2', 3), 'E22K4': ('E-n22-k4', 4), 'P22K7': ('P-n22-k2', 7),
    'P23K7': ('P-n23-k8', 7), 'E30K4': ('E-n30-k3', 4), 'E33K4': ('E-n33-k4', 4),
    'E51K5': ('E-n51-k5', 5), 'E76K7': ('E-n76-k7', 7), 'E76K8': ('E-n76-k7', 8),
    'E76K10': ('E-n76-k7', 10), 'E76K14': ('E-n76-k7', 14), 'E76K15': ('E-n76-k7', 15),
    'E101K8': ('E-n101-k8', 8), 'E101K14': ('E-n101-k8', 14),
    # Not evaluated in the paper; present in the original instance directory
    'P16K8': ('P-n16-k8', 8), 'P21K2': ('P-n21-k2', 2), 'P22K2': ('P-n22-k2', 2),
}
CHEMIN = {
    'E22K4': ('E-n22-k4', 4), 'E23K3': ('E-n23-k3', 3), 'E30K3': ('E-n30-k3', 3),
    'E30K4': ('E-n30-k3', 4), 'E33K4': ('E-n33-k4', 4), 'E76K14': ('E-n76-k7', 14),
    'E76K15': ('E-n76-k7', 15), 'P16K8': ('P-n16-k8', 8), 'P19K2': ('P-n19-k2', 2),
    'P20K2': ('P-n20-k2', 2), 'P21K2': ('P-n21-k2', 2), 'P22K2': ('P-n22-k2', 2),
    'P22K8': ('P-n22-k8', 8), 'P23K8': ('P-n23-k8', 8), 'P40K5': ('P-n40-k5', 5),
    'P45K5': ('P-n45-k5', 5), 'P50K7': ('P-n50-k7', 7), 'P50K8': ('P-n50-k8', 8),
    'P50K10': ('P-n50-k10', 10), 'P51K10': ('P-n51-k10', 10), 'P55K7': ('P-n55-k7', 7),
    'P55K8': ('P-n55-k8', 8), 'P55K10': ('P-n55-k10', 10), 'P55K15': ('P-n55-k15', 15),
    'P60K10': ('P-n60-k10', 10), 'P65K10': ('P-n65-k10', 10), 'P70K10': ('P-n70-k10', 10),
    'P76K4': ('P-n76-k4', 4), 'P76K5': ('P-n76-k5', 5),
    # E51K5 and E76K10 of the circuit suite are the same instances as CMT1 and CMT2
    'CMT1': ('E-n51-k5', 5), 'CMT2': ('E-n76-k10', 10), 'CMT3': ('E-n101-k8', 8),
    'CMT4': ('M-n151-k12', 12), 'CMT5': ('M-n200-k17', 17), 'CMT11': ('M-n121-k7', 7),
    'CMT12': ('M-n101-k10', 10),
}
# Validation instances, absent from every result table of the paper.
VALIDATION = {'A45K6': ('A-n45-k6', 6), 'A60K9': ('A-n60-k9', 9), 'A80K10': ('A-n80-k10', 10)}

# Instances de test de l'etude de transfert (run_transfer_study.py), variante chemin.
# X series of Uchoa et al. (2017), 100 to 147 customers, K equal to the vehicle
# count in the instance name. None was used for design or for training.
TEST_TRANSFERT = {
    'X101K25': ('X-n101-k25', 25), 'X106K14': ('X-n106-k14', 14), 'X110K13': ('X-n110-k13', 13),
    'X115K10': ('X-n115-k10', 10), 'X120K6': ('X-n120-k6', 6), 'X125K30': ('X-n125-k30', 30),
    'X129K18': ('X-n129-k18', 18), 'X134K13': ('X-n134-k13', 13), 'X139K10': ('X-n139-k10', 10),
    'X143K7': ('X-n143-k7', 7), 'X148K46': ('X-n148-k46', 46),
}

# Exact optima computed by dynamic programming (ktrp_exact.c), circuit variant.
OPTIMA_CIRCUIT = {'P16K5': 778, 'P16K7': 824, 'P16K8': 858, 'P19K2': 1110, 'P19K3': 933,
                  'P20K2': 1211, 'P20K3': 1015, 'P21K2': 1262, 'P21K3': 1049, 'P22K2': 1339,
                  'E22K4': 1269, 'P22K7': 1066, 'P23K7': 1054}


def lire_vrp(nom):
    pts, lecture = [], False
    for l in open(os.path.join(DOSSIER_VRP, nom + '.vrp'), encoding='utf-8'):
        l = l.strip()
        if l.startswith('NODE_COORD_SECTION'):
            lecture = True
            continue
        if l.startswith('DEMAND_SECTION'):
            break
        if lecture and l:
            p = l.split()
            pts.append((float(p[1]), float(p[2])))
    return pts


def matrice(pts, variante):
    n = len(pts)
    if variante == 'circuit':
        return [[0 if i == j else int(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])) + 1
                 for j in range(n)] for i in range(n)]
    return [[math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) for j in range(n)]
            for i in range(n)]


def nom_fichier(nom, variante):
    if variante == 'circuit':
        lettre, reste = nom[0], nom[1:]
        n, k = reste.split('K')
        return f"KTRP_test_{lettre}{n}_{k}.txt"
    return f"KTRP_path_{nom}.txt"


def ecrire(chemin, K, M):
    with open(chemin, 'w', encoding='utf-8') as f:
        f.write(f"{K}\n{len(M)}\n")
        for r in M:
            f.write('\t'.join(str(x) if isinstance(x, int) else repr(x) for x in r) + '\t\n')


def generer():
    for variante, table in (('circuit', CIRCUIT), ('path', CHEMIN)):
        d = os.path.join(SORTIE, variante)
        os.makedirs(d, exist_ok=True)
        for nom, (src, K) in table.items():
            ecrire(os.path.join(d, nom_fichier(nom, variante)), K, matrice(lire_vrp(src), variante))
        for nom, (src, K) in VALIDATION.items():
            f = (f"KTRP_test_{nom}.txt" if variante == 'circuit' else f"KTRP_path_{nom}.txt")
            ecrire(os.path.join(d, f), K, matrice(lire_vrp(src), variante))
        n_x = 0
        if variante == 'path':
            for nom, (src, K) in TEST_TRANSFERT.items():
                if os.path.exists(os.path.join(DOSSIER_VRP, src + '.vrp')):
                    ecrire(os.path.join(d, f"KTRP_path_{nom}.txt"), K, matrice(lire_vrp(src), variante))
                    n_x += 1
        print(f"  {variante}: {len(table) + len(VALIDATION) + n_x} files written to {d}")


def comparer(dossier):
    """Compare existing instance files against the reconstructions, entry by entry."""
    for f in sorted(os.listdir(dossier)):
        if not f.startswith('KTRP_test_') or not f.endswith('.txt'):
            continue
        base = f[len('KTRP_test_'):-4]
        lettre, reste = base[0], base[1:]
        if '_' not in reste:
            continue
        n, k = reste.split('_', 1)
        nom = f"{lettre}{n}K{k}"
        if nom not in CIRCUIT:
            print(f"  {f}: unknown source")
            continue
        lignes = [l for l in open(os.path.join(dossier, f)) if l.strip()]
        M = [[float(x) for x in l.split()] for l in lignes[2:]]
        R = matrice(lire_vrp(CIRCUIT[nom][0]), 'circuit')
        diff = sum(1 for i in range(len(R)) for j in range(len(R)) if M[i][j] != R[i][j])
        print(f"  {f}: {'IDENTICAL' if diff == 0 else f'{diff} differing entrie(s)'}")


def controler():
    """Check both objectives on the small instances, using the kernel of this folder."""
    sys.path.insert(0, ICI)
    import ktrp_worker as W
    refs = json.load(open(os.path.join(ICI, 'bks_path.json'))) if os.path.exists(
        os.path.join(ICI, 'bks_path.json')) else {}

    def ppv(d, K):
        n = len(d); rest = set(range(1, n)); R = [[0] for _ in range(K)]; k = 0
        while rest:
            c = min(rest, key=lambda x: d[R[k][-1]][x]); R[k].append(c); rest.remove(c); k = (k + 1) % K
        return [r + [0] for r in R]

    print("  Path variant, reference values of Nucamendi-Guillen et al.:")
    for nom in ('P16K8', 'P19K2', 'E22K4', 'P20K2', 'P23K8'):
        src, K = CHEMIN[nom]
        d = matrice(lire_vrp(src), 'path')
        W.initialiser(d, K, len(d), False, True)
        v = min(W.recherche_tabou({'routes': ppv(d, K), 'duree_tabou': 3, 'max_voisins': 500,
                                   'max_iterations': 1500, 'graine': g})['latence'] for g in (1, 2))
        ref = refs.get(nom)
        ok = ref is not None and abs(v - ref) < 0.01
        print(f"    {nom:6s} obtained {v:9.2f}  reference {ref}  {'OK' if ok else 'MISMATCH'}")
    print("  Circuit variant, exact optima:")
    for nom in ('P16K5', 'P16K7', 'P19K2', 'P20K2', 'E22K4'):
        src, K = CIRCUIT[nom]
        d = matrice(lire_vrp(src), 'circuit')
        W.initialiser(d, K, len(d), True, True)
        v = min(W.recherche_tabou({'routes': ppv(d, K), 'duree_tabou': 3, 'max_voisins': 500,
                                   'max_iterations': 2500, 'graine': g})['latence'] for g in (1, 2))
        print(f"    {nom:6s} obtained {v:7.0f}  optimum {OPTIMA_CIRCUIT[nom]}  "
              f"{'OK' if abs(v - OPTIMA_CIRCUIT[nom]) < 0.5 else 'MISMATCH'}")


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'tout'
    if action in ('generer', 'tout'):
        print("GENERATION")
        generer()
    if action.startswith('comparer'):
        comparer(sys.argv[2] if len(sys.argv) > 2 else ICI)
    if action in ('controler', 'tout'):
        print("VERIFICATION")
        controler()
