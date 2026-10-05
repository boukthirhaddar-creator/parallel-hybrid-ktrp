#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARALLEL PERFORMANCE MEASUREMENTS.
Run alone on the machine: python run_parallel_study.py

Three quantities are measured, all on the same ktrp_worker kernel as the
campaigns. They are reported separately because they answer different questions,
and conflating them overstates what parallelism delivers to the user.

1. THROUGHPUT, in neighbour evaluations per second, as a function of the number P
   of processes, in two modes:
     noyau  P fully independent tabu searches with no coordination. This measures
            the capacity of the machine itself, including any reduction of clock
            frequency when several cores are active and the effect of
            simultaneous multithreading beyond the physical cores;
     cadre  the complete multi-agent framework (epochs, decisions, collective
            memory, learned policy) on the same P processes.
   From these are derived the throughput speedup, the efficiency, the experimental
   sequential fraction of Karp and Flatt, the corresponding Amdahl bound, and the
   cost of coordination alone, namely 1 minus the ratio of the two throughputs.

2. TIME TO REACH A QUALITY TARGET, as a function of P. For each instance, three
   targets fixed in advance at 1 %, 0.5 % and 0.2 % above a stated reference
   value. Success rate, median time by the Kaplan-Meier estimator, which handles
   runs stopped before the target correctly, and speedup against P = 1. This is
   the accepted speedup measure for parallel metaheuristics (Alba and Luque, 2005;
   Aiex, Resende and Ribeiro, 2007, time-to-target plots).

3. QUALITY AT FIXED WALL-CLOCK TIME, as a function of P: the gap to the reference
   recorded at 15, 30, 60, 120 and 240 seconds.

The hardware configuration (processor, physical and logical cores, operating
system, Python version) is recorded automatically, together with the resident
memory when psutil is available.

Conditions: the machine must run no other computation, be connected to mains
power, and have sleep disabled.
"""
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

ICI = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if ICI not in sys.path:
    sys.path.insert(0, ICI)

import ktrp_worker as W            # noqa: E402
import ktrp_coordinator as C        # noqa: E402

# =============================================================================
#                              CONFIGURATION
# =============================================================================
DOSSIER = "run_parallel_study"
METHODE = "partagee"            # politique du cadre evalue : partagee, privee ou hasard

# 1. Debit
INSTANCES_DEBIT = ["instances/circuit/KTRP_test_E76_10.txt",
                   "instances/path/KTRP_path_CMT11.txt"]
P_DEBIT = [1, 2, 3, 4, 6, 8]    # values above the logical core count are skipped
REPETITIONS_DEBIT = 3
DUREE_DEBIT_S = 30

# 2 and 3. Time to reach a target, and quality at fixed time.
# Reference values fixed in advance, with their provenance:
REFERENCES = {
    "instances/circuit/KTRP_test_E76_10.txt": (3519.0, "best known value, reached by all 50 runs of the campaign"),
    "instances/circuit/KTRP_test_E101_8.txt": (5469.0, "best value found by the circuit campaign"),
    "instances/path/KTRP_path_CMT3.txt": (4002.90, "published reference (Ngueveu et al., Nucamendi-Guillen et al.)"),
    "instances/path/KTRP_path_CMT11.txt": (7200.68, "published reference (Ngueveu et al., Nucamendi-Guillen et al.)"),
}
P_CIBLE = [1, 2, 3, 4]
GRAINES = [4001, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073]
DUREE_CIBLE_S = 240
ECARTS_CIBLES = (0.01, 0.005, 0.002)
INSTANTS_RELEVES = (15, 30, 60, 120, 240)
N_EPOQUES = 30
# =============================================================================


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ICI, p)


def lire_instance(chemin: str):
    lignes = [l.strip() for l in open(_abs(chemin), encoding='utf-8', errors='replace') if l.strip()]
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


def est_circuit(chemin: str) -> bool:
    return 'KTRP_path_' not in os.path.basename(chemin)


def materiel() -> Dict[str, Any]:
    info = {'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'systeme': platform.platform(), 'processeur': platform.processor(),
            'coeurs_logiques': os.cpu_count(), 'python': sys.version.split()[0],
            'methode': METHODE}
    try:
        import psutil
        info['coeurs_physiques'] = psutil.cpu_count(logical=False)
        f = psutil.cpu_freq()
        if f:
            info['frequence_mhz'] = {'actuelle': round(f.current), 'max': round(f.max)}
        info['memoire_totale_go'] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        info['coeurs_physiques'] = None
    return info


def memoire_mo() -> Optional[float]:
    """Resident memory of the main process and its children, in MB."""
    try:
        import psutil
        p = psutil.Process()
        return round((p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True)))
                     / 2 ** 20, 1)
    except Exception:
        return None


def ouvrir_pool(P: int, d, K: int, n: int, circuit: bool) -> ProcessPoolExecutor:
    """Pool de P processus, tous demarres avant toute mesure."""
    pool = ProcessPoolExecutor(P, mp_context=mp.get_context('spawn'), initializer=W.initialiser,
                               initargs=(d, K, n, circuit, True))
    # Simultaneous tasks force the pool to start all P processes.
    for f in [pool.submit(time.sleep, 0.5) for _ in range(P)]:
        f.result()
    return pool


def journal(chemin: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chemin):
        return []
    return [json.loads(l) for l in open(chemin, encoding='utf-8') if l.strip()]


def ecrire(chemin: str, enr: Dict[str, Any]) -> None:
    with open(chemin, 'a', encoding='utf-8') as f:
        f.write(json.dumps(enr, ensure_ascii=False) + '\n')


# -----------------------------------------------------------------------------
# 1. DEBIT
# -----------------------------------------------------------------------------
def mesurer_debit(fichier: str) -> None:
    faits = {(r['instance'], r['mode'], r['P'], r['repetition']) for r in journal(fichier)}
    for inst in INSTANCES_DEBIT:
        K, n, d = lire_instance(inst)
        circuit = est_circuit(inst)
        depart = plus_proche_voisin(d, K)
        nom = os.path.splitext(os.path.basename(inst))[0]
        for P in [p for p in P_DEBIT if p <= (os.cpu_count() or 1)]:
            a_faire = [(m, r) for r in range(REPETITIONS_DEBIT) for m in ('noyau', 'cadre')
                       if (nom, m, P, r) not in faits]
            if not a_faire:
                continue
            pool = ouvrir_pool(P, d, K, n, circuit)
            try:
                for mode, rep in a_faire:
                    graine = 5000 + 100 * P + rep
                    cpu0 = time.process_time()
                    t0 = time.perf_counter()
                    if mode == 'noyau':
                        taches = [{'routes': depart, 'duree_tabou': 3, 'max_voisins': 500,
                                   'limite_temps': DUREE_DEBIT_S, 'graine': graine * 31 + i}
                                  for i in range(P)]
                        res = [f.result() for f in [pool.submit(W.recherche_tabou, t) for t in taches]]
                        evaluations = sum(r['evaluations'] for r in res)
                    else:
                        r = C.recherche_cooperative(pool, P, DUREE_DEBIT_S, depart, d, circuit,
                                                    politique=METHODE, graine=graine,
                                                    n_epoques=N_EPOQUES)
                        evaluations = r['evaluations']
                    mur = time.perf_counter() - t0
                    cpu_principal = time.process_time() - cpu0
                    enr = {'instance': nom, 'n_clients': n - 1, 'K': K, 'mode': mode, 'P': P,
                           'repetition': rep, 'evaluations': evaluations, 'mur_s': mur,
                           'debit': evaluations / mur, 'cpu_principal_s': cpu_principal,
                           'memoire_mo': memoire_mo()}
                    ecrire(fichier, enr)
                    print(f"    {nom:18s} {mode:6s} P = {P}  repetition {rep}  "
                          f"{enr['debit']:12,.0f} evaluations/s", flush=True)
            finally:
                pool.shutdown()


def analyser_debit(enr: List[Dict[str, Any]], coeurs_physiques: Optional[int]) -> Dict[str, Any]:
    sortie = {}
    for inst in sorted({r['instance'] for r in enr}):
        print(f"\n  {inst}")
        print(f"    {'P':>3s} {'noyau (eval/s)':>15s} {'cadre (eval/s)':>15s} {'accel. cadre':>12s} "
              f"{'efficacite':>10s} {'Karp-Flatt':>10s} {'surcout':>8s} {'CPU coord.':>10s} {'memoire':>9s}")
        deb = {}
        for P in sorted({r['P'] for r in enr if r['instance'] == inst}):
            ligne = {}
            for mode in ('noyau', 'cadre'):
                v = [r for r in enr if r['instance'] == inst and r['P'] == P and r['mode'] == mode]
                if v:
                    ligne[mode] = st.fmean([r['debit'] for r in v])
                    if mode == 'cadre':
                        ligne['cpu_coord'] = st.fmean([r['cpu_principal_s'] / r['mur_s'] for r in v])
                        mem = [r['memoire_mo'] for r in v if r.get('memoire_mo') is not None]
                        ligne['memoire'] = max(mem) if mem else None
            deb[P] = ligne
        if 1 not in deb or 'cadre' not in deb[1]:
            print("    P = 1 missing: speedup not computable.")
            continue
        base = deb[1]['cadre']
        kf = []
        tab = []
        for P, l in deb.items():
            if 'cadre' not in l:
                continue
            S = l['cadre'] / base
            E = S / P
            e = (1 / S - 1 / P) / (1 - 1 / P) if P > 1 else None
            if e is not None and (coeurs_physiques is None or P <= coeurs_physiques):
                kf.append(e)
            surcout = 1 - l['cadre'] / l['noyau'] if 'noyau' in l else None
            tab.append({'P': P, 'debit_noyau': l.get('noyau'), 'debit_cadre': l['cadre'], 'acceleration': S,
                        'efficacite': E, 'karp_flatt': e, 'surcout_coordination': surcout,
                        'cpu_coordinateur': l.get('cpu_coord'), 'memoire_mo': l.get('memoire')})
            print(f"    {P:3d} {l.get('noyau', float('nan')):15,.0f} {l['cadre']:15,.0f} {S:12.2f} {E:10.2f} "
                  f"{(f'{e:.3f}' if e is not None else '-'):>10s} "
                  f"{(f'{surcout * 100:.1f} %' if surcout is not None else '-'):>8s} "
                  f"{l.get('cpu_coord', 0) * 100:9.1f} % "
                  f"{(str(l.get('memoire')) + ' Mo') if l.get('memoire') else '-':>9s}")
        f = st.fmean(kf) if kf else None
        if f is not None and f > 0:
            print(f"    Experimental sequential fraction (Karp-Flatt, mean over 2 to "
                  f"{coeurs_physiques or 'P'} processes): f = {f:.3f}; Amdahl bound 1/f = {1 / f:.1f}")
        sortie[inst] = {'tableau': tab, 'fraction_sequentielle': f,
                        'borne_amdahl': (1 / f) if f and f > 0 else None}
    return sortie


# -----------------------------------------------------------------------------
# 2 et 3. TEMPS POUR ATTEINDRE UNE CIBLE, QUALITE A TEMPS FIXE
# -----------------------------------------------------------------------------
def mesurer_cibles(fichier: str) -> None:
    faits = {(r['instance'], r['P'], r['graine']) for r in journal(fichier)}
    for inst, (ref, provenance) in REFERENCES.items():
        K, n, d = lire_instance(inst)
        circuit = est_circuit(inst)
        depart = plus_proche_voisin(d, K)
        nom = os.path.splitext(os.path.basename(inst))[0]
        for P in P_CIBLE:
            a_faire = [g for g in GRAINES if (nom, P, g) not in faits]
            if not a_faire:
                continue
            print(f"\n  {nom}: P = {P}, {len(a_faire)} run(s) of {DUREE_CIBLE_S} s", flush=True)
            pool = ouvrir_pool(P, d, K, n, circuit)
            try:
                for g in a_faire:
                    r = C.recherche_cooperative(pool, P, DUREE_CIBLE_S, depart, d, circuit,
                                                politique=METHODE, graine=g, n_epoques=N_EPOQUES,
                                                instants_releves=list(INSTANTS_RELEVES))
                    valide = W.est_valide(r['routes'], K, n, True) and \
                        abs(W.latence(r['routes'], d, circuit) - r['latence']) < 1e-6
                    enr = {'instance': nom, 'n_clients': n - 1, 'K': K, 'P': P, 'graine': g,
                           'reference': ref, 'latence': r['latence'], 'valide': bool(valide),
                           'ameliorations': r['ameliorations'], 'releves': r['releves'],
                           'latence_depart': r['latence_depart'], 'evaluations': r['evaluations']}
                    ecrire(fichier, enr)
                    print(f"    seed {g}  latency {r['latence']:11.2f}  gap "
                          f"{(r['latence'] - ref) / ref * 100:+6.2f} %  {'' if valide else 'INFEASIBLE'}",
                          flush=True)
            finally:
                pool.shutdown()


def temps_cible(r: Dict[str, Any], cible: float) -> Tuple[float, bool]:
    """First instant at which the best collective solution reaches the target;
    (DUREE_CIBLE_S, False) si elle ne l'atteint pas."""
    if r['latence_depart'] <= cible:
        return 0.0, True
    for t, lat in r['ameliorations']:
        if lat <= cible + 1e-9:
            return float(t), True
    return float(DUREE_CIBLE_S), False


def mediane_kaplan_meier(obs: List[Tuple[float, bool]]) -> Optional[float]:
    """Kaplan-Meier median; None when fewer than half the runs reach the target."""
    n = len(obs)
    survie = 1.0
    for t in sorted({t for t, ev in obs if ev}):
        a_risque = sum(1 for u, _ in obs if u >= t)
        morts = sum(1 for u, ev in obs if ev and u == t)
        survie *= 1 - morts / a_risque
        if survie <= 0.5:
            return t
    return None


def analyser_cibles(enr: List[Dict[str, Any]]) -> Dict[str, Any]:
    enr = [r for r in enr if r['valide']]
    sortie = {}
    accel_globales = {P: [] for P in P_CIBLE}
    for inst in sorted({r['instance'] for r in enr}):
        ref = next(r['reference'] for r in enr if r['instance'] == inst)
        print(f"\n  {inst}  (reference {ref})")
        sortie[inst] = {}
        for delta in ECARTS_CIBLES:
            cible = ref * (1 + delta)
            ligne = {}
            for P in P_CIBLE:
                obs = [temps_cible(r, cible) for r in enr if r['instance'] == inst and r['P'] == P]
                if not obs:
                    continue
                ligne[P] = {'succes': sum(ev for _, ev in obs) / len(obs),
                            'mediane_s': mediane_kaplan_meier(obs),
                            'temps': [t for t, _ in obs], 'atteinte': [ev for _, ev in obs]}
            base = ligne.get(1, {}).get('mediane_s')
            morceaux = []
            for P, l in ligne.items():
                m = l['mediane_s']
                acc = (base / m) if (base and m and m > 0) else None
                l['acceleration'] = acc
                if acc is not None and P > 1:
                    accel_globales[P].append(acc)
                morceaux.append(f"P={P}: {l['succes'] * 100:3.0f} %, "
                                f"{(f'{m:6.1f} s' if m is not None else '   non'):>8s}"
                                + (f" (x{acc:.2f})" if acc is not None and P > 1 else ""))
            print(f"    cible +{delta * 100:.1f} % ({cible:.2f}) | " + " | ".join(morceaux))
            sortie[inst][str(delta)] = ligne
        print("    mean gap to the reference against time: " + " | ".join(
            f"P={P}: " + ", ".join(
                f"{s} s {st.fmean([(r['releves'][str(s)] - ref) / ref * 100 for r in enr if r['instance'] == inst and r['P'] == P]):+.2f} %"
                for s in INSTANTS_RELEVES
                if [r for r in enr if r['instance'] == inst and r['P'] == P])
            for P in P_CIBLE if [r for r in enr if r['instance'] == inst and r['P'] == P]))
    print("\n  Median time-to-target speedup, geometric mean over the (instance, target)")
    print("  pairs for which both P = 1 and P reach the target in at least half the runs:")
    for P in P_CIBLE[1:]:
        v = accel_globales[P]
        if v:
            print(f"    P = {P} : x{math.exp(st.fmean([math.log(a) for a in v])):.2f}  ({len(v)} couples)")
        else:
            print(f"    P = {P}: not computable")
    return sortie


def figures(debit: Dict[str, Any], cibles_enr: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib not available: figures not produced.")
        return
    dossier = _abs(DOSSIER)
    if debit:
        # Figure labels are in English, since the figure is used in the paper.
        fig, ax = plt.subplots(figsize=(5, 4))
        pmax = 1
        for inst, v in debit.items():
            tab = v['tableau']
            ax.plot([t['P'] for t in tab], [t['acceleration'] for t in tab], 'o-', label=inst)
            pmax = max(pmax, max(t['P'] for t in tab))
            f = v.get('fraction_sequentielle')
            if f and f > 0:
                ps = [1 + i * (pmax - 1) / 50 for i in range(51)]
                ax.plot(ps, [1 / (f + (1 - f) / q) for q in ps], ':', lw=1,
                        label=f'Amdahl, f = {f:.3f} ({inst})')
        ax.plot([1, pmax], [1, pmax], 'k--', lw=1, label='linear speedup')
        ax.set_xlabel('Number of processes P')
        ax.set_ylabel('Throughput speedup (evaluations per second)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(dossier, 'fig_acceleration_debit.png'), dpi=200)
        plt.close(fig)
    enr = [r for r in cibles_enr if r['valide']]
    for inst in sorted({r['instance'] for r in enr}):
        ref = next(r['reference'] for r in enr if r['instance'] == inst)
        cible = ref * (1 + ECARTS_CIBLES[1])
        fig, ax = plt.subplots(figsize=(5, 4))
        for P in P_CIBLE:
            obs = sorted(temps_cible(r, cible) for r in enr if r['instance'] == inst and r['P'] == P)
            if not obs:
                continue
            n = len(obs)
            xs = [t for t, ev in obs if ev]
            ys = [(i + 1) / n for i in range(len(xs))]
            # The final plateau is extended to the time limit.
            ax.step([0] + xs + [DUREE_CIBLE_S], [0] + ys + [ys[-1] if ys else 0], where='post',
                    label=f'P = {P}')
        ax.set_xlabel('Wall-clock time (s)')
        ax.set_ylabel('Fraction of runs that reached the target')
        ax.set_title(f'{inst}, target +{ECARTS_CIBLES[1] * 100:.1f} %', fontsize=9)
        ax.set_xlim(0, DUREE_CIBLE_S)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(dossier, f'fig_temps_cible_{inst}.png'), dpi=200)
        plt.close(fig)
    print(f"  Figures written to {dossier}")


def main() -> int:
    os.makedirs(_abs(DOSSIER), exist_ok=True)
    info = materiel()
    json.dump(info, open(os.path.join(_abs(DOSSIER), 'hardware.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print("=" * 84)
    print("EVALUATION DU PARALLELISME")
    print("=" * 84)
    print(f"  {info['processeur'] or info['systeme']} ; coeurs logiques {info['coeurs_logiques']}, "
          f"physiques {info.get('coeurs_physiques')} ; Python {info['python']} ; methode {METHODE}")
    for inst in list(INSTANCES_DEBIT) + list(REFERENCES):
        if not os.path.exists(_abs(inst)):
            print(f"  Instance introuvable : {inst}")
            return 1

    f_debit = os.path.join(_abs(DOSSIER), 'throughput.jsonl')
    f_cibles = os.path.join(_abs(DOSSIER), 'targets.jsonl')
    print(f"\n1. DEBIT : {len(INSTANCES_DEBIT)} instance(s), P = {P_DEBIT}, {REPETITIONS_DEBIT} repetitions de "
          f"{DUREE_DEBIT_S} s par mode")
    mesurer_debit(f_debit)
    print(f"\n2. TIME TO REACH A TARGET: {len(REFERENCES)} instance(s), P = {P_CIBLE}, "
          f"{len(GRAINES)} graines, {DUREE_CIBLE_S} s")
    mesurer_cibles(f_cibles)

    print("\n" + "=" * 84)
    print("ANALYSIS")
    print("=" * 84)
    print("\n1. Debit, acceleration et fraction sequentielle (mode cadre ; P = 1 en reference)")
    debit = analyser_debit(journal(f_debit), info.get('coeurs_physiques'))
    print("\n2 and 3. Time to reach a target (success rate, Kaplan-Meier median, speedup)")
    cibles = analyser_cibles(journal(f_cibles))
    json.dump({'materiel': info, 'debit': debit, 'cibles': cibles},
              open(os.path.join(_abs(DOSSIER), 'summary.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False, default=str)
    figures(debit, journal(f_cibles))
    invalides = [r for r in journal(f_cibles) if not r['valide']]
    print(f"\n  {len(invalides)} infeasible solution(s).")
    return 0 if not invalides else 1


if __name__ == '__main__':
    code = main()
    print(f"\nCode de retour : {code}")
