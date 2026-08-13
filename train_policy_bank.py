"""
=============================================================================
 train_policy_bank.py  --  trains every (beta, delta_F, season) cell and
 exports ONE artefact that the pricing tool consumes:  policy_bank.json
=============================================================================
 usage
   python train_policy_bank.py --quick                 # ~10 min sanity run
   python train_policy_bank.py                         # full run
   python train_policy_bank.py --n-train 40000 --jobs 8
   python train_policy_bank.py --scenarios pooled      # deployment bank only

 The run is resumable: every finished cell is written to  shards/<key>.json,
 and re-running skips whatever is already there.
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import lcl_core as L

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# -----------------------------------------------------------------------------
def build_cells(mode):
    """mode: 'pooled' | 'main' | 'all'

    'ANY' is the season-agnostic policy trained with the demand level redrawn
    every episode -- that is the one the live tool should use when the desk
    does not want to declare a season.
    """
    seasons = []
    if mode in ('main', 'all'):
        seasons += list(itertools.product(L.LEVEL_NAMES, L.LEVEL_NAMES))
    if mode in ('pooled', 'all'):
        seasons += ['ANY']
    cells = []
    for beta in L.BETA_LEVELS:
        for dF in L.DELTA_F_LEVELS:
            for sc in seasons:
                cells.append((sc, float(beta), float(dF)))
    return cells


def cell_seed(sc, beta, dF):
    return abs(hash((str(sc), round(beta, 3), round(dF, 3)))) % 100000


def run_cell(args):
    sc, beta, dF, cfg = args
    key = L.policy_key(beta, dF, sc)
    shard = os.path.join(cfg['out'], 'shards', key.replace('|', '_') + '.json')
    if os.path.exists(shard) and not cfg['force']:
        return json.load(open(shard))

    t0 = time.time()
    seed = cell_seed(sc, beta, dF)
    Q, visits, curve = L.train_C(sc, beta, dF, n_train=cfg['n_train'], seed=seed)
    d_B, prof = L.tune_B(sc, beta, dF, n_tune=cfg['n_tune'], seed=seed + 1)
    res = L.evaluate(sc, beta, dF, Q, d_B, n_eval=cfg['n_eval'], seed=seed + 2)

    act, trained, vis = L.greedy_policy(Q, visits, d_B)
    q = Q[0] + Q[1] if L.DOUBLE_Q else Q[0]

    rec = {
        'key': key,
        'beta': beta, 'delta_F': dF,
        'season': 'ANY' if sc == 'ANY' else list(sc),
        'd_static_best': d_B,
        'policy': L.policy_to_json(act, trained, vis,
                                   q if cfg['export_q'] else None),
        'metrics': {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in res.items()},
        'diagnostics': {
            'curve': [round(x, 1) for x in curve.tolist()],
            'tuneB': [round(x, 1) for x in prof.tolist()],
            'coverage': float(trained.mean()),
            'secs': round(time.time() - t0, 1),
        },
    }
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    json.dump(rec, open(shard, 'w'))
    print(f"  [{key}]  A={res['A_rev']:.0f} B={res['B_rev']:.0f}(d*={d_B:.2f}) "
          f"C={res['C_rev']:.0f} | B-A={res['gap_BA']:+.2f}% C-B={res['gap_CB']:+.2f}% "
          f"C-A={res['gap_CA']:+.2f}%±{res['CA_ci95']:.2f} | LF {res['A_lf']:.3f}"
          f"->{res['C_lf']:.3f} | cover={trained.mean():.2f} "
          f"({rec['diagnostics']['secs']:.0f}s)", flush=True)
    return rec


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=50000)
    ap.add_argument('--n-eval', type=int, default=800)
    ap.add_argument('--n-tune', type=int, default=250)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--scenarios', choices=['pooled', 'main', 'all'], default='all')
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--export-q', action='store_true',
                    help='also ship the Q table (lets the tool show value-by-discount)')
    ap.add_argument('--force', action='store_true', help='ignore existing shards')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    if a.quick:
        a.n_train, a.n_eval, a.n_tune = 1500, 150, 60

    os.makedirs(a.out, exist_ok=True)
    cells = build_cells(a.scenarios)
    cfg = {'n_train': a.n_train, 'n_eval': a.n_eval, 'n_tune': a.n_tune,
           'out': a.out, 'export_q': a.export_q, 'force': a.force}

    print('=' * 96)
    print(f'LCL policy bank | {len(cells)} cells | n_train={a.n_train} '
          f'n_eval={a.n_eval} n_tune={a.n_tune} | jobs={a.jobs}')
    print(f'container={L.CONT_CBM:.2f} CBM  T={L.T_HORIZON:.0f}  A0={L.A0}  '
          f'orientations={L.N_ORIENT}  support={L.SUPPORT_FRAC}')
    print('=' * 96)

    t0 = time.time()
    jobs = [(sc, b, d, cfg) for (sc, b, d) in cells]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            recs = pool.map(run_cell, jobs)
    else:
        recs = [run_cell(j) for j in jobs]

    # ---- the single artefact the tool loads ---------------------------------
    bank = {
        **L.env_manifest(),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'training': {'n_train': a.n_train, 'n_eval': a.n_eval,
                     'n_tune': a.n_tune, 'gamma': L.GAMMA,
                     'double_q': L.DOUBLE_Q},
        'policies': {r['key']: r for r in recs},
    }
    path = os.path.join(a.out, 'policy_bank.json')
    json.dump(bank, open(path, 'w'))
    print(f'\npolicy bank -> {path}  ({os.path.getsize(path)/1e6:.2f} MB, '
          f'{len(recs)} policies)')

    # ---- tidy result tables for the paper -----------------------------------
    rows = []
    for r in recs:
        sc = r['season']
        rows.append({'season': 'ANY' if sc == 'ANY' else f'{sc[0]}/{sc[1]}',
                     'level1': 'ANY' if sc == 'ANY' else sc[0],
                     'level2': 'ANY' if sc == 'ANY' else sc[1],
                     'beta': r['beta'], 'delta_F': r['delta_F'],
                     'd_B': r['d_static_best'],
                     'coverage': r['diagnostics']['coverage'],
                     **r['metrics']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'results_all_cells.csv'), index=False)

    m = df[df.season != 'ANY']
    if len(m):
        m.groupby(['level1', 'level2'])[
            ['gap_BA', 'gap_CB', 'gap_CA', 'A_lf', 'C_lf', 'A_imbalance',
             'C_imbalance', 'cannibalisation', 'C_flex_share_cbm',
             'C_no_offer_rate', 'd_B']].mean().round(4).to_csv(
            os.path.join(a.out, 'table_by_scenario.csv'))
        m.groupby('beta')[['gap_BA', 'gap_CB', 'gap_CA', 'd_B']].mean().round(4)\
            .to_csv(os.path.join(a.out, 'table_by_beta.csv'))
        m.groupby('delta_F')[['gap_BA', 'gap_CB', 'gap_CA']].mean().round(4)\
            .to_csv(os.path.join(a.out, 'table_by_deltaF.csv'))
        print('\n--- mean gaps (%) over the scenario cells ---')
        print(f"  B - A  {m.gap_BA.mean():+.2f}   "
              f"C - B  {m.gap_CB.mean():+.2f}   C - A  {m.gap_CA.mean():+.2f}")
        print(f"  LF  {m.A_lf.mean():.3f} -> {m.C_lf.mean():.3f} | "
              f"cannibalisation {m.cannibalisation.mean():.3f}")
        print(f"  corr(A_lf, C-A) = {np.corrcoef(m.A_lf, m.gap_CA)[0,1]:+.3f}  "
              f"corr(A_imbalance, C-A) = "
              f"{np.corrcoef(m.A_imbalance, m.gap_CA)[0,1]:+.3f}")
    print(f'\ntotal {(time.time()-t0)/60:.1f} min  ->  {a.out}')


if __name__ == '__main__':
    main()
