"""
=============================================================================
 lcl_pricer.py  --  runtime side of the system (what the web app / API calls)
=============================================================================
 Training is offline; here nothing is learned.  At quote time the tool
   1. runs the EP heuristic LIVE against the two containers' current stowage
      -> can this box physically go in C1 / C2 ?
   2. looks the discount up in the trained table
      -> state = (t_bin, cap_own_bin, cap_other_bin)  ->  d*
   3. returns fare = r_S * (1 - d*) * CBM

 Everything the backend must persist is in  BookingEngine.state()  (JSON).

 Porting to JS/TS: only two things need porting -- the EPContainer class from
 lcl_core.py and the ~30 lines of table lookup below.  policy_bank.json is
 plain JSON with a flat integer array per policy.
=============================================================================
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

import lcl_core as L

UNIT_TO_DM = {'dm': 1.0, 'cm': 0.1, 'mm': 0.01, 'm': 10.0, 'in': 0.254}


# =============================================================================
class PolicyBank:
    """policy_bank.json + nearest-neighbour selection on (beta, delta_F, season)."""

    def __init__(self, path_or_obj):
        self.bank = (json.load(open(path_or_obj))
                     if isinstance(path_or_obj, str) else path_or_obj)
        self.env = self.bank
        st = self.bank['state']
        self.T_BINS, self.CAP_BINS = st['t_bins'], st['cap_bins']
        self.actions = self.bank['actions']
        self.r_S = float(self.bank['r_S'])
        self.T = float(self.bank['horizon'])
        self.cbm = float(self.bank['container']['cbm'])
        self.betas = sorted(self.bank['axes']['beta'])
        self.dFs = sorted(self.bank['axes']['delta_F'])

    # ---- selection ----------------------------------------------------------
    def _snap(self, x, grid):
        return float(min(grid, key=lambda g: abs(g - x)))

    def select(self, beta, delta_F, season=None):
        """season = ('Peak','OffPeak') | 'Peak/OffPeak' | None -> pooled 'ANY'."""
        b = self._snap(beta, self.betas)
        f = self._snap(delta_F, self.dFs)
        if season is None:
            sc = 'ANY'
        elif isinstance(season, str):
            sc = season.replace('/', '-')
        else:
            sc = f'{season[0]}-{season[1]}'
        key = f'b{b:.2f}|f{f:+.3f}|{sc}'
        if key not in self.bank['policies']:
            key = f'b{b:.2f}|f{f:+.3f}|ANY'
        if key not in self.bank['policies']:
            raise KeyError(f'no policy for beta={beta}, delta_F={delta_F}, '
                           f'season={season}')
        return Policy(self, self.bank['policies'][key],
                      snapped=(b, f, sc), requested=(beta, delta_F))

    def keys(self):
        return sorted(self.bank['policies'])


class Policy:
    def __init__(self, bank: PolicyBank, rec: dict, snapped=None, requested=None):
        self.bank, self.rec = bank, rec
        self.key = rec['key']
        self.snapped, self.requested = snapped, requested
        p = rec['policy']
        self.act = np.asarray(p['d_star_action'], dtype=int)
        self.trained = np.asarray(p['trained'], dtype=bool)
        self.visits = np.asarray(p['visits'], dtype=int)
        self.q = np.asarray(p['q'], dtype=float) if 'q' in p else None
        self.d_static = rec.get('d_static_best')

    def index(self, t_bin, own_bin, oth_bin):
        cb = self.bank.CAP_BINS
        return (t_bin * cb + own_bin) * cb + oth_bin

    def lookup(self, t_bin, own_bin, oth_bin):
        i = self.index(t_bin, own_bin, oth_bin)
        a = int(self.act[i])
        spec = self.bank.actions[a]
        return {'action_id': a,
                'offer': spec['type'] != 'no_offer',
                'd': float(spec.get('d', 0.0)),
                'trained': bool(self.trained[i]),
                'visits': int(self.visits[i]),
                'q': None if self.q is None else self.q[i].tolist(),
                'state_index': i}


# =============================================================================
class BookingEngine:
    """Two containers + live EP gate + learned discount lookup."""

    def __init__(self, bank: PolicyBank, beta: float, delta_F: float,
                 season=None, r_S: Optional[float] = None,
                 containers: Optional[Sequence[L.EPContainer]] = None,
                 cutoff_periods: Optional[float] = None):
        self.bank = bank
        self.policy = bank.select(beta, delta_F, season)
        self.beta, self.delta_F, self.season = beta, delta_F, season
        self.r_S = bank.r_S if r_S is None else float(r_S)   # real KRW/CBM here
        self.T = bank.T if cutoff_periods is None else float(cutoff_periods)
        self.c = list(containers) if containers else [L.EPContainer(),
                                                      L.EPContainer()]

    # ---- persistence --------------------------------------------------------
    def state(self):
        return {'policy_key': self.policy.key, 'beta': self.beta,
                'delta_F': self.delta_F, 'season': self.season,
                'r_S': self.r_S, 'T': self.T,
                'containers': [c.to_dict() for c in self.c]}

    @classmethod
    def from_state(cls, bank: PolicyBank, st: dict):
        return cls(bank, st['beta'], st['delta_F'], st.get('season'),
                   r_S=st.get('r_S'),
                   containers=[L.EPContainer.from_dict(d) for d in st['containers']],
                   cutoff_periods=st.get('T'))

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _dims_dm(dims, unit='dm'):
        k = UNIT_TO_DM[unit]
        return tuple(float(x) * k for x in dims)

    def _bins(self, t, own, oth):
        tb = min(int(max(0.0, t) / self.T * self.bank.T_BINS), self.bank.T_BINS - 1)
        f = lambda r: min(max(int(max(0.0, r) / self.bank.cbm * self.bank.CAP_BINS),
                              0), self.bank.CAP_BINS - 1)
        return tb, f(self.c[own].residual_cbm), f(self.c[oth].residual_cbm)

    # ---- the one call the UI makes -----------------------------------------
    def quote(self, dims, segment: int, t: float, unit: str = 'dm',
              billed_cbm: Optional[float] = None):
        """dims        (L,W,H) of the box in `unit`
           segment     1 or 2 -- the container / sailing the shipper prefers
           t           position in the selling horizon, 0 .. T
           billed_cbm  override (W/M rule etc.); defaults to the geometric CBM
        """
        d3 = self._dims_dm(dims, unit)
        cbm = float(np.prod(d3) / 1000.0) if billed_cbm is None else float(billed_cbm)
        own, oth = segment - 1, 2 - segment
        ors = L.box_orientations(*d3)

        slot_own = self.c[own].find_slot(ors) if self.c[own].residual_cbm >= cbm else None
        slot_oth = self.c[oth].find_slot(ors) if self.c[oth].residual_cbm >= cbm else None

        tb, ob, xb = self._bins(t, own, oth)
        pol = self.policy.lookup(tb, ob, xb)

        avail_F = (slot_own is not None) or (slot_oth is not None)
        offer_F = avail_F and pol['offer']
        if offer_F:
            order = ([own, oth] if self.c[own].residual_cbm >= self.c[oth].residual_cbm
                     else [oth, own])
            slots = {own: slot_own, oth: slot_oth}
            assign = next(k for k in order if slots[k] is not None)
        else:
            assign = None

        d = pol['d'] if offer_F else 0.0
        out = {
            'box': {'dims_dm': d3, 'cbm': round(cbm, 4)},
            'state': {'t_bin': tb, 'cap_own_bin': ob, 'cap_other_bin': xb,
                      'index': pol['state_index'],
                      'residual_cbm': [round(self.c[0].residual_cbm, 3),
                                       round(self.c[1].residual_cbm, 3)]},
            'standard': {
                'available': slot_own is not None,
                'container': f'C{segment}',
                'rate_per_cbm': round(self.r_S, 2),
                'price': round(self.r_S * cbm, 2),
                'reason': None if slot_own is not None else
                          ('volume exhausted' if self.c[own].residual_cbm < cbm
                           else 'no feasible EP placement (shape)'),
            },
            'flexible': {
                'offered': offer_F,
                'discount': round(d, 3),
                'rate_per_cbm': round(self.r_S * (1 - d), 2),
                'price': round(self.r_S * (1 - d) * cbm, 2),
                'assign_preview': None if assign is None else f'C{assign+1}',
                'reason': None if offer_F else
                          ('no feasible EP placement in either container'
                           if not avail_F else
                           'policy withholds the flexible product in this state'),
            },
            'policy': {'key': self.policy.key, 'action_id': pol['action_id'],
                       'trained': pol['trained'], 'visits': pol['visits'],
                       'd_static_benchmark': self.policy.d_static,
                       'q_values': pol['q']},
            '_slots': {'own': slot_own, 'oth': slot_oth, 'own_idx': own,
                       'oth_idx': oth, 'assign': assign, 'cbm': cbm},
        }
        return out

    def accept(self, quote: dict, product: str):
        """product 'S' or 'F'.  Commits the placement; returns the container."""
        s = quote['_slots']
        if product == 'S':
            if s['own'] is None:
                raise ValueError('standard product not available')
            k, d = s['own_idx'], 0.0
            slot = s['own']
        elif product == 'F':
            if not quote['flexible']['offered']:
                raise ValueError('flexible product not offered in this state')
            k = s['assign']
            slot = s['own'] if k == s['own_idx'] else s['oth']
            d = quote['flexible']['discount']
        else:
            raise ValueError("product must be 'S' or 'F'")
        self.c[k].commit(slot, s['cbm'])
        return {'container': f'C{k+1}', 'cbm': s['cbm'], 'discount': d,
                'revenue': round(self.r_S * (1 - d) * s['cbm'], 2),
                'load_factor': [round(self.c[0].used_cbm / self.bank.cbm, 4),
                                round(self.c[1].used_cbm / self.bank.cbm, 4)]}

    # ---- convenience for a dashboard ---------------------------------------
    def summary(self):
        return {'load_factor': [round(c.used_cbm / self.bank.cbm, 4) for c in self.c],
                'residual_cbm': [round(c.residual_cbm, 3) for c in self.c],
                'n_items': [c.n_items for c in self.c]}

    def price_grid(self, segment: int):
        """d*(t_bin, cap_own_bin) for the current other-container bin --
        handy for rendering the rate card in the UI."""
        oth = 2 - segment
        xb = min(max(int(self.c[oth].residual_cbm / self.bank.cbm
                         * self.bank.CAP_BINS), 0), self.bank.CAP_BINS - 1)
        g = []
        for tb in range(self.bank.T_BINS):
            g.append([self.policy.lookup(tb, ob, xb) for ob in range(self.bank.CAP_BINS)])
        return g


# =============================================================================
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/policy_bank.json'
    bank = PolicyBank(path)
    eng = BookingEngine(bank, beta=0.6, delta_F=-0.3, season=('Normal', 'OffPeak'),
                        r_S=95000.0)
    rng = np.random.default_rng(0)
    beta, delta_F = 0.6, -0.3
    total_rev = 0.0
    for i in range(12):
        t    = i / 12 * bank.T
        v    = rng.uniform(0.3, 2.0)
        dims = L.make_box_dims(v, rng)
        seg  = int(rng.integers(1, 3))
        q    = eng.quote(dims, seg, t)

        avail_S = q['standard']['available']
        avail_F = q['flexible']['offered']
        d       = q['flexible']['discount']

        # MNL 확률로 화주 선택 결정 (eq.2)
        choice = L.mnl_choice(avail_S, False, avail_F, d, beta, delta_F,
                              rng.random())
        if choice == 'S_own' and avail_S:
            rec    = eng.accept(q, 'S')
            booked = 'S'
        elif choice == 'F' and avail_F:
            rec    = eng.accept(q, 'F')
            booked = 'F'
        else:
            rec    = None
            booked = '-'   # 미구매

        d_used = d if booked == 'F' else 0.0
        rev    = rec['revenue'] if rec else 0.0
        total_rev += rev
        print(f"t={t:5.1f} seg=C{seg} {v:.2f}CBM  "
              f"booked={booked}  d={d_used:.2f}  "
              f"cont={rec['container'] if rec else '-':2}  "
              f"rev={rev:>10,.0f}")

    print(f"\n적재율  C1={eng.c[0].used_cbm/bank.cbm:.3f}  "
          f"C2={eng.c[1].used_cbm/bank.cbm:.3f}")
    print(f"총수익  {total_rev:,.0f}")
    print(json.dumps(eng.state())[:200] + ' ...')