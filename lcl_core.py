"""
=============================================================================
 LCL Revenue Management  --  core engine  (paper Ch.3-5, faithful rebuild)
=============================================================================
 Notation follows the paper:
   C1,C2      two adjacent containers, V0 = 60 x 23 x 24 dm = 33.12 CBM
   S1,S2      standard products (container-designated)
   F          flexible product (assigned by the consolidator at the deadline)
   r_S        standard unit fare per CBM ,  r_F = (1-d) r_S
   V_j        = a_j - beta * (r_j / r_S)                          ... eq.(1)
              a_S1 = a_S2 = a_S ,  a_F = a_S + delta_F
   P(j|A_t)   = exp(V_j) / ( exp(V_0) + sum_{k in A_t} exp(V_k) ) ... eq.(2)
   s          = (t_bin, cap_own_bin, cap_other_bin)               ... 4.5.1
              capacities are RELATIVE to the arriving shipper's preferred
              container (paper 4.5.1), not absolute (C1,C2)
   a          = d in D, including the NO-OFFER action              ... 4.5.2
   Q update   eq.(5)

 Two deliberate implementation notes
 -----------------------------------
 (1) EP orientations.  Paper 4.4 allows only the two horizontal rotations.
     Here every extreme point is tried against ALL axis-aligned orientations
     (N_ORIENT = 6).  Set N_ORIENT = 2 to recover the paper's text.
 (2) Semi-MDP reward.  Decision epochs occur only when F is offerable.  The
     fares realised between two consecutive decision epochs (standard-product
     bookings) are ACCUMULATED into the reward of the earlier decision, so the
     agent internalises the opportunity cost of the capacity it gives away.
=============================================================================
"""
from __future__ import annotations

import json
import numpy as np

# =============================================================================
# 1. PARAMETERS
# =============================================================================

# ---- container geometry (3.1) : 60 x 23 x 24 dm = 33.12 CBM -----------------
CONT_L, CONT_W, CONT_H = 60.0, 23.0, 24.0          # decimetres
CONT_CBM = CONT_L * CONT_W * CONT_H / 1000.0       # 33.12
N_CONT = 2

# ---- EP heuristic (4.4) ------------------------------------------------------
N_ORIENT = 6            # 6 = all axis-aligned rotations, 2 = paper's text
SUPPORT_FRAC = 0.60     # base-area fraction that must rest on floor/box tops
MAX_EPS = 220           # cap on the extreme-point list (runtime guard)

# ---- selling horizon (3.1) ---------------------------------------------------
T_HORIZON = 40.0

# ---- cargo (5.2.1 step 4) ----------------------------------------------------
BOX_CBM_LOW, BOX_CBM_HIGH = 0.3, 2.0               # Uniform(0.3, 2.0) CBM
MODULE_NW = (2, 3)                                 # cross-section module grid
MODULE_NH = (2, 3)
IRREG_FRAC = 0.35                                  # 35 % irregular cargo
IRREG_SHRINK = 0.10                                # shrunk by up to 10 %
MAX_LEN_FRAC = 0.50                                # reject silly-long boxes

# ---- demand levels (5.2.1 step 1) : mean segment demand in CBM ---------------
MEAN_DEMAND = {'Peak': 62.0, 'Normal': 48.0, 'OffPeak': 32.0}
CV_DEMAND = 0.15                                   # sd / mean (not in paper)
LEVEL_NAMES = ('Peak', 'Normal', 'OffPeak')

# ---- fares (A4) --------------------------------------------------------------
R_S = 100.0                                        # standard fare per CBM

# ---- action grid (4.5.2 eq.4) : index 0 = NO OFFER --------------------------
D_GRID = np.array([0.00, 0.05, 0.10, 0.15, 0.20,
                   0.25, 0.30, 0.35, 0.40])
NO_OFFER = 0
N_ACTIONS = 1 + len(D_GRID)
D_FIXED_B = 0.15                                   # 5.3 policy B default

# ---- MNL (4.1) ---------------------------------------------------------------
#   eq.(1) is  V_j = a_j - beta*(r_j/r_S),  a_S1 = a_S2 = a_S,  a_F = a_S + dF.
#   Because r_S/r_S == 1 the price term is a constant for the standard product,
#   so we let a_S absorb it:   a_S = A0 + beta.   Then
#        V_S = A0 ,   V_F = A0 + delta_F + beta*d ,   V_0 = 0.
#   This keeps the standard-product purchase propensity invariant in beta, so
#   beta is purely the leverage of the flexible discount (otherwise the
#   baseline load factor would move with beta and confound the sweep).
#   A0 is calibrated so the Policy-A load factor lands in the band of 6.3
#   (0.63-0.73 vs an EP ceiling of ~0.77).  Re-check with calibrate_a0().
A0 = 1.5
V_NOBUY = 0.0
BETA_LEVELS = (0.2, 0.4, 0.6, 0.8)
DELTA_F_LEVELS = (0.0, -0.2 * A0)                  # a_F in {a_S, 0.8 a_S}
BETA_BASE, DELTA_F_BASE = 0.6, -0.2 * A0

# ---- offer-set reading (3.1 vs 4.1) -----------------------------------------
#   3.1 says the shipper chooses among {own standard, F, no-purchase}.
#   4.1/2.5 discuss S1, S2 and F coexisting in A_t under IIA.
#   False -> 3.1 reading.  The flexible product is the ONLY way to reach the
#            other container, so its redistribution value is large and B - A
#            can be positive at small d.
#   True  -> 4.1 reading.  The non-preferred container's standard product is
#            also offered at a utility penalty KAPPA, F then substitutes almost
#            purely against the standard products (the ~90 % cannibalisation
#            and the B < A result of 6.1/6.5).
OFFER_CROSS_STANDARD = False
KAPPA = 2.0

# ---- state discretisation (4.5.1) -------------------------------------------
T_BINS, CAP_BINS = 6, 6
N_STATES = T_BINS * CAP_BINS * CAP_BINS            # 216

# ---- Q-learning --------------------------------------------------------------
GAMMA = 1.0
ALPHA_START, ALPHA_END = 0.08, 0.002
EPS_START, EPS_END = 1.00, 0.05
Q_INIT_NOISE = 1e-3                                # breaks argmax ties
DOUBLE_Q = True


# =============================================================================
# 2. EP (EXTREME POINT) HEURISTIC  --  Crainic, Perboli & Tadei (2008)
#    used strictly as a binary feasibility gate (4.4)
# =============================================================================
def box_orientations(dx, dy, dz, n=None):
    """Axis-aligned orientations, de-duplicated.

    n = 6 : every rotation (all corner/axis pairings are tried)
    n = 2 : upright only, two horizontal rotations (paper 4.4)
    """
    n = N_ORIENT if n is None else n
    if n == 2:
        return list({(dx, dy, dz), (dy, dx, dz)})
    return list({(dx, dy, dz), (dx, dz, dy), (dy, dx, dz),
                 (dy, dz, dx), (dz, dx, dy), (dz, dy, dx)})


class EPContainer:
    """Placed items + extreme-point list for one container."""

    __slots__ = ('L', 'W', 'H', 'items', 'eps', 'used_cbm', 'n_items')

    def __init__(self, L=CONT_L, W=CONT_W, H=CONT_H):
        self.L, self.W, self.H = L, W, H
        self.items = []                      # (x,y,z,dx,dy,dz)
        self.eps = [(0.0, 0.0, 0.0)]
        self.used_cbm = 0.0                  # BILLED cbm (4.4 capacity tracking)
        self.n_items = 0

    # ---- serialisation (the web/app backend persists this) ------------------
    def to_dict(self):
        return {'L': self.L, 'W': self.W, 'H': self.H,
                'items': [list(i) for i in self.items],
                'eps': [list(p) for p in self.eps],
                'used_cbm': self.used_cbm, 'n_items': self.n_items}

    @classmethod
    def from_dict(cls, d):
        c = cls(d.get('L', CONT_L), d.get('W', CONT_W), d.get('H', CONT_H))
        c.items = [tuple(i) for i in d['items']]
        c.eps = [tuple(p) for p in d['eps']]
        c.used_cbm = float(d['used_cbm'])
        c.n_items = int(d['n_items'])
        return c

    # ---- geometry -----------------------------------------------------------
    def _inside(self, x, y, z, dx, dy, dz):
        return (x + dx <= self.L + 1e-9 and
                y + dy <= self.W + 1e-9 and
                z + dz <= self.H + 1e-9)

    def _overlaps(self, x, y, z, dx, dy, dz):
        e = 1e-9
        x2, y2, z2 = x + dx, y + dy, z + dz
        for (ix, iy, iz, idx, idy, idz) in self.items:
            if (x2 <= ix + e or ix + idx <= x + e or
                y2 <= iy + e or iy + idy <= y + e or
                z2 <= iz + e or iz + idz <= z + e):
                continue
            return True
        return False

    def _supported(self, x, y, z, dx, dy):
        """Floor or box tops must carry >= SUPPORT_FRAC of the base area."""
        if z <= 1e-9:
            return True
        need = SUPPORT_FRAC * dx * dy
        got = 0.0
        e = 1e-6
        for (ix, iy, iz, idx, idy, idz) in self.items:
            if abs(iz + idz - z) > e:
                continue
            ox = min(x + dx, ix + idx) - max(x, ix)
            oy = min(y + dy, iy + idy) - max(y, iy)
            if ox > 0 and oy > 0:
                got += ox * oy
                if got >= need:
                    return True
        return got >= need

    def _point_in_item(self, px, py, pz):
        e = 1e-6
        for (ix, iy, iz, idx, idy, idz) in self.items:
            if (ix - e < px < ix + idx - e and
                iy - e < py < iy + idy - e and
                iz - e < pz < iz + idz - e):
                return True
        return False

    def _project(self, px, py, pz, axis):
        """Project a corner backwards onto the nearest supporting face/wall."""
        e = 1e-9
        best = 0.0
        for (ix, iy, iz, idx, idy, idz) in self.items:
            if axis == 0:
                if (iy - e <= py < iy + idy - e) and (iz - e <= pz < iz + idz - e) \
                        and ix + idx <= px + e:
                    best = max(best, ix + idx)
            elif axis == 1:
                if (ix - e <= px < ix + idx - e) and (iz - e <= pz < iz + idz - e) \
                        and iy + idy <= py + e:
                    best = max(best, iy + idy)
            else:
                if (ix - e <= px < ix + idx - e) and (iy - e <= py < iy + idy - e) \
                        and iz + idz <= pz + e:
                    best = max(best, iz + idz)
        return best

    def _update_eps(self, x, y, z, dx, dy, dz):
        cand = []
        p = (x + dx, y, z)
        cand.append((p[0], self._project(*p, 1), p[2]))
        cand.append((p[0], p[1], self._project(*p, 2)))
        p = (x, y + dy, z)
        cand.append((self._project(*p, 0), p[1], p[2]))
        cand.append((p[0], p[1], self._project(*p, 2)))
        p = (x, y, z + dz)
        cand.append((self._project(*p, 0), p[1], p[2]))
        cand.append((p[0], self._project(*p, 1), p[2]))

        have = set(self.eps)
        for c in cand:
            if c[0] >= self.L - 1e-6 or c[1] >= self.W - 1e-6 or c[2] >= self.H - 1e-6:
                continue
            if c in have or self._point_in_item(*c):
                continue
            have.add(c)
            self.eps.append(c)

        self.eps = [p for p in self.eps if not self._point_in_item(*p)]
        self.eps.sort(key=lambda p: (p[2], p[1], p[0]))
        if len(self.eps) > MAX_EPS:
            self.eps = self.eps[:MAX_EPS]

    # ---- public API ---------------------------------------------------------
    def find_slot(self, orientations):
        """First feasible (x,y,z,dx,dy,dz) over all EPs x all orientations."""
        for (ex, ey, ez) in self.eps:
            for (dx, dy, dz) in orientations:
                if not self._inside(ex, ey, ez, dx, dy, dz):
                    continue
                if self._overlaps(ex, ey, ez, dx, dy, dz):
                    continue
                if not self._supported(ex, ey, ez, dx, dy):
                    continue
                return (ex, ey, ez, dx, dy, dz)
        return None

    def commit(self, slot, billed_cbm):
        x, y, z, dx, dy, dz = slot
        self.items.append(slot)
        self.n_items += 1
        self.used_cbm += billed_cbm          # 4.4: capacity tracked in billed CBM
        try:
            self.eps.remove((x, y, z))
        except ValueError:
            pass
        self._update_eps(x, y, z, dx, dy, dz)

    @property
    def residual_cbm(self):
        return CONT_CBM - self.used_cbm


# =============================================================================
# 3. DEMAND GENERATION PIPELINE  (5.2.1)
# =============================================================================
def make_box_dims(cbm, rng):
    """Cross-section drawn from the standard module grid; length from volume.

    35 % of boxes are shrunk by up to 10 % on the cross-section, which takes
    them off the grid and creates the unusable slivers that flip the EP gate
    to reject.  Shape affects loading feasibility only, never the fare (5.2.1).
    """
    vol = cbm * 1000.0                                  # dm^3
    w = h = ell = None
    for _ in range(8):
        nw = MODULE_NW[rng.integers(len(MODULE_NW))]
        nh = MODULE_NH[rng.integers(len(MODULE_NH))]
        w, h = CONT_W / nw, CONT_H / nh
        ell = vol / (w * h)
        if ell <= CONT_L * MAX_LEN_FRAC:
            break
    else:
        w, h = CONT_W / 2.0, CONT_H / 2.0
        ell = vol / (w * h)
    if rng.random() < IRREG_FRAC:
        w *= 1.0 - rng.random() * IRREG_SHRINK
        h *= 1.0 - rng.random() * IRREG_SHRINK
    return float(ell), float(w), float(h)


def generate_episode(level1, level2, rng):
    """One selling season.  All primitives are pre-drawn so that policies
    A / B / C are compared on the identical market (common random numbers)."""
    boxes = []
    for seg, lv in ((1, level1), (2, level2)):
        mu = MEAN_DEMAND[lv]
        total = max(0.0, rng.normal(mu, CV_DEMAND * mu))
        acc = 0.0
        while acc < total:
            v = rng.uniform(BOX_CBM_LOW, BOX_CBM_HIGH)
            boxes.append((rng.random() * T_HORIZON, v, seg))
            acc += v
    boxes.sort(key=lambda b: b[0])                      # merge by arrival time
    n = len(boxes)
    tm = np.array([b[0] for b in boxes])
    cbm = np.array([b[1] for b in boxes])
    seg = np.array([b[2] for b in boxes], dtype=np.int8)
    dims = [make_box_dims(c, rng) for c in cbm]
    u = rng.random(n)                                   # CRN for the MNL draw
    return {'n': n, 'time': tm, 'cbm': cbm, 'seg': seg, 'dims': dims, 'u': u,
            'level1': level1, 'level2': level2}


# =============================================================================
# 4. MNL CHOICE  (4.1, eq. 1-2)
#    offer set = {own standard (if EP-feasible), F (if offered & EP-feasible)}
#    the non-preferred container's standard product is never offered (3.1)
# =============================================================================
def mnl_choice(avail_own, avail_oth, avail_F, d, beta, delta_F, u, a0=A0):
    """Returns 'S_own' | 'S_oth' | 'F' | None."""
    exps, keys = [], []
    if avail_own:
        exps.append(np.exp(a0))                         # V_S = A0
        keys.append('S_own')
    if avail_oth and OFFER_CROSS_STANDARD:
        exps.append(np.exp(a0 - KAPPA))                 # non-preferred sailing
        keys.append('S_oth')
    if avail_F:
        exps.append(np.exp(a0 + delta_F + beta * d))    # V_F = A0 + dF + beta*d
        keys.append('F')
    if not keys:
        return None
    denom = np.exp(V_NOBUY) + sum(exps)
    c = 0.0
    for k, e in zip(keys, exps):
        c += e / denom
        if u < c:
            return k
    return None


# =============================================================================
# 5. STATE DISCRETISATION  (4.5.1)   own / other, not C1 / C2
# =============================================================================
def _cap_bin(residual):
    b = int(max(0.0, residual) / CONT_CBM * CAP_BINS)
    return min(max(b, 0), CAP_BINS - 1)


def get_state(t, res_own, res_oth):
    tb = min(int(t / T_HORIZON * T_BINS), T_BINS - 1)
    return tb, _cap_bin(res_own), _cap_bin(res_oth)


def state_index(s):
    return (s[0] * CAP_BINS + s[1]) * CAP_BINS + s[2]


# =============================================================================
# 6. EPISODE SIMULATOR
#    policy 'A' no flexible | 'B' constant d | 'C' Q-learning
# =============================================================================
def run_episode(ep, policy, beta, delta_F, Q=None, d_fixed=D_FIXED_B,
                eps_greedy=0.0, learn=False, rng=None, alpha=ALPHA_START,
                visits=None, a0=A0, trace=False, Vbase=None, collect_V=False):
    c = [EPContainer(), EPContainer()]
    rev = 0.0
    n_book = {'S': 0, 'F': 0}
    cbm_book = {'S': 0.0, 'F': 0.0}
    n_lost = n_ep_blocked = 0
    n_arr = ep['n']
    d_used, act_hist = [], np.zeros(N_ACTIONS, dtype=np.int64)
    booked_flag = np.zeros(n_arr, dtype=bool)
    booked_prod = np.empty(n_arr, dtype=object)

    prev_s = prev_a = None
    pending = 0.0                       # semi-MDP reward accumulator
    vtrace = []                         # (state, revenue realised before it)

    for i in range(n_arr):
        t = ep['time'][i]
        v = ep['cbm'][i]
        own = int(ep['seg'][i]) - 1
        oth = 1 - own
        ors = box_orientations(*ep['dims'][i])

        # ---- EP feasibility drives the available set (4.1) -----------------
        slot_own = c[own].find_slot(ors) if c[own].residual_cbm >= v else None
        slot_oth = c[oth].find_slot(ors) if c[oth].residual_cbm >= v else None
        avail_S = slot_own is not None
        avail_oth = (slot_oth is not None) and OFFER_CROSS_STANDARD
        avail_F = (policy != 'A') and (slot_own is not None or slot_oth is not None)
        if slot_own is None and c[own].residual_cbm >= v:
            n_ep_blocked += 1           # blocked by shape, not by volume

        # ---- action (4.5.2) -------------------------------------------------
        s = a_idx = None
        d = d_fixed
        if policy == 'C' and avail_F:
            s = get_state(t, c[own].residual_cbm, c[oth].residual_cbm)
            if rng is not None and rng.random() < eps_greedy:
                a_idx = int(rng.integers(N_ACTIONS))
            else:
                q = Q[0][s] + Q[1][s] if DOUBLE_Q else Q[0][s]
                a_idx = int(np.argmax(q))
            act_hist[a_idx] += 1
            if a_idx == NO_OFFER:
                avail_F = False
                d = 0.0
            else:
                d = float(D_GRID[a_idx - 1])
        elif policy == 'B' and avail_F:
            act_hist[1 + int(np.argmin(np.abs(D_GRID - d_fixed)))] += 1
            if collect_V:
                s = get_state(t, c[own].residual_cbm, c[oth].residual_cbm)

        # ---- choice (4.1) ---------------------------------------------------
        if not (avail_S or avail_oth or avail_F):
            n_lost += 1
            choice = None
        else:
            choice = mnl_choice(avail_S, avail_oth, avail_F, d, beta, delta_F,
                                ep['u'][i], a0)

        # ---- placement + fare (4.2, 4.3) -----------------------------------
        r_t = 0.0
        if choice in ('S_own', 'S_oth'):
            k = own if choice == 'S_own' else oth
            c[k].commit(slot_own if choice == 'S_own' else slot_oth, v)
            r_t = R_S * v
            n_book['S'] += 1
            cbm_book['S'] += v
            booked_flag[i] = True
            booked_prod[i] = 'S'
        elif choice == 'F':
            # greedy: the container with the larger residual volume first (4.3)
            order = [own, oth] if c[own].residual_cbm >= c[oth].residual_cbm else [oth, own]
            slot = {own: slot_own, oth: slot_oth}
            for k in order:
                if slot[k] is not None:
                    c[k].commit(slot[k], v)
                    r_t = R_S * (1.0 - d) * v
                    n_book['F'] += 1
                    cbm_book['F'] += v
                    booked_flag[i] = True
                    booked_prod[i] = 'F'
                    d_used.append(d)
                    break
        if collect_V and s is not None:
            vtrace.append((s, rev))
        rev += r_t
        pending += r_t

        # ---- Q update, eq.(5), with accumulated reward ----------------------
        if learn and policy == 'C' and s is not None:
            if prev_s is not None:
                rr = pending - r_t
                if Vbase is not None:            # potential-based shaping
                    rr += Vbase[s] - Vbase[prev_s]
                _q_update(Q, prev_s, prev_a, rr, s, alpha, rng)
            if visits is not None:
                visits[s][a_idx] += 1
            prev_s, prev_a = s, a_idx
            pending = r_t

    if learn and policy == 'C' and prev_s is not None:
        rr = pending - (Vbase[prev_s] if Vbase is not None else 0.0)
        _q_update(Q, prev_s, prev_a, rr, None, alpha, rng)

    lf1, lf2 = c[0].used_cbm / CONT_CBM, c[1].used_cbm / CONT_CBM
    tot = c[0].used_cbm + c[1].used_cbm
    out = {'rev': rev, 'lf': tot / (2 * CONT_CBM), 'lf1': lf1, 'lf2': lf2,
           'imbalance': abs(lf1 - lf2),
           'ep_blocked_rate': n_ep_blocked / max(1, n_arr),
           'lost_rate': n_lost / max(1, n_arr),
           'n_arr': n_arr,
           'flex_share_n': n_book['F'] / max(1, n_book['S'] + n_book['F']),
           'flex_share_cbm': cbm_book['F'] / max(1e-9, tot),
           'avg_unit_price': rev / max(1e-9, tot),
           'mean_d': float(np.mean(d_used)) if d_used else np.nan,
           'no_offer_rate': act_hist[NO_OFFER] / max(1, act_hist.sum()),
           'booked_flag': booked_flag, 'booked_prod': booked_prod}
    if trace:
        out['containers'] = c
    if collect_V:
        out['vtrace'] = [(st, rev - before) for (st, before) in vtrace]
    return out


def _q_update(Q, s, a, r, s2, alpha, rng):
    k = int(rng.integers(2)) if (DOUBLE_Q and rng is not None) else 0
    o = 1 - k if DOUBLE_Q else 0
    if s2 is None:
        nxt = 0.0
    else:
        a_star = int(np.argmax(Q[k][s2]))
        nxt = Q[o][s2][a_star]
    Q[k][s][a] += alpha * (r + GAMMA * nxt - Q[k][s][a])


# =============================================================================
# 7. TRAINING / TUNING / EVALUATION
# =============================================================================
def new_Q(seed=0):
    rng = np.random.default_rng(seed)
    shape = (T_BINS, CAP_BINS, CAP_BINS, N_ACTIONS)
    return [rng.normal(0, Q_INIT_NOISE, shape), rng.normal(0, Q_INIT_NOISE, shape)]


def _draw_levels(scenario, rng):
    """scenario = ('Peak','OffPeak')  or  'ANY' (pooled / season-agnostic)."""
    if scenario == 'ANY':
        return (LEVEL_NAMES[rng.integers(3)], LEVEL_NAMES[rng.integers(3)])
    return scenario


def estimate_baseline_V(scenario, beta, delta_F, d=D_FIXED_B, n=2500, seed=99,
                        a0=A0):
    """Monte-Carlo estimate of the remaining revenue from each decision state
    under a fixed-discount baseline.  Used as the potential Phi(s) for
    potential-based reward shaping (Ng, Harada & Russell, 1999), which is
    policy-invariant: it removes the large state-dependent constant from Q and
    leaves the agent to learn only the ACTION ADVANTAGES, which are two orders
    of magnitude smaller than the returns themselves."""
    rng = np.random.default_rng(seed)
    tot = np.zeros((T_BINS, CAP_BINS, CAP_BINS))
    cnt = np.zeros((T_BINS, CAP_BINS, CAP_BINS))
    for _ in range(n):
        l1, l2 = _draw_levels(scenario, rng)
        ep = generate_episode(l1, l2, rng)
        r = run_episode(ep, 'B', beta, delta_F, d_fixed=d, a0=a0, collect_V=True)
        for st, remaining in r['vtrace']:
            tot[st] += remaining
            cnt[st] += 1
    V = np.zeros_like(tot)
    seen = cnt > 0
    V[seen] = tot[seen] / cnt[seen]
    if seen.any():                       # impute unseen states from neighbours
        idx = np.argwhere(seen)
        for st in np.argwhere(~seen):
            V[tuple(st)] = V[tuple(idx[np.argmin(np.abs(idx - st).sum(1))])]
    return V


def train_C(scenario, beta, delta_F, n_train=50000, seed=0, log_every=200,
            a0=A0, Vbase=None, shape=True, d_shape=D_FIXED_B):
    if shape and Vbase is None:
        Vbase = estimate_baseline_V(scenario, beta, delta_F, d=d_shape,
                                    n=max(400, n_train // 20), seed=seed + 7,
                                    a0=a0)
    rng = np.random.default_rng(seed)
    Q = new_Q(seed)
    visits = np.zeros((T_BINS, CAP_BINS, CAP_BINS, N_ACTIONS), dtype=np.int64)
    e_decay = (EPS_END / EPS_START) ** (1.0 / max(1, n_train * 0.7))
    a_decay = (ALPHA_END / ALPHA_START) ** (1.0 / max(1, n_train))
    eps, alpha = EPS_START, ALPHA_START
    curve, buf = [], []
    for k in range(n_train):
        l1, l2 = _draw_levels(scenario, rng)
        ep = generate_episode(l1, l2, rng)
        r = run_episode(ep, 'C', beta, delta_F, Q=Q, eps_greedy=eps, learn=True,
                        rng=rng, alpha=alpha, visits=visits, a0=a0,
                        Vbase=Vbase)
        eps = max(EPS_END, eps * e_decay)
        alpha = max(ALPHA_END, alpha * a_decay)
        buf.append(r['rev'])
        if (k + 1) % log_every == 0:
            curve.append(float(np.mean(buf)))
            buf = []
    return Q, visits, np.array(curve)


B_MIN_D = 0.05      # a 0 % "discount" is not a commercial flexible product;
                    # policy B searches strictly positive discounts only.


def tune_B(scenario, beta, delta_F, n_tune=400, seed=1, a0=A0, grid=None):
    """Pre-search the discount grid and keep the best constant d (5.3)."""
    grid = D_GRID[D_GRID >= B_MIN_D] if grid is None else np.asarray(grid)
    best_d, best_r, prof = float(grid[0]), -np.inf, []
    for d in grid:
        rng = np.random.default_rng(seed)              # CRN across d
        tot = 0.0
        for _ in range(n_tune):
            l1, l2 = _draw_levels(scenario, rng)
            ep = generate_episode(l1, l2, rng)
            tot += run_episode(ep, 'B', beta, delta_F, d_fixed=float(d),
                               a0=a0)['rev']
        m = tot / n_tune
        prof.append(m)
        if m > best_r:
            best_r, best_d = m, float(d)
    return best_d, np.array(prof)


EVAL_KEYS = ('rev', 'lf', 'imbalance', 'ep_blocked_rate', 'lost_rate',
             'flex_share_n', 'flex_share_cbm', 'avg_unit_price', 'mean_d',
             'no_offer_rate')


def evaluate(scenario, beta, delta_F, Q, d_B, n_eval=1000, seed=777, a0=A0,
             d_paper=D_FIXED_B):
    """Paired evaluation on identical episodes (common random numbers).
       A  no flexible product
       B  best constant discount found by tune_B
       P  the paper's fixed d = 0.15 benchmark (5.3)
       C  learned state-dependent policy"""
    rng = np.random.default_rng(seed)
    acc = {p: {k: [] for k in EVAL_KEYS} for p in 'ABPC'}
    canni_num = canni_den = 0
    for _ in range(n_eval):
        l1, l2 = _draw_levels(scenario, rng)
        ep = generate_episode(l1, l2, rng)
        rA = run_episode(ep, 'A', beta, delta_F, a0=a0)
        rB = run_episode(ep, 'B', beta, delta_F, d_fixed=d_B, a0=a0)
        rP = run_episode(ep, 'B', beta, delta_F, d_fixed=d_paper, a0=a0)
        rC = run_episode(ep, 'C', beta, delta_F, Q=Q, eps_greedy=0.0, a0=a0)
        for p, r in (('A', rA), ('B', rB), ('P', rP), ('C', rC)):
            for k in EVAL_KEYS:
                acc[p][k].append(r[k])
        fmask = (rC['booked_prod'] == 'F')
        canni_den += int(fmask.sum())
        canni_num += int((fmask & rA['booked_flag']).sum())

    res = {}
    with np.errstate(invalid='ignore'):
        import warnings
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        for p in 'ABPC':
            for k, v in acc[p].items():
                arr = np.asarray(v, dtype=float)
                res[f'{p}_{k}'] = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')
    for p in 'ABPC':
        res[f'{p}_rev_sd'] = float(np.std(acc[p]['rev']))
    dif = np.array(acc['C']['rev']) - np.array(acc['A']['rev'])
    res['cannibalisation'] = canni_num / max(1, canni_den)
    A, B, P, C = res['A_rev'], res['B_rev'], res['P_rev'], res['C_rev']
    res['gap_BA'] = (B - A) / A * 100
    res['gap_CB'] = (C - B) / B * 100
    res['gap_CA'] = (C - A) / A * 100
    res['gap_PA'] = (P - A) / A * 100        # paper's B - A  (d = 0.15)
    res['gap_CP'] = (C - P) / P * 100        # paper's C - B
    res['d_paper'] = float(d_paper)
    res['CA_ci95'] = float(1.96 * np.std(dif) / np.sqrt(len(dif)) / A * 100)
    res['d_B'] = float(d_B)
    return res


# =============================================================================
# 8. POLICY EXTRACTION + EXPORT  (this is what the pricing tool consumes)
# =============================================================================
def greedy_policy(Q, visits, d_static_fallback=D_FIXED_B):
    """Dense d*(t_bin, cap_own_bin, cap_other_bin) as ACTION INDICES.

    States never visited during training are imputed from the nearest visited
    state in (t, cap_own, cap_other) index space; `trained` flags which is which
    so the tool can show a confidence badge.
    """
    q = Q[0] + Q[1] if DOUBLE_Q else Q[0]
    vis = visits.sum(axis=-1)
    act = np.argmax(q, axis=-1).astype(np.int16)
    trained = vis > 0

    fallback = 1 + int(np.argmin(np.abs(D_GRID - d_static_fallback)))
    idx = np.argwhere(trained)
    for s in np.argwhere(~trained):
        if len(idx) == 0:
            act[tuple(s)] = fallback
            continue
        dist = np.abs(idx - s).sum(axis=1)
        act[tuple(s)] = act[tuple(idx[np.argmin(dist)])]
    return act, trained, vis


def policy_to_json(act, trained, vis, q=None):
    flat = lambda a: a.reshape(-1).tolist()
    out = {'d_star_action': flat(act),
           'trained': [bool(b) for b in flat(trained)],
           'visits': [int(v) for v in flat(vis)]}
    if q is not None:
        out['q'] = q.reshape(N_STATES, N_ACTIONS).round(3).tolist()
    return out


def env_manifest():
    return {
        'schema': 'lcl-flexible-pricing-policy/1.0',
        'container': {'L': CONT_L, 'W': CONT_W, 'H': CONT_H, 'unit': 'dm',
                      'cbm': round(CONT_CBM, 4), 'n': N_CONT},
        'ep': {'n_orientations': N_ORIENT, 'support_frac': SUPPORT_FRAC},
        'horizon': T_HORIZON,
        'r_S': R_S,
        'A0': A0,
        'actions': ([{'id': 0, 'type': 'no_offer'}] +
                    [{'id': i + 1, 'type': 'discount', 'd': float(d)}
                     for i, d in enumerate(D_GRID)]),
        'state': {'t_bins': T_BINS, 'cap_bins': CAP_BINS, 'n_states': N_STATES,
                  'order': ['t_bin', 'cap_own_bin', 'cap_other_bin'],
                  'index': '(t_bin*CAP_BINS + cap_own_bin)*CAP_BINS + cap_other_bin',
                  'cap_bin_edges_cbm': [round(CONT_CBM * i / CAP_BINS, 4)
                                        for i in range(CAP_BINS + 1)],
                  't_bin_edges': [round(T_HORIZON * i / T_BINS, 4)
                                  for i in range(T_BINS + 1)]},
        'axes': {'beta': list(BETA_LEVELS), 'delta_F': list(DELTA_F_LEVELS),
                 'levels': list(LEVEL_NAMES)},
        'notes': 'capacity bins use residual BILLED CBM; state is relative to '
                 'the arriving shipper preferred container',
    }


def policy_key(beta, delta_F, scenario):
    sc = 'ANY' if scenario == 'ANY' else f'{scenario[0]}-{scenario[1]}'
    return f'b{beta:.2f}|f{delta_F:+.3f}|{sc}'


# =============================================================================
# 9. CALIBRATION HELPER
# =============================================================================
def calibrate_a0(a0_grid=(1.0, 1.2, 1.5, 1.8), n=120, beta=BETA_BASE, seed=3):
    """Policy-A baseline load factor per demand cell -- pick A0 so that the
    range brackets the 0.63-0.73 band reported in 6.3."""
    rows = []
    for a in a0_grid:
        lfs = {}
        for l1 in LEVEL_NAMES:
            rng = np.random.default_rng(seed)
            v = [run_episode(generate_episode(l1, l1, rng), 'A', beta, 0.0,
                             a0=a)['lf'] for _ in range(n)]
            lfs[l1] = float(np.mean(v))
        rows.append({'A0': a, **lfs})
    return rows


if __name__ == '__main__':
    import time
    t0 = time.time()
    rng = np.random.default_rng(0)
    ep = generate_episode('Normal', 'OffPeak', rng)
    r = run_episode(ep, 'A', BETA_BASE, DELTA_F_BASE)
    print(f'arrivals={ep["n"]}  LF={r["lf"]:.3f}  rev={r["rev"]:.0f}  '
          f'lost={r["lost_rate"]:.3f}  epblocked={r["ep_blocked_rate"]:.3f}  '
          f'({time.time()-t0:.2f}s)')
    print(json.dumps(calibrate_a0(n=40), indent=1))
