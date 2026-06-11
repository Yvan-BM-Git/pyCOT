#!/usr/bin/env python3
"""
script_complex_COT_ESMO.py
==========================
Asymmetric ESMO transition analysis between social organisations, restricted to
the COVER EDGES of the containment (Hasse) diagram:

    Tribe ⊂ Chief              Chief ⊂ ChiefState
    Tribe ⊂ State              State ⊂ ChiefState
    (Chief ⊄ State and State ⊄ Chief — no containment between them)

For each directed cover edge (A→B or B→A), FOUR transition types are computed
by pairing a PEACE or CONFLICT characteristic state at the source with a PEACE
or CONFLICT characteristic state at the target.  Calling "down" the smaller
org and "up" the larger:

    peace-down  -> peace-up       (PP)  stable expansion / consolidation
    peace-down  -> conflict-up    (PC)  growth into instability
    conflict-down -> peace-up     (CP)  recovery through org-level change
    conflict-down -> conflict-up  (CC)  conflict escalation / diffusion

Going DOWN the lattice reverses the direction:
    peace-up    -> peace-down     (PP)  institutional retraction to stability
    peace-up    -> conflict-down  (PC)  peaceful dissolution into conflict
    conflict-up -> peace-down     (CP)  de-escalation through simplification
    conflict-up -> conflict-down  (CC)  collapse of complex org under conflict

Total: 4 cover edges × 2 directions × 4 state combos = 32 directed ESMOs
      + 4 organisations × 2 intra-org combos (PC, CP) = 8 intra-org ESMOs
      = 40 transitions total.

WHY THIS IS ASYMMETRIC:
  Each signature is derived by comparing CHAR_STATES[source] (INITIAL) to
  CHAR_STATES[target] (GOAL).  Reversing direction swaps INITIAL and GOAL,
  negating all signs -> different LP constraints -> different ESMO counts.
  Within an edge direction, the four combos also differ because peace and
  conflict states have different absolute species levels.

ALL USER SETTINGS ARE IN THE CONFIGURATIONS BLOCK BELOW.
"""

import os, sys, time, hashlib, json
from datetime import datetime, timezone
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.optimize import linprog

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'src'))

from pyCOT.io.functions import read_txt

# ===========================================================================
# BEGIN CONFIGURATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. FILES AND OUTPUT
# ---------------------------------------------------------------------------
RN_FILE    = os.path.join(_SCRIPT_DIR, 'data', 'Complex_example.txt')
OUT_DIR    = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'complex_COT_ESMO')
CACHE_FILE = os.path.join(OUT_DIR, 'esmo_cache_inter.csv')

# ---------------------------------------------------------------------------
# 2. HARD-CODED ORGANISATIONS  (species sets)
# ---------------------------------------------------------------------------
_TRIBE       = frozenset({'C', 'C_Res', 'G', 'Res', 'X'})
_CHIEF       = frozenset({'C', 'C_Res', 'G', 'H', 'H_Res', 'P_H', 'Res', 'X'})
_STATE       = frozenset({'C', 'C_Res', 'G', 'I', 'I_Res', 'L', 'P_I', 'Res', 'X'})
_CHIEF_STATE = frozenset({'C', 'C_Res', 'G', 'H', 'H_Res', 'I', 'I_Res', 'L',
                           'P_H', 'P_I', 'Res', 'X'})

ORG_SETS = {
    'Tribe':      _TRIBE,
    'Chief':      _CHIEF,
    'State':      _STATE,
    'ChiefState': _CHIEF_STATE,
}

# ---------------------------------------------------------------------------
# 3. COVER EDGES  (direct containment in the Hasse diagram)
#    Each (down, up) pair: down ⊂ up, and no intermediate organisation.
#    Containment facts:
#      Chief     = Tribe + {H, H_Res, P_H}
#      State     = Tribe + {I, I_Res, L, P_I}
#      ChiefState = Chief + {I, I_Res, L, P_I}  =  State + {H, H_Res, P_H}
#    Note: Chief ⊄ State and State ⊄ Chief  →  no edge between them.
# ---------------------------------------------------------------------------
COVER_EDGES = [
    ('Tribe', 'Chief'),
    ('Tribe', 'State'),
    ('Chief', 'ChiefState'),
    ('State', 'ChiefState'),
]

ORG_ABBR = {
    'Tribe':      'Tri',
    'Chief':      'Chf',
    'State':      'Sta',
    'ChiefState': 'CfS',
}

ORG_NAMES = ['Tribe', 'Chief', 'State', 'ChiefState']

# ---------------------------------------------------------------------------
# 4. CHARACTERISTIC STATES  (peace and conflict for each organisation)
#    These define the equilibrium population/resource levels that typify each
#    organisational type under peaceful vs conflict conditions.
#    All ESMO signatures are derived by comparing two of these states:
#       delta[s] = GOAL[s] - INITIAL[s]
#       delta >  NEUTRAL_TOL -> sig[s] = +1  (species must GROW)
#       delta < -NEUTRAL_TOL -> sig[s] = -1  (species must SHRINK)
# ---------------------------------------------------------------------------
CHAR_STATES = {

    # ── TRIBE (C+X = 500) ─────────────────────────────────────────────────────
    ('Tribe', 'peace'): {
        'C':      450,   'X':   50,   # C dominant, low displacement
        'H':        0,   'I':    0,   'P_H':   0,   'P_I':    0,
        'G':        5,   'Res': 150,  'C_Res': 100,
        'H_Res':    0,   'I_Res':  0, 'L':     0,
    },
    ('Tribe', 'conflict'): {
        'C':      150,   'X':  350,   # fragmented, high displacement
        'H':        0,   'I':    0,   'P_H':   0,   'P_I':    0,
        'G':       80,   'Res':  15,  'C_Res':  15,
        'H_Res':    0,   'I_Res':  0, 'L':     0,
    },

    # ── CHIEF (C+X+H+P_H = 1000) ──────────────────────────────────────────────
    ('Chief', 'peace'): {
        'C':      200,   'X':   80,
        'H':      550,   'P_H': 170,  # H dominant; peace ≠ C dominant
        'I':        0,   'P_I':   0,
        'G':        5,   'Res':  50,  'C_Res': 100,
        'H_Res':  300,   'I_Res':  0, 'L':     0,
    },
    ('Chief', 'conflict'): {
        'C':      100,   'X':  600,   # social collapse, high displacement
        'H':      150,   'P_H': 150,
        'I':        0,   'P_I':   0,
        'G':       75,   'Res':  20,  'C_Res':  30,
        'H_Res':   60,   'I_Res':  0, 'L':     0,
    },

    # ── STATE (C+X+I+P_I = 10 000) ────────────────────────────────────────────
    ('State', 'peace'): {
        'C':     1500,   'X':  500,
        'H':        0,   'P_H':   0,
        'I':     7000,   'P_I': 1000, # I dominant; rule of law
        'G':       10,   'Res': 500,  'C_Res': 1000,
        'H_Res':    0,   'I_Res': 2000, 'L':  800,
    },
    ('State', 'conflict'): {
        'C':     2000,   'X': 5500,   # institutional collapse
        'H':        0,   'P_H':   0,
        'I':     1500,   'P_I': 1000,
        'G':       80,   'Res': 200,  'C_Res':  300,
        'H_Res':    0,   'I_Res':  400, 'L':   100,
    },

    # ── CHIEFSTATE (C+X+H+I+P_H+P_I = 10 000) ────────────────────────────────
    ('ChiefState', 'peace'): {
        'C':     1000,   'X':  500,
        'H':     2500,   'P_H': 500,
        'I':     4000,   'P_I': 1500, # dual hierarchy, I dominant
        'G':       10,   'Res': 500,  'C_Res':  800,
        'H_Res':  1200,  'I_Res': 1500, 'L':   600,
    },
    ('ChiefState', 'conflict'): {
        'C':      500,   'X': 7000,   # near-total social collapse
        'H':      500,   'P_H': 500,
        'I':      500,   'P_I': 1000,
        'G':       85,   'Res': 100,  'C_Res':  150,
        'H_Res':  100,   'I_Res':  100, 'L':    50,
    },
}

# ---------------------------------------------------------------------------
# 5. TRANSITION TAXONOMY LABELS AND COLORS
# ---------------------------------------------------------------------------
STATE_LABELS  = ['peace', 'conflict']
COMBO_LABELS  = {'pp': 'P->P', 'pc': 'P->C', 'cp': 'C->P', 'cc': 'C->C'}
COMBO_COLORS  = {
    'pp': '#27AE60',   # green:  stable expansion
    'pc': '#E67E22',   # orange: growth into instability
    'cp': '#2980B9',   # blue:   recovery
    'cc': '#E74C3C',   # red:    conflict escalation
}
COMBO_DESC = {
    'pp': 'peace -> peace   (stable expansion)',
    'pc': 'peace -> conflict (growth into instability)',
    'cp': 'conflict -> peace (recovery / de-escalation)',
    'cc': 'conflict -> conflict (conflict escalation)',
}

# ---------------------------------------------------------------------------
# 6. LP PERFORMANCE SETTINGS
# ---------------------------------------------------------------------------
TOL_STRICT  = 1e-9
TOL_VERTEX  = 1e-7
N_RAND_1    = 2000
N_EXPLORE   = 40
N_RAND_3    = 600
PATIENCE    = 200
SEED        = 42
EPS_SIGN    = 1e-3
NEUTRAL_TOL = 5.0    # |delta| below this threshold -> neutral species
                     # (avoids spurious constraints from small numerical differences)

# People species: total must be conserved by every reaction in the network.
# To compare states across organisations with different population sizes, we
# normalise every state to a common reference population before computing the
# signature.  This prevents inter-org LPs from requiring net people growth,
# which is stoichiometrically impossible under conservation.
PEOPLE_SP = {'C', 'X', 'H', 'I', 'P_H', 'P_I'}
P_REF     = 1000.0   # reference population for normalisation

# ---------------------------------------------------------------------------
# 7. FIGURE SETTINGS
# ---------------------------------------------------------------------------
FONT_SIZE = 11

# ===========================================================================
# END CONFIGURATIONS
# ===========================================================================

_OUT_DIR = os.path.realpath(OUT_DIR)
os.makedirs(_OUT_DIR, exist_ok=True)

# ===========================================================================
# CACHE HELPERS  — persistent store for ESMO results
# ===========================================================================
_CACHE_COLS = [
    'run_id', 'state_from_id', 'state_to_id',
    'org_from', 'org_to', 'type_from', 'type_to',
    'combo', 'edge', 'direction',
    'sig_hash', 'sig_json',
    'n_esmos', 'mean_support', 'min_support', 'max_support',
    'time_s', 'n_state_pairs', 'source', 'computed_at',
]

def _make_run_id(sfid, stid):
    return f"{sfid}::{stid}"

def _hash_sig(sig_dict):
    s = json.dumps({k: int(sig_dict[k]) for k in sorted(sig_dict)})
    return hashlib.md5(s.encode()).hexdigest()

def _load_cache():
    _cf = os.path.realpath(CACHE_FILE)
    if os.path.exists(_cf):
        df = pd.read_csv(_cf).set_index('run_id')
        print(f"  Cache: loaded {len(df)} existing results from {_cf}")
        return df
    print(f"  Cache: no existing cache found — will create {_cf}")
    return pd.DataFrame(columns=_CACHE_COLS).set_index('run_id')

def _append_cache(row_dict):
    _cf = os.path.realpath(CACHE_FILE)
    row_df = pd.DataFrame([row_dict])
    row_df.to_csv(_cf, mode='a', header=not os.path.exists(_cf), index=False)

_CACHE = _load_cache()

print(f"Output directory : {_OUT_DIR}")
print(f"LP settings      : N_RAND_1={N_RAND_1}  N_RAND_3={N_RAND_3}  PATIENCE={PATIENCE}")
print(f"NEUTRAL_TOL      : {NEUTRAL_TOL}  (|delta| < {NEUTRAL_TOL} -> neutral)")

# ===========================================================================
# 1. LOAD NETWORK AND BUILD O7 (ChiefState) SUB-NETWORK
# ===========================================================================
print(f"\nLoading: {RN_FILE}")
rn   = read_txt(RN_FILE)
_sm  = rn.stoichiometry_matrix()
_full_sp = [str(s) for s in _sm.species]
_full_rx = [str(r) for r in _sm.reactions]
print(f"  Full network: {len(_full_sp)} species, {len(_full_rx)} reactions")

sp_obj = [sp for sp in rn.species() if str(sp.name) in _CHIEF_STATE]
sub_rn = rn.sub_reaction_network(sp_obj)
sub_sm = sub_rn.stoichiometry_matrix()

_raw_sp = [str(s) for s in sub_sm.species]
_raw_rx = [str(r) for r in sub_sm.reactions]
_S_raw  = np.asarray(sub_sm, dtype=float)

_full_rx_idx = {r: i for i, r in enumerate(_full_rx)}
_order  = sorted(range(len(_raw_rx)), key=lambda k: _full_rx_idx.get(_raw_rx[k], 9999))
_raw_rx = [_raw_rx[k] for k in _order]
_S_raw  = _S_raw[:, _order]

O7_SP = _raw_sp
O7_RX = _raw_rx
S_O7  = _S_raw
N_SP  = len(O7_SP)
N_RX  = len(O7_RX)

print(f"  O7 sub-network: {N_SP} species, {N_RX} reactions")
print(f"  Reactions: {O7_RX}")

# Verify containment relationships
print("\nContainment verification:")
for down, up in COVER_EDGES:
    ok = ORG_SETS[down].issubset(ORG_SETS[up])
    excl = sorted(ORG_SETS[up] - ORG_SETS[down])
    print(f"  {down} ⊂ {up}: {'OK' if ok else 'FAIL'}  "
          f"(exclusive to {up}: {excl})")

# ===========================================================================
# 2. LP HELPERS
# ===========================================================================

def build_lp(sig_dict):
    target_map = {}
    for sp_name, sign in sig_dict.items():
        if sp_name in O7_SP:
            target_map[O7_SP.index(sp_name)] = sign
        else:
            print(f"  [WARN] '{sp_name}' not in O7 sub-network -- skipped")
    A_eq = np.ones((1, N_RX))
    b_eq = np.array([1.0])
    A_ub_rows, b_ub_vals = [], []
    for si, sign in target_map.items():
        A_ub_rows.append(-sign * S_O7[si, :].copy())
        b_ub_vals.append(-EPS_SIGN)
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if A_ub_rows else None
    return A_eq, b_eq, A_ub, b_ub


def is_feasible(A_eq, b_eq, A_ub, b_ub):
    try:
        res = linprog(np.zeros(N_RX), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0.0, None)] * N_RX, method='highs')
        return res.success and res.status == 0
    except Exception:
        return False


def enumerate_vertices(A_eq, b_eq, A_ub, b_ub, seed=SEED):
    rng    = np.random.default_rng(seed)
    bounds = [(0.0, None)] * N_RX
    opts   = {'dual_feasibility_tolerance': 1e-10,
              'primal_feasibility_tolerance': 1e-10, 'presolve': True}
    vertices = []; vertex_keys = {}

    def _key(v):
        return tuple(np.round(v, 7))

    def _try_add(v):
        if not np.all(np.isfinite(v)):
            return False
        k = _key(v)
        if k not in vertex_keys:
            vertex_keys[k] = len(vertices)
            vertices.append(v.copy())
            return True
        return False

    def _solve(c):
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs', options=opts)
            if res.success and res.status == 0:
                return res.x
        except Exception:
            pass
        return None

    if not is_feasible(A_eq, b_eq, A_ub, b_ub):
        return []

    no_new = 0
    for i in range(N_RAND_1):
        v = _solve(rng.standard_normal(N_RX))
        if v is not None:
            if _try_add(v): no_new = 0
            else:           no_new += 1
        if no_new >= PATIENCE and i >= 500:
            break

    n_after_p1 = len(vertices)
    for vi in range(n_after_p1):
        zero_idx = np.where(vertices[vi] < 1e-8)[0]
        explore  = (zero_idx if len(zero_idx) <= N_EXPLORE
                    else rng.choice(zero_idx, N_EXPLORE, replace=False))
        for j in explore:
            c = np.zeros(N_RX); c[j] = -1.0
            v = _solve(c)
            if v is not None and v[j] > 1e-8:
                _try_add(v)

    no_new = 0
    for i in range(N_RAND_3):
        v = _solve(rng.standard_normal(N_RX))
        if v is not None:
            if _try_add(v): no_new = 0
            else:           no_new += 1
        if no_new >= PATIENCE and i >= 200:
            break

    return vertices


def filter_strict(vertices, sig_dict, tol=TOL_STRICT):
    kept = []
    for v in vertices:
        net = S_O7 @ v
        ok  = True
        for sp_name, sign in sig_dict.items():
            if sp_name not in O7_SP:
                continue
            si = O7_SP.index(sp_name)
            if sign > 0 and net[si] <= tol:
                ok = False; break
            if sign < 0 and net[si] >= -tol:
                ok = False; break
        if ok:
            kept.append(v)
    return kept


def compute_esmos(sig_dict):
    A_eq, b_eq, A_ub, b_ub = build_lp(sig_dict)
    t0    = time.perf_counter()
    raw   = enumerate_vertices(A_eq, b_eq, A_ub, b_ub)
    esmos = filter_strict(raw, sig_dict)
    elapsed = time.perf_counter() - t0
    n_rej = len(raw) - len(esmos)
    print(f"    {len(raw):4d} raw  ->  {len(esmos):4d} ESMOs  "
          f"({n_rej} rejected)  [{elapsed:.1f}s]")
    return esmos, elapsed


def classify_nontarget(esmos, sig_dict, tol=TOL_STRICT):
    target_sp = set(sig_dict.keys())
    nontarget = [s for s in O7_SP if s not in target_sp]
    n         = len(esmos)
    counts    = {s: {'grow': 0, 'neutral': 0, 'decline': 0} for s in nontarget}
    for v in esmos:
        net = S_O7 @ v
        for s in nontarget:
            val = net[O7_SP.index(s)]
            if   val >  tol: counts[s]['grow']    += 1
            elif val < -tol: counts[s]['decline']  += 1
            else:            counts[s]['neutral']  += 1
    return nontarget, counts, n


# ===========================================================================
# 3. SIGNATURE DERIVATION
# ===========================================================================

def normalize_state(state_dict):
    """
    Rescale all species in state_dict so that people species sum to P_REF.

    The population conservation law (Σ people species = constant for any
    flux vector) makes inter-org LP infeasible whenever GOAL and INITIAL have
    different total populations.  Normalising to a common P_REF removes that
    constraint: both states then have the same people total, and all deltas
    are structurally achievable.  Resources are scaled by the same factor so
    that per-capita resource ratios are preserved across org scales.
    """
    total = sum(float(state_dict.get(s, 0.0)) for s in PEOPLE_SP)
    if total == 0.0:
        return dict(state_dict)
    scale = P_REF / total
    return {k: float(v) * scale for k, v in state_dict.items()}


def derive_signature(org_from, state_from, org_to, state_to):
    """
    Derive ESMO signature for (org_from, state_from) -> (org_to, state_to).
    Both states are normalised to P_REF people before computing deltas so
    that inter-org comparisons are stoichiometrically feasible.
    delta = GOAL_norm - INITIAL_norm:
      delta >  NEUTRAL_TOL -> +1 (must grow)
      delta < -NEUTRAL_TOL -> -1 (must shrink)
      else                 -> neutral
    """
    initial = normalize_state(CHAR_STATES[(org_from, state_from)])
    goal    = normalize_state(CHAR_STATES[(org_to,   state_to)])
    sig     = {}
    for s in O7_SP:
        delta = goal.get(s, 0.0) - initial.get(s, 0.0)
        if   delta >  NEUTRAL_TOL: sig[s] = +1
        elif delta < -NEUTRAL_TOL: sig[s] = -1
    return sig


# ===========================================================================
# 4. BUILD TRANSITION LIST (cover edges × 2 directions × 4 state combos)
# ===========================================================================

TRANSITIONS = []   # list of dicts; index matches RESULTS list below
for down, up in COVER_EDGES:
    for dir_label, (org_from, org_to) in [('up', (down, up)), ('down', (up, down))]:
        for sf in STATE_LABELS:   # source state
            for st in STATE_LABELS:   # target state
                combo = sf[0] + st[0]   # 'pp', 'pc', 'cp', 'cc'
                TRANSITIONS.append({
                    'edge':          f'{down}-{up}',
                    'down':          down,
                    'up':            up,
                    'direction':     dir_label,
                    'org_from':      org_from,
                    'state_from':    sf,
                    'org_to':        org_to,
                    'state_to':      st,
                    'combo':         combo,
                    'label':         f'{ORG_ABBR[org_from]}_{sf[:1]}  ->  {ORG_ABBR[org_to]}_{st[:1]}',
                    'short':         f'{ORG_ABBR[org_from]}{sf[0].upper()}->{ORG_ABBR[org_to]}{st[0].upper()}',
                    'state_from_id': f'{org_from}::{sf}::norm{int(P_REF)}',
                    'state_to_id':   f'{org_to}::{st}::norm{int(P_REF)}',
                })

N_TRANS = len(TRANSITIONS)
print(f"\n{'='*72}")
print(f"  TRANSITION PLAN: {N_TRANS} directed transitions")
print(f"  (4 cover edges x 2 directions x 4 state combos = 32)")
print(f"{'='*72}")
print(f"\n  {'#':>3}  {'Edge':>18}  dir   combo  label")
print(f"  {'-'*3}  {'-'*18}  ----  -----  -----")
for ti, t in enumerate(TRANSITIONS):
    print(f"  {ti+1:>3}  {t['edge']:>18}  {t['direction']:>4}  "
          f"{t['combo']:>5}  {t['short']}")

# ===========================================================================
# 5. MAIN ANALYSIS LOOP
# ===========================================================================

print(f"\n{'='*72}")
print("  ESMO ENUMERATION")
print(f"{'='*72}")

RESULTS = []   # parallel to TRANSITIONS — each entry has 'n_esmos' always;
               # 'esmos' list is None when loaded from cache (not recomputed)
records = []

for ti, t in enumerate(TRANSITIONS):
    run_id        = _make_run_id(t['state_from_id'], t['state_to_id'])
    sig           = derive_signature(t['org_from'], t['state_from'],
                                     t['org_to'],   t['state_to'])
    grow_sp       = [s for s, v in sig.items() if v == +1]
    shrink_sp     = [s for s, v in sig.items() if v == -1]
    sig_h         = _hash_sig(sig) if sig else ''
    sig_j         = json.dumps({k: int(v) for k, v in sig.items()})

    print(f"\n[{ti+1}/{N_TRANS}] {t['short']}  (run_id={run_id})")

    # ── 1. Exact cache hit ─────────────────────────────────────────────────
    if run_id in _CACHE.index:
        row = _CACHE.loc[run_id]
        n   = int(row['n_esmos'])
        ms  = float(row['mean_support'])
        print(f"  [CACHED]  n_esmos={n}  mean_support={ms:.2f}")
        RESULTS.append({'esmos': None, 'sig': sig, 'n_esmos': n,
                        'mean_support': ms,
                        'min_support': int(row['min_support']),
                        'max_support': int(row['max_support']),
                        'time': float(row['time_s']), 'from_cache': True})
        records.append({
            'edge': t['edge'], 'direction': t['direction'],
            'combo': t['combo'], 'short': t['short'],
            'org_from': t['org_from'], 'state_from': t['state_from'],
            'org_to': t['org_to'], 'state_to': t['state_to'],
            'n_targets': len(sig), 'n_grow': len(grow_sp), 'n_shrink': len(shrink_sp),
            'n_esmos': n, 'Pr_feasible': 1.0 if n > 0 else 0.0, 'n_state_pairs': 1,
            'mean_support': ms, 'min_support': int(row['min_support']),
            'max_support': int(row['max_support']), 'time_s': float(row['time_s']),
        })
        continue

    # ── 2. Empty signature  ────────────────────────────────────────────────
    if not sig:
        print("  WARNING: empty signature -- skipping")
        RESULTS.append({'esmos': [], 'sig': sig, 'n_esmos': 0,
                        'mean_support': 0.0, 'min_support': 0, 'max_support': 0,
                        'time': 0.0, 'from_cache': False})
        records.append({
            'edge': t['edge'], 'direction': t['direction'],
            'combo': t['combo'], 'short': t['short'],
            'org_from': t['org_from'], 'state_from': t['state_from'],
            'org_to': t['org_to'], 'state_to': t['state_to'],
            'n_targets': 0, 'n_grow': 0, 'n_shrink': 0,
            'n_esmos': 0, 'Pr_feasible': 0.0, 'n_state_pairs': 1,
            'mean_support': 0.0, 'min_support': 0, 'max_support': 0, 'time_s': 0.0,
        })
        continue

    # ── 3. Signature-match in cache (same LP, different state labels) ───────
    sig_matches = _CACHE[_CACHE['sig_hash'] == sig_h] if sig_h and len(_CACHE) else pd.DataFrame()
    if len(sig_matches) > 0:
        ref = sig_matches.iloc[0]
        n   = int(ref['n_esmos'])
        ms  = float(ref['mean_support'])
        print(f"  [SIG_MATCH -> {ref.name}]  n_esmos={n}  mean_support={ms:.2f}")
        cache_row = {
            'run_id': run_id, 'state_from_id': t['state_from_id'],
            'state_to_id': t['state_to_id'],
            'org_from': t['org_from'], 'org_to': t['org_to'],
            'type_from': t['state_from'], 'type_to': t['state_to'],
            'combo': t['combo'], 'edge': t['edge'], 'direction': t['direction'],
            'sig_hash': sig_h, 'sig_json': sig_j,
            'n_esmos': n, 'mean_support': ms,
            'min_support': int(ref['min_support']), 'max_support': int(ref['max_support']),
            'time_s': 0.0, 'n_state_pairs': 1,
            'source': f'sig_match::{ref.name}',
            'computed_at': datetime.now(timezone.utc).isoformat(),
        }
        _append_cache(cache_row)
        _CACHE.loc[run_id] = {k: v for k, v in cache_row.items() if k != 'run_id'}
        RESULTS.append({'esmos': None, 'sig': sig, 'n_esmos': n,
                        'mean_support': ms,
                        'min_support': int(ref['min_support']),
                        'max_support': int(ref['max_support']),
                        'time': 0.0, 'from_cache': True})
        records.append({
            'edge': t['edge'], 'direction': t['direction'],
            'combo': t['combo'], 'short': t['short'],
            'org_from': t['org_from'], 'state_from': t['state_from'],
            'org_to': t['org_to'], 'state_to': t['state_to'],
            'n_targets': len(sig), 'n_grow': len(grow_sp), 'n_shrink': len(shrink_sp),
            'n_esmos': n, 'Pr_feasible': 1.0 if n > 0 else 0.0, 'n_state_pairs': 1,
            'mean_support': ms, 'min_support': int(ref['min_support']),
            'max_support': int(ref['max_support']), 'time_s': 0.0,
        })
        continue

    # ── 4. Full computation ────────────────────────────────────────────────
    print(f"  grow: {grow_sp}   shrink: {shrink_sp}")
    esmos, elapsed = compute_esmos(sig)
    supp = [int(np.sum(v > TOL_STRICT)) for v in esmos] if esmos else []
    n    = len(esmos)
    ms   = round(float(np.mean(supp)), 2) if supp else 0.0
    mins = min(supp) if supp else 0
    maxs = max(supp) if supp else 0

    RESULTS.append({'esmos': esmos, 'sig': sig, 'n_esmos': n,
                    'mean_support': ms, 'min_support': mins, 'max_support': maxs,
                    'time': elapsed, 'from_cache': False})
    records.append({
        'edge': t['edge'], 'direction': t['direction'],
        'combo': t['combo'], 'short': t['short'],
        'org_from': t['org_from'], 'state_from': t['state_from'],
        'org_to': t['org_to'], 'state_to': t['state_to'],
        'n_targets': len(sig), 'n_grow': len(grow_sp), 'n_shrink': len(shrink_sp),
        'n_esmos': n, 'Pr_feasible': 1.0 if n > 0 else 0.0, 'n_state_pairs': 1,
        'mean_support': ms, 'min_support': mins, 'max_support': maxs,
        'time_s': round(elapsed, 1),
    })
    cache_row = {
        'run_id': run_id, 'state_from_id': t['state_from_id'],
        'state_to_id': t['state_to_id'],
        'org_from': t['org_from'], 'org_to': t['org_to'],
        'type_from': t['state_from'], 'type_to': t['state_to'],
        'combo': t['combo'], 'edge': t['edge'], 'direction': t['direction'],
        'sig_hash': sig_h, 'sig_json': sig_j,
        'n_esmos': n, 'mean_support': ms, 'min_support': mins, 'max_support': maxs,
        'time_s': round(elapsed, 1), 'n_state_pairs': 1,
        'source': 'computed',
        'computed_at': datetime.now(timezone.utc).isoformat(),
    }
    _append_cache(cache_row)
    _CACHE.loc[run_id] = {k: v for k, v in cache_row.items() if k != 'run_id'}

df = pd.DataFrame(records)
csv_path = os.path.join(_OUT_DIR, 'transition_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nSummary CSV -> {csv_path}")
print(f"\n{'='*72}")
print("  RESULTS TABLE")
print(f"{'='*72}")
print(df[['short','combo','n_targets','n_grow','n_shrink',
          'n_esmos','mean_support','time_s']].to_string(index=False))

# ===========================================================================
# 6. FIGURES
# ===========================================================================

plt.rcParams.update({
    'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2, 'legend.fontsize': FONT_SIZE - 2,
})

_COMBO_COLORS = COMBO_COLORS
_COL_GROW    = '#27AE60'
_COL_NEUTRAL = '#BDC3C7'
_COL_DECLINE = '#E74C3C'
_COL_BAR     = '#2980B9'


def _save(fig, stem):
    p = os.path.join(_OUT_DIR, f'{stem}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close(fig)


# Helper: get result dict for a specific transition.
# Always provides 'n_esmos'; 'esmos' may be None if loaded from cache.
def _get_res(edge, direction, combo):
    for ti, t in enumerate(TRANSITIONS):
        if t['edge'] == edge and t['direction'] == direction and t['combo'] == combo:
            return RESULTS[ti]
    return None

def _n(r):
    """Return ESMO count from a result dict (safe whether computed or cached)."""
    if r is None:
        return 0
    return r.get('n_esmos', len(r.get('esmos') or []))

def _esmos(r):
    """Return actual ESMO vectors; empty list if loaded from cache."""
    if r is None:
        return []
    return r.get('esmos') or []


# ---------------------------------------------------------------------------
# Fig 1: Per-cover-edge 2×2 ESMO count matrix (UP and DOWN side by side)
# ---------------------------------------------------------------------------
# Layout: 4 rows (one per cover edge) × 2 cols (up / down direction)
# Each cell is a 2×2 heatmap: rows=source state, cols=target state

fig1, axes1 = plt.subplots(
    len(COVER_EDGES), 2,
    figsize=(10, 4.5 * len(COVER_EDGES)),
    squeeze=False)

STATE_ABBR = {'peace': 'P', 'conflict': 'C'}
COMBOS_2x2 = [('pp', 'pc'), ('cp', 'cc')]   # [row_peace, row_conflict] x [col_peace, col_conflict]

for ei, (down, up) in enumerate(COVER_EDGES):
    for di, (direction, (org_from, org_to)) in enumerate(
            [('up', (down, up)), ('down', (up, down))]):

        ax = axes1[ei][di]
        mat = np.zeros((2, 2))
        for ri, sf in enumerate(STATE_LABELS):
            for ci, st in enumerate(STATE_LABELS):
                combo = sf[0] + st[0]
                r = _get_res(f'{down}-{up}', direction, combo)
                mat[ri, ci] = _n(r)

        vmax = max(float(mat.max()), 1.0)
        im = ax.imshow(mat, aspect='equal', cmap='YlOrRd',
                       vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['→ peace', '→ conflict'], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['peace ↓', 'conflict ↓'], fontsize=9)
        ax.set_xlabel('target state')
        ax.set_ylabel('source state')
        dir_arrow = ('↑' if direction == 'up' else '↓')
        ax.set_title(
            f'{org_from} {dir_arrow} {org_to}\n(n_ESMOs)',
            fontsize=10, fontweight='bold')

        for ri in range(2):
            for ci in range(2):
                val = int(mat[ri, ci])
                col = 'white' if mat[ri, ci] > vmax * 0.60 else 'black'
                ax.text(ci, ri, str(val), ha='center', va='center',
                        fontsize=13, fontweight='bold', color=col)

        plt.colorbar(im, ax=ax, shrink=0.85)

fig1.suptitle(
    'ESMO counts per directed cover-edge transition\n'
    'Rows = source state (peace / conflict),  Cols = target state\n'
    'Left column = going UP the lattice,  Right column = going DOWN',
    fontsize=FONT_SIZE + 1, y=1.01)
fig1.tight_layout()
_save(fig1, 'fig1_cover_edge_matrices')

# ---------------------------------------------------------------------------
# Fig 2: Aggregated ESMO counts by combo type (PP, PC, CP, CC)
# One bar group per cover edge, coloured by combo
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

for di, (direction, dir_label) in enumerate([('up', 'Going UP ↑'), ('down', 'Going DOWN ↓')]):
    ax    = axes2[di]
    n_ce  = len(COVER_EDGES)
    x     = np.arange(n_ce)
    W     = 0.2
    offsets = [-1.5*W, -0.5*W, 0.5*W, 1.5*W]
    combo_list = ['pp', 'pc', 'cp', 'cc']

    for ci, (combo, offset) in enumerate(zip(combo_list, offsets)):
        vals = []
        for down, up in COVER_EDGES:
            r = _get_res(f'{down}-{up}', direction, combo)
            vals.append(_n(r))
        bars = ax.bar(x + offset, vals, W, color=_COMBO_COLORS[combo],
                      alpha=0.85, label=COMBO_DESC[combo], zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 2, str(v),
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}-{u}' for d, u in COVER_EDGES], rotation=15, ha='right')
    ax.set_ylabel('Number of ESMOs')
    ax.set_title(f'{dir_label}\n(more ESMOs = more reaction strategies available)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

fig2.suptitle(
    'ESMO counts by transition type (PP=peace->peace, PC=peace->conflict,\n'
    'CP=conflict->peace, CC=conflict->conflict)',
    fontsize=FONT_SIZE, y=1.01)
fig2.tight_layout()
_save(fig2, 'fig2_combo_counts')

# ---------------------------------------------------------------------------
# Fig 3: Signature heatmap — constrained species per transition
# ---------------------------------------------------------------------------
n_trans = len(TRANSITIONS)
sig_mat = np.zeros((n_trans, N_SP))
for ti, t in enumerate(TRANSITIONS):
    sig = RESULTS[ti]['sig']
    for si, sp in enumerate(O7_SP):
        sig_mat[ti, si] = sig.get(sp, 0)

cmap_sig = mcolors.ListedColormap([_COL_DECLINE, '#F5F5F5', _COL_GROW])
norm_sig  = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_sig.N)

ytick_lbls = [t['short'] for t in TRANSITIONS]
ytick_cols = [_COMBO_COLORS[t['combo']] for t in TRANSITIONS]

fig3, ax3 = plt.subplots(figsize=(max(10, N_SP * 0.9), max(8, n_trans * 0.35)))
im3 = ax3.imshow(sig_mat, aspect='auto', cmap=cmap_sig, norm=norm_sig,
                 interpolation='nearest')
ax3.set_xticks(range(N_SP))
ax3.set_xticklabels(O7_SP, rotation=45, ha='right')
ax3.set_yticks(range(n_trans))
ax3.set_yticklabels(ytick_lbls, fontsize=8)
for ti, lbl in enumerate(ax3.get_yticklabels()):
    lbl.set_color(ytick_cols[ti])
ax3.set_xlabel('Species (O7 sub-network)')
ax3.set_title(
    'LP signature matrix: green=must GROW (+1), red=must SHRINK (-1), white=neutral\n'
    'Row colors: green=PP, orange=PC, blue=CP, red=CC',
    fontsize=FONT_SIZE)
for ti in range(n_trans):
    for si in range(N_SP):
        v = int(sig_mat[ti, si])
        if v != 0:
            ax3.text(si, ti, '+' if v > 0 else '-',
                     ha='center', va='center', fontsize=8,
                     color='black', fontweight='bold')
plt.colorbar(im3, ax=ax3, ticks=[-1, 0, 1], label='Sign', shrink=0.5)
fig3.tight_layout()
_save(fig3, 'fig3_signature_matrix')

# ---------------------------------------------------------------------------
# Fig 4: Reaction usage grid — one panel per transition
# ---------------------------------------------------------------------------
def _rx_usage_ax(ax, esmos, title, color=_COL_BAR, tol=TOL_STRICT):
    if not esmos:
        ax.text(0.5, 0.5, 'no ESMOs', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title(title, fontsize=7)
        return
    freq  = (np.vstack(esmos) > tol).astype(float).mean(axis=0)
    order = np.argsort(freq)[::-1]
    nz    = freq[order] > 0
    sfreq = freq[order][nz]
    slabs = [O7_RX[i] for i in order[nz]]
    xpos  = np.arange(len(sfreq))
    ax.bar(xpos, sfreq, color=color, alpha=0.75, zorder=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(slabs, rotation=90, fontsize=max(4, FONT_SIZE - 7))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Fraction', fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

# Arrange as: 4 cover edges (row groups of 8 = 2 dir x 4 combos)
# Use 8 columns (up_PP, up_PC, up_CP, up_CC, dn_PP, dn_PC, dn_CP, dn_CC)
n_cols4 = 8
n_rows4 = len(COVER_EDGES)
fig4, axes4 = plt.subplots(n_rows4, n_cols4,
                            figsize=(3.0 * n_cols4, 3.5 * n_rows4),
                            squeeze=False)

col_titles = (['↑PP','↑PC','↑CP','↑CC'] + ['↓PP','↓PC','↓CP','↓CC'])
for ci, ct in enumerate(col_titles):
    axes4[0][ci].set_title(ct, fontsize=9, fontweight='bold')

for ei, (down, up) in enumerate(COVER_EDGES):
    for di, direction in enumerate(['up', 'down']):
        for ci, combo in enumerate(['pp', 'pc', 'cp', 'cc']):
            col = di * 4 + ci
            ax  = axes4[ei][col]
            r   = _get_res(f'{down}-{up}', direction, combo)
            esmos = _esmos(r)
            _rx_usage_ax(ax, esmos,
                         title=f'n={_n(r)}',
                         color=_COMBO_COLORS[combo])
    axes4[ei][0].set_ylabel(f'{down}-{up}', fontsize=9, fontweight='bold')

fig4.suptitle(
    'Reaction usage frequency per transition\n'
    'Columns: ↑=going UP, ↓=going DOWN; PP/PC/CP/CC = state-combo type',
    fontsize=FONT_SIZE, y=1.01)
fig4.tight_layout()
_save(fig4, 'fig4_reaction_usage_grid')

# ---------------------------------------------------------------------------
# Fig 5: Aggregate reaction usage by combo type (across all edges)
# ---------------------------------------------------------------------------
fig5, axes5 = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
for di, direction in enumerate(['up', 'down']):
    for ci, combo in enumerate(['pp', 'pc', 'cp', 'cc']):
        ax = axes5[di][ci]
        all_esmos = []
        for down, up in COVER_EDGES:
            r = _get_res(f'{down}-{up}', direction, combo)
            if r is not None:
                all_esmos.extend(_esmos(r))
        dir_sym = '↑' if direction == 'up' else '↓'
        _rx_usage_ax(ax, all_esmos,
                     title=f'{dir_sym} {COMBO_DESC[combo]}\n(all edges combined, n={len(all_esmos)})',
                     color=_COMBO_COLORS[combo])

fig5.suptitle(
    'Reaction usage aggregated across all cover edges by direction and state-combo\n'
    'High bar = reaction is structurally important for that transition type',
    fontsize=FONT_SIZE, y=1.01)
fig5.tight_layout()
_save(fig5, 'fig5_aggregate_reaction_usage')

# ---------------------------------------------------------------------------
# Fig 6: Non-target species behavior — aggregate by combo type (going UP)
# ---------------------------------------------------------------------------
fig6, axes6 = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
for di, direction in enumerate(['up', 'down']):
    for ci, combo in enumerate(['pp', 'pc', 'cp', 'cc']):
        ax = axes6[di][ci]
        all_esmos = []; sig_agg = {}
        for down, up in COVER_EDGES:
            r = _get_res(f'{down}-{up}', direction, combo)
            ev = _esmos(r)
            if r is not None and ev:
                all_esmos.extend(ev)
                for s, v in r['sig'].items():
                    if s not in sig_agg:
                        sig_agg[s] = v

        if not all_esmos:
            ax.set_visible(False)
            continue

        nt_sp, cnt, n_tot = classify_nontarget(all_esmos, sig_agg)
        if not nt_sp:
            ax.set_title(f'all constrained', fontsize=8)
            ax.set_visible(False)
            continue

        x      = np.arange(len(nt_sp))
        f_grow = np.array([cnt[s]['grow']    / n_tot for s in nt_sp])
        f_neut = np.array([cnt[s]['neutral'] / n_tot for s in nt_sp])
        f_decl = np.array([cnt[s]['decline'] / n_tot for s in nt_sp])
        ax.bar(x, f_grow,                         color=_COL_GROW,    alpha=0.85, label='grows')
        ax.bar(x, f_neut, bottom=f_grow,           color=_COL_NEUTRAL, alpha=0.75, label='neutral')
        ax.bar(x, f_decl, bottom=f_grow + f_neut,  color=_COL_DECLINE, alpha=0.85, label='declines')
        ax.set_xticks(x)
        ax.set_xticklabels(nt_sp, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Fraction', fontsize=7)
        dir_sym = '↑' if direction == 'up' else '↓'
        ax.set_title(f'{dir_sym} {COMBO_DESC[combo]}\n(n={n_tot})', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        if di == 0 and ci == 0:
            ax.legend(loc='upper right', fontsize=7)

fig6.suptitle(
    'Unconstrained species behavior (green=grows, gray=neutral, red=declines)\n'
    'Aggregated across all cover edges by direction and state-combo',
    fontsize=FONT_SIZE, y=1.01)
fig6.tight_layout()
_save(fig6, 'fig6_nontarget_behavior')

# ---------------------------------------------------------------------------
# Fig 7: Directed transition graph (2×2 subplots, one per combo type)
# Node layout mirrors the Hasse diagram; edge width ∝ sqrt(n_ESMOs)
# ---------------------------------------------------------------------------
_NODE_POS = {
    'Tribe':      (0.5, 0.08),
    'Chief':      (0.18, 0.55),
    'State':      (0.82, 0.55),
    'ChiefState': (0.5,  0.92),
}
_NODE_COL = {
    'Tribe':      '#F9E79F',
    'Chief':      '#AED6F1',
    'State':      '#A9DFBF',
    'ChiefState': '#D7BDE2',
}

_all_n_vals = [r['n_esmos'] for r in RESULTS if r['n_esmos'] > 0]
_MAX_N      = max(_all_n_vals) if _all_n_vals else 1

fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)

for ci, combo in enumerate(['pp', 'pc', 'cp', 'cc']):
    ax = axes7[ci // 2][ci % 2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(COMBO_DESC[combo], fontsize=10, fontweight='bold',
                 color=_COMBO_COLORS[combo], pad=8)

    # Draw nodes
    for org, (nx, ny) in _NODE_POS.items():
        circ = plt.Circle((nx, ny), 0.08, color=_NODE_COL[org],
                           ec='#2C3E50', lw=1.8, zorder=5)
        ax.add_patch(circ)
        ax.text(nx, ny, ORG_ABBR[org], ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=6)

    # Draw cover-edge arrows (curved, one per direction)
    for down, up in COVER_EDGES:
        x0, y0 = _NODE_POS[down]
        x1, y1 = _NODE_POS[up]
        dx = x1 - x0; dy = y1 - y0
        dist = (dx**2 + dy**2) ** 0.5
        # Perpendicular unit vector (for label offset)
        px = -dy / dist; py = dx / dist

        for direction, (org_f, org_t), rad in [
                ('up',   (down, up),  +0.22),
                ('down', (up, down),  -0.22)]:
            r = _get_res(f'{down}-{up}', direction, combo)
            n = _n(r)
            if n == 0:
                continue
            lw = 0.8 + 7.0 * (n / _MAX_N) ** 0.5
            xf, yf = _NODE_POS[org_f]
            xt, yt = _NODE_POS[org_t]
            ax.annotate('',
                xy=(xt, yt), xytext=(xf, yf),
                arrowprops=dict(
                    arrowstyle='->', color=_COMBO_COLORS[combo],
                    lw=lw, mutation_scale=18,
                    connectionstyle=f'arc3,rad={rad}'),
                zorder=4)
            # Label at arc midpoint
            mx = (xf + xt) / 2 + rad * 0.45 * dist * px
            my = (yf + yt) / 2 + rad * 0.45 * dist * py
            ax.text(mx, my, str(n), fontsize=7, ha='center', va='center',
                    color=_COMBO_COLORS[combo], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.85, lw=0), zorder=7)

fig7.suptitle(
    'Directed ESMO transition graph — one panel per state-combo type\n'
    'Arrow width ∝ √(N_ESMOs).  Number = ESMO count.  No arrow = stoichiometrically infeasible.',
    fontsize=FONT_SIZE, y=1.01)
fig7.tight_layout()
_save(fig7, 'fig7_transition_graph')

# ---------------------------------------------------------------------------
# Console summary: asymmetry across combo types per edge
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("  ASYMMETRY SUMMARY PER COVER EDGE")
print(f"{'='*72}")
print(f"\n  {'Edge':<18}  combo  {'UP':>6}  {'DOWN':>6}  ratio(UP/DOWN)  interpretation")
print(f"  {'-'*18}  -----  {'-'*6}  {'-'*6}  {'-'*14}  ---------------")
for down, up in COVER_EDGES:
    for combo in ['pp', 'pc', 'cp', 'cc']:
        r_up = _get_res(f'{down}-{up}', 'up',   combo)
        r_dn = _get_res(f'{down}-{up}', 'down',  combo)
        n_up = _n(r_up)
        n_dn = _n(r_dn)
        if n_up + n_dn == 0:
            ratio_str = '   N/A   '
            interp    = 'both infeasible'
        elif n_dn == 0:
            ratio_str = '   inf   '
            interp    = 'only UP feasible'
        else:
            ratio = n_up / n_dn
            ratio_str = f'  {ratio:6.2f}  '
            if abs(ratio - 1.0) < 0.15:
                interp = 'symmetric'
            elif ratio > 1:
                interp = f'going UP easier  ({n_up} vs {n_dn})'
            else:
                interp = f'going DOWN easier ({n_dn} vs {n_up})'
        print(f"  {down}-{up:<13}  {combo}  {n_up:>6}  {n_dn:>6}  {ratio_str}  {interp}")


print(f"\nAll outputs saved to: {_OUT_DIR}")
print("=" * 72)
