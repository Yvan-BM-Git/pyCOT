#!/usr/bin/env python3
"""
script_complex_ESMO.py — ESMO analysis of the ChiefState (O7) socio-political network
=======================================================================================
This script computes Elementary Signed Modes of Operation (ESMOs) for the
band–tribe–chiefdom–state reaction network defined in Complex_example.txt.

WHAT IS AN ESMO?
  An ESMO is a minimal combination of reactions (a flux vector v ≥ 0, normalised
  to sum(v)=1) such that a specified set of "target" species each has a required
  net production sign: some must grow (net > 0), others must shrink (net < 0).
  ESMOs are the extreme vertices of the corresponding sign-constrained polytope,
  found here by randomised LP sampling (HiGHS solver via scipy).

TWO TYPES OF ANALYSIS
  1. STATE-INDEPENDENT (BASE_SIGS):
       Signatures defined by how the social-organisation types 
       relate to displacement (X) and grievance (G).  Non-target species are
       unconstrained — the script reports how they behave across ESMOs.
       Outputs: ESMO counts, non-target species behaviour, reaction-usage
       distributions (with power-law / exponential fits), CSV tables.

  2. STATE-DEPENDENT (PATHWAY / conflict → peace):
       A transition signature derived from comparing an INITIAL_STATE (conflict)
       to a GOAL_STATE (peace) using a coverability rule:
         • Peace species need to reach a MINIMUM — constrained to grow only if
           initial < goal (already above minimum → no constraint).
         • Conflict species need to reach a MAXIMUM — constrained to shrink only
           if initial > goal (already below maximum → no constraint).
       Outputs: pathway ESMO count, neutral-species behaviour, reaction-usage
       distribution.

ALL USER SETTINGS ARE IN THE "INITIAL CONFIGURATIONS" BLOCK BELOW.
"""

import os, sys, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'src'))

from pyCOT.io.functions import read_txt

# ===========================================================================
# BEGIN INITIAL CONFIGURATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. WHAT TO RUN
#    Set each flag to True or False to enable / skip that analysis block.
#    Times are approximate for the Complex_example network on a typical laptop.
# ---------------------------------------------------------------------------
RUN_EFM_DIAGNOSTIC = True   # Count steady-state EFMs (all species balanced). ~2 min.
                             # Output: count printed to console.

RUN_BASE_SIGS      = True   # State-independent ESMO enumeration for BASE_SIGS. ~8 min.
                             # Outputs: fig_si_counts, fig_si_nontarget, fig_si_rxdist
                             #          + per-signature CSV files + esmo_counts.csv

RUN_PATHWAY        = True   # State-dependent pathway ESMOs (conflict→peace). ~45 min.
                             # Outputs: fig_sd_counts, fig_sd_nontarget, fig_sd_rxdist

# ---------------------------------------------------------------------------
# 2. NETWORK FILE AND OUTPUT DIRECTORY
#    Change RN_FILE to point to a different network; change OUT_DIR to redirect
#    all saved figures and CSV files.
# ---------------------------------------------------------------------------
RN_FILE = os.path.join(_SCRIPT_DIR, 'data', 'Complex_example.txt')
OUT_DIR = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'esmo_results')

# ---------------------------------------------------------------------------
# 3. BASE SIGNATURES  (used when RUN_BASE_SIGS = True)
#    Each entry is a label → {species: sign} dict.
#    sign = +1 means that species must have net production > 0 in every ESMO.
#    sign = -1 means net production < 0.
#    Non-listed species are unconstrained (their behaviour is observed, not imposed).
#
#    Current signatures capture four transitions between social-organisation types:
#      -H+XIG : chiefdom (H) declines while displaced (X), state (I), grievance (G) grow
#      -I+XHG : state (I) declines while displaced (X), chiefdom (H), grievance (G) grow
#      +H-XIG : chiefdom (H) grows while displaced (X), state (I), grievance (G) shrink
#      +I-XHG : state (I) grows while displaced (X), chiefdom (H), grievance (G) shrink
# ---------------------------------------------------------------------------
BASE_SIGS = {
    '-H+XIGC': {'H': -1, 'X': +1, 'I': +1, 'G': +1, 'C':+1},
    '-I+XHGC': {'I': -1, 'X': +1, 'H': +1, 'G': +1, 'C':+1},
    '+H-XIGC': {'H': +1, 'X': -1, 'I': -1, 'G': -1, 'C':+1},
    '+I-XHGC': {'I': +1, 'X': -1, 'H': -1, 'G': -1, 'C':+1},
}

# ---------------------------------------------------------------------------
# 4. PATHWAY ANALYSIS — INITIAL AND GOAL STATES  (used when RUN_PATHWAY = True)
#    Describe the conflict state (initial) and the peace target (goal) as
#    population / resource levels for each species.
#    These values define WHICH species are constrained and in which direction
#    (see Section 5 below for the classification rules).
#
#    Species glossary:
#      C     : Collective people (band/tribe)
#      X     : Displaced people
#      H     : Hierarchical people (chiefdom)
#      I     : Institutionalised people (state)
#      Res   : Resources (food, wealth)
#      C_Res : Community resilience / tribal surplus
#      H_Res : Chiefdom-controlled resources
#      I_Res : State-controlled resources
#      G     : Grievance
#      P_H   : Chiefdom military/police force
#      P_I   : State military/police force
#      L     : Legitimacy / rule of law / infrastructure
# ---------------------------------------------------------------------------
INITIAL_STATE = {            # conflict scenario
    'C':      200,
    'X':      250,
    'H':       50,
    'I':      200,
    'Res':     20,
    'C_Res':   20,
    'H_Res':   20,
    'I_Res':   20,
    'G':       50,
    'P_H':     50,
    'P_I':     50,
    'L':       40,
}

GOAL_STATE = {               # peace target
    'C':      350,
    'X':       50,
    'H':        0,
    'I':      250,
    'Res':     50,
    'C_Res':  150,
    'H_Res':    0,
    'I_Res':  200,
    'G':       10,
    'P_H':     20,
    'P_I':     80,
    'L':      100,
}

# ---------------------------------------------------------------------------
# 5. PATHWAY ANALYSIS — SPECIES CLASSIFICATION  (used when RUN_PATHWAY = True)
#    Determines how each species' constraint is derived from INITIAL/GOAL states.
#
#    PEACE_SPECIES    — species that must reach a MINIMUM level.
#                       A species in this set is constrained to GROW only if
#                       INITIAL < GOAL (i.e., it has not yet reached the minimum).
#                       If INITIAL >= GOAL it is already covered → neutral.
#
#    CONFLICT_SPECIES — species that must reach a MAXIMUM level.
#                       A species in this set is constrained to SHRINK only if
#                       INITIAL > GOAL (i.e., it is above the maximum).
#                       If INITIAL <= GOAL it is already below the ceiling → neutral.
#
#    Any species NOT listed in either set is always treated as neutral
#    (H, H_Res, I_Res are structural/resource intermediates with no direct
#    peace/conflict role in the transition scenario).
# ---------------------------------------------------------------------------
PEACE_SPECIES    = {'C', 'I', 'Res', 'C_Res', 'L'}
CONFLICT_SPECIES = {'X', 'G'}

# ---------------------------------------------------------------------------
# 6. FIGURE SETTINGS
#    FONT_SIZE controls all axis labels, tick labels, and legend text.
#    Increase it if exporting for a paper / poster (try 14–16).
# ---------------------------------------------------------------------------
FONT_SIZE = 12

# ---------------------------------------------------------------------------
# 7. PERFORMANCE / ADVANCED SETTINGS
#    These control the LP vertex-enumeration algorithm.
#    The defaults work well for the Complex_example network; change only if
#    you suspect incomplete enumeration (increase N_RAND_1, PATIENCE) or want
#    faster runs at the cost of completeness (decrease them).
# ---------------------------------------------------------------------------
TOL_STRICT = 1e-9   # threshold for "strictly nonzero" sign check
TOL_VERTEX = 1e-7   # tolerance for vertex deduplication
N_RAND_1   = 3000   # Phase-1 random LP samples (broad coverage)
N_EXPLORE  = 40     # neighbourhood reactions explored per discovered vertex
N_RAND_3   = 1000   # Phase-3 cleanup random samples
PATIENCE   = 300    # stop a phase after this many consecutive non-new samples
SEED       = 42     # random seed (set to any integer for reproducibility)
EPS_SIGN   = 1e-3   # ε-displacement for sign constraints — keeps vertices
                    # strictly inside the sign cone; do not set to 0

EXPECTED_EFM       = 75    # approximate EFM count used as a sanity-check reference
EFM_WARN_THRESHOLD = 0.10  # warn if actual count differs by more than this fraction

# ===========================================================================
# END INITIAL CONFIGURATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Derived paths (do not edit)
# ---------------------------------------------------------------------------
_OUT_DIR = os.path.realpath(OUT_DIR)
os.makedirs(_OUT_DIR, exist_ok=True)

print(f"Strict sign tolerance : {TOL_STRICT}")
print(f"Vertex deduplication  : {TOL_VERTEX}")
print(f"LP samples (phase 1/3): {N_RAND_1} / {N_RAND_3}   patience={PATIENCE}")
print(f"Output directory      : {_OUT_DIR}")

# ===========================================================================
# 1. LOAD NETWORK
# ===========================================================================
print(f"\nLoading: {RN_FILE}")
rn  = read_txt(RN_FILE)
_sm = rn.stoichiometry_matrix()

_full_sp = [str(s) for s in _sm.species]
_full_rx = [str(r) for r in _sm.reactions]
_S_full  = np.asarray(_sm, dtype=float)
print(f"  Full network: {len(_full_sp)} species, {len(_full_rx)} reactions")

# ---------------------------------------------------------------------------
# 2. BUILD O7 (ChiefState, no external aid) sub-network
# ---------------------------------------------------------------------------
O7_SPECIES  = {'C', 'X', 'H', 'I', 'Res', 'C_Res', 'H_Res', 'I_Res',
               'G', 'P_H', 'P_I', 'L'}
# r35 (F_I catalytic) and r36 (F_H catalytic) involve non-O7 species and are
# auto-excluded by sub_reaction_network.
EXCLUDE_RX  = set()

sp_obj   = [sp for sp in rn.species() if str(sp.name) in O7_SPECIES]
sub_rn   = rn.sub_reaction_network(sp_obj)
sub_sm   = sub_rn.stoichiometry_matrix()

_raw_sp = [str(s) for s in sub_sm.species]
_raw_rx = [str(r) for r in sub_sm.reactions]
_S_raw  = np.asarray(sub_sm, dtype=float)

_full_rx_idx = {r: i for i, r in enumerate(_full_rx)}
_order = sorted(range(len(_raw_rx)), key=lambda k: _full_rx_idx.get(_raw_rx[k], 9999))
_raw_rx = [_raw_rx[k] for k in _order]
_S_raw  = _S_raw[:, _order]

_keep = [r not in EXCLUDE_RX for r in _raw_rx]
O7_SP = _raw_sp
O7_RX = [r for r, k in zip(_raw_rx, _keep) if k]
S_O7  = _S_raw[:, _keep]

N_SP = len(O7_SP)
N_RX = len(O7_RX)

_excl_present = [r for r in EXCLUDE_RX if r in _raw_rx]
_excl_absent  = [r for r in EXCLUDE_RX if r not in _raw_rx]

print(f"\nO7 (ChiefState) sub-network (after exclusion of {EXCLUDE_RX}):")
print(f"  Species ({N_SP}): {O7_SP}")
print(f"  Reactions ({N_RX}): {O7_RX}")
print(f"  S_O7 shape: {N_SP} × {N_RX}")
if _excl_present:
    print(f"  Explicitly excluded: {_excl_present}")
if _excl_absent:
    print(f"  (already absent from sub-network: {_excl_absent})")

_rank = np.linalg.matrix_rank(S_O7)
print(f"  rank(S_O7) ≈ {_rank}  →  null-space dim ≈ {N_RX - _rank}")
print(f"  (each ESMO / EFM vertex has at most {_rank + 1} non-zero reactions)")

# ===========================================================================
# 3. SIGNATURES
# ===========================================================================
# Sign convention: +1 = require (Sv)_s > 0 ; -1 = require (Sv)_s < 0
#
# The O7 network has a conservation law: net(C+X+H+P_H+I+P_I) = 0
# (every reaction converts people between types; no creation/destruction).
# Resources, Grievance, and Legitimacy are NOT conserved.
# Compound signatures that pair a declining type with a rising one are
# consistent with this conservation law.
ALL_SIGS = BASE_SIGS

# ===========================================================================
# 4. LP HELPERS
# ===========================================================================

def build_lp(S, n_sp, n_rx, sig_dict, sp_list):
    """
    Build LP constraints for scipy.optimize.linprog.

    v ≥ 0  (bounds), sum(v) = 1  (normalisation),
    sign*(Sv)_s ≥ EPS_SIGN  for each target species.
    ε-displacement forces vertices into the strict interior of the sign cone.
    Non-target species are unconstrained.
    """
    target_map = {}
    for sp_name, sign in sig_dict.items():
        if sp_name in sp_list:
            target_map[sp_list.index(sp_name)] = sign
        else:
            print(f"  [WARN] target species '{sp_name}' not in sub-network — skipped")

    A_eq_rows = [np.ones(n_rx)]
    b_eq_vals = [1.0]

    A_ub_rows, b_ub_vals = [], []
    for si, sign in target_map.items():
        A_ub_rows.append(-sign * S[si, :].copy())
        b_ub_vals.append(-EPS_SIGN)

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_vals)
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if A_ub_rows else None

    return A_eq, b_eq, A_ub, b_ub


def is_feasible(A_eq, b_eq, A_ub, b_ub, n_rx):
    try:
        res = linprog(np.zeros(n_rx),
                      A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0.0, None)] * n_rx, method='highs')
        return res.success and res.status == 0
    except Exception:
        return False


def enumerate_vertices(A_eq, b_eq, A_ub, b_ub, n_rx,
                       n_rand1=N_RAND_1, n_explore=N_EXPLORE,
                       n_rand3=N_RAND_3, patience=PATIENCE, seed=SEED):
    """
    Enumerate polytope vertices via randomised LP sampling + neighbourhood
    exploration (Phase 1: random objectives, Phase 2: neighbour search,
    Phase 3: cleanup sweep).
    """
    rng    = np.random.default_rng(seed)
    bounds = [(0.0, None)] * n_rx
    opts   = {'dual_feasibility_tolerance': 1e-10,
              'primal_feasibility_tolerance': 1e-10,
              'presolve': True}

    vertices    = []
    vertex_keys = {}

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

    if not is_feasible(A_eq, b_eq, A_ub, b_ub, n_rx):
        return []

    no_new = 0
    for i in range(n_rand1):
        v = _solve(rng.standard_normal(n_rx))
        if v is not None:
            if _try_add(v): no_new = 0
            else:           no_new += 1
        if no_new >= patience and i >= 500:
            break

    n_after_p1 = len(vertices)
    for vi in range(n_after_p1):
        v_base   = vertices[vi]
        zero_idx = np.where(v_base < 1e-8)[0]
        explore  = (zero_idx if len(zero_idx) <= n_explore
                    else rng.choice(zero_idx, n_explore, replace=False))
        for j in explore:
            c = np.zeros(n_rx); c[j] = -1.0
            v = _solve(c)
            if v is not None and v[j] > 1e-8:
                _try_add(v)

    no_new = 0
    for i in range(n_rand3):
        v = _solve(rng.standard_normal(n_rx))
        if v is not None:
            if _try_add(v): no_new = 0
            else:           no_new += 1
        if no_new >= patience and i >= 200:
            break

    return vertices


def filter_strict(vertices, S, sp_list, sig_dict, tol=TOL_STRICT):
    kept = []
    for v in vertices:
        net = S @ v
        ok  = True
        for sp_name, sign in sig_dict.items():
            if sp_name not in sp_list:
                continue
            si = sp_list.index(sp_name)
            if sign > 0 and net[si] <= tol:
                ok = False; break
            if sign < 0 and net[si] >= -tol:
                ok = False; break
        if ok:
            kept.append(v)
    return kept


def verify_esmos(esmos, S, sp_list, sig_dict, tol=TOL_STRICT):
    violations = 0
    for idx, v in enumerate(esmos):
        net = S @ v
        for sp_name, sign in sig_dict.items():
            if sp_name not in sp_list:
                continue
            si  = sp_list.index(sp_name)
            val = net[si]
            if sign > 0 and val <= tol:
                print(f"    [VIOLATION] ESMO {idx}: {sp_name} net={val:.3e}, expected >0")
                violations += 1
            if sign < 0 and val >= -tol:
                print(f"    [VIOLATION] ESMO {idx}: {sp_name} net={val:.3e}, expected <0")
                violations += 1
    return violations


def compute_esmos(sig_dict, label, S=S_O7, sp_list=O7_SP, rx_list=O7_RX):
    """Full pipeline: build LP → enumerate vertices → verify sign constraints."""
    A_eq, b_eq, A_ub, b_ub = build_lp(S, N_SP, N_RX, sig_dict, sp_list)
    t0    = time.perf_counter()
    raw   = enumerate_vertices(A_eq, b_eq, A_ub, b_ub, N_RX)
    esmos = filter_strict(raw, S, sp_list, sig_dict)
    elapsed = time.perf_counter() - t0
    n_rej = len(raw) - len(esmos)
    print(f"    {len(raw):4d} raw vertices  →  {len(esmos):4d} ESMOs  "
          f"({n_rej} safety-rejected)  [{elapsed:.1f}s]")
    viol = verify_esmos(esmos, S, sp_list, sig_dict)
    if viol:
        print(f"    [WARN] {viol} verification violation(s) found!")
    return esmos, elapsed

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def esmos_to_df(esmos, sig_dict, sp_list, rx_list, S):
    sp_delta_cols = [f'delta_{s}' for s in sp_list]
    base_cols     = ['mode_id'] + rx_list + sp_delta_cols + ['support_size']
    if not esmos:
        return pd.DataFrame(columns=base_cols)
    rows = []
    for idx, v in enumerate(esmos, 1):
        net = S @ v
        row = {'mode_id': idx}
        row.update({r: float(v[j]) for j, r in enumerate(rx_list)})
        row.update({f'delta_{s}': float(net[i]) for i, s in enumerate(sp_list)})
        row['support_size'] = int(np.sum(v > TOL_STRICT))
        rows.append(row)
    return pd.DataFrame(rows, columns=base_cols)


def classify_nontarget(esmos, S, sp_list, sig_dict, tol=TOL_STRICT):
    """Count grow / neutral / decline across ESMOs for each non-target species."""
    target_sp = set(sig_dict.keys())
    nontarget = [s for s in sp_list if s not in target_sp]
    n         = len(esmos)
    counts    = {s: {'grow': 0, 'neutral': 0, 'decline': 0} for s in nontarget}
    for v in esmos:
        net = S @ v
        for s in nontarget:
            val = net[sp_list.index(s)]
            if   val >  tol: counts[s]['grow']    += 1
            elif val < -tol: counts[s]['decline']  += 1
            else:            counts[s]['neutral']  += 1
    return nontarget, counts, n

# ===========================================================================
# PLOTTING SETUP
# ===========================================================================

plt.rcParams.update({
    'font.size':       FONT_SIZE,
    'axes.titlesize':  FONT_SIZE,
    'axes.labelsize':  FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
})

_COL_GROW    = '#27AE60'
_COL_NEUTRAL = '#BDC3C7'
_COL_DECLINE = '#E74C3C'
_COL_BAR     = '#2980B9'


def _save(fig, stem):
    for ext in ('png', 'pdf'):
        p = os.path.join(_OUT_DIR, f'{stem}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close(fig)


from scipy.stats import linregress as _linregress


def _rx_dist_subplot(ax, esmos, rx_list, title, tol=TOL_STRICT):
    """Sorted reaction-usage frequency bar chart with power-law and exponential fits."""
    if not esmos:
        ax.text(0.5, 0.5, 'no ESMOs', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        return
    freq  = (np.vstack(esmos) > tol).astype(float).mean(axis=0)
    order = np.argsort(freq)[::-1]
    sfreq = freq[order]
    slabs = [rx_list[i] for i in order]
    ranks = np.arange(1, len(sfreq) + 1)
    xpos  = ranks - 1
    ax.bar(xpos, sfreq, color=_COL_BAR, alpha=0.75, zorder=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(slabs, rotation=90, fontsize=max(6, FONT_SIZE - 5))
    nz   = sfreq > 0
    x_nz = ranks[nz].astype(float)
    y_nz = sfreq[nz]
    if len(x_nz) >= 4:
        sl_pw, ic_pw, r_pw, *_ = _linregress(np.log(x_nz), np.log(y_nz))
        y_pw  = np.exp(ic_pw) * ranks ** sl_pw
        sl_ex, ic_ex, r_ex, *_ = _linregress(x_nz, np.log(y_nz))
        y_ex  = np.exp(ic_ex) * np.exp(sl_ex * ranks)
        ax.plot(xpos, y_pw, color='#E74C3C', lw=2, zorder=3,
                label=f'Power law    R²={r_pw**2:.3f}')
        ax.plot(xpos, y_ex, color='#27AE60', lw=2, ls='--', zorder=3,
                label=f'Exponential  R²={r_ex**2:.3f}')
        ax.legend()
    top = sfreq[0] if len(sfreq) > 0 else 1.0
    ax.set_ylim(0, min(1.1, top + 0.05))
    ax.set_ylabel('Fraction of ESMOs')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

# ===========================================================================
# MAIN
# ===========================================================================

def main():

    # -----------------------------------------------------------------------
    # 5. DIAGNOSTIC: STEADY-STATE EFM COUNT
    # -----------------------------------------------------------------------
    N_EFM = 0
    if RUN_EFM_DIAGNOSTIC:
        print("\n" + "="*72)
        print("  DIAGNOSTIC — steady-state EFMs  (all species balanced)")
        print("="*72)
        _efm_Aeq = np.vstack([np.ones((1, N_RX)), S_O7])
        _efm_beq = np.zeros(1 + N_SP); _efm_beq[0] = 1.0
        _efm_raw  = enumerate_vertices(_efm_Aeq, _efm_beq, None, None, N_RX,
                                       n_rand1=5000, n_explore=N_EXPLORE,
                                       n_rand3=2000, patience=400)
        N_EFM = len(_efm_raw)
        print(f"  Found {N_EFM} steady-state EFMs")
        if abs(N_EFM - EXPECTED_EFM) / max(EXPECTED_EFM, 1) > EFM_WARN_THRESHOLD:
            print(f"  [NOTE] Differs from expected ~{EXPECTED_EFM}  "
                  f"(ratio {N_EFM/EXPECTED_EFM:.2f}x).  "
                  f"Expected value was from a sampling approach, exact count may differ.")
        else:
            print(f"  [OK] within 10% of expected ~{EXPECTED_EFM}")

    # -----------------------------------------------------------------------
    # 6-10. STATE-INDEPENDENT (BASE_SIGS)
    # -----------------------------------------------------------------------
    if RUN_BASE_SIGS:

        # 6. Enumeration
        print("\n" + "="*72)
        print(f"  ESMO ENUMERATION  ({len(ALL_SIGS)} signatures, free non-target)")
        print("="*72)
        results = {}
        timings = {}
        for sig_label, sig_dict in ALL_SIGS.items():
            print(f"\n[Signature: {sig_label}]  target = {sig_dict}")
            esmos, t = compute_esmos(sig_dict, sig_label)
            results[sig_label] = esmos
            timings[sig_label] = t

        # 7. CSV outputs
        print("\n--- Saving per-signature CSV files ---")
        count_records = []
        for sig_label, sig_dict in ALL_SIGS.items():
            esmos = results[sig_label]
            df    = esmos_to_df(esmos, sig_dict, O7_SP, O7_RX, S_O7)
            fpath = os.path.join(_OUT_DIR, f'esmo_{sig_label}.csv')
            df.to_csv(fpath, index=False)
            n    = len(esmos)
            supp = [int(np.sum(v > TOL_STRICT)) for v in esmos] if n > 0 else []
            count_records.append({
                'signature':          sig_label,
                'n_esmos':            n,
                'mean_support_size':  round(float(np.mean(supp)), 2) if supp else 0.0,
                'min_support_size':   min(supp) if supp else 0,
                'max_support_size':   max(supp) if supp else 0,
                'time_s':             round(timings[sig_label], 1),
            })
            print(f"  {sig_label:20s}: {n:4d} ESMOs  -> {os.path.basename(fpath)}")
        count_df = pd.DataFrame(count_records)
        master_path = os.path.join(_OUT_DIR, 'esmo_counts.csv')
        count_df.to_csv(master_path, index=False)
        print(f"\nMaster table -> {master_path}")
        print("\n" + count_df.to_string(index=False))

        # 8. Non-target species analysis
        nontarget_data = {}
        for sig_label, sig_dict in ALL_SIGS.items():
            nt, cnt, n = classify_nontarget(results[sig_label], S_O7, O7_SP, sig_dict)
            nontarget_data[sig_label] = {'species': nt, 'counts': cnt, 'n': n}

        # 9. Figures
        sig_labels_all = list(ALL_SIGS.keys())
        n_sigs = len(sig_labels_all)
        ncols  = min(n_sigs, 2)
        nrows  = (n_sigs + ncols - 1) // ncols

        # fig_si_counts — bar chart of ESMO count per signature
        x_si   = np.arange(n_sigs)
        cnt_si = [len(results[s]) for s in sig_labels_all]
        fig_sic, ax_sic = plt.subplots(figsize=(max(7, n_sigs * 2.0), 5))
        bars_sic = ax_sic.bar(x_si, cnt_si, 0.6, color=_COL_BAR, alpha=0.85)
        for bar in bars_sic:
            h = bar.get_height()
            if h > 0:
                ax_sic.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
                            ha='center', va='bottom')
        ax_sic.set_xticks(x_si)
        ax_sic.set_xticklabels(sig_labels_all, rotation=15, ha='right')
        ax_sic.set_ylabel('Number of ESMOs')
        ax_sic.set_title('ESMO counts per signature  (free non-target species)')
        ax_sic.grid(axis='y', alpha=0.3)
        ax_sic.set_axisbelow(True)
        fig_sic.tight_layout()
        _save(fig_sic, 'fig_si_counts')

        # fig_si_nontarget — stacked bars: grow/neutral/decline per non-target species
        fig_sint, axes_sint = plt.subplots(nrows, ncols,
                                            figsize=(ncols * 9, nrows * 5),
                                            squeeze=False)
        for plot_i, sig_label in enumerate(sig_labels_all):
            ax   = axes_sint[plot_i // ncols][plot_i % ncols]
            data = nontarget_data[sig_label]
            nt_sp, cnt_d, n_tot = data['species'], data['counts'], data['n']
            if n_tot == 0 or not nt_sp:
                ax.set_title(f'{sig_label}  (no ESMOs)')
                ax.set_visible(False)
                continue
            x      = np.arange(len(nt_sp))
            f_grow = np.array([cnt_d[s]['grow']    / n_tot for s in nt_sp])
            f_neut = np.array([cnt_d[s]['neutral'] / n_tot for s in nt_sp])
            f_decl = np.array([cnt_d[s]['decline'] / n_tot for s in nt_sp])
            ax.bar(x, f_grow,                         color=_COL_GROW,    alpha=0.85, label='grows')
            ax.bar(x, f_neut, bottom=f_grow,           color=_COL_NEUTRAL, alpha=0.75, label='neutral')
            ax.bar(x, f_decl, bottom=f_grow + f_neut,  color=_COL_DECLINE, alpha=0.85, label='declines')
            ax.set_xticks(x)
            ax.set_xticklabels(nt_sp, rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Fraction of ESMOs')
            ax.set_title(f'{sig_label}   (n={n_tot})')
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            if plot_i == 0:
                ax.legend(loc='upper right')
        for plot_i in range(n_sigs, nrows * ncols):
            axes_sint[plot_i // ncols][plot_i % ncols].set_visible(False)
        fig_sint.suptitle('Non-target species behavior — BASE_SIGS ESMOs',
                          fontsize=FONT_SIZE + 1, y=1.01)
        fig_sint.tight_layout()
        _save(fig_sint, 'fig_si_nontarget')

        # fig_si_rxdist — sorted reaction-usage frequency with curve fits
        fig_sirx, axes_sirx = plt.subplots(nrows, ncols,
                                            figsize=(ncols * 9, nrows * 5),
                                            squeeze=False)
        for plot_i, sig_label in enumerate(sig_labels_all):
            ax = axes_sirx[plot_i // ncols][plot_i % ncols]
            _rx_dist_subplot(ax, results[sig_label], O7_RX,
                             title=f'{sig_label}   (n={len(results[sig_label])})')
        for plot_i in range(n_sigs, nrows * ncols):
            axes_sirx[plot_i // ncols][plot_i % ncols].set_visible(False)
        fig_sirx.suptitle('Reaction usage frequency (sorted) — BASE_SIGS',
                          fontsize=FONT_SIZE + 1, y=1.01)
        fig_sirx.tight_layout()
        _save(fig_sirx, 'fig_si_rxdist')

        # 10. Console summary
        print("\n" + "="*72)
        print("  CONSOLE SUMMARY  (state-independent / BASE_SIGS)")
        print("="*72)
        print(f"\nSteady-state EFMs: {N_EFM}  (reference: ~{EXPECTED_EFM})\n")
        print("-- ESMO counts --")
        print(f"  {'Signature':<22}  {'n_ESMOs':>8}  {'mean_supp':>10}  {'min':>5}  {'max':>5}  {'time_s':>7}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*7}")
        for sig in ALL_SIGS:
            esmos = results[sig]
            n = len(esmos)
            if n > 0:
                supp = [int(np.sum(v > TOL_STRICT)) for v in esmos]
                print(f"  {sig:<22}  {n:>8}  {np.mean(supp):>10.2f}  {min(supp):>5}  {max(supp):>5}  {timings[sig]:>7.1f}")
            else:
                print(f"  {sig:<22}  {n:>8}  {'--':>10}  {'--':>5}  {'--':>5}  {timings[sig]:>7.1f}")
        print("\n-- Non-target species behavior (fraction of ESMOs) --")
        for sig_label in ALL_SIGS:
            data  = nontarget_data[sig_label]
            n_tot = data['n']
            if n_tot == 0:
                print(f"  [{sig_label}] no ESMOs")
                continue
            print(f"  [{sig_label}]  n={n_tot}")
            for s in data['species']:
                g = data['counts'][s]['grow']
                z = data['counts'][s]['neutral']
                d = data['counts'][s]['decline']
                print(f"    {s:<10}  grow={g:3d} ({100*g/n_tot:5.1f}%)  "
                      f"neutral={z:3d} ({100*z/n_tot:5.1f}%)  "
                      f"decline={d:3d} ({100*d/n_tot:5.1f}%)")

    # -----------------------------------------------------------------------
    # 11. STATE-DEPENDENT (PATHWAY ANALYSIS)
    # -----------------------------------------------------------------------
    if RUN_PATHWAY:

        # Derive transition signature from INITIAL_STATE / GOAL_STATE using
        # the coverability rules defined in PEACE_SPECIES / CONFLICT_SPECIES.
        NEUTRAL_TOL = 1e-9
        trans_sig   = {}
        must_grow   = []
        must_shrink = []
        neutral_sp  = []

        for s in O7_SP:
            iv, gv = INITIAL_STATE.get(s, 0.0), GOAL_STATE.get(s, 0.0)
            d = gv - iv
            if s in PEACE_SPECIES:
                if d > NEUTRAL_TOL:
                    trans_sig[s] = +1;  must_grow.append(s)
                else:
                    neutral_sp.append(s)
            elif s in CONFLICT_SPECIES:
                if d < -NEUTRAL_TOL:
                    trans_sig[s] = -1;  must_shrink.append(s)
                else:
                    neutral_sp.append(s)
            else:
                neutral_sp.append(s)

        print("\n" + "="*72)
        print("  PATHWAY ANALYSIS  (conflict → peace)  — coverability formulation")
        print("="*72)
        print(f"  Peace   species (minimum bound): {sorted(PEACE_SPECIES)}")
        print(f"  Conflict species (maximum bound): {sorted(CONFLICT_SPECIES)}")
        print(f"  Must GROW   — peace below goal     : {must_grow}")
        print(f"  Must SHRINK — conflict above goal  : {must_shrink}")
        print(f"  NEUTRAL     — covered or neutral   : {neutral_sp}")
        print(f"  Transition signature ({len(trans_sig)} targets): {trans_sig}")

        print("\n--- Enumerating pathway ESMOs ---")
        trans_esmos, trans_time = compute_esmos(trans_sig, 'conflict_to_peace')
        print(f"  Found {len(trans_esmos)} pathway ESMOs  [{trans_time:.1f}s]")
        if trans_esmos:
            supp_all = [int(np.sum(v > TOL_STRICT)) for v in trans_esmos]
            print(f"  Support size — min={min(supp_all)}, max={max(supp_all)}, "
                  f"mean={np.mean(supp_all):.1f}, median={float(np.median(supp_all)):.1f}")

        # fig_sd_counts — single bar showing total pathway ESMO count
        fig_sdc, ax_sdc = plt.subplots(figsize=(4, 5))
        n_te = len(trans_esmos)
        ax_sdc.bar([0], [n_te], 0.4, color=_COL_BAR, alpha=0.85)
        ax_sdc.text(0, n_te + 1, str(n_te), ha='center', va='bottom')
        ax_sdc.set_xticks([0])
        ax_sdc.set_xticklabels(['conflict → peace'])
        ax_sdc.set_ylabel('Number of pathway ESMOs')
        ax_sdc.set_title('Pathway ESMO count\n(state-dependent transition)')
        ax_sdc.grid(axis='y', alpha=0.3)
        ax_sdc.set_axisbelow(True)
        fig_sdc.tight_layout()
        _save(fig_sdc, 'fig_sd_counts')

        # fig_sd_nontarget — grow/neutral/decline for unconstrained species
        if trans_esmos and neutral_sp:
            nt_trans, cnt_trans, n_trans = classify_nontarget(
                trans_esmos, S_O7, O7_SP, trans_sig)
            if nt_trans:
                x_nt = np.arange(len(nt_trans))
                f_g  = np.array([cnt_trans[s]['grow']    / n_trans for s in nt_trans])
                f_n  = np.array([cnt_trans[s]['neutral'] / n_trans for s in nt_trans])
                f_d  = np.array([cnt_trans[s]['decline'] / n_trans for s in nt_trans])
                fig_sdn, ax_sdn = plt.subplots(figsize=(max(7, len(nt_trans) * 1.4), 5))
                ax_sdn.bar(x_nt, f_g,              color=_COL_GROW,    alpha=0.85, label='grows')
                ax_sdn.bar(x_nt, f_n, bottom=f_g,  color=_COL_NEUTRAL, alpha=0.75, label='neutral')
                ax_sdn.bar(x_nt, f_d, bottom=f_g + f_n,
                           color=_COL_DECLINE, alpha=0.85, label='declines')
                ax_sdn.set_xticks(x_nt)
                ax_sdn.set_xticklabels(nt_trans, rotation=45, ha='right')
                ax_sdn.set_ylim(0, 1.05)
                ax_sdn.set_ylabel('Fraction of pathway ESMOs')
                ax_sdn.set_title(
                    f'Neutral species behavior — conflict→peace pathways  (n={n_trans})')
                ax_sdn.legend(loc='upper right')
                ax_sdn.grid(axis='y', alpha=0.3)
                ax_sdn.set_axisbelow(True)
                fig_sdn.tight_layout()
                _save(fig_sdn, 'fig_sd_nontarget')
                print("\n-- Neutral species in transition ESMOs --")
                for s in nt_trans:
                    g = cnt_trans[s]['grow']; z = cnt_trans[s]['neutral']; d = cnt_trans[s]['decline']
                    print(f"  {s:<10}  grow={g:4d} ({100*g/n_trans:5.1f}%)  "
                          f"neutral={z:4d} ({100*z/n_trans:5.1f}%)  "
                          f"decline={d:4d} ({100*d/n_trans:5.1f}%)")

        # fig_sd_rxdist — sorted reaction-usage frequency with curve fits
        if trans_esmos:
            fig_sdrx, ax_sdrx = plt.subplots(figsize=(max(12, N_RX * 0.5 + 2), 5))
            _rx_dist_subplot(ax_sdrx, trans_esmos, O7_RX,
                             title=f'Reaction usage — conflict→peace pathways  (n={len(trans_esmos)})')
            fig_sdrx.tight_layout()
            _save(fig_sdrx, 'fig_sd_rxdist')

    print(f"\nAll outputs saved to: {_OUT_DIR}")
    print("="*72)


if __name__ == '__main__':
    main()
