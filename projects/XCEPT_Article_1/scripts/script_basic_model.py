#!/usr/bin/env python3
"""
script_basic_model.py
=====================
Basic band-tribe conflict model.
Loads the 5-species, 11-reaction network from Basic_example.txt (r0-r10).
Produces ODE and stochastic timeseries + phase portraits for two scenarios.

Species:   C (Collective), X (Displaced), Res (Resources),
           C_Res (Community Resilience), G (Grievance)
Reactions: r0-r7 (band-tribe basics) + r8, r9, r10 (conflict dynamics)

Modifiers:
  alpha_solidarity — scales r2 (surplus growth) and r5 (reintegration)
  alpha_reactivity — scales r8 (grievance generation) and r9 (displacement by grievance)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PYCOT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, os.path.join(_PYCOT_ROOT, 'src'))

from pyCOT.io.functions     import read_txt
from pyCOT.simulations.core import build_reaction_dict

# -- Output directory ----------------------------------------------------------
_OUT_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', 'outputs', 'basic_model'))
os.makedirs(_OUT_DIR, exist_ok=True)

# ===========================================================================
# NETWORK + STOICHIOMETRY  (loaded from Basic_example.txt, same pattern as
# script_intermediate_model.py loads from Intrmediate_example.txt)
# ===========================================================================
_RN_FILE = os.path.join(_SCRIPT_DIR, 'data', 'Basic_example.txt')
_rn      = read_txt(_RN_FILE)
_sm      = _rn.stoichiometry_matrix()

# Canonical species and reaction order for this script
SPECIES   = ['C', 'X', 'Res', 'C_Res', 'G']
REACTIONS = [f'r{i}' for i in range(11)]   # r0 through r10

N_SPECIES   = len(SPECIES)
N_REACTIONS = len(REACTIONS)

# Reorder pyCOT's internal ordering to match our canonical order
_pycot_sp = list(_sm.species)
_pycot_rx = list(_sm.reactions)
_row_idx  = [_pycot_sp.index(s) for s in SPECIES]
_col_idx  = [_pycot_rx.index(r) for r in REACTIONS]
STOICH    = np.array(_sm, dtype=float)[np.ix_(_row_idx, _col_idx)]   # (5, 11)

# Input stoichiometry for enabling degrees (stochastic simulation)
_rxn_dict    = build_reaction_dict(_rn)
INPUT_STOICH = np.zeros((N_SPECIES, N_REACTIONS))
for j, rxn_name in enumerate(REACTIONS):
    reactants, _ = _rxn_dict[rxn_name]
    for sp, coef in reactants:
        if sp in SPECIES:
            INPUT_STOICH[SPECIES.index(sp), j] = coef

# Species index shortcuts
iC, iX, iRes, iCRes, iG = 0, 1, 2, 3, 4

# ===========================================================================
# USER-EDITABLE PARAMETERS
# ===========================================================================

# -- Initial conditions --------------------------------------------------------
C0    = 10.0   # collective population
X0_IC =  3.0   # displaced population
Res0  = 5.0   # resources
CRes0 =  2.0   # community resilience
G0    =  1.0   # grievance

# -- Base kinetic constants (one per reaction, ordered r0-r10) ----------------
#
# Modifier reactions:
#   index 2  = r2  (alpha_solidarity: surplus growth)
#   index 5  = r5  (alpha_solidarity: reintegration)
#   index 8  = r8  (alpha_reactivity: grievance generation)     <- old r15
#   index 9  = r9  (alpha_reactivity: displacement by grievance) <- old r16
#
# Note: r6 is now first-order (Res =>) unlike the intermediate model (2Res =>).
#       r10 (G =>) is grievance decay, formerly r19 in the intermediate model.
KAPPA = np.array([
    0.2,    # r0  => Res                 constant resource inflow
    0.02,    # r1  C + Res => C           band consumption
    0.10,    # r2  C+Res+C_Res=>C+2C_Res [SOLIDARITY] surplus growth
    0.01,    # r3  C_Res => Res           redistribution
    0.01,    # r4  C => X                 spontaneous displacement
    0.10,    # r5  X + C_Res => C         [SOLIDARITY] reintegration
    0.005,   # r6  2Res =>                resource decay  (second-order: rate = k*Res^2)
    0.01,    # r7  2C_Res =>              C_Res decay     (second-order: rate = k*CRes^2)
    0.01,    # r8  X+C+Res => X+C+G       [REACTIVITY] grievance generation
    0.05,    # r9  2G+C => X              [REACTIVITY] displacement by grievance
    0.005,   # r10 2G =>                  grievance decay (second-order: rate = k*G^2)
])

# Michaelis-Menten saturation constants (ODE only)
# Only reactions listed as saturated in the reaction table carry a KM.
# r9 (2G+C->X) has NO saturation per table -- pure mass action k*G^2*C.
KM_r1 = 3.0   # Res   saturation in r1: band consumption
KM_r2 = 3.0   # Res   saturation in r2: surplus growth
KM_r3 = 2.0   # C_Res saturation in r3: resilience redistribution
KM_r8 = 3.0   # Res   saturation in r8: grievance generation

# -- ODE integration settings --------------------------------------------------
T_END   = 400.0
N_STEPS = 2000

# -- Stochastic simulation settings --------------------------------------------
N_STOCH_STEPS = 300
DT_STOCH      = 1.0

# -- Scenario A (dashes -- in all plots) ---------------------------------------
# High community investment in resilience and reintegration; low conflict tension.
ALPHA_REACT_A = 0.5    # scales r8 and r9  (conflict reactivity)
ALPHA_SOL_A   = 2    # scales r2 and r5  (community solidarity)
LABEL_A       = 'High solidarity, low reactivity'
SEED_A        = 42

# -- Scenario B (solid  --  in all plots) --------------------------------------
# Low investment in reintegration; high grievance dynamics.
ALPHA_REACT_B = 3.0    # higher -> more grievance -> more displacement
ALPHA_SOL_B   = 0.3    # lower  -> slower reintegration, weaker surplus
LABEL_B       = 'Low solidarity, high reactivity'
SEED_B        = 42

# ===========================================================================
# COLOUR PALETTE
# ===========================================================================
_CRED  = '#C0392B'   # X / displacement
_CBLUE = '#2980B9'   # C (collective)
_CGRN  = '#27AE60'   # C_Res (community resilience)
_CORNG = '#F39C12'   # Res (resources)
_CGRAY = '#7F8C8D'   # G (grievance)

# ===========================================================================
# ODE
# ===========================================================================

def _ode_rhs(t, y, alpha_react, alpha_sol):
    """Michaelis-Menten ODE RHS for the basic 5-species, 11-reaction model."""
    C, X, Res, CRes, G = y
    # protect denominators from zero
    Res_  = max(Res,  1e-9)
    CRes_ = max(CRes, 1e-9)
    G_    = max(G,    1e-9)

    v0  = KAPPA[0]
    v1  = KAPPA[1]                * C   * Res_  / (Res_  + KM_r1)
    v2  = alpha_sol  * KAPPA[2]   * C   * Res_  * CRes_ / (Res_  + KM_r2)
    v3  = KAPPA[3]                * CRes_ / (CRes_ + KM_r3)        # r3: MM in C_Res
    v4  = KAPPA[4]                * C
    v5  = alpha_sol  * KAPPA[5]   * X   * CRes_
    v6  = KAPPA[6]                * Res_  ** 2                     # r6: 2Res =>  second-order
    v7  = KAPPA[7]                * CRes_ ** 2                     # r7: 2C_Res => second-order
    v8  = alpha_react * KAPPA[8]  * X   * C   * Res_  / (Res_  + KM_r8)
    v9  = alpha_react * KAPPA[9]  * G_  ** 2  * C                  # r9: pure mass action, no saturation
    v10 = KAPPA[10]               * G_  ** 2                       # r10: 2G =>  second-order

    # dX = STOICH @ v  (stoich coefficients: r6 gives -2*Res, r7 gives -2*CRes, r10 gives -2*G)
    dC    = -v4 + v5 - v9
    dX    = +v4 - v5 + v9
    dRes  = +v0 - v1 - v2 + v3 - 2*v6 - v8
    dCRes = +v2 - v3 - v5 - 2*v7
    dG    = +v8 - 2*v9 - 2*v10
    return [dC, dX, dRes, dCRes, dG]


def run_ode(alpha_react, alpha_sol):
    """Integrate the ODE. Returns (t, y) where y.shape = (5, N_STEPS+1)."""
    y0  = np.array([C0, X0_IC, Res0, CRes0, G0])
    sol = solve_ivp(
        _ode_rhs,
        (0.0, T_END),
        y0,
        method='LSODA',
        args=(alpha_react, alpha_sol),
        t_eval=np.linspace(0.0, T_END, N_STEPS + 1),
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y   # sol.y[species_idx, time_idx]


# ===========================================================================
# STOCHASTIC  (Poisson tau-leaping)
# ===========================================================================
#
# WHY KAPPA_STOCH IS SEPARATE FROM KAPPA
# ----------------------------------------
# KAPPA (above) drives Michaelis-Menten ODE kinetics.  The rate of reaction j is:
#   v_j = k_j * (product of reactant concentrations, with MM saturation)
# e.g. r2:  v2 = k * C * Res * C_Res / (Res + Km)
#
# The stochastic simulation uses enabling degrees:
#   lambda_j = k_j * pi_j(x),  where pi_j = min_s { x[s] / INPUT_STOICH[s,j] }
# e.g. r2:  lambda2 = k * min(C, Res, C_Res)          (bottleneck reactant)
#
# At initial conditions C=10, Res=5, C_Res=2:
#   ODE rate (r2):   k * 10 * 5 * 2 / (5+3)  =  12.5 * k
#   Stoch lambda(r2):  k * min(10, 5, 2)      =   2.0 * k   (6x lower!)
#
# The ODE rate grows multiplicatively with every reactant (synergistic product).
# The enabling degree captures only the BOTTLENECK reactant (scarcest one limits
# the reaction).  There is no single universal rescaling factor because the ratio
# depends on the current state and reaction stoichiometry.
#
# Practical calibration guideline:
#   KAPPA_STOCH[j] ~ KAPPA[j] * (ODE rate at x0) / (enabling degree at x0)
# This matches propensities at t=0 but they diverge later — treat both sets as
# independent parameterisations of the same qualitative model.
#
# MODIFIER REACTIONS (alpha scalings apply to both ODE and stochastic):
#   index 2 = r2  (alpha_solidarity: surplus growth)
#   index 5 = r5  (alpha_solidarity: reintegration)
#   index 8 = r8  (alpha_reactivity: grievance generation)
#   index 9 = r9  (alpha_reactivity: displacement by grievance)

KAPPA_STOCH = np.array([
    2,    # r0  => Res                  (source: pi=1, matches ODE directly)
    0.025,   # r1  C + Res => C            (pi = min(C,Res); ODE/pi ratio ~0.025 at x0)
    0.50,    # r2  C+Res+C_Res=>C+2C_Res  [SOLIDARITY]  (pi = min(C,Res,CRes)=2; ODE/pi~6x)
    0.02,    # r3  C_Res => Res            (pi = CRes; ODE/pi ratio ~0.01)
    0.01,    # r4  C => X                  (pi = C; ratio ~1, so same as ODE)
    0.1,    # r5  X + C_Res => C          [SOLIDARITY]  (pi = min(X,CRes)=2; ODE/pi~0.3)
    0.05,    # r6  2Res =>                 (pi = Res/2; ODE rate=k*Res^2; ratio~2.5 at x0)
    0.02,    # r7  2C_Res =>               (pi = CRes/2; ODE rate=k*CRes^2; ratio~1 at x0)
    0.05,    # r8  X+C+Res => X+C+G       [REACTIVITY]  (pi = min(X,C,Res)=3; ODE/pi~6x)
    0.1,    # r9  2G+C => X              [REACTIVITY]  (pi = min(G/2,C)=0.5; ODE/pi~3x)
    0.01,    # r10 2G =>                   (pi = G/2=0.5; ODE rate=k*G^2; ratio~2 at x0)
])


def _enabling_degrees(x):
    """pi_j(x) = min_{s: INPUT_STOICH[s,j]>0}  x[s] / INPUT_STOICH[s,j]."""
    pi = np.ones(N_REACTIONS)
    for j in range(N_REACTIONS):
        col    = INPUT_STOICH[:, j]
        active = col > 0
        if active.any():
            pi[j] = np.min(x[active] / col[active])
    return np.clip(pi, 0.0, None)


def _resolve_contention(x, n, max_iter=20):
    """Scale down firings that would push any species negative."""
    n = n.copy()
    for _ in range(max_iter):
        delta = STOICH @ n
        if np.all(x + delta >= -1e-10):
            break
        for s in range(N_SPECIES):
            if x[s] + delta[s] < -1e-10:
                consuming = STOICH[s] < 0
                if consuming.any():
                    total_drain = -(STOICH[s] @ n)
                    if total_drain > 1e-12:
                        scale = max(0.0, x[s] / total_drain)
                        n[consuming] *= scale
                        delta = STOICH @ n
    return n


def _step_poisson(x, kappa_eff, dt, rng):
    pi  = _enabling_degrees(x)
    lam = kappa_eff * pi
    n   = rng.poisson(lam * dt).astype(float)
    n   = np.minimum(n, np.floor(pi))
    n   = _resolve_contention(x, n)
    return np.maximum(x + STOICH @ n, 0.0)


def run_stochastic(alpha_react, alpha_sol, seed=42):
    """Run Poisson tau-leaping for N_STOCH_STEPS steps. Returns X[n+1, 5]."""
    kappa_eff     = KAPPA_STOCH.copy()
    kappa_eff[2] *= alpha_sol    # r2  solidarity (surplus growth)
    kappa_eff[5] *= alpha_sol    # r5  solidarity (reintegration)
    kappa_eff[8] *= alpha_react  # r8  reactivity (grievance generation)
    kappa_eff[9] *= alpha_react  # r9  reactivity (displacement by grievance)

    rng  = np.random.default_rng(seed)
    X    = np.zeros((N_STOCH_STEPS + 1, N_SPECIES))
    X[0] = np.array([C0, X0_IC, Res0, CRes0, G0])
    for t in range(N_STOCH_STEPS):
        X[t + 1] = _step_poisson(X[t], kappa_eff, DT_STOCH, rng)
    return X


# ===========================================================================
# PLOTTING HELPERS
# ===========================================================================

def _gradient_line(ax, xv, yv, cmap_name, lw, ls, alpha, start_shade=0.3):
    """Draw xv vs yv as a gradient-coloured trajectory (early=light, late=dark)."""
    n    = len(xv)
    cmap = plt.colormaps[cmap_name]
    for i in range(n - 1):
        c = cmap(start_shade + (1 - start_shade) * i / max(n - 1, 1))
        ax.plot(xv[i:i + 2], yv[i:i + 2], color=c, lw=lw, ls=ls, alpha=alpha)


def _add_colorbars(fig, ax, cmap_A, cmap_B):
    """Attach scenario A and B colorbars to the right of ax."""
    sm_A = plt.cm.ScalarMappable(cmap=cmap_A, norm=plt.Normalize(0, 1))
    sm_B = plt.cm.ScalarMappable(cmap=cmap_B, norm=plt.Normalize(0, 1))
    sm_A.set_array([])
    sm_B.set_array([])
    cb_A = fig.colorbar(sm_A, ax=ax, pad=0.02, fraction=0.03, aspect=25)
    cb_B = fig.colorbar(sm_B, ax=ax, pad=0.14, fraction=0.03, aspect=25)
    cb_A.set_label(f'{LABEL_A}  (early to late)', fontsize=7)
    cb_B.set_label(f'{LABEL_B}  (early to late)', fontsize=7)


# ===========================================================================
# FIGURE B1 -- ODE TIMESERIES  (3 panels: C/X | G | Res/C_Res)
# ===========================================================================

def plot_ode_timeseries(t_A, y_A, t_B, y_B, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(t_A, y_A[iC], color=_CBLUE, ls='--', lw=1.8, label=f'C  ({LABEL_A})')
    ax.plot(t_B, y_B[iC], color=_CBLUE, ls='-',  lw=2.0, label=f'C  ({LABEL_B})')
    ax.plot(t_A, y_A[iX], color=_CRED,  ls='--', lw=1.8, label=f'X  ({LABEL_A})')
    ax.plot(t_B, y_B[iX], color=_CRED,  ls='-',  lw=2.0, label=f'X  ({LABEL_B})')
    ax.set_title('Collective (C) and Displaced (X)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(t_A, y_A[iG], color=_CGRAY, ls='--', lw=1.8, label=f'G  ({LABEL_A})')
    ax.plot(t_B, y_B[iG], color=_CGRAY, ls='-',  lw=2.0, label=f'G  ({LABEL_B})')
    ax.set_title('Grievance (G)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(t_A, y_A[iRes],  color=_CORNG, ls='--', lw=1.8, label=f'Res   ({LABEL_A})')
    ax.plot(t_B, y_B[iRes],  color=_CORNG, ls='-',  lw=2.0, label=f'Res   ({LABEL_B})')
    ax.plot(t_A, y_A[iCRes], color=_CGRN,  ls='--', lw=1.8, label=f'C_Res ({LABEL_A})')
    ax.plot(t_B, y_B[iCRes], color=_CGRN,  ls='-',  lw=2.0, label=f'C_Res ({LABEL_B})')
    ax.set_title('Resources (Res) and Community Resilience (C_Res)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'ODE Timeseries  --  {LABEL_A}  (--) vs  {LABEL_B}  (-)',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# FIGURE B1b -- ODE PHASE PORTRAITS  (2 panels: X vs C | Res vs C_Res)
# ===========================================================================

def plot_ode_phase(t_A, y_A, t_B, y_B, save_path=None):
    cmap_A = plt.colormaps['Blues']
    cmap_B = plt.colormaps['Oranges']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    _gradient_line(ax, y_A[iX], y_A[iC], 'Blues',   lw=1.3, ls='--', alpha=0.75)
    _gradient_line(ax, y_B[iX], y_B[iC], 'Oranges', lw=1.9, ls='-',  alpha=0.88)
    ax.scatter(y_A[iX][0],  y_A[iC][0],  color='black',      s=80,  zorder=6, marker='o', label='Start (shared)')
    ax.scatter(y_A[iX][-1], y_A[iC][-1], color='steelblue',  s=90,  zorder=6, marker='X', label=f'End -- {LABEL_A}')
    ax.scatter(y_B[iX][-1], y_B[iC][-1], color='darkorange', s=110, zorder=6, marker='*', label=f'End -- {LABEL_B}')
    ax.set_xlabel('X  (Displaced)'); ax.set_ylabel('C  (Collective)')
    ax.set_title('Phase portrait: Displaced vs Collective', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    _add_colorbars(fig, ax, cmap_A, cmap_B)

    ax = axes[1]
    _gradient_line(ax, y_A[iRes], y_A[iCRes], 'Blues',   lw=1.3, ls='--', alpha=0.75)
    _gradient_line(ax, y_B[iRes], y_B[iCRes], 'Oranges', lw=1.9, ls='-',  alpha=0.88)
    ax.scatter(y_A[iRes][0],  y_A[iCRes][0],  color='black',      s=80,  zorder=6, marker='o', label='Start (shared)')
    ax.scatter(y_A[iRes][-1], y_A[iCRes][-1], color='steelblue',  s=90,  zorder=6, marker='X', label=f'End -- {LABEL_A}')
    ax.scatter(y_B[iRes][-1], y_B[iCRes][-1], color='darkorange', s=110, zorder=6, marker='*', label=f'End -- {LABEL_B}')
    ax.set_xlabel('Res  (Resources)'); ax.set_ylabel('C_Res  (Community Resilience)')
    ax.set_title('Phase portrait: Resources vs Community Resilience', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    _add_colorbars(fig, ax, cmap_A, cmap_B)

    fig.suptitle(
        f'ODE Phase Portraits  --  Blues (--): {LABEL_A}   |   Oranges (-): {LABEL_B}',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# FIGURE B2 -- STOCHASTIC TIMESERIES  (3 panels: C/X | G | Res/C_Res)
# ===========================================================================

def plot_stoch_timeseries(X_A, X_B, save_path=None):
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    t_A, t_B = np.arange(n_A), np.arange(n_B)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(t_A, X_A[:, iC], color=_CBLUE, ls='--', lw=1.8, label=f'C  ({LABEL_A})')
    ax.plot(t_B, X_B[:, iC], color=_CBLUE, ls='-',  lw=2.0, label=f'C  ({LABEL_B})')
    ax.plot(t_A, X_A[:, iX], color=_CRED,  ls='--', lw=1.8, label=f'X  ({LABEL_A})')
    ax.plot(t_B, X_B[:, iX], color=_CRED,  ls='-',  lw=2.0, label=f'X  ({LABEL_B})')
    ax.set_title('Collective (C) and Displaced (X)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(t_A, X_A[:, iG], color=_CGRAY, ls='--', lw=1.8, label=f'G  ({LABEL_A})')
    ax.plot(t_B, X_B[:, iG], color=_CGRAY, ls='-',  lw=2.0, label=f'G  ({LABEL_B})')
    ax.set_title('Grievance (G)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(t_A, X_A[:, iRes],  color=_CORNG, ls='--', lw=1.8, label=f'Res   ({LABEL_A})')
    ax.plot(t_B, X_B[:, iRes],  color=_CORNG, ls='-',  lw=2.0, label=f'Res   ({LABEL_B})')
    ax.plot(t_A, X_A[:, iCRes], color=_CGRN,  ls='--', lw=1.8, label=f'C_Res ({LABEL_A})')
    ax.plot(t_B, X_B[:, iCRes], color=_CGRN,  ls='-',  lw=2.0, label=f'C_Res ({LABEL_B})')
    ax.set_title('Resources (Res) and Community Resilience (C_Res)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'Stochastic Timeseries  --  {LABEL_A}  (--) vs  {LABEL_B}  (-)',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# FIGURE B2b -- STOCHASTIC PHASE PORTRAITS  (2 panels: X vs C | Res vs C_Res)
# ===========================================================================

def plot_stoch_phase(X_A, X_B, save_path=None):
    cmap_A = plt.colormaps['Blues']
    cmap_B = plt.colormaps['Oranges']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    _gradient_line(ax, X_A[:, iX], X_A[:, iC], 'Blues',   lw=1.3, ls='--', alpha=0.75)
    _gradient_line(ax, X_B[:, iX], X_B[:, iC], 'Oranges', lw=1.9, ls='-',  alpha=0.88)
    ax.scatter(X_A[0,  iX], X_A[0,  iC], color='black',      s=80,  zorder=6, marker='o', label='Start (shared)')
    ax.scatter(X_A[-1, iX], X_A[-1, iC], color='steelblue',  s=90,  zorder=6, marker='X', label=f'End -- {LABEL_A}')
    ax.scatter(X_B[-1, iX], X_B[-1, iC], color='darkorange', s=110, zorder=6, marker='*', label=f'End -- {LABEL_B}')
    ax.set_xlabel('X  (Displaced)'); ax.set_ylabel('C  (Collective)')
    ax.set_title('Phase portrait: Displaced vs Collective', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    _add_colorbars(fig, ax, cmap_A, cmap_B)

    ax = axes[1]
    _gradient_line(ax, X_A[:, iRes], X_A[:, iCRes], 'Blues',   lw=1.3, ls='--', alpha=0.75)
    _gradient_line(ax, X_B[:, iRes], X_B[:, iCRes], 'Oranges', lw=1.9, ls='-',  alpha=0.88)
    ax.scatter(X_A[0,  iRes], X_A[0,  iCRes], color='black',      s=80,  zorder=6, marker='o', label='Start (shared)')
    ax.scatter(X_A[-1, iRes], X_A[-1, iCRes], color='steelblue',  s=90,  zorder=6, marker='X', label=f'End -- {LABEL_A}')
    ax.scatter(X_B[-1, iRes], X_B[-1, iCRes], color='darkorange', s=110, zorder=6, marker='*', label=f'End -- {LABEL_B}')
    ax.set_xlabel('Res  (Resources)'); ax.set_ylabel('C_Res  (Community Resilience)')
    ax.set_title('Phase portrait: Resources vs Community Resilience', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    _add_colorbars(fig, ax, cmap_A, cmap_B)

    fig.suptitle(
        f'Stochastic Phase Portraits  --  Blues (--): {LABEL_A}   |   Oranges (-): {LABEL_B}',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    print('=' * 65)
    print('Basic Band-Tribe Model')
    print(f'  {N_SPECIES} species, {N_REACTIONS} reactions  (r0-r10)')
    print(f'  Network: {_RN_FILE}')
    print('=' * 65)
    print(f"X0 = C={C0}, X={X0_IC}, Res={Res0}, C_Res={CRes0}, G={G0}")
    print()
    print(f'STOICH shape: {STOICH.shape}')
    print(f'INPUT_STOICH shape: {INPUT_STOICH.shape}')
    print(f'SPECIES:   {SPECIES}')
    print(f'REACTIONS: {REACTIONS}')
    print()
    print(f'Scenario A  alpha_react={ALPHA_REACT_A}, alpha_sol={ALPHA_SOL_A}'
          f'  ->  {LABEL_A}')
    print(f'Scenario B  alpha_react={ALPHA_REACT_B}, alpha_sol={ALPHA_SOL_B}'
          f'  ->  {LABEL_B}')
    print()

    # -- Part A: Deterministic ODE ---------------------------------------------
    print('--- Part A: ODE (scipy LSODA) ---')
    t_A, y_A = run_ode(ALPHA_REACT_A, ALPHA_SOL_A)
    t_B, y_B = run_ode(ALPHA_REACT_B, ALPHA_SOL_B)

    path = os.path.join(_OUT_DIR, 'figB1_ode_timeseries.png')
    plot_ode_timeseries(t_A, y_A, t_B, y_B, save_path=path)

    path = os.path.join(_OUT_DIR, 'figB1_ode_phase.png')
    plot_ode_phase(t_A, y_A, t_B, y_B, save_path=path)
    print('  Part A done.\n')

    # -- Part B: Stochastic Poisson tau-leaping --------------------------------
    print('--- Part B: Stochastic Poisson tau-leaping ---')
    XS_A = run_stochastic(ALPHA_REACT_A, ALPHA_SOL_A, seed=SEED_A)
    XS_B = run_stochastic(ALPHA_REACT_B, ALPHA_SOL_B, seed=SEED_B)

    path = os.path.join(_OUT_DIR, 'figB2_stoch_timeseries.png')
    plot_stoch_timeseries(XS_A, XS_B, save_path=path)

    path = os.path.join(_OUT_DIR, 'figB2_stoch_phase.png')
    plot_stoch_phase(XS_A, XS_B, save_path=path)
    print('  Part B done.\n')

    print(f'All outputs saved to: {_OUT_DIR}')
