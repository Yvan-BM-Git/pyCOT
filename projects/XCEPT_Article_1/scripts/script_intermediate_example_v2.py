#!/usr/bin/env python3
"""
script_intermediate_example_v2.py
==================================
Band-tribe-chiefdom conflict model  --  stochastic kinetics only.

Extends Basic_example.txt (r0-r10) with chiefdom reactions r11-r21.
The tribe-band baseline (r0-r10) is parameterised to match the STABLE
TRIBE-BAND case from script_basic_model.py (Scenario A: alpha_sol=2,
alpha_react=0.5 baked in).  The chiefdom emerges as a small perturbation
(H_INIT, HRES_INIT) on top of this harmonious baseline.

Species:   C, X, H, Res, C_Res, H_Res, G, P_H, F_H   (9 species)
Reactions: r0-r10  (band-tribe + conflict, shared with Basic_example.txt)
           r11     (chiefdom maintenance: H + H_Res => H)
           r12     (resource extraction: C + H + Res => C + H + H_Res)  [CHIEF]
           r13     (resilience extraction: H + C_Res => H + H_Res)      [CHIEF]
           r14     (hierarchy decay: H => C)
           r15     (hierarchy expansion: C + H + H_Res => 2H)           [CHIEF]
           r16     (militarize displaced: H + X + H_Res => H + P_H)     [CHIEF]
           r17     (police maintenance: P_H + H_Res => P_H)             [CHIEF]
           r18     (surplus redistribution: 2H_Res => H_Res + C_Res)    [CHIEF] KEY
           r19     (G suppression: P_H + G => P_H)                      [CHIEF]
           r20     (demobilize: P_H => C)                               [CHIEF]
           r21     (foreign aid: H + F_H => H + F_H + H_Res)
           F_H is a catalytic external input (never consumed by the network).

Two chiefdom strategies (controlled via allocation multipliers on [CHIEF] reactions):

  PROTECTION    -- moderate r12, minimal r13, high r18 + r20
      Chiefdom acts as redistribution layer: taxes Res (r12, not C_Res),
      converts H_Res surplus back to C_Res (r18 HIGH -- the KEY mechanism),
      and demobilises police quickly back to the collective (r20 HIGH).
      Community resilience is the PRIMARY tool against grievance and displacement;
      P_H plays a supporting role only.  Sustainable without aid.

  EXPLOITATION  -- heavy r12 + r13, strong r15 + r16 + r19, r18 + r20 suppressed
      Chiefdom fights scarcity with violence: dual extraction drains both Res
      (r12) and C_Res (r13 HIGH), hierarchy expands (r15), P_H is the PRIMARY
      grievance tool (r16+r19 HIGH).  r18 suppressed (no wealth returned to
      community).  r20 suppressed (police garrison permanent).
      FAILS without aid: r13 breaks C_Res -> r5 stops -> as C->0, r12 stops
      -> H_Res exhausted -> collapse to X-dominated state.
      With aid: H+P_H police-state sustained externally; community never recovers.

Each strategy is run in two foreign-aid conditions:
  NO AID  : F_H = 0 throughout  (r21 never fires)
  WITH AID: F_H = F_H_LEVEL unconditionally (AID_X/G_THRESHOLD = 0).
            External powers back the chiefdom authority regardless of
            community outcomes -- they support the government, not the crisis.
            r21 injects H_Res continuously.  Under PROTECTION, extra H_Res
            flows back to community via r18 (redistribution) -> C_Res grows
            higher, C population larger than no-aid case.  Under EXPLOITATION,
            extra H_Res fuels r15 expansion cycle -> H+police state sustained
            even as C collapses -- aid prolongs the parasitic phase.

Figures produced (tag = noaid / aid):
  figV2a_stocks_<tag>.png          --  stock timeseries (4 panels, F_H in panel 2)
  figV2b_mode_<tag>.png            --  mode projections (5 modes x 2 scenarios)
  figV2c_timescale_reint_<tag>.png --  timescale classification, reintegration
  figV2c_timescale_expl_<tag>.png  --  timescale classification, exploitation
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PYCOT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, os.path.join(_PYCOT_ROOT, 'src'))

from pyCOT.io.functions     import read_txt
from pyCOT.simulations.core import build_reaction_dict

# -- Output directory ----------------------------------------------------------
_OUT_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', 'outputs', 'intermediate_v2'))
os.makedirs(_OUT_DIR, exist_ok=True)

# ===========================================================================
# NETWORK + STOICHIOMETRY  (9 species, 21 reactions r0-r20)
# ===========================================================================
_RN_FILE = os.path.join(_SCRIPT_DIR, 'data', 'Intrmediate_example.txt')
_rn      = read_txt(_RN_FILE)
_sm      = _rn.stoichiometry_matrix()

SPECIES   = ['C', 'X', 'H', 'Res', 'C_Res', 'H_Res', 'G', 'P_H', 'F_H']
REACTIONS = [f'r{i}' for i in range(22)]   # r0 through r21

N_SPECIES   = len(SPECIES)    # 9
N_REACTIONS = len(REACTIONS)  # 22

_pycot_sp = list(_sm.species)
_pycot_rx = list(_sm.reactions)
_row_idx  = [_pycot_sp.index(s) for s in SPECIES]
_col_idx  = [_pycot_rx.index(r) for r in REACTIONS]
STOICH    = np.array(_sm, dtype=float)[np.ix_(_row_idx, _col_idx)]   # (9, 22)

_rxn_dict    = build_reaction_dict(_rn)
INPUT_STOICH = np.zeros((N_SPECIES, N_REACTIONS))
for j, rxn_name in enumerate(REACTIONS):
    reactants, _ = _rxn_dict[rxn_name]
    for sp, coef in reactants:
        if sp in SPECIES:
            INPUT_STOICH[SPECIES.index(sp), j] = coef

# Species index shortcuts
iC, iX, iH, iRes, iCRes, iHRes, iG, iPH, iFH = range(9)

# ===========================================================================
# USER-EDITABLE PARAMETERS
# ===========================================================================

# -- Initial conditions -------------------------------------------------------
# Tribe-band at harmony: matches Scenario A initial conditions from
# script_basic_model.py.  H_INIT / HRES_INIT seed the chiefdom as a small
# perturbation -- the key variable to explore.
C_INIT    = 500.0
X_INIT    =  50.0
RES_INIT  =  10.0
CRES_INIT =  50.0
G_INIT    =  30.0
H_INIT    =  50.0   # <<<  editable: initial chiefdom size (small perturbation)
HRES_INIT =  100.0   # <<<  editable: initial chiefdom resources
PH_INIT   =  10.0
FH_INIT   =  0.0   # foreign aid starts inactive; injected adaptively

#                  C         X         H       Res       C_Res     H_Res     G         P_H      F_H
X0 = np.array([C_INIT, X_INIT, H_INIT, RES_INIT, CRES_INIT, HRES_INIT, G_INIT, PH_INIT, FH_INIT])

# -- Stochastic simulation settings -------------------------------------------
N_STOCH_STEPS = 5000
DT_STOCH      = 1.0
SEED_PROT     = 42
SEED_EXPL     = 42

# -- Window sizes for mode and timescale analysis -----------------------------
WINDOW_SIZES = [10, 50, 200, 500]

# -- Multi-seed settings for Q2/Q3 statistical analysis ----------------------
N_SEEDS = 10                       # <<<  editable: number of independent seeds
SEEDS   = list(range(N_SEEDS))    # seeds 0 … N_SEEDS-1; change N_SEEDS to scale

# -- figV2c transparency parameters (alpha ∝ primary-species share of C+H+X) -
ALPHA_DOM_THRESHOLD = 0.5   # <<<  fraction at or above which -> full opacity
ALPHA_MIN           = 0.25  # <<<  alpha when species share = 0  (still visible)
ALPHA_MAX           = 1.0   # <<<  alpha when species share >= threshold

# ===========================================================================
# STOCHASTIC KINETICS
# ===========================================================================
#
# r0-r10  BASELINE  (Scenario A from script_basic_model.py, values baked in):
#   KAPPA_BASIC × alpha -- values taken directly from that script's KAPPA_STOCH:
#     r2 = 0.50 * ALPHA_SOL=2.0   = 1.00   (community solidarity / surplus growth)
#     r5 = 0.10 * ALPHA_SOL=2.0   = 0.20   (community solidarity / reintegration)
#     r8 = 0.05 * ALPHA_REACT=0.5 = 0.025  (conflict reactivity / grievance)
#     r9 = 0.10 * ALPHA_REACT=0.5 = 0.05   (conflict reactivity / displacement)
#
# r11-r20  CHIEFDOM  (budget constraints documented inline):
#
#   TWO EXTRACTION PATHWAYS (both [CHIEF]-controlled):
#     r12: C+H+Res => C+H+H_Res  -- taxes the shared resource pool (Res)
#          Enabling degree: min(C, H, Res).  Harms band-tribe indirectly (less Res).
#          Reintegration: A=1.5 -> eff = 1.5*0.05*min(C,H,Res) ~ 0.375/step at IC
#          Exploitation:  A=2.0 -> eff = 2.0*0.05*min(C,H,Res) ~ 0.50/step at IC
#
#     r13: H+C_Res => H+H_Res    -- taxes community resilience directly (C_Res)
#          Enabling degree: min(H, C_Res).  The sharper weapon: directly weakens
#          community's capacity for surplus (r2) and reintegration (r5).
#          Reintegration: A=0.3 -> eff = 0.3*0.05*min(H,C_Res) ~ 0.03/step at IC
#          Exploitation:  A=3.0 -> eff = 3.0*0.05*min(H,C_Res) ~ 0.30/step at IC
#
#   C_Res budget: r2 produces ~2 C_Res/step at IC.
#     reintegration: r13_eff=0.03 << 2 -> community resilience preserved ✓
#     exploitation:  r13_eff=0.30 -> 15% drain vs r2 production; as C_Res falls,
#                    r13 self-limits (min(H,C_Res) shrinks), but so does r2 -> collapse spiral
#
#   H_Res budget: inflow = r12 + r13; outflow = r11 + r15 + r16 + r17 + r18
#     r18 (redistribution) is second-order (pi=H_Res/2) -- self-regulating:
#       reintegration (A_r18=3): H_Res stabilises where r12+r13 = r16+r17+r18
#       exploitation  (A_r18=0.1): H_Res rises (inflow > tiny outflow) until r15 saturates
#
# r21  FOREIGN AID  (H + F_H => H + F_H + H_Res):
#   F_H is catalytic -- never consumed.  Controlled externally in run_stochastic.
#   KAPPA_AID sets potency; F_H level (and when it activates) set below.

KAPPA_AID = 0.05   # <<<  editable: foreign aid potency per unit min(H, F_H)

# ---------------------------------------------------------------------------
# KAPPA derivation: all values derived from simultaneous SS balance at the
# target IC (C=500, X=50, Res=50, C_Res=200, H_Res=100, G=50, P_H=100).
# Enabling degrees at IC: Res-bottleneck → 50 for r1,r2,r12; H→100 for r11,r14.
#
# Coherence principles applied:
#   (A) κ_r1 = κ_r11 = κ_r17 = κ_e = 0.01   (equal energy needs for all humans)
#   (B) κ_r6 = κ_r7 = 0.005                  (same physical decay for material)
#   (C) κ_r15 = 5 × κ_r14                    (A_r15_prot=0.2 gives r15=r14 balance)
#   (D) κ_r8 = κ_r9 + κ_r10                  (G steady state)
#   (E) κ_r20 ≈ κ_r4                          (demobilize ≈ displacement rate)
#
# Joint Res / C_Res / H_Res balance equations (protection mode, A_r18=5.0):
#   R18   = A_r18·κ_r18·50 = 5.0·0.003·50 = 0.75
#   H_Res: R12+R13 = r11+r17+r15+r16+R18 = 1.0+1.0+0.1+0.1+0.75 = 2.95
#   C_Res: r2+R18  = R13+r5+r7+r3  → 3.3+0.75 = R13+1.1+0.5+1.2  → R13 = 1.25
#          → κ_r13 = R13/(A_r13·H) = 1.25/100 = 0.0125
#          → R12 = 2.95-1.25 = 1.70 → κ_r12 = 1.70/(A_r12·Res) = 1.70/100 = 0.017
#   Res:   κ_r0 = r1+r2-r3+r6+R12 = 0.5+3.3-1.2+0.125+1.70 = 4.425 → κ_r0 = 4.5
# ---------------------------------------------------------------------------

KAPPA_STOCH = np.array([
    # -- band-tribe (Res/C_Res SS, updated for chiefdom r12 Res drain) ---------------
    2.5,    # r0  => Res              resource inflow  (κ_r0=4.5: raised from 2.9 to absorb r12 drain)
    0.01,   # r1  C+Res=>C            C energy consumption     (= κ_e, principle A)
    0.066,  # r2  C+Res+C_Res=>2C_Res solidarity               (SS derived: r2=3.3/step at IC)
    0.01,  # r3  C_Res=>Res          C_Res redistribution     (SS derived)
    0.001,  # r4  C=>X               spontaneous displacement  (r5 must dominate over r4 in protection)
    0.03,  # r5  X+C_Res=>C         reintegration            (SS derived: r5=1.1/step at IC)
    0.03,  # r6  2Res=>              Res decay (2nd order)    (= κ_r7, principle B)
    0.005,  # r7  2C_Res=>            C_Res decay (2nd order)  (= κ_r6, principle B)
    0.006,  # r8  X+C+Res=>G         grievance generation      (= κ_r9+κ_r10, principle D)
    0.004,  # r9  2G+C=>X            displacement from G       (SS derived)
    0.002,  # r10 2G=>               grievance decay (2nd ord) (SS derived)
    # -- chiefdom (joint Res/C_Res/H_Res balance with A_r12=2.0, A_r13=1.0, A_r18=5.0) -
    0.02,   # r11 H+H_Res=>H         H energy consumption      (= κ_e, principle A)
    0.017,  # r12 C+H+Res=>H_Res     Res extraction  [CHIEF]   (R12=1.70/step: H_Res inflow minus r13)
    0.0125, # r13 H+C_Res=>H_Res     C_Res extraction [CHIEF]  (R13=1.25/step: C_Res balance with r18)
    0.01,   # r14 H=>C               hierarchy decay           (principle C: κ_r15=5κ_r14; same overhead as r11 makes elite status fragile)
    0.03,  # r15 C+H+H_Res=>2H      expansion [CHIEF]         (= 5κ_r14, principle C)
    0.001,  # r16 H+X+H_Res=>P_H     militarize [CHIEF]        (low baseline; amplified in protection)
    0.02,   # r17 P_H+H_Res=>P_H     P_H energy consumption    (= κ_e, principle A)
    0.01,  # r18 2H_Res=>H_Res+C_Res surplus redistribution [CHIEF] KEY  (R18=0.75 at H_Res=100)
    0.003,  # r19 P_H+G=>P_H         G suppression [CHIEF]
    0.002,  # r20 P_H=>C             demobilize [CHIEF]        (≈ κ_r4, principle E)
    # -- foreign aid -----------------------------------------------------------------
    KAPPA_AID, # r21 H+F_H=>H+F_H+H_Res  F_H catalytic, set externally
])

# -- Chiefdom allocation vectors ----------------------------------------------
# Controlled reactions [CHIEF_CTRL_INDICES]:
#   r12 (Res extraction), r13 (C_Res extraction), r15 (expansion),
#   r16 (militarize), r17 (police upkeep), r18 (redistribution),
#   r19 (G suppression), r20 (demobilize)
# Index mapping: 0=r12, 1=r13, 2=r15, 3=r16, 4=r17, 5=r18, 6=r19, 7=r20
CHIEF_CTRL_INDICES = [12, 13, 15, 16, 17, 18, 19, 20]

# ─────────────────────────────────────────────────────────────────────────────
# PROTECTION STRATEGY  =  investment in the PROTECTION MODE
# Mode: 2*r12 + r16 + r18
#   Fuel (2*r12): double-weight resource extraction into H_Res (r13 NOT in mode).
#   Redistribution path (r18 KEY): H_Res -> C_Res, returns wealth to community.
#   Protective-force path (r16): X->P_H absorbs displaced (r19 NOT in mode).
# Strategy: amplify mode reactions; r13/r19 kept at baseline, r15 suppressed.
# r17 baseline (police upkeep needed for r16/r19 to work); r20 baseline (P_H->C).
# Expected outcome WITHOUT AID: C majority, H moderate, X low, C_Res sustained
#   (r18 dominates r13 drain; net C_Res slightly positive).
# Expected outcome WITH AID:    same or better -- extra H_Res redistributed via
#   r18 further boosts C_Res.
# Approx effective rates at IC (C=500, X=50, H=50, Res=50, C_Res=50, H_Res=100, P_H=50):
#   H stability: r15_prot=0.2*0.05*H=0.01*H = κ_r14*H=0.01*H → BALANCED ✓ (principle C: 5×)
#   H_Res: at H_Res<50, all reactions H_Res-bottlenecked; H_Res_ss = R12+R13 / (r11+r17+r15+r16+r18 coeff)
#          R12=0.85, R13=0.625, outflow coeff=0.02+0.02+0.01+0.002+0.025=0.077 → H_Res_ss≈19
#   With-aid H_Res_ss: R21=0.05*50=2.5 extra → (1.475+2.5)/0.077=52 → r18=5.0*0.01*26=1.3/step
#   C_Res with-aid: r2+r18=3.3+1.3=4.6 >> no-aid (r2+r18=3.3+0.5=3.8) → C_Res grows higher ✓
#   C with-aid: larger C_Res → stronger r5 reintegration → C grows faster and higher than no-aid ✓
A_CHIEF_PROTECT = np.array([8.0, 1.0, 5.0, 8.0, 1.0, 5.0, 2.0, 1.0])
# r12 x8.0: IN mode (×2 weight, fuel) -> double H_Res inflow from Res
# r13 x1.0: NOT in mode (baseline only) -> minimal C_Res drain; r18 compensates
# r15 x5.0: NOT in mode (expansion suppressed) -> H stays moderate
# r16 x8.0: IN mode (protective force) -> X militarized into P_H
# r17 x1.0: neutral (police upkeep infrastructure)
# r18 x5.0: IN mode (redistribution KEY) -> H_Res flows back to community as C_Res
# r19 x2.0: NOT in mode (baseline G suppression) -> P_H functions passively
# r20 x1.0: NOT in mode (demobilize baseline) -> some P_H returns to C

# ─────────────────────────────────────────────────────────────────────────────
# EXPLOITATION STRATEGY  =  investment in the EXPLOITATION MODE
# Mode: r12 + r13 + 2*r15  (r14 background, NOT in mode)
#   Fuel (r12+r13): extract from Res (r12) and C_Res (r13) into H_Res.
#   Expansion cycle (2*r15): C+H+H_Res->2H, double-weight; H_Res consumed,
#   C depleted, H grows; no C_Res ever produced.
#   r18 suppressed: no redistribution back to community.
# WHY IT FAILS WITHOUT AID:
#   r13 drains C_Res -> r5 (X+C_Res->C) weakens -> X accumulates.
#   r15 converts C to H -> C falls -> r12 (needs C) eventually slows.
#   H_Res inflow drops -> r15 exhausted -> r14 decays H->C but C_Res gone
#   -> no r2 -> X cannot return -> X-dominated collapse.
# WHY AID SUSTAINS BUT DOES NOT FIX: r21 replenishes H_Res -> r15 cycle keeps
#   running -> H sustained; but C_Res still broken -> community never recovers.
# Approx effective rates at IC (C=500, X=50, H=50, Res=50, C_Res=50, H_Res=100, P_H=50):
#   r12=2.5*0.017*50=2.125, r13=3.0*0.0125*50=1.875 (H_Res inflow=4.0)
#   r11=0.02*50=1.0, r17=0.02*50=1.0, r15=3.0*0.05*50=7.5 (H_Res drain=9.5 >> inflow!)
#   H_Res crashes in ~100/(9.5-4.0)=18 steps; H peaks 50+7.0*18≈176 in first 18 steps
#   C_Res crashes: r13 drain>>r2 gain → C_Res=0 by step ~60
#   After C_Res=0: r15 limited by H_Res_ss≈16 → r15=2.4/step, r14=0.01*H; H peaks ~256 at step ~195
#   r14=0.01 → H half-life=69 steps; H→0 by step ~540; arc fully visible within N=2000 ✓
A_CHIEF_EXPLOIT = np.array([3.0, 10.0, 8.0, 0.5, 1.0, 0.05, 5.0, 0.1])
# r12 x3.0: IN mode (fuel) -> heavy Res extraction
# r13 x10.0: IN mode (fuel) -> aggressive C_Res drain (community broken)
# r15 x8.0: IN mode (×2 weight, expansion cycle) -> C+H+H_Res->2H, H_Res consumed
# r16 x1.0: NOT in mode (baseline militarize) -> police maintained but not primary
# r17 x1.0: neutral (police upkeep infrastructure)
# r18 x0.05: NOT in mode (redistribution suppressed) -> H_Res never returned
# r19 x1.0: NOT in mode (baseline G suppression) -> P_H functions passively
# r20 x0.1: NOT in mode (garrison stays) -> P_H not returned to community

LABEL_PROT = 'Chiefdom protection   (invest in 2*r12+r16+r18 mode)'
LABEL_EXPL = 'Chiefdom exploitation (invest in r12+r13+2*r15 mode)'

# -- Adaptive foreign aid parameters ------------------------------------------
# F_H is set externally (not consumed by any reaction) in run_stochastic.
# When conditions are met, F_H = F_H_LEVEL, enabling r20 to fire.
# r20 rate per step = KAPPA_AID * min(H, F_H_LEVEL) -- proportional to H.
F_H_LEVEL       = 100.0  # <<<  editable: F_H injected when aid conditions met
AID_X_THRESHOLD = 50.0    # <<<  editable: 0 = unconditional aid (always active when use_aid=True)
AID_G_THRESHOLD = 10.0    # <<<  editable: 0 = unconditional aid
AID_DECAY       = 0.0    # <<<  editable: F_H withdrawal per step when below threshold

# ===========================================================================
# STOCHASTIC SIMULATION  (Poisson tau-leaping)
# ===========================================================================

def _full_alloc(a_chief):
    """Build 21-element allocation vector; non-chiefdom reactions get a=1."""
    a = np.ones(N_REACTIONS)
    for i, j in enumerate(CHIEF_CTRL_INDICES):
        a[j] = a_chief[i]
    return a


def _enabling_degrees(x):
    pi = np.ones(N_REACTIONS)
    for j in range(N_REACTIONS):
        col    = INPUT_STOICH[:, j]
        active = col > 0
        if active.any():
            pi[j] = np.min(x[active] / col[active])
    return np.clip(pi, 0.0, None)


def _resolve_contention(x, n, max_iter=20):
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


def _step_poisson(x, a, dt, rng):
    pi  = _enabling_degrees(x)
    lam = a * KAPPA_STOCH * pi
    n   = rng.poisson(lam * dt).astype(float)
    n   = np.minimum(n, np.floor(pi))
    n   = _resolve_contention(x, n)
    return np.maximum(x + STOICH @ n, 0.0), n


def run_stochastic(a_chief, seed=42, use_aid=False):
    """
    Poisson tau-leaping simulation.

    use_aid: if True, F_H is injected adaptively:
               F_H = F_H_LEVEL  when  x[iX] >= AID_X_THRESHOLD  or  x[iG] >= AID_G_THRESHOLD
               F_H decreases by AID_DECAY otherwise (0 = hold level until condition clears)
             F_H is catalytic (STOICH[iFH,:]=0), so it is never consumed by the network.
             The F_H value used at each step is recorded in X[t, iFH].

    Returns X[n+1, 9]  (stocks) and N[n, 21]  (firing counts per step).
    """
    a   = _full_alloc(a_chief)
    rng = np.random.default_rng(seed)
    X   = np.zeros((N_STOCH_STEPS + 1, N_SPECIES))
    N   = np.zeros((N_STOCH_STEPS,     N_REACTIONS))
    X[0] = X0.copy()
    for t in range(N_STOCH_STEPS):
        x = X[t].copy()
        # -- Adaptive foreign aid injection ------------------------------------
        if use_aid:
            if x[iX] >= AID_X_THRESHOLD or x[iG] >= AID_G_THRESHOLD:
                x[iFH] = F_H_LEVEL
            else:
                x[iFH] = max(0.0, x[iFH] - AID_DECAY)
        else:
            x[iFH] = 0.0
        X[t, iFH] = x[iFH]   # record the F_H level used at this step
        X[t + 1], N[t] = _step_poisson(x, a, DT_STOCH, rng)
        # F_H is catalytic so X[t+1][iFH] = x[iFH] after STOICH @ n;
        # no correction needed.
    return X, N


# ===========================================================================
# COLOUR PALETTE
# ===========================================================================
_CRED    = '#C0392B'   # X, grievance, exploitation
_CBLUE   = '#2980B9'   # C (collective)
_CGRN    = '#27AE60'   # C_Res (community resilience)
_CORNG   = '#F39C12'   # Res (resources)
_CGRAY   = '#7F8C8D'   # G (grievance)
_CPURP   = '#8E44AD'   # P_H (military/police)
_CTEAL   = '#1ABC9C'   # H (chiefdom)
_CBROWN  = '#795548'   # H_Res (chiefdom resources)
_CAID    = '#1ABC9C'   # F_H (foreign aid) -- distinct dashed teal

_WINDOW_COLORS = {
    10:  '#E74C3C',
    50:  '#F39C12',
    200: '#27AE60',
    500: '#2980B9',
}

# ===========================================================================
# FONT SIZES  (edit here to rescale all figure text uniformly)
# ===========================================================================
FONT_SIZES = {
    'suptitle':   10,   # overall figure title
    'title':      12,   # subplot / panel title
    'axis_label':  12,   # x / y axis labels
    'tick_label':  12,   # tick mark labels
    'legend':      12,   # legend entries
    'annotation':  12,   # bar / scatter text annotations
    'small':       12,   # secondary labels (point counts etc.)
}

# ===========================================================================
# FIGURE V2a -- STOCK TIMESERIES  (4-panel comparison)
# ===========================================================================

def plot_stocks(X_prot, X_expl, title_tag='', save_path=None):
    """4-panel: C/X | H/P_H/F_H | G | Res/C_Res/H_Res."""
    n = X_prot.shape[0]
    t = np.arange(n)
    FS = FONT_SIZES

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes.flatten()

    panels = [
        (ax[0], 'Collective (C) and Displaced (X)', [
            (X_prot[:, iC],  _CBLUE, '-',  2.0, 'C  (protection)'),
            (X_expl[:, iC],  _CBLUE, '--', 1.5, 'C  (exploitation)'),
            (X_prot[:, iX],  _CRED,  '-',  2.0, 'X  (protection)'),
            (X_expl[:, iX],  _CRED,  '--', 1.5, 'X  (exploitation)'),
        ]),
        (ax[1], 'Chiefdom (H), Police (P_H), Foreign Aid (F_H)', [
            (X_prot[:, iH],  _CTEAL, '-',  2.0, 'H  (protection)'),
            (X_expl[:, iH],  _CTEAL, '--', 1.5, 'H  (exploitation)'),
            (X_prot[:, iPH], _CPURP, '-',  2.0, 'P_H  (protection)'),
            (X_expl[:, iPH], _CPURP, '--', 1.5, 'P_H  (exploitation)'),
            (X_prot[:, iFH], _CAID,  '-',  1.5, 'F_H  (protection)'),
            (X_expl[:, iFH], _CAID,  '--', 1.0, 'F_H  (exploitation)'),
        ]),
        (ax[2], 'Grievance (G)', [
            (X_prot[:, iG], _CGRAY, '-',  2.0, 'G  (protection)'),
            (X_expl[:, iG], _CGRAY, '--', 1.5, 'G  (exploitation)'),
        ]),
        (ax[3], 'Resources: Res, C_Res, H_Res', [
            (X_prot[:, iRes],  _CORNG,  '-',  2.0, 'Res  (protection)'),
            (X_expl[:, iRes],  _CORNG,  '--', 1.5, 'Res  (exploitation)'),
            (X_prot[:, iCRes], _CGRN,   '-',  2.0, 'C_Res  (protection)'),
            (X_expl[:, iCRes], _CGRN,   '--', 1.5, 'C_Res  (exploitation)'),
            (X_prot[:, iHRes], _CBROWN, '-',  2.0, 'H_Res  (protection)'),
            (X_expl[:, iHRes], _CBROWN, '--', 1.5, 'H_Res  (exploitation)'),
        ]),
    ]
    for a, title, lines in panels:
        for y, col, ls, lw, lbl in lines:
            a.plot(t, y, color=col, ls=ls, lw=lw, label=lbl, alpha=0.9)
        a.set_title(title, fontsize=FS['title'], fontweight='bold')
        a.set_xlabel('Step', fontsize=FS['axis_label'])
        a.set_ylabel('Level', fontsize=FS['axis_label'])
        a.tick_params(labelsize=FS['tick_label'])
        a.legend(fontsize=FS['legend'])
        a.grid(True, alpha=0.25)

    fig.suptitle(
        f'Stock Dynamics -- Protection (-) vs Exploitation (--)\n{title_tag}',
        fontsize=FS['suptitle'], fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# MODE ANALYSIS
# ===========================================================================
# Five canonical modes (process-vector directions):
#
#   Community recovery:  r0 + r2 + r5
#     The self-reinforcing band-tribe loop: resource renewal -> C_Res surplus
#     -> reintegration of displaced people.
#
#   Conflict amplification:  r4 + 2*r8 + r9
#     The destabilising spiral: C->X, X+C+Res->G, 2G+C->X.
#     r8 weighted x2: grievance generation drives both accumulation and r9.
#
#   Chiefdom protection:  2*r12 + r16 + r18
#     FUEL: 2*r12 (taxes Res, double weight) -> H_Res accumulates.
#           r13 is NOT in this mode; protection does not drain C_Res.
#     PATH A (redistribution): r18 (2H_Res -> H_Res + C_Res) returns wealth.
#     PATH B (protective force): r16 (X+H_Res -> P_H) absorbs displaced.
#     Net: Res -2, X -1, H_Res balanced, P_H +1, C_Res +1.
#     r19 is NOT in this mode; suppression is background, not the mechanism.
#
#   Chiefdom exploitation:  r12 + r13 + 2*r15
#     FUEL: r12 (Res -> H_Res) + r13 (C_Res -> H_Res).
#     EXPANSION CYCLE (2*r15): C+H+H_Res->2H (double weight); H_Res consumed,
#     C depleted, H grows; no C_Res ever produced.
#     r14 is NOT in this mode (background hierarchy decay, uncontrolled).
#     Net: Res -1, C_Res -1, C -2, H +2, H_Res balanced.
#     r18 suppressed: no redistribution; community slowly starved.
#
#   Foreign aid:  r21
#     Catalytic H_Res injection.  Under PROTECTION: r18 redistribution route
#     converts extra H_Res to C_Res -> community strengthened further.
#     Under EXPLOITATION: r15 expansion cycle consumes extra H_Res -> H+police
#     state sustained externally but community never recovers.
#
# Projection (2-D scatter per mode):
#   v_avg = sum(firings in block) / w   (per-step average, scale-independent)
#   d_hat = d / |d|                     (normalised canonical direction)
#   proj_mag  = d_hat . v_avg           (scalar projection, magnitude of alignment)
#   cos_sim   = proj_mag / |v_avg|      (cosine similarity, 0..1 for non-neg v)
#   Scatter: x=proj_mag, y=cos_sim, colour=window size (timescale)
#   Interpretation: high proj_mag + high cos_sim => mode fires strongly AND
#   dominates the overall process; high mag + low cos => mode active but
#   many other reactions also firing simultaneously.

_d_comm  = np.zeros(N_REACTIONS)
_d_comm[[0, 2, 5]] = 1.0

_d_conf  = np.zeros(N_REACTIONS)
_d_conf[[4, 9]] = 1.0; _d_conf[8] = 2.0

_d_prot  = np.zeros(N_REACTIONS)
_d_prot[12] = 2.0; _d_prot[[16, 18]] = 1.0   # 2*r12 + r16 + r18

_d_expl  = np.zeros(N_REACTIONS)
_d_expl[[12, 13]] = 1.0; _d_expl[15] = 2.0   # r12 + r13 + 2*r15

_d_aid   = np.zeros(N_REACTIONS)
_d_aid[21] = 1.0

MODES = {
    'Community recovery  (r0+r2+r5)':                       _d_comm,
    'Conflict amplification  (r4+2*r8+r9)':                 _d_conf,
    'Chiefdom protection  (2*r12+r16+r18)':                  _d_prot,
    'Chiefdom exploitation  (r12+r13+2*r15)':               _d_expl,
    'Foreign aid  (r21)':                                    _d_aid,
}

# Pre-compute normalised mode vectors
MODES_HAT = {name: d / np.linalg.norm(d) for name, d in MODES.items()}


def plot_mode_projections(scenarios, save_path=None):
    """
    2-D scatter per mode: scalar projection (x) vs cosine similarity (y).
    Each point = one w-step non-overlapping block; v_avg = sum/w.
    Colour = window size.
    scenarios: list of (list_of_N_arrs, label) — 4 scenarios, all seeds overlaid.

    r21 (foreign aid, column index 21) is excluded from every process vector so
    that all scenarios are projected in the same 21-dimensional subspace.
    The Foreign aid mode is omitted from the plot.

    Per-mode shared x-axis: xlim = mu ± 3 × max_sigma across scenarios.
    Points outside this range are dropped before plotting.
    """
    FS         = FONT_SIZES
    # Exclude Foreign aid mode from display
    plot_modes = {k: v for k, v in MODES.items() if 'Foreign aid' not in k}
    n_modes    = len(plot_modes)
    n_scen     = len(scenarios)
    n_seeds    = len(scenarios[0][0])

    # ── Pass 1: collect all (w, proj, cos) per (mode, scenario) ─────────────
    # Process vectors truncated to first 21 reactions (exclude r21).
    all_data = {}   # (row, col) -> list of (w, proj, cos)
    d_hats   = {}   # row -> d_hat_21d

    for row, (mode_name, _) in enumerate(plot_modes.items()):
        dh = MODES_HAT[mode_name][:21].copy()
        norm = np.linalg.norm(dh)
        dh   = dh / norm if norm > 1e-12 else dh   # renormalise in 21D
        d_hats[row] = dh

        for col, (N_arrs_list, _) in enumerate(scenarios):
            points = []
            for N_arr in N_arrs_list:
                n_steps = N_arr.shape[0]
                for w in WINDOW_SIZES:
                    n_blocks = n_steps // w
                    for b in range(n_blocks):
                        v_blk  = N_arr[b*w:(b+1)*w, :21].sum(axis=0) / w
                        proj   = float(np.dot(dh, v_blk))
                        v_norm = float(np.linalg.norm(v_blk))
                        cos_s  = (proj / v_norm) if v_norm > 1e-12 else 0.0
                        points.append((w, proj, cos_s))
            all_data[(row, col)] = points

    # ── Per-mode x-scale: mu ± 3 × max_sigma, shared across all 4 scenarios ─
    mode_xlims = {}
    for row in range(n_modes):
        all_projs = np.array([p for col in range(n_scen)
                               for _, p, _ in all_data[(row, col)]])
        mu     = float(np.mean(all_projs))
        sigmas = [float(np.std([p for _, p, _ in all_data[(row, col)]]))
                  for col in range(n_scen)]
        sig    = max(sigmas)
        if row == 0:
            mode_xlims[row] = (0,1.2)
        elif row == 1:
            mode_xlims[row] = (0,0.5)
        elif row == 2:
            mode_xlims[row] = (0,1.5)
        elif row == 3:
            mode_xlims[row] = (0,5.2)

    # ── Pass 2: plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_modes, n_scen,
                             figsize=(6 * n_scen, 3.5 * n_modes),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.25)

    for row, (mode_name, _) in enumerate(plot_modes.items()):
        x_lo, x_hi = mode_xlims[row]
        for col, (N_arrs_list, label) in enumerate(scenarios):
            ax = axes[row, col]
            for w in WINDOW_SIZES:
                pts = [(p, c) for ww, p, c in all_data[(row, col)] if ww == w]
                if not pts:
                    continue
                projs = np.array([p for p, _ in pts])
                coss  = np.array([c for _, c in pts])
                keep  = (projs >= x_lo) & (projs <= x_hi)
                ax.scatter(projs[keep], coss[keep],
                           color=_WINDOW_COLORS[w], s=4, alpha=0.25,
                           label=f'w={w}', rasterized=True)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{mode_name}\n{label}',
                         fontsize=FS['title'], fontweight='bold')
            ax.set_xlabel('Projection magnitude  (d̂·v_avg)',
                          fontsize=FS['axis_label'])
            ax.set_ylabel('Cosine similarity', fontsize=FS['axis_label'])
            ax.tick_params(labelsize=FS['tick_label'])
            ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.4)
            ax.grid(True, alpha=0.25)

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=_WINDOW_COLORS[w], markersize=7,
                      label=f'w={w}')
               for w in WINDOW_SIZES]
    fig.legend(handles=handles, loc='lower center', ncol=len(WINDOW_SIZES),
               fontsize=FS['legend'], frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Mode projections: projection magnitude vs cosine similarity  ({n_seeds} seeds)\n'
        'r21 (foreign aid) excluded from process vector  ·  x-axis: shared per mode, clipped at μ ± 3σ  ·  colour = window size',
        fontsize=FS['suptitle'])
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# MODE DOMINANCE TIMESERIES  (figV2f_dominance)
# ===========================================================================
# For each (scenario × window_size) panel:
#   x = time step, y = cosine similarity of process vector with canonical mode.
#   Opacity (alpha) ∝ absolute projection magnitude |d̂·v_avg|, capped at 95th
#   percentile across all modes/scenarios/windows to suppress outliers.
#   Transparent = mode directionally aligned but negligible activity.
#   Opaque      = mode dominates both direction AND magnitude of process activity.
# All process vectors exclude r21 (foreign aid) so scenarios are comparable.
# Mean cos-similarity and mean |projection| averaged across all seeds.

_MODE_COLORS_DOM = [
    '#27AE60',   # Community recovery      (green)
    '#C0392B',   # Conflict amplification  (red)
    '#2980B9',   # Chiefdom protection     (blue)
    '#F39C12',   # Chiefdom exploitation   (orange)
]


def plot_mode_dominance_timeseries(scenarios, save_path=None):
    """
    4 scenarios × 4 window sizes grid.
    Each panel: 4 canonical modes as scatter (x=time, y=cosine similarity);
    opacity ∝ |projection magnitude| clipped at global 95th percentile.
    r21 excluded; mean across seeds.
    """
    FS        = FONT_SIZES
    plot_modes = {k: v for k, v in MODES.items() if 'Foreign aid' not in k}
    mode_names = list(plot_modes.keys())
    n_modes    = len(mode_names)   # 4
    n_scen     = len(scenarios)    # 4
    n_w        = len(WINDOW_SIZES) # 4
    n_seeds    = len(scenarios[0][0])
    n_steps    = N_STOCH_STEPS

    # Normalised mode direction vectors in 21D (exclude r21)
    d_hats_21d = np.zeros((n_modes, 21))
    for mi, mn in enumerate(mode_names):
        dh   = MODES_HAT[mn][:21].copy()
        norm = np.linalg.norm(dh)
        d_hats_21d[mi] = dh / norm if norm > 1e-12 else dh

    # ── Pass 1: mean cosine similarity and mean |projection| ─────────────────
    all_cos  = {}   # (sci, wi, mi) -> (n_times,)
    all_proj = {}   # (sci, wi, mi) -> (n_times,)

    for sci, (N_arrs_list, _) in enumerate(scenarios):
        for wi, w in enumerate(WINDOW_SIZES):
            n_times = n_steps - w + 1
            cos_sum  = np.zeros((n_modes, n_times))
            proj_sum = np.zeros((n_modes, n_times))

            for N_arr in N_arrs_list:
                N21    = N_arr[:, :21].astype(float)
                csum   = np.vstack([np.zeros((1, 21)), np.cumsum(N21, axis=0)])
                v_cums = (csum[w:] - csum[:n_times]) / w          # (n_times, 21)
                projs  = v_cums @ d_hats_21d.T                    # (n_times, n_modes)
                v_norms = np.linalg.norm(v_cums, axis=1, keepdims=True)
                cos_s  = np.where(v_norms > 1e-12, projs / v_norms, 0.0)
                cos_sum  += cos_s.T        # (n_modes, n_times)
                proj_sum += np.abs(projs).T

            for mi in range(n_modes):
                all_cos [(sci, wi, mi)] = cos_sum [mi] / n_seeds
                all_proj[(sci, wi, mi)] = proj_sum[mi] / n_seeds

    # Global 95th-percentile cap on |projection| magnitude
    all_flat = np.concatenate(list(all_proj.values()))
    pos_vals = all_flat[all_flat > 0]
    proj_cap = float(np.percentile(pos_vals, 95)) if len(pos_vals) > 0 else 1.0

    # ── Pass 2: plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_scen, n_w,
                             figsize=(5 * n_w, 3.5 * n_scen),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    for sci, (_, label) in enumerate(scenarios):
        for wi, w in enumerate(WINDOW_SIZES):
            ax    = axes[sci, wi]
            times = np.arange(w - 1, n_steps)   # length = n_times

            for mi, (mn, mcol) in enumerate(zip(mode_names, _MODE_COLORS_DOM)):
                cos_mean  = all_cos [(sci, wi, mi)]
                proj_mean = all_proj[(sci, wi, mi)]

                alphas   = np.clip(proj_mean / proj_cap, 0.03, 1.0)
                base_rgba = np.array(matplotlib.colors.to_rgba(mcol))
                rgba_arr  = np.tile(base_rgba, (len(times), 1))
                rgba_arr[:, 3] = alphas

                ax.scatter(times, cos_mean, c=rgba_arr,
                           s=2, rasterized=True, linewidths=0)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, n_steps)
            ax.set_title(f'{label}\nw = {w}', fontsize=FS['title'])
            ax.set_xlabel('Step', fontsize=FS['axis_label'])
            ax.set_ylabel('Cosine similarity', fontsize=FS['axis_label'])
            ax.tick_params(labelsize=FS['tick_label'])
            ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.3)
            ax.grid(True, alpha=0.20)

    # Shared legend
    mode_short = [mn.split('  ')[0] for mn in mode_names]
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_MODE_COLORS_DOM[mi], markersize=8,
               label=mode_short[mi])
        for mi in range(n_modes)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=n_modes,
               fontsize=FS['legend'], frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Mode dominance over time  ({n_seeds} seeds, mean)  ·  r21 excluded\n'
        'y = cosine similarity of process vector with mode  ·  '
        f'opacity ∝ |projection magnitude|  (cap ≈ {proj_cap:.2f}  =  95th pct)\n'
        'Opaque = mode dominant in size and direction  ·  transparent = mode weak',
        fontsize=FS['suptitle'])
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# TIMESCALE MODE CLASSIFICATION
# ===========================================================================
# Three species groups:
#   MP: {C, C_Res}   community production
#   MH: {H, H_Res}   chiefdom production
#   MC: {X, G}       conflict indicators

MP_IDX = np.array([iC,  iCRes])
MH_IDX = np.array([iH,  iHRes])
MC_IDX = np.array([iX,  iG])

_MODE_LABELS = ['problem', 'challenge', 'overproduction', 'steady-state']
_MODE_CODE   = {v: i for i, v in enumerate(_MODE_LABELS)}
_MODE_COLORS = ['#C0392B', '#F39C12', '#27AE60', '#2980B9']
_MODE_CMAP   = matplotlib.colors.ListedColormap(_MODE_COLORS)
_MODE_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5]
_MODE_NORM   = matplotlib.colors.BoundaryNorm(_MODE_BOUNDS, _MODE_CMAP.N)
_MODE_RGB    = np.array([matplotlib.colors.to_rgb(c) for c in _MODE_COLORS])  # (4, 3)
_GRAY_RGB    = np.array(matplotlib.colors.to_rgb('#DCDCDC'))


def _classify(dx_sub, tol=0.05):
    has_pos = np.any(dx_sub >  tol)
    has_neg = np.any(dx_sub < -tol)
    if has_pos and has_neg:
        return 'challenge'
    if has_pos:
        return 'overproduction'
    if has_neg:
        return 'problem'
    return 'steady-state'


def compute_timescale_modes(N_arr):
    """
    Causal sliding-window classification for each window in WINDOW_SIZES.
    Returns (n_w, n_steps) grids for MP, MH, MC groups (NaN where history < w).
    """
    n_steps = N_arr.shape[0]
    n_w     = len(WINDOW_SIZES)
    grid_MP = np.full((n_w, n_steps), np.nan)
    grid_MH = np.full((n_w, n_steps), np.nan)
    grid_MC = np.full((n_w, n_steps), np.nan)

    for wi, w in enumerate(WINDOW_SIZES):
        for t in range(w - 1, n_steps):
            v_cum = N_arr[t - w + 1 : t + 1].sum(axis=0)
            dx    = STOICH @ v_cum/w
            grid_MP[wi, t] = float(_MODE_CODE[_classify(dx[MP_IDX])])
            grid_MH[wi, t] = float(_MODE_CODE[_classify(dx[MH_IDX])])
            grid_MC[wi, t] = float(_MODE_CODE[_classify(dx[MC_IDX])])

    return grid_MP, grid_MH, grid_MC


def plot_timescale_modes(N_arrs, scenario_label, X_arrs=None, save_path=None):
    """
    3-panel raster: {C,C_Res} | {H,H_Res} | {X,G} across window sizes.
    N_arrs: list of N_arr (one per seed).  Each window-size band is subdivided
    into len(N_arrs) thin rows (one per seed), stacked vertically.

    X_arrs: optional list of X_arr (one per seed, shape N_STOCH_STEPS+1 × N_SPECIES).
    When supplied, each cell's opacity reflects the primary species' share of
    the total human population (C + H + X):
      alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * clip(share / ALPHA_DOM_THRESHOLD, 0, 1)
    Configurable via ALPHA_DOM_THRESHOLD, ALPHA_MIN, ALPHA_MAX at the top of the script.
    """
    FS      = FONT_SIZES
    n_seeds = len(N_arrs)
    n_steps = N_arrs[0].shape[0]
    n_w     = len(WINDOW_SIZES)
    n_rows  = n_w * n_seeds

    # ── Classification grids (n_rows × n_steps) ───────────────────────────────
    grid_MP = np.full((n_rows, n_steps), np.nan)
    grid_MH = np.full((n_rows, n_steps), np.nan)
    grid_MC = np.full((n_rows, n_steps), np.nan)

    for si, N_arr in enumerate(N_arrs):
        gMP, gMH, gMC = compute_timescale_modes(N_arr)
        for wi in range(n_w):
            r = wi * n_seeds + si
            grid_MP[r, :] = gMP[wi, :]
            grid_MH[r, :] = gMH[wi, :]
            grid_MC[r, :] = gMC[wi, :]

    # ── Per-cell alpha grids (n_rows × n_steps) ───────────────────────────────
    # Primary species for each panel: C, H, X respectively.
    def _build_alpha(sp_idx):
        ag = np.full((n_rows, n_steps), ALPHA_MAX)
        if X_arrs is None:
            return ag
        for si, X_arr in enumerate(X_arrs):
            st    = X_arr[:n_steps]                              # (n_steps, N_SPECIES)
            total = st[:, iC] + st[:, iH] + st[:, iX]
            share = np.where(total > 1e-6, st[:, sp_idx] / total, 0.0)
            alpha_t = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.clip(
                share / ALPHA_DOM_THRESHOLD, 0.0, 1.0)
            for wi in range(n_w):
                ag[wi * n_seeds + si, :] = alpha_t
        return ag

    alpha_MP = _build_alpha(iC)
    alpha_MH = _build_alpha(iH)
    alpha_MC = _build_alpha(iX)

    # ── Convert classification + alpha to RGBA image ──────────────────────────
    def _to_rgba(grid, alpha_grid):
        nan_mask = np.isnan(grid)
        cls      = np.where(nan_mask, 0, grid).astype(int)
        rgba     = np.ones((*grid.shape, 4), dtype=float)
        rgba[:, :, :3] = _MODE_RGB[cls]
        rgba[:, :, 3]  = np.where(nan_mask, 1.0, alpha_grid)
        rgba[nan_mask, :3] = _GRAY_RGB   # insufficient history → gray, fully opaque
        return rgba

    group_labels = [
        'Community group  {C, C_Res}',
        'Chiefdom group   {H, H_Res}',
        'Conflict group   {X, G}',
    ]
    grids  = [grid_MP,  grid_MH,  grid_MC]
    alphas = [alpha_MP, alpha_MH, alpha_MC]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.30)

    extent = [-0.5, n_steps - 0.5, -0.5, n_rows - 0.5]

    for ax, grid, ag, glabel in zip(axes, grids, alphas, group_labels):
        rgba = _to_rgba(grid, ag)
        ax.imshow(rgba, aspect='auto', origin='lower',
                  extent=extent, interpolation='nearest')

        group_centers = [wi * n_seeds + (n_seeds - 1) / 2 for wi in range(n_w)]
        ax.set_yticks(group_centers)
        ax.set_yticklabels([f'w={w}' for w in WINDOW_SIZES],
                           fontsize=FS['tick_label'])
        ax.set_ylabel('Window size w', fontsize=FS['axis_label'])
        ax.set_title(glabel, fontsize=FS['title'], fontweight='bold')
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.grid(False)

        for wi in range(1, n_w):
            ax.axhline(wi * n_seeds - 0.5, color='white', lw=1.5)

        for wi, w in enumerate(WINDOW_SIZES):
            n_valid = max(0, n_steps - w + 1)
            ax.text(n_steps + 2, group_centers[wi],
                    f'{n_valid}×{n_seeds}',
                    va='center', ha='left',
                    fontsize=FS['small'], color='#555555')

    axes[-1].set_xlabel('Time step', fontsize=FS['axis_label'])
    axes[-1].tick_params(labelsize=FS['tick_label'])

    patches = [
        mpatches.Patch(color=_MODE_COLORS[0], label='problem (net negative)'),
        mpatches.Patch(color=_MODE_COLORS[1], label='challenge (mixed)'),
        mpatches.Patch(color=_MODE_COLORS[2], label='overproduction (net positive)'),
        mpatches.Patch(color=_MODE_COLORS[3], label='steady-state (near zero)'),
        mpatches.Patch(color='#DCDCDC',       label='insufficient history (< w steps)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=FS['legend'], frameon=True, bbox_to_anchor=(0.5, -0.02))

    alpha_note = (f'  ·  opacity ∝ species share of C+H+X  '
                  f'[{ALPHA_MIN:.2f}–{ALPHA_MAX:.2f}],  full at ≥{ALPHA_DOM_THRESHOLD:.0%}') \
                  if X_arrs is not None else ''
    fig.suptitle(
        f'Timescale-dependent process-mode classification — {scenario_label}\n'
        f'Causal sliding window: dx = S·v_cum(t)  ·  {n_seeds} seeds stacked per window band'
        f'  ·  grey = insufficient history{alpha_note}',
        fontsize=FS['suptitle'])
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# Q2 -- H_Res ROUTING FRACTION  (figV2b_routing + figV2d_routing)
# ===========================================================================
# Routing is computed directly from H_Res-consuming reactions via INPUT_STOICH,
# making it valid for ALL four scenarios (no-aid and aid alike).
#
# Community-benefit outflow (numerator):
#   r16  coef=1  (H+X+H_Res -> P_H, protective militarize)
#   r18  coef=2  (2H_Res -> H_Res+C_Res, redistribution)
#
# Total H_Res outflow (denominator):
#   r11(1) + r15(1) + r16(1) + r17(1) + r18(2)
#
# Routing fraction = community-benefit / total  ∈ [0, 1]
#   1 = all H_Res goes to redistribution/peacekeeping
#   0 = all H_Res goes to elite expansion / maintenance
#
#   figV2b_routing  (plot_hres_routing_summary)
#     4-bar summary chart: mean routing fraction (last 50 %, w=200) ± 1 SD
#     across N_SEEDS seeds.  Strategy colour × aid hatch pattern.
#
#   figV2d_routing  (plot_routing_timescale_multi)
#     1×4 panels, one per window size.
#     Mean ± 1 SD band per scenario (4 scenarios).
#     Short windows: bands overlap.  Long windows: protection clearly above exploitation.
#     Annotated at w=200: "strategies first separate".

_HRES_COMM_RXNS  = {16: 1, 18: 2}          # community-benefit H_Res consumers
_HRES_TOTAL_RXNS = {11: 1, 15: 1, 16: 1, 17: 1, 18: 2}  # all H_Res consumers


def compute_hres_routing(N_arr):
    """
    Causal sliding-window H_Res community-benefit routing fraction.
    frac = (r16×1 + r18×2) / (r11×1 + r15×1 + r16×1 + r17×1 + r18×2)
    Returns {w: (time_indices, fraction_array)} with NaN where total outflow = 0.
    """
    n_steps = N_arr.shape[0]
    result  = {}
    for w in WINDOW_SIZES:
        times, fracs = [], []
        for t in range(w - 1, n_steps):
            v_cum  = N_arr[t - w + 1 : t + 1].sum(axis=0)
            comm   = float(sum(v_cum[j] * c for j, c in _HRES_COMM_RXNS.items()))
            total  = float(sum(v_cum[j] * c for j, c in _HRES_TOTAL_RXNS.items()))
            fracs.append(comm / total if total > 1e-6 else np.nan)
            times.append(t)
        result[w] = (np.array(times), np.array(fracs))
    return result


def compute_routing_stats(N_arrs):
    """
    Aggregate routing fraction across multiple N_arr seeds.
    Returns {w: {'times', 'mean', 'std', 'tail_mean', 'tail_std'}}
    where 'tail' = last 50 % of simulation steps (per-seed mean, then mean/std across seeds).
    """
    all_routing = [compute_hres_routing(N) for N in N_arrs]
    half        = N_STOCH_STEPS // 2
    result      = {}
    for w in WINDOW_SIZES:
        ts_arr        = all_routing[0][w][0]
        fracs         = np.array([r[w][1] for r in all_routing])   # (n_seeds, n_times)
        mean_arr      = np.nanmean(fracs, axis=0)
        std_arr       = np.nanstd( fracs, axis=0)
        mask          = ts_arr >= half
        tail_per_seed = np.nanmean(fracs[:, mask], axis=1)          # one value per seed
        result[w] = {
            'times':     ts_arr,
            'mean':      mean_arr,
            'std':       std_arr,
            'tail_mean': float(np.nanmean(tail_per_seed)),
            'tail_std':  float(np.nanstd( tail_per_seed)),
        }
    return result


def plot_hres_routing_summary(scenarios_stats, save_path=None):
    """
    figV2b_routing: 4-bar summary chart with error bars (mean ± 1 SD, w=200, last 50 %).
    scenarios_stats: list of (routing_stats_dict, label, colour, hatch)
      ordered [prot_na, prot_aid, expl_na, expl_aid].
    """
    FS    = FONT_SIZES
    W_REP = 200
    n     = len(scenarios_stats)

    means  = np.array([s[W_REP]['tail_mean'] for s, *_ in scenarios_stats])
    stds   = np.array([s[W_REP]['tail_std']  for s, *_ in scenarios_stats])
    labels = [lbl                             for _, lbl, *_ in scenarios_stats]
    cols   = [col                             for _, _, col, _ in scenarios_stats]
    hatches = [h                              for _, _, _, h in scenarios_stats]
    n_seeds = len(scenarios_stats[0][0][W_REP]['times'])   # proxy for info string

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(n)

    ax.bar(x, means, yerr=stds, capsize=7,
           color=cols, hatch=hatches, alpha=0.85, edgecolor='white',
           error_kw={'elinewidth': 1.8, 'ecolor': '#222222', 'capthick': 1.8})
    ax.axhline(0.5, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(1.5, color='black', lw=1.2, ls='--', alpha=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.015, f'{m:.3f}\n±{s:.3f}',
                ha='center', va='bottom', fontsize=FS['small'], color='#333333')

    ax.text(0.5, -0.17, 'Protection',   ha='center',
            transform=ax.get_xaxis_transform(),
            fontsize=FS['annotation'], fontweight='bold', color=_CBLUE)
    ax.text(2.5, -0.17, 'Exploitation', ha='center',
            transform=ax.get_xaxis_transform(),
            fontsize=FS['annotation'], fontweight='bold', color=_CRED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS['tick_label'])
    ax.set_ylabel('H_Res community-benefit routing fraction\n(mean, last 50 %, window w=200)',
                  fontsize=FS['axis_label'])
    ax.set_ylim(0, min(1.05, float(np.max(means + stds)) * 1.6 + 0.12))
    ax.tick_params(labelsize=FS['tick_label'])
    ax.grid(True, axis='y', alpha=0.25)

    from matplotlib.patches import Patch as _Patch
    legend_els = [
        _Patch(facecolor='#888888',          label='No aid'),
        _Patch(facecolor='#888888', hatch='//', label='With aid'),
        Line2D([0], [0], color='black', lw=1, ls='--', alpha=0.5, label='50 % threshold'),
    ]
    ax.legend(handles=legend_els, fontsize=FS['legend'], loc='upper right')
    ax.set_title(
        f'Q2 — H_Res community-benefit routing fraction\n'
        f'r16(×1) + r18(×2)  /  total H_Res outflow  ·  w=200  ·  mean ± 1 SD  ({N_SEEDS} seeds)',
        fontsize=FS['title'], fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


def plot_routing_timescale_multi(scenarios_stats, save_path=None):
    """
    figV2d_routing: 1×4 panels (one per window size).
    Mean ± 1 SD shaded band per scenario; annotates w=200 as the separation timescale.
    scenarios_stats: list of (routing_stats_dict, label, colour, linestyle).
    """
    FS  = FONT_SIZES
    n_w = len(WINDOW_SIZES)
    fig, axes = plt.subplots(1, n_w, figsize=(5 * n_w, 5), sharey=True)
    fig.subplots_adjust(wspace=0.10)

    for ax, w in zip(axes, WINDOW_SIZES):
        for stats, label, col, ls in scenarios_stats:
            ts   = stats[w]['times']
            mean = stats[w]['mean']
            std  = stats[w]['std']
            valid = ~np.isnan(mean)
            ax.plot(ts[valid], mean[valid], color=col, ls=ls,
                    lw=1.6, alpha=0.90, label=label)
            ax.fill_between(ts[valid],
                            np.clip(mean[valid] - std[valid], 0.0, 1.0),
                            np.clip(mean[valid] + std[valid], 0.0, 1.0),
                            color=col, alpha=0.15)
        ax.axhline(0.5, color='black', lw=0.7, ls='--', alpha=0.45)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Window  w = {w}', fontsize=FS['title'], fontweight='bold')
        ax.set_xlabel('Time step', fontsize=FS['axis_label'])
        ax.tick_params(labelsize=FS['tick_label'])
        ax.grid(True, alpha=0.25)

        if w == 200:
            ax.annotate('strategies\nfirst separate',
                        xy=(int(N_STOCH_STEPS * 0.28), 0.28),
                        xytext=(int(N_STOCH_STEPS * 0.05), 0.62),
                        fontsize=FS['small'], color='#444444',
                        arrowprops=dict(arrowstyle='->', color='#444444', lw=1.2))

    axes[0].set_ylabel('H_Res routing fraction\n(community benefit / total outflow)',
                       fontsize=FS['axis_label'])

    handles = [Line2D([0], [0], color=col, ls=ls, lw=2, label=lbl)
               for _, lbl, col, ls in scenarios_stats]
    handles += [Line2D([0], [0], color='black', lw=1, ls='--', alpha=0.5,
                       label='50 % threshold')]
    fig.legend(handles=handles, loc='lower center', ncol=len(scenarios_stats) + 1,
               fontsize=FS['legend'], frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f'H_Res routing fraction across timescales  —  all 4 scenarios  ({N_SEEDS} seeds, mean ± 1 SD)\n'
        '1 = all H_Res to community benefit  |  0 = all to elite expansion / maintenance\n'
        'Panel where strategy bands first separate = observable timescale of routing difference',
        fontsize=FS['suptitle'], fontweight='bold')
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# Q3 -- H_Res LEVERAGE ANALYSIS  (figV2e)
# ===========================================================================
# Gross H_Res outflow per reaction (INPUT_STOICH coefficients, not net STOICH):
#   r11  coef=1  Other maintenance      (chiefdom energy: H+H_Res->H)
#   r15  coef=1  Elite expansion        (C+H+H_Res->2H)
#   r16  coef=1  Redistribution/peace   (H+X+H_Res->P_H)
#   r17  coef=1  Other maintenance      (P_H+H_Res->P_H police upkeep)
#   r18  coef=2  Redistribution/peace   (2H_Res->H_Res+C_Res)
# r12/r13 produce H_Res (positive STOICH row) -- excluded (inflow, not outflow).

_HRES_OUTFLOW_RXNS = {
    'r11': (11, 'Other maintenance'),
    'r15': (15, 'Elite expansion'),
    'r16': (16, 'Redistribution/peacekeeping'),
    'r17': (17, 'Other maintenance'),
    'r18': (18, 'Redistribution/peacekeeping'),
}
_HRES_CATEGORIES = ['Redistribution/peacekeeping', 'Elite expansion', 'Other maintenance']
_HRES_CAT_COLORS = ['#2980B9', '#C0392B', '#7F8C8D']


def compute_hres_outflow(N_arr):
    """
    Total gross H_Res outflow grouped into three categories.
    Returns (cat_totals dict, fracs dict).
    """
    cat_totals = {c: 0.0 for c in _HRES_CATEGORIES}
    for _, (j, cat) in _HRES_OUTFLOW_RXNS.items():
        coef = float(INPUT_STOICH[iHRes, j])
        cat_totals[cat] += float(N_arr[:, j].sum()) * coef
    grand = sum(cat_totals.values())
    fracs = {c: (cat_totals[c] / grand if grand > 1e-12 else 0.0)
             for c in _HRES_CATEGORIES}
    return cat_totals, fracs


def compute_leverage_stats(N_arrs):
    """
    Aggregate H_Res leverage fractions across multiple N_arr seeds.
    Returns {cat: {'mean': float, 'std': float}} for each category.
    """
    all_fracs = [compute_hres_outflow(N)[1] for N in N_arrs]
    return {cat: {
        'mean': float(np.mean([f[cat] for f in all_fracs])),
        'std':  float(np.std( [f[cat] for f in all_fracs])),
    } for cat in _HRES_CATEGORIES}


def plot_hres_leverage_multi(scenarios_multi, save_path=None):
    """
    Two-panel figure (Q3 left, Q2 right) with multi-seed error bars.
    Left:  stacked bar chart – H_Res outflow fractions; error bars on redistribution segment.
    Right: delta-leverage grouped bars with propagated uncertainty.
    scenarios_multi: list of (list_of_N_arrs, label)
      ordered [prot_na, expl_na, prot_aid, expl_aid].
    """
    assert len(scenarios_multi) == 4, \
        'Requires exactly 4 scenarios: prot_na, expl_na, prot_aid, expl_aid'
    FS = FONT_SIZES

    stats_all = [(compute_leverage_stats(N_arrs), lbl)
                 for N_arrs, lbl in scenarios_multi]
    labels    = [lbl for _, lbl in stats_all]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 7))
    fig.subplots_adjust(wspace=0.38)

    # ── Left panel: stacked bars with error bars on redistribution (Q3) ──────
    x       = np.arange(len(scenarios_multi))
    bottoms = np.zeros(len(scenarios_multi))
    for cat, col in zip(_HRES_CATEGORIES, _HRES_CAT_COLORS):
        means = np.array([s[cat]['mean'] for s, _ in stats_all])
        stds  = np.array([s[cat]['std']  for s, _ in stats_all])
        ax_l.bar(x, means, bottom=bottoms, color=col,
                 label=cat, alpha=0.85, edgecolor='white')
        if cat == 'Redistribution/peacekeeping':
            ax_l.errorbar(x, bottoms + means, yerr=stds, fmt='none',
                          capsize=5, elinewidth=1.5, ecolor='black', capthick=1.5,
                          zorder=5)
        for i, (v, b) in enumerate(zip(means, bottoms)):
            if v > 0.04:
                ax_l.text(i, b + v / 2, f'{v:.1%}',
                          ha='center', va='center',
                          fontsize=FS['annotation'], fontweight='bold', color='white')
        bottoms += means

    ax_l.axvline(1.5, color='black', lw=1.5, ls='--', alpha=0.7)
    ax_l.text(0.75, 1.03, 'Protection',
              transform=ax_l.get_xaxis_transform(),
              ha='center', fontsize=FS['annotation'], fontweight='bold')
    ax_l.text(2.75, 1.03, 'Exploitation',
              transform=ax_l.get_xaxis_transform(),
              ha='center', fontsize=FS['annotation'], fontweight='bold')
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=FS['tick_label'])
    ax_l.set_ylabel('Fraction of total H_Res gross outflow', fontsize=FS['axis_label'])
    ax_l.set_ylim(0, 1.14)
    ax_l.legend(loc='upper right', fontsize=FS['legend'])
    ax_l.grid(True, axis='y', alpha=0.25)
    ax_l.tick_params(labelsize=FS['tick_label'])
    ax_l.set_title(
        'Q3 – H_Res allocation across all 4 scenarios\n'
        f'Gross outflow: r11(×1)+r15(×1)+r16(×1)+r17(×1)+r18(×2)  ·  {N_SEEDS} seeds\n'
        'Error bars on redistribution segment (mean ± 1 SD)',
        fontsize=FS['title'], fontweight='bold')

    # ── Right panel: delta-leverage with propagated error bars (Q2) ──────────
    n_cats    = len(_HRES_CATEGORIES)
    x_cat     = np.arange(n_cats)
    bar_w     = 0.35
    cat_short = ['Redist /\nPeacekeeping', 'Elite\nExpansion', 'Other\nMaintenance']

    for offset, (na_idx, aid_idx, strat_label, strat_col) in enumerate([
            (0, 2, 'Protection  (+aid minus no-aid)',  _CBLUE),
            (1, 3, 'Exploitation  (+aid minus no-aid)', _CRED)]):
        s_na  = stats_all[na_idx][0]
        s_aid = stats_all[aid_idx][0]
        vals  = np.array([s_aid[c]['mean'] - s_na[c]['mean'] for c in _HRES_CATEGORIES])
        errs  = np.array([np.sqrt(s_aid[c]['std']**2 + s_na[c]['std']**2)
                          for c in _HRES_CATEGORIES])
        ax_r.bar(x_cat + offset * bar_w, vals, yerr=errs, width=bar_w,
                 color=strat_col, label=strat_label, alpha=0.85, edgecolor='white',
                 capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        for i, v in enumerate(vals):
            if abs(v) > 0.005:
                va = 'bottom' if v >= 0 else 'top'
                yp = v + 0.004 if v >= 0 else v - 0.004
                ax_r.text(i + offset * bar_w, yp, f'{v:+.1%}',
                          ha='center', va=va,
                          fontsize=FS['annotation'], fontweight='bold', color=strat_col)

    ax_r.axhline(0, color='black', lw=0.9)
    ax_r.set_xticks(x_cat + bar_w / 2)
    ax_r.set_xticklabels(cat_short, fontsize=FS['tick_label'])
    ax_r.set_ylabel('Change in H_Res outflow fraction\n(with-aid  minus  no-aid)',
                    fontsize=FS['axis_label'])
    ax_r.legend(loc='upper right', fontsize=FS['legend'])
    ax_r.grid(True, axis='y', alpha=0.25)
    ax_r.tick_params(labelsize=FS['tick_label'])
    ax_r.set_title(
        'Q2 – Where does marginal aid H_Res go?\n'
        f'Delta = frac(with-aid) minus frac(no-aid)  ·  {N_SEEDS} seeds  ·  error = propagated SD',
        fontsize=FS['title'], fontweight='bold')

    fig.suptitle(
        'H_Res as leverage point: strategy multipliers flip the dominant pathway  (Q3)\n'
        'Marginal aid amplifies the already-dominant route within each strategy  (Q2)',
        fontsize=FS['suptitle'], fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    print('=' * 72)
    print('Intermediate Band-Tribe-Chiefdom Model  v2  (stochastic only)')
    print(f'  {N_SPECIES} species, {N_REACTIONS} reactions  (r0-r21)')
    print(f'  Network: {_RN_FILE}')
    print('  Dual extraction: r12 (taxes Res) + r13 (taxes C_Res)')
    print('  Redistribution pivot: r18 (2H_Res => H_Res + C_Res)')
    print('=' * 72)
    print(f'Baseline (Scenario A from script_basic_model.py):')
    print(f'  C={C_INIT}, X={X_INIT}, Res={RES_INIT}, C_Res={CRES_INIT}, G={G_INIT}')
    print(f'Chiefdom seed: H_INIT={H_INIT}, HRES_INIT={HRES_INIT}')
    print(f'Foreign aid:   F_H_LEVEL={F_H_LEVEL},  X-threshold={AID_X_THRESHOLD},  '
          f'G-threshold={AID_G_THRESHOLD},  decay={AID_DECAY}')
    print()
    print(f'Protection  allocation (invest in protection mode: 2*r12+r16+r18):')
    print(f'  {dict(zip([f"r{j}" for j in CHIEF_CTRL_INDICES], A_CHIEF_PROTECT))}')
    print(f'Exploitation allocation (invest in exploitation mode: r12+r13+2*r15):')
    print(f'  {dict(zip([f"r{j}" for j in CHIEF_CTRL_INDICES], A_CHIEF_EXPLOIT))}')
    print()

    # -----------------------------------------------------------------------
    # Run 4 scenarios × N_SEEDS seeds.
    # SEEDS[0] is the primary seed; its X_arr is used for stock / mode figures.
    # All seeds' N_arrs are used for multi-seed Q2/Q3 analysis.
    # -----------------------------------------------------------------------
    print(f'--- Running stochastic simulations  ({N_SEEDS} seeds × 4 scenarios) ---')
    print(f'  Seeds: {SEEDS}')

    print(f'  Protection,   no aid   (N={N_STOCH_STEPS}, dt={DT_STOCH}) ...')
    _runs_prot_na   = [run_stochastic(A_CHIEF_PROTECT, seed=s, use_aid=False) for s in SEEDS]
    print(f'  Exploitation, no aid   ...')
    _runs_expl_na   = [run_stochastic(A_CHIEF_EXPLOIT, seed=s, use_aid=False) for s in SEEDS]
    print(f'  Protection,   with aid ...')
    _runs_prot_aid  = [run_stochastic(A_CHIEF_PROTECT, seed=s, use_aid=True)  for s in SEEDS]
    print(f'  Exploitation, with aid ...')
    _runs_expl_aid  = [run_stochastic(A_CHIEF_EXPLOIT, seed=s, use_aid=True)  for s in SEEDS]
    print()

    # Primary-seed trajectories (for stock / mode / timescale figures)
    X_prot_na,  N_prot_na  = _runs_prot_na[0]
    X_expl_na,  N_expl_na  = _runs_expl_na[0]
    X_prot_aid, N_prot_aid = _runs_prot_aid[0]
    X_expl_aid, N_expl_aid = _runs_expl_aid[0]

    # All-seed N_arr and X_arr lists (for multi-seed Q2/Q3/figV2c figures)
    N_multi_prot_na  = [N for _, N in _runs_prot_na]
    N_multi_expl_na  = [N for _, N in _runs_expl_na]
    N_multi_prot_aid = [N for _, N in _runs_prot_aid]
    N_multi_expl_aid = [N for _, N in _runs_expl_aid]

    X_multi_prot_na  = [X for X, _ in _runs_prot_na]
    X_multi_expl_na  = [X for X, _ in _runs_expl_na]
    X_multi_prot_aid = [X for X, _ in _runs_prot_aid]
    X_multi_expl_aid = [X for X, _ in _runs_expl_aid]

    # Pre-compute routing stats (expensive per-seed window sweep, done once)
    print('--- Pre-computing routing stats across seeds ---')
    rs_prot_na  = compute_routing_stats(N_multi_prot_na)
    rs_expl_na  = compute_routing_stats(N_multi_expl_na)
    rs_prot_aid = compute_routing_stats(N_multi_prot_aid)
    rs_expl_aid = compute_routing_stats(N_multi_expl_aid)
    print()

    _aid_tag = (f'Adaptive foreign aid  '
                f'(F_H={F_H_LEVEL}  when  X>={AID_X_THRESHOLD}  or  G>={AID_G_THRESHOLD})')

    # -----------------------------------------------------------------------
    # Figure set 1: WITHOUT foreign aid
    # Expected: Protection -> C majority, H moderate, X low
    #           Exploitation -> collapses: C_Res broken, H_Res exhausted, X dominates
    # -----------------------------------------------------------------------
    print('--- Figure set 1: Without foreign aid ---')
    plot_stocks(
        X_prot_na, X_expl_na,
        title_tag='Without foreign aid',
        save_path=os.path.join(_OUT_DIR, 'figV2a_stocks_noaid.png'))

    plot_mode_projections(
        [(N_multi_prot_na,  'Protection   (no aid)'),
         (N_multi_prot_aid, 'Protection   (with aid)'),
         (N_multi_expl_na,  'Exploitation (no aid)'),
         (N_multi_expl_aid, 'Exploitation (with aid)')],
        save_path=os.path.join(_OUT_DIR, 'figV2b_mode.png'))

    plot_mode_dominance_timeseries(
        [(N_multi_prot_na,  'Protection   (no aid)'),
         (N_multi_prot_aid, 'Protection   (with aid)'),
         (N_multi_expl_na,  'Exploitation (no aid)'),
         (N_multi_expl_aid, 'Exploitation (with aid)')],
        save_path=os.path.join(_OUT_DIR, 'figV2f_dominance.png'))

    plot_timescale_modes(
        N_multi_prot_na, 'Protection  (no aid)',
        X_arrs=X_multi_prot_na,
        save_path=os.path.join(_OUT_DIR, 'figV2c_timescale_prot_noaid.png'))

    plot_timescale_modes(
        N_multi_expl_na, 'Exploitation  (no aid)',
        X_arrs=X_multi_expl_na,
        save_path=os.path.join(_OUT_DIR, 'figV2c_timescale_expl_noaid.png'))

    # -----------------------------------------------------------------------
    # Figure set 2: WITH adaptive foreign aid
    # Expected: Protection -> same or stronger community outcomes
    #           Exploitation -> H+P_H police state, C_Res never recovers
    # -----------------------------------------------------------------------
    print('--- Figure set 2: With adaptive foreign aid ---')
    plot_stocks(
        X_prot_aid, X_expl_aid,
        title_tag=_aid_tag,
        save_path=os.path.join(_OUT_DIR, 'figV2a_stocks_aid.png'))

    plot_timescale_modes(
        N_multi_prot_aid, 'Protection  (with aid)',
        X_arrs=X_multi_prot_aid,
        save_path=os.path.join(_OUT_DIR, 'figV2c_timescale_prot_aid.png'))

    plot_timescale_modes(
        N_multi_expl_aid, 'Exploitation  (with aid)',
        X_arrs=X_multi_expl_aid,
        save_path=os.path.join(_OUT_DIR, 'figV2c_timescale_expl_aid.png'))

    # -----------------------------------------------------------------------
    # Q2 -- H_Res routing fraction (multi-seed)
    # figV2b_routing: 4-bar summary chart (mean ± SD, w=200, last 50 %)
    # figV2d_routing: 1×4 panels, mean ± SD band per scenario
    # -----------------------------------------------------------------------
    print('--- Q2: H_Res routing fraction (figV2b_routing + figV2d_routing) ---')
    plot_hres_routing_summary(
        [(rs_prot_na,  'No aid',    _CBLUE, ''),
         (rs_prot_aid, 'With aid',  _CBLUE, '//'),
         (rs_expl_na,  'No aid',    _CRED,  ''),
         (rs_expl_aid, 'With aid',  _CRED,  '//')],
        save_path=os.path.join(_OUT_DIR, 'figV2b_routing.png'))

    plot_routing_timescale_multi(
        [(rs_prot_na,  'Protection  (no aid)',    _CBLUE, '-'),
         (rs_prot_aid, 'Protection  (with aid)',  _CBLUE, '--'),
         (rs_expl_na,  'Exploitation  (no aid)',  _CRED,  '-'),
         (rs_expl_aid, 'Exploitation  (with aid)',_CRED,  '--')],
        save_path=os.path.join(_OUT_DIR, 'figV2d_routing.png'))

    # -----------------------------------------------------------------------
    # Q3 + Q2 combined -- H_Res leverage (left) and delta-leverage (right)
    # Both panels now show multi-seed mean ± SD / propagated error.
    # -----------------------------------------------------------------------
    print('--- Q3/Q2: H_Res leverage + delta-leverage (figV2e_leverage.png) ---')
    _scen_lev_multi = [
        (N_multi_prot_na,  'Prot\n(no aid)'),
        (N_multi_expl_na,  'Expl\n(no aid)'),
        (N_multi_prot_aid, 'Prot\n(aid)'),
        (N_multi_expl_aid, 'Expl\n(aid)'),
    ]
    plot_hres_leverage_multi(
        _scen_lev_multi,
        save_path=os.path.join(_OUT_DIR, 'figV2e_leverage.png'))

    # -----------------------------------------------------------------------
    # Q2 / Q3 Summary table
    # -----------------------------------------------------------------------
    print()
    print(f'--- Q2 / Q3 Summary table  ({N_SEEDS} seeds) ---')
    print(f'{"Scenario":<26}  {"Redist/Peace (mean±SD)":<26}  {"Elite (mean±SD)":<22}  '
          f'{"Routing frac w=200 (mean±SD)"}')
    print('-' * 110)
    for sname, N_arrs, rs in [
            ('Prot (no aid)', N_multi_prot_na,  rs_prot_na),
            ('Expl (no aid)', N_multi_expl_na,  rs_expl_na),
            ('Prot (aid)',    N_multi_prot_aid,  rs_prot_aid),
            ('Expl (aid)',    N_multi_expl_aid,  rs_expl_aid)]:
        lev   = compute_leverage_stats(N_arrs)
        f_red = lev['Redistribution/peacekeeping']
        f_eli = lev['Elite expansion']
        rf200 = rs[200]
        print(f'{sname:<26}  '
              f'{f_red["mean"]:.3f} ± {f_red["std"]:.3f}       '
              f'{f_eli["mean"]:.3f} ± {f_eli["std"]:.3f}     '
              f'{rf200["tail_mean"]:.3f} ± {rf200["tail_std"]:.3f}')

    print()
    print(f'All outputs saved to: {_OUT_DIR}')
