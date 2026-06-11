#!/usr/bin/env python3
"""
script_intra_org_esmo.py
========================
Non-parametric ESMO analysis of intra-organisational transitions:
peace -> conflict (PC / destabilisation) and conflict -> peace (CP / recovery)
within each of the four social organisations (Tribe, Chief, State, ChiefState).

DESIGN
------
Each organisation is assigned a STATE_LIBRARY of M peace-states and N
conflict-states (currently 5+5 per org).  All M×N PC pairs and all N×M CP
pairs are enumerated.  For each pair the ESMO LP is solved once.

KEY STRUCTURAL DISTINCTNESS CRITERION
States within each type (peace or conflict) are designed so that different
pairs produce different LP sign-patterns (signatures) over the O7 species.
Pairs that share an identical signature map to the same LP polytope and are
therefore equivalent; the cache deduplicates them via a signature hash.

PERSISTENT CACHE (esmo_cache_intra.csv)
Every computed result is appended immediately to a CSV file.  On restart the
script skips any (state_from_id, state_to_id) pairs already present in the
file — even across partial runs or multiple contributors.  The cache is
append-friendly: multiple users can generate separate CSVs and merge them
by concatenating rows.

PROBABILITY MODEL (Eq. 2 of the paper)
After all pairs are computed, for each org and direction (PC / CP):
  Pr_feasible     = |{pairs : n_esmos > 0}| / (M*N)
  mean_N          = mean(n_esmos over all pairs)
  std_N           = std(n_esmos over all pairs)
  Pr_feasible serves as a parameter-free structural feasibility probability.

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
OUT_DIR    = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'intra_org_esmo')
CACHE_FILE = os.path.join(OUT_DIR, 'esmo_cache_intra.csv')

# ---------------------------------------------------------------------------
# 2. ORGANISATION SPECIES SETS
# ---------------------------------------------------------------------------
_CHIEF_STATE = frozenset({'C', 'C_Res', 'G', 'H', 'H_Res', 'I', 'I_Res', 'L',
                           'P_H', 'P_I', 'Res', 'X'})

ORG_NAMES = ['Tribe', 'Chief', 'State', 'ChiefState']

ORG_ABBR = {
    'Tribe':      'Tri',
    'Chief':      'Chf',
    'State':      'Sta',
    'ChiefState': 'CfS',
}

# ---------------------------------------------------------------------------
# 3. STATE LIBRARY
# Key: (org_name, state_id)  — state_id encodes type and variant.
# Value: dict mapping O7-species names to abundance values.
# Species absent from an org are set to 0 (they will be outside the org's
# closed set so they cannot appear in constrained positions anyway).
#
# DISTINCTNESS CRITERION: states within each (org, type) group must differ
# in WHICH species have meaningfully different values (not just proportional
# scaling), so that different source/goal pairs produce different LP
# sign-patterns.  At least 3–4 distinct signatures per direction per org.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# POPULATION CONSERVATION NOTE
# Every reaction in the network converts one person-type to another without
# creating or destroying people.  The conserved quantity per organisation is:
#   P = C + X + H + I + P_H + P_I  (species absent from an org are kept at 0)
# Tribe: P = 500   (C + X = 500,               H=I=P_H=P_I=0)
# Chief: P = 1000  (C + X + H + P_H = 1000,    I=P_I=0)
# State: P = 10000 (C + X + I + P_I = 10000,   H=P_H=0)
# ChiefState: P = 10000  (all six active)
#
# Peace / conflict distinction:
#   peace   → X small, G small; social structure functioning as expected
#   conflict → X large, G large; social displacement and grievance dominant
# Within an org, the composition of non-X people (C vs H, C vs I, etc.)
# can vary freely — a chiefdom at peace can be H-dominant; a state at peace
# can be I-dominant.  Signature diversity comes from varying these ratios and
# from atypical resource configurations (e.g. resource-rich conflict).
# ---------------------------------------------------------------------------

STATE_LIBRARY = {

    # ── TRIBE  (P = C + X = 500; active non-people: G, Res, C_Res) ──────────
    # Peace: X small (20-120), G small (3-25).  Five states vary in
    # (Res, C_Res) to produce distinct LP sign-patterns.
    ('Tribe', 'peace_v1'): dict(
        C=450, X=50,  G=5,  Res=150, C_Res=100,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'peace_v2'): dict(
        C=430, X=70,  G=8,  Res=80,  C_Res=200,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'peace_v3'): dict(
        C=460, X=40,  G=3,  Res=250, C_Res=60,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'peace_v4'): dict(
        C=380, X=120, G=25, Res=50,  C_Res=80,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'peace_v5'): dict(
        C=470, X=30,  G=10, Res=20,  C_Res=30,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),

    # Conflict: X large (250-420), G large (50-100).
    # conflict_v4 has atypically high Res/C_Res (resource-rich conflict) →
    # creates reversed Res/C_Res sign in signatures paired with peace_v4/v5.
    ('Tribe', 'conflict_v1'): dict(
        C=150, X=350, G=80,  Res=15,  C_Res=15,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'conflict_v2'): dict(
        C=200, X=300, G=60,  Res=5,   C_Res=8,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'conflict_v3'): dict(
        C=80,  X=420, G=100, Res=20,  C_Res=20,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'conflict_v4'): dict(
        C=250, X=250, G=50,  Res=100, C_Res=80,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),
    ('Tribe', 'conflict_v5'): dict(
        C=180, X=320, G=70,  Res=40,  C_Res=3,
        H=0, H_Res=0, P_H=0, I=0, I_Res=0, L=0, P_I=0),

    # ── CHIEF  (P = C + X + H + P_H = 1000; active non-people: G, Res, C_Res, H_Res) ─
    # Peace: X small (80-120), G small (3-15).  H can dominate or equal C.
    # Five states vary in C/H ratio and P_H size → different H/P_H sign directions.
    ('Chief', 'peace_v1'): dict(
        C=200, H=550, X=80,  P_H=170,
        G=5,  Res=80,  C_Res=50,  H_Res=300,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'peace_v2'): dict(
        C=350, H=380, X=120, P_H=150,
        G=10, Res=60,  C_Res=70,  H_Res=200,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'peace_v3'): dict(
        C=500, H=250, X=100, P_H=150,
        G=8,  Res=120, C_Res=100, H_Res=120,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'peace_v4'): dict(
        C=300, H=300, X=100, P_H=300,
        G=15, Res=50,  C_Res=40,  H_Res=250,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'peace_v5'): dict(
        C=150, H=650, X=80,  P_H=120,
        G=3,  Res=100, C_Res=30,  H_Res=400,
        I=0, I_Res=0, L=0, P_I=0),

    # Conflict: X large (350-680), G large (50-100).
    # Varying H/P_H levels across conflict states produces diverse H/P_H signs:
    # v1: H small+P_H moderate (collapse); v2: P_H large (militarised);
    # v3: H collapses; v4: H survives with resources; v5: H dominant.
    ('Chief', 'conflict_v1'): dict(
        C=100, H=150, X=600, P_H=150,
        G=80,  Res=15, C_Res=15, H_Res=30,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'conflict_v2'): dict(
        C=150, H=200, X=350, P_H=300,
        G=60,  Res=20, C_Res=20, H_Res=50,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'conflict_v3'): dict(
        C=200, H=50,  X=680, P_H=70,
        G=100, Res=10, C_Res=10, H_Res=10,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'conflict_v4'): dict(
        C=200, H=300, X=400, P_H=100,
        G=50,  Res=60, C_Res=30, H_Res=150,
        I=0, I_Res=0, L=0, P_I=0),
    ('Chief', 'conflict_v5'): dict(
        C=100, H=450, X=350, P_H=100,
        G=70,  Res=8,  C_Res=8,  H_Res=40,
        I=0, I_Res=0, L=0, P_I=0),

    # ── STATE  (P = C + X + I + P_I = 10000; active non-people: G, Res, C_Res, I_Res, L) ─
    # Peace: X small (300-1200), G small (2-12).  I can dominate (I > C).
    # Five states vary in C/I ratio and P_I size → different I/P_I sign directions.
    ('State', 'peace_v1'): dict(
        C=1500, I=7000, X=500,  P_I=1000,
        G=5,  Res=300, C_Res=150, I_Res=2000, L=800,
        H=0, H_Res=0, P_H=0),
    ('State', 'peace_v2'): dict(
        C=3500, I=4500, X=800,  P_I=1200,
        G=8,  Res=200, C_Res=200, I_Res=1500, L=500,
        H=0, H_Res=0, P_H=0),
    ('State', 'peace_v3'): dict(
        C=6000, I=2000, X=1200, P_I=800,
        G=12, Res=400, C_Res=300, I_Res=600,  L=300,
        H=0, H_Res=0, P_H=0),
    ('State', 'peace_v4'): dict(
        C=2000, I=5000, X=500,  P_I=2500,
        G=3,  Res=150, C_Res=100, I_Res=3000, L=1000,
        H=0, H_Res=0, P_H=0),
    ('State', 'peace_v5'): dict(
        C=2500, I=6000, X=300,  P_I=1200,
        G=2,  Res=500, C_Res=80,  I_Res=3500, L=1200,
        H=0, H_Res=0, P_H=0),

    # Conflict: X large (2500-5500), G large (150-300).
    # Varying I/P_I levels produces diverse I/P_I signs:
    # v1: I collapses; v2: I partially survives; v3: large P_I (militarised);
    # v4: resource-rich conflict; v5: I dominant despite high X.
    ('State', 'conflict_v1'): dict(
        C=2000, I=1500, X=5500, P_I=1000,
        G=300, Res=30,  C_Res=30,  I_Res=200,  L=50,
        H=0, H_Res=0, P_H=0),
    ('State', 'conflict_v2'): dict(
        C=3000, I=3500, X=2500, P_I=1000,
        G=200, Res=15,  C_Res=20,  I_Res=300,  L=100,
        H=0, H_Res=0, P_H=0),
    ('State', 'conflict_v3'): dict(
        C=2000, I=2500, X=3000, P_I=2500,
        G=150, Res=20,  C_Res=15,  I_Res=400,  L=80,
        H=0, H_Res=0, P_H=0),
    ('State', 'conflict_v4'): dict(
        C=3000, I=2000, X=4000, P_I=1000,
        G=180, Res=200, C_Res=100, I_Res=500,  L=60,
        H=0, H_Res=0, P_H=0),
    ('State', 'conflict_v5'): dict(
        C=1500, I=4500, X=3500, P_I=500,
        G=250, Res=10,  C_Res=10,  I_Res=600,  L=40,
        H=0, H_Res=0, P_H=0),

    # ── CHIEFSTATE  (P = C + X + H + I + P_H + P_I = 10000; all 12 active) ──
    # Peace: X small (400-800), G small (3-10).  H and I can both be large.
    # Five states vary in the C/H/I split and P_H/P_I ratio.
    ('ChiefState', 'peace_v1'): dict(
        C=1000, H=2500, I=4000, X=500,  P_H=500,  P_I=1500,
        G=5,  Res=100, C_Res=50, H_Res=300, I_Res=1500, L=600),
    ('ChiefState', 'peace_v2'): dict(
        C=800,  H=5000, I=2500, X=400,  P_H=800,  P_I=500,
        G=3,  Res=80,  C_Res=30, H_Res=600, I_Res=800,  L=400),
    ('ChiefState', 'peace_v3'): dict(
        C=1500, H=1000, I=5500, X=600,  P_H=200,  P_I=1200,
        G=4,  Res=120, C_Res=60, H_Res=200, I_Res=2000, L=900),
    ('ChiefState', 'peace_v4'): dict(
        C=1000, H=2000, I=3000, X=500,  P_H=1500, P_I=2000,
        G=8,  Res=70,  C_Res=40, H_Res=400, I_Res=1000, L=500),
    ('ChiefState', 'peace_v5'): dict(
        C=3000, H=2000, I=3000, X=800,  P_H=400,  P_I=800,
        G=10, Res=150, C_Res=80, H_Res=300, I_Res=1200, L=500),

    # Conflict: X large (2500-7000), G large (180-400).
    # v1: everything fragments (X dominates all); v2: H survives, I collapses;
    # v3: I survives, H collapses; v4: militarised (large P_H+P_I);
    # v5: resource-rich conflict (Res/H_Res/I_Res survive).
    ('ChiefState', 'conflict_v1'): dict(
        C=500,  H=500,  I=500,  X=7000, P_H=500,  P_I=1000,
        G=400, Res=10,  C_Res=10, H_Res=20,  I_Res=50,  L=20),
    ('ChiefState', 'conflict_v2'): dict(
        C=1000, H=3500, I=500,  X=3500, P_H=1000, P_I=500,
        G=250, Res=15,  C_Res=15, H_Res=100, I_Res=40,  L=20),
    ('ChiefState', 'conflict_v3'): dict(
        C=1500, H=500,  I=3500, X=3500, P_H=300,  P_I=700,
        G=200, Res=12,  C_Res=12, H_Res=25,  I_Res=200, L=80),
    ('ChiefState', 'conflict_v4'): dict(
        C=1000, H=1500, I=2000, X=2500, P_H=1500, P_I=1500,
        G=300, Res=20,  C_Res=15, H_Res=50,  I_Res=100, L=30),
    ('ChiefState', 'conflict_v5'): dict(
        C=1500, H=2000, I=2000, X=3000, P_H=500,  P_I=1000,
        G=180, Res=100, C_Res=50, H_Res=150, I_Res=400, L=100),
}

# Map each (org, state_id) to its type
STATE_TYPES = {k: ('peace' if 'peace' in k[1] else 'conflict')
               for k in STATE_LIBRARY}

# Human-readable descriptions (for reporting)
STATE_DESC = {
    # Tribe: P = C+X = 500
    ('Tribe', 'peace_v1'):    'C=450 X=50  — high C, Res-rich peace',
    ('Tribe', 'peace_v2'):    'C=430 X=70  — C_Res-rich peace (resilience-built)',
    ('Tribe', 'peace_v3'):    'C=460 X=40  — Res-abundant peace (surplus economy)',
    ('Tribe', 'peace_v4'):    'C=380 X=120 — tensioned peace (moderate displacement)',
    ('Tribe', 'peace_v5'):    'C=470 X=30  — Res-depleted peace (subsistence)',
    ('Tribe', 'conflict_v1'): 'C=150 X=350 — displacement conflict (Res/C_Res depleted)',
    ('Tribe', 'conflict_v2'): 'C=200 X=300 — resource-collapse conflict',
    ('Tribe', 'conflict_v3'): 'C=80  X=420 — extreme displacement conflict',
    ('Tribe', 'conflict_v4'): 'C=250 X=250 — resource-rich conflict (atypical Res high)',
    ('Tribe', 'conflict_v5'): 'C=180 X=320 — C_Res-collapse conflict',
    # Chief: P = C+X+H+P_H = 1000
    ('Chief', 'peace_v1'):    'C=200 H=550 P_H=170 — H-dominant peace (strong hierarchy)',
    ('Chief', 'peace_v2'):    'C=350 H=380 P_H=150 — balanced C/H peace',
    ('Chief', 'peace_v3'):    'C=500 H=250 P_H=150 — C-dominant chiefdom peace',
    ('Chief', 'peace_v4'):    'C=300 H=300 P_H=300 — militarised peace (large P_H)',
    ('Chief', 'peace_v5'):    'C=150 H=650 P_H=120 — H-dominant H_Res-rich peace',
    ('Chief', 'conflict_v1'): 'C=100 H=150 X=600  — social collapse (H+P_H moderate)',
    ('Chief', 'conflict_v2'): 'C=150 H=200 X=350  — militarised conflict (P_H=300)',
    ('Chief', 'conflict_v3'): 'C=200 H=50  X=680  — hierarchy-collapse conflict',
    ('Chief', 'conflict_v4'): 'C=200 H=300 X=400  — resource-rich conflict (H survives)',
    ('Chief', 'conflict_v5'): 'C=100 H=450 X=350  — H-dominant conflict',
    # State: P = C+X+I+P_I = 10000
    ('State', 'peace_v1'):    'C=1500 I=7000 P_I=1000 — I-dominant peace',
    ('State', 'peace_v2'):    'C=3500 I=4500 P_I=1200 — balanced C/I peace',
    ('State', 'peace_v3'):    'C=6000 I=2000 P_I=800  — C-dominant state peace',
    ('State', 'peace_v4'):    'C=2000 I=5000 P_I=2500 — militarised I-dominant peace',
    ('State', 'peace_v5'):    'C=2500 I=6000 P_I=1200 — I-dominant Res-rich peace',
    ('State', 'conflict_v1'): 'C=2000 I=1500 X=5500  — institutional collapse conflict',
    ('State', 'conflict_v2'): 'C=3000 I=3500 X=2500  — I-surviving conflict (resource-poor)',
    ('State', 'conflict_v3'): 'C=2000 I=2500 X=3000  — militarised conflict (P_I=2500)',
    ('State', 'conflict_v4'): 'C=3000 I=2000 X=4000  — resource-rich conflict',
    ('State', 'conflict_v5'): 'C=1500 I=4500 X=3500  — I-dominant conflict',
    # ChiefState: P = C+X+H+I+P_H+P_I = 10000
    ('ChiefState', 'peace_v1'):    'C=1000 H=2500 I=4000 — I-leaning dual-hierarchy peace',
    ('ChiefState', 'peace_v2'):    'C=800  H=5000 I=2500 — H-dominant dual-hierarchy peace',
    ('ChiefState', 'peace_v3'):    'C=1500 H=1000 I=5500 — I-dominant dual-hierarchy peace',
    ('ChiefState', 'peace_v4'):    'C=1000 H=2000 I=3000 P_H=1500 P_I=2000 — militarised peace',
    ('ChiefState', 'peace_v5'):    'C=3000 H=2000 I=3000 — C-heavy balanced peace',
    ('ChiefState', 'conflict_v1'): 'C=500  H=500  I=500  X=7000 — total fragmentation',
    ('ChiefState', 'conflict_v2'): 'C=1000 H=3500 I=500  X=3500 — H-survives conflict',
    ('ChiefState', 'conflict_v3'): 'C=1500 H=500  I=3500 X=3500 — I-survives conflict',
    ('ChiefState', 'conflict_v4'): 'C=1000 H=1500 I=2000 X=2500 — militarised conflict',
    ('ChiefState', 'conflict_v5'): 'C=1500 H=2000 I=2000 X=3000 — resource-rich conflict',
}

# ---------------------------------------------------------------------------
# 4. LP PERFORMANCE SETTINGS
# ---------------------------------------------------------------------------
TOL_STRICT  = 1e-9
TOL_VERTEX  = 1e-7
N_RAND_1    = 2000
N_EXPLORE   = 40
N_RAND_3    = 600
PATIENCE    = 200
SEED        = 42
EPS_SIGN    = 1e-3
NEUTRAL_TOL = 5.0

# ---------------------------------------------------------------------------
# 5. FIGURE SETTINGS
# ---------------------------------------------------------------------------
FONT_SIZE   = 11
COMBO_COLORS = {
    'pc': '#E67E22',   # orange: destabilisation (peace → conflict)
    'cp': '#2980B9',   # blue:   recovery       (conflict → peace)
}

# ===========================================================================
# END CONFIGURATIONS
# ===========================================================================

_OUT_DIR = os.path.realpath(OUT_DIR)
os.makedirs(_OUT_DIR, exist_ok=True)

print(f"Output directory : {_OUT_DIR}")
print(f"Cache file       : {os.path.realpath(CACHE_FILE)}")
print(f"LP settings      : N_RAND_1={N_RAND_1}  N_RAND_3={N_RAND_3}  PATIENCE={PATIENCE}")
print(f"NEUTRAL_TOL      : {NEUTRAL_TOL}")

# ===========================================================================
# 1. LOAD NETWORK AND BUILD O7 SUB-NETWORK
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

# ===========================================================================
# 2. CACHE HELPERS
# ===========================================================================
_CACHE_COLS = [
    'run_id', 'state_from_id', 'state_to_id',
    'org', 'type_from', 'type_to', 'combo',
    'sig_hash', 'sig_json',
    'n_esmos', 'mean_support', 'min_support', 'max_support',
    'time_s', 'source', 'computed_at',
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
        print(f"  Cache: loaded {len(df)} existing results")
        return df
    print(f"  Cache: starting fresh")
    return pd.DataFrame(columns=_CACHE_COLS).set_index('run_id')

def _append_cache(row_dict, cache_df):
    _cf = os.path.realpath(CACHE_FILE)
    row_df = pd.DataFrame([row_dict])
    row_df.to_csv(_cf, mode='a', header=not os.path.exists(_cf), index=False)
    cache_df.loc[row_dict['run_id']] = {k: v for k, v in row_dict.items()
                                         if k != 'run_id'}
    return cache_df

_CACHE = _load_cache()

# ===========================================================================
# 3. LP HELPERS  (identical to script_complex_COT_ESMO for reproducibility)
# ===========================================================================

def build_lp(sig_dict):
    target_map = {}
    for sp_name, sign in sig_dict.items():
        if sp_name in O7_SP:
            target_map[O7_SP.index(sp_name)] = sign
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

# ===========================================================================
# 4. SIGNATURE DERIVATION
# ===========================================================================

def derive_signature(state_from_dict, state_to_dict):
    sig = {}
    for s in O7_SP:
        delta = float(state_to_dict.get(s, 0.0)) - float(state_from_dict.get(s, 0.0))
        if   delta >  NEUTRAL_TOL: sig[s] = +1
        elif delta < -NEUTRAL_TOL: sig[s] = -1
    return sig

# ===========================================================================
# 5. BUILD PAIR LIST
# ===========================================================================

print(f"\nBuilding pair list ...")
PAIRS = []   # list of dicts describing each (from, to) computation

for org in ORG_NAMES:
    peace_ids    = [(org, sid) for (o, sid) in STATE_LIBRARY if o == org
                    and STATE_TYPES[(o, sid)] == 'peace']
    conflict_ids = [(org, sid) for (o, sid) in STATE_LIBRARY if o == org
                    and STATE_TYPES[(o, sid)] == 'conflict']

    for pkey in peace_ids:
        for ckey in conflict_ids:
            # PC: peace -> conflict
            PAIRS.append({
                'org':          org,
                'combo':        'pc',
                'from_key':     pkey,
                'to_key':       ckey,
                'state_from_id': f'{org}::{pkey[1]}',
                'state_to_id':   f'{org}::{ckey[1]}',
                'type_from':    'peace',
                'type_to':      'conflict',
            })
            # CP: conflict -> peace
            PAIRS.append({
                'org':          org,
                'combo':        'cp',
                'from_key':     ckey,
                'to_key':       pkey,
                'state_from_id': f'{org}::{ckey[1]}',
                'state_to_id':   f'{org}::{pkey[1]}',
                'type_from':    'conflict',
                'type_to':      'peace',
            })

N_PAIRS  = len(PAIRS)
N_CACHED = sum(1 for p in PAIRS
               if _make_run_id(p['state_from_id'], p['state_to_id']) in _CACHE.index)

print(f"  Total pairs    : {N_PAIRS}")
print(f"  Already cached : {N_CACHED}")
print(f"  To compute     : {N_PAIRS - N_CACHED}")

# ===========================================================================
# 6. MAIN COMPUTATION LOOP
# ===========================================================================

print(f"\n{'='*72}")
print("  ESMO ENUMERATION (intra-org PC / CP pairs)")
print(f"{'='*72}")

RESULTS = []   # one dict per PAIR

for pi, p in enumerate(PAIRS):
    run_id = _make_run_id(p['state_from_id'], p['state_to_id'])
    sf_dict = STATE_LIBRARY[p['from_key']]
    st_dict = STATE_LIBRARY[p['to_key']]
    sig     = derive_signature(sf_dict, st_dict)
    sig_h   = _hash_sig(sig) if sig else ''
    sig_j   = json.dumps({k: int(v) for k, v in sig.items()})

    label = f"[{pi+1}/{N_PAIRS}] {p['org']} {p['combo'].upper()}  " \
            f"{p['from_key'][1]} -> {p['to_key'][1]}"

    # ── 1. Exact cache hit ─────────────────────────────────────────────────
    if run_id in _CACHE.index:
        row = _CACHE.loc[run_id]
        n   = int(row['n_esmos'])
        print(f"{label}  [CACHED n={n}]")
        RESULTS.append({**p, 'sig': sig, 'n_esmos': n,
                        'mean_support': float(row['mean_support']),
                        'min_support':  int(row['min_support']),
                        'max_support':  int(row['max_support']),
                        'time_s': float(row['time_s']), 'from_cache': True})
        continue

    # ── 2. Signature-match (same LP, can copy result) ──────────────────────
    if sig_h and len(_CACHE) > 0 and 'sig_hash' in _CACHE.columns:
        sig_matches = _CACHE[_CACHE['sig_hash'] == sig_h]
        if len(sig_matches) > 0:
            ref = sig_matches.iloc[0]
            n   = int(ref['n_esmos'])
            ms  = float(ref['mean_support'])
            print(f"{label}  [SIG_MATCH n={n}]")
            cache_row = {
                'run_id': run_id,
                'state_from_id': p['state_from_id'],
                'state_to_id':   p['state_to_id'],
                'org': p['org'], 'type_from': p['type_from'], 'type_to': p['type_to'],
                'combo': p['combo'], 'sig_hash': sig_h, 'sig_json': sig_j,
                'n_esmos': n, 'mean_support': ms,
                'min_support': int(ref['min_support']),
                'max_support': int(ref['max_support']),
                'time_s': 0.0,
                'source': f'sig_match::{ref.name}',
                'computed_at': datetime.now(timezone.utc).isoformat(),
            }
            _CACHE = _append_cache(cache_row, _CACHE)
            RESULTS.append({**p, 'sig': sig, 'n_esmos': n,
                            'mean_support': ms,
                            'min_support': int(ref['min_support']),
                            'max_support': int(ref['max_support']),
                            'time_s': 0.0, 'from_cache': True})
            continue

    # ── 3. Full computation ────────────────────────────────────────────────
    if not sig:
        print(f"{label}  [EMPTY SIGNATURE — skipping]")
        RESULTS.append({**p, 'sig': sig, 'n_esmos': 0,
                        'mean_support': 0.0, 'min_support': 0, 'max_support': 0,
                        'time_s': 0.0, 'from_cache': False})
        continue

    grow_sp   = sorted(s for s, v in sig.items() if v == +1)
    shrink_sp = sorted(s for s, v in sig.items() if v == -1)
    print(f"\n{label}")
    print(f"  grow: {grow_sp}")
    print(f"  shrink: {shrink_sp}")

    esmos, elapsed = compute_esmos(sig)
    supp = [int(np.sum(v > TOL_STRICT)) for v in esmos] if esmos else []
    n    = len(esmos)
    ms   = round(float(np.mean(supp)), 2) if supp else 0.0
    mins = min(supp) if supp else 0
    maxs = max(supp) if supp else 0

    cache_row = {
        'run_id': run_id,
        'state_from_id': p['state_from_id'],
        'state_to_id':   p['state_to_id'],
        'org': p['org'], 'type_from': p['type_from'], 'type_to': p['type_to'],
        'combo': p['combo'], 'sig_hash': sig_h, 'sig_json': sig_j,
        'n_esmos': n, 'mean_support': ms, 'min_support': mins, 'max_support': maxs,
        'time_s': round(elapsed, 1),
        'source': 'computed',
        'computed_at': datetime.now(timezone.utc).isoformat(),
    }
    _CACHE = _append_cache(cache_row, _CACHE)
    RESULTS.append({**p, 'sig': sig, 'n_esmos': n,
                    'mean_support': ms, 'min_support': mins, 'max_support': maxs,
                    'time_s': round(elapsed, 1), 'from_cache': False})

# ===========================================================================
# 7. AGGREGATE PROBABILITY STATISTICS
# ===========================================================================

print(f"\n{'='*72}")
print("  STRUCTURAL FEASIBILITY ANALYSIS")
print(f"{'='*72}")

stats = {}   # (org, combo) -> dict of metrics
for org in ORG_NAMES:
    for combo in ['pc', 'cp']:
        subset = [r for r in RESULTS if r['org'] == org and r['combo'] == combo]
        n_vals = np.array([r['n_esmos'] for r in subset])
        n_pairs = len(n_vals)
        pr_feas = float(np.mean(n_vals > 0)) if n_pairs > 0 else 0.0
        mean_n  = float(np.mean(n_vals))     if n_pairs > 0 else 0.0
        std_n   = float(np.std(n_vals))      if n_pairs > 0 else 0.0
        stats[(org, combo)] = {
            'n_pairs': n_pairs, 'Pr_feasible': pr_feas,
            'mean_N': mean_n, 'std_N': std_n,
            'min_N': int(n_vals.min()) if n_pairs else 0,
            'max_N': int(n_vals.max()) if n_pairs else 0,
            'n_vals': n_vals,
        }

print(f"\n  {'Org':<14}  combo  pairs  Pr_feas  mean_N   std_N   min  max  CP/PC_ratio")
print(f"  {'-'*14}  -----  -----  -------  ------  ------  ---  ---  -----------")
for org in ORG_NAMES:
    pc = stats[(org, 'pc')]
    cp = stats[(org, 'cp')]
    ratio = cp['mean_N'] / pc['mean_N'] if pc['mean_N'] > 0 else float('inf')
    dom = 'PC>CP (destab.)' if ratio < 0.95 else ('CP>PC (recovery)' if ratio > 1.05 else 'symmetric')
    print(f"  {org:<14}  pc  {pc['n_pairs']:>5}  {pc['Pr_feasible']:.3f}  "
          f"{pc['mean_N']:>6.0f}  {pc['std_N']:>6.0f}  "
          f"{pc['min_N']:>3}  {pc['max_N']:>3}")
    print(f"  {'':<14}  cp  {cp['n_pairs']:>5}  {cp['Pr_feasible']:.3f}  "
          f"{cp['mean_N']:>6.0f}  {cp['std_N']:>6.0f}  "
          f"{cp['min_N']:>3}  {cp['max_N']:>3}  ratio={ratio:.2f}  {dom}")
    print()

# Save aggregate stats CSV
stats_records = []
for org in ORG_NAMES:
    for combo in ['pc', 'cp']:
        s = stats[(org, combo)]
        stats_records.append({
            'org': org, 'combo': combo,
            'n_pairs': s['n_pairs'], 'Pr_feasible': round(s['Pr_feasible'], 4),
            'mean_N': round(s['mean_N'], 2), 'std_N': round(s['std_N'], 2),
            'min_N': s['min_N'], 'max_N': s['max_N'],
        })
pd.DataFrame(stats_records).to_csv(
    os.path.join(_OUT_DIR, 'intra_stats.csv'), index=False)
print(f"\nStats CSV -> {os.path.join(_OUT_DIR, 'intra_stats.csv')}")

# Also dump the signature diversity report (how many distinct signatures per org/combo)
print(f"\n{'='*72}")
print("  SIGNATURE DIVERSITY (distinct LP problems per org/combo)")
print(f"{'='*72}")
print(f"  {'Org':<14}  combo  n_pairs  n_distinct_sigs  coverage")
print(f"  {'-'*14}  -----  -------  ---------------  --------")
for org in ORG_NAMES:
    for combo in ['pc', 'cp']:
        subset = [r for r in RESULTS if r['org'] == org and r['combo'] == combo]
        sigs_seen = set()
        for r in subset:
            h = _hash_sig(r['sig']) if r['sig'] else '__empty__'
            sigs_seen.add(h)
        n_d = len(sigs_seen)
        cov = n_d / len(subset) if subset else 0.0
        print(f"  {org:<14}  {combo}   {len(subset):>5}    {n_d:>12}    {cov:.2f}")

# ===========================================================================
# 8. FIGURES
# ===========================================================================

plt.rcParams.update({
    'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2, 'legend.fontsize': FONT_SIZE - 2,
})

def _save(fig, stem):
    p = os.path.join(_OUT_DIR, f'{stem}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 1: Distribution of ESMO counts — violin plot per org and combo
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, len(ORG_NAMES), figsize=(14, 5), sharey=False)

for oi, org in enumerate(ORG_NAMES):
    ax = axes1[oi]
    data_pc = stats[(org, 'pc')]['n_vals']
    data_cp = stats[(org, 'cp')]['n_vals']

    parts = ax.violinplot([data_pc, data_cp], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc_body, color in zip(parts['bodies'], [COMBO_COLORS['pc'], COMBO_COLORS['cp']]):
        pc_body.set_facecolor(color); pc_body.set_alpha(0.65)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[part].set_color('#2C3E50')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['PC\n(destab.)', 'CP\n(recovery)'], fontsize=9)
    ax.set_title(org, fontweight='bold')
    ax.set_ylabel('N ESMOs' if oi == 0 else '')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    for xi, (data, combo) in enumerate([(data_pc, 'pc'), (data_cp, 'cp')], start=1):
        mn = stats[(org, combo)]['mean_N']
        ax.scatter([xi], [mn], zorder=5, s=40, color='white',
                   edgecolors=COMBO_COLORS[combo], linewidths=1.5)
        ax.text(xi + 0.12, mn, f'μ={mn:.0f}', va='center', fontsize=7)

fig1.suptitle(
    'Distribution of ESMO counts across all state pairs (intra-org)\n'
    'PC = peace→conflict (destabilisation),  CP = conflict→peace (recovery)\n'
    'White dot = mean; violin = full distribution',
    fontsize=FONT_SIZE, y=1.03)
fig1.tight_layout()
_save(fig1, 'fig1_esmo_distributions')

# ---------------------------------------------------------------------------
# Fig 2: Feasibility probability and mean N side by side
# ---------------------------------------------------------------------------
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(ORG_NAMES))
W = 0.35

for ax, metric, ylabel, title_sfx in [
        (ax2a, 'Pr_feasible',  'Structural feasibility Pr(N > 0)',
         'Structural feasibility probability'),
        (ax2b, 'mean_N',       'Mean ESMO count  E[N]',
         'Mean structural accessibility E[N]')]:

    for ci, (combo, offset) in enumerate(zip(['pc', 'cp'], [-W/2, W/2])):
        vals = [stats[(org, combo)][metric] for org in ORG_NAMES]
        errs = ([stats[(org, combo)]['std_N'] for org in ORG_NAMES]
                if metric == 'mean_N' else None)
        bars = ax.bar(x + offset, vals, W, color=COMBO_COLORS[combo],
                      alpha=0.80, label=('PC destab.' if combo == 'pc' else 'CP recovery'),
                      zorder=3, yerr=errs, capsize=4,
                      error_kw={'ecolor': '#2C3E50', 'lw': 1.2})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (max(vals)*0.01 if vals else 0),
                    f'{v:.2f}' if metric == 'Pr_feasible' else f'{v:.0f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ORG_NAMES, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title_sfx)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

fig2.suptitle(
    'Non-parametric structural transition probabilities (intra-org)\n'
    'Error bars on E[N] show ±1 SD over all state pairs',
    fontsize=FONT_SIZE, y=1.03)
fig2.tight_layout()
_save(fig2, 'fig2_feasibility_and_mean')

# ---------------------------------------------------------------------------
# Fig 3: Scatter of all (PC, CP) pairs — each org a panel
# x = n_esmos PC,  y = n_esmos CP for the same peace/conflict state pair
# (only valid when a peace state p and conflict state c define BOTH a PC pair
# and a CP pair; here we match them by the shared (peace, conflict) state pair)
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, len(ORG_NAMES), figsize=(14, 4.5), sharey=False)

for oi, org in enumerate(ORG_NAMES):
    ax = axes3[oi]
    # Match PC and CP results by (peace_id, conflict_id) pairing
    pc_dict = {(r['from_key'], r['to_key']): r['n_esmos']
               for r in RESULTS if r['org'] == org and r['combo'] == 'pc'}
    cp_dict = {(r['to_key'], r['from_key']): r['n_esmos']
               for r in RESULTS if r['org'] == org and r['combo'] == 'cp'}
    matched_pc, matched_cp = [], []
    for pair_key, n_pc in pc_dict.items():
        n_cp = cp_dict.get(pair_key)
        if n_cp is not None:
            matched_pc.append(n_pc)
            matched_cp.append(n_cp)

    if matched_pc:
        ax.scatter(matched_pc, matched_cp, alpha=0.6, s=30,
                   color='#8E44AD', edgecolors='#5B2C6F', lw=0.5, zorder=3)
        all_vals = matched_pc + matched_cp
        mx = max(all_vals) * 1.05 if all_vals else 10
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.5, label='PC = CP')
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)

    ax.set_xlabel('N ESMOs  (PC, destabilisation)', fontsize=9)
    ax.set_ylabel('N ESMOs  (CP, recovery)' if oi == 0 else '', fontsize=9)
    ax.set_title(org, fontweight='bold')
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    if oi == 0:
        ax.legend(fontsize=8)

fig3.suptitle(
    'Matched PC vs CP ESMO counts per state pair  (intra-org)\n'
    'Points above the diagonal: recovery has more pathways than destabilisation for that pair',
    fontsize=FONT_SIZE, y=1.03)
fig3.tight_layout()
_save(fig3, 'fig3_pc_vs_cp_scatter')

print(f"\nAll outputs saved to: {_OUT_DIR}")
print("=" * 72)
