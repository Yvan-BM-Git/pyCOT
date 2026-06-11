#!/usr/bin/env python3
"""
script_erc_complementarity_stats.py
=====================================
Batch-process reaction networks, compute ERC complementarity statistics,
save results to CSV, and produce scatter plots analogous to
script_erc_synergy_stats.py.

Statistics collected per network
---------------------------------
  n_species, n_reactions, n_ercs, n_pairs_max
  n_complementary_pairs      – unordered (E1,E2) pairs with supl(E1,E2)∪supl(E2,E1) ≠ ∅
  n_pure_complementary_pairs – complementary pairs that are NOT synergetic
  n_fundamental_edges        – unique unordered {E_prod,E_cons} in any fundamental comp.
  n_producer_ERCs            – distinct ERCs acting as minimal producers in ≥1 fund. comp.
  n_consumer_ERCs            – distinct ERCs acting as minimal consumers in ≥1 fund. comp.
  ratio_complementary        = n_complementary_pairs / C(n_ercs,2)
  ratio_pure                 = n_pure_complementary_pairs / C(n_ercs,2)
  ratio_fundamental          = n_fundamental_edges / C(n_ercs,2)
  time_ercs, time_complementarity

Definitions (Section 5 of the paper)
--------------------------------------
  supl(E, E')  = prod(R_E) ∩ req(E')           (supply from E to E')
  req(E)       = supp(R_E) \\ prod(R_E)          (species required by E)
  complementary pair: incomparable (E,E') with supl(E,E')∪supl(E',E) ≠ ∅
  purely complementary: complementary AND NOT synergetic
  minprod(s): inclusion-minimal ERCs producing s
  mincons(s): inclusion-minimal ERCs requiring s
  fundamental complementarity E ⟺[s] E': E ∈ minprod(s), E' ∈ mincons(s)
"""

import os
import sys
import time
from itertools import combinations
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx

# -- Path setup ----------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PYCOT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(_PYCOT_ROOT, 'src'))

from pyCOT.io.functions import read_txt
from pyCOT.analysis.ERC_Hierarchy import ERC, ERC_Hierarchy, species_list_to_names

# -- Configuration -------------------------------------------------------------
SCAN_FOLDERS = [
    os.path.join(_PYCOT_ROOT, 'data', 'biomodels', 'biomodels_all_txt', 'Biomodels_txt_sample'),
    # os.path.join(_PYCOT_ROOT, 'networks', 'testing', 'performance_benchmark'),
    # os.path.join(_PYCOT_ROOT, 'networks', 'testing'),
]

OUT_DIR  = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'outputs', 'complementarity_stats'))
CSV_FILE = os.path.join(OUT_DIR, 'complementarity_stats.csv')

MAX_ERCS      = 200    # skip networks with more ERCs than this
MAX_TIME_ERCS = 120    # max seconds for ERC computation

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Synergy helper (lightweight: returns True/False for a pair)
# Used only to classify "purely complementary" pairs.
# =============================================================================

def _has_basic_synergy(erc1, erc2, hierarchy, RN):
    """Return True iff (erc1,erc2) has at least one basic synergy."""
    if (erc1 in hierarchy.get_contain(erc2) or
            erc2 in hierarchy.get_contain(erc1)):
        return False
    cl1   = erc1.get_closure_names(RN)
    cl2   = erc2.get_closure_names(RN)
    joint = cl1 | cl2
    sub1  = {e.label for e in hierarchy.get_contain(erc1)}
    sub2  = {e.label for e in hierarchy.get_contain(erc2)}
    for target in hierarchy.ercs:
        if target is erc1 or target is erc2:
            continue
        if target.label in sub1 or target.label in sub2:
            continue
        for gen in target.min_generators:
            gen_sp = set(species_list_to_names(gen))
            if (gen_sp.issubset(joint) and
                    not gen_sp.issubset(cl1) and
                    not gen_sp.issubset(cl2)):
                return True
    return False


# =============================================================================
# Supply and complementarity helpers
# =============================================================================

def _is_incomparable(erc1, erc2, hierarchy):
    return (erc1 not in hierarchy.get_contain(erc2) and
            erc2 not in hierarchy.get_contain(erc1))


def _supply(erc_prod, erc_cons, RN):
    """supl(erc_prod, erc_cons) = prod(R_{erc_prod}) ∩ req(erc_cons)."""
    return erc_prod.get_produced_species(RN) & erc_cons.get_required_species(RN)


def _compute_minprod_mincons(ercs, hierarchy, RN):
    """
    For each species s, identify:
      minprod[s]: inclusion-minimal ERCs producing s
      mincons[s]: inclusion-minimal ERCs requiring s

    An ERC E is a minimal producer of s iff no descendant of E in the
    hierarchy also produces s (equivalently, no E' ⊊ E produces s).
    """
    producers = defaultdict(set)   # species_name → set of ERC labels
    consumers = defaultdict(set)

    for erc in ercs:
        for s in erc.get_produced_species(RN):
            producers[s].add(erc.label)
        for s in erc.get_required_species(RN):
            consumers[s].add(erc.label)

    label_to_erc = {e.label: e for e in ercs}

    def _minimal_ercs(label_set):
        """Return the ERCs in label_set with no strict subset also in label_set."""
        minimal = []
        for lbl in label_set:
            erc = label_to_erc[lbl]
            # descendants = ERCs strictly contained within erc
            descendants = {e.label for e in hierarchy.get_contain(erc)}
            if not descendants.intersection(label_set):
                minimal.append(erc)
        return minimal

    minprod = {s: _minimal_ercs(lbls) for s, lbls in producers.items()}
    mincons = {s: _minimal_ercs(lbls) for s, lbls in consumers.items()}
    return minprod, mincons


def compute_complementarity_stats(ercs, hierarchy, RN):
    """
    Compute all complementarity statistics for the given ERC collection.

    Returns a dict with keys:
      n_complementary_pairs, n_pure_complementary_pairs,
      n_fundamental_edges, n_producer_ERCs, n_consumer_ERCs
    """
    minprod, mincons = _compute_minprod_mincons(ercs, hierarchy, RN)

    comp_pairs   = set()   # frozenset pair keys for complementary pairs
    pure_pairs   = set()   # subset of comp_pairs that are not synergetic
    fund_pairs   = set()   # frozenset pair keys for fundamental edges
    prod_ercs    = set()   # ERC labels acting as fundamental producers
    cons_ercs    = set()   # ERC labels acting as fundamental consumers

    for e1, e2 in combinations(ercs, 2):
        if not _is_incomparable(e1, e2, hierarchy):
            continue

        s12 = _supply(e1, e2, RN)   # e1 → e2
        s21 = _supply(e2, e1, RN)   # e2 → e1

        if not s12 and not s21:
            continue

        pair_key = frozenset([e1.label, e2.label])
        comp_pairs.add(pair_key)

        # Purely complementary: complementary and NOT synergetic
        if not _has_basic_synergy(e1, e2, hierarchy, RN):
            pure_pairs.add(pair_key)

        # Fundamental complementarities: E_prod ∈ minprod(s), E_cons ∈ mincons(s)
        for s in s12:
            if e1 in minprod.get(s, []) and e2 in mincons.get(s, []):
                fund_pairs.add(pair_key)
                prod_ercs.add(e1.label)
                cons_ercs.add(e2.label)
        for s in s21:
            if e2 in minprod.get(s, []) and e1 in mincons.get(s, []):
                fund_pairs.add(pair_key)
                prod_ercs.add(e2.label)
                cons_ercs.add(e1.label)

    return {
        'n_complementary_pairs':      len(comp_pairs),
        'n_pure_complementary_pairs': len(pure_pairs),
        'n_fundamental_edges':        len(fund_pairs),
        'n_producer_ERCs':            len(prod_ercs),
        'n_consumer_ERCs':            len(cons_ercs),
    }


# =============================================================================
# File discovery
# =============================================================================

def collect_files(folders):
    seen, files = set(), []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(folder, fname)
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                files.append(path)
    return files


# =============================================================================
# Main processing loop
# =============================================================================

all_files = collect_files(SCAN_FOLDERS)
print(f"Found {len(all_files)} .txt files across {len(SCAN_FOLDERS)} folder(s).")

records = []
skipped = []

for idx, fpath in enumerate(all_files):
    fname = os.path.basename(fpath)
    print(f"\n[{idx+1}/{len(all_files)}] {fname}")

    try:
        RN   = read_txt(fpath)
        n_sp = len(RN.species())
        n_rx = len(RN.reactions())
        print(f"  {n_sp} species, {n_rx} reactions")

        t0    = time.time()
        ercs  = ERC.ERCs(RN)
        t_erc = time.time() - t0

        # Exclude E_∅ (empty closure)
        ercs  = [e for e in ercs if len(e.get_closure_names(RN)) > 0]
        n_ercs = len(ercs)
        print(f"  {n_ercs} ERCs (E_∅ excluded)  ({t_erc:.1f}s)")

        if n_ercs > MAX_ERCS:
            print(f"  SKIP: too many ERCs ({n_ercs} > {MAX_ERCS})")
            skipped.append((fname, f'too many ERCs: {n_ercs}'))
            continue
        if t_erc > MAX_TIME_ERCS:
            print(f"  SKIP: ERC computation too slow ({t_erc:.1f}s)")
            skipped.append((fname, f'ERC timeout: {t_erc:.1f}s'))
            continue

        hierarchy = ERC_Hierarchy(RN, ercs)

        t0  = time.time()
        cst = compute_complementarity_stats(ercs, hierarchy, RN)
        t_c = time.time() - t0

        print(f"  Complementary={cst['n_complementary_pairs']}  "
              f"Pure={cst['n_pure_complementary_pairs']}  "
              f"Fundamental={cst['n_fundamental_edges']}  ({t_c:.1f}s)")
        print(f"  Producer ERCs={cst['n_producer_ERCs']}  "
              f"Consumer ERCs={cst['n_consumer_ERCs']}")

        n_pairs = n_ercs * (n_ercs - 1) // 2   # C(n,2)
        records.append({
            'file':       fname,
            'n_species':  n_sp,
            'n_reactions': n_rx,
            'n_ercs':     n_ercs,
            'n_pairs_max': n_pairs,
            'n_complementary_pairs':      cst['n_complementary_pairs'],
            'n_pure_complementary_pairs': cst['n_pure_complementary_pairs'],
            'n_fundamental_edges':        cst['n_fundamental_edges'],
            'n_producer_ERCs':            cst['n_producer_ERCs'],
            'n_consumer_ERCs':            cst['n_consumer_ERCs'],
            'ratio_complementary': (cst['n_complementary_pairs']      / n_pairs if n_pairs > 0 else 0),
            'ratio_pure':          (cst['n_pure_complementary_pairs'] / n_pairs if n_pairs > 0 else 0),
            'ratio_fundamental':   (cst['n_fundamental_edges']        / n_pairs if n_pairs > 0 else 0),
            'ratio_producers':     (cst['n_producer_ERCs'] / n_ercs if n_ercs > 0 else 0),
            'ratio_consumers':     (cst['n_consumer_ERCs'] / n_ercs if n_ercs > 0 else 0),
            'time_ercs':            round(t_erc, 2),
            'time_complementarity': round(t_c,   2),
        })

    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        skipped.append((fname, str(exc)))

# =============================================================================
# Save CSV
# =============================================================================

df = pd.DataFrame(records)
df.to_csv(CSV_FILE, index=False)
print(f"\nSaved {len(df)} records to {CSV_FILE}")
if skipped:
    print(f"Skipped {len(skipped)} networks:")
    for fn, reason in skipped:
        print(f"  {fn}: {reason}")

if df.empty:
    print("No data to plot.")
    import sys; sys.exit(0)

# =============================================================================
# Plots
# =============================================================================

COMP_STYLES = {
    'complementary': {'color': '#A8D8EA', 'label': 'Complementary',       'marker': 'o', 'zorder': 2},
    'pure':          {'color': '#E67E22', 'label': 'Purely complementary', 'marker': 's', 'zorder': 3},
    'fundamental':   {'color': '#27AE60', 'label': 'Fundamental',          'marker': '^', 'zorder': 4},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---------------------------------------------------------------------------
# Plot 1 — Complementary pair counts vs theoretical maximum C(n_ercs, 2)
# ---------------------------------------------------------------------------
x_max    = df['n_pairs_max'].values
y_comp   = df['n_complementary_pairs'].values
y_pure   = df['n_pure_complementary_pairs'].values
y_fund   = df['n_fundamental_edges'].values

diag_max = max(
    x_max.max(),
    max(y_comp.max(), y_pure.max(), y_fund.max())
) * 1.05

ax1.scatter(x_max, y_comp,
            color=COMP_STYLES['complementary']['color'],
            marker=COMP_STYLES['complementary']['marker'],
            s=55, alpha=0.80, edgecolors='white', lw=0.5,
            label=COMP_STYLES['complementary']['label'],
            zorder=COMP_STYLES['complementary']['zorder'])
ax1.scatter(x_max, y_pure,
            color=COMP_STYLES['pure']['color'],
            marker=COMP_STYLES['pure']['marker'],
            s=55, alpha=0.80, edgecolors='white', lw=0.5,
            label=COMP_STYLES['pure']['label'],
            zorder=COMP_STYLES['pure']['zorder'])
ax1.scatter(x_max, y_fund,
            color=COMP_STYLES['fundamental']['color'],
            marker=COMP_STYLES['fundamental']['marker'],
            s=55, alpha=0.80, edgecolors='white', lw=0.5,
            label=COMP_STYLES['fundamental']['label'],
            zorder=COMP_STYLES['fundamental']['zorder'])

ax1.plot([0, diag_max], [0, diag_max], 'k--', lw=0.9, alpha=0.35,
         label='y = x  (all pairs complementary)')
ax1.set_xlim(-diag_max * 0.02, diag_max)
ax1.set_ylim(-diag_max * 0.02, diag_max)
ax1.set_xlabel('C(|ERCs|, 2)  —  theoretical max unordered pairs', fontsize=11)
ax1.set_ylabel('Complementary pairs', fontsize=11)
ax1.set_title(
    'Complementary pairs vs theoretical maximum\n'
    'Points far below diagonal → complementarities are rare',
    fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Plot 2 — Producer/consumer ERC fractions vs complementary pair fraction
#
# X: ratio_fundamental  ∈ [0,1]
# Y: ratio_producers = n_producer_ERCs / n_ercs  (fraction of ERCs as producers)
#    ratio_consumers = n_consumer_ERCs / n_ercs  (fraction of ERCs as consumers)
# Node size ∝ n_ercs
# ---------------------------------------------------------------------------
n_ercs_vals = df['n_ercs'].values
_n_min, _n_max = n_ercs_vals.min(), n_ercs_vals.max()
_n_range = max(_n_max - _n_min, 1)
node_s = 30 + 370 * (n_ercs_vals - _n_min) / _n_range

xv = df['ratio_fundamental'].values

ax2.scatter(xv, df['ratio_producers'].values,
            s=node_s,
            color='#27AE60', marker='^',
            alpha=0.75, edgecolors='#333333', lw=0.4,
            label='Producer ERCs / |ERCs|',
            zorder=4)
ax2.scatter(xv, df['ratio_consumers'].values,
            s=node_s,
            color='#2980B9', marker='v',
            alpha=0.75, edgecolors='#333333', lw=0.4,
            label='Consumer ERCs / |ERCs|',
            zorder=3)

ax2.axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
ax2.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(
    'Fundamental complementarity fraction  =  fund. edges / C(|ERCs|, 2)',
    fontsize=11)
ax2.set_ylabel(
    'Fraction of ERCs  (producers or consumers in fundamental pairs)',
    fontsize=11)
ax2.set_title(
    'Producer/consumer ERC fractions vs fundamental complementarity fraction\n'
    'Node size ∝ |ERCs|  ·  both axes ∈ [0, 1]',
    fontsize=11, fontweight='bold')

# Size legend
_ticks = [_n_min, (_n_min + _n_max) // 2, _n_max]
size_handles = [
    ax2.scatter([], [], s=30 + 370 * (n - _n_min) / _n_range,
                color='#888888', marker='o', alpha=0.7,
                edgecolors='#333333', lw=0.4, label=f'|ERCs| = {n}')
    for n in _ticks
]
type_handles = [
    ax2.scatter([], [], s=80, color='#27AE60', marker='^', alpha=0.9,
                label='Producer ERCs / |ERCs|'),
    ax2.scatter([], [], s=80, color='#2980B9', marker='v', alpha=0.9,
                label='Consumer ERCs / |ERCs|'),
]
ax2.legend(handles=type_handles + size_handles,
           fontsize=8, loc='upper left',
           title='Role  /  scale', title_fontsize=8)
ax2.grid(True, alpha=0.25)

fig.suptitle(
    f'ERC complementarity statistics  —  {len(df)} networks  '
    f'(E_∅ excluded,  max_ercs = {MAX_ERCS})',
    fontsize=12, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(OUT_DIR, 'complementarity_stats.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {fig_path}")
plt.show()

# =============================================================================
# Summary table
# =============================================================================
print("\n=== Summary ===")
print(df[['file', 'n_species', 'n_reactions', 'n_ercs',
          'n_complementary_pairs', 'n_pure_complementary_pairs', 'n_fundamental_edges',
          'ratio_complementary', 'ratio_pure', 'ratio_fundamental',
          'ratio_producers', 'ratio_consumers',
         ]].to_string(index=False))
