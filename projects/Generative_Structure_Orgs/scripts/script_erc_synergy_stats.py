#!/usr/bin/env python3
"""
script_erc_synergy_stats.py
============================
Batch-process a folder of reaction network .txt files, compute ERC synergy
statistics for each, save the results to a CSV, and plot scatter plots showing
how synergy counts scale with network size.

Statistics collected per network:
  n_species, n_reactions, n_ercs,
  n_basic, n_maximal, n_fundamental,
  ratio_basic    = n_basic    / C(n_ercs,2)   (fraction of pairs with basic synergy)
  ratio_maximal  = n_maximal  / C(n_ercs,2)
  ratio_fundamental = n_fundamental / C(n_ercs,2)
  time_ercs, time_synergies

Synergy detection uses the corrected implementations from script_erc_synergy_viz.py
(the library's get_maximal_synergies has an inverted containment check; see
the docstring in script_erc_synergy_viz.py for details).

Default folders scanned:
  1. data/biomodels/biomodels_interesting/
  2. networks/testing/performance_benchmark/
  3. networks/testing/   (other .txt files)
Add or change SCAN_FOLDERS below.

Networks that fail (parse error, timeout, too many ERCs) are skipped and logged.
"""

import os
import sys
import time
import traceback
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
    os.path.join(_PYCOT_ROOT, 'data', 'biomodels', 'biomodels_all_txt', 'Biomodels_txt_sample')
    #os.path.join(_PYCOT_ROOT, 'networks', 'testing', 'performance_benchmark'),
    #os.path.join(_PYCOT_ROOT, 'networks', 'testing'),
    #os.path.join(_PYCOT_ROOT, 'networks', 'FarmVariants'),
]

OUT_DIR  = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'outputs', 'synergy_stats'))
CSV_FILE = os.path.join(OUT_DIR, 'synergy_stats.csv')

MAX_ERCS = 200        # skip networks with more than this many ERCs (too slow)
MAX_TIME_ERCS = 120   # max seconds for ERC computation per network

os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# Corrected synergy functions (same logic as script_erc_synergy_viz.py)
# ===========================================================================

def _compute_basic(erc1, erc2, hierarchy, RN):
    """Basic synergies: (erc1,erc2)→T for each jointly-coverable-but-not-individually generator."""
    if (erc1 in hierarchy.get_contain(erc2) or
            erc2 in hierarchy.get_contain(erc1)):
        return []
    cl1   = erc1.get_closure_names(RN)
    cl2   = erc2.get_closure_names(RN)
    joint = cl1 | cl2
    contained_by_1 = {e.label for e in hierarchy.get_contain(erc1)}
    contained_by_2 = {e.label for e in hierarchy.get_contain(erc2)}

    result = []
    for target in hierarchy.ercs:
        if target is erc1 or target is erc2:
            continue
        if target.label in contained_by_1 or target.label in contained_by_2:
            continue
        for gen in target.min_generators:
            gen_sp = set(species_list_to_names(gen))
            if not gen_sp.issubset(joint):
                continue
            if gen_sp.issubset(cl1) or gen_sp.issubset(cl2):
                continue
            result.append((erc1, erc2, target))
            break
    return result


def _filter_maximal(basics, RN):
    """
    Keep only maximal synergies (Definition 19): for each pair (E1,E2), keep
    targets T such that no other target T' in the basic synergies for that pair
    has cl(T') ⊋ cl(T).
    """
    if not basics:
        return []
    by_pair = defaultdict(list)
    for e1, e2, target in basics:
        key = tuple(sorted([e1.label, e2.label]))
        by_pair[key].append((e1, e2, target))

    result = []
    for syns in by_pair.values():
        closures = {s[2].label: s[2].get_closure_names(RN) for s in syns}
        for e1, e2, T in syns:
            cl_T = closures[T.label]
            dominated = any(
                cl_T < closures[T2.label]     # T strictly smaller than T2
                for _, _, T2 in syns
                if T2.label != T.label
            )
            if not dominated:
                result.append((e1, e2, T))
    return result


def _filter_fundamental(maximals, hierarchy, RN):
    """
    Keep only fundamental synergies (Definition 20): maximal synergy (E1,E2)→T
    is fundamental iff no strictly smaller pair (E1'⊆E1, E2'⊆E2, at least one
    strict) has a maximal synergy to T.
    """
    if not maximals:
        return []

    label_to_erc = {erc.label: erc for erc in hierarchy.ercs}

    def _desc_plus_self(erc):
        if erc.label not in hierarchy.graph:
            return {erc.label}
        return {erc.label} | set(nx.descendants(hierarchy.graph, erc.label))

    desc_cache = {}
    for e1, e2, _ in maximals:
        for e in (e1, e2):
            if e.label not in desc_cache:
                desc_cache[e.label] = _desc_plus_self(e)

    result = []
    for e1, e2, target in maximals:
        is_fund = True
        d1, d2  = desc_cache[e1.label], desc_cache[e2.label]

        for l1 in d1:
            if not is_fund:
                break
            for l2 in d2:
                if l1 == l2:
                    continue
                orig = {tuple(sorted([e1.label, e2.label]))}
                if tuple(sorted([l1, l2])) in orig:
                    continue
                sub1 = label_to_erc.get(l1)
                sub2 = label_to_erc.get(l2)
                if sub1 is None or sub2 is None:
                    continue
                sub_bas = _compute_basic(sub1, sub2, hierarchy, RN)
                sub_max = _filter_maximal(sub_bas, RN)
                if any(s[2].label == target.label for s in sub_max):
                    is_fund = False
                    break

        if is_fund:
            result.append((e1, e2, target))
    return result


def compute_synergy_stats(ercs, hierarchy, RN):
    """
    Returns a dict with:
      _pairs   : unique (E1,E2) unordered pairs with ≥1 synergy  [≤ C(n,2)]
      _targets : unique ERC labels that appear as synergy targets  [≤ n_ercs]

    Ratios are computed in the caller as:
      pair_fraction    = n_*_pairs   / C(n_ercs, 2)
      target_fraction  = n_*_targets / n_ercs
    """
    b_pairs, m_pairs, f_pairs       = set(), set(), set()
    b_targets, m_targets, f_targets = set(), set(), set()

    for e1, e2 in combinations(ercs, 2):
        b = _compute_basic(e1, e2, hierarchy, RN)
        if not b:
            continue
        pair_key = tuple(sorted([e1.label, e2.label]))
        b_pairs.add(pair_key)
        b_targets.update(t.label for _, _, t in b)

        m = _filter_maximal(b, RN)
        if m:
            m_pairs.add(pair_key)
            m_targets.update(t.label for _, _, t in m)

        f = _filter_fundamental(m, hierarchy, RN)
        if f:
            f_pairs.add(pair_key)
            f_targets.update(t.label for _, _, t in f)

    return {
        'n_basic_pairs':         len(b_pairs),
        'n_maximal_pairs':       len(m_pairs),
        'n_fundamental_pairs':   len(f_pairs),
        'n_basic_targets':       len(b_targets),
        'n_maximal_targets':     len(m_targets),
        'n_fundamental_targets': len(f_targets),
    }


# ===========================================================================
# File discovery
# ===========================================================================

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


# ===========================================================================
# Main loop
# ===========================================================================

all_files = collect_files(SCAN_FOLDERS)
print(f"Found {len(all_files)} .txt files across {len(SCAN_FOLDERS)} folders.")

records = []
skipped = []

for idx, fpath in enumerate(all_files):
    fname = os.path.basename(fpath)
    print(f"\n[{idx+1}/{len(all_files)}] {fname}")

    try:
        RN = read_txt(fpath)
        n_sp = len(RN.species())
        n_rx = len(RN.reactions())
        print(f"  {n_sp} species, {n_rx} reactions")

        t0 = time.time()
        ercs = ERC.ERCs(RN)
        t_ercs = time.time() - t0

        # Filter out E_∅ (empty-closure ERC).
        # generators() always adds clos(∅) as the first entry; for networks
        # without inflow reactions this produces an ERC with an empty closure.
        # The paper (Def 14) notes E_∅ is "redundant to every generator" and
        # excluded from all generative analysis.  Counting it inflates n_ercs
        # above n_reactions for networks where every reaction has a unique support.
        ercs = [e for e in ercs if len(e.get_closure_names(RN)) > 0]
        n_ercs = len(ercs)
        print(f"  {n_ercs} ERCs (E_∅ excluded)  ({t_ercs:.1f}s)")

        if n_ercs > MAX_ERCS:
            print(f"  SKIP: too many ERCs ({n_ercs} > {MAX_ERCS})")
            skipped.append((fname, f'too many ERCs: {n_ercs}'))
            continue
        if t_ercs > MAX_TIME_ERCS:
            print(f"  SKIP: ERC computation too slow ({t_ercs:.1f}s)")
            skipped.append((fname, f'ERC timeout: {t_ercs:.1f}s'))
            continue

        hierarchy = ERC_Hierarchy(RN, ercs)

        t0  = time.time()
        syn = compute_synergy_stats(ercs, hierarchy, RN)
        t_syn = time.time() - t0
        print(f"  Pairs   — Basic={syn['n_basic_pairs']}  "
              f"Maximal={syn['n_maximal_pairs']}  "
              f"Fundamental={syn['n_fundamental_pairs']}  ({t_syn:.1f}s)")
        print(f"  Targets — Basic={syn['n_basic_targets']}  "
              f"Maximal={syn['n_maximal_targets']}  "
              f"Fundamental={syn['n_fundamental_targets']}")

        n_pairs = n_ercs * (n_ercs - 1) // 2   # C(n,2) = theoretical max
        records.append({
            'file':         fname,
            'n_species':    n_sp,
            'n_reactions':  n_rx,
            'n_ercs':       n_ercs,
            'n_pairs_max':  n_pairs,             # theoretical max = C(n_ercs,2)
            # Synergic pair counts: unique (E1,E2) with ≥1 synergy  [≤ n_pairs_max]
            'n_basic_pairs':         syn['n_basic_pairs'],
            'n_maximal_pairs':       syn['n_maximal_pairs'],
            'n_fundamental_pairs':   syn['n_fundamental_pairs'],
            # Unique target ERCs activated by synergies  [≤ n_ercs]
            'n_basic_targets':       syn['n_basic_targets'],
            'n_maximal_targets':     syn['n_maximal_targets'],
            'n_fundamental_targets': syn['n_fundamental_targets'],
            # Synergic pair fraction: n_*_pairs / C(n,2)  ∈ [0,1]
            'ratio_basic':       syn['n_basic_pairs']       / n_pairs if n_pairs > 0 else 0,
            'ratio_maximal':     syn['n_maximal_pairs']     / n_pairs if n_pairs > 0 else 0,
            'ratio_fundamental': syn['n_fundamental_pairs'] / n_pairs if n_pairs > 0 else 0,
            # Target fraction: n_*_targets / n_ercs  ∈ [0,1]
            'target_ratio_basic':       syn['n_basic_targets']       / n_ercs if n_ercs > 0 else 0,
            'target_ratio_maximal':     syn['n_maximal_targets']     / n_ercs if n_ercs > 0 else 0,
            'target_ratio_fundamental': syn['n_fundamental_targets'] / n_ercs if n_ercs > 0 else 0,
            'time_ercs':      round(t_ercs, 2),
            'time_synergies': round(t_syn,  2),
        })

    except Exception as exc:
        print(f"  ERROR: {exc}")
        skipped.append((fname, str(exc)))

# ===========================================================================
# Save CSV
# ===========================================================================

df = pd.DataFrame(records)
df.to_csv(CSV_FILE, index=False)
print(f"\nSaved {len(df)} records to {CSV_FILE}")
if skipped:
    print(f"Skipped {len(skipped)} networks:")
    for fn, reason in skipped:
        print(f"  {fn}: {reason}")

if df.empty:
    print("No data to plot.")
    sys.exit(0)

# ===========================================================================
# Plots  (two focused panels)
# ===========================================================================

SYN_STYLES = {
    'basic':       {'color': '#A8D8EA', 'label': 'Basic',       'marker': 'o', 'zorder': 2},
    'maximal':     {'color': '#E67E22', 'label': 'Maximal',     'marker': 's', 'zorder': 3},
    'fundamental': {'color': '#27AE60', 'label': 'Fundamental', 'marker': '^', 'zorder': 4},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---------------------------------------------------------------------------
# Plot 1 — Synergic pairs vs theoretical maximum C(n_ercs, 2)
#
# X: C(n_ercs, 2) = n*(n-1)/2  [theoretical max of unordered pairs]
# Y: n_*_pairs  [actual synergic pairs, always ≤ x]
# Diagonal y=x = 100 % saturation.  Points far below confirm synergies are rare.
# ---------------------------------------------------------------------------
x_max    = df['n_pairs_max'].values
diag_max = max(x_max.max(),
               df[['n_basic_pairs','n_maximal_pairs','n_fundamental_pairs']].values.max()) * 1.05

for key, sty in SYN_STYLES.items():
    y = df[f'n_{key}_pairs'].values
    ax1.scatter(x_max, y, color=sty['color'], marker=sty['marker'],
                s=55, alpha=0.80, edgecolors='white', lw=0.5,
                label=sty['label'], zorder=sty['zorder'])

ax1.plot([0, diag_max], [0, diag_max], 'k--', lw=0.9, alpha=0.35,
         label='y = x  (all pairs synergic)')
ax1.set_xlim(-diag_max * 0.02, diag_max)
ax1.set_ylim(-diag_max * 0.02, diag_max)
ax1.set_xlabel('C(|ERCs|, 2)  —  theoretical max unordered pairs', fontsize=11)
ax1.set_ylabel('Synergic pairs  (unique (E1, E2) with ≥ 1 synergy)', fontsize=11)
ax1.set_title(
    'Synergic pairs vs theoretical maximum\n'
    'Points far below diagonal → synergies are rare',
    fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Plot 2 — Synergetic target fraction vs ERC pairs reactants fraction
#
# X: ratio_* = n_*_pairs / C(n_ercs,2)  ∈ [0,1]
#    Fraction of all ERC pairs that ARE reactants in ≥1 synergy.
# Y: target_ratio_* = n_*_targets / n_ercs  ∈ [0,1]
#    Fraction of ERCs that appear as a synergy TARGET.
# Node size ∝ n_ercs so the network scale is visible.
# ---------------------------------------------------------------------------
n_ercs_vals = df['n_ercs'].values
# Map n_ercs → scatter size: min 30 pts², max 400 pts²
_n_min, _n_max = n_ercs_vals.min(), n_ercs_vals.max()
_n_range = max(_n_max - _n_min, 1)
node_s = 30 + 370 * (n_ercs_vals - _n_min) / _n_range

for key, sty in SYN_STYLES.items():
    xv = df[f'ratio_{key}'].values
    yv = df[f'target_ratio_{key}'].values
    ax2.scatter(xv, yv, s=node_s, color=sty['color'], marker=sty['marker'],
                alpha=0.75, edgecolors='#333333', lw=0.4,
                label=sty['label'], zorder=sty['zorder'])

ax2.axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
ax2.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(
    'ERC pairs reactants fraction  =  synergic pairs / C(|ERCs|, 2)',
    fontsize=11)
ax2.set_ylabel(
    'Synergetic target fraction  =  unique targets / |ERCs|',
    fontsize=11)
ax2.set_title(
    'Synergetic target fraction vs ERC pairs reactants fraction\n'
    'Node size ∝ |ERCs|  ·  both axes ∈ [0, 1]',
    fontsize=11, fontweight='bold')

# Size legend: show 3 representative n_ercs values
from matplotlib.lines import Line2D as _L2D
_ticks = [_n_min, (_n_min + _n_max) // 2, _n_max]
size_handles = [
    ax2.scatter([], [], s=30 + 370 * (n - _n_min) / _n_range,
                color='#888888', marker='o', alpha=0.7,
                edgecolors='#333333', lw=0.4, label=f'|ERCs| = {n}')
    for n in _ticks
]
# Merge synergy-type legend + size legend
type_handles = [ax2.scatter([], [], s=80, color=sty['color'],
                             marker=sty['marker'], alpha=0.9, label=sty['label'])
                for sty in SYN_STYLES.values()]
ax2.legend(handles=type_handles + size_handles,
           fontsize=8, loc='upper left',
           title='Type  /  scale', title_fontsize=8)
ax2.grid(True, alpha=0.25)

fig.suptitle(
    f'ERC synergy statistics  —  {len(df)} networks  '
    f'(E_∅ excluded,  max_ercs = {MAX_ERCS})',
    fontsize=12, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(OUT_DIR, 'synergy_stats.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {fig_path}")
plt.show()

# ===========================================================================
# Summary table
# ===========================================================================
print("\n=== Summary ===")
print(df[['file', 'n_species', 'n_reactions', 'n_ercs',
          'n_basic_pairs', 'n_maximal_pairs', 'n_fundamental_pairs',
          'ratio_basic', 'ratio_maximal', 'ratio_fundamental',
          'target_ratio_basic', 'target_ratio_maximal', 'target_ratio_fundamental',
         ]].to_string(index=False))
