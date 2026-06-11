#!/usr/bin/env python3
"""
plot_inter_org_figures.py
=========================
Regenerate inter-org ESMO figures directly from the cached summary CSV.
No LP computation required.

Figures produced
----------------
fig_inter1_heatmaps.png       — 2×2 ESMO count matrices per cover edge
fig_inter2_asymmetry.png      — up/down asymmetry ratios per edge and combo
fig_inter3_struct_entropy.png — N_in, N_out, and structural entropy ΔH per org
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'complex_COT_ESMO')
SUMMARY_FILE = os.path.join(OUT_DIR, 'transition_summary.csv')

COVER_EDGES = [
    ('Tribe', 'Chief'),
    ('Tribe', 'State'),
    ('Chief', 'ChiefState'),
    ('State', 'ChiefState'),
]
EDGE_LABELS = {
    ('Tribe',  'Chief'):      'Tribe–Chief',
    ('Tribe',  'State'):      'Tribe–State',
    ('Chief',  'ChiefState'): 'Chief–CfState',
    ('State',  'ChiefState'): 'State–CfState',
}
ORG_NAMES = ['Tribe', 'Chief', 'State', 'ChiefState']
ORG_LABELS = {'Tribe': 'Tribe', 'Chief': 'Chief',
              'State': 'State', 'ChiefState': 'ChiefState'}

COMBO_COLORS = {
    'pp': '#27AE60',
    'pc': '#E67E22',
    'cp': '#2980B9',
    'cc': '#E74C3C',
}
COMBO_LABELS = {
    'pp': 'PP  peace→peace',
    'pc': 'PC  peace→conflict',
    'cp': 'CP  conflict→peace',
    'cc': 'CC  conflict→conflict',
}
FONT_SIZE = 11
# ---------------------------------------------------------------------------

df = pd.read_csv(SUMMARY_FILE)
print(f"Loaded {len(df)} rows from {os.path.realpath(SUMMARY_FILE)}")

def _n(edge_str, direction, combo):
    row = df[(df['edge'] == edge_str) & (df['direction'] == direction) &
             (df['combo'] == combo)]
    if len(row) == 0:
        return 0
    return int(row['n_esmos'].values[0])

# Pre-compute total N per source node for normalized probabilities (P sums to 1).
# Each node sends along exactly 2 outgoing cover edges × 2 combos = 4 transitions.
_ORG_ABBR = {'Tribe': 'Tr', 'Chief': 'Ch', 'State': 'St', 'ChiefState': 'CS'}
_SRC_EDGES = {
    'TrP': [('Tribe-Chief', 'up', 'pp'), ('Tribe-Chief', 'up', 'pc'),
            ('Tribe-State', 'up', 'pp'), ('Tribe-State', 'up', 'pc')],
    'TrC': [('Tribe-Chief', 'up', 'cp'), ('Tribe-Chief', 'up', 'cc'),
            ('Tribe-State', 'up', 'cp'), ('Tribe-State', 'up', 'cc')],
    'ChP': [('Tribe-Chief', 'down', 'pp'), ('Tribe-Chief', 'down', 'pc'),
            ('Chief-ChiefState', 'up', 'pp'), ('Chief-ChiefState', 'up', 'pc')],
    'ChC': [('Tribe-Chief', 'down', 'cp'), ('Tribe-Chief', 'down', 'cc'),
            ('Chief-ChiefState', 'up', 'cp'), ('Chief-ChiefState', 'up', 'cc')],
    'StP': [('Tribe-State', 'down', 'pp'), ('Tribe-State', 'down', 'pc'),
            ('State-ChiefState', 'up', 'pp'), ('State-ChiefState', 'up', 'pc')],
    'StC': [('Tribe-State', 'down', 'cp'), ('Tribe-State', 'down', 'cc'),
            ('State-ChiefState', 'up', 'cp'), ('State-ChiefState', 'up', 'cc')],
    'CSP': [('Chief-ChiefState', 'down', 'pp'), ('Chief-ChiefState', 'down', 'pc'),
            ('State-ChiefState', 'down', 'pp'), ('State-ChiefState', 'down', 'pc')],
    'CSC': [('Chief-ChiefState', 'down', 'cp'), ('Chief-ChiefState', 'down', 'cc'),
            ('State-ChiefState', 'down', 'cp'), ('State-ChiefState', 'down', 'cc')],
}
_NODE_TOTAL = {node: sum(_n(e, d, c) for e, d, c in edges)
               for node, edges in _SRC_EDGES.items()}

def _p(edge_str, direction, combo, down_org, up_org):
    """Normalized transition probability: N / total_N_from_source_node."""
    src_org  = down_org if direction == 'up' else up_org
    src_state = 'P' if combo[0] == 'p' else 'C'
    src_node  = _ORG_ABBR[src_org] + src_state
    n = _n(edge_str, direction, combo)
    return n / _NODE_TOTAL[src_node] if _NODE_TOTAL[src_node] > 0 else 0.0

def _save(fig, stem):
    p = os.path.join(os.path.realpath(OUT_DIR), f'{stem}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close(fig)

plt.rcParams.update({
    'font.size':       FONT_SIZE,
    'axes.titlesize':  FONT_SIZE,
    'axes.labelsize':  FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
})

# ===========================================================================
# Fig 1: Cover-edge ESMO heatmaps
#   Layout: 4 rows (one per edge) × 2 cols (up / down)
#   Each cell: 2×2 inner grid (source: peace/conflict) × (target: peace/conflict)
# ===========================================================================
COMBOS_2x2 = [('pp', 'pc'), ('cp', 'cc')]   # row=source, col=target

fig1, axes1 = plt.subplots(4, 2, figsize=(9, 14), squeeze=False)
STATE_ROW = ['peace', 'conflict']
STATE_COL = ['peace', 'conflict']

global_max = max(df['n_esmos'])

for ei, (down, up) in enumerate(COVER_EDGES):
    edge_str = f'{down}-{up}'
    for di, (direction, org_from, org_to, dir_sym) in enumerate([
            ('up',   down, up,   '↑'),
            ('down', up,   down, '↓')]):
        ax = axes1[ei][di]
        mat = np.zeros((2, 2))
        for ri, sf in enumerate(STATE_ROW):
            for ci, st in enumerate(STATE_COL):
                combo = sf[0] + st[0]
                mat[ri, ci] = _n(edge_str, direction, combo)

        vmax = max(float(mat.max()), 1.0)
        im = ax.imshow(mat, aspect='equal', cmap='YlOrRd',
                       vmin=0, vmax=global_max, interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['→ peace', '→ conflict'], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['peace ↓', 'conflict ↓'], fontsize=9)
        ax.set_title(f'{org_from} {dir_sym} {org_to}',
                     fontsize=10, fontweight='bold')

        for ri in range(2):
            for ci in range(2):
                val = int(mat[ri, ci])
                txt_col = 'white' if mat[ri, ci] > global_max * 0.55 else 'black'
                ax.text(ci, ri, str(val), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=txt_col)

        cb = plt.colorbar(im, ax=ax, shrink=0.80)
        cb.ax.tick_params(labelsize=8)

fig1.suptitle(
    'ESMO counts for all 32 directed inter-organizational transitions\n'
    'Rows = source state (peace / conflict),  Columns = target state\n'
    'Left: going UP the lattice (→ larger org)   Right: going DOWN (→ smaller org)\n'
    'Color intensity encodes ESMO count on a common scale across all panels',
    fontsize=FONT_SIZE, y=1.01)
fig1.tight_layout()
_save(fig1, 'fig_inter1_heatmaps')


# ===========================================================================
# Fig 2: ESMO counts by combo type + conditional relative frequencies
#
# RELATIVE FREQUENCY DEFINITION (inter-org):
#   For a given (edge, direction), the source state type is known (peace or
#   conflict).  The relative frequency f conditions on that source type and
#   asks: "what fraction of structural pathways lead to each target state?"
#
#     f_PP = N_PP / (N_PP + N_PC)   — from peace: fraction ending in peace
#     f_PC = N_PC / (N_PP + N_PC)   — from peace: fraction ending in conflict
#     f_CP = N_CP / (N_CP + N_CC)   — from conflict: fraction ending in peace
#     f_CC = N_CC / (N_CP + N_CC)   — from conflict: fraction ending in conflict
#
#   PP + PC sum to 1 (given peace source).
#   CP + CC sum to 1 (given conflict source).
#   This is the structural analogue of a Markov transition probability.
# ===========================================================================
F2_FONT  = 13   # base font size for this figure
F2_ANNOT = 9    # annotation font size

n_edges = len(COVER_EDGES)
x       = np.arange(n_edges)
W       = 0.20
offsets = [-1.5*W, -0.5*W, 0.5*W, 1.5*W]
combos  = ['pp', 'pc', 'cp', 'cc']

fig2, axes2 = plt.subplots(1, 2, figsize=(17, 8), sharey=True)

for ax2, direction, dir_label in [
        (axes2[0], 'up',   'Going UP  (→ larger org)'),
        (axes2[1], 'down', 'Going DOWN  (→ smaller org)')]:

    # X-axis labels reflect the actual source→target direction
    if direction == 'up':
        xlabels = [f'{d}–{u}' for d, u in COVER_EDGES]
    else:
        xlabels = [f'{u}–{d}' for d, u in COVER_EDGES]

    # Shade alternating edge groups for readability
    for ei in range(n_edges):
        if ei % 2 == 0:
            ax2.axvspan(ei - 0.5, ei + 0.5, color='#F0F0F0', zorder=0)

    # Vertical separator between the peace-source (PP,PC) and conflict-source
    # (CP,CC) bars within each edge group
    for ei in range(n_edges):
        ax2.axvline(ei, color='#AAAAAA', lw=0.8, ls=':', zorder=2)

    for ci, (combo, offset) in enumerate(zip(combos, offsets)):
        vals = [_n(f'{d}-{u}', direction, combo) for d, u in COVER_EDGES]
        bars = ax2.bar(x + offset, vals, W,
                       color=COMBO_COLORS[combo], alpha=0.85,
                       label=COMBO_LABELS[combo], zorder=3,
                       edgecolor='white', linewidth=0.5)

        for ei, (bar, v, (down, up)) in enumerate(zip(bars, vals, COVER_EDGES)):
            if v == 0:
                continue
            edge_str = f'{down}-{up}'

            # Normalized probability: N / total_N_from_source_node (sums to 1)
            p = _p(edge_str, direction, combo, down, up)

            # Two-line annotation: absolute count + normalized probability
            bx = bar.get_x() + bar.get_width() / 2
            ax2.text(bx, v + 18,
                     f'{v}\n({p:.2f})',
                     ha='center', va='bottom',
                     fontsize=F2_ANNOT, fontweight='bold',
                     linespacing=1.25, color='#1a1a1a')

    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=10, ha='right', fontsize=F2_FONT)
    ax2.set_ylabel('ESMO count  N', fontsize=F2_FONT)
    ax2.set_title(dir_label, fontweight='bold', fontsize=F2_FONT + 1)
    ax2.tick_params(axis='y', labelsize=F2_FONT - 1)
    ax2.legend(fontsize=F2_FONT - 2, loc='upper left',
               title='State combination', title_fontsize=F2_FONT - 2)
    ax2.grid(axis='y', alpha=0.35)
    ax2.set_axisbelow(True)

fig2.axes[0].set_ylim(0, max(df['n_esmos']) * 1.38)

fig2.suptitle(
    'ESMO counts and normalized transition probabilities — inter-organizational transitions\n'
    'Bar height = ESMO count N.  Annotation: N above, (P) below = normalized probability.\n'
    'P = N / total_N_from_source_node  (all 4 outgoing transitions per node sum to 1)\n'
    'Consistent with the Markov hierarchy graph.',
    fontsize=F2_FONT - 1, y=1.02)
fig2.tight_layout()
_save(fig2, 'fig_inter2_combo_counts')


# ===========================================================================
# Fig 3: Structural entropy analysis
#   Top panel: N_in and N_out per org (grouped bars)
#   Bottom panel: ΔH per org (signed bar, centripetal=green, centrifugal=red)
# ===========================================================================

# Compute N_in and N_out per org
incoming = {org: [] for org in ORG_NAMES}
outgoing  = {org: [] for org in ORG_NAMES}

for _, row in df.iterrows():
    incoming[row['org_to']].append(row['n_esmos'])
    outgoing[row['org_from']].append(row['n_esmos'])

N_in  = {org: np.mean(incoming[org]) for org in ORG_NAMES}
N_out = {org: np.mean(outgoing[org])  for org in ORG_NAMES}
dH    = {org: np.log(N_in[org]) - np.log(N_out[org]) for org in ORG_NAMES}

print("\nStructural entropy summary:")
for org in ORG_NAMES:
    print(f"  {org:12s}  N_in={N_in[org]:.1f}  N_out={N_out[org]:.1f}  dH={dH[org]:+.3f}")

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

x3 = np.arange(len(ORG_NAMES))
ORG_DISPLAY = ['Tribe', 'Chief', 'State', 'ChiefState']
W3 = 0.35

# Panel A: N_in and N_out
bars_in  = ax3a.bar(x3 - W3/2, [N_in[o]  for o in ORG_DISPLAY], W3,
                    color='#2980B9', alpha=0.85, label=r'$\overline{N}_{\rm in}$', zorder=3)
bars_out = ax3a.bar(x3 + W3/2, [N_out[o] for o in ORG_DISPLAY], W3,
                    color='#E67E22', alpha=0.85, label=r'$\overline{N}_{\rm out}$', zorder=3)

for bars, vals in [(bars_in, [N_in[o] for o in ORG_DISPLAY]),
                   (bars_out, [N_out[o] for o in ORG_DISPLAY])]:
    for bar, v in zip(bars, vals):
        ax3a.text(bar.get_x() + bar.get_width()/2, v + 10,
                  f'{v:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3a.set_ylabel(r'Mean ESMO count  $\overline{N}$')
ax3a.set_title('Mean structural accessibility: incoming vs outgoing transitions',
               fontweight='bold')
ax3a.legend(fontsize=10)
ax3a.grid(axis='y', alpha=0.3)
ax3a.set_axisbelow(True)
ax3a.set_ylim(0, max(max(N_in.values()), max(N_out.values())) * 1.18)

# Panel B: ΔH signed bar chart
dH_vals = [dH[o] for o in ORG_DISPLAY]
bar_colors = ['#27AE60' if v >= 0 else '#E74C3C' for v in dH_vals]
ax3b.bar(x3, dH_vals, 0.55, color=bar_colors, alpha=0.85, zorder=3)
ax3b.axhline(0, color='black', lw=1.0, ls='--', zorder=2)
ax3b.set_xticks(x3)
ax3b.set_xticklabels(ORG_DISPLAY, fontsize=11)
ax3b.set_ylabel(r'$\Delta H = \ln\overline{N}_{\rm in} - \ln\overline{N}_{\rm out}$')
ax3b.set_title(
    r'Structural entropy index  $\Delta H_{\mathcal{O}}$'
    '\n(green > 0: centripetal attractor;  red < 0: centrifugal transient)',
    fontweight='bold')
ax3b.grid(axis='y', alpha=0.3)
ax3b.set_axisbelow(True)

for xi, (org, v) in enumerate(zip(ORG_DISPLAY, dH_vals)):
    va = 'bottom' if v >= 0 else 'top'
    offset = 0.005 if v >= 0 else -0.005
    ax3b.text(xi, v + offset, f'{v:+.3f}',
              ha='center', va=va, fontsize=10, fontweight='bold',
              color='#27AE60' if v >= 0 else '#E74C3C')

fig3.suptitle(
    'Structural entropy of organizational regimes  (inter-organizational transitions)\n'
    r'$\Delta H > 0$: incoming transitions richer → structural attractor'
    r'    $\Delta H < 0$: outgoing transitions richer → structural transient',
    fontsize=FONT_SIZE, y=1.02)
fig3.tight_layout()
_save(fig3, 'fig_inter3_struct_entropy')


# ===========================================================================
# Fig 4: Directed transition graph — 2×2 panels, one per combo type
# (clean standalone version with larger fonts)
# ===========================================================================
_NODE_POS = {
    'Tribe':      (0.50, 0.08),
    'Chief':      (0.18, 0.55),
    'State':      (0.82, 0.55),
    'ChiefState': (0.50, 0.92),
}
_NODE_COL = {
    'Tribe':      '#F9E79F',
    'Chief':      '#AED6F1',
    'State':      '#A9DFBF',
    'ChiefState': '#D7BDE2',
}
_NODE_ABBR = {
    'Tribe':      'Tribe',
    'Chief':      'Chief',
    'State':      'State',
    'ChiefState': 'CfState',
}

all_n = df['n_esmos'].values
MAX_N = float(all_n.max())

fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)

for ci, combo in enumerate(['pp', 'pc', 'cp', 'cc']):
    ax = axes4[ci // 2][ci % 2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(COMBO_LABELS[combo], fontsize=11, fontweight='bold',
                 color=COMBO_COLORS[combo], pad=10)

    # Nodes
    for org, (nx, ny) in _NODE_POS.items():
        circ = plt.Circle((nx, ny), 0.09, color=_NODE_COL[org],
                           ec='#2C3E50', lw=2.0, zorder=5)
        ax.add_patch(circ)
        ax.text(nx, ny, _NODE_ABBR[org], ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=6)

    # Arrows
    for down, up in COVER_EDGES:
        x0, y0 = _NODE_POS[down]
        x1, y1 = _NODE_POS[up]
        dx = x1 - x0; dy = y1 - y0
        dist = (dx**2 + dy**2) ** 0.5
        px = -dy / dist; py = dx / dist

        for direction, (org_f, org_t), rad in [
                ('up',   (down, up),  +0.22),
                ('down', (up, down),  -0.22)]:
            n = _n(f'{down}-{up}', direction, combo)
            if n == 0:
                continue
            lw = 0.8 + 7.5 * (n / MAX_N) ** 0.5
            xf, yf = _NODE_POS[org_f]
            xt, yt = _NODE_POS[org_t]
            ax.annotate('',
                xy=(xt, yt), xytext=(xf, yf),
                arrowprops=dict(
                    arrowstyle='->', color=COMBO_COLORS[combo],
                    lw=lw, mutation_scale=20,
                    connectionstyle=f'arc3,rad={rad}'),
                zorder=4)
            mx = (xf + xt) / 2 + rad * 0.45 * dist * px
            my = (yf + yt) / 2 + rad * 0.45 * dist * py
            ax.text(mx, my, str(n), fontsize=8, ha='center', va='center',
                    color=COMBO_COLORS[combo], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.90, lw=0), zorder=7)

fig4.suptitle(
    'Directed ESMO transition graph — one panel per state-combination type\n'
    'Arrow width ∝ √N_ESMOs.  Number = ESMO count.\n'
    'PP = peace→peace,  PC = peace→conflict,  CP = conflict→peace,  CC = conflict→conflict',
    fontsize=FONT_SIZE, y=1.01)
fig4.tight_layout()
_save(fig4, 'fig_inter4_transition_graph')

print("\nAll inter-org figures saved.")
