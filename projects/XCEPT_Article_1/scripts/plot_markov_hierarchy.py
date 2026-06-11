#!/usr/bin/env python3
"""
plot_markov_hierarchy.py
========================
Structural Markov graph on (org, state_type) nodes using ESMO-based
conditional relative frequencies from the inter-org analysis.

Nodes (8):
  TrP/TrC = Tribe-peace/conflict
  ChP/ChC = Chief-peace/conflict
  StP/StC = State-peace/conflict
  CSP/CSC = ChiefState-peace/conflict

Edges (32): directed along cover edges of the org lattice.
  f-value = conditional relative frequency (same as fig_inter2_combo_counts):
    f(PP)+f(PC)=1  (from peace)
    f(CP)+f(CC)=1  (from conflict)

Key finding — dominant 4-cycle (by ESMO-weighted probability):
  TrC -> ChC -> CSC -> StP -> TrC
  Tribal conflict escalates -> Chiefdom conflict -> Dual-hierarchy conflict
  -> State-peace (fragile) -> Tribal conflict (structural collapse)

Output:
  markov_hierarchy.html   — interactive PyVis graph
  fig_markov_loops.png    — matplotlib summary with dominant loops
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'complex_COT_ESMO')
SUMMARY_FILE = os.path.join(OUT_DIR, 'transition_summary.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# 'P'  — N-weighted probability (sums to 1 per source node across all 4 outgoing edges) [default]
# 'f'  — conditional relative frequency per directed edge (f(PP)+f(PC)=1 per edge)
# 'both' — show f on top, P below
SHOW_VALUE = 'P'

# Driver for arrow thickness/opacity.  Must be one column in edf: 'f' or 'P'.
WEIGHT_COL = 'P'

def _edge_label(f_val, p_val):
    if SHOW_VALUE == 'f':
        return f'{f_val:.2f}'
    if SHOW_VALUE == 'both':
        return f'{f_val:.2f}\n{p_val:.2f}'
    return f'{p_val:.2f}'  # default: P

# -- 1. Load and compute f-values ---------------------------------------------
df = pd.read_csv(SUMMARY_FILE)
print(f"Loaded {len(df)} rows from {os.path.realpath(SUMMARY_FILE)}")

def _node_id(org, state):
    abbr = {'Tribe': 'Tr', 'Chief': 'Ch', 'State': 'St', 'ChiefState': 'CS'}
    return abbr[org] + ('P' if state == 'peace' else 'C')

COMPLEMENT = {'pp': 'pc', 'pc': 'pp', 'cp': 'cc', 'cc': 'cp'}

rows = []
for _, r in df.iterrows():
    src, dst = _node_id(r['org_from'], r['state_from']), _node_id(r['org_to'], r['state_to'])
    n    = int(r['n_esmos'])
    comp = COMPLEMENT[r['combo']]
    comp_rows = df[(df['edge'] == r['edge']) & (df['direction'] == r['direction'])
                   & (df['combo'] == comp)]
    n_comp = int(comp_rows['n_esmos'].values[0]) if len(comp_rows) else 0
    f = n / (n + n_comp) if (n + n_comp) > 0 else 0.0
    rows.append({'src': src, 'dst': dst, 'n': n, 'f': f,
                 'combo': r['combo'], 'edge': r['edge'], 'direction': r['direction']})

edf = pd.DataFrame(rows)

# Normalised probability: P(src->dst) = N / sum_N_from_src
total_n = edf.groupby('src')['n'].sum()
edf['P'] = edf.apply(lambda r: r['n'] / total_n[r['src']], axis=1)

# -- 2. Dominant-path analysis -------------------------------------------------
print("\n" + "="*62)
print("  DOMINANT TRANSITIONS (greedy max-P from each node)")
print("="*62)

dominant = {}
for node, sub in edf.groupby('src'):
    best = sub.loc[sub['P'].idxmax()]
    dominant[node] = best['dst']
    print(f"  {node} -> {best['dst']}  "
          f"P={best['P']:.3f}  f={best['f']:.3f}  N={int(best['n'])}  combo={best['combo']}")

# Trace dominant cycle
print("\n  Greedy path trace:")
visited, current = [], 'TrC'
for _ in range(20):
    if current in visited:
        idx   = visited.index(current)
        cycle = visited[idx:]
        print(f"  CYCLE: {' -> '.join(cycle)} -> {current}")
        break
    visited.append(current)
    nxt = dominant[current]
    row = edf[(edf['src'] == current) & (edf['dst'] == nxt)].iloc[0]
    print(f"    {current} -> {nxt}  P={row['P']:.3f}")
    current = nxt

# -- 3. Notable loops ----------------------------------------------------------
LOOPS = [
    ("Dominant 4-cycle  (main structural attractor)",
     ['TrC', 'ChC', 'CSC', 'StP', 'TrC']),
    ("Conflict escalation 2-cycle  ChC <-> CSC",
     ['ChC', 'CSC', 'ChC']),
    ("Recovery-relapse 2-cycle  TrC <-> StP  (symmetric f=0.578)",
     ['TrC', 'StP', 'TrC']),
    ("ChiefState-conflict <-> State-peace  2-cycle",
     ['CSC', 'StP', 'CSC']),
]

print("\n" + "="*62)
print("  NOTABLE STRUCTURAL LOOPS")
print("="*62)
for name, cycle in LOOPS:
    print(f"\n  {name}")
    print(f"    Path: {' -> '.join(cycle)}")
    for i in range(len(cycle) - 1):
        s, d = cycle[i], cycle[i+1]
        row = edf[(edf['src'] == s) & (edf['dst'] == d)]
        if len(row):
            rx = row.iloc[0]
            print(f"      {s}->{d}: N={int(rx['n']):4d}  f={rx['f']:.3f}  P={rx['P']:.3f}"
                  f"  ({rx['combo'].upper()})")

# -- 4. Matplotlib figure ------------------------------------------------------
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      12,
    'axes.titlesize': 12,
})

NODE_POS = {
    'TrP': np.array([0.5, 0.0]),
    'TrC': np.array([1.5, 0.0]),
    'ChP': np.array([-0.5, 2.0]),
    'ChC': np.array([0.5, 2.0]),
    'StP': np.array([1.5, 2.0]),
    'StC': np.array([2.5, 2.0]),
    'CSP': np.array([0.5, 4.0]),
    'CSC': np.array([1.5, 4.0]),
}

# Muted two-tone palette: very pale blue-gray for peace, very pale rose for conflict
FACE  = {n: ('#DCE9F5' if n.endswith('P') else '#F5DEDE') for n in NODE_POS}
ECOLR = {n: ('#2E4057' if n.endswith('P') else '#6B2222') for n in NODE_POS}
LABEL = {
    'TrP': 'Tribe\npeace',   'TrC': 'Tribe\nconflict',
    'ChP': 'Chief\npeace',   'ChC': 'Chief\nconflict',
    'StP': 'State\npeace',   'StC': 'State\nconflict',
    'CSP': 'CfState\npeace', 'CSC': 'CfState\nconflict',
}

CYCLE4     = [('TrC', 'ChC'), ('ChC', 'CSC'), ('CSC', 'StP'), ('StP', 'TrC')]
CYCLE4_SET = set(CYCLE4)
CYCLE_NODES= {'TrC', 'ChC', 'CSC', 'StP'}
CYCLE_COL  = '#1A1A2E'   # near-black navy for cycle node borders
NODE_R     = 0.235        # radius in data units
COMBO_COLORS = {
    'pp': '#3D7A4A',  # muted forest green  — peace  -> peace
    'pc': '#C8810A',  # amber               — peace  -> conflict
    'cp': '#2563A6',  # steel blue          — conflict -> peace
    'cc': '#A83232',  # brick red           — conflict -> conflict
}

def _arrow(ax, src, dst, color, lw, alpha, rad, zorder=2):
    d = np.array(dst) - np.array(src)
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return
    u  = d / dist
    s0 = np.array(src) + NODE_R * u
    d0 = np.array(dst) - NODE_R * u
    ax.annotate('', xy=tuple(d0), xytext=tuple(s0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                mutation_scale=18, alpha=alpha),
                zorder=zorder)


fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(-1.3, 3.5); ax.set_ylim(-0.8, 5.1)
ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor('white')

# All 32 transitions — single formula driven by WEIGHT_COL; rad=0.15 separates pairs
# P in [0.10, 0.40]:  lw = w*12  → [1.2, 4.8];  alpha = w*2.2  → [0.22, 0.88]
for _, r in edf.iterrows():
    w = r[WEIGHT_COL]
    _arrow(ax, NODE_POS[r['src']], NODE_POS[r['dst']],
           color=COMBO_COLORS[r['combo']],
           lw=w * 12, alpha=w * 2.2, rad=0.15, zorder=2)

# Nodes
for node, pos in NODE_POS.items():
    in_cycle  = node in CYCLE_NODES
    lw_border = 3.2 if in_cycle else 1.5
    ec        = CYCLE_COL if in_cycle else ECOLR[node]
    circ = plt.Circle(pos, NODE_R, facecolor=FACE[node],
                      edgecolor=ec, linewidth=lw_border, zorder=8)
    ax.add_patch(circ)
    ax.text(pos[0], pos[1], LABEL[node],
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#000000', zorder=9)

# Level labels (left margin)
for y, lab in [(0.0, 'Band / Tribe'), (2.0, 'Chiefdom / State'), (4.0, 'Dual hierarchy')]:
    ax.text(-1.25, y, lab, fontsize=11, ha='left', va='center',
            color='#000000', style='italic')
    ax.plot([-1.10, -1.05], [y, y], color='#AAAAAA', lw=0.8)

# Legend
legend_h = [
    mpatches.Patch(fc='#DCE9F5', ec='#2E4057', lw=1.5, label='Peace state'),
    mpatches.Patch(fc='#F5DEDE', ec='#6B2222', lw=1.5, label='Conflict state'),
    mpatches.Patch(fc='#DCE9F5', ec=CYCLE_COL, lw=3.0, label='Node in dominant cycle'),
    plt.Line2D([0], [0], color=COMBO_COLORS['pp'], lw=2.5, label='PP  peace -> peace'),
    plt.Line2D([0], [0], color=COMBO_COLORS['pc'], lw=2.5, label='PC  peace -> conflict'),
    plt.Line2D([0], [0], color=COMBO_COLORS['cp'], lw=2.5, label='CP  conflict -> peace'),
    plt.Line2D([0], [0], color=COMBO_COLORS['cc'], lw=2.5, label='CC  conflict -> conflict'),
    plt.Line2D([0], [0], color='#555555', lw=1.0,
               label='Thickness / opacity proportional to f'),
]
ax.legend(handles=legend_h, loc='lower left', fontsize=11,
          framealpha=0.97, edgecolor='#BBBBBB',
          title='Legend', title_fontsize=11.5)

# Annotation box — values follow SHOW_VALUE setting
_lbl_desc = {'P': 'N-weighted probability', 'f': 'cond. relative frequency',
             'both': 'cond. freq. (top) / prob. (bottom)'}
def _ev(s, d):
    row = edf[(edf['src'] == s) & (edf['dst'] == d)].iloc[0]
    return _edge_label(row['f'], row['P'])
box = (
    f"Dominant 4-cycle\n({_lbl_desc[SHOW_VALUE]})\n\n"
    "TrC -> ChC -> CSC -> StP -> TrC\n\n"
    f"(1) Tribe-conflict  ->  Chief-conflict\n"
    f"    Tribal conflict escalates  (CC, {_ev('TrC','ChC')})\n\n"
    f"(2) Chief-conflict  ->  CfState-conflict\n"
    f"    Chiefdom conflict escalates  (CC, {_ev('ChC','CSC')})\n\n"
    f"(3) CfState-conflict  ->  State-peace\n"
    f"    Dual hierarchy simplifies  (CP, {_ev('CSC','StP')})\n\n"
    f"(4) State-peace  ->  Tribe-conflict\n"
    f"    State-peace structurally collapses  (PC, {_ev('StP','TrC')})\n\n"
    f"Edge labels: {_lbl_desc[SHOW_VALUE]}\n\n"
    f"Embedded 2-cycles:\n"
    f"  ChC <-> CSC   {_ev('ChC','CSC')} / {_ev('CSC','ChC')}   (escalation trap)\n"
    f"  TrC <-> StP   {_ev('TrC','StP')} / {_ev('StP','TrC')}   (recovery-relapse)"
)
ax.text(0.988, 0.988, box,
        transform=ax.transAxes, fontsize=10.5,
        ha='right', va='top', color='#000000',
        bbox=dict(boxstyle='round,pad=0.6', fc='#F9F9F9',
                  ec='#AAAAAA', lw=0.8, alpha=0.97),
        zorder=10)

ax.set_title(
    'Structural Markov graph on (organization, state) pairs\n'
    'All 32 ESMO inter-organizational transitions shown. '
    'Thick arrows: dominant 4-cycle. '
    'Gray arcs: embedded 2-cycles.',
    fontsize=12, pad=14, color='#000000')

fig.tight_layout()
fig_path = os.path.join(os.path.realpath(OUT_DIR), 'fig_markov_loops.png')
fig.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {fig_path}")
plt.close(fig)

# -- 5. PyVis HTML -------------------------------------------------------------
try:
    from pyvis.network import Network

    PYVIS_POS = {
        'TrP': (-130, 380), 'TrC': (130, 380),
        'ChP': (-340, 110), 'ChC': (-90, 110),
        'StP': (90,  110),  'StC': (340, 110),
        'CSP': (-130, -160),'CSC': (130, -160),
    }
    CYCLE_NODES_HTML = {'TrC', 'ChC', 'CSC', 'StP'}

    net = Network(height='800px', width='100%', directed=True, notebook=False)
    net.set_options("""
    {
      "nodes": {
        "shape": "circle",
        "font": {"size": 13, "align": "center", "bold": true},
        "borderWidth": 3,
        "size": 42
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.18},
        "arrows": {"to": {"enabled": true, "scaleFactor": 1.1}},
        "font": {"size": 10, "align": "middle",
                 "strokeWidth": 2, "strokeColor": "#ffffff"}
      },
      "physics": {"enabled": false}
    }
    """)

    for node, (x, y) in PYVIS_POS.items():
        in_cycle = node in CYCLE_NODES_HTML
        bg       = '#DCE9F5' if node.endswith('P') else '#F5DEDE'
        border   = CYCLE_COL if in_cycle else ('#2E4057' if node.endswith('P') else '#6B2222')
        bw       = 4 if in_cycle else 2
        net.add_node(node,
                     label=node,
                     title=LABEL[node].replace('\n', ' - '),
                     x=x, y=y, physics=False,
                     color={'background': bg, 'border': border,
                            'highlight': {'background': '#EAF2FB', 'border': CYCLE_COL}},
                     shape='circle', size=44,
                     borderWidth=bw,
                     font={'size': 14, 'bold': True, 'color': '#111111'})

    for _, r in edf.iterrows():
        s, d = r['src'], r['dst']
        f, p, combo = r['f'], r['P'], r['combo']
        is_cycle = (s, d) in CYCLE4_SET
        w      = r[WEIGHT_COL]
        color  = COMBO_COLORS[combo]
        net.add_edge(s, d,
                     label=_edge_label(f, p),
                     title=(f'{s}->{d}  {combo.upper()}\n'
                            f'N={int(r["n"])}  f={f:.3f}  P={p:.3f}'),
                     width=w * 15,
                     color={'color': color, 'opacity': w * 2.2},
                     font={'size': 11, 'color': '#333333'})

    html_path = os.path.join(os.path.realpath(OUT_DIR), 'markov_hierarchy.html')
    net.write_html(html_path)
    print(f"Saved: {html_path}")

except ImportError:
    print("  [PyVis not available — skipping HTML]")

print("\nDone.")
