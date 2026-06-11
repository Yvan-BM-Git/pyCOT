#!/usr/bin/env python3
"""
plot_intra_org_figures.py
=========================
Regenerate intra-org ESMO figures directly from the cached CSV.
No LP computation required.

Figures produced
----------------
fig1_esmo_distributions.png  — violin plots, shared y-axis [Y_MIN, Y_MAX]
fig2_feasibility.png         — left: Pr(N>0) with k/n labels; right: mean N +/- SD
fig3_pc_vs_cp_scatter.png    — matched PC vs CP scatter per org
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_SCRIPT_DIR, '..', 'outputs', 'intra_org_esmo')
CACHE_FILE  = os.path.join(OUT_DIR, 'esmo_cache_intra.csv')

ORG_NAMES = ['Tribe', 'Chief', 'State', 'ChiefState']
FONT_SIZE  = 11
COMBO_COLORS = {'pc': '#E67E22', 'cp': '#2980B9'}

# Fig 1 shared y-axis limits (user-specified)
Y_MIN, Y_MAX = 300, 2250
# ---------------------------------------------------------------------------

df = pd.read_csv(CACHE_FILE)
print(f"Loaded {len(df)} rows from {os.path.realpath(CACHE_FILE)}")

# Build stats dict
stats = {}
for org in ORG_NAMES:
    for combo in ['pc', 'cp']:
        n_vals = df[(df['org'] == org) & (df['combo'] == combo)]['n_esmos'].values
        n_pairs    = len(n_vals)
        n_feasible = int(np.sum(n_vals > 0))
        stats[(org, combo)] = dict(
            n_pairs    = n_pairs,
            n_feasible = n_feasible,
            Pr         = float(n_feasible / n_pairs) if n_pairs else 0.0,
            mean_N     = float(np.mean(n_vals))      if n_pairs else 0.0,
            std_N      = float(np.std(n_vals))        if n_pairs else 0.0,
            min_N      = int(n_vals.min())            if n_pairs else 0,
            max_N      = int(n_vals.max())            if n_pairs else 0,
            n_vals     = n_vals,
        )

# Warn about clipping in fig1
for org in ORG_NAMES:
    for combo in ['pc', 'cp']:
        n_vals = stats[(org, combo)]['n_vals']
        clipped = int(np.sum(n_vals < Y_MIN))
        if clipped:
            print(f"  [fig1 WARN] {org} {combo}: {clipped} values below "
                  f"y_min={Y_MIN} (min={n_vals.min()})")

plt.rcParams.update({
    'font.size':         FONT_SIZE,
    'axes.titlesize':    FONT_SIZE,
    'axes.labelsize':    FONT_SIZE,
    'xtick.labelsize':   FONT_SIZE - 2,
    'ytick.labelsize':   FONT_SIZE - 2,
    'legend.fontsize':   FONT_SIZE - 2,
})

def _save(fig, stem):
    p = os.path.join(os.path.realpath(OUT_DIR), f'{stem}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close(fig)


# ===========================================================================
# Fig 1: Distribution of ESMO counts — shared y-axis violin
# ===========================================================================
fig1, axes1 = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

for oi, org in enumerate(ORG_NAMES):
    ax = axes1[oi]
    data_pc = stats[(org, 'pc')]['n_vals']
    data_cp = stats[(org, 'cp')]['n_vals']

    parts = ax.violinplot([data_pc, data_cp], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for body, col in zip(parts['bodies'],
                         [COMBO_COLORS['pc'], COMBO_COLORS['cp']]):
        body.set_facecolor(col)
        body.set_alpha(0.65)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[part].set_color('#2C3E50')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['PC\n(destab.)', 'CP\n(recovery)'], fontsize=9)
    ax.set_title(org, fontweight='bold')
    if oi == 0:
        ax.set_ylabel('N ESMOs')
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    for xi, combo in enumerate(['pc', 'cp'], start=1):
        mn = stats[(org, combo)]['mean_N']
        if Y_MIN <= mn <= Y_MAX:
            ax.scatter([xi], [mn], zorder=5, s=40, color='white',
                       edgecolors=COMBO_COLORS[combo], linewidths=1.5)
            ax.text(xi + 0.13, mn, f'μ={mn:.0f}', va='center', fontsize=7)

fig1.suptitle(
    f'ESMO count distributions across all state pairs (intra-org) — '
    f'shared y-axis [{Y_MIN}, {Y_MAX}]\n'
    'PC = peace→conflict (destabilisation),  '
    'CP = conflict→peace (recovery)\n'
    'White dot = mean; violin = full distribution  '
    '(State CP: 8 values below 500 are clipped)',
    fontsize=FONT_SIZE - 1, y=1.04)
fig1.tight_layout()
_save(fig1, 'fig1_esmo_distributions')


# ===========================================================================
# Fig 2: Single panel — mean N ± SD bars annotated with relative frequency.
#
# For each org the relative frequency of each direction is:
#   Pr_PC = mean_N_PC / (mean_N_PC + mean_N_CP)
#   Pr_CP = mean_N_CP / (mean_N_PC + mean_N_CP)
# Bars show absolute mean N (left y-axis, with ±SD error bars).
# Each bar is annotated with both the absolute value and its relative freq.
# A twin right y-axis shows the relative frequency (0–1) on the same scale.
# ===========================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5))

x = np.arange(len(ORG_NAMES))
W = 0.35

# Precompute per-org totals for relative frequency
org_totals = {org: stats[(org, 'pc')]['mean_N'] + stats[(org, 'cp')]['mean_N']
              for org in ORG_NAMES}

for combo, offset in zip(['pc', 'cp'], [-W / 2, W / 2]):
    vals_mn = [stats[(org, combo)]['mean_N'] for org in ORG_NAMES]
    vals_sd = [stats[(org, combo)]['std_N']  for org in ORG_NAMES]
    bars = ax2.bar(x + offset, vals_mn, W,
                   color=COMBO_COLORS[combo], alpha=0.80, zorder=3,
                   label='PC  peace→conflict' if combo == 'pc' else 'CP  conflict→peace',
                   yerr=vals_sd, capsize=4,
                   error_kw={'ecolor': '#2C3E50', 'lw': 1.2})
    for bar, org, mn in zip(bars, ORG_NAMES, vals_mn):
        rel = mn / org_totals[org]
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f'{mn:.0f}\n({rel:.2f})',
                 ha='center', va='bottom', fontsize=7.5, linespacing=1.3)

ax2.set_xticks(x)
ax2.set_xticklabels(ORG_NAMES, rotation=15, ha='right')
ax2.set_ylabel('Mean ESMO count  E[N]')
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.25)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_axisbelow(True)

# Twin right axis: relative frequency scale (0–1)
# Scale = left_value / max_org_total so the right axis reads the rel freq
# for whichever org has the largest total.  Annotations give exact per-org values.
max_total = max(org_totals.values())
ax2r = ax2.twinx()
ax2r.set_ylim(0, ax2.get_ylim()[1] / max_total)
ax2r.set_ylabel('Relative frequency  N / (N_PC + N_CP)', labelpad=8)
ax2r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

fig2.suptitle(
    'Structural accessibility by direction (intra-org)\n'
    'Bars = mean ESMO count ± 1 SD.  '
    'Annotation: absolute value and relative frequency N / (N_PC + N_CP)',
    fontsize=FONT_SIZE - 1, y=1.03)
fig2.tight_layout()
_save(fig2, 'fig2_feasibility')


# ===========================================================================
# Fig 3: Matched PC vs CP scatter — each org a panel
#
# Each point = one (peace_vi, conflict_vj) pair.
# x = N_ESMOs for the PC direction (peace -> conflict)
# y = N_ESMOs for the CP direction (conflict -> peace)
# Many pairs share the same sig-hash and therefore the same N value,
# causing overlapping points.  alpha=0.6 encodes point density: darker
# regions have more overlapping pairs with that (N_PC, N_CP) combination.
# ===========================================================================
fig3, axes3 = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)

for oi, org in enumerate(ORG_NAMES):
    ax = axes3[oi]
    sub = df[df['org'] == org]

    # Build lookup: (peace_variant, conflict_variant) -> n_esmos
    pc_dict, cp_dict = {}, {}
    for _, row in sub[sub['combo'] == 'pc'].iterrows():
        p_var = row['state_from_id'].split('::')[1]
        c_var = row['state_to_id'].split('::')[1]
        pc_dict[(p_var, c_var)] = row['n_esmos']
    for _, row in sub[sub['combo'] == 'cp'].iterrows():
        c_var = row['state_from_id'].split('::')[1]
        p_var = row['state_to_id'].split('::')[1]
        cp_dict[(p_var, c_var)] = row['n_esmos']

    matched_pc, matched_cp = [], []
    for key, n_pc in pc_dict.items():
        n_cp = cp_dict.get(key)
        if n_cp is not None:
            matched_pc.append(n_pc)
            matched_cp.append(n_cp)

    if matched_pc:
        # alpha=0.6: overlapping points appear darker, encoding point density
        ax.scatter(matched_pc, matched_cp, alpha=0.6, s=30,
                   color='#8E44AD', edgecolors='#5B2C6F', lw=0.5, zorder=3)
        all_vals = matched_pc + matched_cp
        mx = max(all_vals) * 1.05
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.5, label='PC = CP')
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)

    ax.set_xlabel('N ESMOs  (PC, destabilisation)', fontsize=9)
    if oi == 0:
        ax.set_ylabel('N ESMOs  (CP, recovery)', fontsize=9)
    ax.set_title(org, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if oi == 0:
        ax.legend(fontsize=8)

fig3.suptitle(
    'Matched PC vs CP ESMO counts per state pair  (intra-org)\n'
    'Points above the diagonal: recovery has more pathways than destabilisation\n'
    'Darkness ∝ overlap density: darker = more pairs sharing that (N_PC, N_CP)',
    fontsize=FONT_SIZE - 1, y=1.04)
fig3.tight_layout()
_save(fig3, 'fig3_pc_vs_cp_scatter')

print("\nAll figures saved.")
