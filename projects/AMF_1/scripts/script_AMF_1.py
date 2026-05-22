# ========================================
# AMF – TRADE BALANCE MODEL
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from matplotlib.colors import TwoSlopeNorm

# Absolute paths for pyCOT import (adjust if necessary)
sys.path.append('/Users/yvanomarbalderamoreno/Downloads/pyCOT/src')
sys.path.append('/Users/yvanomarbalderamoreno/Downloads/pyCOT')

from pyCOT.io.functions import read_txt
from pyCOT.simulations.ode import simulation

VIS_DIR = 'projects/AMF_2/outputs_2'
os.makedirs(VIS_DIR, exist_ok=True)

# ========================================
# PLOTTING FUNCTIONS
# ========================================

def plot_amf_dynamics_new(time_series, title="AMF Dynamics", save_path=None):

    time = time_series['Time'].values
    all_species = [c for c in time_series.columns if c != 'Time']

    groups = {
        "Biomass (Plant & Fungus)": ['Plant_active', 'Plant_limited', 'Fungus', 'Mycelium'],
        "Carbon Pools":             ['C_plant', 'C_fungus'],
        "Nitrogen":                 ['N_plant', 'N_fungus', 'N_avail'],
        "Phosphorus":               ['P_plant', 'P_fungus', 'P_avail'],
    }

    fig, axes = plt.subplots(4, 1, figsize=(4, 12))
    axes = axes.flatten()

    for ax, (group, group_vars) in zip(axes, groups.items()):

        for var in [v for v in group_vars if v in all_species]:

            lw = 2.5 if var in ['Plant_active', 'Fungus', 'C_plant'] else 1.8

            ax.plot(
                time,
                time_series[var],
                linewidth=lw,
                label=var
            )

        ax.set_title(group, fontweight='bold', fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.close()

def plot_flux_dynamics_new(flux_df, title="Flux Dynamics", save_path=None):

    time = flux_df['Time'].values
    all_reactions = [c for c in flux_df.columns if c != 'Time']

    groups = {
        "Plant Core & Growth": ['R1', 'R2', 'R3', 'R4', 'R11', 'R16'],
        "Mortality":           ['R7', 'R8', 'R12', 'R17'],
        "Mycelium Uptake":     ['R9', 'R10'],
        "Plant Direct Uptake": ['R5', 'R6'],
        "Symbiotic Exchange":  ['R14', 'R15', 'R13'],
        "Inputs":              ['R18', 'R19'],
    }

    nrows = 6
    ncols = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(3, 15))
    axes = axes.flatten()

    for ax, (group, rxns) in zip(axes, groups.items()):

        for rxn in [r for r in rxns if r in all_reactions]:

            ls = '--' if rxn in ['R4', 'R11', 'R13', 'R16'] else '-'

            ax.plot(
                time,
                flux_df[rxn],
                linewidth=1.8,
                linestyle=ls,
                label=rxn
            )

        ax.set_title(group, fontweight='bold', fontsize=12)

        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")

        # ====================================
        # INPUT PANEL Y-LIMITS
        # ====================================
        if group == "Inputs":
            ax.set_ylim(0, 3)

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.close()

# ========================================
# BIOLOGICAL BASE PARAMETERS
# ========================================

# Baseline biomass initial conditions
x0_dict = {
    'Plant_active': 5.5,   # [1.00-7.50] g biomass/kg soil
    'Plant_limited': 1.5,  # [0.50-2.50] g biomass/kg soil
    'C_plant': 45.0,       # [35.0-50.0] % leaf C
    'N_plant': 2.0,        # [1.00-3.00] % leaf N
    'P_plant': 0.4,        # [0.05-0.88] % leaf P
    'Fungus': 20,          # [6.00-90.0] % root length colonized
    'Mycelium': 2.0,       # [0.10-3.00] m mycelium/g soil
    'C_fungus': 40.0,      # [35.0-50.0] % fungal C
    'N_fungus': 5.0,       # [1.00-6.00] % fungal N
    'P_fungus': 0.5,       # [0.10-2.00] % fungal P
    'N_avail': 30,         # [10-40] mg N/kg dry soil
    'P_avail': 25          # [5-20] mg P/kg dry soil
}

_SPECIES_ORDER = [
    'Plant_active', 'Plant_limited', 'C_plant', 'N_plant', 'P_plant',
    'Fungus', 'Mycelium', 'C_fungus', 'N_fungus', 'P_fungus',
    'N_avail', 'P_avail'
]

# RESULTS: Explicit ranges for the 24 parameters [Lower_bound, Upper_bound]

def create_base_params():

    return [

        [2.5, 10.0],   # R1(mmk): Photosynthesis
        [1e-5],        # R2(mak): Activation
        [0.02],        # R3(mak): Senescence
        [8e-6],        # R4(mak): Plant growth
        [5e-4],        # R5(mak): Direct N uptake by roots
        [0.005],       # R6(mak): Direct P uptake by roots
        [5e-8],        # R7(mak): Plant_active recycling
        [5e-3],        # R8(mak): Plant_limited recycling
        [1.5, 30.0],   # R9(mmk): Mycelial N uptake
        [1.2, 10.0],   # R10(mmk): Mycelial P uptake
        [5e-7],        # R11(mak): Mycelium growth
        [0.01],        # R12(mak): Mycelium mortality
        [0.0001],      # R13(mak): Fungal C uptake
        [0.8, 20.0],   # R14(mmk): Fungus→plant N transfer
        [0.5, 20.0],   # R15(mmk): Fungus→plant P transfer
        [6e-7],        # R16(mak): Fungal growth
        [0.001],       # R17(mak): Fungal mortality
        [0.0],         # R18(mak): N input
        [0.0],         # R19(mak): P input
    ]

def create_rate_list():

    return [
        'mmk', 'mak', 'mak', 'mak', 'mak', 'mak', 'mak', 'mak',
        'mmk', 'mmk', 'mak', 'mak', 'mak', 'mmk', 'mmk', 'mak',
        'mak', 'mak', 'mak'
    ]

# ========================================
# SCENARIO CONFIGURATION
# ========================================

def get_scenario_config(scenario: str, species_order=None):

    if species_order is None:
        species_order = _SPECIES_ORDER

    params = create_base_params()

    IDX_R18 = 17  # N input
    IDX_R19 = 18  # P input

    # Trade Balance Model (Johnson 2010)
    # Limitation = "Low"
    # Luxury = "High"

    if scenario == 'I':

        x0_dict['N_avail'] = 15
        x0_dict['P_avail'] = 5

        params[IDX_R18][0] = 0.005
        params[IDX_R19][0] = 0.01

        params_name = "I – N↓ P↓"
        scenario_name = "I – C-limited mutualism"

    elif scenario == 'II':

        x0_dict['N_avail'] = 45
        x0_dict['P_avail'] = 5

        params[IDX_R18][0] = 0.5
        params[IDX_R19][0] = 0.09

        params_name = "II – N↑ P↓"
        scenario_name = "II – Strong mutualism"

    elif scenario == 'III':

        x0_dict['N_avail'] = 15
        x0_dict['P_avail'] = 30

        params[IDX_R18][0] = 0.005
        params[IDX_R19][0] = 2.5

        params_name = "III – N↓ P↑"
        scenario_name = "III – Commensalism"

    elif scenario == 'IV':

        x0_dict['N_avail'] = 45
        x0_dict['P_avail'] = 30

        params[IDX_R18][0] = 1.9
        params[IDX_R19][0] = 2.5

        params_name = "IV – N↑ P↑"
        scenario_name = "IV – Parasitism"

    else:
        raise ValueError("Invalid scenario")

    x0 = [x0_dict[s] for s in species_order]

    return params, x0, params_name, scenario_name

# ========================================
# INDICATORS
# ========================================
def calculate_amf_indicators(time_series, flux_vector, t_span, steady_frac=0.2):
    """Calculate mycorrhizal function indicators at steady state"""
    t_ss_init = t_span[1] - steady_frac * (t_span[1] - t_span[0])
    time_arr  = flux_vector['Time'].values
    n_crit    = np.searchsorted(time_arr, t_ss_init)

    def ss(rxn): return float(flux_vector[rxn].iloc[n_crit:].mean())

    v1  = ss('R1')  # Photosynthesis (C supply)
    v5 = ss('R5')   # Direct N uptake
    v6 = ss('R6')   # Direct P uptake
    v13 = ss('R13') # Fungal C demand (C uptake from plant)
    v14 = ss('R14') # N transfer fungus→plant
    v15 = ss('R15') # P transfer fungus→plant
    C_demand = v13/(v1 + 1e-10)  # Proportion of C supply demanded by fungus

    P_benefit = v15 / (v6 + v15 + 1e-10)  # Proportion of P acquired via mycorrhizas
    ind = {'C_supply': v1, 'C_demand': C_demand, 'P_benefit': P_benefit}
    return ind, n_crit


# ========================================
# PLOT TRADE BALANCE MODEL
# ========================================
def plot_trade_balance_model(rn, save_path=None):
    rate_list = create_rate_list()
    n=1
    t_span = (0, 360 * n) 
    n_steps = 500 * n
    species_order = [s.name for s in rn.species()]
    scenarios = ['III', 'IV', 'I', 'II']
    results = {}

    colors = {'I': '#4A90D9', 'II': '#27AE60', 'III': '#F39C12', 'IV': '#E74C3C'}
    labels = {
        'I':   'I – C-limited mutualism (N↓P↓)',
        'II':  'II – Strong mutualism (N↑P↓)',
        'III': 'III – Commensalism (N↓P↑)',
        'IV':  'IV – Parasitism (N↑P↑)',
    }

    print("=" * 65)
    print("  AMF TRADE BALANCE MODEL (Biological Parameters)")
    print("=" * 65)

    ts_dict = {}; flux_dict = {}

    for scen in scenarios:
        params, x0, params_name, scenario_name = get_scenario_config(scen, species_order)
        print(f"\n[{scen}] {scenario_name}")

        # Slightly adjust tolerances to avoid numerical warnings
        time_series, flux_vector = simulation(
            rn, x0=x0, rate=rate_list, spec_vector=params,
            t_span=t_span, n_steps=n_steps, method='LSODA', rtol=1e-8, atol=1e-10
        )
        ts_dict[scen] = time_series
        flux_dict[scen] = flux_vector
        
        plot_amf_dynamics_new(time_series, title=f"Scenario {scenario_name}", save_path=os.path.join(VIS_DIR, '1_time_series', f'{scen}_ts.png'))
        plot_flux_dynamics_new(flux_vector, title=f"Scenario {scenario_name}", save_path=os.path.join(VIS_DIR, '2_flux_vector', f'{scen}_fd.png'))
        
        ind, _ = calculate_amf_indicators(time_series, flux_vector, t_span)
        results[scen] = ind

    # ── Figure 1: Indicator temporal trajectories ──────────
    ind_colors = {'C_supply': '#2ECC71', 'C_demand': '#E67E22', 'P_benefit': '#3498DB'}
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
    axes1 = axes1.flatten()

    for ax, scen in zip(axes1, scenarios):
        _, _, params_name, scenario_name = get_scenario_config(scen, species_order)
        t   = flux_dict[scen]['Time'].values
        v1  = flux_dict[scen]['R1'].values
        v13 = flux_dict[scen]['R13'].values
        v15 = flux_dict[scen]['R15'].values
        v6  = flux_dict[scen]['R6'].values
        pb  = v15 / (v15 + v6 + 1e-10)

        ax.plot(t, v1,  color=ind_colors['C_supply'],  lw=2.2, label='v1 – C supply')
        ax.plot(t, v13, color=ind_colors['C_demand'],  lw=2.2, label='v13 – C demand')
        ax.plot(t, pb,  color=ind_colors['P_benefit'], lw=2.2, label='P benefit = v15/(v15+v6)')

        t_ss = t_span[1] - 0.2 * (t_span[1] - t_span[0])
        ax.axvspan(t_ss, t_span[1], alpha=0.07, color='gray', label='SS Zone')
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)

        ax.set_title(f"{scenario_name}", fontweight='bold', fontsize=12, color=colors[scen])
        ax.set_xlabel("Time"); ax.set_ylabel("Flux / Ratio")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig1.suptitle("AMF Trade Balance Model\nBiological Indicator Trajectories", fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        p1 = save_path + '_indicators_ts.png'
        os.makedirs(os.path.dirname(p1), exist_ok=True)
        fig1.savefig(p1, dpi=300, bbox_inches='tight')
        print(f"Saved: {p1}")
    plt.close(fig1)

    # ── Figure 2: Phase map 2×2 ────────────────────────────────
    layout = {'III': (0, 0), 'IV': (0, 1), 'I': (1, 0), 'II': (1, 1)}
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    bar_keys    = ['C_supply', 'C_demand', 'P_benefit']
    bar_xlabels = ['C supply', 'C demand', 'P benefit']
    bar_cols    = [ind_colors[k] for k in bar_keys]

    for scen, (row, col) in layout.items():
        ax   = axes2[row][col]
        vals = [results[scen][k] for k in bar_keys]
        bars = ax.bar(bar_xlabels, vals, color=bar_cols, alpha=0.85, edgecolor='white', linewidth=1.2, width=0.55)
        for bar, val in zip(bars, vals):
            ypos = val + 0.003 if val >= 0 else val - 0.006
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.4f}", ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
        ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.set_ylabel("Steady State Value", fontsize=9)
        ax.set_title(labels[scen], fontweight='bold', fontsize=11, color=colors[scen])
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_ylim(0, max(1, max(vals) * 1.05))

    fig2.text(0.5, 0.01, "N axis → Limitation (left) | Luxury (right)", ha='center', fontsize=11, style='italic', color='#555')
    fig2.text(0.01, 0.5, "P axis → Limitation (bottom) | Luxury (top)", va='center', rotation='vertical', fontsize=11, style='italic', color='#555')
    fig2.suptitle("AMF Trade Balance Model – N × P Phase Map", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    if save_path:
        p2 = save_path + '_phase_map.png'
        fig2.savefig(p2, dpi=300, bbox_inches='tight')
        print(f"Saved: {p2}")
    # plt.close(fig2)
    plt.show()

    return results

# ========================================
# PLOT MYCORRHIZAL FUNCTION
# ========================================
def plot_mycorrhizal_function(rn, save_path=None):

    rate_list = create_rate_list()

    n = 1
    t_span = (0, 360 * n)
    n_steps = 500 * n

    species_order = [s.name for s in rn.species()]
    scenarios = ['I', 'II', 'III', 'IV']

    steady_frac = 0.2

    colors = {
        'I': '#4A90D9',
        'II': '#27AE60',
        'III': '#F39C12',
        'IV': '#E74C3C'
    }

    titles = {
        'I':   'I – C-lim.\n(N↓P↓)',
        'II':  'II – Mutualism\n(N↑P↓)',
        'III': 'III – Commens.\n(N↓P↑)',
        'IV':  'IV – Parasitism\n(N↑P↑)',
    }

    col_v1 = '#2ECC71'
    col_v13 = '#E67E22'
    col_pr = '#3498DB'

    print("\n" + "=" * 65)
    print("  AMF MYCORRHIZAL FUNCTION – MODULATION BY N AND P")
    print("=" * 65)

    mf_results = {}

    for scen in scenarios:

        params, x0, _, _ = get_scenario_config(scen, species_order)

        _, flux_vector = simulation(
            rn,
            x0=x0,
            rate=rate_list,
            spec_vector=params,
            t_span=t_span,
            n_steps=n_steps,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )

        t_ss = t_span[1] - steady_frac * (t_span[1] - t_span[0])

        n_c = np.searchsorted(
            flux_vector['Time'].values,
            t_ss
        )

        ss = lambda r: float(
            flux_vector[r].iloc[n_c:].mean()
        )

        # ------------------------------------
        # Steady-state indicators
        # ------------------------------------

        v1 = ss('R1')   # C supply (photosynthesis)

        v13 = ss('R13') # C demand (fungal C uptake)

        C_demand = v13 / (v1 + 1e-10)

        v15 = ss('R15') # P transfer fungus→plant

        v6 = ss('R6')   # Direct P uptake

        pb = v15 / (v15 + v6 + 1e-10)

        mf_results[scen] = {
            'v1': v1,
            'v13': v13,
            'C_demand': C_demand,
            'v15': v15,
            'v6': v6,
            'P_benefit': pb
        }

    # ======================================
    # Heatmap
    # ======================================
    # Layout follows Johnson 2010 Figure 1:
    #
    # N luxury      → top row
    # N limitation  → bottom row
    #
    # P luxury      → left column
    # P limitation  → right column

    layout = {
        'III': (1, 0),
        'IV':  (0, 0),
        'I':   (1, 1),
        'II':  (0, 1)
    }

    metrics = [
        'v1',
        'C_demand',
        'P_benefit'
    ]

    met_labels = [
        'C supply\n(v1 = R1)\nPhotosynthesis',
        'C demand\n(v13/v1 = R13/R1)\nFungal C uptake',
        'P exchange\n(v15/(v15+v6))\nMycorrhizal P benefit'
    ]

    met_colors = [
        col_v1,
        col_v13,
        col_pr
    ]

    matrices = {
        m: np.zeros((2, 2))
        for m in metrics
    }

    for scen, (row, col) in layout.items():

        for m in metrics:

            matrices[m][row, col] = mf_results[scen][m]

    fig2, axes2 = plt.subplots(
        1,
        3,
        figsize=(17, 6)
    )

    scen_in_cell = {
        v: k for k, v in layout.items()
    }

    for ax, m, mlabel, mcol in zip(
        axes2,
        metrics,
        met_labels,
        met_colors
    ):

        mat = matrices[m]

        vmin = mat.min()
        vmax = mat.max()

        vc = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2

        try:

            norm = TwoSlopeNorm(
                vmin=vmin - 1e-6,
                vcenter=vc,
                vmax=vmax + 1e-6
            )

            # Reverse colormap for C demand
            # because lower values are better

            cmap = (
                'RdYlGn'
                if m != 'C_demand'
                else 'RdYlGn_r'
            )

            im = ax.imshow(
                mat,
                cmap=cmap,
                norm=norm,
                aspect='auto'
            )

        except Exception:

            im = ax.imshow(
                mat,
                cmap='YlOrRd',
                aspect='auto'
            )

        plt.colorbar(
            im,
            ax=ax,
            shrink=0.82
        )

        for r in range(2):

            for c in range(2):

                val = mat[r, c]

                ax.text(
                    c,
                    r,
                    f"{val:.4f}",
                    ha='center',
                    va='center',
                    fontsize=13,
                    fontweight='bold',
                    color='white'
                    if abs(val) > 0.4 * max(abs(vmin), abs(vmax))
                    else 'black'
                )

                ax.text(
                    c,
                    r - 0.38,
                    f"Scen. {scen_in_cell.get((r,c),'?')}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    style='italic'
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            'P luxury',
            'P limitation'
        ])

        ax.set_yticks([0, 1])
        ax.set_yticklabels([
            'N luxury',
            'N limitation'
        ])

        ax.set_title(
            mlabel,
            fontweight='bold',
            fontsize=11,
            color=mcol,
            pad=10
        )

        for (row, col), scen in scen_in_cell.items():

            ax.add_patch(
                plt.Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    linewidth=3,
                    edgecolor=colors[scen],
                    facecolor='none',
                    zorder=5
                )
            )

    legend_patches = [
        mpatches.Patch(
            color=colors[e],
            label=titles[e].replace('\n', ' ')
        )
        for e in scenarios
    ]

    fig2.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
        title='Scenarios',
        title_fontsize=9
    )

    fig2.suptitle(
        "AMF Mycorrhizal Function – N × P Phase Map\n"
        "C supply, C demand, and P exchange ratio (steady state)",
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:

        p2 = save_path + '_mycorrhizal_function_heatmap.png'

        fig2.savefig(
            p2,
            dpi=300,
            bbox_inches='tight'
        )

        print(f"Saved: {p2}")

    plt.close(fig2)

    # ======================================
    # Summary table & Validation
    # (Johnson 2009)
    # ======================================

    print("\n" + "=" * 80)
    print("  EXPECTED RANGE EVALUATION (Johnson 2009, Fig. 1)")
    print("=" * 80)

    # Expected ranges [min, max]
    # 'ratio' corresponds to v13/v1

    ranges = {

        'I': {
            'v1': [0.10, 0.40],
            'ratio': [0.05, 0.20],
            'pb': [0.70, 1]
        },

        'II': {
            'v1': [0.70, 2.50],
            'ratio': [0.20, 0.70],
            'pb': [0.70, 1]
        },

        'III': {
            'v1': [0.10, 0.40],
            'ratio': [0.05, 0.20],
            'pb': [0.05, 0.40]
        },

        'IV': {
            'v1': [0.70, 2.50],
            'ratio': [0.20, 0.70],
            'pb': [0.05, 0.40]
        }
    }

    scen_titles = {

        'I':
            'Scenario I — C-limited mutualism (N↓ P↓)',

        'II':
            'Scenario II — Strong mutualism (N↑ P↓)',

        'III':
            'Scenario III — Commensalism (N↓ P↑)',

        'IV':
            'Scenario IV — Parasitism (N↑ P↑)',
    }

    def evaluate(val, r_min, r_max, is_ratio=False):

        if r_min <= val <= r_max:

            return "🟢 correct"

        elif val < r_min:

            if val < 0.01 and not is_ratio:
                return "🔴 near zero"

            if is_ratio:
                return "🟠 low ratio"

            return "🔴 low"

        else:

            if val > 0.95 and r_max <= 0.95:
                return "🟠 spurious/high"

            return "🔴 too high"

    for scen in scenarios:

        print(f"\n{scen_titles[scen]}")

        print(
            f"{'Indicator':<25} | "
            f"{'Expected':<15} | "
            f"{'Obtained':<10} | "
            f"{'Status'}"
        )

        print("-" * 75)

        r = mf_results[scen]

        v1 = r['v1']

        cd = r['C_demand']

        pb = r['P_benefit']

        # ------------------------------------
        # 1. C Supply (v1)
        # ------------------------------------

        rng_v1 = ranges[scen]['v1']

        st_v1 = evaluate(
            v1,
            rng_v1[0],
            rng_v1[1]
        )

        print(
            f"{'C supply (v1)':<25} | "
            f"{rng_v1[0]:.2f} - {rng_v1[1]:.2f}     | "
            f"{v1:<10.4f} | "
            f"{st_v1}"
        )

        # ------------------------------------
        # 2. C Demand (v13/v1)
        # ------------------------------------

        rng_rat = ranges[scen]['ratio']

        st_rat = evaluate(
            cd,
            rng_rat[0],
            rng_rat[1]
        )

        print(
            f"{'C demand (v13/v1)':<25} | "
            f"{rng_rat[0]:.2f} - {rng_rat[1]:.2f}     | "
            f"{cd:<10.4f} | "
            f"{st_rat}"
        )

        # ------------------------------------
        # 3. P Benefit
        # ------------------------------------

        rng_pb = ranges[scen]['pb']

        st_pb = evaluate(
            pb,
            rng_pb[0],
            rng_pb[1]
        )

        print(
            f"{'P benefit (v15/(v15+v6))':<25} | "
            f"{rng_pb[0]:.2f} - {rng_pb[1]:.2f}     | "
            f"{pb:<10.4f} | "
            f"{st_pb}"
        )

    print("\n" + "=" * 80)

    return mf_results

# ========================================
# MAIN
# ========================================
if __name__ == '__main__':
    FILE_PATH = 'data/Ecological_models/AMF_2.txt'
    rn = read_txt(FILE_PATH)
    
    print(f"Species:    {[s.name for s in rn.species()]}")
    print(f"Reactions:  {[r.name() for r in rn.reactions()]}")
    
    save_base = os.path.join(VIS_DIR, '3_trade_balance', 'trade_balance_v6')
    os.makedirs(os.path.dirname(save_base), exist_ok=True)
    
    results = plot_trade_balance_model(rn, save_path=save_base)
    mf_results = plot_mycorrhizal_function(rn, save_path=save_base)
    
    print("\nSimulation completed successfully.") 