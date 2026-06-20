# ==========================================
# MONTE CARLO ROBUSTNESS ANALYSIS FOR AMF
# WITH COLOR-CODED TRAJECTORIES BY INDICATOR - CORREGIDO
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pyCOT.simulations.ode import simulation

# ========================================
# BIOLOGICAL BASE PARAMETERS
# ========================================

x0_dict = {
    'Pact': 5.5,    # [1.00-7.50] g biomass/kg soil
    'Plim': 1.5,    # [0.50-2.50] g biomass/kg soil
    'Cp':   45.0,   # [35.0-50.0] % leaf C
    'Np':   2.0,    # [1.00-3.00] % leaf N
    'Pp':   0.4,    # [0.05-0.88] % leaf P
    'F':    20,     # [6.00-90.0] % root length colonized
    'M':    2.0,    # [0.10-3.00] m mycelium/g soil
    'Cf':   40.0,   # [35.0-50.0] % fungal C
    'Nf':   5.0,    # [1.00-6.00] % fungal N
    'Pf':   0.5,    # [0.10-2.00] % fungal P
    'Ns':   30,     # [10-40] mg N/kg dry soil
    'Ps':   25      # [5-20] mg P/kg dry soil
}

_SPECIES_ORDER = [
    'Pact', 'Plim', 'Cp', 'Np', 'Pp',
    'F', 'M', 'Cf', 'Nf', 'Pf',
    'Ns', 'Ps'
]


def create_base_params():
    return [
        [2.5, 10.0],   # r1(mmk):  Photosynthesis
        [1e-5],        # r2(mak):  Activation
        [0.02],        # r3(mak):  Senescence
        [8e-6],        # r4(mak):  Plant growth
        [5e-4],        # r5(mak):  Direct N uptake by roots
        [0.005],       # r6(mak):  Direct P uptake by roots
        [5e-8],        # r7(mak):  Pact recycling
        [5e-3],        # r8(mak):  Plim recycling
        [1.5, 30.0],   # r9(mmk):  Mycelial N uptake
        [1.2, 10.0],   # r10(mmk): Mycelial P uptake
        [5e-7],        # r11(mak): Mycelium growth
        [0.01],        # r12(mak): Mycelium mortality
        [0.0001],      # r13(mak): Fungal C uptake
        [0.8, 20.0],   # r14(mmk): F to plant N transfer
        [0.5, 20.0],   # r15(mmk): F to plant P transfer
        [6e-7],        # r16(mak): Fungal growth
        [0.001],       # r17(mak): Fungal mortality
        [0.0],         # r18(mak): N input
        [0.0],         # r19(mak): P input
    ]


def create_rate_list():
    return [
        'mmk', 'mak', 'mak', 'mak', 'mak', 'mak', 'mak', 'mak',
        'mmk', 'mmk', 'mak', 'mak', 'mak', 'mmk', 'mmk', 'mak',
        'mak', 'mak', 'mak'
    ]

def get_scenario_config(scenario: str, species_order=None):

    if species_order is None:
        species_order = _SPECIES_ORDER

    params  = create_base_params()
    
    # IMPORTANTE: Creamos una copia local para no modificar el x0_dict global
    local_x0_dict = x0_dict.copy()
    
    IDX_R18 = 17
    IDX_R19 = 18

    if scenario == 'I':
        local_x0_dict['Ns'] = 15
        local_x0_dict['Ps'] = 5
        params[IDX_R18][0] = 0.001
        params[IDX_R19][0] = 0.012
        params_name   = "I – N↓ P↓"
        scenario_name = "I – C-limited mutualism"

    elif scenario == 'II':
        local_x0_dict['Ns'] = 45
        local_x0_dict['Ps'] = 5
        params[IDX_R18][0] = 0.5
        params[IDX_R19][0] = 0.12
        params_name   = "II – N↑ P↓"
        scenario_name = "II – Strong mutualism"

    elif scenario == 'III':
        local_x0_dict['Ns'] = 5
        local_x0_dict['Ps'] = 50
        params[IDX_R18][0] = 0.001
        params[IDX_R19][0] = 2.0
        params_name   = "III – N↓ P↑"
        scenario_name = "III – Commensalism"

    elif scenario == 'IV':
        local_x0_dict['Ns'] = 90
        local_x0_dict['Ps'] = 100
        params[IDX_R18][0] = 0.01
        params[IDX_R19][0] = 5
        params_name   = "IV – N↑ P↑"
        scenario_name = "IV – Parasitism"

    else:
        raise ValueError("Invalid scenario")

    # Extraemos basándonos en la copia local modificada
    x0 = [local_x0_dict[s] for s in species_order]

    return params, x0, params_name, scenario_name



# ==========================================
# PERTURBATION UTILITIES
# ==========================================

def perturb_value(val, variance=0.10):
    """
    Gaussian perturbation around nominal value.

    sigma = variance * nominal value
    """

    sigma = abs(val) * variance

    sampled = np.random.normal(
        loc=val,
        scale=sigma
    )

    # avoid negative biological values
    return max(sampled, 0.0)


def perturb_vector(vec, variance=0.10):

    return [
        perturb_value(v, variance)
        for v in vec
    ]


def perturb_nested_params(params, variance=0.10):
    """
    Perturb list-of-lists parameter structure.
    """

    perturbed = []

    for group in params:

        new_group = [
            perturb_value(v, variance)
            for v in group
        ]

        perturbed.append(new_group)

    return perturbed


# ==========================================
# STEADY STATE INDICATORS
# ==========================================

def compute_steady_state_indicators(
    flux_vector,
    t_span,
    steady_frac=0.2
):
    """
    Compute steady-state indicators.
    """

    t_ss = t_span[1] - steady_frac * (
        t_span[1] - t_span[0]
    )

    idx = np.searchsorted(
        flux_vector['Time'].values,
        t_ss
    )

    def ss(rxn):

        return float(
            flux_vector[rxn].iloc[idx:].mean()
        )

    c_supply = ss('r1')
    v13 = ss('r13')
    v15 = ss('r15')
    v6  = ss('r6')

    c_demand = v13 / (c_supply + 1e-12)

    p_benefit = v15 / (v15 + v6 + 1e-12)

    return {
        'c_supply': c_supply,
        'c_demand': c_demand,
        'p_benefit': p_benefit
    }


# ==========================================
# ECOLOGICAL / JOHNSON CONDITIONS
# ==========================================

def evaluate_johnson_conditions(
    indicators,
    scenario
):
    """
    Evaluate ecological inequalities:
    c_supply  = v1
    c_demand  = v13 / v1
    p_benefit = v15 / (v15 + v6)
    """
    c_supply  = indicators['c_supply']
    c_demand = indicators['c_demand']
    p_benefit    = indicators['p_benefit']
    L_high = 0.5
    S_high = 0.25
    C_high = 0.75
    P_high = 0.25
    if scenario == 'I':
        cond_order = (p_benefit > c_supply >= c_demand)
        cond_p_benefit = (p_benefit >= L_high)
        cond_c_supply = (c_supply < L_high)
        all_ok = (
            cond_order
            and cond_p_benefit
            and cond_c_supply
        )
        return {
            'order_ok': cond_order,
            'p_benefit_ok': cond_p_benefit,
            'c_supply_ok': cond_c_supply,
            'all_ok': all_ok
        }
    elif scenario == 'II':
        cond_c_demand = (c_supply > c_demand > S_high)
        cond_p_benefit = (p_benefit > S_high)
        cond_order = (c_supply > p_benefit > S_high)
        all_ok = (cond_c_demand and cond_p_benefit and cond_order)
        return {
            'c_demand_ok': cond_c_demand,
            'p_benefit_ok': cond_p_benefit,
            'order_ok': cond_order,
            'all_ok': all_ok
        }
    elif scenario == 'III':
        cond_c_supply = (c_demand < c_supply <= C_high)
        cond_p_benefit = (c_demand < p_benefit <= C_high)
        all_ok = (cond_c_supply and cond_p_benefit)
        return {
            'c_supply_ok': cond_c_supply,
            'p_benefit_ok': cond_p_benefit,
            'all_ok': all_ok
        } 
    elif scenario == 'IV':
        cond_order = (
            p_benefit < P_high < c_supply < c_demand 
        )

        return {
            'order_ok': cond_order,
            'all_ok': cond_order
        }

    else:

        raise ValueError("Invalid scenario")


# ==========================================
# VISUALIZATION CONFIGURATION
# ==========================================

# Colores base para cada indicador - CLAVES en minúsculas para coincidir con trajectories
INDICATOR_BASE_COLORS = {
    'c_supply':  '#2ECC71',  # Verde
    'c_demand':  '#E67E22',  # Naranja
    'p_benefit': '#3498DB'   # Azul
}

# Nombres legibles para la leyenda
INDICATOR_NAMES = {
    'c_supply': 'C supply',
    'c_demand': 'C demand',
    'p_benefit': 'P benefit'
}

SCENARIO_COLORS = {
    'I':   '#2E86C1',  # Azul
    'II':  '#28B463',  # Verde
    'III': '#E67E22',  # Naranja
    'IV':  '#E74C3C'   # Rojo
}

SCENARIO_LABELS = {
    'I':   'I – C-limited mutualism (N↓P↓)',
    'II':  'II – Strong mutualism (N↑P↓)',
    'III': 'III – Commensalism (N↓P↑)',
    'IV':  'IV – Parasitism (N↑P↑)',
}


# ==========================================
# FUNCTION TO COLLECT TRAJECTORIES WITH SUCCESS/FAILURE
# ==========================================

def collect_monte_carlo_trajectories_with_labels(
    rn,
    scenario,
    variance=0.10,
    n_simulations=100,
    t_span=(0, 360*2),
    n_steps=500,  # ← CORREGIDO: ahora 500 para coincidir con run_single_scenario
    steady_frac=0.2
):
    """
    Collects ALL trajectories from Monte Carlo simulations for a given scenario,
    along with their success/failure labels based on ALL CONDITIONS.
    Returns trajectories separated by indicator and by success/failure.
    """
    
    # from script_AMF_3 import get_scenario_config, create_rate_list
    
    rate_list = create_rate_list()
    species_order = [s.name for s in rn.species()]
    
    # Obtener configuración base
    params_base, x0_base, _, _ = get_scenario_config(scenario, species_order)
    
    # Almacenar TODAS las trayectorias con sus etiquetas
    # Estructura: trajectories[indicador][status] = lista de arrays
    trajectories = {
        'time': None,
        'c_supply': {'success': [], 'failure': []},
        'c_demand': {'success': [], 'failure': []},
        'p_benefit': {'success': [], 'failure': []}
    }
    
    successful_sims = 0
    failed_sims = 0
    all_ok_count = 0
    
    for k in range(n_simulations):
        if (k + 1) % 20 == 0:
            print(f"    Simulation {k+1}/{n_simulations} (success_cond: {all_ok_count}, success_sim: {successful_sims}, failed_sim: {failed_sims})")
        
        # Perturbación
        x0_mc = perturb_vector(x0_base, variance)
        params_mc = perturb_nested_params(params_base, variance)
        
        try:
            # Simulación
            ts, flux = simulation(
                rn, x0=x0_mc, rate=rate_list, spec_vector=params_mc,
                t_span=t_span, n_steps=n_steps,
                method='LSODA', rtol=1e-8, atol=1e-10
            )
            
            # Calcular indicadores a lo largo del tiempo
            t = flux['Time'].values
            c_supply = flux['r1'].values
            v13 = flux['r13'].values
            v15 = flux['r15'].values
            v6 = flux['r6'].values
            c_demand = v13 / (c_supply + 1e-12)
            p_benefit = v15 / (v15 + v6 + 1e-12)
            
            # Calcular indicadores en estado estacionario para evaluar condición
            indicators_ss = compute_steady_state_indicators(flux, t_span, steady_frac)
            
            # Evaluar si cumple ALL CONDITIONS
            ev = evaluate_johnson_conditions(indicators_ss, scenario)
            is_success = ev['all_ok']
            
            # Guardar tiempo (solo una vez)
            if trajectories['time'] is None:
                trajectories['time'] = t
            
            # Guardar trayectoria en la categoría correspondiente
            if is_success:
                trajectories['c_supply']['success'].append(c_supply)
                trajectories['c_demand']['success'].append(c_demand)
                trajectories['p_benefit']['success'].append(p_benefit)
                all_ok_count += 1
            else:
                trajectories['c_supply']['failure'].append(c_supply)
                trajectories['c_demand']['failure'].append(c_demand)
                trajectories['p_benefit']['failure'].append(p_benefit)
            
            successful_sims += 1
            
        except Exception as e:
            failed_sims += 1
            continue
    
    print(f"    Final: {all_ok_count} ALL_CONDITIONS_OK, {successful_sims - all_ok_count} FAILED_CONDITIONS, {failed_sims} simulation failures")
    
    # Convertir a arrays numpy
    for indicator in ['c_supply', 'c_demand', 'p_benefit']:
        trajectories[indicator]['success'] = np.array(trajectories[indicator]['success']) if trajectories[indicator]['success'] else np.array([])
        trajectories[indicator]['failure'] = np.array(trajectories[indicator]['failure']) if trajectories[indicator]['failure'] else np.array([])
    
    return trajectories, successful_sims, failed_sims, all_ok_count


def plot_monte_carlo_trajectories_by_indicator(
    rn,
    all_trajectories,
    show_success=True,
    show_failure=True,
    show_legend=True,
    show_median=False,  # ← NUEVO PARÁMETRO
    save_path=None
):
    """
    Creates 2x2 figure with trajectories colored by INDICATOR:
    - c_supply:  GREEN
    - c_demand:  ORANGE
    - p_benefit: BLUE
    
    Success vs Failure distinguished by INTENSITY/OPACITY:
    - SUCCESS: saturated colors (alpha based on density)
    - FAILURE: faded colors (higher transparency, alpha=0.15)
    """
    
    scenarios = ['I', 'II', 'III', 'IV']
    
    # Layout para subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    layout = {
        'III': (0, 0), 
        'IV': (0, 1), 
        'I': (1, 0), 
        'II': (1, 1)
    }
    
    for scen in scenarios:
        row, col = layout[scen]
        ax = axes[row][col]
        
        if scen not in all_trajectories:
            ax.text(0.5, 0.5, f'No data for scenario {scen}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        trajectories = all_trajectories[scen]['trajectories']
        successful_cond = all_trajectories[scen]['all_ok_count']
        total_sims = all_trajectories[scen]['successful_sims']
        failed_cond = total_sims - successful_cond
        
        if total_sims == 0:
            ax.text(0.5, 0.5, f'No successful simulations\nfor scenario {scen}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        time = trajectories['time']
        
        # ==========================================
        # PLOTEAR CADA INDICADOR POR SEPARADO
        # ==========================================
        for indicator, color in INDICATOR_BASE_COLORS.items():
            
            # ==========================================
            # TRAYECTORIAS EXITOSAS (solo si show_success=True)
            # ==========================================
            if show_success and successful_cond > 0:
                n_traj_success = len(trajectories[indicator]['success'])
                if n_traj_success > 0:
                    alpha_success = 0.4
                    linewidth = 0.6
                    
                    # Plotear todas las trayectorias exitosas individuales
                    for i in range(n_traj_success):
                        ax.plot(time, trajectories[indicator]['success'][i], 
                            color=color, alpha=alpha_success, linewidth=linewidth)
            
            # ==========================================
            # MEDIANA DE ÉXITO (independiente de show_success)
            # ==========================================
            if show_median and successful_cond > 0:
                n_traj_success = len(trajectories[indicator]['success'])
                if n_traj_success > 0:
                    median_success = np.median(trajectories[indicator]['success'], axis=0)
                    ax.plot(time, median_success, 
                        color=color, 
                        linewidth=3.0, 
                        alpha=1.0,
                        linestyle='-')
            
            # ==========================================
            # TRAYECTORIAS FALLIDAS (PUNTEADAS)
            # ==========================================
            if show_failure and failed_cond > 0:
                n_traj_failure = len(trajectories[indicator]['failure'])
                if n_traj_failure > 0:
                    alpha_failure = 0.2
                    linewidth = 0.6
                    
                    # Plotear todas las trayectorias fallidas individuales
                    for i in range(n_traj_failure):
                        ax.plot(time, trajectories[indicator]['failure'][i], 
                            color=color, 
                            alpha=alpha_failure, 
                            linewidth=linewidth,
                            linestyle='--')         
        
        # ==========================================
        # AÑADIR LÍNEAS DE REFERENCIA PARA LA LEYENDA
        # ==========================================
        if show_legend:
            from matplotlib.lines import Line2D
            legend_elements = []
            
            for indicator, color in INDICATOR_BASE_COLORS.items():
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=2.5, 
                          label=INDICATOR_NAMES[indicator])
                )
            
            # Añadir mediana a la leyenda si está activada
            if show_median and successful_cond > 0:
                legend_elements.append(
                    Line2D([0], [0], color='black', linewidth=2.5, 
                          linestyle='-', label=f'Median (Success)')
                )
            
            # Añadir fallos a la leyenda
            if show_failure and failed_cond > 0:
                legend_elements.append(
                    Line2D([0], [0], color='#E74C3C', linewidth=2, 
                          linestyle='--', label=f'Failure (n={failed_cond})', alpha=0.6)
                )
            
            ax.legend(handles=legend_elements, fontsize=8, loc='best')
        
        # ==========================================
        # CONFIGURAR GRÁFICO
        # ==========================================
        success_rate = 100 * successful_cond / total_sims if total_sims > 0 else 0
        
        ax.set_title(f"{SCENARIO_LABELS[scen]}\n(Success rate: {success_rate:.1f}%)", 
                    fontweight='bold', fontsize=11, color=SCENARIO_COLORS[scen])
        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel("Indicator values", fontsize=10)
        
        # Ajustar límites y
        all_values = []
        if show_failure:
            for indicator in ['c_supply', 'c_demand', 'p_benefit']:
                if len(trajectories[indicator]['failure']) > 0:
                    all_values.extend(trajectories[indicator]['failure'].flatten())
        
        # También incluir valores de la mediana para límites
        if show_median and successful_cond > 0:
            for indicator in ['c_supply', 'c_demand', 'p_benefit']:
                if len(trajectories[indicator]['success']) > 0:
                    median_val = np.median(trajectories[indicator]['success'], axis=0)
                    all_values.extend(median_val)
        
        if all_values:
            max_val = np.percentile(all_values, 99)
            ax.set_ylim(0, max_val * 1.1)
        
        ax.grid(True, alpha=0.25)
        
        # Añadir caja con estadísticas
        stats_text = f"Total: {total_sims}\n✓ Success: {successful_cond} ({success_rate:.1f}%)\n✗ Failure: {failed_cond}"
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Añadir etiquetas globales
    fig.text(0.53, 0.02,
             r"Limitation $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$ Luxury",
             ha='center', fontsize=14, color='#555')
    fig.text(0.53, 0.001, r"N supply",
             ha='center', fontsize=18, fontweight='bold', color='#555')
    fig.text(0.001, 0.53, r"P supply",
             va='center', rotation='vertical', fontsize=18, fontweight='bold', color='#555')
    fig.text(0.02, 0.52,
             r"Limitation $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$ Luxury",
             va='center', rotation='vertical', fontsize=14, color='#555')
    
    # Añadir título general
    fig.suptitle('Monte Carlo Trajectories - Color Coding:\n'
                'Green = C_supply | Orange = C_demand | Blue = P_benefit\n'
                'Solid = Median of Success | Dashed = Failure trajectories',
                fontsize=12, y=0.98, fontweight='bold')
    
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: {save_path}")
    
    plt.show()
    
    return fig


# ==========================================
# MAIN MONTE CARLO FUNCTION
# ==========================================

# def monte_carlo_amf_robustness(
#     rn,
#     scenario='all',
#     variance=0.10,
#     n_simulations=100,
#     t_span=(0, 360),
#     n_steps=500,
#     steady_frac=0.2,
#     random_seed=123,
#     plot_trajectories=True,
#     show_success=True,
#     show_failure=True,
#     show_median=False,  # ← NUEVO PARÁMETRO
#     save_figure_path=None
# ):

#     np.random.seed(random_seed)

#     # ==========================================
#     # MODO: EJECUTAR TODOS LOS ESCENARIOS
#     # ==========================================
    
#     if scenario == 'all':
#         scenarios_to_run = ['I', 'II', 'III', 'IV']
        
#         print("=" * 70)
#         print(" MONTE CARLO ROBUSTNESS ANALYSIS - ALL SCENARIOS ")
#         print("=" * 70)
#         print(f"Variance:      {variance*100:.1f}%")
#         print(f"Simulations:   {n_simulations}")
#         print(f"n_steps:       {n_steps}")  # ← AÑADIDO: mostrar n_steps para verificar
#         print(f"Plot mode:     COLOR-CODED BY INDICATOR (Green=C_supply, Orange=C_demand, Blue=P_benefit)")
#         print("=" * 70)
        
#         # Diccionario para almacenar resultados de todos los escenarios
#         all_results = {}
#         all_trajectories = {}
        
#         for sc in scenarios_to_run:
#             print(f"\n\n{'='*70}")
#             print(f" EXECUTING SCENARIO {sc} ")
#             print(f"{'='*70}")
            
#             # Ejecutar el escenario individual (solo estadísticas de estado estacionario)
#             df_scenario = run_single_scenario(
#                 rn, sc, variance, n_simulations, 
#                 t_span, n_steps, steady_frac
#             )
#             all_results[sc] = df_scenario
            
#             # Recolectar TODAS las trayectorias con etiquetas de éxito/fracaso
#             if plot_trajectories:
#                 print(f"\n  Collecting trajectories for scenario {sc} with success/failure labels...")
#                 trajectories, n_success, n_failed, n_ok = collect_monte_carlo_trajectories_with_labels(
#                     rn, sc, variance, n_simulations, t_span, n_steps, steady_frac  # ← n_steps se pasa correctamente
#                 )
#                 all_trajectories[sc] = {
#                     'trajectories': trajectories,
#                     'successful_sims': n_success,
#                     'failed_sims': n_failed,
#                     'all_ok_count': n_ok
#                 }
        
#         # Crear tabla resumen
#         print("\n\n")
#         print("=" * 70)
#         print(" SUMMARY TABLE - ALL SCENARIOS ")
#         print("=" * 70)
#         print()
        
#         # Crear DataFrame resumen
#         summary_data = []
        
#         for sc in scenarios_to_run:
#             df_sc = all_results[sc]
            
#             # Calcular estadísticas para ALL CONDITIONS
#             all_ok_pct = 100 * df_sc['all_ok'].sum() / len(df_sc)
            
#             # Calcular medias de indicadores
#             mean_c_supply = df_sc['c_supply'].mean()
#             mean_c_demand = df_sc['c_demand'].mean()
#             mean_p_benefit = df_sc['p_benefit'].mean()
            
#             # Añadir información de simulaciones exitosas
#             n_success = all_trajectories[sc]['successful_sims'] if plot_trajectories else n_simulations
            
#             summary_data.append({
#                 'Scenario': sc,
#                 'Success Rate (%)': f'{all_ok_pct:.2f}',
#                 'Mean c_supply': f'{mean_c_supply:.4f}',
#                 'Mean c_demand': f'{mean_c_demand:.4f}',
#                 'Mean p_benefit': f'{mean_p_benefit:.4f}',
#                 'Valid Sims': f'{n_success}/{n_simulations}'
#             })
        
#         summary_df = pd.DataFrame(summary_data)
#         print(summary_df.to_string(index=False))
#         print("\n" + "=" * 70)
        
#         # También mostrar tabla detallada por condición para cada escenario
#         print("\n\n")
#         print("=" * 70)
#         print(" DETAILED CONDITIONS BY SCENARIO ")
#         print("=" * 70)
        
#         for sc in scenarios_to_run:
#             df_sc = all_results[sc]
            
#             print(f"\n{'='*70}")
#             print(f" SCENARIO {sc} ")
#             print(f"{'='*70}")
            
#             # Obtener las columnas de condiciones
#             condition_cols = [col for col in df_sc.columns 
#                              if col.endswith('_ok') and col != 'all_ok']
            
#             for col in condition_cols:
#                 pct = 100 * df_sc[col].sum() / len(df_sc)
#                 print(f"{col:<20}: {pct:.2f}%")
            
#             print("-" * 70)
#             all_ok_pct = 100 * df_sc['all_ok'].sum() / len(df_sc)
#             print(f"{'ALL CONDITIONS':<20}: {all_ok_pct:.2f}%")
            
#             print("\nIndicator statistics:")
#             print(df_sc[['c_supply', 'c_demand', 'p_benefit']].describe())
            
#             if plot_trajectories:
#                 print(f"\nTrajectory collection: {all_trajectories[sc]['successful_sims']} successful simulations")
#                 print(f"  - ALL_CONDITIONS_OK: {all_trajectories[sc]['all_ok_count']}")
#                 print(f"  - CONDITIONS_FAILED: {all_trajectories[sc]['successful_sims'] - all_trajectories[sc]['all_ok_count']}")
                
#                 # VERIFICACIÓN: Mostrar si coinciden
#                 table_ok = all_ok_pct
#                 trajectory_ok = 100 * all_trajectories[sc]['all_ok_count'] / all_trajectories[sc]['successful_sims']
#                 print(f"\n  ✓ VERIFICACIÓN: Tabla ALL CONDITIONS = {table_ok:.1f}% | Trajectory ALL_CONDITIONS_OK = {trajectory_ok:.1f}%")
#                 if abs(table_ok - trajectory_ok) < 0.1:
#                     print(f"  ✓ COINCIDEN CORRECTAMENTE")
#                 else:
#                     print(f"  ✗ NO COINCIDEN - Revisar parámetros")
        
#         # ==========================================
#         # GENERAR FIGURA POR INDICADOR
#         # ==========================================
#         if plot_trajectories:
#             print("\n\n")
#             print("=" * 70)
#             print(" GENERATING MONTE CARLO TRAJECTORY FIGURE ")
#             print(" GREEN = C_supply | ORANGE = C_demand | BLUE = P_benefit")
#             print(" Saturated = SUCCESS | Faded = FAILURE")
#             print("=" * 70)
            
#             fig = plot_monte_carlo_trajectories_by_indicator(
#                 rn,
#                 all_trajectories,
#                 show_success=show_success,
#                 show_failure=show_failure,
#                 show_legend=True,
#                 show_median=show_median,  # ← PASAR EL PARÁMETRO
#                 save_path=save_figure_path
#             )
        
#         # Retornar diccionario con todos los resultados
#         return all_results, all_trajectories
    
#     # ==========================================
#     # MODO: ESCENARIO INDIVIDUAL
#     # ==========================================
#     else:
#         df_result = run_single_scenario(
#             rn, scenario, variance, n_simulations,
#             t_span, n_steps, steady_frac
#         )
        
#         # Recolectar trayectorias con etiquetas si se solicita
#         if plot_trajectories:
#             print(f"\n  Collecting trajectories for scenario {scenario} with success/failure labels...")
#             trajectories, n_success, n_failed, n_ok = collect_monte_carlo_trajectories_with_labels(
#                 rn, scenario, variance, n_simulations, t_span, n_steps, steady_frac
#             )
            
#             # Crear figura para un solo escenario
#             fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
#             time = trajectories['time']
#             failed_cond = n_success - n_ok
            
#             # Plotear cada indicador
#             for indicator, color in INDICATOR_BASE_COLORS.items():
#                 # Éxitos
#                 if n_ok > 0 and len(trajectories[indicator]['success']) > 0:
#                     alpha_success = max(0.08, min(0.4, 20.0 / n_ok))
#                     for i in range(len(trajectories[indicator]['success'])):
#                         ax.plot(time, trajectories[indicator]['success'][i], 
#                                color=color, alpha=alpha_success, linewidth=0.6)
                
#                 # Fallos
#                 if failed_cond > 0 and len(trajectories[indicator]['failure']) > 0:
#                     for i in range(len(trajectories[indicator]['failure'])):
#                         ax.plot(time, trajectories[indicator]['failure'][i], 
#                                color=color, alpha=0.15, linewidth=0.4)
            
#             # Configurar gráfico
#             success_rate = 100 * n_ok / n_success if n_success > 0 else 0
            
#             ax.set_title(f'Monte Carlo Trajectories - Scenario {scenario}\n'
#                         f'Success rate: {success_rate:.1f}% ({n_ok} simulations) | Failure: {failed_cond} simulations', 
#                         fontweight='bold', fontsize=12)
#             ax.set_xlabel("Time (days)", fontsize=10)
#             ax.set_ylabel("Indicator values", fontsize=10)
            
#             # Leyenda
#             from matplotlib.lines import Line2D
#             legend_elements = []
#             for indicator, color in INDICATOR_BASE_COLORS.items():
#                 legend_elements.append(
#                     Line2D([0], [0], color=color, linewidth=2.5, label=INDICATOR_NAMES[indicator])
#                 )
#             legend_elements.append(
#                 Line2D([0], [0], color='gray', linewidth=2, label=f'Success (n={n_ok})', alpha=0.8)
#             )
#             legend_elements.append(
#                 Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', 
#                       label=f'Failure (n={failed_cond})', alpha=0.5)
#             )
#             ax.legend(handles=legend_elements, fontsize=10, loc='best')
            
#             ax.grid(True, alpha=0.25)
            
#             # Ajustar límites
#             all_values = []
#             for indicator in ['c_supply', 'c_demand', 'p_benefit']:
#                 if n_ok > 0 and len(trajectories[indicator]['success']) > 0:
#                     all_values.extend(trajectories[indicator]['success'].flatten())
#                 if failed_cond > 0 and len(trajectories[indicator]['failure']) > 0:
#                     all_values.extend(trajectories[indicator]['failure'].flatten())
            
#             if all_values:
#                 max_val = np.percentile(all_values, 99)
#                 ax.set_ylim(0, max_val * 1.1)
            
#             if save_figure_path:
#                 plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
#                 print(f"Figure saved: {save_figure_path}")
            
#             plt.show()
        
#         return df_result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_amf_robustness(
    rn,
    scenario='all',
    variance=0.10,
    n_simulations=100,
    t_span=(0, 360*2),
    n_steps=500,
    steady_frac=0.2,
    random_seed=123,
    plot_trajectories=True,
    show_success=True,
    show_failure=True,
    show_median=False,
    save_figure_path=None,
    save_excel_path=None  # ← NUEVO PARÁMETRO PARA EL EXCEL
):

    np.random.seed(random_seed)

    # ==========================================
    # MODO: EJECUTAR TODOS LOS ESCENARIOS
    # ==========================================
    
    if scenario == 'all':
        scenarios_to_run = ['I', 'II', 'III', 'IV']
        
        print("=" * 70)
        print(" MONTE CARLO ROBUSTNESS ANALYSIS - ALL SCENARIOS ")
        print("=" * 70)
        print(f"Variance:      {variance*100:.1f}%")
        print(f"Simulations:   {n_simulations}")
        print(f"n_steps:       {n_steps}") 
        print(f"Plot mode:     COLOR-CODED BY INDICATOR (Green=C_supply, Orange=C_demand, Blue=P_benefit)")
        print("=" * 70)
        
        # Diccionario para almacenar resultados de todos los escenarios
        all_results = {}
        all_trajectories = {}
        
        for sc in scenarios_to_run:
            print(f"\n\n{'='*70}")
            print(f" EXECUTING SCENARIO {sc} ")
            print(f"{'='*70}")
            
            # Ejecutar el escenario individual
            df_scenario = run_single_scenario(
                rn, sc, variance, n_simulations, 
                t_span, n_steps, steady_frac
            )
            all_results[sc] = df_scenario
            
            # Recolectar TODAS las trayectorias con etiquetas de éxito/fracaso
            if plot_trajectories:
                print(f"\n  Collecting trajectories for scenario {sc} with success/failure labels...")
                trajectories, n_success, n_failed, n_ok = collect_monte_carlo_trajectories_with_labels(
                    rn, sc, variance, n_simulations, t_span, n_steps, steady_frac
                )
                all_trajectories[sc] = {
                    'trajectories': trajectories,
                    'successful_sims': n_success,
                    'failed_sims': n_failed,
                    'all_ok_count': n_ok
                }
        
        # Crear tabla resumen
        print("\n\n")
        print("=" * 70)
        print(" SUMMARY TABLE - ALL SCENARIOS ")
        print("=" * 70)
        print()
        
        summary_data = []
        for sc in scenarios_to_run:
            df_sc = all_results[sc]
            all_ok_pct = 100 * df_sc['all_ok'].sum() / len(df_sc)
            mean_c_supply = df_sc['c_supply'].mean()
            mean_c_demand = df_sc['c_demand'].mean()
            mean_p_benefit = df_sc['p_benefit'].mean()
            n_success = all_trajectories[sc]['successful_sims'] if plot_trajectories else n_simulations
            
            summary_data.append({
                'Scenario': sc,
                'Success Rate (%)': f'{all_ok_pct:.2f}',
                'Mean c_supply': f'{mean_c_supply:.4f}',
                'Mean c_demand': f'{mean_c_demand:.4f}',
                'Mean p_benefit': f'{mean_p_benefit:.4f}',
                'Valid Sims': f'{n_success}/{n_simulations}'
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print("\n" + "=" * 70)
        
        # Mostrar tabla detallada por condición para cada escenario
        for sc in scenarios_to_run:
            df_sc = all_results[sc]
            print(f"\n{'='*70}\n SCENARIO {sc} \n{'='*70}")
            condition_cols = [col for col in df_sc.columns if col.endswith('_ok') and col != 'all_ok']
            for col in condition_cols:
                pct = 100 * df_sc[col].sum() / len(df_sc)
                print(f"{col:<20}: {pct:.2f}%")
            print("-" * 70)
            all_ok_pct = 100 * df_sc['all_ok'].sum() / len(df_sc)
            print(f"{'ALL CONDITIONS':<20}: {all_ok_pct:.2f}%")
            
            if plot_trajectories:
                table_ok = all_ok_pct
                trajectory_ok = 100 * all_trajectories[sc]['all_ok_count'] / all_trajectories[sc]['successful_sims']
                if abs(table_ok - trajectory_ok) < 0.1:
                    pass # Coinciden
                else:
                    print(f"  ✗ NO COINCIDEN - Revisar parámetros")
        
        # ==========================================
        # GUARDAR RESULTADOS EN EXCEL
        # ==========================================
        if save_excel_path:
            print(f"\nSaving results to Excel: {save_excel_path} ...")
            try:
                with pd.ExcelWriter(save_excel_path) as writer:
                    # Crear un DataFrame con los parámetros usados
                    params_df = pd.DataFrame({
                        'Parameter': ['Variance', 'Simulations', 't_span', 'n_steps', 'steady_frac', 'random_seed'],
                        'Value': [variance, n_simulations, str(t_span), n_steps, steady_frac, random_seed]
                    })
                    
                    for sc in scenarios_to_run:
                        params_df.to_excel(writer, sheet_name=f'Scenario_{sc}', index=False, startrow=0)
                        
                        # Extraer las columnas solicitadas más la columna de éxito general
                        df_to_save = all_results[sc][['x0', 'spec_vector', 'c_supply', 'c_demand', 'p_benefit', 'all_ok']].copy()
                        
                        # Convertir listas/arrays a texto (str) para que Excel no colapse con las sub-listas
                        df_to_save['x0'] = df_to_save['x0'].apply(lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else str(x))
                        df_to_save['spec_vector'] = df_to_save['spec_vector'].apply(lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else str(x))
                        
                        df_to_save.to_excel(writer, sheet_name=f'Scenario_{sc}', index=False, startrow=len(params_df) + 2)
                print("  ✓ Excel saved successfully.")
                
            except KeyError as e:
                print(f"  ✗ Error saving Excel: Faltan columnas. Asegúrate de que tu función 'run_single_scenario' está guardando 'x0' y 'spec_vector' en el DataFrame. Detalle: {e}")
            except Exception as e:
                print(f"  ✗ Error saving Excel: {e}")

        # ==========================================
        # GENERAR FIGURA POR INDICADOR
        # ==========================================
        if plot_trajectories:
            print("\n\n" + "=" * 70)
            print(" GENERATING MONTE CARLO TRAJECTORY FIGURE ")
            print("=" * 70)
            
            fig = plot_monte_carlo_trajectories_by_indicator(
                rn, all_trajectories, show_success=show_success,
                show_failure=show_failure, show_legend=True,
                show_median=show_median, save_path=save_figure_path
            )
        
        return all_results, all_trajectories
    
    # ==========================================
    # MODO: ESCENARIO INDIVIDUAL
    # ==========================================
    else:
        df_result = run_single_scenario(
            rn, scenario, variance, n_simulations,
            t_span, n_steps, steady_frac
        )

        # ==========================================
        # GUARDAR RESULTADOS EN EXCEL (MODO INDIVIDUAL)
        # ==========================================
        if save_excel_path:
            print(f"\nSaving results to Excel: {save_excel_path} ...")
            try:
                with pd.ExcelWriter(save_excel_path) as writer:
                    params_df = pd.DataFrame({
                        'Parameter': ['Scenario', 'Variance', 'Simulations', 't_span', 'n_steps', 'steady_frac', 'random_seed'],
                        'Value': [scenario, variance, n_simulations, str(t_span), n_steps, steady_frac, random_seed]
                    })
                    params_df.to_excel(writer, sheet_name=f'Scenario_{scenario}', index=False, startrow=0)
                    
                    # Extraer las columnas solicitadas más la columna de éxito general
                    df_to_save = df_result[['x0', 'spec_vector', 'c_supply', 'c_demand', 'p_benefit', 'all_ok']].copy()
                    
                    # Convertir listas/arrays a texto (str)
                    df_to_save['x0'] = df_to_save['x0'].apply(lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else str(x))
                    df_to_save['spec_vector'] = df_to_save['spec_vector'].apply(lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else str(x))
                    
                    df_to_save.to_excel(writer, sheet_name=f'Scenario_{scenario}', index=False, startrow=len(params_df) + 2)
                print("  ✓ Excel saved successfully.")
                
            except KeyError as e:
                print(f"  ✗ Error saving Excel: Faltan columnas. Asegúrate de que tu función 'run_single_scenario' está guardando 'x0' y 'spec_vector' en el DataFrame. Detalle: {e}")
            except Exception as e:
                print(f"  ✗ Error saving Excel: {e}")        
        
        # Recolectar trayectorias con etiquetas si se solicita
        if plot_trajectories:
            print(f"\n  Collecting trajectories for scenario {scenario} with success/failure labels...")
            trajectories, n_success, n_failed, n_ok = collect_monte_carlo_trajectories_with_labels(
                rn, scenario, variance, n_simulations, t_span, n_steps, steady_frac
            )
            
            # Crear figura para un solo escenario
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            time = trajectories['time']
            failed_cond = n_success - n_ok
            
            # Plotear cada indicador
            for indicator, color in INDICATOR_BASE_COLORS.items():
                if n_ok > 0 and len(trajectories[indicator]['success']) > 0:
                    alpha_success = max(0.08, min(0.4, 20.0 / n_ok))
                    for i in range(len(trajectories[indicator]['success'])):
                        ax.plot(time, trajectories[indicator]['success'][i], 
                               color=color, alpha=alpha_success, linewidth=0.6)
                if failed_cond > 0 and len(trajectories[indicator]['failure']) > 0:
                    for i in range(len(trajectories[indicator]['failure'])):
                        ax.plot(time, trajectories[indicator]['failure'][i], 
                               color=color, alpha=0.15, linewidth=0.4)
            
            success_rate = 100 * n_ok / n_success if n_success > 0 else 0
            ax.set_title(f'Monte Carlo Trajectories - Scenario {scenario}\n'
                        f'Success rate: {success_rate:.1f}% ({n_ok} simulations) | Failure: {failed_cond} simulations', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel("Indicator values", fontsize=10)
            
            from matplotlib.lines import Line2D
            legend_elements = []
            for indicator, color in INDICATOR_BASE_COLORS.items():
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2.5, label=INDICATOR_NAMES[indicator]))
            legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, label=f'Success (n={n_ok})', alpha=0.8))
            legend_elements.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label=f'Failure (n={failed_cond})', alpha=0.5))
            
            ax.legend(handles=legend_elements, fontsize=10, loc='best')
            ax.grid(True, alpha=0.25)
            
            if save_figure_path:
                plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved: {save_figure_path}")
            
            plt.show()
        
        return df_result

def run_single_scenario(
    rn,
    scenario,
    variance,
    n_simulations,
    t_span,
    n_steps,
    steady_frac
):
    """
    Ejecuta un único escenario de Monte Carlo perturbando sobre la base del escenario actual.
    """
    
    print("=" * 70)
    print(" MONTE CARLO ROBUSTNESS ANALYSIS ")
    print("=" * 70)

    print(f"Scenario:      {scenario}")
    print(f"Variance:      {variance*100:.1f}%")
    print(f"Simulations:   {n_simulations}")

    # --------------------------------------
    # Base configuration
    # --------------------------------------

    from script_AMF_3 import get_scenario_config, create_rate_list
    
    species_order = [
        s.name for s in rn.species()
    ]

    # Aquí obtenemos los parámetros y x0 ESPECÍFICOS para este escenario
    params_base, x0_base, _, scenario_name = \
        get_scenario_config(
            scenario,
            species_order
        )

    rate_list = create_rate_list()

    # --------------------------------------
    # Statistics containers
    # --------------------------------------

    stats = []
    condition_counts = {}
    success_all = 0

    # ======================================
    # MONTE CARLO LOOP
    # ======================================

    for k in range(n_simulations):

        print(
            f"Simulation {k+1}/{n_simulations}",
            end='\r'
        )

        # ----------------------------------
        # Perturb parameters and ICs
        # (Se perturba SOBRE las bases del escenario actual)
        # ----------------------------------

        x0_mc = perturb_vector(
            x0_base,
            variance
        )

        params_mc = perturb_nested_params(
            params_base,
            variance
        )

        try:

            # ------------------------------
            # Run simulation
            # ------------------------------

            ts, flux = simulation(
                rn,
                x0=x0_mc,
                rate=rate_list,
                spec_vector=params_mc,
                t_span=t_span,
                n_steps=n_steps,
                method='LSODA',
                rtol=1e-8,
                atol=1e-10
            )

            # ------------------------------
            # Compute indicators
            # ------------------------------

            indicators = \
                compute_steady_state_indicators(
                    flux,
                    t_span,
                    steady_frac
                )

            # ------------------------------
            # Evaluate conditions
            # ------------------------------

            ev = evaluate_johnson_conditions(
                indicators,
                scenario
            )

            # ------------------------------
            # Update counters
            # ------------------------------

            success_all += int(ev['all_ok'])

            for key in ev.keys():
                if key not in condition_counts:
                    condition_counts[key] = 0
                condition_counts[key] += int(ev[key])

            # ------------------------------
            # Store statistics
            # ------------------------------

            row = {
                'simulation': k + 1,
                'x0': x0_mc,                  
                'spec_vector': params_mc,     
                'c_supply': indicators['c_supply'],
                'c_demand': indicators['c_demand'],
                'p_benefit': indicators['p_benefit']
            }

            for key, val in ev.items():
                row[key] = val

            stats.append(row)

        except Exception as e:

            print(f"\nSimulation failed: {e}")

            stats.append({
                'simulation': k + 1,
                'x0': x0_mc,                  
                'spec_vector': params_mc,     
                'c_supply': np.nan,
                'c_demand': np.nan,
                'p_benefit': np.nan,
                'all_ok': False
            })

    # ======================================
    # FINAL RESULTS
    # ======================================

    df = pd.DataFrame(stats)

    print("\n")
    print("=" * 70)
    print(f" ROBUSTNESS RESULTS – {scenario_name}")
    print("=" * 70)

    for key, count in condition_counts.items():
        pct = 100 * count / n_simulations
        print(f"{key:<20}: {pct:.2f}%")

    print("-" * 70)

    pct_all = 100 * success_all / n_simulations
    print(f"{'ALL CONDITIONS':<20}: {pct_all:.2f}%")

    print("=" * 70)

    # ======================================
    # DESCRIPTIVE STATISTICS
    # ======================================

    print("\nIndicator statistics:\n")

    print(
        df[
            ['c_supply', 'c_demand', 'p_benefit']
        ].describe()
    )

    return df

# def run_single_scenario(
#     rn,
#     scenario,
#     variance,
#     n_simulations,
#     t_span,
#     n_steps,
#     steady_frac
# ):
#     """
#     Ejecuta un único escenario de Monte Carlo.
#     """
    
#     print("=" * 70)
#     print(" MONTE CARLO ROBUSTNESS ANALYSIS ")
#     print("=" * 70)

#     print(f"Scenario:      {scenario}")
#     print(f"Variance:      {variance*100:.1f}%")
#     print(f"Simulations:   {n_simulations}")

#     # --------------------------------------
#     # Base configuration
#     # --------------------------------------

#     # from script_AMF_3 import get_scenario_config, create_rate_list
    
#     species_order = [
#         s.name for s in rn.species()
#     ]

#     params_base, x0_base, _, scenario_name = \
#         get_scenario_config(
#             scenario,
#             species_order
#         )

#     rate_list = create_rate_list()

#     # --------------------------------------
#     # Statistics containers
#     # --------------------------------------

#     stats = []
#     condition_counts = {}
#     success_all = 0

#     # ======================================
#     # MONTE CARLO LOOP
#     # ======================================

#     for k in range(n_simulations):

#         print(
#             f"Simulation {k+1}/{n_simulations}",
#             end='\r'
#         )

#         # ----------------------------------
#         # Perturb parameters and ICs
#         # ----------------------------------

#         x0_mc = perturb_vector(
#             x0_base,
#             variance
#         )

#         params_mc = perturb_nested_params(
#             params_base,
#             variance
#         )

#         try:

#             # ------------------------------
#             # Run simulation
#             # ------------------------------

#             ts, flux = simulation(
#                 rn,
#                 x0=x0_mc,
#                 rate=rate_list,
#                 spec_vector=params_mc,
#                 t_span=t_span,
#                 n_steps=n_steps,
#                 method='LSODA',
#                 rtol=1e-8,
#                 atol=1e-10
#             )

#             # ------------------------------
#             # Compute indicators
#             # ------------------------------

#             indicators = \
#                 compute_steady_state_indicators(
#                     flux,
#                     t_span,
#                     steady_frac
#                 )

#             # ------------------------------
#             # Evaluate conditions
#             # ------------------------------

#             ev = evaluate_johnson_conditions(
#                 indicators,
#                 scenario
#             )

#             # ------------------------------
#             # Update counters
#             # ------------------------------

#             success_all += int(ev['all_ok'])

#             for key in ev.keys():
#                 if key not in condition_counts:
#                     condition_counts[key] = 0
#                 condition_counts[key] += int(ev[key])

#             # ------------------------------
#             # Store statistics
#             # ------------------------------

#             row = {
#                 'simulation': k + 1,
#                 'x0': x0_mc,                  # <-- AÑADIDO
#                 'spec_vector': params_mc,     # <-- AÑADIDO
#                 'c_supply': indicators['c_supply'],
#                 'c_demand': indicators['c_demand'],
#                 'p_benefit': indicators['p_benefit']
#             }

#             for key, val in ev.items():
#                 row[key] = val

#             stats.append(row)

#         except Exception as e:

#             print(f"\nSimulation failed: {e}")

#             stats.append({
#                 'simulation': k + 1,
#                 'x0': x0_mc,                  # <-- AÑADIDO (Útil para saber por qué falló)
#                 'spec_vector': params_mc,     # <-- AÑADIDO
#                 'c_supply': np.nan,
#                 'c_demand': np.nan,
#                 'p_benefit': np.nan,
#                 'all_ok': False
#             })

#     # ======================================
#     # FINAL RESULTS
#     # ======================================

#     df = pd.DataFrame(stats)

#     print("\n")
#     print("=" * 70)
#     print(f" ROBUSTNESS RESULTS – {scenario_name}")
#     print("=" * 70)

#     for key, count in condition_counts.items():
#         pct = 100 * count / n_simulations
#         print(f"{key:<20}: {pct:.2f}%")

#     print("-" * 70)

#     pct_all = 100 * success_all / n_simulations
#     print(f"{'ALL CONDITIONS':<20}: {pct_all:.2f}%")

#     print("=" * 70)

#     # ======================================
#     # DESCRIPTIVE STATISTICS
#     # ======================================

#     print("\nIndicator statistics:\n")

#     print(
#         df[
#             ['c_supply', 'c_demand', 'p_benefit']
#         ].describe()
#     )

#     return df


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':

    from pyCOT.io.functions import read_txt
    from pyCOT.simulations.ode import simulation
    # from script_AMF_3 import get_scenario_config, create_rate_list

    FILE_PATH = 'data/Ecological_models/AMF.txt'

    rn = read_txt(FILE_PATH)

    # ==========================================
    # EJECUTAR ANÁLISIS CON TRAYECTORIAS POR INDICADOR
    # ==========================================
    print("\n" + "="*80)
    print(" MONTE CARLO ROBUSTNESS ANALYSIS WITH INDICATOR-COLORED TRAJECTORIES ")
    print(" GREEN = C_supply | ORANGE = C_demand | BLUE = P_benefit")
    print(" SATURATED = Success (ALL CONDITIONS) | FADED = Failure")
    print("="*80 + "\n")
    
    results_all, trajectories_all = monte_carlo_amf_robustness(
        rn,
        scenario='all',
        variance=0.10,
        n_simulations=100,
        n_steps=500,
        plot_trajectories=True,
        show_success=True,
        show_failure=False,
        show_median=True,  # ← CAMBIAR A True PARA MOSTRAR LA MEDIANA
        save_figure_path='projects/AMF_3/outputs_3/monte_carlo_trajectories_by_indicator_exitos.png',
        save_excel_path='projects/AMF_3/outputs_3/monte_carlo_results_all_scenarios.xlsx'  # ← GUARDAR RESULTADOS EN EXCEL
    )
    # results_all, trajectories_all = monte_carlo_amf_robustness(
    #     rn,
    #     scenario='all',
    #     variance=0.10,
    #     n_simulations=100,
    #     n_steps=500,
    #     plot_trajectories=True,
    #     show_success=False,
    #     show_failure=True,
    #     show_median=True,  # ← CAMBIAR A True PARA MOSTRAR LA MEDIANA
    #     save_figure_path='projects/AMF_3/outputs_3/monte_carlo_trajectories_by_indicator_fracasos.png'
    # )
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETED SUCCESSFULLY ")
    print("="*80)