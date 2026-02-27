#!/usr/bin/env python3
"""
SCALE OF EFFORT ANALYSIS (ODE Simulation)
==========================================
Question 2: What happens if I increase with 30%:
    i.   Economic output (E)
    ii.  Trust - social cohesion (T)
    iii. Violence (V)

This script analyzes the impact of 30% increases in key variables using:
- Approach A: Boost initial conditions (+30% starting value)
- Approach B: Boost production rates (+30% reaction kinetics)

All analysis uses ODE simulation with Monte Carlo sampling for uncertainty.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path for pyCOT imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_pycot_root = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(_pycot_root, 'src'))

from pyCOT.io.functions import read_txt
from pyCOT.simulations.ode import simulation
from uncertainty_engine import (
    UncertaintyEngine, MonteCarloResult,
    INITIAL_CONDITION_RANGES, BASELINE_KINETICS,
    get_kinetic_boost_for_variable,
    results_to_dataframe, print_mc_summary
)

# ============================================================================
# MODEL LOADING
# ============================================================================

FILE_PATH = os.path.join(SCRIPT_DIR, 'data', 'Resource_Community_Insurgency_Loops_model3.txt')


def load_network():
    """Load the reaction network."""
    rn = read_txt(FILE_PATH)
    species = [s.name for s in rn.species()]
    reactions = [r.name() for r in rn.reactions()]
    return rn, species, reactions


RN, SPECIES, REACTIONS = load_network()
RXN_IDX = {name: idx for idx, name in enumerate(REACTIONS)}
SP_IDX = {name: idx for idx, name in enumerate(SPECIES)}


# ============================================================================
# BOOST CONFIGURATIONS
# ============================================================================

BOOST_FACTOR = 1.30  # +30%

# Boost types for initial conditions
INITIAL_BOOST_TYPES = {
    'baseline': {},
    'E_boost': {'E': BOOST_FACTOR},
    'T_boost': {'T': BOOST_FACTOR},
    'V_boost': {'V': BOOST_FACTOR},
    'E+T_boost': {'E': BOOST_FACTOR, 'T': BOOST_FACTOR},
}

# Boost types for reaction rates
RATE_BOOST_TYPES = {
    'baseline': {},
    'E_rate_boost': get_kinetic_boost_for_variable('E', BOOST_FACTOR),
    'T_rate_boost': get_kinetic_boost_for_variable('T', BOOST_FACTOR),
    'V_rate_boost': get_kinetic_boost_for_variable('V', BOOST_FACTOR),
}


# ============================================================================
# SIMULATION-BASED SOLVER
# ============================================================================

def simulate_with_boost(
    initial_state: Dict[str, float],
    kinetics: Dict[str, float],
    couplings: Dict[str, float],
    strategy_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run ODE simulation with optional boost to initial conditions or rates.

    Args:
        initial_state: Sampled initial conditions (may include boost)
        kinetics: Sampled kinetic parameters (may include boost)
        couplings: Sampled coupling parameters (not used in simulation)
        strategy_params: Contains 'boost_type', 'boost_approach'
        fixed_params: Contains 'simulation_time', 'climate'

    Returns:
        Result dictionary with simulation outcomes
    """
    simulation_time = fixed_params.get('simulation_time', 50)
    climate = fixed_params.get('climate', 0.5)

    # Build initial condition vector in species order
    x0 = [initial_state.get(sp, 1.0) for sp in SPECIES]

    # Apply climate effect (reduces all reaction rates in harsh conditions)
    climate_factor = 0.5 + 0.5 * climate  # 0.5-1.0 range

    # Build kinetic parameter vector
    spec_vector = []
    for rxn in REACTIONS:
        rate = kinetics.get(rxn, 0.01) * climate_factor
        spec_vector.append([rate])

    # Run ODE simulation
    try:
        ts_df, flux_df = simulation(
            RN,
            rate='mak',  # Mass action kinetics
            spec_vector=spec_vector,
            x0=x0,
            t_span=(0, simulation_time),
            n_steps=200,
            verbose=False
        )

        # Extract final state
        final_row = ts_df.iloc[-1]

        # Calculate metrics
        initial_AG = initial_state.get('AG_RL', 0) + initial_state.get('AG_SL', 0)
        final_AG = final_row.get('AG_RL', 0) + final_row.get('AG_SL', 0)

        ag_reduction_pct = (initial_AG - final_AG) / initial_AG * 100 if initial_AG > 0 else 0
        ag_reduction_pct = max(0, min(100, ag_reduction_pct))  # Clamp to 0-100

        return {
            'feasible': True,
            'status': 'Completed',
            'ag_reduction_pct': ag_reduction_pct,
            'final_V': final_row.get('V', 0),
            'final_T': final_row.get('T', 0),
            'final_E': final_row.get('E', 0),
            'final_Gov': final_row.get('Gov', 0),
            'initial_AG': initial_AG,
            'final_AG': final_AG,
            'initial_E': initial_state.get('E', 0),
            'initial_T': initial_state.get('T', 0),
            'initial_V': initial_state.get('V', 0),
        }

    except Exception as e:
        return {
            'feasible': False,
            'status': f'Error: {str(e)[:50]}',
            'ag_reduction_pct': 0,
            'final_V': initial_state.get('V', 0),
            'final_T': initial_state.get('T', 0),
            'final_E': initial_state.get('E', 0),
            'final_Gov': initial_state.get('Gov', 0),
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_scale_of_effort_analysis(
    n_samples: int = 100,
    scenarios: List[str] = None,
    climates: List[float] = None,
    simulation_time: float = 50,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run scale of effort analysis with both boost approaches.

    Args:
        n_samples: Monte Carlo samples per configuration
        scenarios: List of scenarios to test
        climates: List of climate conditions
        simulation_time: Duration of each simulation
        verbose: Print progress

    Returns:
        Tuple of (initial_boost_results_df, rate_boost_results_df)
    """
    if scenarios is None:
        scenarios = ['low', 'medium', 'severe']
    if climates is None:
        climates = [0.3, 0.5, 0.7]

    engine = UncertaintyEngine(n_samples=n_samples, seed=42)

    initial_results = []
    rate_results = []

    # ========== APPROACH A: Initial Condition Boost ==========
    print("\n" + "="*70)
    print("APPROACH A: INITIAL CONDITION BOOST (+30%)")
    print("="*70)

    total_configs = len(INITIAL_BOOST_TYPES) * len(scenarios) * len(climates)
    current = 0

    for boost_name, boost_dict in INITIAL_BOOST_TYPES.items():
        for scenario in scenarios:
            for climate in climates:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total_configs}] {boost_name}, {scenario}, climate={climate}")

                mc_result = engine.run_monte_carlo(
                    solver_func=simulate_with_boost,
                    scenario=scenario,
                    strategy_params={
                        'boost_type': boost_name,
                        'boost_approach': 'initial',
                    },
                    fixed_params={
                        'simulation_time': simulation_time,
                        'climate': climate,
                    },
                    initial_boost=boost_dict if boost_dict else None,
                    kinetic_boost=None,
                    n_samples=n_samples,
                    verbose=False,
                )

                row = {
                    'boost_type': boost_name,
                    'boost_approach': 'initial_condition',
                    'scenario': scenario,
                    'climate': climate,
                    'n_samples': mc_result.n_samples,
                    'feasibility_rate': mc_result.feasibility_rate,
                    'ag_reduction_mean': mc_result.ag_reduction_mean,
                    'ag_reduction_std': mc_result.ag_reduction_std,
                    'ag_reduction_p5': mc_result.ag_reduction_p5,
                    'ag_reduction_p95': mc_result.ag_reduction_p95,
                    'final_violence_mean': mc_result.final_violence_mean,
                    'final_trust_mean': mc_result.final_trust_mean,
                    'robustness_25': mc_result.robustness_25,
                    'robustness_50': mc_result.robustness_50,
                }
                initial_results.append(row)

    # ========== APPROACH B: Reaction Rate Boost ==========
    print("\n" + "="*70)
    print("APPROACH B: REACTION RATE BOOST (+30%)")
    print("="*70)

    total_configs = len(RATE_BOOST_TYPES) * len(scenarios) * len(climates)
    current = 0

    for boost_name, boost_dict in RATE_BOOST_TYPES.items():
        for scenario in scenarios:
            for climate in climates:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total_configs}] {boost_name}, {scenario}, climate={climate}")

                mc_result = engine.run_monte_carlo(
                    solver_func=simulate_with_boost,
                    scenario=scenario,
                    strategy_params={
                        'boost_type': boost_name,
                        'boost_approach': 'rate',
                    },
                    fixed_params={
                        'simulation_time': simulation_time,
                        'climate': climate,
                    },
                    initial_boost=None,
                    kinetic_boost=boost_dict if boost_dict else None,
                    n_samples=n_samples,
                    verbose=False,
                )

                row = {
                    'boost_type': boost_name,
                    'boost_approach': 'reaction_rate',
                    'scenario': scenario,
                    'climate': climate,
                    'n_samples': mc_result.n_samples,
                    'feasibility_rate': mc_result.feasibility_rate,
                    'ag_reduction_mean': mc_result.ag_reduction_mean,
                    'ag_reduction_std': mc_result.ag_reduction_std,
                    'ag_reduction_p5': mc_result.ag_reduction_p5,
                    'ag_reduction_p95': mc_result.ag_reduction_p95,
                    'final_violence_mean': mc_result.final_violence_mean,
                    'final_trust_mean': mc_result.final_trust_mean,
                    'robustness_25': mc_result.robustness_25,
                    'robustness_50': mc_result.robustness_50,
                }
                rate_results.append(row)

    initial_df = pd.DataFrame(initial_results)
    rate_df = pd.DataFrame(rate_results)

    # Save results
    os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)
    initial_df.to_csv(os.path.join(SCRIPT_DIR, 'outputs', 'scale_of_effort_initial_boost.csv'), index=False)
    rate_df.to_csv(os.path.join(SCRIPT_DIR, 'outputs', 'scale_of_effort_rate_boost.csv'), index=False)
    print(f"\nResults saved to: outputs/scale_of_effort_*.csv")

    return initial_df, rate_df


def create_visualizations(
    initial_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    save_dir: str = None
):
    """Create visualizations for scale of effort analysis."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # 1. Comparison of boost effects on AG reduction
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Initial condition boost
    ax1 = axes[0]
    boost_comparison = initial_df.groupby('boost_type')['ag_reduction_mean'].mean().sort_values(ascending=True)
    colors = ['green' if 'baseline' in b else 'steelblue' for b in boost_comparison.index]
    ax1.barh(range(len(boost_comparison)), boost_comparison.values, color=colors)
    ax1.set_yticks(range(len(boost_comparison)))
    ax1.set_yticklabels(boost_comparison.index)
    ax1.set_xlabel('Expected AG Reduction (%)')
    ax1.set_title('Approach A: Initial Condition Boost', fontweight='bold')
    if 'baseline' in boost_comparison.index:
        ax1.axvline(boost_comparison['baseline'], color='red', linestyle='--', alpha=0.7)

    # Rate boost
    ax2 = axes[1]
    boost_comparison = rate_df.groupby('boost_type')['ag_reduction_mean'].mean().sort_values(ascending=True)
    colors = ['green' if 'baseline' in b else 'steelblue' for b in boost_comparison.index]
    ax2.barh(range(len(boost_comparison)), boost_comparison.values, color=colors)
    ax2.set_yticks(range(len(boost_comparison)))
    ax2.set_yticklabels(boost_comparison.index)
    ax2.set_xlabel('Expected AG Reduction (%)')
    ax2.set_title('Approach B: Reaction Rate Boost', fontweight='bold')
    if 'baseline' in boost_comparison.index:
        ax2.axvline(boost_comparison['baseline'], color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boost_comparison_boxplots.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/boost_comparison_boxplots.png")

    # 2. Impact by scenario - Initial condition boost
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, scenario in enumerate(initial_df['scenario'].unique()):
        ax = axes[idx]
        data = initial_df[initial_df['scenario'] == scenario]
        pivot = data.pivot_table(values='ag_reduction_mean', index='boost_type', aggfunc='mean')
        baseline_val = pivot.loc['baseline'] if 'baseline' in pivot.index else 0

        # Calculate difference from baseline
        diff = pivot - baseline_val
        colors = ['green' if v >= 0 else 'red' for v in diff.values.flatten()]

        ax.barh(range(len(diff)), diff.values.flatten(), color=colors)
        ax.set_yticks(range(len(diff)))
        ax.set_yticklabels(diff.index)
        ax.set_xlabel('Change in AG Reduction (pp)')
        ax.set_title(f'{scenario.upper()} Scenario', fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('Impact of +30% Initial Condition Boost (vs Baseline)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'initial_boost_by_scenario.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/initial_boost_by_scenario.png")

    # 3. E boost distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_data = initial_df[initial_df['boost_type'] == 'baseline']['ag_reduction_mean']

    for approach, df, label in [('Initial', initial_df, 'Initial Condition'),
                                  ('Rate', rate_df, 'Reaction Rate')]:
        if approach == 'Initial':
            e_data = df[df['boost_type'] == 'E_boost']['ag_reduction_mean']
        else:
            e_data = df[df['boost_type'] == 'E_rate_boost']['ag_reduction_mean']
        if len(e_data) > 0:
            ax.hist(e_data, bins=20, alpha=0.5, label=f'{label} +30% E')

    ax.axvline(baseline_data.mean(), color='red', linestyle='--', label='Baseline mean')
    ax.set_xlabel('AG Reduction (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of AG Reduction with +30% Economic Output', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'E_boost_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/E_boost_distribution.png")

    # 4. T boost distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    for approach, df, label in [('Initial', initial_df, 'Initial Condition'),
                                  ('Rate', rate_df, 'Reaction Rate')]:
        if approach == 'Initial':
            t_data = df[df['boost_type'] == 'T_boost']['ag_reduction_mean']
        else:
            t_data = df[df['boost_type'] == 'T_rate_boost']['ag_reduction_mean']
        if len(t_data) > 0:
            ax.hist(t_data, bins=20, alpha=0.5, label=f'{label} +30% T')

    ax.axvline(baseline_data.mean(), color='red', linestyle='--', label='Baseline mean')
    ax.set_xlabel('AG Reduction (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of AG Reduction with +30% Trust', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'T_boost_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/T_boost_distribution.png")

    # 5. V boost distribution (negative impact expected)
    fig, ax = plt.subplots(figsize=(10, 6))

    for approach, df, label in [('Initial', initial_df, 'Initial Condition'),
                                  ('Rate', rate_df, 'Reaction Rate')]:
        if approach == 'Initial':
            v_data = df[df['boost_type'] == 'V_boost']['ag_reduction_mean']
        else:
            v_data = df[df['boost_type'] == 'V_rate_boost']['ag_reduction_mean']
        if len(v_data) > 0:
            ax.hist(v_data, bins=20, alpha=0.5, label=f'{label} +30% V')

    ax.axvline(baseline_data.mean(), color='red', linestyle='--', label='Baseline mean')
    ax.set_xlabel('AG Reduction (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of AG Reduction with +30% Violence', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'V_boost_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/V_boost_distribution.png")

    # 6. Initial vs Rate boost comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    variables = ['E', 'T', 'V']
    x = np.arange(len(variables))
    width = 0.35

    initial_means = []
    rate_means = []
    baseline_mean = initial_df[initial_df['boost_type'] == 'baseline']['ag_reduction_mean'].mean()

    for var in variables:
        init_data = initial_df[initial_df['boost_type'] == f'{var}_boost']['ag_reduction_mean'].mean()
        rate_data = rate_df[rate_df['boost_type'] == f'{var}_rate_boost']['ag_reduction_mean'].mean()
        initial_means.append(init_data - baseline_mean)
        rate_means.append(rate_data - baseline_mean)

    ax.bar(x - width/2, initial_means, width, label='Initial Condition Boost', color='steelblue')
    ax.bar(x + width/2, rate_means, width, label='Reaction Rate Boost', color='orange')

    ax.set_ylabel('Change in AG Reduction (pp vs baseline)')
    ax.set_xlabel('Variable Boosted (+30%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Economy (E)', 'Trust (T)', 'Violence (V)'])
    ax.set_title('Comparison: Initial Condition vs Reaction Rate Boost', fontweight='bold')
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'initial_vs_rate_boost.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/initial_vs_rate_boost.png")


def generate_report(
    initial_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    save_dir: str = None
):
    """Generate text report with key findings."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'reports')
    os.makedirs(save_dir, exist_ok=True)

    report = []
    report.append("="*70)
    report.append("SCALE OF EFFORT ANALYSIS REPORT (ODE Simulation)")
    report.append("Question: Impact of +30% Increase in E, T, V")
    report.append("="*70)
    report.append("")

    # Baseline reference
    baseline_initial = initial_df[initial_df['boost_type'] == 'baseline']
    baseline_mean = baseline_initial['ag_reduction_mean'].mean()

    report.append("1. BASELINE REFERENCE")
    report.append("-"*50)
    report.append(f"Baseline AG reduction (expected): {baseline_mean:.1f}%")
    report.append(f"Baseline simulation success rate: {baseline_initial['feasibility_rate'].mean()*100:.1f}%")
    report.append("")

    # Approach A: Initial Condition Boost
    report.append("2. APPROACH A: INITIAL CONDITION BOOST (+30%)")
    report.append("-"*50)

    for boost_type in ['E_boost', 'T_boost', 'V_boost', 'E+T_boost']:
        data = initial_df[initial_df['boost_type'] == boost_type]
        if len(data) > 0:
            mean_val = data['ag_reduction_mean'].mean()
            diff = mean_val - baseline_mean
            sign = '+' if diff >= 0 else ''

            report.append(f"\n{boost_type}:")
            report.append(f"  Expected AG reduction: {mean_val:.1f}% ({sign}{diff:.1f} pp vs baseline)")
            report.append(f"  Worst case (5th pct): {data['ag_reduction_p5'].mean():.1f}%")
            report.append(f"  Success rate: {data['feasibility_rate'].mean()*100:.1f}%")
            report.append(f"  Robustness (25% target): {data['robustness_25'].mean()*100:.1f}%")

    # Approach B: Reaction Rate Boost
    report.append("\n\n3. APPROACH B: REACTION RATE BOOST (+30%)")
    report.append("-"*50)

    baseline_rate = rate_df[rate_df['boost_type'] == 'baseline']
    baseline_rate_mean = baseline_rate['ag_reduction_mean'].mean()

    for boost_type in ['E_rate_boost', 'T_rate_boost', 'V_rate_boost']:
        data = rate_df[rate_df['boost_type'] == boost_type]
        if len(data) > 0:
            mean_val = data['ag_reduction_mean'].mean()
            diff = mean_val - baseline_rate_mean
            sign = '+' if diff >= 0 else ''

            report.append(f"\n{boost_type}:")
            report.append(f"  Expected AG reduction: {mean_val:.1f}% ({sign}{diff:.1f} pp vs baseline)")
            report.append(f"  Worst case (5th pct): {data['ag_reduction_p5'].mean():.1f}%")
            report.append(f"  Success rate: {data['feasibility_rate'].mean()*100:.1f}%")
            report.append(f"  Robustness (25% target): {data['robustness_25'].mean()*100:.1f}%")

    # Comparison
    report.append("\n\n4. COMPARISON: INITIAL vs RATE BOOST")
    report.append("-"*50)

    for var in ['E', 'T', 'V']:
        init_data = initial_df[initial_df['boost_type'] == f'{var}_boost']['ag_reduction_mean'].mean()
        rate_data = rate_df[rate_df['boost_type'] == f'{var}_rate_boost']['ag_reduction_mean'].mean()

        report.append(f"\n{var} (+30%):")
        report.append(f"  Initial condition: {init_data:.1f}% AG reduction")
        report.append(f"  Reaction rate: {rate_data:.1f}% AG reduction")
        report.append(f"  Difference: {abs(init_data - rate_data):.1f} pp")

    # Key findings
    report.append("\n\n5. KEY FINDINGS")
    report.append("-"*50)

    # Find best boost
    all_boosts = pd.concat([
        initial_df[initial_df['boost_type'] != 'baseline'][['boost_type', 'ag_reduction_mean']],
        rate_df[rate_df['boost_type'] != 'baseline'][['boost_type', 'ag_reduction_mean']]
    ])
    if len(all_boosts) > 0:
        best = all_boosts.groupby('boost_type')['ag_reduction_mean'].mean().idxmax()
        best_val = all_boosts.groupby('boost_type')['ag_reduction_mean'].mean().max()

        report.append(f"\nMost effective boost: {best}")
        report.append(f"  Expected AG reduction: {best_val:.1f}%")
        report.append(f"  Improvement over baseline: {best_val - baseline_mean:.1f} pp")

    # Violence boost impact
    v_init = initial_df[initial_df['boost_type'] == 'V_boost']['ag_reduction_mean'].mean()
    report.append(f"\nViolence increase (+30%) impact:")
    report.append(f"  AG reduction: {v_init:.1f}% (baseline: {baseline_mean:.1f}%)")
    if v_init < baseline_mean:
        report.append(f"  This confirms violence undermines peace interventions.")
    else:
        report.append(f"  Note: System dynamics may absorb initial violence spike.")

    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)

    # Save report
    report_text = '\n'.join(report)
    with open(os.path.join(save_dir, 'scale_of_effort_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"Saved: {save_dir}/scale_of_effort_report.txt")

    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main(n_samples: int = 100, quick: bool = False):
    """Run scale of effort analysis."""
    if quick:
        scenarios = ['medium']
        climates = [0.5]
        n_samples = min(n_samples, 50)
    else:
        scenarios = ['low', 'medium', 'severe']
        climates = [0.3, 0.5, 0.7]

    # Run analysis (5 years simulation time)
    initial_df, rate_df = run_scale_of_effort_analysis(
        n_samples=n_samples,
        scenarios=scenarios,
        climates=climates,
        simulation_time=5,
        verbose=True,
    )

    # Create visualizations
    create_visualizations(initial_df, rate_df)

    # Generate report
    report = generate_report(initial_df, rate_df)
    print("\n" + report)

    return initial_df, rate_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scale of Effort Analysis (ODE Simulation)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Monte Carlo samples per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced scenarios')
    args = parser.parse_args()

    main(n_samples=args.samples, quick=args.quick)
