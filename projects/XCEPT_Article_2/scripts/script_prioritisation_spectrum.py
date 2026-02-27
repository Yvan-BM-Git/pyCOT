#!/usr/bin/env python3
"""
PRIORITISATION SPECTRUM ANALYSIS (ODE Simulation)
==================================================
Question 1: What happens if I prioritise security over development?
            Or development over security?

This script analyzes the full spectrum of security vs development prioritisation
using ODE simulation with Monte Carlo sampling for uncertainty quantification.

Key Outputs:
- Expected values across the prioritisation spectrum
- Worst-case scenarios (5th percentile)
- Robustness measures (% achieving targets)
- Time evolution of key variables
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
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
# PRIORITISATION STRATEGY DEFINITION
# ============================================================================

@dataclass
class PrioritisationStrategy:
    """
    Strategy defined by security/development prioritisation ratio.

    security_priority = 0: Pure development (economic + trust focus)
    security_priority = 1: Pure security (DDR + governance focus)

    The strategy modifies reaction rates to emphasize different interventions.
    """
    security_priority: float  # 0-1

    @property
    def name(self) -> str:
        if self.security_priority == 0:
            return "Pure Development"
        elif self.security_priority == 1:
            return "Pure Security"
        elif self.security_priority == 0.5:
            return "Balanced"
        else:
            return f"{self.security_priority:.0%} Security"

    @property
    def governance_boost(self) -> float:
        """Governance reaction rate multiplier: 1.0 (dev) to 3.0 (sec)"""
        return 1.0 + 2.0 * self.security_priority

    @property
    def economic_boost(self) -> float:
        """Economic reaction rate multiplier: 3.0 (dev) to 1.0 (sec)"""
        return 3.0 - 2.0 * self.security_priority

    @property
    def trust_boost(self) -> float:
        """Trust reaction rate multiplier: 2.5 (dev) to 1.0 (sec)"""
        return 2.5 - 1.5 * self.security_priority

    @property
    def ddr_boost(self) -> float:
        """DDR reaction rate multiplier: 1.0 (dev) to 3.0 (sec)"""
        return 1.0 + 2.0 * self.security_priority

    @property
    def violence_reduction_boost(self) -> float:
        """Violence reduction rate multiplier: 1.0 (dev) to 2.0 (sec)"""
        return 1.0 + 1.0 * self.security_priority


# 11 prioritisation levels: 0%, 10%, 20%, ..., 100%
PRIORITISATION_LEVELS = [i / 10.0 for i in range(11)]


# ============================================================================
# REACTION CATEGORIES
# ============================================================================

# Categorize reactions for intervention based on what they affect
GOV_REACTIONS = ['r9', 'r10']               # Governance building
ECON_REACTIONS = ['r7', 'r8', 'r16', 'r17'] # Economic production
TRUST_REACTIONS = ['r22', 'r23', 'r24']     # Trust building
DDR_REACTIONS = ['r21']                      # Demobilization (AG reduction)
VIOLENCE_RED_REACTIONS = ['r29', 'r30', 'r31']  # Violence reduction


# ============================================================================
# SIMULATION-BASED SOLVER
# ============================================================================

def simulate_with_prioritisation(
    initial_state: Dict[str, float],
    kinetics: Dict[str, float],
    couplings: Dict[str, float],
    strategy_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run ODE simulation with prioritisation strategy.

    The strategy modifies reaction rates to emphasize security or development.

    Args:
        initial_state: Sampled initial conditions
        kinetics: Sampled kinetic parameters (baseline rates)
        couplings: Sampled coupling parameters (not used in simulation)
        strategy_params: Contains 'strategy' (PrioritisationStrategy)
        fixed_params: Contains 'simulation_time', 'climate'

    Returns:
        Result dictionary with simulation outcomes
    """
    strategy = strategy_params['strategy']
    simulation_time = fixed_params.get('simulation_time', 50)
    climate = fixed_params.get('climate', 0.5)

    # Build initial condition vector in species order
    x0 = [initial_state.get(sp, 1.0) for sp in SPECIES]

    # Apply climate effect (reduces all reaction rates in harsh conditions)
    climate_factor = 0.5 + 0.5 * climate  # 0.5-1.0 range

    # Build kinetic parameter vector with strategy boosts
    spec_vector = []
    for rxn in REACTIONS:
        base_rate = kinetics.get(rxn, 0.01) * climate_factor

        # Apply strategy-specific boosts based on reaction category
        if rxn in GOV_REACTIONS:
            rate = base_rate * strategy.governance_boost
        elif rxn in ECON_REACTIONS:
            rate = base_rate * strategy.economic_boost
        elif rxn in TRUST_REACTIONS:
            rate = base_rate * strategy.trust_boost
        elif rxn in DDR_REACTIONS:
            rate = base_rate * strategy.ddr_boost
        elif rxn in VIOLENCE_RED_REACTIONS:
            rate = base_rate * strategy.violence_reduction_boost
        else:
            rate = base_rate

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
            'security_priority': strategy.security_priority,
            'ag_reduction_pct': ag_reduction_pct,
            'final_V': final_row.get('V', 0),
            'final_T': final_row.get('T', 0),
            'final_E': final_row.get('E', 0),
            'final_Gov': final_row.get('Gov', 0),
            'initial_AG': initial_AG,
            'final_AG': final_AG,
        }

    except Exception as e:
        return {
            'feasible': False,
            'status': f'Error: {str(e)[:50]}',
            'security_priority': strategy.security_priority,
            'ag_reduction_pct': 0,
            'final_V': initial_state.get('V', 0),
            'final_T': initial_state.get('T', 0),
            'final_E': initial_state.get('E', 0),
            'final_Gov': initial_state.get('Gov', 0),
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_prioritisation_analysis(
    n_samples: int = 100,
    scenarios: List[str] = None,
    climates: List[float] = None,
    simulation_time: float = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full prioritisation spectrum analysis with Monte Carlo sampling.

    Args:
        n_samples: Monte Carlo samples per configuration
        scenarios: List of scenarios to test ('low', 'medium', 'severe')
        climates: List of climate conditions (0-1, higher is more favorable)
        simulation_time: Duration of each simulation
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    if scenarios is None:
        scenarios = ['low', 'medium', 'severe']
    if climates is None:
        climates = [0.3, 0.5, 0.7]

    engine = UncertaintyEngine(n_samples=n_samples, seed=42)

    all_results = []

    total_configs = len(PRIORITISATION_LEVELS) * len(scenarios) * len(climates)
    current = 0

    print("\n" + "="*70)
    print("PRIORITISATION SPECTRUM ANALYSIS (ODE Simulation)")
    print("="*70)
    print(f"\nConfigurations: {total_configs}")
    print(f"MC samples per config: {n_samples}")
    print(f"Simulation time: {simulation_time} time units")
    print(f"Total simulations: {total_configs * n_samples}")

    for scenario in scenarios:
        for climate in climates:
            for priority in PRIORITISATION_LEVELS:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total_configs}] {scenario}, climate={climate}, "
                          f"security={priority:.0%}")

                strategy = PrioritisationStrategy(security_priority=priority)

                # Run Monte Carlo
                mc_result = engine.run_monte_carlo(
                    solver_func=simulate_with_prioritisation,
                    scenario=scenario,
                    strategy_params={'strategy': strategy},
                    fixed_params={
                        'simulation_time': simulation_time,
                        'climate': climate,
                    },
                    n_samples=n_samples,
                    verbose=False,
                )

                # Store results
                row = {
                    'scenario': scenario,
                    'climate': climate,
                    'security_priority': priority,
                    'strategy_name': strategy.name,
                    'n_samples': mc_result.n_samples,
                    'feasibility_rate': mc_result.feasibility_rate,
                    'ag_reduction_mean': mc_result.ag_reduction_mean,
                    'ag_reduction_std': mc_result.ag_reduction_std,
                    'ag_reduction_p5': mc_result.ag_reduction_p5,
                    'ag_reduction_p25': mc_result.ag_reduction_p25,
                    'ag_reduction_median': mc_result.ag_reduction_median,
                    'ag_reduction_p75': mc_result.ag_reduction_p75,
                    'ag_reduction_p95': mc_result.ag_reduction_p95,
                    'final_violence_mean': mc_result.final_violence_mean,
                    'final_violence_p5': mc_result.final_violence_p5,
                    'final_violence_p95': mc_result.final_violence_p95,
                    'final_trust_mean': mc_result.final_trust_mean,
                    'robustness_25': mc_result.robustness_25,
                    'robustness_50': mc_result.robustness_50,
                }
                all_results.append(row)

    df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)
    df.to_csv(os.path.join(SCRIPT_DIR, 'outputs', 'prioritisation_mc_results.csv'), index=False)
    print(f"\nResults saved to: outputs/prioritisation_mc_results.csv")

    return df


def create_visualizations(df: pd.DataFrame, save_dir: str = None):
    """Create visualizations for prioritisation analysis."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # 1. Expected value across spectrum
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # AG Reduction by priority
    ax1 = axes[0, 0]
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario].groupby('security_priority').agg({
            'ag_reduction_mean': 'mean',
            'ag_reduction_std': 'mean',
        }).reset_index()
        ax1.plot(data['security_priority'] * 100, data['ag_reduction_mean'],
                 'o-', label=scenario, linewidth=2)
        ax1.fill_between(data['security_priority'] * 100,
                         data['ag_reduction_mean'] - data['ag_reduction_std'],
                         data['ag_reduction_mean'] + data['ag_reduction_std'],
                         alpha=0.2)
    ax1.set_xlabel('Security Priority (%)')
    ax1.set_ylabel('AG Reduction (%)')
    ax1.set_title('Expected AG Reduction by Prioritisation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Success rate by priority
    ax2 = axes[0, 1]
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario].groupby('security_priority')['feasibility_rate'].mean()
        ax2.plot(data.index * 100, data.values * 100, 'o-', label=scenario, linewidth=2)
    ax2.set_xlabel('Security Priority (%)')
    ax2.set_ylabel('Simulation Success Rate (%)')
    ax2.set_title('Simulation Success by Prioritisation', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Robustness (25% target)
    ax3 = axes[1, 0]
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario].groupby('security_priority')['robustness_25'].mean()
        ax3.plot(data.index * 100, data.values * 100, 'o-', label=scenario, linewidth=2)
    ax3.set_xlabel('Security Priority (%)')
    ax3.set_ylabel('Robustness (%)')
    ax3.set_title('Probability of Achieving 25% AG Reduction', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Worst case (5th percentile)
    ax4 = axes[1, 1]
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario].groupby('security_priority')['ag_reduction_p5'].mean()
        ax4.plot(data.index * 100, data.values, 'o-', label=scenario, linewidth=2)
    ax4.set_xlabel('Security Priority (%)')
    ax4.set_ylabel('AG Reduction (5th percentile)')
    ax4.set_title('Worst-Case AG Reduction', fontweight='bold')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prioritisation_expected_value.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/prioritisation_expected_value.png")

    # 2. Heatmap: Priority vs Scenario
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(
        values='ag_reduction_mean',
        index='security_priority',
        columns='scenario',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'AG Reduction (%)'})
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Security Priority')
    ax.set_title('Expected AG Reduction: Security Priority vs Scenario', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prioritisation_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/prioritisation_heatmap.png")

    # 3. Confidence bands
    fig, ax = plt.subplots(figsize=(12, 6))
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario].groupby('security_priority').agg({
            'ag_reduction_mean': 'mean',
            'ag_reduction_p5': 'mean',
            'ag_reduction_p95': 'mean',
        }).reset_index()

        x = data['security_priority'] * 100
        ax.plot(x, data['ag_reduction_mean'], 'o-', label=f'{scenario} (mean)', linewidth=2)
        ax.fill_between(x, data['ag_reduction_p5'], data['ag_reduction_p95'], alpha=0.2)

    ax.set_xlabel('Security Priority (%)')
    ax.set_ylabel('AG Reduction (%)')
    ax.set_title('AG Reduction with 90% Confidence Bands (5th-95th percentile)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prioritisation_confidence_bands.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/prioritisation_confidence_bands.png")

    # 4. Robustness heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(
        values='robustness_25',
        index='security_priority',
        columns='scenario',
        aggfunc='mean'
    ) * 100
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax,
                vmin=0, vmax=100, cbar_kws={'label': 'Robustness (%)'})
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Security Priority')
    ax.set_title('Robustness: Probability of Achieving 25% AG Reduction', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prioritisation_robustness_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/prioritisation_robustness_heatmap.png")


def generate_report(df: pd.DataFrame, save_dir: str = None):
    """Generate text report with key findings."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'reports')
    os.makedirs(save_dir, exist_ok=True)

    report = []
    report.append("="*70)
    report.append("PRIORITISATION SPECTRUM ANALYSIS REPORT (ODE Simulation)")
    report.append("Question: Security vs Development Trade-offs")
    report.append("="*70)
    report.append("")

    # Overall summary
    report.append("1. OVERALL SUMMARY")
    report.append("-"*50)
    report.append(f"Total configurations analyzed: {len(df)}")
    report.append(f"MC samples per configuration: {df['n_samples'].iloc[0]}")
    report.append(f"Average simulation success rate: {df['feasibility_rate'].mean()*100:.1f}%")
    report.append("")

    # Best prioritisation by scenario
    report.append("2. OPTIMAL PRIORITISATION BY SCENARIO")
    report.append("-"*50)
    for scenario in df['scenario'].unique():
        data = df[df['scenario'] == scenario]
        best = data.loc[data['ag_reduction_mean'].idxmax()]
        report.append(f"\n{scenario.upper()}:")
        report.append(f"  Best security priority: {best['security_priority']*100:.0f}%")
        report.append(f"  Expected AG reduction: {best['ag_reduction_mean']:.1f}%")
        report.append(f"  Worst-case AG reduction: {best['ag_reduction_p5']:.1f}%")
        report.append(f"  Robustness (25% target): {best['robustness_25']*100:.1f}%")

    # Key findings
    report.append("\n3. KEY FINDINGS")
    report.append("-"*50)

    # Compare pure security vs pure development
    pure_sec = df[df['security_priority'] == 1.0].groupby('scenario')['ag_reduction_mean'].mean()
    pure_dev = df[df['security_priority'] == 0.0].groupby('scenario')['ag_reduction_mean'].mean()
    balanced = df[df['security_priority'] == 0.5].groupby('scenario')['ag_reduction_mean'].mean()

    report.append("\nComparison (Expected AG Reduction):")
    report.append(f"{'Scenario':<12} {'Pure Dev':<12} {'Balanced':<12} {'Pure Sec':<12}")
    for scenario in df['scenario'].unique():
        report.append(f"{scenario:<12} {pure_dev.get(scenario, 0):<12.1f} "
                     f"{balanced.get(scenario, 0):<12.1f} {pure_sec.get(scenario, 0):<12.1f}")

    # Robustness comparison
    report.append("\n4. ROBUSTNESS COMPARISON")
    report.append("-"*50)
    report.append("\nProbability of achieving 25% AG reduction:")
    for priority in [0.0, 0.3, 0.5, 0.7, 1.0]:
        data = df[df['security_priority'] == priority]
        rob = data['robustness_25'].mean() * 100
        report.append(f"  Security {priority*100:3.0f}%: {rob:.1f}% robust")

    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)

    # Save report
    report_text = '\n'.join(report)
    with open(os.path.join(save_dir, 'prioritisation_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"Saved: {save_dir}/prioritisation_report.txt")

    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main(n_samples: int = 100, quick: bool = False):
    """Run prioritisation analysis."""
    if quick:
        scenarios = ['medium']
        climates = [0.5]
        n_samples = min(n_samples, 50)
    else:
        scenarios = ['low', 'medium', 'severe']
        climates = [0.3, 0.5, 0.7]

    # Run analysis (5 years simulation time)
    results_df = run_prioritisation_analysis(
        n_samples=n_samples,
        scenarios=scenarios,
        climates=climates,
        simulation_time=5,
        verbose=True,
    )

    # Create visualizations
    create_visualizations(results_df)

    # Generate report
    report = generate_report(results_df)
    print("\n" + report)

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prioritisation Spectrum Analysis (ODE Simulation)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Monte Carlo samples per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced scenarios')
    args = parser.parse_args()

    main(n_samples=args.samples, quick=args.quick)
