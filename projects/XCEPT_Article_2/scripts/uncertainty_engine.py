#!/usr/bin/env python3
"""
Uncertainty Engine - Monte Carlo Sampling for Conflict Analysis

This module provides infrastructure for decision-making under uncertainty
through Monte Carlo sampling of initial conditions, kinetic parameters,
and coupling coefficients.

Key Features:
1. Sample from reasonable parameter ranges
2. Run N simulations per scenario
3. Aggregate results with statistics (mean, percentiles, robustness)
4. Support decision-making with expected values and worst-case analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PARAMETER RANGES - Uncertain Initial Conditions
# ============================================================================

INITIAL_CONDITION_RANGES = {
    'severe': {
        'description': 'Active insurgency with widespread violence',
        # Population on resource land
        'SR_RL': (5.0, 12.0),    # Strong resilient: 500k-1.2M
        'WR_RL': (8.0, 15.0),    # Weak resilient: 800k-1.5M
        'AG_RL': (0.5, 1.2),     # Armed groups: 50k-120k
        # Population on stressed land
        'SR_SL': (8.0, 15.0),    # Strong resilient: 800k-1.5M
        'WR_SL': (12.0, 20.0),   # Weak resilient: 1.2M-2M
        'AG_SL': (1.0, 2.5),     # Armed groups: 100k-250k
        # Land
        'RL': (15.0, 25.0),      # Restored land
        'SL': (80.0, 120.0),     # Stressed land
        # Key state variables
        'E': (30.0, 70.0),       # Economy: weak
        'T': (10.0, 30.0),       # Trust: low
        'V': (35.0, 65.0),       # Violence: high
        'Gov': (5.0, 15.0),      # Governance: weak
    },
    'medium': {
        'description': 'Regional instability with active tensions',
        'SR_RL': (10.0, 15.0),
        'WR_RL': (12.0, 18.0),
        'AG_RL': (0.4, 1.0),
        'SR_SL': (6.0, 10.0),
        'WR_SL': (8.0, 12.0),
        'AG_SL': (1.0, 2.0),
        'RL': (15.0, 25.0),
        'SL': (80.0, 120.0),
        'E': (60.0, 100.0),
        'T': (30.0, 50.0),
        'V': (20.0, 40.0),
        'Gov': (15.0, 25.0),
    },
    'low': {
        'description': 'Stable region with minor tensions',
        'SR_RL': (12.0, 18.0),
        'WR_RL': (8.0, 12.0),
        'AG_RL': (0.3, 0.8),
        'SR_SL': (10.0, 14.0),
        'WR_SL': (6.0, 10.0),
        'AG_SL': (0.8, 1.5),
        'RL': (15.0, 25.0),
        'SL': (80.0, 120.0),
        'E': (80.0, 120.0),
        'T': (50.0, 70.0),
        'V': (5.0, 15.0),
        'Gov': (25.0, 40.0),
    },
}

# Coupling parameters - uncertain
COUPLING_RANGES = {
    'alpha': (0.2, 0.6),   # Trust efficiency for violence reduction
    'beta': (0.3, 0.7),    # DDR-governance coupling
    'gamma': (0.3, 0.7),   # DDR-economy coupling
    'delta': (0.4, 0.8),   # Institution-economy coupling
}

# Baseline kinetic constants (to be sampled with ±30% uncertainty)
BASELINE_KINETICS = {
    # Land dynamics
    'r1': 0.01, 'r2': 0.01, 'r3': 0.01, 'r4': 0.02, 'r5': 0.02, 'r6': 0.02, 'r7': 0.02,
    # Governance/Economy
    'r8': 0.05, 'r9': 0.03, 'r10': 0.03, 'r11': 0.02, 'r12': 1.0,
    # Migration
    'r13': 0.01, 'r14': 0.01, 'r15': 0.02, 'r16': 0.02, 'r17': 0.01,
    # Resilience
    'r18': 0.02, 'r19': 0.02, 'r20': 0.01, 'r21': 0.03,
    # Trust
    'r22': 0.02, 'r23': 0.02, 'r24': 0.02, 'r25': 0.05, 'r26': 0.02,
    # Violence
    'r27': 0.02, 'r28': 0.03, 'r29': 0.02, 'r30': 0.02, 'r31': 0.1,
    # AG dynamics
    'r32': 0.0002, 'r33': 0.01, 'r34': 0.01, 'r35': 0.01, 'r36': 0.005,
}

KINETIC_UNCERTAINTY = 0.30  # ±30% from baseline


# ============================================================================
# UNCERTAINTY ENGINE CLASS
# ============================================================================

@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    n_samples: int
    feasibility_rate: float

    # AG reduction statistics
    ag_reduction_mean: float
    ag_reduction_std: float
    ag_reduction_p5: float
    ag_reduction_p25: float
    ag_reduction_median: float
    ag_reduction_p75: float
    ag_reduction_p95: float

    # Violence statistics
    final_violence_mean: float
    final_violence_std: float
    final_violence_p5: float
    final_violence_p95: float

    # Trust statistics
    final_trust_mean: float
    final_trust_std: float
    final_trust_p5: float
    final_trust_p95: float

    # Economy statistics
    final_economy_mean: float
    final_economy_std: float

    # Robustness
    robustness_25: float  # % achieving 25% AG reduction
    robustness_50: float  # % achieving 50% AG reduction

    # Raw data for custom analysis
    raw_results: List[Dict] = field(default_factory=list)


class UncertaintyEngine:
    """
    Monte Carlo sampling and uncertainty quantification engine.

    Provides methods to:
    1. Sample initial conditions from uncertainty ranges
    2. Sample kinetic parameters within ±30% of baseline
    3. Sample coupling parameters from ranges
    4. Aggregate results with comprehensive statistics
    """

    def __init__(self, n_samples: int = 500, seed: Optional[int] = 42):
        """
        Initialize uncertainty engine.

        Args:
            n_samples: Number of Monte Carlo samples per configuration
            seed: Random seed for reproducibility (None for random)
        """
        self.n_samples = n_samples
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def sample_initial_conditions(self, scenario: str) -> Dict[str, float]:
        """
        Sample initial conditions from uncertainty ranges for a scenario.

        Args:
            scenario: One of 'severe', 'medium', 'low'

        Returns:
            Dictionary with sampled initial conditions
        """
        if scenario not in INITIAL_CONDITION_RANGES:
            raise ValueError(f"Unknown scenario: {scenario}")

        ranges = INITIAL_CONDITION_RANGES[scenario]
        sampled = {}

        for species, value in ranges.items():
            if species == 'description':
                continue
            if isinstance(value, tuple) and len(value) == 2:
                low, high = value
                sampled[species] = np.random.uniform(low, high)

        return sampled

    def sample_kinetics(self,
                        baseline: Optional[Dict[str, float]] = None,
                        uncertainty: float = KINETIC_UNCERTAINTY,
                        boost: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Sample kinetic constants within uncertainty range.

        Args:
            baseline: Baseline kinetic constants (uses BASELINE_KINETICS if None)
            uncertainty: Fractional uncertainty (default ±30%)
            boost: Optional dict of {reaction: boost_factor} to apply after sampling

        Returns:
            Dictionary with sampled kinetic constants
        """
        if baseline is None:
            baseline = BASELINE_KINETICS

        sampled = {}
        for rxn, base_value in baseline.items():
            low = base_value * (1 - uncertainty)
            high = base_value * (1 + uncertainty)
            sampled[rxn] = np.random.uniform(low, high)

        # Apply boost if specified
        if boost is not None:
            for rxn, factor in boost.items():
                if rxn in sampled:
                    sampled[rxn] *= factor

        return sampled

    def sample_couplings(self, ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Sample coupling parameters from ranges.

        Args:
            ranges: Dictionary of {param: (low, high)} (uses COUPLING_RANGES if None)

        Returns:
            Dictionary with sampled coupling parameters
        """
        if ranges is None:
            ranges = COUPLING_RANGES

        sampled = {}
        for param, (low, high) in ranges.items():
            sampled[param] = np.random.uniform(low, high)

        return sampled

    def aggregate_results(self, results: List[Dict[str, Any]]) -> MonteCarloResult:
        """
        Aggregate Monte Carlo results into statistics.

        Args:
            results: List of result dictionaries from LP solver

        Returns:
            MonteCarloResult with comprehensive statistics
        """
        n_samples = len(results)

        # Count feasible solutions
        feasible = [r for r in results if r.get('feasible', False)]
        feasibility_rate = len(feasible) / n_samples if n_samples > 0 else 0.0

        if not feasible:
            # Return empty result if no feasible solutions
            return MonteCarloResult(
                n_samples=n_samples,
                feasibility_rate=0.0,
                ag_reduction_mean=0.0,
                ag_reduction_std=0.0,
                ag_reduction_p5=0.0,
                ag_reduction_p25=0.0,
                ag_reduction_median=0.0,
                ag_reduction_p75=0.0,
                ag_reduction_p95=0.0,
                final_violence_mean=0.0,
                final_violence_std=0.0,
                final_violence_p5=0.0,
                final_violence_p95=0.0,
                final_trust_mean=0.0,
                final_trust_std=0.0,
                final_trust_p5=0.0,
                final_trust_p95=0.0,
                final_economy_mean=0.0,
                final_economy_std=0.0,
                robustness_25=0.0,
                robustness_50=0.0,
                raw_results=results,
            )

        # Extract metrics from feasible solutions
        ag_reductions = [r.get('ag_reduction_pct', 0) for r in feasible]
        final_v = [r.get('final_V', 0) for r in feasible]
        final_t = [r.get('final_T', 0) for r in feasible]
        final_e = [r.get('final_E', 0) for r in feasible]

        # Robustness: % achieving target reductions
        robustness_25 = np.mean([1 if r >= 25 else 0 for r in ag_reductions])
        robustness_50 = np.mean([1 if r >= 50 else 0 for r in ag_reductions])

        return MonteCarloResult(
            n_samples=n_samples,
            feasibility_rate=feasibility_rate,
            # AG reduction
            ag_reduction_mean=np.mean(ag_reductions),
            ag_reduction_std=np.std(ag_reductions),
            ag_reduction_p5=np.percentile(ag_reductions, 5),
            ag_reduction_p25=np.percentile(ag_reductions, 25),
            ag_reduction_median=np.percentile(ag_reductions, 50),
            ag_reduction_p75=np.percentile(ag_reductions, 75),
            ag_reduction_p95=np.percentile(ag_reductions, 95),
            # Violence
            final_violence_mean=np.mean(final_v),
            final_violence_std=np.std(final_v),
            final_violence_p5=np.percentile(final_v, 5),
            final_violence_p95=np.percentile(final_v, 95),
            # Trust
            final_trust_mean=np.mean(final_t),
            final_trust_std=np.std(final_t),
            final_trust_p5=np.percentile(final_t, 5),
            final_trust_p95=np.percentile(final_t, 95),
            # Economy
            final_economy_mean=np.mean(final_e),
            final_economy_std=np.std(final_e),
            # Robustness
            robustness_25=robustness_25,
            robustness_50=robustness_50,
            raw_results=results,
        )

    def run_monte_carlo(self,
                        solver_func: Callable,
                        scenario: str,
                        strategy_params: Dict[str, Any],
                        fixed_params: Optional[Dict[str, Any]] = None,
                        kinetic_boost: Optional[Dict[str, float]] = None,
                        initial_boost: Optional[Dict[str, float]] = None,
                        n_samples: Optional[int] = None,
                        verbose: bool = True) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a given configuration.

        Args:
            solver_func: Function that takes (initial_state, kinetics, couplings, strategy_params, fixed_params)
                         and returns result dict
            scenario: One of 'severe', 'medium', 'low'
            strategy_params: Strategy parameters to pass to solver
            fixed_params: Fixed parameters (not sampled)
            kinetic_boost: Optional dict of {reaction: boost_factor} for reaction rate boost
            initial_boost: Optional dict of {species: boost_factor} for initial condition boost
            n_samples: Number of samples (uses self.n_samples if None)
            verbose: Print progress

        Returns:
            MonteCarloResult with aggregated statistics
        """
        if n_samples is None:
            n_samples = self.n_samples

        results = []

        for i in range(n_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Sample {i + 1}/{n_samples}")

            # Sample initial conditions
            initial_state = self.sample_initial_conditions(scenario)

            # Apply initial condition boost if specified
            if initial_boost is not None:
                for species, factor in initial_boost.items():
                    if species in initial_state:
                        initial_state[species] *= factor

            # Sample kinetic parameters
            kinetics = self.sample_kinetics(boost=kinetic_boost)

            # Sample coupling parameters
            couplings = self.sample_couplings()

            try:
                # Run solver
                result = solver_func(
                    initial_state=initial_state,
                    kinetics=kinetics,
                    couplings=couplings,
                    strategy_params=strategy_params,
                    fixed_params=fixed_params or {},
                )
                results.append(result)
            except Exception as e:
                # Record failed attempt
                results.append({
                    'feasible': False,
                    'error': str(e)[:100],
                })

        return self.aggregate_results(results)


def create_boosted_initial_conditions(scenario: str,
                                       boost_type: str,
                                       boost_factor: float = 1.30) -> Dict[str, Tuple[float, float]]:
    """
    Create modified initial condition ranges with 30% boost.

    Args:
        scenario: Base scenario ('severe', 'medium', 'low')
        boost_type: One of 'E', 'T', 'V', 'E+T'
        boost_factor: Multiplicative factor (default 1.30 = +30%)

    Returns:
        Modified ranges dictionary
    """
    ranges = INITIAL_CONDITION_RANGES[scenario].copy()

    if 'E' in boost_type:
        low, high = ranges['E']
        ranges['E'] = (low * boost_factor, high * boost_factor)

    if 'T' in boost_type:
        low, high = ranges['T']
        ranges['T'] = (low * boost_factor, high * boost_factor)

    if 'V' in boost_type:
        low, high = ranges['V']
        ranges['V'] = (low * boost_factor, high * boost_factor)

    return ranges


def get_kinetic_boost_for_variable(variable: str, boost_factor: float = 1.30) -> Dict[str, float]:
    """
    Get kinetic boost dictionary for a variable.

    Args:
        variable: One of 'E', 'T', 'V'
        boost_factor: Multiplicative factor (default 1.30 = +30%)

    Returns:
        Dictionary of {reaction: boost_factor} for relevant reactions
    """
    if variable == 'E':
        # Boost E-producing reactions: r9, r10
        return {'r9': boost_factor, 'r10': boost_factor}
    elif variable == 'T':
        # Boost T-producing reactions: r22, r23
        return {'r22': boost_factor, 'r23': boost_factor}
    elif variable == 'V':
        # Boost V-producing reactions: r27, r28, r29, r30
        return {'r27': boost_factor, 'r28': boost_factor, 'r29': boost_factor, 'r30': boost_factor}
    else:
        raise ValueError(f"Unknown variable: {variable}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def results_to_dataframe(mc_results: Dict[str, MonteCarloResult]) -> pd.DataFrame:
    """
    Convert Monte Carlo results dictionary to DataFrame.

    Args:
        mc_results: Dict mapping configuration names to MonteCarloResult objects

    Returns:
        DataFrame with one row per configuration
    """
    rows = []
    for config_name, result in mc_results.items():
        row = {
            'configuration': config_name,
            'n_samples': result.n_samples,
            'feasibility_rate': result.feasibility_rate,
            'ag_reduction_mean': result.ag_reduction_mean,
            'ag_reduction_std': result.ag_reduction_std,
            'ag_reduction_p5': result.ag_reduction_p5,
            'ag_reduction_p25': result.ag_reduction_p25,
            'ag_reduction_median': result.ag_reduction_median,
            'ag_reduction_p75': result.ag_reduction_p75,
            'ag_reduction_p95': result.ag_reduction_p95,
            'final_violence_mean': result.final_violence_mean,
            'final_violence_p5': result.final_violence_p5,
            'final_violence_p95': result.final_violence_p95,
            'final_trust_mean': result.final_trust_mean,
            'final_trust_p5': result.final_trust_p5,
            'final_trust_p95': result.final_trust_p95,
            'robustness_25': result.robustness_25,
            'robustness_50': result.robustness_50,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_mc_summary(result: MonteCarloResult, name: str = ""):
    """Print summary of Monte Carlo result."""
    print(f"\n{'='*60}")
    print(f"Monte Carlo Summary: {name}")
    print(f"{'='*60}")
    print(f"Samples: {result.n_samples}")
    print(f"Feasibility: {result.feasibility_rate*100:.1f}%")
    print(f"\nAG Reduction:")
    print(f"  Mean: {result.ag_reduction_mean:.1f}%")
    print(f"  Std:  {result.ag_reduction_std:.1f}%")
    print(f"  5th percentile (worst case): {result.ag_reduction_p5:.1f}%")
    print(f"  Median: {result.ag_reduction_median:.1f}%")
    print(f"  95th percentile (best case): {result.ag_reduction_p95:.1f}%")
    print(f"\nRobustness:")
    print(f"  Achieving 25% reduction: {result.robustness_25*100:.1f}%")
    print(f"  Achieving 50% reduction: {result.robustness_50*100:.1f}%")
    print(f"\nFinal Violence: {result.final_violence_mean:.1f} (range: {result.final_violence_p5:.1f} - {result.final_violence_p95:.1f})")
    print(f"Final Trust: {result.final_trust_mean:.1f} (range: {result.final_trust_p5:.1f} - {result.final_trust_p95:.1f})")


if __name__ == "__main__":
    # Test the uncertainty engine
    print("Testing Uncertainty Engine...")

    engine = UncertaintyEngine(n_samples=10, seed=42)

    # Test sampling
    print("\n1. Sample initial conditions (severe):")
    ic = engine.sample_initial_conditions('severe')
    for k, v in ic.items():
        print(f"  {k}: {v:.2f}")

    print("\n2. Sample kinetics:")
    kinetics = engine.sample_kinetics()
    print(f"  r9 (baseline {BASELINE_KINETICS['r9']}): {kinetics['r9']:.4f}")
    print(f"  r21 (baseline {BASELINE_KINETICS['r21']}): {kinetics['r21']:.4f}")

    print("\n3. Sample couplings:")
    couplings = engine.sample_couplings()
    for k, v in couplings.items():
        print(f"  {k}: {v:.3f}")

    print("\n4. Test kinetic boost for E (+30%):")
    boost = get_kinetic_boost_for_variable('E', 1.30)
    print(f"  Boosted reactions: {boost}")

    print("\nUncertainty Engine tests passed!")
