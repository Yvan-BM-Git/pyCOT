#!/usr/bin/env python3
"""
XCEPT Conflict Analysis - Main Entry Point
===========================================
Lake Chad Basin Conflict Model Analysis

This script runs comprehensive conflict resolution analysis comparing
government strategies vs armed group strategies under various scenarios.

Usage:
    python run_analysis.py                    # Run with default parameters
    python run_analysis.py --quick            # Quick test run (fewer scenarios)
    python run_analysis.py --scenario severe  # Run specific scenario only

The analysis uses the Resource_Community_Insurgency_Loops_model3 reaction network
which models conflict dynamics through 36 reactions across 7 categories.

Outputs are saved to:
    - outputs/          : CSV results files
    - visualizations/   : PNG plots and HTML interactive visualizations
    - reports/          : Text analysis reports
"""

import os
import sys
import argparse

# Ensure proper path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYCOT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PYCOT_ROOT, 'src'))

# Create output directories
os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'reports'), exist_ok=True)

# ============================================================================
# CONFIGURABLE INITIAL CONDITIONS
# ============================================================================
# Modify these values to test different starting scenarios
# All population values in units of 100,000 people

INITIAL_CONDITIONS = {
    # Severe Conflict: Active insurgency (Lake Chad Basin ~2014-2015)
    'severe': {
        'description': 'Active insurgency with widespread violence',
        'state': {
            'SR_RL': 8.0,    # 800K strong-resilient on resource land
            'SR_SL': 12.0,   # 1.2M strong-resilient on stressed land
            'WR_RL': 10.0,   # 1M weak-resilient on resource land
            'WR_SL': 15.0,   # 1.5M weak-resilient on stressed land
            'AG_RL': 0.9,    # 90K armed groups on resource land
            'AG_SL': 1.9,    # 190K armed groups on stressed land
            'RL': 20.0,      # Restored land
            'SL': 100.0,     # Stressed land
            'E': 50.0,       # Economy (weak)
            'T': 20.0,       # Trust (low)
            'V': 50.0,       # Violence (high)
            'Gov': 10.0,     # Governance (weak)
        }
    },

    # Medium Conflict: Ongoing tensions (current situation ~2020-2023)
    'medium': {
        'description': 'Regional instability with active tensions',
        'state': {
            'SR_RL': 12.0,
            'SR_SL': 8.0,
            'WR_RL': 15.0,
            'WR_SL': 10.0,
            'AG_RL': 0.7,
            'AG_SL': 1.7,
            'RL': 20.0,
            'SL': 100.0,
            'E': 80.0,
            'T': 40.0,
            'V': 30.0,
            'Gov': 20.0,
        }
    },

    # Low Conflict: Stabilizing region
    'low': {
        'description': 'Stable region with minor tensions',
        'state': {
            'SR_RL': 15.0,
            'SR_SL': 12.0,
            'WR_RL': 10.0,
            'WR_SL': 8.0,
            'AG_RL': 0.5,
            'AG_SL': 1.5,
            'RL': 20.0,
            'SL': 100.0,
            'E': 100.0,
            'T': 60.0,
            'V': 10.0,
            'Gov': 30.0,
        }
    },
}

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

GOVERNMENT_STRATEGIES = {
    'governance_only': 'State-building focus: institutions, security forces',
    'economic_only': 'Development focus: livelihoods, infrastructure',
    'security_only': 'Military focus: counter-insurgency, DDR',
    'balanced': 'Multi-pronged: equal investment across all dimensions',
    'integrated': 'Coordinated approach with strong inter-sector coupling',
}

ARMED_GROUP_STRATEGIES = {
    'aggressive': 'High violence, moderate recruitment',
    'recruitment': 'Focus on recruitment over violence',
    'territorial': 'Territorial expansion focus',
    'defensive': 'Defensive posture',
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Climate conditions (0=harsh, 1=favorable)
CLIMATE_CONDITIONS = [0.3, 0.5, 0.7, 0.9]

# Budget levels ($ millions per year)
PEACE_BUDGETS = [500, 1000, 2000, 3000]
AG_BUDGETS = [50, 100, 200, 500]

# Target AG reductions
AG_REDUCTION_TARGETS = [0.25, 0.50, 0.75]


def print_configuration():
    """Print current analysis configuration."""
    print("\n" + "="*70)
    print("XCEPT CONFLICT ANALYSIS CONFIGURATION")
    print("="*70)

    print("\nInitial Condition Scenarios:")
    for name, config in INITIAL_CONDITIONS.items():
        state = config['state']
        total_ag = state['AG_RL'] + state['AG_SL']
        print(f"  {name:12s}: {config['description']}")
        print(f"               AG={total_ag:.1f} ({int(total_ag*100000):,} fighters), "
              f"V={state['V']:.0f}, Gov={state['Gov']:.0f}")

    print("\nGovernment Strategies:")
    for name, desc in GOVERNMENT_STRATEGIES.items():
        print(f"  {name:20s}: {desc}")

    print("\nArmed Group Strategies:")
    for name, desc in ARMED_GROUP_STRATEGIES.items():
        print(f"  {name:15s}: {desc}")

    print(f"\nClimate conditions: {CLIMATE_CONDITIONS}")
    print(f"Peace budgets: ${PEACE_BUDGETS} M/year")
    print(f"AG budgets: ${AG_BUDGETS} M/year")
    print(f"AG reduction targets: {[f'{t:.0%}' for t in AG_REDUCTION_TARGETS]}")


def run_calibrated_analysis(quick=False, scenario=None):
    """Run the calibrated conflict game analysis."""
    print("\n" + "="*70)
    print("RUNNING CALIBRATED CONFLICT GAME ANALYSIS")
    print("="*70)

    # Change to script directory for proper relative paths
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        # Import the analysis module
        from script_calibrated_conflict import (
            run_calibrated_analysis as _run_analysis,
            print_calibrated_results,
            create_calibrated_visualizations
        )

        # Run analysis
        results_df = _run_analysis()

        # Save results
        output_path = os.path.join(SCRIPT_DIR, 'outputs', 'calibrated_conflict_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print summary
        print_calibrated_results(results_df)

        # Create visualizations
        os.chdir(os.path.join(SCRIPT_DIR, 'outputs'))
        create_calibrated_visualizations(results_df)

        return results_df

    finally:
        os.chdir(original_dir)


def run_adaptive_simulation(quick=False, scenario=None):
    """Run adaptive budget simulation with ODE dynamics."""
    print("\n" + "="*70)
    print("RUNNING ADAPTIVE BUDGET ODE SIMULATION")
    print("="*70)

    # Change to script directory for proper relative paths
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        # Import and run the adaptive simulation
        import script_two_budget_adaptive
        # The script runs on import via __main__ block
        # We can also call specific functions if needed

    finally:
        os.chdir(original_dir)


def run_transition_analysis(quick=False, scenario=None):
    """Run transition resolution analysis comparing mono vs multi-pronged strategies."""
    print("\n" + "="*70)
    print("RUNNING TRANSITION RESOLUTION ANALYSIS")
    print("="*70)

    # Change to script directory for proper relative paths
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        # Import the transition analysis module
        import script_transition_resolution

    finally:
        os.chdir(original_dir)


def run_organization_analysis():
    """Run organization analysis and visualization."""
    print("\n" + "="*70)
    print("RUNNING ORGANIZATION STRUCTURE ANALYSIS")
    print("="*70)

    # Change to script directory for proper relative paths
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        import script_print_conflict_organizations

    finally:
        os.chdir(original_dir)


def main():
    """Main entry point for XCEPT conflict analysis."""
    parser = argparse.ArgumentParser(
        description='XCEPT Conflict Analysis - Lake Chad Basin Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                    Run full analysis
  python run_analysis.py --quick            Quick test run
  python run_analysis.py --scenario severe  Analyze severe conflict only
  python run_analysis.py --calibrated       Run calibrated game analysis only
  python run_analysis.py --adaptive         Run adaptive ODE simulation only
  python run_analysis.py --transition       Run transition analysis only
  python run_analysis.py --organizations    Run organization analysis only
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Run quick analysis with fewer scenarios')
    parser.add_argument('--scenario', choices=['low', 'medium', 'severe'],
                        help='Run analysis for specific scenario only')
    parser.add_argument('--calibrated', action='store_true',
                        help='Run calibrated conflict game analysis only')
    parser.add_argument('--adaptive', action='store_true',
                        help='Run adaptive budget ODE simulation only')
    parser.add_argument('--transition', action='store_true',
                        help='Run transition resolution analysis only')
    parser.add_argument('--organizations', action='store_true',
                        help='Run organization structure analysis only')
    parser.add_argument('--config', action='store_true',
                        help='Show configuration and exit')

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("XCEPT CONFLICT ANALYSIS")
    print("Lake Chad Basin Conflict Model")
    print("="*70)
    print(f"\nModel: Resource_Community_Insurgency_Loops_model3.txt")
    print(f"Script directory: {SCRIPT_DIR}")

    if args.config:
        print_configuration()
        return

    print_configuration()

    # Determine which analyses to run
    run_all = not (args.calibrated or args.adaptive or args.transition or args.organizations)

    try:
        if run_all or args.calibrated:
            run_calibrated_analysis(quick=args.quick, scenario=args.scenario)

        if run_all or args.transition:
            run_transition_analysis(quick=args.quick, scenario=args.scenario)

        if run_all or args.organizations:
            run_organization_analysis()

        if run_all or args.adaptive:
            run_adaptive_simulation(quick=args.quick, scenario=args.scenario)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nOutput files saved to:")
        print(f"  - {os.path.join(SCRIPT_DIR, 'outputs')}/")
        print(f"  - {os.path.join(SCRIPT_DIR, 'visualizations')}/")
        print(f"  - {os.path.join(SCRIPT_DIR, 'reports')}/")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
