#!/usr/bin/env python3
"""
XCEPT Article 2 - Main Analysis Entry Point (ODE Simulation)
=============================================================

This script runs the analyses for answering:

Question 1 (Prioritisation):
    What happens if I prioritise security over development?
    Or development over security?

Question 2 (Scale of Effort):
    What happens if I increase with 30%:
    i.   Economic output (E)
    ii.  Trust - social cohesion (T)
    iii. Violence (V)

All analyses use ODE SIMULATION with Monte Carlo sampling to handle uncertainty in:
- Initial conditions
- Kinetic parameters

Usage:
    python run_article2_analysis.py                 # Run both analyses
    python run_article2_analysis.py --prioritisation # Q1 only
    python run_article2_analysis.py --scale         # Q2 only
    python run_article2_analysis.py --config        # Show configuration
    python run_article2_analysis.py --quick         # Quick test (fewer samples)
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


def print_configuration():
    """Print analysis configuration."""
    print("\n" + "="*70)
    print("XCEPT ARTICLE 2 - ANALYSIS CONFIGURATION (ODE Simulation)")
    print("="*70)

    print("\n" + "-"*50)
    print("QUESTION 1: PRIORITISATION SPECTRUM")
    print("-"*50)
    print("""
Analyzes the trade-off between Security and Development focus using ODE simulation.

Simulation time: 5 time units (5 years)

Security Priority Levels:
  0%   = Pure Development (economic + trust focus)
  50%  = Balanced approach
  100% = Pure Security (DDR + governance focus)

Tests 11 levels: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

Reaction rate boosts vary with priority:
  - Governance:  1x (dev) to 3x (sec)
  - Economic:    3x (dev) to 1x (sec)
  - Trust:       2.5x (dev) to 1x (sec)
  - DDR:         1x (dev) to 3x (sec)
""")

    print("\n" + "-"*50)
    print("QUESTION 2: SCALE OF EFFORT (+30%)")
    print("-"*50)
    print("""
Analyzes impact of 30% increase in key variables.

Simulation time: 5 time units (5 years)

Variables tested:
  i.   E (Economic output)
  ii.  T (Trust/social cohesion)
  iii. V (Violence)

Two approaches:
  A. Initial Condition Boost: Start with 30% more E/T/V
  B. Reaction Rate Boost: Generate E/T/V 30% faster

Boost conditions:
  - baseline (no boost)
  - E_boost / E_rate_boost
  - T_boost / T_rate_boost
  - V_boost / V_rate_boost
  - E+T_boost (combined)
""")

    print("\n" + "-"*50)
    print("UNCERTAINTY HANDLING (Monte Carlo)")
    print("-"*50)
    print("""
All parameters are sampled from uncertainty ranges:

Initial Conditions:
  - Severe: E(30-70), T(10-30), V(35-65), Gov(5-15)
  - Medium: E(60-100), T(30-50), V(20-40), Gov(15-25)
  - Low:    E(80-120), T(50-70), V(5-15), Gov(25-40)

Kinetic Parameters: ±30% from baseline values
Coupling Parameters: Sampled from typical ranges

Output Statistics:
  - Expected value (mean)
  - Worst-case (5th percentile)
  - Best-case (95th percentile)
  - Robustness (% achieving target)
""")


def run_prioritisation_analysis(n_samples: int = 100, quick: bool = False):
    """Run Question 1: Prioritisation Spectrum Analysis (ODE Simulation)."""
    print("\n" + "="*70)
    print("RUNNING QUESTION 1: PRIORITISATION SPECTRUM ANALYSIS (ODE Simulation)")
    print("="*70)

    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        from script_prioritisation_spectrum import main as run_main

        # Run analysis
        results_df = run_main(n_samples=n_samples, quick=quick)

        return results_df

    finally:
        os.chdir(original_dir)


def run_scale_of_effort_analysis(n_samples: int = 100, quick: bool = False):
    """Run Question 2: Scale of Effort Analysis (ODE Simulation)."""
    print("\n" + "="*70)
    print("RUNNING QUESTION 2: SCALE OF EFFORT ANALYSIS (ODE Simulation)")
    print("="*70)

    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        from script_scale_of_effort import main as run_main

        # Run analysis
        initial_df, rate_df = run_main(n_samples=n_samples, quick=quick)

        return initial_df, rate_df

    finally:
        os.chdir(original_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='XCEPT Article 2 - Strategy & Scale Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_article2_analysis.py                  Run full analysis
    python run_article2_analysis.py --quick          Quick test run
    python run_article2_analysis.py --prioritisation Q1 only
    python run_article2_analysis.py --scale          Q2 only
    python run_article2_analysis.py --config         Show configuration
        """
    )

    parser.add_argument('--config', action='store_true',
                        help='Show configuration and exit')
    parser.add_argument('--prioritisation', action='store_true',
                        help='Run prioritisation analysis (Q1) only')
    parser.add_argument('--scale', action='store_true',
                        help='Run scale of effort analysis (Q2) only')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with fewer samples')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of Monte Carlo samples per configuration')

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("XCEPT ARTICLE 2")
    print("Strategy Prioritisation & Scale of Effort Analysis")
    print("="*70)
    print(f"\nModel: Resource_Community_Insurgency_Loops_model3.txt")
    print(f"Script directory: {SCRIPT_DIR}")

    if args.config:
        print_configuration()
        return 0

    print_configuration()

    # Determine what to run
    run_q1 = args.prioritisation or not (args.prioritisation or args.scale)
    run_q2 = args.scale or not (args.prioritisation or args.scale)

    n_samples = 50 if args.quick else args.samples

    print(f"\nMonte Carlo samples per configuration: {n_samples}")
    if args.quick:
        print("(Quick mode: reduced scenarios and samples)")

    try:
        if run_q1:
            run_prioritisation_analysis(n_samples=n_samples, quick=args.quick)

        if run_q2:
            run_scale_of_effort_analysis(n_samples=n_samples, quick=args.quick)

        print("\n" + "="*70)
        print("ALL ANALYSES COMPLETE")
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
