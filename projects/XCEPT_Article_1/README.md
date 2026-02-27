# XCEPT Article 1 - Lake Chad Basin Conflict Model Analysis

This project implements conflict resolution analysis using Chemical Organization Theory (COT) applied to the Lake Chad Basin conflict model.

## Overview

The analysis explores government vs armed group strategy effectiveness under various conflict scenarios, budget levels, and climate conditions. Key findings show that **multi-pronged interventions** addressing governance + economics + trust + security simultaneously are required for sustainable peace - single-focus strategies consistently fail.

## Quick Start

```bash
# From the scripts directory
cd projects/XCEPT_Article_1/scripts

# Run the full analysis
python run_analysis.py

# Run with specific options
python run_analysis.py --quick           # Quick test run (fewer scenarios)
python run_analysis.py --calibrated      # Run calibrated game analysis only
python run_analysis.py --transition      # Run transition analysis only
python run_analysis.py --organizations   # Run organization structure analysis
python run_analysis.py --adaptive        # Run adaptive ODE simulation
python run_analysis.py --config          # Show configuration and exit
```

## Project Structure

```
XCEPT_Article_1/
├── scripts/
│   ├── data/
│   │   └── Resource_Community_Insurgency_Loops_model3.txt  # Reaction network model
│   ├── run_analysis.py                    # Main entry point
│   ├── script_calibrated_conflict.py      # Game-theoretic analysis
│   ├── script_transition_resolution.py    # Mono vs multi-pronged comparison
│   ├── script_two_budget_adaptive.py      # ODE simulation with adaptive strategies
│   ├── script_multiscale_conflict.py      # Temporal scale analysis
│   └── script_print_conflict_organizations.py  # Organization visualization
├── outputs/                               # CSV results files
├── visualizations/                        # PNG plots and HTML visualizations
└── reports/                              # Text analysis reports
```

## Scripts Description

### `run_analysis.py` - Main Entry Point
Orchestrates all analyses with configurable initial conditions. Allows modulating:
- Conflict scenarios (low, medium, severe)
- Climate conditions
- Budget levels
- Strategy combinations

### `script_calibrated_conflict.py` - Calibrated Conflict Game
Implements a two-budget game comparing:
- **Government strategies**: governance-only, economic-only, security-only, balanced, integrated
- **Armed group strategies**: aggressive, recruitment-focused, territorial, defensive

Uses real-world calibration based on Lake Chad Basin data (UN OCHA appeals, DDR costs, population estimates).

### `script_transition_resolution.py` - Strategy Comparison
Analyzes the Three Interlocking Loops framework:
1. Resource-Scarcity Loop: Climate → Degradation → Resilience Loss
2. Scarcity-Conflict Loop: Violence → Trust Erosion → Institutional Vacuum
3. Insurgency-Grievance Loop: Grievances → Recruitment → Violence

**Key finding**: Single-focus strategies fail because violence destroys trust-building, economic development requires security, and governance needs economic foundation.

### `script_two_budget_adaptive.py` - ODE Simulation
Runs dynamic simulations with:
- Manual consumption tracking
- Seasonal climate variation
- Adaptive government budget allocation
- Multiple conflict scenarios

### `script_multiscale_conflict.py` - Temporal Analysis
Analyzes intervention effectiveness across temporal scales to determine if strategies work through:
- Short-term accumulation (immediate effects)
- Long-term adaptation (system reorganization)

### `script_print_conflict_organizations.py` - Organization Analysis
Computes and visualizes all conflict organizations from the reaction network:
- Colors nodes by conflict species presence
- Creates interactive HTML visualization
- Generates detailed text reports

## Model Description

The **Resource_Community_Insurgency_Loops_model3** reaction network includes:
- **12 species**: SR_RL, SR_SL, WR_RL, WR_SL, AG_RL, AG_SL, RL, SL, E, T, V, Gov
- **36 reactions** organized in 7 categories:
  - Land and Resource Dynamics (r1-r8)
  - Economic Production and Governance (r9-r12)
  - Population Migration (r13-r17)
  - Resilience Transitions (r18-r21)
  - Trust and Social Cohesion (r22-r26)
  - Violence Generation (r27-r31)
  - Armed Group Dynamics (r32-r36)

## Requirements

- Python 3.11+
- pyCOT library (from parent repository)
- Dependencies: numpy, pandas, matplotlib, seaborn, pulp, scipy, pyvis

## Installation

1. Clone the pyCOT repository
2. Install dependencies: `pip install -e .` or `poetry install`
3. Navigate to scripts directory and run analysis

## Output Files

- `calibrated_conflict_results.csv` - Game-theoretic analysis results
- `strategy_comparison_results.csv` - Strategy comparison data
- `calibrated_conflict_results.png` - Visualization of strategy effectiveness
- `strategy_comparison_results.png` - Mono vs multi-pronged comparison
- `conflict_orgs_*.html` - Interactive organization visualization
- `conflict_report_*.txt` - Detailed analysis reports

## Key Research Findings

1. **Multi-pronged strategies required**: Success rates 70-90% vs 20-40% for single-focus
2. **Budget ratio matters**: Peace budget must significantly exceed AG budget
3. **Climate exacerbates conflict**: Poor climate increases all intervention costs
4. **Coupling is critical**: DDR requires governance capacity and economic alternatives
5. **Trust building needs security**: Violence destroys trust-building effectiveness

## Citation

If you use this analysis, please cite the XCEPT project and the pyCOT library.

## License

GNU v.3
