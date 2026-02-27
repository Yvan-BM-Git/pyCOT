# XCEPT Article 2 - Strategy Prioritisation & Scale of Effort Analysis

This project analyzes conflict resolution strategies under uncertainty using Monte Carlo sampling.

## Research Questions

### Question 1: Strategy Prioritisation
**What happens if I prioritise security over development? Or development over security?**

Analyzes the full spectrum from pure development focus (0% security) to pure security focus (100% security), measuring:
- Expected AG reduction
- Worst-case outcomes (5th percentile)
- Robustness (probability of achieving targets)

### Question 2: Scale of Effort
**What happens if I increase with 30%:**
- **i.** Economic output (E)
- **ii.** Trust - social cohesion (T)
- **iii.** Violence (V)

Tests two approaches:
- **Approach A**: Initial condition boost (+30% starting value)
- **Approach B**: Reaction rate boost (+30% production rates)

## Key Feature: Decision-Making Under Uncertainty

All analyses use Monte Carlo sampling because we don't know exact:
- Initial conditions
- Kinetic parameters
- Coupling coefficients

Results provide:
- **Expected values** (mean across samples)
- **Worst-case** (5th percentile)
- **Best-case** (95th percentile)
- **Robustness** (% of samples achieving targets)

## Quick Start

```bash
cd projects/XCEPT_Article_2/scripts

# Run full analysis
python run_article2_analysis.py

# Quick test run
python run_article2_analysis.py --quick

# Run specific analysis
python run_article2_analysis.py --prioritisation  # Q1 only
python run_article2_analysis.py --scale           # Q2 only

# Show configuration
python run_article2_analysis.py --config
```

## Project Structure

```
XCEPT_Article_2/
├── scripts/
│   ├── data/
│   │   └── Resource_Community_Insurgency_Loops_model3.txt
│   ├── uncertainty_engine.py              # Monte Carlo core module
│   ├── run_article2_analysis.py           # Main entry point
│   ├── script_prioritisation_spectrum.py  # Q1 analysis
│   └── script_scale_of_effort.py          # Q2 analysis
├── outputs/                               # CSV results
├── visualizations/                        # PNG plots
└── reports/                              # Text reports
```

## Uncertainty Ranges

### Initial Conditions (by scenario)

| Variable | Severe | Medium | Low |
|----------|--------|--------|-----|
| E (Economy) | 30-70 | 60-100 | 80-120 |
| T (Trust) | 10-30 | 30-50 | 50-70 |
| V (Violence) | 35-65 | 20-40 | 5-15 |
| Gov | 5-15 | 15-25 | 25-40 |

### Kinetic Parameters
All reaction rates sampled with ±30% uncertainty from baseline.

### Coupling Parameters
- α (Trust efficiency): 0.2-0.6
- β (DDR-governance): 0.3-0.7
- γ (DDR-economy): 0.3-0.7
- δ (Institution-economy): 0.4-0.8

## Output Files

### Question 1 (Prioritisation)
- `prioritisation_mc_results.csv` - All Monte Carlo results
- `prioritisation_expected_value.png` - Mean outcomes by priority level
- `prioritisation_confidence_bands.png` - 90% confidence intervals
- `prioritisation_heatmap.png` - Priority vs scenario effectiveness
- `prioritisation_robustness_heatmap.png` - Target achievement rates
- `prioritisation_report.txt` - Key findings summary

### Question 2 (Scale of Effort)
- `scale_of_effort_initial_boost.csv` - Initial condition boost results
- `scale_of_effort_rate_boost.csv` - Reaction rate boost results
- `boost_comparison_boxplots.png` - Comparison of all boost types
- `E_boost_distribution.png` - Economic output impact
- `T_boost_distribution.png` - Trust impact
- `V_boost_distribution.png` - Violence impact
- `initial_vs_rate_boost.png` - Approach comparison
- `scale_of_effort_report.txt` - Key findings summary

## Methodology

### Prioritisation Spectrum (Q1)
Budget allocation varies continuously with security priority `p ∈ [0, 1]`:

| Category | p=0 (Dev) | p=0.5 (Balanced) | p=1 (Sec) |
|----------|-----------|------------------|-----------|
| Governance | 15% | 42% | 70% |
| Economic | 70% | 40% | 10% |
| Trust | 25% | 15% | 5% |
| DDR | 10% | 40% | 70% |

### Scale of Effort (Q2)

**Approach A - Initial Condition Boost:**
- Multiply initial E, T, or V by 1.30

**Approach B - Reaction Rate Boost:**
- E boost: r9, r10 rates × 1.30
- T boost: r22, r23 rates × 1.30
- V boost: r27, r28, r29, r30 rates × 1.30

## Requirements

- Python 3.11+
- pyCOT library
- pulp, numpy, pandas, matplotlib, seaborn

## License

GNU v.3
