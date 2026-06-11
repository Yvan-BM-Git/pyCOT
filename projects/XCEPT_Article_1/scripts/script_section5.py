#!/usr/bin/env python3
"""
script_section5.py
==================
Generates all figures for Section 5 of the paper
(Chemical Organization Theory and the Statistical Geometry of Regimes).

Figures produced:
  fig2_feasibility.png          -- intra-org PC/CP ESMO feasibility bars
  fig_inter2_combo_counts.png   -- inter-org ESMO counts by state-combo type
  fig_inter3_struct_entropy.png -- structural entropy per organization
  fig_markov_loops.png          -- Markov hierarchy graph (matplotlib)
  markov_hierarchy.html         -- Markov hierarchy graph (interactive PyVis)

All outputs go to outputs/section5/.

Heavy computation (LP/ESMO counting) is cached in:
  outputs/complex_COT_ESMO/transition_summary.csv   (inter-org, 32 rows)
  outputs/intra_org_esmo/esmo_cache_intra.csv        (intra-org)

If these CSVs do not exist, run the following scripts first:
  python script_complex_COT_ESMO.py   (generates transition_summary.csv)
  python script_intra_org_esmo.py     (generates esmo_cache_intra.csv)
Then re-run this script.
"""

import os
import sys
import shutil
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'outputs', 'section5'))
os.makedirs(_OUT_DIR, exist_ok=True)

# Cache files expected by the plotting scripts
_CACHE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'outputs', 'complex_COT_ESMO'))
_INTRA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'outputs', 'intra_org_esmo'))

CACHE_INTER = os.path.join(_CACHE_DIR, 'transition_summary.csv')
CACHE_INTRA = os.path.join(_INTRA_DIR, 'esmo_cache_intra.csv')


def _require_cache():
    ok = True
    for path, script in [
        (CACHE_INTER, 'script_complex_COT_ESMO.py'),
        (CACHE_INTRA, 'script_intra_org_esmo.py'),
    ]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            print(f"  Run: python {script}")
            ok = False
    if not ok:
        sys.exit(1)


def _run(script_name):
    path = os.path.join(_SCRIPT_DIR, script_name)
    print(f"\n--- Running {script_name} ---")
    subprocess.run([sys.executable, path], check=True, cwd=_SCRIPT_DIR)


def _copy_outputs():
    copies = [
        # (src_dir, filename, dst_dir)
        (_CACHE_DIR, 'fig_inter2_combo_counts.png', _OUT_DIR),
        (_CACHE_DIR, 'fig_inter3_struct_entropy.png', _OUT_DIR),
        (_CACHE_DIR, 'fig_markov_loops.png', _OUT_DIR),
        (_CACHE_DIR, 'markov_hierarchy.html', _OUT_DIR),
        (_INTRA_DIR, 'fig2_feasibility.png', _OUT_DIR),
        # also copy cache CSVs for completeness
        (_CACHE_DIR, 'transition_summary.csv', _OUT_DIR),
        (_INTRA_DIR, 'esmo_cache_intra.csv', _OUT_DIR),
    ]
    for src_dir, fname, dst_dir in copies:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst_dir)
            print(f"  Copied: {fname} -> {dst_dir}")


if __name__ == '__main__':
    _require_cache()
    _run('plot_intra_org_figures.py')
    _run('plot_inter_org_figures.py')
    _run('plot_markov_hierarchy.py')
    _copy_outputs()
    print(f"\nAll Section 5 figures saved to: {_OUT_DIR}")
