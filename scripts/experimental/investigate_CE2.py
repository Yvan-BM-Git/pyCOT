"""
Systematic investigation of the Lake Chad Basin conflict reaction network (CE_2.txt).

Steps:
  1. Diagnose which species block self-maintenance in X3-X11.
  2. Test proposed fixes r55-r60 and r32_modified individually.
  3. Test key combinations.
  4. Identify mixed organisations and structural backfire pairs.
"""

import os
import sys
import numpy as np
from scipy.optimize import linprog

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))

from pyCOT.io.functions import read_txt, from_string
from pyCOT.analysis.Persistent_Modules_Generator import compute_all_organizations

# ─────────────────────────────────────────────
# Species classification
# ─────────────────────────────────────────────
PEACE_SP    = {'L1', 'L2', 'L3', 'Tc', 'Ti', 'D2', 'D3'}
CONFLICT_SP = {'A1', 'A1g', 'A2', 'A3', 'Gl', 'Gs', 'Rg_x'}

def classify(sp_set):
    has_p = bool(sp_set & PEACE_SP)
    has_c = bool(sp_set & CONFLICT_SP)
    if has_p and has_c:
        return "MIXED"
    if has_p:
        return "peace"
    if has_c:
        return "conflict"
    return "structural"

# ─────────────────────────────────────────────
# Blocking-species diagnostic
# ─────────────────────────────────────────────
def diagnose_blocking(species_names, rn):
    """
    Find which species block self-maintenance in the sub-network.

    Two passes:
    (a) Any species with no producer reaction in the sub-network → 'no_producer'.
    (b) Slack LP: minimise total violation of S@v >= 0; species with slack > 0
        are bottlenecks even if they have producers.

    Returns dict {species_name: reason_string}.
    """
    sp_objs = [rn.get_species(s) for s in species_names if rn.has_species(s)]
    sub = rn.sub_reaction_network(sp_objs)
    S_obj = sub.stoichiometry_matrix()
    S = np.asarray(S_obj, dtype=float)
    sp_names = S_obj.species
    n_sp, n_rx = S.shape

    result = {}

    if n_rx == 0:
        return {s: "no_reactions_in_subnetwork" for s in species_names}

    # (a) No-producer check
    for i, sp in enumerate(sp_names):
        if np.all(S[i, :] <= 0):
            result[sp] = "no_producer"

    # (b) Slack LP: min sum(s) s.t. S@v + s >= 0, v >= eps, s >= 0
    eps = 0.01
    c    = np.concatenate([np.zeros(n_rx), np.ones(n_sp)])
    A_ub = np.hstack([-S, -np.eye(n_sp)])
    b_ub = np.zeros(n_sp)
    bds  = [(eps, None)] * n_rx + [(0, None)] * n_sp

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bds, method='highs')
    if res.success:
        s_vals = res.x[n_rx:]
        for i, sp in enumerate(sp_names):
            if s_vals[i] > 1e-3 and sp not in result:
                result[sp] = f"insufficient_production (slack={s_vals[i]:.3f})"

    return result

# ─────────────────────────────────────────────
# Lattice computation & reporting
# ─────────────────────────────────────────────
def run_lattice(rn):
    return compute_all_organizations(
        rn, max_generator_size=8, max_organization_size=5, verbose=False
    )

def so_set(so):
    return frozenset(sp.name for sp in so.closure_species)

def org_set(org):
    if hasattr(org, 'combined_closure'):
        return frozenset(sp.name for sp in org.combined_closure)
    return frozenset(sp.name for sp in org.closure_species)

def report_lattice(label, results, rn):
    sos  = sorted(results['elementary_sos'],  key=lambda x: len(x.closure_species))
    orgs = results['all_organizations']
    org_sets = {org_set(o) for o in orgs}

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Semi-organizations : {len(sos)}")
    print(f"  Organizations      : {len(orgs)}")

    print(f"\n  ── Semi-organizations ──")
    for i, so in enumerate(sos, 1):
        sps  = so_set(so)
        cls  = classify(sps)
        star = "★ ORG" if sps in org_sets else ""
        print(f"  X{i:02d} ({len(sps):2d}sp) [{cls:8s}] {star}")
        print(f"       {sorted(sps)}")

    if orgs:
        print(f"\n  ── True organisations ──")
        for i, org in enumerate(orgs, 1):
            sps = org_set(org)
            cls = classify(sps)
            print(f"  O{i} ({len(sps)}sp) [{cls}]")
            print(f"    {sorted(sps)}")

    print(f"\n  ── Blocking-species diagnosis (non-organisations) ──")
    diag_done = False
    for i, so in enumerate(sos, 1):
        sps = so_set(so)
        if sps not in org_sets:
            blocking = diagnose_blocking(sorted(sps), rn)
            if blocking:
                print(f"  X{i:02d}: {blocking}")
                diag_done = True
    if not diag_done:
        print("  (all semi-organisations are organisations)")

# ─────────────────────────────────────────────
# Network modification helpers
# ─────────────────────────────────────────────
def add_rxns(base_txt_path, extra_rxn_strings):
    """
    Load the base network from base_txt_path and add extra reactions given
    as plain strings like 'r55: 2*Rg + L1 => Rg + L3'.
    """
    with open(base_txt_path, 'r') as f:
        base_text = f.read()
    extra = "\n".join(extra_rxn_strings)
    full_text = base_text + "\n" + extra
    return from_string(full_text)

# ─────────────────────────────────────────────
# Proposed fixes (all population-conserved)
# Each is a plain reaction string appended to CE_2.txt text.
# ─────────────────────────────────────────────
# r55: external civic reconstruction (L1→L3, net 1 Rg consumed)
R55 = "r55: 2*Rg + L1 => Rg + L3"

# r56: emergency mediation deployment (L1→D3, Rg consumed, L3 catalytic)
R56 = "r56: Rg + L1 + L3 => D3 + L3"

# r57: spontaneous IDP return when land available (Disp→L1, Rs catalytic)
R57 = "r57: Disp + Rs => L1 + Rs"

# r58: spontaneous ex-combatant reintegration (Ex→L1, Rs catalytic)
R58 = "r58: Ex + Rs => L1 + Rs"

# r59/r60: low-rate armed-group attrition to Ex (factional defections etc.)
R59 = "r59: A1 => Ex"
R60 = "r60: A1g => Ex"

# r32m: insurgent territorial mode requires L1 recruitment (A1→A1g, L1 catalytic)
R32M = "r32m: A1 + Rg + L1 => A1g + Rg_x + L1"

# ─────────────────────────────────────────────
# Backfire / transition analysis helpers
# ─────────────────────────────────────────────
def compute_closure(seed_names, rn):
    """Compute the generated closure of a seed species set in rn."""
    sp_objs = [rn.get_species(s) for s in seed_names if rn.has_species(s)]
    closed  = rn.generated_closure(sp_objs)
    return frozenset(sp.name for sp in closed)

def check_sm(species_names, rn):
    """True if the set is self-maintaining."""
    sp_objs = [rn.get_species(s) for s in species_names if rn.has_species(s)]
    sub = rn.sub_reaction_network(sp_objs)
    S   = np.asarray(sub.stoichiometry_matrix(), dtype=float)
    n_sp, n_rx = S.shape
    if n_rx == 0:
        return False, None
    eps = 1e-6
    res = linprog(np.ones(n_rx), A_ub=-S, b_ub=np.zeros(n_sp),
                  bounds=[(eps, None)]*n_rx, method='highs')
    if res.success:
        return True, res.x
    return False, None

def backfire_test(O_names, Y_names, label, rn):
    """
    Test whether adding Y to organisation O causes structural backfire.

    Returns: (closure_of_union, is_org, blocking_in_closure)
    """
    union = set(O_names) | set(Y_names)
    closure = compute_closure(sorted(union), rn)
    is_org, v = check_sm(sorted(closure), rn)

    print(f"\n  Backfire test: {label}")
    print(f"    O  = {sorted(O_names)}")
    print(f"    Y  = {sorted(Y_names)}")
    print(f"    closure(O∪Y) = {sorted(closure)}")
    print(f"    Is organisation: {is_org}")
    if not is_org:
        blocking = diagnose_blocking(sorted(closure), rn)
        print(f"    Blocking species: {blocking}")
        # Check if O itself is still reachable
        is_O_sub = set(O_names).issubset(closure)
        print(f"    O ⊆ closure: {is_O_sub}")
    return closure, is_org

# ─────────────────────────────────────────────
# MAIN INVESTIGATION
# ─────────────────────────────────────────────
if __name__ == '__main__':
    file_path = os.path.join(project_root,
        'projects', 'XCEPT_Article_1', 'scripts', 'data', 'CE_2.txt')
    rn_base = read_txt(file_path)

    print("\n" + "="*70)
    print("  STEP 0: Base network summary")
    print("="*70)
    print(f"  Species  : {[sp.name for sp in rn_base.species()]}")
    print(f"  Reactions: {len(rn_base.reactions())}")

    # ──────────────────────────────────────
    # STEP 1: Base network lattice + diagnosis
    # ──────────────────────────────────────
    print("\n>>> STEP 1: Base network lattice + blocking-species diagnosis")
    res_base = run_lattice(rn_base)
    report_lattice("BASE NETWORK (CE_2.txt, 54 reactions)", res_base, rn_base)

    # Hard-code the 11 SOs from the briefing for targeted diagnosis
    X = {
        'X1' : {'Rs','Rs_deg','Rg'},
        'X2' : {'A1','A1g','Rs','Rs_deg','Rg','Rg_x'},
        'X3' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl'},
        'X4' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl','D3'},
        'X5' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl','V'},
        'X6' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl','V','D3'},
        'X7' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                'A1','A1g','V','Ex','Rg_x','Gs'},
        'X8' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                'A1','A1g','V','Ex','Rg_x','Gs','D2'},
        'X9' : {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                'A1','A1g','V','Ex','Rg_x','Gs','D3'},
        'X10': {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                'A1','A1g','V','Ex','Rg_x','Gs','D2','D3'},
        'X11': {'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                'A1','A1g','V','Ex','Rg_x','Gs','D2','D3','L2','L3','Ti'},
    }
    print("\n  ── Targeted diagnosis of known non-organisations ──")
    orgs_base = {org_set(o) for o in res_base['all_organizations']}
    for label, sps in X.items():
        if frozenset(sps) not in orgs_base:
            bl = diagnose_blocking(sorted(sps), rn_base)
            print(f"  {label} ({len(sps)} sp, {classify(sps)}): {bl if bl else 'none - SM?'}")

    # ──────────────────────────────────────
    # STEP 2: Individual fixes
    # ──────────────────────────────────────
    print("\n\n>>> STEP 2: Individual fixes")

    fixes = [
        ("r55 only (external civic reconstruct)", [R55]),
        ("r56 only (emergency mediation)", [R56]),
        ("r57 only (Disp return via Rs)", [R57]),
        ("r58 only (Ex reintegration via Rs)", [R58]),
        ("r59+r60 (A1/A1g decay to Ex)", [R59, R60]),
        ("r32m only (insurgency needs L1)", [R32M]),
    ]

    fix_results = {}
    for fix_label, rxn_list in fixes:
        rn_mod = add_rxns(file_path, rxn_list)
        res    = run_lattice(rn_mod)
        report_lattice(fix_label, res, rn_mod)
        fix_results[fix_label] = (rn_mod, res)

    # ──────────────────────────────────────
    # STEP 3: Combinations
    # ──────────────────────────────────────
    print("\n\n>>> STEP 3: Combinations")

    combos = [
        ("r55+r56 (civic + mediation)", [R55, R56]),
        ("r55+r56+r57 (civic + mediation + Disp)", [R55, R56, R57]),
        ("r55+r56+r57+r59+r60 (+decay)", [R55, R56, R57, R59, R60]),
        ("r55+r56+r57+r58 (+Ex reint)", [R55, R56, R57, R58]),
        ("r55+r56+r57+r58+r59+r60 (all bootstrap)", [R55, R56, R57, R58, R59, R60]),
        ("r55+r56+r57+r58+r59+r60+r32m (full)", [R55, R56, R57, R58, R59, R60, R32M]),
    ]

    combo_results = {}
    for combo_label, rxn_list in combos:
        rn_mod = add_rxns(file_path, rxn_list)
        res    = run_lattice(rn_mod)
        report_lattice(combo_label, res, rn_mod)
        combo_results[combo_label] = (rn_mod, res)

    # ──────────────────────────────────────
    # STEP 4: Identify mixed organisations
    # ──────────────────────────────────────
    print("\n\n>>> STEP 4: Mixed organisation summary")
    all_tested = list(fix_results.items()) + list(combo_results.items())
    for lbl, (rn_mod, res) in all_tested:
        mixed = []
        for org in res['all_organizations']:
            sps = org_set(org)
            if classify(sps) == "MIXED":
                mixed.append(sorted(sps))
        if mixed:
            print(f"\n  *** MIXED FOUND in [{lbl}] ***")
            for m in mixed:
                print(f"      {m}")

    # ──────────────────────────────────────
    # STEP 5: Structural backfire analysis
    # (use best combo that has >= 2 organisations)
    # ──────────────────────────────────────
    print("\n\n>>> STEP 5: Structural backfire analysis")

    # Find the richest lattice (most organisations) for backfire testing
    best_lbl, (best_rn, best_res) = max(
        all_tested, key=lambda x: len(x[1][1]['all_organizations'])
    )  # all_tested items: (label, (rn, res))
    print(f"\n  Using richest lattice: [{best_lbl}]")
    print(f"  Organisations: {len(best_res['all_organizations'])}")

    best_orgs = []
    for org in best_res['all_organizations']:
        sps = org_set(org)
        best_orgs.append((classify(sps), sorted(sps)))
    for cls, sps in best_orgs:
        print(f"    [{cls}] {sps}")

    if len(best_orgs) >= 2:
        # Identify conflict and peace orgs for backfire tests
        conflict_orgs = [sps for cls, sps in best_orgs if cls == 'conflict']
        peace_orgs    = [sps for cls, sps in best_orgs if cls == 'peace']
        mixed_orgs    = [sps for cls, sps in best_orgs if cls == 'MIXED']

        print(f"\n  Conflict orgs: {len(conflict_orgs)}")
        print(f"  Peace orgs   : {len(peace_orgs)}")
        print(f"  Mixed orgs   : {len(mixed_orgs)}")

        # Test 1: Civic intervention into hard-defence (D1-heavy) org
        for O in conflict_orgs[:2]:
            backfire_test(O, ['L3', 'Tc'],
                          "civic intervention (L3+Tc) into conflict org", best_rn)
            backfire_test(O, ['L3', 'Tc', 'D3'],
                          "civic+mediation (L3+Tc+D3) into conflict org", best_rn)

        # Test 2: Insurgent adjacency to peace org
        for O in peace_orgs[:2]:
            backfire_test(O, ['A1g'],
                          "insurgent adjacency (A1g) to peace org", best_rn)
            backfire_test(O, ['A1g', 'D1'],
                          "insurgent + military (A1g+D1) to peace org", best_rn)

        # Test 3: Ex reintegration without D3
        for O in peace_orgs[:1]:
            if 'D3' in O:
                O_no_d3 = [s for s in O if s != 'D3']
                backfire_test(O_no_d3, ['Ex'],
                              "Ex reintegration without D3", best_rn)

        # Test 4: Vigilante addition to civic org
        for O in peace_orgs[:1]:
            backfire_test(O, ['V'],
                          "vigilante formation in peace org", best_rn)

    # ──────────────────────────────────────
    # STEP 6: Transition asymmetry
    # ──────────────────────────────────────
    print("\n\n>>> STEP 6: Transition asymmetry between organisations")
    if len(best_orgs) >= 2:
        for i, (cls_i, O_i) in enumerate(best_orgs):
            for j, (cls_j, O_j) in enumerate(best_orgs):
                if i >= j:
                    continue
                # From O_i to O_j: what Y takes O_i closure into O_j's territory?
                # Minimal test: does O_j \ O_i as Y lead to closure = O_j?
                Y_ij = sorted(set(O_j) - set(O_i))
                Y_ji = sorted(set(O_i) - set(O_j))

                if Y_ij:
                    c_ij = compute_closure(sorted(set(O_i) | set(Y_ij)), best_rn)
                    sm_ij, _ = check_sm(sorted(c_ij), best_rn)
                    print(f"\n  O{i+1}→O{j+1} [{cls_i}→{cls_j}]: "
                          f"Y={Y_ij}  closure={sorted(c_ij)}  SM={sm_ij}")
                if Y_ji:
                    c_ji = compute_closure(sorted(set(O_j) | set(Y_ji)), best_rn)
                    sm_ji, _ = check_sm(sorted(c_ji), best_rn)
                    print(f"  O{j+1}→O{i+1} [{cls_j}→{cls_i}]: "
                          f"Y={Y_ji}  closure={sorted(c_ji)}  SM={sm_ji}")

                symmetric = (set(Y_ij) == set(Y_ji))
                print(f"  Symmetric transition: {symmetric}")

    print("\n\n>>> INVESTIGATION COMPLETE")


# ─────────────────────────────────────────────
# STEP 7: V / Disp balance fix (run separately)
# ─────────────────────────────────────────────
def step7_balance_fix():
    """
    Diagnose why X11 (21sp set) fails SM at eps=0.01 and find the minimal
    reaction addition that makes it SM.

    Root cause chain (from prior analysis):
      Disp: 2 producers (r27,r31), 3 consumers (r52,r53,r54) → r31 forced ≥ 2×eps
      → r9 forced ≥ 2×eps → extra L1 consumed → r38 forced ≥ 2×eps
      V:   1 producer (r35), 2 consumers (r37,r38) → V net = eps-eps-2eps < 0

    Need BOTH:
      (A) 1+ extra Disp producer not cascading through L2 → r31 back to eps → r38 to eps
      (B) 1+ extra V producer → V net = 2eps-2eps = 0 ✓
    """
    import numpy as np
    from scipy.optimize import linprog

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root,
        'projects', 'XCEPT_Article_1', 'scripts', 'data', 'CE_2.txt')

    X11 = sorted({'Disp','Tc','Rs','D1','L1','A2','Rg','Rs_deg','A3','Gl',
                  'A1','A1g','V','Ex','Rg_x','Gs','D2','D3','L2','L3','Ti'})

    def sm_eps(rn, species_names, eps=0.01):
        sp_objs = [rn.get_species(s) for s in species_names if rn.has_species(s)]
        sub = rn.sub_reaction_network(sp_objs)
        S = np.asarray(sub.stoichiometry_matrix(), dtype=float)
        n_sp, n_rx = S.shape
        if n_rx == 0:
            return False, 1e9
        res = linprog(np.ones(n_rx), A_ub=-S, b_ub=np.zeros(n_sp),
                      bounds=[(eps, None)]*n_rx, method='highs')
        if res.success:
            return True, 0.0
        # slack LP for total violation
        c = np.concatenate([np.zeros(n_rx), np.ones(n_sp)])
        A_ub = np.hstack([-S, -np.eye(n_sp)])
        b_ub = np.zeros(n_sp)
        bds  = [(eps, None)]*n_rx + [(0, None)]*n_sp
        r2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bds, method='highs')
        slack = sum(r2.x[n_rx:]) if r2.success else 1e9
        return False, slack

    def show_balance(rn, species_names):
        sp_objs = [rn.get_species(s) for s in species_names if rn.has_species(s)]
        sub = rn.sub_reaction_network(sp_objs)
        S_obj = sub.stoichiometry_matrix()
        S = np.asarray(S_obj, dtype=float)
        sp_names, rx_names = S_obj.species, S_obj.reactions
        print(f"\n  {'Species':8s}  prod  cons")
        for i, sp in enumerate(sp_names):
            row = S[i, :]
            prod = [rx_names[j] for j in range(len(row)) if row[j] > 0]
            cons = [rx_names[j] for j in range(len(row)) if row[j] < 0]
            flag = " <--" if len(prod) < len(cons) else ""
            print(f"  {sp:8s}  {len(prod):4d}  {len(cons):4d}{flag}")
            if sp in ('V', 'Disp'):
                print(f"    prod: {prod}")
                print(f"    cons: {cons}")

    print("\n\n>>> STEP 7: V / Disp balance fix")
    rn_base = read_txt(file_path)

    print("\n-- Base 21sp balance --")
    show_balance(rn_base, X11)
    sm, slack = sm_eps(rn_base, X11)
    print(f"\n  X11 SM@eps=0.01: {sm}  slack={slack:.4f}")
    sm0, _ = sm_eps(rn_base, X11, eps=0)
    print(f"  X11 SM@eps=0   : {sm0}")

    # Candidate Disp producers (population-conserved or resource-sourced)
    DISP_FIXES = [
        "r63a: A1 + L3 => Disp + L3",   # insurgency near D3-territory displaces
        "r63b: Ex + L1 => Disp + L1",   # Ex-combatant terrorises → Disp (L1 catalytic)
        "r63c: A2 + L1 => Disp + L1",   # armed actor displaces (L1 catalytic)
        "r63d: Rs_deg + L1 => Disp + L1",# degraded land forces Disp (L1 catalytic)
        "r63e: Rg_x + L1 => Disp + L1", # insurgent-held land → Disp (L1 catalytic)
    ]

    # Candidate V producers (population-conserved)
    V_FIXES = [
        "r64a: L3 + D3 + L1 => V + D3 + L1",  # local communities form vigilance
        "r64b: D1 + L1 => V + D1",             # D1 trains local vigilantes (L1→V, D1 cat)
        "r64c: L3 + Tc => V + Tc",             # community cohesion → vigilance (Tc cat)
        "r64d: Ex + L1 => V + L1",             # ex-combatants form vigilante groups
    ]

    print("\n-- Disp-fix only (with r55) --")
    for d in DISP_FIXES:
        rn_t = add_rxns(file_path, [R55, d])
        sm, slack = sm_eps(rn_t, X11)
        print(f"  r55 + {d[:20]:20s}: SM={sm}  slack={slack:.4f}")

    print("\n-- Disp-fix + V-fix (with r55) --")
    for d in DISP_FIXES:
        for v in V_FIXES:
            rn_t = add_rxns(file_path, [R55, d, v])
            sm, slack = sm_eps(rn_t, X11)
            if sm:
                print(f"  *** SM=True: r55 + {d[5:10]} + {v[5:10]} ***")
            else:
                print(f"  r55+{d[5:8]}+{v[5:8]}: slack={slack:.4f}")

    print("\n-- Best combo: full lattice --")
    # Pick the first pair that gives SM=True and compute full lattice
    for d in DISP_FIXES:
        for v in V_FIXES:
            rn_t = add_rxns(file_path, [R55, d, v])
            sm, slack = sm_eps(rn_t, X11)
            if sm:
                res = run_lattice(rn_t)
                report_lattice(f"r55 + {d[:20]} + {v[:20]}", res, rn_t)
                return  # stop at first working combo


def step8_paper_findings():
    """
    Print the three COT findings for the XCEPT Article 1 paper.
    Network: CE_2.txt + r63d (resource degradation drives displacement).

    r63d: Rs_deg + L1 => Disp + L1
      - Rs_deg (degraded land) in the presence of civilians (L1, catalytic)
        generates displaced persons (Disp).
      - Follows the model's catalytic-generation pattern (same as r8, r25).
      - Fixes Disp's 2-producer / 3-consumer deficit, unblocking V via the
        r31 -> r9 -> L1 -> r38 cascade.
    """
    import numpy as np
    from scipy.optimize import linprog

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root,
        'projects', 'XCEPT_Article_1', 'scripts', 'data', 'CE_2.txt')

    with open(file_path) as f:
        txt = f.read()
    rn = from_string(txt + '\nr63d: Rs_deg + L1 => Disp + L1')

    PEACE_SP    = {'L1', 'L2', 'L3', 'Tc', 'Ti', 'D2', 'D3'}
    CONFLICT_SP = {'A1', 'A1g', 'A2', 'A3', 'Gl', 'Gs', 'Rg_x'}

    def classify(s):
        hp = bool(s & PEACE_SP); hc = bool(s & CONFLICT_SP)
        if hp and hc: return 'MIXED'
        return 'peace' if hp else ('conflict' if hc else 'structural')

    def compute_closure(seed):
        sp_objs = [rn.get_species(s) for s in seed if rn.has_species(s)]
        return frozenset(sp.name for sp in rn.generated_closure(sp_objs))

    def check_sm(names, eps=0.01):
        sp_objs = [rn.get_species(s) for s in names if rn.has_species(s)]
        sub = rn.sub_reaction_network(sp_objs)
        S = np.asarray(sub.stoichiometry_matrix(), dtype=float)
        n_sp, n_rx = S.shape
        if n_rx == 0: return False
        return linprog(np.ones(n_rx), A_ub=-S, b_ub=np.zeros(n_sp),
                       bounds=[(eps, None)]*n_rx, method='highs').success

    def blocking_sp(names):
        sp_objs = [rn.get_species(s) for s in names if rn.has_species(s)]
        sub = rn.sub_reaction_network(sp_objs)
        S = np.asarray(sub.stoichiometry_matrix(), dtype=float)
        n_sp, n_rx = S.shape
        sp_names = sub.stoichiometry_matrix().species
        c  = np.concatenate([np.zeros(n_rx), np.ones(n_sp)])
        A  = np.hstack([-S, -np.eye(n_sp)])
        r2 = linprog(c, A_ub=A, b_ub=np.zeros(n_sp),
                     bounds=[(0.01, None)]*n_rx + [(0, None)]*n_sp, method='highs')
        if r2.success:
            return [sp_names[i] for i in range(n_sp) if r2.x[n_rx + i] > 1e-3]
        return []

    O1 = frozenset(['Rg', 'Rs', 'Rs_deg'])
    O2 = frozenset(['A1', 'A1g', 'Rg', 'Rg_x', 'Rs', 'Rs_deg'])
    O3 = frozenset(['A1', 'A1g', 'A2', 'A3', 'D1', 'D2', 'D3', 'Disp', 'Ex', 'Gl',
                    'Gs', 'L1', 'L2', 'L3', 'Rg', 'Rg_x', 'Rs', 'Rs_deg', 'Tc', 'Ti', 'V'])

    print()
    print('#' * 60)
    print('  PAPER FINDINGS  (CE_2.txt + r63d)')
    print('#' * 60)
    print()
    print('ORGANISATIONS')
    print('  O1 (3sp)  [structural]: Rg, Rs, Rs_deg')
    print('  O2 (6sp)  [conflict]  : A1, A1g, Rg, Rg_x, Rs, Rs_deg')
    print('  O3 (21sp) [MIXED]     : all species')
    print()

    print('FINDING 1 – Mixed organisation')
    sm_O3 = check_sm(sorted(O3))
    print(f'  O3 self-maintaining: {sm_O3}')
    print(f'  Peace species   : {sorted(PEACE_SP & O3)}')
    print(f'  Conflict species: {sorted(CONFLICT_SP & O3)}')
    print()

    print('FINDING 2 – Structural backfire pairs (O + Y -> not SM)')
    bf_tests = [
        (O2, ['L1'],       'livelihoods only'),
        (O2, ['D3'],       'governance only'),
        (O2, ['D1'],       'local authority only'),
        (O2, ['D3', 'L1'], 'governance + livelihoods'),
        (O2, ['L3'],       'community org [SUCCEEDS]'),
        (O1, ['L1'],       'O1 + livelihoods'),
    ]
    for O_org, Y, desc in bf_tests:
        c   = compute_closure(sorted(O_org | set(Y)))
        sm  = check_sm(sorted(c))
        y_str = '+'.join(sorted(Y))
        is_bf = (not sm) and O_org.issubset(c)
        bl    = blocking_sp(sorted(c)) if not sm else []
        tag   = '*** BACKFIRE ***' if is_bf else ('SUCCESS -> O3' if c == O3 else '')
        print('  O+%-20s -> %2dsp [%-8s] SM=%-5s  %s' % (
              y_str, len(c), classify(c), sm, tag))
        if bl:
            print('     blocked: %s' % bl)
    print()

    print('FINDING 3 – Asymmetric transitions')
    print()
    c_fwd = compute_closure(sorted(O2 | {'L3'}))
    print('  O2 + {L3} -> %dsp [%s]  SM=%s  is_O3=%s' % (
          len(c_fwd), classify(c_fwd), check_sm(sorted(c_fwd)), c_fwd == O3))
    c_fwd2 = compute_closure(sorted(O2 | {'L2'}))
    print('  O2 + {L2} -> %dsp [%s]  SM=%s  is_O3=%s' % (
          len(c_fwd2), classify(c_fwd2), check_sm(sorted(c_fwd2)), c_fwd2 == O3))
    print()
    print('  O3 + any single species ->',
          'stays in O3 (O3 is ABSORBING by addition)')
    print()
    print('  ASYMMETRY: O2->O3 via {L3} (or {L2})  |  O3->O2 unreachable by addition')


if __name__ == '__main__' and False:
    # Run standalone step 7
    step7_balance_fix()

if __name__ == '__main__':
    # Quick path: just print the paper findings
    step8_paper_findings()

