# Script: Reaction Network Simulation with Extended Reactions
# Using three kinetic types: cosine, saturated, and mass action

# ========================================
# 1. LIBRARY LOADING AND CONFIGURATION
# ========================================
import os
import sys
import numpy as np

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pyCOT modules
from pyCOT.io.functions import read_txt 
from pyCOT.plot_dynamics import plot_series_ode
from pyCOT.simulations import *
from pyCOT.plot_dynamics import *
from pyCOT.abstractions import abstraction_ordinary


# ========================================
# 2. CUSTOM KINETIC LAWS DEFINITION
# ========================================

def rate_cosine(substrates, concentrations, species_idx, spec_vector, t=None):
    """
    Cosine inflow kinetics: A*(1 + cos(w*t))/2 (always positive, oscillates between 0 and A)
    Used for: R1 (seasonal water inflow)
    Parameters: [A, w] where A is amplitude, w is frequency
    """
    A, w = spec_vector
    
    # Handle time parameter - it should be passed from the modified simulation
    if t is not None:
        current_t = float(t)
    else:
        print("WARNING: No time parameter received, using t=0")
        current_t = 0.0
    
    # Use (1 + cos(wt))/2 to ensure always positive and smooth oscillation
    rate = A * (1 + np.cos(w * current_t)) / 2
    
    # Debug print
    if int(current_t * 100) % 100 == 0:  # Print every 1 time unit
        print(f"Time: {current_t:.2f}, Rate: {rate:.4f}")
    
    return rate

rate_cosine.expression = lambda substrates, reaction: (
    f"A_{reaction} * (1 + cos(w_{reaction} * t)) / 2"
)

def rate_saturated(substrates, concentrations, species_idx, spec_vector):
    """
    Saturated kinetics: (Vmax * [substrate1] * [substrate2]) / (Km + [substrate1])
    Used for: R2, R3, R5, R6, R7, R9 (resource processing with saturation)
    Parameters: [Vmax, Km] where Vmax is max rate, Km is half-saturation
    """
    Vmax, Km = spec_vector
    
    if len(substrates) == 1:
        # Single substrate reaction
        substrate_name = substrates[0][0]
        substrate_conc = concentrations[species_idx[substrate_name]]
        return (Vmax * substrate_conc) / (Km + substrate_conc)
    
    elif len(substrates) >= 2:
        # Two substrate reaction - use first for saturation, multiply by both
        substrate1_name = substrates[0][0]  # Resource being processed
        substrate2_name = substrates[1][0]  # Population/catalyst
        
        substrate1_conc = concentrations[species_idx[substrate1_name]]
        substrate2_conc = concentrations[species_idx[substrate2_name]]
        
        return (Vmax * substrate1_conc * substrate2_conc) / (Km + substrate1_conc)
    
    else:
        return 0

rate_saturated.expression = lambda substrates, reaction: (
    f"(Vmax_{reaction} * [{substrates[0][0]}] * [{substrates[1][0] if len(substrates) > 1 else '1'}]) / (Km_{reaction} + [{substrates[0][0]}])"
    if len(substrates) >= 1 else "0"
)

# ========================================
# 3. CREATE EXTENDED REACTION NETWORK
# ========================================

# Save the extended network to a file
network_file = 'networks/Riverland_model/Scenario1_baseline_only_reactions.txt'
rn = read_txt(network_file)

print("Loaded extended reaction network:")
print("R1: => W_V                    (seasonal water inflow)")
print("R2: W_V + V => V + V_W        (water processing by V)")
print("R3: W_V + V_n => V_n + V_W    (water processing with V needs)")
print("R4: W_V => W_R                (water transfer)")
print("R5: W_R + R => R + R_W        (water processing by R)")
print("R6: W_R + R_n => R_n + R_W    (water processing with R needs)")
print("R7: V_W + V_n => V            (satisfaction of V)")
print("R8: V => V_n                  (V need generation)")
print("R9: R_n + R_W => R            (satisfaction of R)")
print("R10: R => R_n                 (R need generation)")
print("R11: V_W + V_W =>             (V water degradation)")
print("R12: R_W + R_W =>             (R water degradation)")

# ========================================
# 4. SIMULATION PARAMETERS WITH SYMMETRY
# ========================================

# Define initial conditions
# Species order: [W_V, V, V_W, V_n, W_R, R, R_W, R_n]
x0 = [0, 5, 0, 0, 0, 10, 0, 0]  # Start with populations but no water

# Define kinetic types for each reaction
rate_list = [
    'cosine',     # R1: => W_V (seasonal inflow)
    'saturated',  # R2: W_V + V => V + V_W 
    'saturated',  # R3: W_V + V_n => V_n + V_W
    'mak',        # R4: W_V => W_R (mass action)
    'saturated',  # R5: W_R + R => R + R_W
    'saturated',  # R6: W_R + R_n => R_n + R_W
    'saturated',  # R7: V_W + V_n => V
    'mak',        # R8: V => V_n (mass action)
    'saturated',  # R9: R_n + R_W => R
    'mak',        # R10: R => R_n (mass action)
    'mak',        # R11: V_W + V_W => (mass action degradation)
    'mak',        # R12: R_W + R_W => (mass action degradation)
    'mak'         # R13: 2W_R => (mass action degradation)
]

# Define parameters with SYMMETRY between R and V reactions
# For testing oscillations, set most rates to zero except water inflow
water_processing_params = [2, 0.5]     # [Vmax, Km] for R2, R3, R5, R6 - SET TO ZERO FOR TESTING
satisfaction_params = [0.4, 1.0]         # [Vmax, Km] for R7, R9 - SET TO ZERO FOR TESTING
need_generation_rate = [0.2]             # [k] for R8, R10 - SET TO ZERO FOR TESTING
water_degradation_rate = [0.4]          # [k] for R11, R12 - SET TO ZERO FOR TESTING

spec_vector = [[20, 1.0],                   # R1: [A, w] - amplitude=5.0, frequency=1.0 (clear oscillation)
water_processing_params,       # R2: [Vmax, Km] - water processing by V
water_processing_params,        # R3: [Vmax, Km] - need-based water processing by V
[0.5],                        # R4: [k] - water transfer rate - SET TO ZERO FOR TESTING
water_processing_params,       # R5: [Vmax, Km] - water processing by R (SAME as R2)
water_processing_params,        # R6: [Vmax, Km] - need-based water processing by R (SAME as R3)
satisfaction_params,           # R7: [Vmax, Km] - satisfaction of V
need_generation_rate,          # R8: [k] - need generation by V
satisfaction_params,           # R9: [Vmax, Km] - satisfaction of R (SAME as R7)
need_generation_rate,          # R10: [k] - need generation by R (SAME as R8)
water_degradation_rate,        # R11: [k] - V water degradation
water_degradation_rate,        # R12: [k] - R water degradation (SAME as R11)
water_degradation_rate         # R13: [k] - 2W_R water degradation (SAME as R12)
]

# Dictionary of additional kinetic laws
additional_laws = {
    'cosine': rate_cosine,
    'saturated': rate_saturated
}

# ========================================
# 5. RUN SIMULATION
# ========================================

# ========================================
# 5. RUN SIMULATION WITH TIME-DEPENDENT RATE FIX
# ========================================

# We need to use a custom simulation function since pyCOT doesn't pass time to rate functions
def custom_simulation_with_time(rn, rate_list, spec_vector, x0, t_span, n_steps, additional_laws):
    """Custom simulation that properly handles time-dependent kinetics"""
    
    species = [specie.name for specie in rn.species()]
    reactions = [reaction.name() for reaction in rn.reactions()]
    species_idx = {s: i for i, s in enumerate(species)}
    
    # Build reaction dictionary
    rn_dict = {}
    reactants_vectors = rn.reactants_matrix().T
    products_vectors = rn.products_matrix().T
    for i in range(len(reactions)):
        reactants = [(species[j], reactants_vectors[i][j]) for j in range(len(species)) if reactants_vectors[i][j] > 0]
        products = [(species[j], products_vectors[i][j]) for j in range(len(species)) if products_vectors[i][j] > 0]
        rn_dict[reactions[i]] = (reactants, products)
    
    # Rate laws dictionary
    rate_laws = {
        'mak': lambda reactants, concentrations, species_idx, spec_vector, t=None: 
               spec_vector[0] * np.prod([concentrations[species_idx[sp]] ** coef for sp, coef in reactants]),
        'cosine': rate_cosine,
        'saturated': rate_saturated
    }
    
    # Add additional laws
    if additional_laws:
        rate_laws.update(additional_laws)
    
    # ODE function with proper time handling
    def ode_system(t, x):
        x = np.maximum(x, 0)  # Ensure non-negative concentrations
        dxdt = np.zeros_like(x)
        
        for i, reaction in enumerate(reactions):
            kinetic = rate_list[i]
            reactants = rn_dict[reaction][0]
            products = rn_dict[reaction][1]
            params = spec_vector[i]
            
            # Call rate function with time parameter for time-dependent kinetics
            if kinetic == 'cosine':
                rate_value = rate_laws[kinetic](reactants, x, species_idx, params, t=t)
            else:
                rate_value = rate_laws[kinetic](reactants, x, species_idx, params)
            
            # Apply stoichiometry
            for sp, coef in reactants:
                dxdt[species_idx[sp]] -= coef * rate_value
            for sp, coef in products:
                dxdt[species_idx[sp]] += coef * rate_value
        
        return dxdt
    
    # Solve ODE
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(ode_system, t_span, x0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
    
    # Create time series DataFrame
    time_series_data = {"Time": sol.t}
    for i, sp in enumerate(species):
        time_series_data[sp] = sol.y[i]
    time_series_df = pd.DataFrame(time_series_data)
    
    # Create flux vector DataFrame (simplified)
    flux_data = {"Time": sol.t}
    for i in range(len(reactions)):
        flux_data[f"Flux_r{i+1}"] = np.zeros_like(sol.t)  # Placeholder
    flux_vector_df = pd.DataFrame(flux_data)
    
    return time_series_df, flux_vector_df

# Time span and steps - optimized to see oscillations clearly
t_span = (0, 200)   # Short time span to see oscillations
n_steps = 1000     # High resolution

# Run custom simulation with time-dependent rate handling
time_series, flux_vector = custom_simulation_with_time(
    rn, 
    rate_list, 
    spec_vector, 
    x0, 
    t_span, 
    n_steps,
    additional_laws
)

# ========================================
# 6. DISPLAY RESULTS
# ========================================

print("\nSimulation completed successfully!")
print("\nTime Series Shape:", time_series.shape)
print("Species order: W_V, V, V_W, V_n, W_R, R, R_W, R_n")

# Plot the results
plot_series_ode(time_series, filename="extended_network_results.png")

print("\nFinal concentrations:")
final_concentrations = time_series.iloc[-1]
species_names = ['W_V', 'V', 'V_W', 'V_n', 'W_R', 'R', 'R_W', 'R_n']
for i, species in enumerate(species_names):
    print(f"{species}: {final_concentrations.iloc[i]:.4f}")

# Check oscillation in W_V (first 100 time points)
print(f"\nW_V oscillation check (first 10 values): {time_series.iloc[:10, 0].values}")


abstract_time_series = abstraction_ordinary(time_series, threshold=0.05)

plot_abstraction_size(abstract_time_series)

plot_abstraction_sets(abstract_time_series)

plot_abstraction_graph_movie_html(abstract_time_series, filename="abstraction_graph_movie_html.html", interval=400, title="Abstraction Graph - Time")

print("\nFinal concentrations:")
final_concentrations = time_series.iloc[-1]
species_names = ['W_V']
for i, species in enumerate(species_names):
    print(f"{species}: {final_concentrations.iloc[i]:.4f}")


# Check oscillation in W_V (first 100 time points)
print(f"\nW_V oscillation check (first 10 values): {time_series.iloc[:3].values}")



