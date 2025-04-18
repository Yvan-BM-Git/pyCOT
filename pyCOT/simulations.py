import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from matplotlib.widgets import Slider
from scipy.optimize import linprog     

###################################################################################
# Function to calculate the reaction rate 
def reaction_rate_mak(reaction_name, species_concentrations, RN_dict, k):
    """
    Calculate the reaction rate for a given reaction.
    """
    if k is None:
        raise ValueError("The speed constants ‘k’ are not correctly defined.")
    
    # print("Constantes de velocidad:", k)  # Verifica el valor de k
    reactants_coeff, _ = RN_dict[reaction_name]
    
    rate = k[reaction_name]
    
    # Multiply the concentrations of reactants raised to their stoichiometric coefficients
    for reactant, coef in reactants_coeff:
        rate *= species_concentrations.get(reactant, 0) ** coef
    
    return rate


###################################################################################
# Generate ODEs for the reaction network
# def generate_odes_mak(species_concentrations, RN_dict, k):
#     """
#     Generate the ODE system for the reaction network.

#     Parameters:
#     species_concentrations (dict): Mapping of species to indices.
#     RN_dict (dict): Reaction network dictionary mapping reactions to stoichiometry.
#     k (dict): Dictionary of reaction rate constants.

#     Returns:
#     function: A function representing the ODE system.
#     """
#     def odes(t, y):
#         # Initialize derivatives for all species
#         dSdt = {species: 0 for species in species_concentrations.keys()}
        
#         # Map current species concentrations
#         current_concentrations = dict(zip(species_concentrations.keys(), y))
        
#         # Compute derivatives for each species
#         for species in current_concentrations:
#             for reaction_name, (reactants, products) in RN_dict.items():
#                 # Reactants contribution
#                 for reactant, coef in reactants:
#                     if reactant == species:
#                         dSdt[species] -= coef * reaction_rate_mak(reaction_name, current_concentrations, RN_dict, k)
#                 # Products contribution
#                 for product, coef in products:
#                     if product == species:
#                         dSdt[species] += coef * reaction_rate_mak(reaction_name, current_concentrations, RN_dict, k)

#         # Ensure all species are accounted for in dSdt
#         for key in species_concentrations.keys():
#             if key not in dSdt:
#                 raise KeyError(f"Unexpected key '{key}' not found in dSdt.")
        
#         return np.array([dSdt[species] for species in species_concentrations.keys()])
    
#     return odes

def generate_odes_mak(species_concentrations, RN_dict, k):
    """
    Genera el sistema de ecuaciones diferenciales para la red de reacciones.
    """
    import numpy as np
    
    printed = False  # Flag para controlar la impresión de ecuaciones

    def odes(t, y, printed_once=[False]):  # Se usa una lista mutable como "flag" persistente
        nonlocal printed
        
        # Inicializar derivadas de todas las especies
        dSdt = {species: 0 for species in species_concentrations.keys()}
        
        # Mapear concentraciones actuales
        current_concentrations = dict(zip(species_concentrations.keys(), y))
        
        # Diccionario para almacenar las ecuaciones diferenciales en texto
        equation_dict = {species: [] for species in current_concentrations}
        
        # Calcular las derivadas de cada especie
        for reaction_name, (reactants, products) in RN_dict.items():
            rate = reaction_rate_mak(reaction_name, current_concentrations, RN_dict, k)
            
            # Contribución de los reactantes
            for reactant, coef in reactants:
                dSdt[reactant] -= coef * rate
                equation_dict[reactant].append(f"- {coef} * k_{reaction_name} * " + " * ".join([f"[{r}]" for r, c in reactants]))

            # Contribución de los productos
            for product, coef in products:
                dSdt[product] += coef * rate
                equation_dict[product].append(f"+ {coef} * k_{reaction_name} * " + " * ".join([f"[{r}]" for r, c in reactants]))

        # Imprimir ecuaciones solo en la primera llamada
        if not printed_once[0]:
            print("\nDifferential equations of the system:")
            for species, terms in equation_dict.items():
                unique_terms = list(set(terms))  # Convertir a set y luego a lista para eliminar duplicados
                equation_str = f"d[{species}]/dt = " + " ".join(sorted(unique_terms))  # Ordenar para consistencia
                print(equation_str)
            printed_once[0] = True  # Cambiar el flag para evitar imprimir nuevamente
        
        return np.array([dSdt[species] for species in species_concentrations.keys()])
    
    return odes







###################################################################################
# Solve the ODE system 
def simulate_ode_mak(RN, x0 = None, t_span = None, n_steps = None, k0 = None):
    """
    Solve the ODE system for the given reaction network and return the results as a time series.

    This function constructs the reaction network, generates the ODE system based on the stoichiometry 
    of the reactions, and solves the system using an ODE solver. It returns the time series of species 
    concentrations over the specified time span.

    Parameters:
    RN (object): Reaction network object containing species and reaction details.
    x0 (list, optional): Initial concentrations of species. If not provided, they will be generated automatically.
    t_span (tuple, optional): Time span for the simulation (start, end). Defaults to (0, 100).
    n_steps (int, optional): Number of time steps for the simulation. Defaults to 100.
    k0 (dict, optional): Dictionary of reaction rate constants. If not provided, they will be generated automatically.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the time series of species concentrations over time.
    """
    # Build the reaction network dictionary
    RN_dict = {}
    for i in range(len(RN.RnStr)):
        reactants_coeff = []
        products_coeff = []
        for j in range(len(RN.SpStr)):
            coef_r = RN.RnMsupp[i][j]
            if coef_r > 0:
                reactants_coeff.append((RN.SpStr[j], coef_r))
            coef_p = RN.RnMprod[i][j]
            if coef_p > 0:
                products_coeff.append((RN.SpStr[j], coef_p))
        RN_dict[RN.RnStr[i]] = (reactants_coeff, products_coeff)

    # Map species to indices
    species_concentrations = {species: idx for idx, species in enumerate(RN.SpStr)}

    # Define the initial state vector
    if x0 is None:
        x0 = generate_random_vector(len(RN.SpStr), seed=3).round(1)

    # Number of steps
    if n_steps is None:
        n_steps = 100
        
    if t_span is None:    
        t_span = [0, 100]

    # Automatically generate rate constants if not provided
    if k0 is None: 
        k0 = generate_random_vector(len(RN.RnStr), seed=3).round(2) 
        print("Constantes de velocidad:", k0)  # Verifica el valor de k
        k = {RN.RnStr[i]: float(k0[i]) for i in range(len(RN.RnStr))}        
    # Asegúrate de que `k` siempre esté definido
    k = {RN.RnStr[i]: float(k0[i]) for i in range(len(RN.RnStr))}

    # Generate ODE system
    odes = generate_odes_mak(species_concentrations, RN_dict, k)

    # Solve the ODE system
    sol = solve_ivp(odes, t_span, x0, t_eval=np.linspace(t_span[0],t_span[1], n_steps))

    # Extract the vector of times and the solutions
    data_concentrations = {"Time": sol.t}
    for i, species in enumerate(RN.SpStr):     
    # for i, species in enumerate(sorted(RN.SpStr)):
        data_concentrations[species] = sol.y[i]

    time_series = pd.DataFrame(data_concentrations) 
    time_series = time_series[["Time"] + sorted(time_series.columns.difference(["Time"]))]

    # Calculate flux vector at each time step
    flux_vector = []
    for t_idx in range(len(sol.t)):
        concentrations = sol.y[:, t_idx]
        flux = []
        for rxn, (reactants, products) in RN_dict.items():
            rate = k[rxn]
            for species, coeff in reactants:
                rate *= concentrations[species_concentrations[species]] ** coeff
            flux.append(rate)
        flux_vector.append(flux)

    # Calculate flux vector at each time step
    flux_data = {"Time": sol.t}
    for i, rxn in enumerate(RN.RnStr):
        flux_data[f"Flux_{i}"] = []

    for t_idx in range(len(sol.t)):
        concentrations = sol.y[:, t_idx]
        for i, rxn in enumerate(RN.RnStr):
            rate = k[rxn]
            reactants = RN_dict[rxn][0]
            for species, coeff in reactants:
                rate *= concentrations[species_concentrations[species]] ** coeff
            flux_data[f"Flux_{i}"].append(rate)

    flux_vector = pd.DataFrame(flux_data)
    
    return time_series, flux_vector   
###################################################################################








###################################################################################
# Function to generate the state vector with random positive values 
def generate_random_vector(n_species, seed=None):
    """
    Generates a state vector with random values between 0 and 1 for each species.
    The same seed will always produce the same vector.

    Parameters:
    n_species (int): Number of species in the system.
    seed (int or None): Seed for random number generator. If None, the seed is not set.

    Returns:
    np.ndarray: A vector of size n_species with values between 0 and 1.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    return np.random.rand(n_species)  # Generates a vector of size n_species with values between 0 and 1

###################################################################################
# Function to create the stoichiometric matrix 
def universal_stoichiometric_matrix(RN):
    """
    Creates the stoichiometric matrix based on the reaction network (RN).
    
    Parameters:
    RN (object): The reaction network object that contains species and reaction details.
    
    Returns:
    np.ndarray: The stoichiometric matrix representing the reactions, with integer values.
    """
    # Initialize the stoichiometric matrix with zeros
    matrix = np.zeros((len(RN.SpStr), len(RN.RnStr)))  # #species x #reactions

    # Iterate over reactions
    for i in range(len(RN.RnStr)):  # For each reaction 
        for j in range(len(RN.SpStr)):  # Same for products
            coef_p = RN.RnMprod[i][j]  # Coefficient of the product in reaction i for species j
            coef_r = RN.RnMsupp[i][j]  # Coefficient of the reactant in reaction i for species j
            matrix[j, i] = coef_p - coef_r  # Product coefficients are positive (transposition here)  

    return matrix  # Returns the stoichiometric matrix with integer values


###################################################################################
# Function to update the state vector at each time step 
def simulate_discrete_random(RN, S, x, n_iter=10):
    """
    Iterates the state vector over a number of iterations, updating the state 
    based on the stoichiometric matrix and random values, and stores the flux history.

    Parameters:
    RN (object): The reaction network object.
    S (np.ndarray): The stoichiometric matrix.
    x (np.ndarray): The initial state vector. 
    n_iter (int): Number of iterations to run.
    
    Returns:
    df_state: DataFrame with the time and updated species concentrations.
    df_flux: DataFrame with the time and flux vectors v(x,k) for each iteration.
    """
    history = [x.copy()]  # Store the initial state in the history
    flux_history = []  # Store the flux vectors (v)
    
    for i in range(n_iter):  # Iterate for n_iter steps
        v = generate_random_vector(S.shape[1])  # Generate a new random state vector for the reaction network
        x = x + S @ v  # Update the state vector using the stoichiometric matrix and random values
        x = np.maximum(x, 0)  # Ensure all values are positive (no negative concentrations)
        
        history.append(x.copy())  # Store the updated state in the history
        flux_history.append(v.copy())  # Store the flux vector
    
    # Convert the history into a numpy array
    history = np.array(history)
    
    # Create a time vector t
    t = np.arange(n_iter)  # Time steps based on the history shape
    
    # Concatenate the time vector with the history of states
    history = np.column_stack((t, history[:n_iter]))  # Combine time with the state history
    
    # Create DataFrame for species concentrations
    columns = ['Time'] + RN.SpStr  # The first column is 'Time', followed by species names
    df_state = pd.DataFrame(history, columns=columns)  # Create the DataFrame with time and species concentrations
    
    # Create DataFrame for flux history
    flux_columns = ['Time'] + [f'Flux_{i}' for i in range(S.shape[1])]  # Column names for fluxes
    flux_history = np.column_stack((np.arange(0, n_iter), flux_history))  # Combine time with flux history
    df_flux = pd.DataFrame(flux_history, columns=flux_columns)  # Create DataFrame for fluxes

    return df_state, df_flux # Return the state and flux DataFrames

###########################################################################################
# Function that calculates the Michaelis-Menten reaction rate
def rate_mmk(species_concentrations, S, Vmax, Km):
    """
    Calculates the Michaelis-Menten reaction rate for a given substrate.

    Parameters:
    species_concentrations (dict): Current concentrations of the species.
        The dictionary keys are the species names, and the values are the corresponding concentrations.
    S (str): The name of the substrate in the reaction network, as defined in the `species_concentrations` dictionary.
    Vmax (float): The maximum reaction rate, in the corresponding units (e.g., M/s).
    Km (float): The Michaelis constant, representing the substrate concentration at which the reaction rate is half of Vmax (in the same units as S).

    Returns:
    float: The reaction rate calculated using the Michaelis-Menten equation, in the same units as Vmax.
    """
    # Get the current concentration of the substrate
    S_conc = species_concentrations.get(S, 0)
    
    # Apply the Michaelis-Menten equation
    rate = (Vmax * S_conc) / (Km + S_conc)
    
    return rate

def reaction_rate_mmk(RN, reaction_name, species_concentrations, k, Vmax_dict, Km_dict):
    """
    Calculates the reaction rate for a given reaction, considering Michaelis-Menten kinetics 
    when applicable and general kinetics in other cases.

    Parameters:
    RN (dict): Dictionary defining the reaction network. The keys are reaction names,
               and the values are tuples of stoichiometric coefficients for reactants and products, for example:
               {'reaction1': ([(reactant1, coef1), (reactant2, coef2)], [(product1, coef1), ...])}.
    reaction_name (str): The name of the reaction for which the reaction rate will be calculated.
    species_concentrations (dict): Dictionary with the current concentrations of species.
                                   The keys are species names, and the values are the corresponding concentrations.
    k (dict): Dictionary with reaction rate constants for general reactions.
              The keys are reaction names, and the values are the corresponding constants.
    Vmax_dict (dict): Dictionary with maximum velocities for reactions following Michaelis-Menten kinetics.
                      The keys are the names of the substrates, and the values are the Vmax values.
    Km_dict (dict): Dictionary with Michaelis constants for reactions following Michaelis-Menten kinetics.
                    The keys are the names of the substrates, and the values are the Km values.

    Returns:
    float: The calculated reaction rate for the specified reaction. It may include terms based on 
           Michaelis-Menten kinetics or general kinetics, depending on the reaction and its definition in the parameters.

    """
    # Get reactants and products for the specific reaction
    reactants_coeff, products_coeff = RN[reaction_name]
    rate = k.get(reaction_name, 1)  # Use rate constant (default k=1 if not defined)

    # Check if the reaction follows Michaelis-Menten kinetics
    for reactant, coef in reactants_coeff: # For each reactant in the reaction
        if reactant in Vmax_dict:  # Check if the reactant is a substrate in MM kinetics
            rate = rate_mmk(species_concentrations, reactant, Vmax_dict[reactant], Km_dict[reactant])
            break  # Exit loop if Michaelis-Menten is used
        else:
            # Apply general kinetics for the reactant if not Michaelis-Menten
            rate *= species_concentrations.get(reactant, 0) ** coef
    
    return rate

# Function that generates the ODE system for the reaction network
# def generate_odes_mmk(RN, species_concentrations, k, Vmax_dict, Km_dict):
#     """
#     Generates the system of ordinary differential equations (ODEs) for a reaction network, 
#     considering Michaelis-Menten kinetics and general kinetics.

#     Parameters:
#     RN (dict): Dictionary defining the reaction network. The keys are reaction names,
#                and the values are tuples of lists of reactants and products. Each list contains tuples 
#                in the form (species_name, stoichiometric_coefficient), for example:
#                {'reaction1': ([(reactant1, coef1), (reactant2, coef2)], [(product1, coef1), ...])}.
#     species_concentrations (dict): Dictionary mapping species names to their indices in the concentration vector.
#                                     For example, {'A': 0, 'B': 1, 'C': 2}.
#     k (dict): Dictionary with reaction rate constants for general reactions. 
#               The keys are reaction names, and the values are the corresponding constants.
#     Vmax_dict (dict): Dictionary with maximum velocities for reactions following Michaelis-Menten kinetics.
#                       The keys are the names of the substrates, and the values are the Vmax values.
#     Km_dict (dict): Dictionary with Michaelis constants for reactions following Michaelis-Menten kinetics.
#                     The keys are the names of the substrates, and the values are the Km values.

#     Returns:
#     function: A function that takes time `t` (float) and the concentration vector `y` (array-like) as input,
#               and returns an array of derivatives for all species in the system.
#     """
#     def odes(t, y):
#         """
#         System of ordinary differential equations representing the reaction network dynamics.

#         Parameters:
#         t (float): Current time (not explicitly used but required by ODE solvers).
#         y (array-like): Vector of current species concentrations, in the same order as in species_concentrations.

#         Returns:
#         np.ndarray: Array of concentration derivatives for all species.
#         """
#         # Initialize derivatives for all species
#         dSdt = {species: 0 for species in species_concentrations.keys()}
        
#         # Map current species concentrations
#         current_concentrations = dict(zip(species_concentrations.keys(), y))
        
#         # Compute derivatives for each species
#         for species in current_concentrations:
#             for reaction_name, (reactants, products) in RN.items():
#                 # Contribution from reactants
#                 for reactant, coef in reactants:
#                     if reactant == species:
#                         dSdt[species] -= coef * reaction_rate_mmk(RN, reaction_name, current_concentrations, k, Vmax_dict, Km_dict)
#                 # Contribution from products
#                 for product, coef in products:
#                     if product == species:
#                         dSdt[species] += coef * reaction_rate_mmk(RN, reaction_name, current_concentrations, k, Vmax_dict, Km_dict)
        
#         # Convert derivatives to the format required by the integrator
#         return np.array([dSdt[species] for species in species_concentrations.keys()])
    
#     return odes

def generate_odes_mmk(RN, species_concentrations, k, Vmax_dict, Km_dict):
    """
    Genera el sistema de ecuaciones diferenciales para la red de reacciones,
    considerando cinética de Michaelis-Menten y cinética general.
    """
    printed = False  # Flag para controlar la impresión de ecuaciones

    def odes(t, y, printed_once=[False]):
        nonlocal printed
        
        # Inicializar derivadas de todas las especies
        dSdt = {species: 0 for species in species_concentrations.keys()}
        
        # Mapear concentraciones actuales
        current_concentrations = dict(zip(species_concentrations.keys(), y))
        
        # Diccionario para almacenar las ecuaciones diferenciales en texto
        equation_dict = {species: [] for species in current_concentrations}
        
        # Calcular las derivadas de cada especie
        for reaction_name, (reactants, products) in RN.items():
            rate = reaction_rate_mmk(RN, reaction_name, current_concentrations, k, Vmax_dict, Km_dict)
            
            # Contribución de los reactantes
            for reactant, coef in reactants:
                dSdt[reactant] -= coef * rate 
                equation_dict[reactant].append(f"- {coef} * r_{reaction_name}")
                # (f"- {coef} * k_{reaction_name} * " + " * ".join([f"[{r}]" for r, c in reactants])) # 
                
            # # Contribución de los productos
            # for product, coef in products:
            #     dSdt[product] += coef * rate
            #     equation_dict[product].append(f"+ {coef} * r_{reaction_name}")
            #     # (f"+ {coef} * k_{reaction_name} * " + " * ".join([f"[{r}]" for r, c in reactants])) # 
        
        # Imprimir ecuaciones solo en la primera llamada
        if not printed_once[0]:
            print("\nDifferential equations of the system:")
            for species, terms in equation_dict.items():
                unique_terms = list(set(terms))  # Eliminar duplicados
                equation_str = f"d[{species}]/dt = " + " ".join(sorted(unique_terms))
                print(equation_str)
            printed_once[0] = True  # Cambiar el flag para evitar imprimir nuevamente
        
        return np.array([dSdt[species] for species in species_concentrations.keys()])
    
    return odes






# Function that solves the ODE system with Michaelis-Menten kinetics
def simulate_ode_mmk(RN, x0=None, t_span=None, n_steps=None, k0=None, Vmax_dict=None, Km_dict=None):
    """
    Simulates the dynamics of a chemical reaction network using a system of ordinary differential equations (ODEs). 
    Integrates Michaelis-Menten kinetic models and specific reaction parameters to calculate concentrations and fluxes.

    Parameters:
    ----------
    RN : object
        Object describing a reaction network. It must include:
        - `SpStr`: List of chemical species names.
        - `RnStr`: List of reaction names.
        - `RnMsupp`: Stoichiometric matrix of reactants.
        - `RnMprod`: Stoichiometric matrix of products.

    x0 : list, optional
        Vector of initial concentrations of chemical species (length equal to `SpStr`).
        If not provided, it is generated randomly.

    t_span : tuple, optional
        Time interval for the simulation, defined as `(start, end)`. 
        Default value: `(0, 100)`.

    n_steps : int, optional
        Number of time steps to evaluate the solution within `t_span`.
        Default value: `100`.

    k0 : dict, optional
        Dictionary with reaction rate constants (one per reaction). 
        If not provided, random values are generated.

    Vmax_dict : dict, optional
        Dictionary with maximum velocity values (Vmax) for each species.
        If not provided, random values are generated.

    Km_dict : dict, optional
        Dictionary with Michaelis constant values (Km) for each species.
        If not provided, random values are generated.

    Returns:
    -------
    time_series : pandas.DataFrame
        DataFrame with species concentrations over time.
        - Columns: `Time` and the name of each species in `RN.SpStr`.

    flux_vector : pandas.DataFrame
        DataFrame with reaction fluxes over time.
        - Columns: `Time` and `Flux_i` (where `i` is the reaction index).

    Notes:
    ------
    - Generates the differential equations using the Michaelis-Menten model and the provided kinetic constants.
    - The fluxes of each reaction are computed by evaluating the rates at each time step.
    """
    # Build the reaction network dictionary
    RN_dict = {}
    for i in range(len(RN.RnStr)):
        reactants_coeff = []
        products_coeff = []
        for j in range(len(RN.SpStr)):
            coef_r = RN.RnMsupp[i][j]
            if coef_r > 0:
                reactants_coeff.append((RN.SpStr[j], coef_r))
            coef_p = RN.RnMprod[i][j]
            if coef_p > 0:
                products_coeff.append((RN.SpStr[j], coef_p))
        RN_dict[RN.RnStr[i]] = (reactants_coeff, products_coeff)
    
    # Map species to indices
    species_concentrations = {species: idx for idx, species in enumerate(RN.SpStr)}

    # Define the initial state vector
    if x0 is None:
        x0 = generate_random_vector(len(RN.SpStr), seed=3).round(1)*20

    # Number of steps
    if n_steps is None:
        n_steps = 100
        
    if t_span is None:    
        t_span = (0, 100)

    # Automatically generate rate constants if not provided
    if k0 is None: 
        k0 = generate_random_vector(len(RN.RnStr), seed=3).round(2) 
        k = {RN.RnStr[i]: float(k0[i]) for i in range(len(RN.RnStr))}   

    else:
        # Handling already existing k 
        k = {species: float(k0[species]) if species in k0 else 0.5 for species in RN.RnStr}

    # Automatically generate Vmax values if not provided 
    if Vmax_dict is None:
        Vmax = generate_random_vector(len(RN.SpStr), seed=5).round(1)*300
        Vmax_dict = {RN.SpStr[i]: float(Vmax[i]) for i in range(len(RN.SpStr))}
    else: 
        Vmax_dict = {species: float(Vmax_dict[species]) if species in Vmax_dict else 1.0 for species in RN.SpStr}

          
    # Automatically generate Km values if not provided 
    if Km_dict is None:
        Km = generate_random_vector(len(RN.SpStr), seed=3).round(2)*5000
        Km_dict = {RN.SpStr[i]: float(Km[i]) for i in range(len(RN.SpStr))}
    else:
        # Handling already existing Km_dict 
        Km_dict = {species: float(Km_dict[species]) if species in Km_dict else 0.5 for species in RN.SpStr}


    # Validate t_span
    if not isinstance(t_span, (list, tuple)) or len(t_span) != 2:
        raise ValueError("t_span must be a list or tuple with two elements: [start, end].")

    # Validate x0
    if len(x0) != len(RN.SpStr):
        raise ValueError("x0 must have the same length as the number of species in RN.SpStr.")

    # Validate species_concentrations
    if len(species_concentrations) != len(RN.SpStr):
        raise ValueError("The mapping of species in species_concentrations is inconsistent with RN.SpStr.")

        
    ## Generate the ODE system using the Michaelis-Menten model
    odes = generate_odes_mmk(RN_dict, species_concentrations, k, Vmax_dict, Km_dict)
    
    # Solve the ODE system
    sol = solve_ivp(odes, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], n_steps))

    # Extract the time vector and the solutions
    data_concentrations = {"Time": sol.t}
    for i, species in enumerate(RN.SpStr):
        data_concentrations[species] = sol.y[i]

    time_series = pd.DataFrame(data_concentrations) 
    time_series = time_series[["Time"] + sorted(time_series.columns.difference(["Time"]))]



    ###########################################################################################
    # Calculate flux vector at each time step
    flux_vector = []
    for t_idx in range(len(sol.t)):
        concentrations = sol.y[:, t_idx]
        flux = []
        for rxn, (reactants, products) in RN_dict.items():
            rate = k[rxn]
            for species, coeff in reactants:
                rate *= concentrations[species_concentrations[species]] ** coeff
            flux.append(rate)
        flux_vector.append(flux)

    # Calculate flux vector at each time step
    flux_data = {"Time": sol.t}
    for i, rxn in enumerate(RN.RnStr):
        flux_data[f"Flux_{i}"] = []

    for t_idx in range(len(sol.t)):
        concentrations = sol.y[:, t_idx]
        for i, rxn in enumerate(RN.RnStr):
            rate = k[rxn]
            reactants = RN_dict[rxn][0]
            for species, coeff in reactants:
                rate *= concentrations[species_concentrations[species]] ** coeff
            flux_data[f"Flux_{i}"].append(rate)

    flux_vector = pd.DataFrame(flux_data)
    
    return time_series, flux_vector      

##############################################################################################
# Function that simulates the reaction network using the specified kinetics (MAK or MMK)
def simulations(RN, rate='mak', spec_vector=None, x0=None, n_steps=300, t_span=(0, 100), k0=None, Vmax_dict=None, Km_dict=None):
    # Check if the rate is valid
    if isinstance(rate, str):
        rate = [rate] * len(RN.RnStr)
    elif len(rate) != len(RN.RnStr):
        raise ValueError("The rate list must be the same length as the number of reactions.")

    RN_dict = {}
    for i in range(len(RN.RnStr)):
        reactants_coeff = []
        products_coeff = []
        for j in range(len(RN.SpStr)):
            coef_r = RN.RnMsupp[i][j]
            if coef_r > 0:
                reactants_coeff.append((RN.SpStr[j], coef_r))
            coef_p = RN.RnMprod[i][j]
            if coef_p > 0:
                products_coeff.append((RN.SpStr[j], coef_p))
        RN_dict[RN.RnStr[i]] = (reactants_coeff, products_coeff)

    species_concentrations = {species: idx for idx, species in enumerate(RN.SpStr)}

    # Define the parameters for the simulation
    if x0 is None:
        x0 = generate_random_vector(len(RN.SpStr), seed=3).round(1) * 20

    # =========================
    # Load parameters from spec_vector
    # =========================
    if spec_vector is not None:
        k = {}
        Vmax_dict = {}
        Km_dict = {}
        for i, kind in enumerate(rate):
            rxn = RN.RnStr[i]
            reactants = RN_dict[rxn][0]
            if kind == 'mak':
                k[rxn] = spec_vector[i][0]
            elif kind == 'mmk' and reactants:
                sustrato = reactants[0][0]
                if sustrato not in Vmax_dict:
                    Vmax_dict[sustrato] = spec_vector[i][0]
                    Km_dict[sustrato] = spec_vector[i][1]
    else:
        # =========================
        # Generate random parameters if no spec_vector was passed
        # =========================
        if k0 is None:
            k0 = generate_random_vector(len(RN.RnStr), seed=3).round(2)
            k = {RN.RnStr[i]: float(k0[i]) for i in range(len(RN.RnStr))}
        else:
            k = {species: float(k0[species]) if species in k0 else 0.5 for species in RN.RnStr}

        if Vmax_dict is None:
            Vmax = generate_random_vector(len(RN.SpStr), seed=5).round(1) * 300
            Vmax_dict = {RN.SpStr[i]: float(Vmax[i]) for i in range(len(RN.SpStr))}
        else:
            Vmax_dict = {species: float(Vmax_dict[species]) if species in Vmax_dict else 1.0 for species in RN.SpStr}

        if Km_dict is None:
            Km = generate_random_vector(len(RN.SpStr), seed=3).round(2) * 5000
            Km_dict = {RN.SpStr[i]: float(Km[i]) for i in range(len(RN.SpStr))}
        else:
            Km_dict = {species: float(Km_dict[species]) if species in Km_dict else 0.5 for species in RN.SpStr}
    # =========================

    # Function to calculate the reaction rates
    def reaction_rates(concentrations, t):
        rates = np.zeros(len(RN.RnStr))

        for i, reaction in enumerate(RN.RnStr):
            reactants = RN_dict[reaction][0]
            products = RN_dict[reaction][1]
            kind = rate[i]

            if kind == 'mak':  # Acción de masas
                rate_value = k[reaction]
                for reactant, coeff in reactants:
                    rate_value *= concentrations[species_concentrations[reactant]] ** coeff
                rates[i] = rate_value

            elif kind == 'mmk':  # Michaelis-Menten
                # Se escoge el primer reactivo como sustrato para Vmax/Km
                if len(reactants) == 0:
                    rates[i] = 0.0
                    continue
                substrate = reactants[0][0]
                substrate_conc = concentrations[species_concentrations[substrate]]
                rate_value = Vmax_dict[substrate] * substrate_conc / (Km_dict[substrate] + substrate_conc)
                rates[i] = rate_value

        dxdt = np.zeros_like(concentrations)
        for i, reaction in enumerate(RN.RnStr):
            for reactant, coeff in RN_dict[reaction][0]:
                dxdt[species_concentrations[reactant]] -= coeff * rates[i]
            for product, coeff in RN_dict[reaction][1]:
                dxdt[species_concentrations[product]] += coeff * rates[i]

        return dxdt

    # Imprimir las ecuaciones diferenciales
    print("\nDifferential equations of the system:")

    for species in RN.SpStr:
        eq_terms = []
        for i, reaction in enumerate(RN.RnStr):
            rate_expr = f"v{i+1}"
            for reactant, coeff in RN_dict[reaction][0]:
                if reactant == species:
                    term = f"-{coeff}*{rate_expr}" if coeff > 1 else f"-{rate_expr}"
                    eq_terms.append(term)
            for product, coeff in RN_dict[reaction][1]:
                if product == species:
                    term = f"+{coeff}*{rate_expr}" if coeff > 1 else f"+{rate_expr}"
                    eq_terms.append(term)
        if eq_terms:
            rhs = " ".join(eq_terms)
            print(f"d[{species}]/dt = {rhs}")

    # Print the analytical expressions of the velocities
    print("\nAnalytical expressions of velocities v(x,param):")

    for i, reaction in enumerate(RN.RnStr):
        vlabel = f"v{i+1}"
        kind = rate[i].lower()
        reactants = RN_dict[reaction][0]

        if kind == 'mak': 
            expr = f"k_{reaction}"
            for reactant, coeff in reactants:
                term = f"[{reactant}]^{coeff}" if coeff > 1 else f"[{reactant}]"
                expr += f" * {term}"
            print(f"{vlabel} = {expr}  (mak)")

        elif kind == 'mmk':
            if reactants:
                sustrato = reactants[0][0] 
                expr = f"(Vmax_{sustrato} * [{sustrato}]) / (Km_{sustrato} + [{sustrato}])"
                print(f"{vlabel} = {expr}  (mmk)") 
            else:
                print(f"{vlabel} = 0  (mmk without defined substrate)")

    # Create the vector ‘spec’ in order according to rate_list
    spec_vector = []
    for i, reaction in enumerate(RN.RnStr):
        kind = rate[i].lower()
        if kind == 'mak':
            spec_vector.append([k[reaction]])
        elif kind == 'mmk':
            reactants = RN_dict[reaction][0]
            if reactants:
                substrate = reactants[0][0]
                spec_vector.append([Vmax_dict[substrate], Km_dict[substrate]])
            else:
                spec_vector.append([0.0, 0.0])  # Caso de mmk sin sustrato definido

    print("\nVector spec with parameters in order:")
    print("spec_vector =",spec_vector)
    for i, params in enumerate(spec_vector):
        print(f"  v{i+1}: {params}")
    print("")

    ########################################################
    time_points = np.linspace(t_span[0], t_span[1], n_steps)
    result = odeint(reaction_rates, x0, time_points)

    data_concentrations = {"Time": time_points}
    for i, species in enumerate(RN.SpStr):
        data_concentrations[species] = result[:, i]

    time_series = pd.DataFrame(data_concentrations)
    time_series = time_series[["Time"] + sorted(time_series.columns.difference(["Time"]))]

    flux_data = {"Time": time_points}
    for i, rxn in enumerate(RN.RnStr):
        flux_data[f"Flux_r{i+1}"] = []

    for t_idx in range(len(time_points)):
        concentrations = result[t_idx, :]
        for i, rxn in enumerate(RN.RnStr):
            kind = rate[i]
            reactants = RN_dict[rxn][0]

            if kind == 'mak':
                rate_value = k[rxn]
                for species, coeff in reactants:
                    rate_value *= concentrations[species_concentrations[species]] ** coeff

            elif kind == 'mmk':
                if len(reactants) == 0:
                    rate_value = 0.0
                else:
                    substrate = reactants[0][0]
                    substrate_conc = concentrations[species_concentrations[substrate]]
                    rate_value = Vmax_dict[substrate] * substrate_conc / (Km_dict[substrate] + substrate_conc)

            flux_data[f"Flux_r{i+1}"].append(rate_value)

    flux_vector = pd.DataFrame(flux_data)

    return time_series, flux_vector











###########################################################################################
# Function that computes the time series of a metapopulation reaction-diffusion model
def simulate_odes_metapop_mak(RN, x0=None, t_span=None, n_steps=None, k=None, exchange_rates=None, D=None, grid_shape=None):
    """
    Simulates the dynamics of a spatiotemporal metapopulation with reaction-diffusion terms.
    Specifically, it models the evolution of species in a reaction network distributed over a 
    discrete space (a grid of patches), considering both chemical interactions within each patch 
    and exchange between patches through diffusion and exchange rates.

    Parameters:
        RN (dict): Object describing the reaction network.
        x0 (array-like, optional): Initial conditions of the species. Default is None.
        t_span (tuple, optional): Time range for the simulation. Default is None.
        n_steps (int, optional): Number of steps for the simulation. Default is None.
        k (array-like, optional): Reaction rate constants. Default is None.
        exchange_rates (array-like, optional): Exchange rates between modules (diffusion between nodes). Default is None.
        D (array-like, optional): Spatial diffusion coefficients for each species. Default is None.
        grid_shape (tuple, optional): Shape of the spatial grid (rows, columns). Default is None.

    Returns:
        dict: Simulation results for each patch (key: patch index).
    """
    # Validate that RN and its attributes are not None
    if RN is None or not hasattr(RN, 'SpStr') or not hasattr(RN, 'RnStr'):
        raise ValueError("The reaction network (RN) is invalid or incomplete.")
    
    # Default parameters if not specified
    if grid_shape is None:
        grid_shape = (2, 2)                   # Default configuration if not provided
    if x0 is None:
        x0 = np.random.rand(grid_shape[0], grid_shape[1], len(RN.SpStr)) * 2 # Random initial conditions
    if n_steps is None:
        n_steps = 500                         # Default number of steps    
    if t_span is None:
        t_span = (0, 100)                      # Default simulation time
    if k is None:
        k = np.random.rand(len(RN.RnStr)) * 1 # Random rate constants
    if exchange_rates is None:
        exchange_rates = np.ones((grid_shape[0], grid_shape[1], len(RN.SpStr)))  # No exchange by default 
    if D is None:
        D = np.random.rand(len(RN.SpStr))     # Default uniform diffusion coefficients
    
    # Function describing the system of differential equations with diffusion
    def ode_system(x_flat, t, k, RN, exchange_rates, D, grid_shape):
        # Reconstruct the spatial grid
        x = x_flat.reshape(grid_shape[0], grid_shape[1], -1)
        dxdt = np.zeros_like(x)
        
        # Stoichiometric matrices
        S = np.array(RN.RnMsupp)
        P = np.array(RN.RnMprod)
        
        # Reaction terms
        for i in range(len(RN.SpStr)):      # For each species
            for r in range(len(RN.RnStr)):  # For each reaction
                rate = k[r] * np.prod(x[:, :, :] ** S[r, :], axis=-1)
                dxdt[:, :, i] += rate * (P[r, i] - S[r, i])
        
        # Diffusion terms (diffusion in the grid)
        for i in range(len(RN.SpStr)):      # For each species
            dxdt[:, :, i] += D[i] * (
                np.roll(x[:, :, i],  1, axis=0) +  # Upper neighbor
                np.roll(x[:, :, i], -1, axis=0) +  # Lower neighbor
                np.roll(x[:, :, i],  1, axis=1) +  # Left neighbor
                np.roll(x[:, :, i], -1, axis=1) -  # Right neighbor
                4 * x[:, :, i]  # Center
            )
        
        # Exchange rates between nodes (patches)
        for i in range(len(RN.SpStr)):
            dxdt[:, :, i] += exchange_rates[:, :, i]
        
        return dxdt.flatten()
    # Verify that t_span is a tuple with two values
    if not (isinstance(t_span, tuple) and len(t_span) == 2):
        raise TypeError(f"t_span must be a tuple (t0, tf), but received {t_span} of type {type(t_span)}.")
    
    # Define the time range
    t = np.linspace(t_span[0], t_span[1], n_steps)
    
    # Solve the system of ODEs
    result = odeint(ode_system, x0.flatten(), t, args=(k, RN, exchange_rates, D, grid_shape))
    
    # Reconstruct results as a dictionary 
    modules = {
        f'Patch ({i+1}, {j+1})': result[:, (i * grid_shape[1] + j    ) * len(RN.SpStr):
                                           (i * grid_shape[1] + j + 1) * len(RN.SpStr)]
        for i in range(grid_shape[0]) for j in range(grid_shape[1])
    }
    # Number of patches
    num_patches = grid_shape[0] * grid_shape[1]
    print(f"Number of patches: {num_patches}")
    
    return modules

###########################################################################################
# Function to simulate the dynamics of a metapopulation with Michaelis-Menten kinetics and diffusion
def simulate_odes_metapop_mmk(RN, x0=None, t_span=None, n_steps=None, Vmax=None, Km=None, D=None, grid_shape=None):
    """
    Simulates the dynamics of a metapopulation with Michaelis-Menten kinetics and diffusion on a spatial grid.

    Parameters:
        RN (dict): Object describing the reaction network.
        x0 (array-like, optional): Initial conditions of the species.
        t_span (tuple, optional): Simulation time range.
        n_steps (int, optional): Number of simulation steps.
        Vmax (array-like, optional): Maximum reaction rate for each reaction.
        Km (array-like, optional): Michaelis-Menten constant for each reaction.
        D (array-like, optional): Diffusion coefficients for each species.
        grid_shape (tuple, optional): Shape of the spatial grid (rows, columns).

    Returns:
        dict: Simulation results for each patch (key: patch index).
    """
    if RN is None or not hasattr(RN, 'SpStr') or not hasattr(RN, 'RnStr'):
        raise ValueError("The reaction network (RN) is invalid or incomplete.")
    
    if grid_shape is None:
        grid_shape = (2, 2)
    if x0 is None:
        x0 = np.random.rand(grid_shape[0], grid_shape[1], len(RN.SpStr)) * 2
    if n_steps is None:
        n_steps = 500
    if t_span is None:
        t_span = (0, 100)
    if Vmax is None:
        Vmax = np.random.rand(len(RN.RnStr))
    if Km is None:
        Km = np.random.rand(len(RN.RnStr))
    if D is None:
        D = np.random.rand(len(RN.SpStr))
    
    def ode_system(x_flat, t, Vmax, Km, RN, D, grid_shape):
        x = x_flat.reshape(grid_shape[0], grid_shape[1], -1)
        dxdt = np.zeros_like(x)
        
        S = np.array(RN.RnMsupp)
        P = np.array(RN.RnMprod)
        
        for i in range(len(RN.SpStr)):
            for r in range(len(RN.RnStr)):
                rate = Vmax[r] * (x[:, :, :] ** S[r, :]) / (Km[r] + x[:, :, :])
                dxdt[:, :, i] += np.sum(rate * (P[r, i] - S[r, i]), axis=-1)
        
        for i in range(len(RN.SpStr)):
            dxdt[:, :, i] += D[i] * (
                np.roll(x[:, :, i],  1, axis=0) + 
                np.roll(x[:, :, i], -1, axis=0) + 
                np.roll(x[:, :, i],  1, axis=1) + 
                np.roll(x[:, :, i], -1, axis=1) - 
                4 * x[:, :, i]
            )
        
        return dxdt.flatten()
    
    t = np.linspace(t_span[0], t_span[1], n_steps)
    result = odeint(ode_system, x0.flatten(), t, args=(Vmax, Km, RN, D, grid_shape))
    
    modules = {
        f'Patch ({i+1}, {j+1})': result[:, (i * grid_shape[1] + j) * len(RN.SpStr):
                                           (i * grid_shape[1] + j + 1) * len(RN.SpStr)]
        for i in range(grid_shape[0]) for j in range(grid_shape[1])
    }
    
    print(f"Number of patches: {grid_shape[0] * grid_shape[1]}")
    return modules

###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# Function to create the simulations of the metapopulation dynamics of a reaction-diffusion model PDE
def simulate_pde_rd(RN, x0=None, t_span=None, n_steps=None, k=None, exchange_rates=None, D=None, grid_shape=None):
    if RN is None or not hasattr(RN, 'SpStr') or not hasattr(RN, 'RnStr'):
        raise ValueError("The reaction network (RN) is invalid or incomplete.")
    
    if grid_shape is None:
        grid_shape = (2, 2)
    if x0 is None:
        x0 = np.random.rand(grid_shape[0], grid_shape[1], len(RN.SpStr)) * 2
        print(f"Values of x0: {x0}")
    if n_steps is None:
        n_steps = 500
    if t_span is None:
        t_span = (0, 100)
    if k is None:
        k = np.random.rand(len(RN.RnStr)) * 1 # Random rate constants into the interval [0, 1]
    if exchange_rates is None:
        exchange_rates = np.ones((grid_shape[0], grid_shape[1], len(RN.SpStr)))  # No exchange by default        
    if D is None:
        D = np.random.rand(len(RN.SpStr)) # Default diffusion coefficients 
    
    def ode_system(x_flat, t, k, RN, D, grid_shape):
        x = x_flat.reshape(grid_shape[0], grid_shape[1], -1)
        dxdt = np.zeros_like(x)
        
        S = np.array(RN.RnMsupp)
        P = np.array(RN.RnMprod)
        
        for i in range(len(RN.SpStr)):
            for r in range(len(RN.RnStr)):
                rate = k[r] * np.prod(x[:, :, :] ** S[r, :], axis=-1) # Calculate the reaction rate 
                dxdt[:, :, i] += rate * (P[r, i] - S[r, i])           # Mass action kinetics
            
            # Difusión con diferencias finitas centradas en una malla 2D con coeficientes D_i específicos para cada especie.
            dxdt[:, :, i] += D[i] * (
                np.roll(x[:, :, i],  1, axis=0) +  # Vecino superior
                np.roll(x[:, :, i], -1, axis=0) +  # Vecino inferior
                np.roll(x[:, :, i],  1, axis=1) +  # Vecino derecho
                np.roll(x[:, :, i], -1, axis=1) -  # Vecino izquierdo
                4 * x[:, :, i]  # Centro (multiplicado por 4)
            )

            # # Difusión en la dirección vertical
            # dxdt[:, :, i] += D[i] * (
            #     np.roll(x[:, :, i],  1, axis=0) +  # Vecino superior
            #     np.roll(x[:, :, i], -1, axis=0) -  # Vecino inferior
            #     2 * x[:, :, i]  # Centro (multiplicado por 2 en lugar de 4)
            # )

            # # Difusión en la dirección horizontal
            # dxdt[:, :, i] += D[i] * (
            #     np.roll(x[:, :, i],  1, axis=1) +  # Vecino derecho
            #     np.roll(x[:, :, i], -1, axis=1) -  # Vecino izquierdo
            #     2 * x[:, :, i]  # Centro (multiplicado por 2 en lugar de 4)
            # )

        # Exchange rates between nodes (patches)
        for i in range(len(RN.SpStr)):
            dxdt[:, :, i] += exchange_rates[:, :, i]            
        
        return dxdt.flatten()
    
    if not (isinstance(t_span, tuple) and len(t_span) == 2):
        raise TypeError(f"t_span must be a tuple (t0, tf), but received {t_span} of type {type(t_span)}.")
    
    t = np.linspace(t_span[0], t_span[1], n_steps)
    
    result = odeint(ode_system, x0.flatten(), t, args=(k, RN, D, grid_shape))
    
    return result.reshape(n_steps, grid_shape[0], grid_shape[1], len(RN.SpStr))

###########################################################################################
# Function for solve the Linear Programming Problem: S.v>=0, v>0
def minimize_sv(S, epsilon=1,method='highs'):
    """
    Minimizes c @ v subject to S @ v >= 0 and v > 0.
    
    Parameters:
    ----------
    S : numpy.ndarray
        Stoichiometric matrix of the reaction network (RN).
    epsilon : float, optional
        Small positive value to ensure all coordinates of v remain strictly positive (default is 1).

    Returns:
    -------
    numpy.ndarray
        Optimal process vector v with all strictly positive coordinates.

    Raises:
    ------
    ValueError
        If no feasible solution is found.
    """
    n_species, n_reactions = S.shape  # Dimensions of S
    c = np.ones(n_reactions)          # Objective function: minimize the sum of v
    A_ub = -S                         # Reformulate S @ v >= 0 as -S @ v <= 0
    b_ub = np.zeros(n_species)        # Inequality constraints
    bounds = [(epsilon, None)] * n_reactions  # v > 0 (avoiding exact zero values) 
    
    # Solve the linear programming problem: minimize c @ v subject to (A_ub @ v <= b_ub)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                     method=method) # 'highs' uses the Dual Simplex method (good for many constraints)
                                     # 'highs-ipm' uses the Interior Point method (faster for large sparse problems)
                    
    
    if result.success:
        return result.x
    else:
        raise ValueError("No feasible solution was found.")
    
import numpy as np
from scipy.optimize import linprog

def maximize_sv(S, epsilon=1, method='highs'):
    """
    Maximizes c @ v subject to S @ v >= 0 and v > 0.
    
    Parameters:
    ----------
    S : numpy.ndarray
        Stoichiometric matrix of the reaction network (RN).
    epsilon : float, optional
        Small positive value to ensure all coordinates of v remain strictly positive (default is 1).

    Returns:
    -------
    numpy.ndarray
        Optimal process vector v with all strictly positive coordinates.

    Raises:
    ------
    ValueError
        If no feasible solution is found.
    """
    n_species, n_reactions = S.shape  # Dimensions of S
    c = -np.ones(n_reactions)         # Objective function: maximize the sum of -v (equivalent to minimize c @ v)
    A_ub = -S                         # Reformulate S @ v >= 0 as -S @ v <= 0
    b_ub = np.zeros(n_species)        # Inequality constraints
    bounds = [( None,epsilon)] * n_reactions  # v > 0 (avoiding exact zero values) 
    
    # Solve the linear programming problem: maximize c @ v subject to (A_ub @ v <= b_ub)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    
    if result.success:
        return result.x
    else:
        raise ValueError("No feasible solution was found.")

    
########################################################################################### 
# Function to calculate the echenlon form of a matrix
def row_echelon_form(A):
    A = A.astype(float)  # Asegurar que los cálculos sean en punto flotante
    rows, cols = A.shape
    for i in range(min(rows, cols)):
        # Encontrar el pivote y hacer intercambio de filas si es necesario
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] != 0:
            A[[i, max_row]] = A[[max_row, i]]  # Intercambio de filas
        
        # Hacer ceros debajo del pivote
        for j in range(i + 1, rows):
            if A[i, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
    
    return A