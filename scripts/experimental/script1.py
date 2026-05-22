# Script 1: Exploring a Reaction Network with pyCOT

# ======================================== source .venv/bin/activate
# 1. LIBRARY LOADING AND CONFIGURATION: 
# ======================================== Depurador de Python: Archivo actual (pyCOT)
# Import necessary libraries
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the root directory to the PYTHONPATH

# Import the necessary modules from pyCOT  
from pyCOT.io.functions import read_txt, print_reaction_network
from pyCOT.visualization.rn_visualize import rn_visualize_html, rn_visualize_png_in_out

# ========================================
# 2. CREATING THE REACTION_NETWORK OBJECT
# ======================================== 
file_path = 'data/examples/autopoietic.txt'  # Change this path according to the desired file
# file_path = 'data/examples/Farm.txt'            
file_path = 'data/ai_generated/EColi.txt'      

# Option 1: Create a ReactionNetwork from a string
rn = read_txt(file_path)  # Creates the variable rn containing an object of ReactionNetwork class

# Option 2: Create a ReactionNetwork from a .txt file with exact names (case-sensitive)
# file_path = 'data/examples/PassiveUncomforableIndignated_problemsolution.txt' 
# rn = read_txt(file_path, exact_names=True) # exact_names=True to preserve the distinction between species with s ≠ S (e.g., 's' and 'S'):

rn_comments = rn.reaction_comments
print("Diccionario con las reacciones que tienen comentarios:\n",rn_comments)

""" 
Some of the methods of the ReactionNetwork class are: 
- rn.species(): Returns a list of Species objects present in the network.
- rn.reactions(): Returns a list of Reaction objects defined in the network.
- rn.stoichiometry_matrix(): Returns an object of class `StoichiometryMatrix`, which represents a matrix 
    where each entry corresponds to the net stoichiometric coefficient of a species in a specific reaction.
"""

# ========================================
# 3. SPECIES OF THE REACTION NETWORK
# ========================================
# Print the list of Species objects
"""
The Species object has the following attributes:
- index: The index of the species in the reaction network.
- name: The name of the species (e.g., 'l', 's1', 's2').
- quantity: The initial quantity or concentration of the species in the reaction network (can be None if not set).
"""
print("-"*100)
# print("List of Species objects:") 
species_list = rn.species()
# print(species_list)
# for specie in species_list:
#     print(specie) 

species = [specie.name for specie in species_list]
print("\nSpecies Set =",species) # Prints the list of the set of species 
print(f"Number of species: {len(species)}")

# ========================================
# 4. REACTIONS OF THE REACTION NETWORK
# ========================================
# Print the list of Reactions objects
"""
rn.reactions() returns a list of Reaction objects, each representing a reaction in the reaction network.
The Reaction class has the following attributes:    
- node: The node of the reaction in the reaction network.
    - index: The index of the reaction node in the graph.
    - name: The name of the reaction (e.g., 'R1', 'R2').
    - rate: The kinetic rate law associated with the reaction (can be None if not yet defined).
- edges: A list of Edge objects representing the edges of the reaction.
    - index: The index of the edge in the reaction graph.
    - source_index: The index of the source node (species or reaction) from which the edge originates.
    - target_index: The index of the target node (species or reaction) to which the edge points.
    - species_name: The name of the chemical species involved in the reaction.
    - type: The role of the species in the reaction, either 'reactant' or 'product'.
    - coefficient: The stoichiometric coefficient of the species in the reaction.
"""
print("-"*100)
# print("List of Reactions objects:")
reactions_list = rn.reactions()
# print(reactions_list)

# Print the reactions set as a list
reactions = [reaction.name() for reaction in reactions_list]
print("\nReactions Set =",reactions) # Prints the list of the set of reactions
print(f"Number of reactions: {len(reactions)}")

# # ========================================
# # 5. REACTION NETWORK IN TEXT FORMAT
# # ======================================== 
# Printing the reaction network
print("-"*100)
print("Reaction network:")
print_reaction_network(rn)

# # # ========================================
# # # 6. STOICHIOMETRIC MATRICES
# # # ========================================
# # Print the stoichiometric matrix 
# print("-" * 100)
# print("Stoichiometric matrix (products - reactants):")
# print(rn.stoichiometry_matrix())

# print("\nMatrix of stoichiometric coefficients of the reactants:")
# print(rn.reactants_matrix())

# print("\nMatrix of stoichiometric coefficients of the products:")
# print(rn.products_matrix())

# # ========================================
# # 7. REACTANTS AND PRODUCTS OF EACH REACTION
# # ========================================
# # Print the reactants and products of a reaction
# print("-"*100)
# reaction = rn.get_reaction('R2')
# print(f"Reaction {reaction}")
# reactants = [edge.species_name for edge in reaction.support_edges()]
# coef_r = [edge.coefficient for edge in reaction.support_edges()]
# print(f"Reactants of {reaction.name()}={reactants}, Coefficients of the reactants={coef_r}")
# products = [edge.species_name for edge in reaction.products_edges()]
# coef_p = [edge.coefficient for edge in reaction.products_edges()]
# print(f"Products of {reaction.name()}={products}, Coefficients of the products={coef_p}")

# # Print reactants and products of all reactions
# print("-"*100)
# print("Reactants of all reactions:")
# for r in reactions:
#     reaction = rn.get_reaction(r)
#     reactants = [edge.species_name for edge in reaction.support_edges()]
#     print(f"{r}: {reactants}")

# print("-"*100)
# print("\nProducts of all reactions:")
# for r in reactions:
#     reaction = rn.get_reaction(r)
#     products = [edge.species_name for edge in reaction.products_edges()]
#     print(f"{r}: {products}")

# # ========================================
# # 8. VISUALIZATION OF THE REACTION NETWORK
# # ========================================  
print("-"*100)
rn_visualize_html(rn, filename="reaction_network.html")  