######################################################################################
import os
import sys
# Add the root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCOT.rn_visualize import *
from pyCOT.reaction_network import *
from pyCOT.closure_structure import * 
from pyCOT.file_manipulation import *  

#####################################################################################
# Examples

# File path
file_path = 'Txt/autopoietic.txt' 
# file_path = 'Txt/2019fig1.txt'
# file_path = 'Txt/2019fig2.txt'
# file_path = 'Txt/non_connected_example.txt' 
# file_path = 'Txt/PassiveUncomforableIndignated_problemsolution.txt'
# file_path = 'Txt/Farm.txt'

# Lists of species sets and reactions with colors
S=[["water"],["eggs","chickens","infr"]]
lst_color_specs = [("blue", S[0]), ("yellow", S[1])]

lst_color_reacs = [("purple", ["R1"]), ("orange", ["R2","R3","R4"]), ("green", ["R12","R13","R14"])]

testRN=load_pyCOT_from_file(file_path)
# print(testRN)

# # print(testRN.RN)
# print(testRN.SpStr)
# print(testRN.SpBt)

# print(testRN.RnStr) 
# print(testRN.RnBt)

# print(testRN.RnMsupp) #Coeficientes de los Reactantes
# print(testRN.RnMprod) #Coeficientes de los Productos

#Armar la función del vizualizador 


# print(testRN.RnMprod) # SpStr, SpBt, RnStr, RnBt, RnMsupp, RnMprod
# print(testRN) 
# print(dir(testRN))


# Visualize the network with lst_color_spcs and lst_color_reacs applied  
# rn_get_visualization(testRN)
rn_visualize(testRN, node_size=2000) 
# rn_visualize(testRN,curvature=True) #None=False (Without curvature), True=curvedCCW (Counter Clockwise) 
# rn_visualize(testRN,physics_enabled=True) 
# rn_visualize(testRN, filename='rn_visualize.html') 
# rn_visualize(testRN, lst_color_spcs=[("yellow", ["l","s1"])], filename="rn_visualize.html")
# rn_visualize(testRN, lst_color_spcs=lst_color_specs, filename="rn_visualize.html")
# rn_visualize(testRN, lst_color_reacs=lst_color_reacs, filename="rn_visualize.html")
# rn_visualize(testRN, lst_color_spcs=lst_color_specs, lst_color_reacs=lst_color_reacs, filename="rn_visualize.html")
# rn_visualize(testRN, global_species_color='yellow', global_reaction_color='orange', filename="rn_visualize.html")
# rn_visualize(testRN, global_species_color='blue', global_reaction_color='orange',global_input_edge_color='blue', global_output_edge_color='gray', filename="rn_visualize.html")
# rn_visualize(testRN, lst_color_spcs=lst_color_specs, lst_color_reacs=lst_color_reacs, global_input_edge_color='blue', global_output_edge_color='gray', filename="rn_visualize.html")