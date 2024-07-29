from dataclasses import dataclass
from numbers import Real
from typing import Literal

from rustworkx import PyDiGraph, InvalidNode

@dataclass(slots=True)
class Species:
    index: int
    name: str
    quantity: Real

@dataclass(slots=True)
class ReactionNode:
    index: int
    name: str
    rate: Real | None = None

@dataclass(slots=True)
class ReactionEdge:
    index: int
    source: int
    target: int
    type: Literal["reactant", "product"]
    coefficient: Real

@dataclass(slots=True)
class Reaction:
    node: ReactionNode 
    edges: list[ReactionEdge]

class ReactionNetwork(PyDiGraph):

    ########################################################################################################
    #### Basic methods for species #######################################################################
    ########################################################################################################
    def add_species(self, name: str, quantity: Real) -> int:
        """Add a new species to the reaction network."""
        if self.has_species(name):
            raise ValueError(f"Species '{name}' already exists.")

        new_index = self.add_node(None)
        self[new_index] = Species(new_index, name, quantity)

        return new_index
    

    def species(self) -> list[Species]:
        """Get the list of species in the reaction network."""
        nodes_data = self.nodes()
        return [node_data for node_data in nodes_data if isinstance(node_data, Species)]


    def get_species_by_index(self, index: int) -> Species:
        species = self[index]
        
        if not isinstance(species, Species):
            raise ValueError(f"Node at index {index} is not a species.")
        else:
            return species


    def get_species(self, name: str) -> Species:
        def filter_species_by_name(payload: Species | ReactionNode) -> bool:
            return isinstance(payload, Species) and payload.name == name
        
        indices = self.filter_nodes(filter_species_by_name)

        if len(indices) == 0:
            raise InvalidNode(f"Species '{name}' does not exist.")

        if len(indices) > 1:  # TODO: Estudiar cómo evitar llegar a este punto
            raise ValueError(f"Malformed reaction network. Species '{name}' is not unique.")

        return self.get_species_by_index(indices[0])
    

    def has_species(self, name: str) -> bool:
        try:
            self.get_species(name)
        except InvalidNode:
            is_present = False
        else:
            is_present = True

        return is_present


    ########################################################################################################
    #### Basic methods for reactions #######################################################################
    ########################################################################################################  
    def add_reaction(
            self, name: str, support: list[str], products: list[str], 
            support_coefficients: list[Real], products_coefficients: list[Real], rate: Real = None
    ) -> int:
        """Add a new reaction to the reaction network."""
        if self.has_reaction(name):
            raise ValueError(f"Reaction '{name}' already exists.")
        reaction_node_index = self.add_node(None)
        self[reaction_node_index] = ReactionNode(reaction_node_index, name, rate)

        for reactant in support:
            if not self.has_species(reactant):
                raise ValueError(f"Reactant '{reactant}' must be a declared species.")
        
        for product in products:
            if not self.has_species(product):
                raise ValueError(f"Product '{product}' must be a declared species.")

        support_indices = [self.get_species(reactant).index for reactant in support]
        n_reactants = len(support)
        products_indices = [self.get_species(product).index for product in products]
        n_products = len(products)

        support_edges_indices = self.add_edges_from(list(zip(support_indices, [reaction_node_index] * n_reactants, [None] * n_reactants)))
        for i, edge_index in enumerate(support_edges_indices):
            edge_data = ReactionEdge(edge_index, support_indices[i], reaction_node_index, "reactant", support_coefficients[i])
            self.update_edge_by_index(edge_index, edge_data)

        products_edges_indices = self.add_edges_from(list(zip([reaction_node_index] * n_products, products_indices, [None] * n_products)))
        for i, edge_index in enumerate(products_edges_indices):
            edge_data = ReactionEdge(edge_index, reaction_node_index, products_indices[i], "product", products_coefficients[i])
            self.update_edge_by_index(edge_index, edge_data)

        return reaction_node_index
    

    def get_reaction_edges_by_index(self, reaction_index: int) -> list[ReactionEdge]:
        return [edge[2] for edge in self.incident_edge_index_map(reaction_index, all_edges = True).values()]
    

    def reactions(self) -> list[Reaction]:
        """Get the list of reactions in the reaction network."""
        reaction_nodes = [node_data for node_data in self.nodes() if isinstance(node_data, ReactionNode)]
        return [
            Reaction(reaction_node, self.get_reaction_edges_by_index(reaction_node.index)) 
            for reaction_node in reaction_nodes
        ]


    def get_reaction(self, name: str) -> Reaction:
        def filter_reactions_by_name(payload: Species | ReactionNode) -> bool:
            return isinstance(payload, ReactionNode) and payload.name == name
        
        index = self.filter_nodes(filter_reactions_by_name)

        if len(index) == 0:
            raise InvalidNode(f"Reaction '{name}' does not exist.")

        if len(index) > 1:  # TODO: Estudiar qué evitar llegar a este punto
            raise ValueError(f"Malformed reaction network. Reaction '{name}' is not unique.")

        return Reaction(self[index], self.get_reaction_edges_by_index(index))


    def has_reaction(self, name: str) -> bool:
        try:
            self.get_reaction(name)
        except InvalidNode:
            is_present = False
        else:
            is_present = True

        return is_present
    
    

    def is_active_reaction(self, reaction_name: str) -> bool: # TODO: Considerar overloads
        """
        Check if a reaction is active.

        Parameters
        ----------
        reaction : str
            The reaction name.

        Returns
        -------
        bool
            True if the reaction is active, False otherwise.
        """
        reaction = self.get_reaction(reaction_name)
        support_edges = (edge for edge in reaction.edges if edge.type == "reactant")

        for edge in support_edges:
            reactant = self.get_species_by_index(edge.source)
            if reactant.quantity < edge.coefficient:
                active = False
                break
        else:
            active = True

        return active

    
    ########################################################################################################
    #### obtaining sets of species/reactions producing/consuming one another################################
    ########################################################################################################  
    # def get_reactions_from_species(self, species_names: str | Collection[str]) -> list[NodeIndices]: # TODO: Considerar el tipo de retorno deseado. Ojalá list[Reaction] # TODO: Separar en get_active_reactions_from_species y get_reactions_from_species
    #     """
    #     Obtain the reactions activated by a given species set.

    #     Parameters
    #     ----------
    #     species : str | Collection[str]
    #         The species set.

    #     Returns
    #     -------
    #     list[NodeIndices]
    #         Indices of the reactions activated by the species set.
    #     """
    #     if isinstance(species_names, str):
    #         species_names = [species_names]
        
    #     species_indices = [self.get_species(specie).index for specie in species_names]
    #     candidates = [reaction_index for reaction_index in self.adj_direction(species_indices, direction = False).keys()]
            
    #     return [reaction_index for reaction_index in candidates.keys() if self.is_active_reaction(self[reaction_index].name, self[species_indices].name)]
    
    
    # def get_supp_from_reactions(self, reaction: str | Collection[str]) -> list[Species]:
    #     if isinstance(reaction, str):
    #         reaction = [reaction]
        
    #     reaction_indices = [self.get_reaction(reac) for reac in reaction]

    #     return [self[self.adj_direction(reaction_index, direction = True)] for reaction_index in reaction_indices] # TODO: Estudiar caso de que no exista la reacción
    

    # def get_prod_from_reactions(self, reaction: str | Collection[str]) -> dict[int, Real]:
    #     if isinstance(reaction, str):
    #         reaction = [reaction]
        
    #     reaction_indices = [self.get_reaction(reac) for reac in reaction]

    #     out = {}
    #     for reaction_index in reaction_indices:
    #         reactions_dict = self.adj_direction(reaction_index, direction = False)
    #         out.update(reactions_dict)
            
    #     return out # No estoy aplicando el umbral # TODO: Estudiar caso de que no exista la reacción
    

    # def get_species_from_reactions(self, reaction: str | Collection[str]) -> dict[int, Real]:
    #     if isinstance(reaction, str):
    #         reaction = [reaction]
        
    #     reaction_indices = [self.get_reaction(reac) for reac in reaction]

    #     out = {}
    #     for reaction_index in reaction_indices:
    #         reactions_dict = self.adj(reaction_index)
    #         out.update(reactions_dict)
            
    #     return out # No estoy aplicando el umbral # TODO: Estudiar caso de que no exista la reacción
    

    # def get_prod_from_species(self):
    #     ...
    

    # def get_reactions_consuming_species(self):
    #     ...


    # def get_reactions_producing_species(self):
    #     ...