import pytest
from pyCOT.rn_rustworkx import Reaction, ReactionNode, ReactionEdge

@pytest.fixture
def reaction_node():
    return ReactionNode(index=2, name="R1", rate=0.5)

@pytest.fixture
def reaction_edge_reactant():
    return ReactionEdge(index=0, source_index=0, target_index=2, species_name="S1", type="reactant", coefficient=1.0)

@pytest.fixture
def reaction_edge_product():
    return ReactionEdge(index=1, source_index=2, target_index=1, species_name="S2", type="product", coefficient=2.0)

@pytest.fixture
def reaction(reaction_node, reaction_edge_reactant, reaction_edge_product):
    return Reaction(node=reaction_node, edges=[reaction_edge_reactant, reaction_edge_product])

def test_reaction_name(reaction: Reaction):
    assert reaction.name() == "R1"

def test_reaction_support_edges(reaction: Reaction, reaction_edge_reactant: ReactionEdge):
    assert reaction.support_edges() == [reaction_edge_reactant]

def test_reaction_products_edges(reaction: Reaction, reaction_edge_product: ReactionEdge):
    assert reaction.products_edges() == [reaction_edge_product]

def test_reaction_support_indices(reaction: Reaction, reaction_edge_reactant: ReactionEdge):
    assert reaction.support_indices() == [reaction_edge_reactant.source_index]

def test_reaction_support_names(reaction: Reaction, reaction_edge_reactant: ReactionEdge):
    assert reaction.support_names() == [reaction_edge_reactant.species_name]

def test_reaction_products_indices(reaction: Reaction, reaction_edge_product: ReactionEdge):
    assert reaction.products_indices() == [reaction_edge_product.target_index]

def test_reaction_products_names(reaction: Reaction, reaction_edge_product: ReactionEdge):
    assert reaction.products_names() == [reaction_edge_product.species_name]

def test_reaction_species_indices(reaction: Reaction, reaction_edge_reactant: ReactionEdge, reaction_edge_product: ReactionEdge):
    indices = reaction.species_indices()
    assert reaction_edge_reactant.source_index in indices
    assert reaction_edge_product.target_index in indices

def test_reaction_species_names(reaction: Reaction, reaction_edge_reactant: ReactionEdge, reaction_edge_product: ReactionEdge):
    names = reaction.species_names()
    assert reaction_edge_reactant.species_name in names
    assert reaction_edge_product.species_name in names