# pyCOT Analysis Module
# Contains analysis tools for reaction networks

from pyCOT.analysis.process_structure import (
    ProcessAggregation,
    ProcessDisaggregation,
)

# NOTE: Other imports commented out due to missing dependencies
# from pyCOT.analysis.ERC_Hierarchy import ERCHierarchy
# from pyCOT.analysis.ERC_Synergy_Complementarity import ERCSynergyComplementarity
# from pyCOT.analysis.Persistent_Modules_Generator import PersistentModulesGenerator
# from pyCOT.analysis.SORN_Generators import SORNGenerators

__all__ = [
    "ProcessAggregation",
    "ProcessDisaggregation",
    # "ERCHierarchy",
    # "ERCSynergyComplementarity",
    # "PersistentModulesGenerator",
    # "SORNGenerators",
]
