"""
Graph Factory.
"""
import os
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from Core.Graph.RKGraph import RKGraph
from Core.Graph.TreeGraphLSH import TreeGraphLSH
from Core.Graph.TreeGraphDynamic import TreeGraphDynamic


class GraphFactory():
    def __init__(self):
        self.creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._create_passage_graph,
            "tree_graph_lsh": self._create_tree_graph_lsh,
            "tree_graph_dynamic": self._create_tree_graph_dynamic
        }


    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        return self.creators[config.graph.graph_type](config, **kwargs)

    @staticmethod
    def _create_er_graph(config, **kwargs):
        return ERGraph(
            config.graph, **kwargs
        )

    @staticmethod
    def _create_rkg_graph(config, **kwargs):
        return RKGraph(config.graph, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _create_tree_graph_balanced(config, **kwargs):
        return TreeGraphBalanced(config, **kwargs)

    @staticmethod
    def _create_passage_graph(config, **kwargs):
        return PassageGraph(config.graph, **kwargs)
    
    @staticmethod
    def _create_tree_graph_lsh(config, **kwargs):
        return TreeGraphLSH(config, **kwargs)

    @staticmethod
    def _create_tree_graph_dynamic(config, **kwargs):
        return TreeGraphDynamic(config, **kwargs)

get_graph = GraphFactory().get_graph
