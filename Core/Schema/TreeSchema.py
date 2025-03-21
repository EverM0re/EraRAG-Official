from typing import Dict, List, Set, Tuple, Optional

class TreeNode:
    def __init__(self, text: str, index: int, children: Set[int], embedding) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embedding = embedding

class TreeSchema:
    def __init__(
        self, all_nodes: List[TreeNode]  = None, layer_to_nodes: List[TreeNode] = None
    ) -> None:
        self.all_nodes = all_nodes
        self.layer_to_nodes = layer_to_nodes

    @property
    def num_layers(self) -> int:
        if (self.layer_to_nodes is None):
            return 0
        return len(self.layer_to_nodes)

    @property
    def num_nodes(self) -> int:
        if (self.all_nodes is None):
            return 0
        return len(self.all_nodes)

    @property
    def leaf_nodes(self) -> int:
        if (self.num_layers == 0):
            return None
        return self.layer_to_nodes[0]

    @property
    def root_nodes(self) -> int:
        if (self.num_layers == 0):
            return None
        return self.layer_to_nodes[-1]