from Core.Storage.BaseGraphStorage import BaseGraphStorage
from Core.Schema.TreeSchema import TreeNode, TreeSchema
from Core.Common.Logger import logger

from typing import Dict, Any, List

import os
import pickle


class TreeGraphStorage(BaseGraphStorage):
    def __init__(self):
        super().__init__()

    name: str = "tree_data.pkl"
    _tree: TreeSchema = TreeSchema()

    def clear(self):
        self._tree = TreeSchema()

    async def _persist(self, force):
        if (os.path.exists(self.tree_pkl_file) and not force):
            return
        logger.info(f"Writing graph into {self.tree_pkl_file}")
        TreeGraphStorage.write_tree_graph(self.tree, self.tree_pkl_file)

    def write_tree_graph(tree, tree_pkl_file):
        with open(tree_pkl_file, "wb") as file:
            pickle.dump(tree, file)

    async def load_tree_graph(self, force) -> bool:
        # Attempting to load the graph from the specified pkl file
        logger.info(f"Attempting to load the tree from: {self.tree_pkl_file}")
        if os.path.exists(self.tree_pkl_file):
            try:
                with open(self.tree_pkl_file, "rb") as file:
                    self._tree = pickle.load(file)
                logger.info(
                    f"Successfully loaded tree from: {self.tree_pkl_file} with {len(self._tree.all_nodes)} nodes and {self._tree.num_layers} layers")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load tree from: {self.tree_pkl_file} with {e}! Need to re-build the tree.")
                return False
        else:
            # Pkl file doesn't exist; need to construct the tree from scratch
            logger.info("Pkl file does not exist! Need to build the tree from scratch.")
            return False

    async def write_tree_leaves(self):
        TreeGraphStorage.write_tree_graph(tree=self.tree, tree_pkl_file=self.tree_leaves_pkl_file)
    
    async def load_tree_graph_from_leaves(self, force = False) -> bool:
        # Attempting to load the graph from the specified pkl file
        logger.info(f"Attempting to load the tree leaves from: {self.tree_leaves_pkl_file}")
        if os.path.exists(self.tree_leaves_pkl_file):
            try:
                with open(self.tree_leaves_pkl_file, "rb") as file:
                    self._tree = pickle.load(file)
                logger.info(
                    f"Successfully loaded tree from: {self.tree_leaves_pkl_file} with {len(self._tree.leaf_nodes)} leaves")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load tree from: {self.tree_leaves_pkl_file} with {e}! Need to re-build the tree.")
                return False
        else:
            # Pkl file doesn't exist; need to construct the tree from scratch
            logger.info("Pkl file does not exist! Need to build the tree from scratch.")
            return False
        
    async def load_full_tree_graph(self, force: bool = False) -> bool:

        logger.info(f"Attempting to load the full tree from: {self.tree_pkl_file}")
        
        if os.path.exists(self.tree_pkl_file):
            try:
                with open(self.tree_pkl_file, "rb") as f:
                    self._tree = pickle.load(f)
                logger.info(f"Successfully loaded full tree with {len(self._tree.leaf_nodes)} leaves and {self._tree.num_layers} layers.")
                return True
            except Exception as e:
                logger.error(f"Failed to load full tree from: {self.tree_pkl_file} - {e}")
                return False
        else:
            logger.warning("Full tree file does not exist. Need to build from scratch.")
            return False

    @property
    def tree(self):
        return self._tree

    @property
    def tree_pkl_file(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    @property
    def tree_leaves_pkl_file(self):
        assert self.namespace is not None
        path = self.tree_pkl_file
        name, extension = os.path.splitext(path)
        path = name + "_leaves" + extension
        return path

    @property
    def root_nodes(self):
        return self.tree.root_nodes

    @property
    def leaf_nodes(self):
        return self.tree.leaf_nodes

    @property
    def num_layers(self):
        return self.tree.num_layers

    @property
    def num_nodes(self):
        return self.tree.num_nodes

    def add_layer(self):
        if (self.num_layers == 0):
            self._tree.layer_to_nodes = []
            self._tree.all_nodes = []
        self._tree.layer_to_nodes.append([])

    def get_layer(self, layer: int):
        return self._tree.layer_to_nodes[layer]
    
    def replace_layer(self, layer: int, nodes: List[TreeNode]):
        assert layer < self.num_layers, "Layer to replace is out of range"
        self._tree.layer_to_nodes[layer] = nodes

    def upsert_node(self, node_id: int, node_data: Dict[str, Any]) ->TreeNode:
        node = TreeNode(index=node_id, text=node_data['text'], children=node_data['children'],
                        embedding=node_data['embedding'])
        layer = node_data['layer']
        self._tree.layer_to_nodes[layer].append(node)
        self._tree.all_nodes.append(node)
        return node

    async def load_graph(self, force: bool = False) -> bool:
        return await self.load_tree_graph(force)

    async def persist(self, force):
        return await self._persist(force)

    async def get_nodes_data(self):
        return [{"content": node.text, "index": node.index} for node in self.tree.all_nodes]

    async def get_node_metadata(self):
        return ["index"]

    def get_node_num(self):
        return self.num_nodes

    async def get_node(self, node_id):
        return self.tree.all_nodes[node_id]

    @property
    def nodes(self):
        return self.tree.all_nodes

    async def neighbors(self, node):

        if not node.children:
            return []
        else:
            [self.tree.all_nodes[node_idx] for node_idx in node.children]

    async def get_community_schema(self):
        return None
