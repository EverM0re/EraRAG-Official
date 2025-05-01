import asyncio
import numpy as np
import pickle
import json
import os
import random
from typing import Optional

from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Prompt.RaptorPrompt import SUMMARIZE
from Core.Storage.TreeGraphStorage import TreeGraphStorage
from Core.Schema.TreeSchema import TreeNode, TreeSchema
from Core.Storage.NameSpace import Workspace
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Any
from pathlib import Path

Embedding = List[float]

# from TreeNode
class DynTreeNodeAux():
    def __init__(self, index: int, layer: int,  parent: Optional[int] = None, update_flag: bool = False, valid_flag: bool = True):    
        self.index = index
        self.parent = parent
        self.layer = layer
        self.update_flag = update_flag        
        self.valid_flag = valid_flag


class DynAux:
    # Based on DynTreeNodeAux
    def __init__(self, workspace, shape: Tuple[int, int], Force: bool=False):
        self.workspace = workspace
        self.ns_clustering = workspace.make_for("ns_clustering")
        # self.signature_file = self.ns_clustering.get_save_path("signature.npy")
        # self.hyperplane_file = self.ns_clustering.get_save_path("hyperplanes.npy")
        self.signature_file = "/ssddata/zhengjun/Dynamic_test/signature.npy"
        self.hyperplane_file = "/ssddata/zhengjun/Dynamic_test/hyperplanes.npy"

        # if reset the tree, delete the signature and hyperplane files
        if (Force):
            if os.path.exists(self.signature_file):
                os.remove(self.signature_file)

            if os.path.exists(self.hyperplane_file):
                os.remove(self.hyperplane_file)

        self.NodeAux = []
        self.signature_map = {}
        self.hyperplanes = self.get_hyperplanes(shape)
        self.affected_entities = set()


    def save_hyperplanes(self, hyperplanes: np.ndarray):
        np.save(self.hyperplane_file, hyperplanes)

    def load_hyperplanes(self):
        if os.path.exists(self.hyperplane_file):
            self.hyperplanes = np.load(self.hyperplane_file)
            return True
        return False

    def get_hyperplanes(self, shape: Tuple[int, int], force: bool = False):
        if os.path.exists(self.hyperplane_file) and not force:
            hp =  np.load(self.hyperplane_file)
            logger.info("\n✅ Hyperplane loaded!")
        else:
            hp = np.random.randn(*shape)
            np.save(self.hyperplane_file, hp)
            logger.info("\n⚠️ No existing hyperplane! Regenerated hyperplane!")
        return hp
    
    def init_tree_aux(self, tree: TreeSchema):
        self.NodeAux = [DynTreeNodeAux(node.index, node.children) for node in tree.all_nodes]
        for layer in tree.layer_to_nodes: 
            layer_index = tree.layer_to_nodes.index(layer)
            for node in layer:
                self.NodeAux[node.index].layer = layer_index
                for child in node.children or []:
                    self.NodeAux[child].parent = node.index
    
    def add_node_aux(self, node: TreeNode, layer: int):
        self.NodeAux.append(DynTreeNodeAux(node.index, layer))
        assert len(self.NodeAux) == node.index + 1, "NodeAux index is not equal to node index"

    def update_children(self, node_index: int, children: set[int]):
        if children is None:
            return
        for child in children:
            self.NodeAux[child].parent = node_index
    
    def set_parent(self, node_index: int, parent_index: int):
        self.NodeAux[node_index].parent = parent_index

    def set_valid_flag(self, node_index: int, valid_flag: bool):
        self.NodeAux[node_index].valid_flag = valid_flag

class TreeGraphDynamic(BaseGraph):
    max_workers: int = 16
    leaf_workers: int = 32
    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  
        self.config = config.graph 
        random.seed(self.config.random_seed)
        self.workspace = Workspace(config.working_dir, config.exp_name)  

        hyperplane_shape = (self.config.num_hyperplanes, config.embedding.dimensions)
        self.aux = DynAux(self.workspace, hyperplane_shape, False)

    def _create_task_for(self, func):
        def _pool_func(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(func(**params))
            loop.close()
        return _pool_func

    def _create_task_with_return(self, func):
        def _pool_func(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(func(**params))
            loop.close()
            return result
        return _pool_func
    
    def _compute_signature(self, vec):
        # logger.info(f"vec type: {type(vec)}, vec: {vec}")
        # logger.info(f"hyperplanes type: {type(self.aux.hyperplanes)}, shape: {self.aux.hyperplanes.shape if self.aux.hyperplanes is not None else 'None'}")
        projections = np.dot(vec, self.aux.hyperplanes.T)
        binary_hash = (projections > 0).astype(int)
        return int(''.join(map(str, binary_hash)), 2)


    def _analyze_bucket_distribution(self, buckets):
        size_counts = defaultdict(int)
        total_items = 0
        
        for items in buckets.values():
            size = len(items)
            size_counts[size] += 1
            total_items += size
        
        if not size_counts:
                return {}
    
        sorted_sizes = sorted(size_counts.items())

        return {
            'total_buckets': len(buckets),
            'size_distribution': dict(sorted_sizes),
            'max_size': max(size_counts.keys()) if size_counts else 0,
            'min_size': min(size_counts.keys()) if size_counts else 0,
            'avg_size': round(total_items / len(buckets), 2) if buckets else 0
        }

    def _print_bucket_stats(self, buckets, clusters):
        stats = self._analyze_bucket_distribution(buckets)
        logger.info("\n=== LSH Bucket Distribution Analysis ===")
        logger.info(f"Total Buckets: {stats['total_buckets']}")
        logger.info(f"Biggest Buckets: {stats['max_size']} Elements")
        logger.info(f"Smallest Buckets: {stats['min_size']} Elements")
        logger.info(f"Average Size: {stats['avg_size']} Elements")
        logger.info("\nBucket Size:")
        logger.info(f"{'Size':<8} | {'Count':<8} | {'Percentage':<10}")
        cumulative = 0
        total = stats['total_buckets']
        for size, count in stats['size_distribution'].items():
            percent = count / total * 100
            cumulative += percent
            logger.info(f"{size:<8} | {count:<8} | {cumulative:>8.1f}%")
        
        # Check point for each layer of clustering
        logger.info("\n=== Clustering Result ===")
        logger.info(f"Total Clusters: {len(clusters)}")
        sizes = [len(c) for c in clusters]
        logger.info(f"Cluster Distribution: Biggest {max(sizes)}, Smallest {min(sizes)}, Average {np.mean(sizes):.1f}")


    # hyper plain realization
    async def _perform_clustering(
        self, nodes: List[TreeNode], refine: bool = False
    ) -> List[np.ndarray]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])
        node_ids = np.array([node.index for node in nodes])
        n_samples = embeddings.shape[0]
        logger.info("Perform Clustering: n_samples = {n_samples}".format(n_samples=n_samples))

        # Defined in GraphConfig.py
        num_hyperplanes = self.config.num_hyperplanes
        min_size = self.config.lower_limit
        max_size = self.config.upper_limit

        # Clear current temporary files
        # self._clear_previous_clustering_files()

        # Data Preprocessing
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        n_samples, dim = embeddings.shape

        # 验证dim的维度和hyperplanes的维度是否一致
        assert dim == self.aux.hyperplanes.shape[1], "Dim is not equal to hyperplane shape"

        # Create initial hash bucket
        buckets = defaultdict(list)

        for idx, vec in enumerate(embeddings):
            signature = self._compute_signature(vec)
            self.aux.signature_map[node_ids[idx]] = signature
            buckets[signature].append(idx)
        
        # Process buckets -> Generate final clusters
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])
        queue = deque([(bid, items) for bid, items in sorted_buckets])
        clusters = []
        current_cluster = []
        labels_map = {}
        
        while queue:
            bid, items = queue.popleft()

            if len(items) >= max_size:
                clusters.append(items[:max_size])
                remaining = items[max_size:]
                if remaining:
                    queue.appendleft((bid, remaining))
                continue
            
            # Attempt to join
            if len(current_cluster) + len(items) <= max_size:
                current_cluster.extend(items)
            else:
                # Calculate Joinable Amount
                available = max_size - len(current_cluster)
                current_cluster.extend(items[:available])
                queue.appendleft((bid, items[available:]))
            
            # Size check: min_size
            if len(current_cluster) >= min_size:
                clusters.append(current_cluster)
                current_cluster = []

        # Process final cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        # Print bucket distribution
        self._print_bucket_stats(buckets, clusters)

        # Turn cluster result into easy to process labels
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels_map[idx] = cluster_id
        labels = np.array([labels_map.get(i, -1) for i in range(n_samples)])
        
        return labels



    async def _clustering(
            self, nodes: List[TreeNode], refine: bool = False
        ) -> List[List[TreeNode]]:
        
        # Perform the clustering
        clusters = await self._perform_clustering(nodes, refine)
        unique_values, inverse_indices = np.unique(clusters, return_inverse=True)
        sorted_indices = np.argsort(inverse_indices)
        clustered_indices = np.split(sorted_indices, np.cumsum(np.bincount(inverse_indices))[:-1])
        node_clusters = [[nodes[i] for i in cluster] for cluster in clustered_indices]

        return node_clusters
    
    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node(0, chunk_info.content)
        return leaf_node

    async def _create_node_without_embedding(self, layer: int, text: str, children_indices: Set[int] = None):
        # embedding = self._embed_text(text)
        logger.info(
            "Create node_id = {node_id}, children = {children}".format(node_id=self._graph.num_nodes, children=children_indices))
        new_node = self._graph.upsert_node(node_id=self._graph.num_nodes,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": None, "parent": None})
        self.aux.add_node_aux(new_node, layer)
        # 更新children的父亲
        self.aux.update_children(new_node.index, children_indices)
        return new_node

    async def _extract_entity_relationship_without_embedding(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node_without_embedding(0, chunk_info.content)
        return leaf_node

    async def _create_entity_node_without_embedding(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node_without_embedding(0, chunk_info.content)
        return leaf_node
    
    async def _extract_cluster_relationship_without_embedding(self, layer: int, cluster: List[TreeNode]) -> TreeNode:
        # Build a non-leaf node from a cluster of nodes
        summarized_text = await self._summarize_from_cluster(cluster, self.config.summarization_length)
        parent_node = await self._create_node_without_embedding(layer, summarized_text, {node.index for node in cluster})
        return parent_node

    async def _summarize_from_cluster(self, node_list: List[TreeNode], summarization_length=150) -> str:
        # Give a summarization from a cluster of nodes
        node_texts = f"\n\n".join([' '.join(node.text.splitlines()) for node in node_list])
        content = SUMMARIZE.format(context=node_texts)
        return await self.llm.aask(content, max_tokens=summarization_length)

    async def _process_layer_embeddings_and_indices(self, layer):
        current_layer = self._graph.get_layer(layer)
        texts = [node.text for node in current_layer]

        embeddings = self.embedding_model._get_text_embeddings(texts)
        start_id = self._graph.get_node_num() - len(self._graph.get_layer(layer))
        for i in range(start_id, len(self._graph.nodes)):
            self._graph.nodes[i].embedding = embeddings[i - start_id]
        for node, embedding in zip(self._graph.get_layer(layer), embeddings):
            node.embedding = embedding
            start_id += 1
    
    # Build logic
    async def _build_tree_from_leaves(self):
        for layer in range(self.config.num_layers):
            logger.info("length of layer: {length}".format(length=len(self._graph.get_layer(layer))))
            if len(self._graph.get_layer(layer)) <= self.config.reduction_dimension + 1:
                break

            self._graph.add_layer()

            clusters = await self._clustering(nodes = self._graph.get_layer(layer), refine = False)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for i in range(0, self.max_workers):
                    cluster_tasks = [pool.submit(self._create_task_for(self._extract_cluster_relationship_without_embedding), layer = layer + 1, cluster = cluster) for (j, cluster) in enumerate(clusters) if j % self.max_workers == i]
                    as_completed(cluster_tasks)

            logger.info("To batch embed current layer")
            await self._process_layer_embeddings_and_indices(self._graph.num_layers - 1)

            logger.info("Layer: {layer}".format(layer=layer))

        logger.info(self._graph.num_layers)

    async def _split_node(self, node_index: int):
        logger.info("split node: {node_index} with num of children: {children_num}".format(node_index=node_index, children_num=len(self._graph.nodes[node_index].children)))
        node = self._graph.nodes[node_index]
        children = node.children
        # split the children into multiple parts
        num_parts = (len(children)-self.config.lower_limit) // self.config.avg_size + 1
        signatures = [(self.aux.signature_map[child], child) for child in children]
        # sort the children by signature
        signatures.sort(key=lambda x: x[0])
        # split the signatures into multiple parts
        signatures_parts = np.array_split(signatures, num_parts)
        # create new nodes for each part
        # the first part is the original node
        new_children = [self._graph.nodes[child] for signature, child in signatures_parts[0]]
        node.children = {child.index for child in new_children}
        # Not updating text now
        node.text = ""
        # Set parent
        self.aux.update_children(node.index, node.children)
        self._remove_a_child_from_parent(node.index)
        self.aux.affected_entities.add(node.index)

        current_layer_index = self.aux.NodeAux[node.index].layer
        assert current_layer_index is not None, "Current layer index is None"

        for signatures_part in signatures_parts[1:]:
            new_children = {self._graph.nodes[child] for signature, child in signatures_part}
            new_node = await self._create_node_without_embedding(current_layer_index, "", {child.index for child in new_children})
            assert new_node.index !=0 , "New node index is 0"
            self.aux.affected_entities.add(new_node.index)

    def _set_node_invalid(self, node_index: int):
        node = self._graph.nodes[node_index]
        node.text = ""
        node.embedding = None
        node.children = set()
        self.aux.set_valid_flag(node.index, False)
        if self.aux.NodeAux[node_index].parent is not None:
            parent_index = self.aux.NodeAux[node_index].parent
            self._graph.nodes[parent_index].children.remove(node_index)

        
    def _merge_node(self, node_index: int, node_index_to_merge: int):
        logger.info("merge node: {node_index} and {node_index_to_merge}".format(node_index=node_index, node_index_to_merge=node_index_to_merge))
        node = self._graph.nodes[node_index]
        node_to_merge = self._graph.nodes[node_index_to_merge]
        children = node.children

        # 将node_to_merge的children加入到node中
        assert len(node.children) + len(node_to_merge.children) <= self.config.upper_limit, "Merge node will exceed the upper limit"
        for child in node_to_merge.children:
            node.children.add(child)
        # 从父节点中移除node_to_merge
        self.aux.update_children(node.index, node_to_merge.children)
        self._set_node_invalid(node_index_to_merge)
        node.text = ""
        node.embedding = None

        self.aux.affected_entities.add(node.index)
        self.aux.affected_entities.add(node_to_merge.index)


    async def _split_or_merge(self, layer: int):
        if layer == 0:
            return
        # 移除空节点
        self._remove_empty_cluster(layer)
        current_layer = self._graph.get_layer(layer)
        node_ids = [node.index for node in current_layer]

        for index in range(len(node_ids)):
            if len(self._graph.nodes[node_ids[index]].children) > self.config.upper_limit:
                await self._split_node(node_ids[index])
            elif len(self._graph.nodes[node_ids[index]].children) < self.config.lower_limit:
                if index > 0:
                    # 处理第一个元素的情况
                    if len(self._graph.nodes[node_ids[index - 1]].children) + len(self._graph.nodes[node_ids[index]].children) <= self.config.upper_limit:
                        self._merge_node(node_ids[index - 1], node_ids[index])
                        # self._remove_empty_cluster(layer)

    def _remove_a_child_from_parent(self, node_index: int):
        node_aux = self.aux.NodeAux[node_index]
        if node_aux.parent is not None:
            logger.info("node_aux.parent: {parent}".format(parent=node_aux.parent))
            self._graph.nodes[node_aux.parent].children.remove(node_index)
            self.aux.affected_entities.add(node_aux.parent)
            node_aux.parent = None

    def _remove_empty_cluster(self, layer: int):
        current_layer = self._graph.get_layer(layer)
        new_current_layer = []
        for node in current_layer:
            if len(node.children) != 0:
                new_current_layer.append(node)
            else:
                self._remove_a_child_from_parent(node.index)
        self._graph.replace_layer(layer, new_current_layer)

    async def _refine_one_layer(self, layer: int):
        logger.info("refine layer: {layer}".format(layer=layer))
        if layer != 0:
            # output the total number of nodes in the layer
            total_children = 0
            for node in self._graph.get_layer(layer):
                if node.children is not None:
                    total_children += len(node.children)
            logger.info("total children: {total_children}".format(total_children=total_children))
            
            await self._split_or_merge(layer)

        current_layer_nodes = self._graph.get_layer(layer)

        # 需要重新处理当前层的affected entities
        current_layer_affected_entities = []
        for node in current_layer_nodes:
            # 找出当前层中需要重新处理的节点
            if node.index in self.aux.affected_entities and self.aux.NodeAux[node.index].valid_flag:
                logger.info("affected node.index: {node_index}".format(node_index=node.index))
                node.embedding = None
                current_layer_affected_entities.append(node.index)
                # 如果节点有父亲，则将其从父亲中移除
                self._remove_a_child_from_parent(node.index)
        # Record current affected length
        logger.info("len of current affected entities: {length}".format(length=len(current_layer_affected_entities)))

        if len(current_layer_affected_entities) == 0:
            return 
            
        if layer != 0:
            # If not first layer, resummary
            for node_index in current_layer_affected_entities:
                logger.info("current summarize node_index: {node_index}".format(node_index=node_index))
                current_node = self._graph.nodes[node_index]
                children_list = [self._graph.nodes[child] for child in current_node.children]
                logger.info("len of children_list: {children_list}".format(children_list=len(children_list)))
                current_node.text = await self._summarize_from_cluster(children_list, self.config.summarization_length)
                # output the text of the new node
                logger.info("new node text: {text}".format(text=current_node.text))

        # update embedding
        texts = [self._graph.nodes[node_index].text for node_index in current_layer_affected_entities]
        logger.info("len of texts: {length}".format(length=len(texts)))
        embeddings = self.embedding_model._get_text_embeddings(texts)
        for node_id, embedding in zip(current_layer_affected_entities, embeddings):
            self._graph.nodes[node_id].embedding = embedding

        # Update signature
        current_layer_signatures = []
        for node in current_layer_nodes:
            if self.aux.NodeAux[node.index].valid_flag:
                if node.index not in self.aux.signature_map:
                    self.aux.signature_map[node.index] = self._compute_signature(node.embedding)
                current_layer_signatures.append((self.aux.signature_map[node.index], node.index))
        logger.info("len of current layer signatures: {length}".format(length=len(current_layer_signatures)))


        # 如果当前层是最后一层，则不需要更新cluster
        if layer == self._graph.num_layers - 1:
            if len(current_layer_nodes) < self.config.upper_limit or layer  == self.config.num_layers - 1:
                return
            else:
                # 如果当前层是最后一层，则需要更新新建一个node
                self._graph.add_layer()
                await self._create_node_without_embedding(layer+1, "", {node.index for node in current_layer_nodes})

        # sort the signatures by signature
        current_layer_signatures.sort(key=lambda x: x[0])

        # display the signatures
        if layer != 0:
            logger.info("current layer signatures: {signatures}".format(signatures=current_layer_signatures))

        # need to index in the list of signatures
        next_layer = self._graph.get_layer(layer + 1)
        parent_idx = next_layer[0].index
        assert parent_idx != None, "First parent is None"

        # New insertion complete, ready for clustering
        record_parent_idx = []
        for idx, (signature, node_index) in enumerate(current_layer_signatures):
            # record the parent index, even if the parent is None
            record_parent_idx.append(self.aux.NodeAux[node_index].parent)
            if self.aux.NodeAux[node_index].parent is not None:
                parent_idx = self.aux.NodeAux[node_index].parent
                continue

            self.aux.NodeAux[node_index].parent = parent_idx
            self._graph.nodes[parent_idx].children.add(node_index)
            self.aux.affected_entities.add(parent_idx)
        
        # 输出父亲list
        logger.info("record_parent_idx: {record_parent_idx}".format(record_parent_idx=record_parent_idx))

    def _reorder_node_id(self):
        new_node_id = 0
        new_order_map = {}
        
        # 首先创建映射，只包含有效的节点
        for node in self._graph.nodes:
            if node.embedding is not None:
                new_order_map[node.index] = new_node_id
                new_node_id = new_node_id + 1
    
        # 然后只处理有效的节点
        new_all_nodes = []
        for node in self._graph.nodes:
            if node.index in new_order_map:  # 检查节点是否在映射中
                node.index = new_order_map[node.index]
                if node.children is not None:
                    new_children = set()
                    for child in node.children:
                        if child in new_order_map:  # 检查子节点是否在映射中
                            new_children.add(new_order_map[child])
                    node.children = new_children
                new_all_nodes.append(node)
            else:
                assert node.embedding is None, "Node embedding is not None"
        logger.info("len of new_all_nodes: {length}".format(length=len(new_all_nodes)))
        logger.info("len of original nodes: {length}".format(length=len(self._graph.nodes)))
        self._graph._tree.all_nodes = new_all_nodes

    # Add information to the tree given the additional dynamic chunks
    async def _refine_graph(self, new_chunks: List[Any]): 
        is_tree_load = await self._graph.load_full_tree_graph()
        assert is_tree_load == True, "No existing tree for insertion mode!"
        self.aux.init_tree_aux(self._graph._tree)
        assert len(new_chunks) > 0, "No new chunks to insert!"

        # new_chunks = new_chunks[:10]

        # 直接遍历处理每个 chunk
        for chunk in new_chunks:
            new_node = await self._create_entity_node_without_embedding(chunk_key_pair=chunk)
            parent_idx = self.aux.NodeAux[new_node.index].parent
            logger.info("new_node.parent: {parent}".format(parent=parent_idx))
            assert new_node.index != 0, "New node index is 0" 
            self.aux.affected_entities.add(new_node.index)

        assert len(self.aux.affected_entities) == len(new_chunks), "Affected entities size is not equal to new chunks size"

        logger.info(f"Refining graph with {len(new_chunks)} new chunks")
        for layer in range(self._graph.num_layers):  # build a new layer
            logger.info("length of layer: {length}".format(length=len(self._graph.get_layer(layer))))
            await self._refine_one_layer(layer)
            for node in self._graph.get_layer(layer):
                if not self.aux.NodeAux[node.index].valid_flag:
                    continue
                if node.embedding is None:
                    logger.error(f"Node index {node.index} has no embedding")
                assert node.embedding is not None, "Node embedding is None"

        self._reorder_node_id()
                
    async def _batch_embed_and_assign(self, layer):
        current_layer = self._graph.get_layer(layer)
        texts = [node.text for node in current_layer]

        embeddings = self.embedding_model._get_text_embeddings(texts)
        start_id = self._graph.get_node_num() - len(self._graph.get_layer(layer))
        for i in range(start_id, len(self._graph.nodes)):
            self._graph.nodes[i].id = i
            self._graph.nodes[i].embedding = embeddings[i - start_id]
        for node, embedding in zip(self._graph.get_layer(layer), embeddings):
            node.embeddings = embedding
            node.index = start_id
            start_id += 1

    async def _build_graph(self, chunks: List[Any]):
        if not self.config.force:  # 只在非force模式下尝试加载
            is_load = await self._graph.load_tree_graph_from_leaves()
            if is_load:
                logger.info(f"Loaded {len(self._graph.leaf_nodes)} Leaf Embeddings")
                await self._build_tree_from_leaves()
                return
        # 如果force=True或者加载失败，则重新构建
        self._graph.clear()  # clear the storage before rebuilding
        self._graph.add_layer()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # leaf_tasks = []
            # for index, chunk in enumerate(chunks):
            #     logger.info(index)
            #     leaf_tasks.append(pool.submit(self._create_task_for(self._extract_entity_relationship), chunk_key_pair=chunk))
            for i in range(0, self.max_workers):
                leaf_tasks = [pool.submit(self._create_task_for(self._extract_entity_relationship_without_embedding), chunk_key_pair=chunk) for index, chunk in enumerate(chunks) if index % self.max_workers == i]
                as_completed(leaf_tasks)
        logger.info(len(chunks))
        logger.info(f"To batch embed leaves")
        await self._batch_embed_and_assign(self._graph.num_layers - 1)
        logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
        await self._graph.write_tree_leaves()
        await self._build_tree_from_leaves()


    @property
    def entity_metakey(self):
        return "index"