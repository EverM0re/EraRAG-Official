import asyncio
import numpy as np
import pickle
import json
import os
import random

from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Prompt.RaptorPrompt import SUMMARIZE
from Core.Storage.TreeGraphStorage import TreeGraphStorage
from Core.Schema.TreeSchema import TreeNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Any

Embedding = List[float]
BUCKET_MAP_FILE = "/ssddata/zhengjun/temp/bucket_map.json"
EMBEDDINGS_FILE = "/ssddata/zhengjun/temp/embeddings.npy"
HYPERPLANE_FILE = "/ssddata/zhengjun/temp/hyperplanes.npy"


class TreeGraphDynamic(BaseGraph):
    max_workers: int = 16
    leaf_workers: int = 32
    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  # Tree index
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  # Embedding model
        self.config = config.graph # Only keep the graph config
        random.seed(self.config.random_seed)


    def _load_bucket_map(self):
        if os.path.exists(BUCKET_MAP_FILE):
            with open(BUCKET_MAP_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_bucket_map(self):
        with open(BUCKET_MAP_FILE, "w") as f:
            json.dump(self.bucket_map, f)

    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            return np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        return {}

    def _save_embeddings(self):
        np.save(EMBEDDINGS_FILE, self.embedding_cache)

    def _save_hyperplanes(self, hyperplanes: np.ndarray):
        np.save(HYPERPLANE_FILE, hyperplanes)

    def _load_hyperplanes(self) -> np.ndarray:
        if os.path.exists(HYPERPLANE_FILE):
            return np.load(HYPERPLANE_FILE)


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

    # hyper plain realization
    async def _perform_clustering(
        self, embeddings: np.ndarray
    ) -> List[np.ndarray]:
        
        n_samples = embeddings.shape[0]
        logger.info("Perform Clustering: n_samples = {n_samples}".format(n_samples=n_samples))

        # Defined in GraphConfig.py
        num_hyperplanes = self.config.num_hyperplanes
        min_size = self.config.lower_limit
        max_size = self.config.upper_limit

        # Data Preprocessing
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        n_samples, dim = embeddings.shape

        # Generate hash function
        hyperplanes = np.random.randn(num_hyperplanes, dim)
        self._save_hyperplanes(hyperplanes)
        logger.info(f"✅ Hyperplanes saved.")

        def get_bucket_id(vec):
            projections = np.dot(vec, hyperplanes.T)
            binary_hash = (projections > 0).astype(int)
            return int(''.join(map(str, binary_hash)), 2)
        
        def analyze_bucket_distribution(buckets):
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


        # See the stat of each layer of bucketing
        def print_bucket_stats(buckets):
            stats = analyze_bucket_distribution(buckets)
            print("\n=== LSH桶分布分析 ===")
            print(f"总桶数: {stats['total_buckets']}")
            print(f"最大桶: {stats['max_size']} 元素")
            print(f"最小桶: {stats['min_size']} 元素")
            print(f"平均桶: {stats['avg_size']} 元素")
            print("\n桶大小分布表:")
            print(f"{'大小':<8} | {'桶数量':<8} | {'累积占比':<10}")
            cumulative = 0
            total = stats['total_buckets']
            for size, count in stats['size_distribution'].items():
                percent = count / total * 100
                cumulative += percent
                print(f"{size:<8} | {count:<8} | {cumulative:>8.1f}%")


        # Create initial hash bucket
        buckets = defaultdict(list)

        for idx, vec in enumerate(embeddings):
            bucket_id = get_bucket_id(vec)
            self.embedding_cache[idx] = vec
            self.bucket_map[idx] = bucket_id
            buckets[bucket_id].append(idx)
        
        # Print bucket distribution
        print_bucket_stats(buckets)
        
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
        
        # Check point for each layer of clustering
        print("\n=== 最终聚类结果 ===")
        print(f"总聚类数: {len(clusters)}")
        sizes = [len(c) for c in clusters]
        print(f"聚类大小分布: 最大 {max(sizes)}, 最小 {min(sizes)}, 平均 {np.mean(sizes):.1f}")

        # Turn cluster result into easy to process labels
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels_map[idx] = cluster_id
        labels = np.array([labels_map.get(i, -1) for i in range(n_samples)])
        
        #  Save hyperplane, embedding and bucket data
        self._save_hyperplanes(hyperplanes)
        self.embedding_cache = {idx: vec for idx, vec in enumerate(embeddings)}
        self._save_embeddings()
        self.bucket_map = {idx: get_bucket_id(vec) for idx, vec in enumerate(embeddings)}
        self._save_bucket_map()
        logger.info("✅ 已保存 hyperplanes、embeddings、bucket map")

        return labels


    async def _clustering(
            self, nodes: List[TreeNode]
        ) -> List[List[TreeNode]]:
        
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])
        # Perform the clustering
        clusters = await self._perform_clustering(embeddings)

        # todo：进行桶的调整
        unique_values, inverse_indices = np.unique(clusters, return_inverse=True)
        sorted_indices = np.argsort(inverse_indices)
        clustered_indices = np.split(sorted_indices, np.cumsum(np.bincount(inverse_indices))[:-1])
        node_clusters = [[nodes[i] for i in cluster] for cluster in clustered_indices]

        return node_clusters

    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)

    async def _create_node(self, layer: int, text: str, children_indices: Set[int] = None):
        embedding = self._embed_text(text)
        node_id = self._graph.num_nodes  # Give it an index
        logger.info(
            "Create node_id = {node_id}, children = {children}".format(node_id=node_id, children=children_indices))
        return self._graph.upsert_node(node_id=node_id,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": embedding})

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node(0, chunk_info.content)
        return leaf_node

    async def _extract_cluster_relationship(self, layer: int, cluster: List[TreeNode]) -> TreeNode:
        # Build a non-leaf node from a cluster of nodes
        summarized_text = await self._summarize_from_cluster(cluster, self.config.summarization_length)
        parent_node = await self._create_node(layer, summarized_text, {node.index for node in cluster})
        return parent_node

    async def _create_node_without_embedding(self, layer: int, text: str, children_indices: Set[int] = None):
        # embedding = self._embed_text(text)
        logger.info(
            "Create node_id = unassigned, children = {children}".format(node_id=0, children=children_indices))
        return self._graph.upsert_node(node_id=0,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": []})

    async def _extract_entity_relationship_without_embedding(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
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

    async def _build_tree_from_leaves(self):
        for layer in range(self.config.num_layers):  # build a new layer
            logger.info("length of layer: {length}".format(length=len(self._graph.get_layer(layer))))
            if len(self._graph.get_layer(layer)) <= self.config.reduction_dimension + 1:
                break

            self._graph.add_layer()

            clusters = await self._clustering(nodes = self._graph.get_layer(layer))

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for i in range(0, self.max_workers):
                    cluster_tasks = [pool.submit(self._create_task_for(self._extract_cluster_relationship_without_embedding), layer = layer + 1, cluster = cluster) for (j, cluster) in enumerate(clusters) if j % self.max_workers == i]
                    as_completed(cluster_tasks)

            logger.info("To batch embed current layer")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)


            logger.info("Layer: {layer}".format(layer=layer))

        logger.info(self._graph.num_layers)
        

    # Construct the graph from scratch
    async def _build_graph(self, chunks: List[Any]):
        is_load = await self._graph.load_tree_graph_from_leaves()
        if is_load:
            logger.info(f"Loaded {len(self._graph.leaf_nodes)} Leaf Embeddings")
        else:
            self._graph.clear()  # clear the storage before rebuilding
            self._graph.add_layer()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for i in range(0, self.max_workers):
                    leaf_tasks = [pool.submit(self._create_task_for(self._extract_entity_relationship_without_embedding), chunk_key_pair=chunk) for index, chunk in enumerate(chunks) if index % self.max_workers == i]
                    as_completed(leaf_tasks)
            logger.info(len(chunks))
            logger.info(f"To batch embed leaves")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)
            logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
            await self._graph.write_tree_leaves()
        await self._build_tree_from_leaves()
    

    # Add information to the tree given the additional dynamic chunks
    async def _refine_graph(
            self, new_chunks: List[Any]
        ) -> List[int]:

        logger.info(f"Refining graph with {len(new_chunks)} new chunks")

        # load bucket id and embedding data
        self.bucket_map = self._load_bucket_map()
        self.embedding_cache = self._load_embeddings()

        new_node_indices = []
        new_embeddings = []
        for chunk in new_chunks:
            node_id = self._graph.num_nodes
            embedding = self._embed_text(chunk["text"])
            new_node_indices.append(node_id)
            new_embeddings.append(embedding)
            self.embedding_cache[node_id] = embedding

        new_embeddings = np.array(new_embeddings)
        bucket_changes, labels = await self._map_to_existing_buckets(new_embeddings, new_node_indices)

        affected_buckets = set(bucket_changes.keys())
        logger.info(f"Affected buckets: {affected_buckets}")
        await self._recluster_affected_nodes(affected_buckets)

        return labels 

    async def _map_to_existing_buckets(
            self, new_embeddings: np.ndarray, new_node_indices: List[int]
        ) -> Tuple[Dict[int, List[int]], List[int]]:

        # Load exsisting hyperplane data
        hyperplanes = self._load_hyperplanes()

        bucket_changes = defaultdict(list)
        labels = []

        def get_bucket_id(vec):
            projections = np.dot(vec, hyperplanes.T)
            binary_hash = (projections > 0).astype(int)
            return int(''.join(map(str, binary_hash)), 2)

        for node_idx, vec in zip(new_node_indices, new_embeddings):
            bucket_id = get_bucket_id(vec)
            self.bucket_map[str(node_idx)] = bucket_id
            bucket_changes[bucket_id].append(node_idx)
            labels.append(bucket_id)  # label = bucket_id

        self._save_bucket_map()
        return bucket_changes, labels


    async def _recluster_affected_nodes(self, affected_buckets: Set[int]):
        affected_nodes = []
        for bucket_id in affected_buckets:
            affected_nodes.extend(self.bucket_map[bucket_id]) 

        # 重新 summary 父节点
        updated_parents = set()
        for node in affected_nodes:
            parent_id = self._graph.get_parent(node)
            if parent_id:
                updated_parents.add(parent_id)
        
        # 重新对受影响的父节点进行 summary
        for parent_id in updated_parents:
            children = self._graph.get_children(parent_id)
            new_summary = await self._summarize_from_cluster(children, self.config.summarization_length)
            self._graph.update_summary(parent_id, new_summary)

        # 逐层向上
        if updated_parents:
            await self._recluster_affected_nodes(updated_parents)

        
    @property
    def entity_metakey(self):
        return "index"