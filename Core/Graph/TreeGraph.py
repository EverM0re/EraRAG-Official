from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Prompt.RaptorPrompt import SUMMARIZE
from Core.Storage.TreeGraphStorage import TreeGraphStorage
from Core.Schema.TreeSchema import TreeNode

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from typing import List, Set, Any

Embedding = List[float]

import numpy as np
import umap
import random
from sklearn.mixture import GaussianMixture

class TreeGraph(BaseGraph):
    max_workers: int = 16
    leaf_workers: int = 32
    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  # Tree index
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  # Embedding model
        self.config = config.graph # Only keep the graph config
        random.seed(self.config.random_seed)

    async def _GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        if  len(embeddings) >  self.config.threshold_cluster_num:
            max_clusters  = len(embeddings) // 100
            n_clusters = np.arange(max_clusters - 1, max_clusters)
        else:
            max_clusters = min(50, len(embeddings))
            n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            # logger.info("GMM Cluster n = {n}".format(n=n))
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]

        gm = GaussianMixture(n_components=optimal_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, optimal_clusters

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

    async def _process_cluster(self, i, global_clusters, embeddings, dim, threshold):
        logger.info("Processing cluster i={i}", i=i)
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            return
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = umap.UMAP(
                n_neighbors=10, n_components=dim, metric=self.config.cluster_metric
            ).fit_transform(global_cluster_embeddings_)
            # import pdb
            # pdb.set_trace()
            local_clusters, n_local_clusters = await self._GMM_cluster(
                reduced_embeddings_local, threshold
            )

        return i, local_clusters, n_local_clusters

    async def _perform_clustering(
        self, embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
    ) -> List[np.ndarray]:
        logger.info("Length of embeddings: {length}".format(length=len(embeddings)))
        reduced_embeddings_global = umap.UMAP(
            n_neighbors=int((len(embeddings) - 1) ** 0.5), n_components=min(dim, len(embeddings) -2), metric=self.config.cluster_metric
        ).fit_transform(embeddings)

        logger.info("Finished UMAP")
        global_clusters, n_global_clusters = await self._GMM_cluster(
            reduced_embeddings_global, threshold
        )
        
        logger.info("Finished GMM clustering, {n} clusters".format(n=n_global_clusters))

        # import pdb
        # pdb.set_trace()

        if verbose:
            logger.info(f"Global Clusters: {n_global_clusters}")

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        completed_list = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for j in range(0, self.max_workers):
                cluster_tasks = [pool.submit(self._create_task_with_return(self._process_cluster),
                                          i=i,
                                          global_clusters=global_clusters,
                                          embeddings=embeddings,
                                          dim=dim,
                                          threshold=threshold) for i in range(n_global_clusters) if i % self.max_workers == j]
                completed_list.extend(as_completed(cluster_tasks))
                time.sleep(4)

        for task in completed_list:
            i, local_clusters, n_local_clusters = task.result()
            global_indices = np.where(np.array([i in gc for gc in global_clusters]))[0]
            # global_cluster_embeddings_ = embeddings[global_indices]
            for j in range(n_local_clusters):
                # tmp = [j in lc for lc in local_clusters]
                # import pdb
                # pdb.set_trace()
                indices = global_indices[np.array([j in lc for lc in local_clusters])]
                # local_cluster_embeddings_ = global_cluster_embeddings_[
                #     np.array([j in lc for lc in local_clusters])
                # ]
                # indices = np.where(
                #     (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                # )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        logger.info(f"Total Clusters: {total_clusters}")
        return all_local_clusters


    async def _clustering(self, nodes: List[TreeNode], max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose, depth: int = 0) -> List[List[TreeNode]]:
        logger.info("Clustering: dep = {depth}", depth=depth)
        if depth >= 20: return [nodes]
        
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])


        # Perform the clustering
        clusters = await self._perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        if len(np.unique(np.concatenate(clusters))) == 1:
            logger.info("Only one cluster length = {len}, return".format(len = len(nodes)))
            return [nodes]

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster and len(cluster_nodes) > self.config.reduction_dimension + 1:
                if verbose:
                    logger.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
    
                node_clusters.extend(
                    await self._clustering(
                        cluster_nodes, max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose, depth + 1
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

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

            clusters = await self._clustering(
                nodes = self._graph.get_layer(layer),
                max_length_in_cluster =  self.config.max_length_in_cluster,
                tokenizer = self.ENCODER,
                reduction_dimension = self.config.reduction_dimension,
                threshold = self.config.threshold,
                verbose = self.config.verbose,
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for i in range(0, self.max_workers):
                    cluster_tasks = [pool.submit(self._create_task_for(self._extract_cluster_relationship_without_embedding), layer = layer + 1, cluster = cluster) for (j, cluster) in enumerate(clusters) if j % self.max_workers == i]
                    # self._run_tasks(cluster_tasks)
                    as_completed(cluster_tasks)
                    time.sleep(3)

            logger.info("To batch embed current layer")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)
            # for cluster in clusters:  # for each cluster, create a new node
            #     await self._extract_cluster_relationship(layer + 1, cluster)

            logger.info("Layer: {layer}".format(layer=layer))
            # logger.info(self._graph.get_layer(layer + 1))

        logger.info(self._graph.num_layers)
        

    async def _build_graph(self, chunks: List[Any]):
        if self.config.build_tree_from_leaves:
            await self._graph.load_tree_graph_from_leaves()
            logger.info(f"Loaded {len(self._graph.leaf_nodes)} Leaf Embeddings")
        else:
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
                    # time.sleep(2)
            logger.info(len(chunks))
            logger.info(f"To batch embed leaves")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)
            logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
            await self._graph.write_tree_leaves()
        await self._build_tree_from_leaves()
        
    @property
    def entity_metakey(self):
        return "index"
