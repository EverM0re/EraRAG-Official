from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import numpy as np
import asyncio
from collections import defaultdict
from Core.Common.Utils import truncate_list_by_token_size
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Retriever.RetrieverFactory import register_retriever_method


class EntityRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["ppr", "vdb", "from_relation", "tf_df", "all", "by_neighbors", "link_entity", "get_all", "from_relation_by_agent"]
        self.type = "entity"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="entity", method_name="ppr")
    async def _find_relevant_entities_by_ppr(self, query, seed_entities: list[dict], link_entity=False):

        if len(seed_entities) == 0:
            return None
        if link_entity:
            seed_entities = await self.link_query_entities(seed_entities)
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        ppr_node_matrix = await self._run_personalized_pagerank(query, seed_entities)
        topk_indices = np.argsort(ppr_node_matrix)[-self.config.top_k:]
        nodes = await self.graph.get_node_by_indices(topk_indices)

        return nodes, ppr_node_matrix

    @register_retriever_method(type="entity", method_name="vdb")
    async def _find_relevant_entities_vdb(self, seed, tree_node=False, top_k=None):
        try:
            if top_k is None:
                top_k = self.config.top_k
                
            # Custom search logic
            if self.config.custom_search and tree_node:
                return await self._custom_tree_search(seed, top_k)
            
            # Original logic
            node_datas = await self.entities_vdb.retrieval_nodes(query=seed, top_k=top_k, graph=self.graph,
                                                                 tree_node=tree_node)

            if not len(node_datas):
                return None
                
            valid_node_datas = []
            max_node_id = len(self.graph._graph.nodes) - 1
            for node in node_datas:
                if node is None:
                    continue
                if tree_node:
                    if 0 <= node.index <= max_node_id:
                        valid_node_datas.append(node)
                    else:
                        logger.warning(f"Invalid node index {node.index} found in vector index, max valid index is {max_node_id}")
                else:
                    entity_name = node.get("entity_name")
                    if entity_name and await self.graph.has_node(entity_name):
                        valid_node_datas.append(node)
                    else:
                        logger.warning(f"Invalid entity {entity_name} found in vector index")
            
            if not valid_node_datas:
                logger.warning("No valid nodes found after filtering")
                return None
                
            node_datas = valid_node_datas
            if tree_node:
                node_datas = [node.text for node in node_datas]
                return node_datas
                
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip(node_datas, node_degrees)
                if n is not None
            ]

            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")

    async def _custom_tree_search(self, seed, top_k):
        """
        Custom search that retrieves a portion from leaf nodes and the rest from non-leaf nodes
        """
        try:
            # Get all nodes from the tree
            all_nodes = await self.entities_vdb.retrieval_nodes(query=seed, top_k=self.graph.num_nodes, graph=self.graph, tree_node=True)
            
            if not all_nodes:
                return None
                
            # Separate leaf nodes and non-leaf nodes
            leaf_nodes = []
            non_leaf_nodes = []
            
            # Get leaf nodes from tree structure
            leaf_node_indices = set()
            if hasattr(self.graph, 'tree') and self.graph.tree and hasattr(self.graph.tree, 'leaf_nodes'):
                # Get indices of leaf nodes (layer 0)
                for leaf_node in self.graph.tree.leaf_nodes:
                    if hasattr(leaf_node, 'index'):
                        leaf_node_indices.add(leaf_node.index)
            
            for node in all_nodes:
                if node is None:
                    continue
                    
                # Check if node is a leaf node
                if hasattr(node, 'index') and node.index in leaf_node_indices:
                    leaf_nodes.append(node)
                else:
                    non_leaf_nodes.append(node)
            
            # Calculate how many nodes to retrieve from each type
            leaf_count = int(top_k * self.config.portion)
            non_leaf_count = top_k - leaf_count
            
            # Ensure we don't exceed available nodes
            leaf_count = min(leaf_count, len(leaf_nodes))
            non_leaf_count = min(non_leaf_count, len(non_leaf_nodes))
            
            # Combine results
            result_nodes = []
            result_nodes.extend(leaf_nodes[:leaf_count])
            result_nodes.extend(non_leaf_nodes[:non_leaf_count])
            
            # Extract text from nodes
            node_texts = [node.text for node in result_nodes if hasattr(node, 'text')]
            
            logger.info(f"Custom search: retrieved {len(leaf_nodes[:leaf_count])} leaf nodes and {len(non_leaf_nodes[:non_leaf_count])} non-leaf nodes")
            
            return node_texts
            
        except Exception as e:
            logger.exception(f"Failed in custom tree search: {e}")
            # Fallback to original method
            return await self.entities_vdb.retrieval_nodes(query=seed, top_k=top_k, graph=self.graph, tree_node=True)

    @register_retriever_method(type="entity", method_name="tf_df")
    async def _find_relevant_entities_tf_df(self, seed, corpus, top_k, candidates_idx):
        try:
            graph_nodes = list(await self.graph.get_nodes())
            corpus = dict({id: (await self.graph.get_node(id))['description'] for id in graph_nodes})
            candidates_idx = list(id for id in graph_nodes)
            index = TFIDFIndex()

            index._build_index_from_list([corpus[_] for _ in candidates_idx])
            idxs = index.query(query_str=seed, top_k=top_k)

            new_candidates_idx = [candidates_idx[_] for _ in idxs]
            cur_contexts = [corpus[_] for _ in new_candidates_idx]

            return cur_contexts, new_candidates_idx

        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")

    @register_retriever_method(type="entity", method_name="all")
    async def _find_relevant_entities_all(self, key):
        graph_nodes = list(await self.graph.get_nodes())
        corpus = dict({id: (await self.graph.get_node(id))[key] for id in graph_nodes})
        candidates_idx = list(id for id in graph_nodes)
        return corpus, candidates_idx

    @register_retriever_method(type="entity", method_name="link_entity")
    async def _link_entities(self, query_entities):
        # For entity link, we only consider the top-ranked entity
        entities = await asyncio.gather(
            *[self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph=self.graph) for query_entity in
              query_entities]
        )
        entities = list(map(lambda x: x[0], entities))
        return entities

    @register_retriever_method(type="entity", method_name="get_all")
    async def _get_all_entities(self):
        nodes = await self.graph.nodes_data()
        return nodes

    @register_retriever_method(type="entity", method_name="from_relation_by_agent")
    async def _find_relevant_entities_by_relationships_agent(self, query: str, total_entity_relation_list: list[dict],
                                                             total_relations_dict: defaultdict[list], width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            total_entity_relation_list: list,  whose element is {"entity": entity_name, "relation": relation, "score": score, "head": bool}
            total_relations_dict: defaultdict[list], key is (src, rel), value is tar
        Returns:
            flag: bool,  indicator that shows whether to reason or not
            relations, heads
            cluster_chain_of_entities: list[list], reasoning paths
            candidates: list[str], entity candidates
            relations: list[str], related relation
            heads: list[bool]
        """
        # ✅
        try:
            from Core.Prompt.TogPrompt import score_entity_candidates_prompt
            total_candidates = []
            total_scores = []
            total_relations = []
            total_topic_entities = []
            total_head = []

            for index, entity in enumerate(total_entity_relation_list):
                candidate_list = total_relations_dict[(entity["entity"], entity["relation"])]

                # score these candidate entities
                if len(candidate_list) == 1:
                    scores = [entity["score"]]
                elif len(candidate_list) == 0:
                    scores = [0.0]
                else:
                    # agent
                    prompt = score_entity_candidates_prompt.format(query, entity["relation"]) + '; '.join(
                        candidate_list) + ';' + '\nScore: '
                    result = await self.llm.aask(msg=[
                        {"role": "user",
                         "content": prompt}
                    ])

                    # clean
                    import re
                    scores = re.findall(r'\d+\.\d+', result)
                    scores = [float(number) for number in scores]
                    if len(scores) != len(candidate_list):
                        logger.info("All entities are created with equal scores.")
                        scores = [1 / len(candidate_list)] * len(candidate_list)

                # update
                if len(candidate_list) == 0:
                    candidate_list.append("[FINISH]")
                candidates_relation = [entity['relation']] * len(candidate_list)
                topic_entities = [entity['entity']] * len(candidate_list)
                head_num = [entity['head']] * len(candidate_list)
                total_candidates.extend(candidate_list)
                total_scores.extend(scores)
                total_relations.extend(candidates_relation)
                total_topic_entities.extend(topic_entities)
                total_head.extend(head_num)

            # entity_prune according to width
            zipped = list(zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores))
            sorted_zipped = sorted(zipped, key=lambda x: x[4], reverse=True)
            relations = list(map(lambda x: x[0], sorted_zipped))[:width]
            candidates = list(map(lambda x: x[1], sorted_zipped))[:width]
            topics = list(map(lambda x: x[2], sorted_zipped))[:width]
            heads = list(map(lambda x: x[3], sorted_zipped))[:width]
            scores = list(map(lambda x: x[4], sorted_zipped))[:width]

            # merge and output
            merged_list = list(zip(relations, candidates, topics, heads, scores))
            filtered_list = [(rel, ent, top, hea, score) for rel, ent, top, hea, score in merged_list if score != 0]
            if len(filtered_list) == 0:
                return False, []
            else:
                return True, filtered_list
        except Exception as e:
            logger.exception(f"Failed to find relevant entities by relation agent: {e}")

    @register_retriever_method(type="entity", method_name="from_relation")
    async def _find_relevant_entities_by_relationships(self, seed):
        entity_names = set()
        for e in seed:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])

        node_datas = await asyncio.gather(
            *[self.graph.get_node(entity_name) for entity_name in entity_names]
        )

        node_degrees = await asyncio.gather(
            *[self.graph.node_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
        ]

        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )

        return node_datas

    @register_retriever_method(type="entity", method_name="by_neighbors")
    async def _find_relevant_entities_by_neighbor(self, seed):
        return list(await self.graph.get_neighbors(seed))
