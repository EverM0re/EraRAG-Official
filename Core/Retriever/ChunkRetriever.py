from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import asyncio
import numpy as np
from Core.Common.Utils import truncate_list_by_token_size, split_string_by_multi_markers, min_max_normalize, to_str_by_maxtokens
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Common.Constants import GRAPH_FIELD_SEP,TOKEN_TO_CHAR_RATIO
class ChunkRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["entity_occurrence", "ppr", "from_relation", "aug_ppr"]
        self.type = "chunk"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="chunk", method_name="entity_occurrence")
    async def _find_relevant_chunks_from_entity_occurrence(self, node_datas: list[dict]):

        if len(node_datas) == 0:
            return None
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in node_datas
        ]
        edges = await asyncio.gather(
            *[self.graph.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])
        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await asyncio.gather(
            *[self.graph.get_node(e) for e in all_one_hop_nodes]
        )
        all_one_hop_text_units_lookup = {
            k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None
        }
        all_text_units_lookup = {}
        for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
            for c_id in this_text_units:
                if c_id in all_text_units_lookup:
                    continue
                relation_counts = 0
                for e in this_edges:
                    if (
                            e[1] in all_one_hop_text_units_lookup
                            and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
                all_text_units_lookup[c_id] = {
                    "data": await self.doc_chunk.get_data_by_key(c_id),
                    "order": index,
                    "relation_counts": relation_counts,
                }
        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        # for node_data in node_datas:
        all_text_units = sorted(
            all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
        )
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size=self.config.local_max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units

    @register_retriever_method(type="chunk", method_name="from_relation")
    async def _find_relevant_chunks_from_relationships(self, seed: list[dict]):
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in seed
        ]

        all_text_units_lookup = {}

        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await self.doc_chunk.get_data_by_key(c_id),
                        "order": index,
                    }

        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        all_text_units = sorted(all_text_units, key=lambda x: x["order"])
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size=self.config.max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units

    @register_retriever_method(type="chunk", method_name="ppr")
    async def _find_relevant_chunks_by_ppr(self, query, seed_entities: list[dict], link_entity=False):
        # 
        if link_entity:
            seed_entities = await self.link_query_entities(seed_entities) 
        entity_to_edge_mat = await self.entities_to_relationships.get()
        relationship_to_chunk_mat = await self.relationships_to_chunks.get()
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        if len(seed_entities) == 0:
            node_ppr_matrix = np.ones(self.graph.node_num) / self.graph.node_num

        else:
            node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
        edge_prob = entity_to_edge_mat.T.dot(node_ppr_matrix)
        ppr_chunk_prob = relationship_to_chunk_mat.T.dot(edge_prob)
        ppr_chunk_prob = min_max_normalize(ppr_chunk_prob)
        # Return top k documents
        sorted_doc_ids = np.argsort(ppr_chunk_prob, kind='mergesort')[::-1]
        sorted_scores = ppr_chunk_prob[sorted_doc_ids]
        top_k = self.config.top_k
        sorted_docs = await self.doc_chunk.get_data_by_indices(sorted_doc_ids[:top_k])
        return sorted_docs, sorted_scores[:top_k]

    @register_retriever_method(type="chunk", method_name="aug_ppr")
    async def _find_relevant_chunks_by_ppr(self, query, seed_entities: list[dict]):
        # 
        entity_to_edge_mat = await self.entities_to_relationships.get()
        relationship_to_chunk_mat = await self.relationships_to_chunks.get()
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
        edge_prob = entity_to_edge_mat.T.dot(node_ppr_matrix)
        ppr_chunk_prob = relationship_to_chunk_mat.T.dot(edge_prob)
        # Return top k documents
        sorted_doc_ids = np.argsort(ppr_chunk_prob, kind='mergesort')[::-1]
        sorted_entity_ids = np.argsort(node_ppr_matrix, kind='mergesort')[::-1]
        sorted_relationship_ids = np.argsort(edge_prob, kind='mergesort')[::-1]

        sorted_docs = await self.doc_chunk.get_data_by_indices(sorted_doc_ids)
        sorted_entities = await self.graph.get_node_by_indices(sorted_entity_ids)
        sorted_relationships = await self.graph.get_edge_by_indices(sorted_relationship_ids)
        sorted_entities, sorted_relationships, sorted_docs
        return to_str_by_maxtokens(max_chars={
                "entities": self.config.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
                "relationships": self.config.relationships_max_tokens * TOKEN_TO_CHAR_RATIO,
                "chunks": self.config.local_max_token_for_text_unit * TOKEN_TO_CHAR_RATIO,
            }, entities=sorted_entities, relationships=sorted_relationships, chunks=sorted_docs)
