"""
Here is the colbert index for our GraphRAG
"""
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from pathlib import Path

from Core.Common.Logger import logger
import os
from typing import Any
from colbert.data import Queries
from Core.Index.BaseIndex import BaseIndex, ColbertNodeResult, ColbertEdgeResult


class ColBertIndex(BaseIndex):
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(self, config):
        super().__init__(config)
        self.index_config = ColBERTConfig(
            root=os.path.dirname(self.config.persist_path),
            experiment=os.path.basename(self.config.persist_path),
            doc_maxlen=self.config.doc_maxlen,
            query_maxlen=self.config.query_maxlen,
            nbits=self.config.nbits,
            kmeans_niters=self.config.kmeans_niters,
        )

    async def _update_index(self, elements, meta_data):

        with Run().context(
                RunConfig(nranks=self.config.ranks, experiment=self.index_config.experiment,
                          root=self.index_config.root)
        ):
            indexer = Indexer(checkpoint=self.config.model_name, config=self.index_config)            
            # Store the index
            elements = [element["content"] for element in elements]
            indexer.index(name=self.config.index_name, collection=elements, overwrite=True)
            self._index = Searcher(
                index=self.config.index_name, collection=elements, checkpoint=self.config.model_name
            )

    async def _load_index(self):
        try:
            colbert_config = ColBERTConfig.load_from_index(
                Path(self.config.persist_path) / "indexes" / self.config.index_name)
            searcher = Searcher(
                index=self.config.index_name, index_root=(Path(self.config.persist_path) / "indexes"),
                config=colbert_config
            )
            self._index = searcher
            return True
        except Exception as e:
            logger.error("Loading colbert index failed", exc_info=e)

    async def upsert(self, data: list[Any]):
        pass

    def exist_index(self):

        return os.path.exists(self.config.persist_path)

    async def retrieval(self, query, top_k=None):

        if top_k is None:
            top_k = self._get_retrieve_top_k()

        results =  tuple(self._index.search(query, k=top_k))
        return results


    async def retrieval_nodes(self, query, top_k, graph, need_score = False, tree_node = False):

        result =  ColbertNodeResult(*(await self.retrieval(query, top_k)))
        
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        else:
            return await result.get_node_data(graph, need_score)

    async def retrieval_edges(self, query, top_k, graph, need_score = False):
        results = await self.retrieval(query, top_k)
        result =  ColbertEdgeResult(*results)
        
        return await result.get_edge_data(graph, need_score)
    async def retrieval_batch(self, queries, top_k=None):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        try:
            if isinstance(queries, str):
                queries = Queries(path=None, data={0: queries})
            elif not isinstance(queries, Queries):
                queries = Queries(data=queries)

            return self._index.search_all(queries, k=top_k).data
        except Exception as e:
            logger.exception(f"fail to search {queries} for {e}")
            return []

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        # Stores the index for Colbert-index upon its creation.
        pass

    def _get_index(self):
        return ColBertIndex(self.config)

    async def _similarity_score(self, object_q, object_d):
        encoded_q = self._index.encode(object_q, full_length_search=False)
        encoded_d = self._index.checkpoint.docFromText(object_d).float()
        real_score = encoded_q[0].matmul(encoded_d[0].T).max(
            dim=1).values.sum().detach().cpu().numpy()
        return real_score

    async def get_max_score(self, queries_):
        assert isinstance(queries_, list)
        encoded_query = self._index.encode(queries_, full_length_search=False)
        encoded_doc = self._index.checkpoint.docFromText(queries_).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return max_score
