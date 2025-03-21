import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from llama_index.legacy.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.legacy.schema import BaseNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore.types import RefDocInfo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFIndex(BaseIndex[IndexDict]):

    def __init__(self
    ) -> None:
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None




    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError("TFIDFIndex does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("TFIDFIndex does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError("TFIDFIndex does not support deletion yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        raise NotImplementedError("TFIDFStoreIndex does not support ref_doc_info.")

    def _build_index_from_nodes(self, nodes: List[BaseNode]):
        raise NotImplementedError("TFIDFStoreIndex does not support insertion yet.")
    def _build_index_from_list(self, docs_list: List[str]):

        self.tfidf_matrix = self.vectorizer.fit_transform(docs_list)



    def persist(self, persist_dir: str) -> None:
        # Check if the destination directory exists
        if os.path.exists(persist_dir):
            # Remove the existing destination directory
            shutil.rmtree(persist_dir)

        shutil.copytree(
            Path(self.index_path) / self.index_name, Path(persist_dir) / self.index_name
        )
        self._storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def load_from_disk(cls, persist_dir: str, index_name: str = ""):
        raise NotImplementedError("TFIDFStoreIndex does not support load_from_disk yet.")
      

    def query(self, query_str: str, top_k: int = 10):
        """
        Retrieval the Tf-Idf

        Returns: list of search results.
        """
        query_emb = self.vectorizer.transform([query_str])
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()

        top_k = min(top_k, len(cosine_sim))
        idxs = cosine_sim.argsort()[::-1][:top_k]
        
        return idxs
        

    def query_batch(self, queries, top_k):
        raise NotImplementedError("TFIDFStoreIndex does not support query_batch yet.")
        