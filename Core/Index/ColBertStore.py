import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from llama_index.legacy.data_structs.data_structs import IndexDict
from llama_index.legacy.schema import BaseNode, NodeWithScore

from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore.types import RefDocInfo
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig

class ColbertIndex(BaseIndex[IndexDict]):
    """
    Store for ColBERT v2 with PLAID indexing.

    ColBERT is a neural retrieval method that tends to work
    well in a zero-shot setting on out of domain datasets, due
    to it's use of token-level encodings (rather than sentence or
    chunk level)

    Parameters:

    index_path: directory containing PLAID index files.
    model_name: ColBERT hugging face model name.
        Default: "colbert-ir/colbertv2.0".
    show_progress: whether to show progress bar when building index.
        Default: False. noop for ColBERT for now.
    nbits: number of bits to quantize the residual vectors. Default: 2.
    kmeans_niters: number of kmeans clustering iterations. Default: 1.
    gpus: number of GPUs to use for indexing. Default: 0.
    rank: number of ranks to use for indexing. Default: 1.
    doc_maxlen: max document length. Default: 120.
    query_maxlen: max query length. Default: 60.
    kmeans_niters: number of kmeans iterations. Default: 4.

    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        persist_path: str = "storage/colbert_index",
        index_name: str = "",
        nbits: int = 2,
        gpus: int = 0,
        ranks: int = 1,
        doc_maxlen: int = 120,
        query_maxlen: int = 60,
        kmeans_niters: int = 4,
        store: Optional[Searcher] = None
    ) -> None:
        self.model_name = model_name
        self.index_path = persist_path
        self.index_name = index_name
        self.nbits = nbits
        self.gpus = gpus
        self.ranks = ranks
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.kmeans_niters = kmeans_niters
        self.store = store




    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError("ColbertStoreIndex does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("ColbertStoreIndex does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError("ColbertStoreIndex does not support deletion yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        raise NotImplementedError("ColbertStoreIndex does not support ref_doc_info.")

    def _build_index_from_nodes(self, nodes: List[BaseNode]):
        raise NotImplementedError("ColbertStoreIndex does not support insertion yet.")
    def _build_index_from_list(self, docs_list: List[str]) -> IndexDict:
        """Generate a PLAID index from the ColBERT checkpoint via its hugging face
        model_name.
        """


    
    def persist(self, persist_dir: str) -> None:
        # Check if the destination directory exists
        if os.path.exists(persist_dir):
            # Remove the existing destination directory
            shutil.rmtree(persist_dir)

        # Copy PLAID vectors
        shutil.copytree(
            Path(self.index_path) / self.index_name, Path(persist_dir) / self.index_name
        )
        self._storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def load_from_disk(cls, persist_dir: str, index_name: str = "") -> "ColbertIndex":
 

        colbert_config = ColBERTConfig.load_from_index(Path(persist_dir) / index_name)
        searcher = Searcher(
            index=index_name, index_root=persist_dir, config=colbert_config
        )
        return cls(store = searcher)
      

    def query(self, query_str: str, top_k: int = 10):
        """
        Retrieval the Colbert v2.

        Returns: list of search results.
        """
        doc_ids, _, scores = self.store.search(text=query_str, k=top_k)

   

        nodes_with_score = []


        return nodes_with_score
    def query_batch(self, queries, top_k):
        ranking = self.store.search_all(queries, k = top_k)
        return ranking
        
    @property
    def index_searcher(self):
        return self.store