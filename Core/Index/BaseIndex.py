import os
from abc import ABC, abstractmethod
from Core.Common.Utils import clean_storage
from Core.Common.Logger import logger
from Core.Schema.VdbResult import * 

class BaseIndex(ABC):
    def __init__(self, config):
        self.config = config
        self._index = None

    async def build_index(self, elements, meta_data, force=False):
        logger.info("Starting insert elements of the given graph into vector database")
 
        from_load = False
        if self.exist_index() and not force:
            logger.info("Loading index from the file {}".format(self.config.persist_path))
            from_load = await self._load_index()
        else:
        
            self._index = self._get_index()
        if not from_load:
            # Note: When you successfully load the index from a file, you don't need to rebuild it.
            await self.clean_index()
            logger.info("Building index for input elements")
            await self._update_index(elements, meta_data)
            self._storage_index()
            logger.info("Index successfully built and stored.")
        logger.info("✅ Finished starting insert entities of the given graph into vector database")

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    async def retrieval(self, query, top_k):
        pass

    @abstractmethod
    def _get_index(self):
        pass

    @abstractmethod
    async def retrieval_batch(self, queries, top_k):
        pass

    @abstractmethod
    async def _update_index(self, elements, meta_data):
        pass

    @abstractmethod
    def _get_retrieve_top_k(self):
        return 10

    @abstractmethod
    def _storage_index(self):
        pass

    @abstractmethod
    async def _load_index(self) -> bool:
        pass

    async def similarity_score(self, object_q, object_d):
        return await self._similarity_score(object_q, object_d)

    
    async def _similarity_score(self, object_q, object_d):
        pass

    
    async def get_max_score(self, query):
        pass

    async def clean_index(self):
       clean_storage(self.config.persist_path)
       
    @abstractmethod
    async def retrieval_nodes(self, query, top_k, graph):
        pass


    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        pass