import asyncio
from Core.Chunk.ChunkFactory import create_chunk_method
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.ChunkKVStorage import ChunkKVStorage
from typing import List, Union


class DocChunk:
    def __init__(self, config, token_model, namesapce):
        self.config = config
        self.chunk_method = create_chunk_method(self.config.chunk_method)
        self._chunk = ChunkKVStorage(namespace=namesapce)
        self.token_model = token_model

    @property
    def namespace(self):
        return None

    # TODO: Try to rewrite here, not now
    @namespace.setter
    def namespace(self, namespace):
        self.namespace = namespace

    async def build_chunks(self, docs: Union[str, List[str]], force=True):
        logger.info("Starting chunk the given documents")
  
        is_exist = await self._load_chunk(force)
        if not is_exist or force:

            # TODO: Now we only support the str, list[str], Maybe for more types.
            if isinstance(docs, str):
                docs = [docs]

            if isinstance(docs, list):
                if all(isinstance(doc, dict) for doc in docs):
                    docs = {
                        mdhash_id(doc["content"].strip(), prefix="doc-"): {
                            "content": doc["content"].strip(),
                            "title": doc.get("title", ""),
                        }
                        for doc in docs
                    }
                else:
                    docs = {
                        mdhash_id(doc.strip(), prefix="doc-"): {
                            "content": doc.strip(),
                            "title": "",
                        }
                        for doc in docs
                    }

            flatten_list = list(docs.items())
            docs = [doc[1]["content"] for doc in flatten_list]
            doc_keys = [doc[0] for doc in flatten_list]
            title_list = [doc[1]["title"] for doc in flatten_list]
            tokens = self.token_model.encode_batch(docs, num_threads=16)

            chunks = await self.chunk_method(
                tokens,
                doc_keys=doc_keys,
                tiktoken_model=self.token_model,
                title_list=title_list,
                overlap_token_size=self.config.chunk_overlap_token_size,
                max_token_size=self.config.chunk_token_size,
            )

            for chunk in chunks:
                chunk["chunk_id"] = mdhash_id(chunk["content"], prefix="chunk-")
                await self._chunk.upsert(chunk["chunk_id"], TextChunk(**chunk))

            await self._chunk.persist()
        logger.info("✅ Finished the chunking stage")

    async def _load_chunk(self, force=False):
        if force:
            return False
        return await self._chunk.load_chunk()

    async def get_chunks(self):
        return await self._chunk.get_chunks()

    async def get_index_by_merge_key(self, chunk_id):
        return await self._chunk.get_index_by_merge_key(chunk_id)

    @property
    async def size(self):
        return await self._chunk.size()

    async def get_index_by_key(self, key):
        return await self._chunk.get_index_by_key(key)

    async def get_data_by_key(self, chunk_id):

        chunk = await self._chunk.get_by_key(chunk_id)
        return chunk.content

    async def get_data_by_index(self, index):
        chunk = await self._chunk.get_data_by_index(index)
        return chunk.content

    async def get_key_by_index(self, index):
        return await self._chunk.get_key_by_index(index)

    async def get_data_by_indices(self, indices):
        return await asyncio.gather(
            *[self.get_data_by_index(index) for index in indices]
        )
