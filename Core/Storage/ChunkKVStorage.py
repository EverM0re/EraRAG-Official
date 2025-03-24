import pickle
from dataclasses import dataclass, field
from typing import Dict,  List, Optional, Union
from Core.Common.Utils import split_string_by_multi_markers 
from Core.Common.Constants import GRAPH_FIELD_SEP
import numpy as np
import os
import numpy.typing as npt
from Core.Common.Logger import logger
from Core.Storage.BaseKVStorage import BaseKVStorage
from Core.Schema.ChunkSchema import TextChunk
@dataclass
class ChunkKVStorage(BaseKVStorage):
    data_name = "chunk_data_idx.pkl"
    chunk_name = "chunk_data_key.pkl"
    _data: Dict[int, TextChunk] = field(init=False, default_factory=dict)
    _chunk: Dict[str, TextChunk] = field(init=False, default_factory=dict)
    _key_to_index: Dict[str, int] = field(init=False, default_factory=dict)
    _np_keys: Optional[npt.NDArray[np.object_]] = field(init=False, default=None)

    async def size(self) -> int:
        return len(self._data)

    async def get_by_key(self, key:str) -> TextChunk:
        return self._data.get(self._key_to_index.get(key, None), None)

    async def get_data_by_index(self, index) -> TextChunk:
        return self._data.get(index, None)


        
    async def get_index_by_merge_key(self, merge_chunk_id: str) -> list[int]:
        key_list  = split_string_by_multi_markers(merge_chunk_id, [GRAPH_FIELD_SEP])
        index_list = [self._key_to_index.get(chunk_id, None) for chunk_id in key_list]
        return index_list
    
    async def get_index_by_key(self, key:str) -> int:
        return self._key_to_index.get(key, None)




    async def upsert_batch(self, keys, values) -> None:
        for key, value in zip(keys, values):
            self._chunk[key] = value
            index = self._key_to_index.get(key, None)
            if index is None:
                index = value.index
                self._key_to_index[key] = index
                self._data[index] = value
    async def upsert(self, key, value) -> None:
        
        self._chunk[key] = value
        index = self._key_to_index.get(key, None)
        if index is None:
            index = value.index
            self._key_to_index[key] = index
        # If index is already in the data, we need to update the value
        self._data[index] = value

    async def delete_by_key(self, key) -> None:

        index = self._key_to_index.pop(key, None)
        if index is not None:
            self._data.pop(index, None)
        else:
            logger.warning(f"Key '{key}' not found in indexed key-value storage.")

    
    async def chunk_datas(self):
        inserting_chunks = {key: value for key, value in self._chunk.items() if key in self._chunk}

        
        return list(inserting_chunks.items())

    @property
    def dat_idx_pkl_file(self):
       return self.namespace.get_save_path(self.data_name)
    @property
    def dat_key_pkl_file(self):
        return self.namespace.get_save_path(self.chunk_name)
    
    async def load_chunk(self):
        # Attempting to load the graph from the specified pkl file
        logger.info(f"Attempting to load the chunk data from: {self.dat_idx_pkl_file} and {self.dat_key_pkl_file}" )
        if os.path.exists(self.dat_idx_pkl_file) and os.path.exists(self.dat_key_pkl_file):
            try:
                with open(self.dat_idx_pkl_file, "rb") as file:
                    self._data = pickle.load(file)
                with open(self.dat_key_pkl_file, "rb") as file:
                    self._chunk = pickle.load(file)
                self._key_to_index = {key: value.index for key, value in self._chunk.items()}
                logger.info(
                    f"Successfully loaded chunk data (idx and key) from: {self.dat_idx_pkl_file}")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load chunk data from: {self.dat_idx_pkl_file} and {self.dat_key_pkl_file} with {e}! Need to re-chunk the documents.")
                return False
        else:
            # Pkl file doesn't exist; need to construct the tree from scratch
            logger.info("Pkl file does not exist! Need to chunk the documents from scratch.")
            return False
    async def _persist(self):
        logger.info(f"Writing data into {self.dat_idx_pkl_file} and {self.dat_key_pkl_file}")
  
        self.write_chunk_data(self._data, self.dat_idx_pkl_file)
        self.write_chunk_data(self._chunk, self.dat_key_pkl_file)
    @staticmethod
    def write_chunk_data(data, pkl_file):
        with open(pkl_file, "wb") as file:
            pickle.dump(data, file)

    async def persist(self):
        # Attempting to save the graph to the specified pkl file
        await self._persist()

    async def get_chunks(self):
        return list(self._chunk.items())
    
    async def size(self):
        print(len(self._data))
        print(len(self._chunk))
        assert len(self._data) == len(self._chunk)
        return len(self._data)