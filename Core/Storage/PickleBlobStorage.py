import pickle
from dataclasses import dataclass, field
from typing import Optional, Any
from Core.Common.Logger import logger
from Core.Storage.BaseBlobStorage import BaseBlobStorage


@dataclass
class PickleBlobStorage(BaseBlobStorage):
    RESOURCE_NAME = "blob_data.pkl"
    _data: Optional[Any] = field(init=False, default=None)

    async def get(self):
        return self._data

    async def set(self, blob) -> None:
        self._data = blob

    async def load(self, force):
        if force:
            logger.info(f"Forcing rebuild the mapping for: {self.namespace.get_load_path(self.RESOURCE_NAME)}.")
            self._data = None
            return False
        if self.namespace:
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
            if data_file_name:
                try:
                    with open(data_file_name, "rb") as f:
                        self._data = pickle.load(f)
                    logger.info("Successfully loaded data file for blob storage {data_file_name}.")
                    return True
                except Exception as e:
                    logger.error(f"Error loading data file for blob storage {data_file_name}: {e}")
                    return False
            else:
                logger.info(f"No data file found for blob storage {data_file_name}. Loading empty storage.")
                self._data = None
                return False
        else:
            self._data = None
            logger.info("Creating new volatile blob storage.")
            return False

    async def persist(self):
        if self.namespace:
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                with open(data_file_name, "wb") as f:
                    pickle.dump(self._data, f)
                logger.info(
                    f"Saving blob storage '{data_file_name}'."
                )
            except Exception as e:
                logger.error(f"Error saving data file for blob storage {data_file_name}: {e}")
