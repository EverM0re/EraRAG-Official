from Core.Common.Utils import load_json, write_json
from Core.Common.Logger import logger
from Core.Storage.BaseKVStorage import (
    BaseKVStorage,
)


class JsonKVStorage(BaseKVStorage):
    def __init__(self, namespace, name):
        super().__init__()
        self._data = {}
        self.name: str = "{name}.json".format(name=name)
        self.namespace = namespace

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    @property
    def json_data(self):
        return self._data

    @property
    def _file_name(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    async def persist(self):
        write_json(self._data, self._file_name)
        logger.info(f"Write KV {self._file_name} with {len(self._data)} data")

    async def load(self):
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self._file_name} with {len(self._data)} data")

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}

    async def is_empty(self):
        return len(self._data) == 0
