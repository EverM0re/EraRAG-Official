from typing import Any
from Core.Common.Utils import mdhash_id
from collections import defaultdict

from Core.Schema.ChunkSchema import TextChunk


class ChunkingFactory:
    chunk_methods: dict = defaultdict(Any)

    def register_chunking_method(
            self,
            method_name: str,
            method_func=None  # can be any classes or functions
    ):
        if self.has_chunk_method(method_name):
            return

        self.chunk_methods[method_name] = method_func

    def has_chunk_method(self, key: str) -> Any:
        return key in self.chunk_methods

    def get_method(self, key) -> Any:
        return self.chunk_methods.get(key)


# Registry instance
CHUNKING_REGISTRY = ChunkingFactory()


def register_chunking_method(method_name):
    """ Register a new chunking method
    
    This is a decorator that can be used to register a new chunking method.
    The method will be stored in the self.methods dictionary.
    
    Parameters
    ----------
    method_name: str
        The name of the chunking method.
    """

    def decorator(func):
        """ Register a new chunking method """
        CHUNKING_REGISTRY.register_chunking_method(method_name, func)

    return decorator


def create_chunk_method(method_name):
    chunking_method = CHUNKING_REGISTRY.get_method(method_name)
    return chunking_method
