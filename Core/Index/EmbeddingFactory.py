"""
RAG Embedding Factory.
@Reference: https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/factories/embedding.py
@Provide: OllamaEmbedding, OpenAIEmbedding
"""

from __future__ import annotations

from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("HuggingFaceEmbedding not found, please install it by `pip install huggingface_hub`")
from Config.EmbConfig import EmbeddingType
from Config.LLMConfig import LLMType
from Core.Common.BaseFactory import GenericFactory
from Option.Config2 import Config

class RAGEmbeddingFactory(GenericFactory):
    """Create LlamaIndex Embedding with MetaGPT's embedding config."""

    def __init__(self):
        creators = {
            EmbeddingType.OPENAI: self._create_openai,
            EmbeddingType.OLLAMA: self._create_ollama,
            EmbeddingType.HF: self._create_hf,
        }
        super().__init__(creators)

    def get_rag_embedding(self, key: EmbeddingType = None, config: Config = None) -> BaseEmbedding:
        """Key is EmbeddingType."""
        return super().get_instance(key or self._resolve_embedding_type(config), config = config)

    @staticmethod
    def _resolve_embedding_type(config) -> EmbeddingType | LLMType:
        """Resolves the embedding type.

        If the embedding type is not specified, for backward compatibility, it checks if the LLM API type is either OPENAI or AZURE.
        Raise TypeError if embedding type not found.
        """
 
        if config.embedding.api_type:
            return config.embedding.api_type


        raise TypeError("To use RAG, please set your embedding in Config2.yaml.")

    def _create_openai(self, config) -> OpenAIEmbedding:
        params = dict(
            api_key = config.embedding.api_key or config.llm.api_key,
            api_base = config.embedding.base_url or config.llm.base_url,
        )

        self._try_set_model_and_batch_size(params, config)

        return OpenAIEmbedding(**params)

 
    def _create_ollama(self, config) -> OllamaEmbedding:
        params = dict(
            base_url=config.embedding.base_url,
        )

        self._try_set_model_and_batch_size(params, config)

        return OllamaEmbedding(**params)

    def _create_hf(self, config) -> HuggingFaceEmbedding:
        
        # For huggingface-hub embedding model, we only need to set the model_name
        params = dict(
            model_name=config.embedding.model,
            cache_folder=config.embedding.cache_folder,
            device = "cuda",
            target_devices = ["cuda:7"],
            embed_batch_size = 128,
        )
        if config.embedding.cache_folder == "":
            del params["cache_folder"]
        return HuggingFaceEmbedding(**params)
    
    @staticmethod
    def _try_set_model_and_batch_size(params: dict, config):
  
        """Set the model_name and embed_batch_size only when they are specified."""
        if config.embedding.model:
            params["model_name"] = config.embedding.model

        if config.embedding.embed_batch_size:
            params["embed_batch_size"] = config.embedding.embed_batch_size

        if config.embedding.dimensions:
             params["dimensions"] = config.embedding.dimensions
    def _raise_for_key(self, key: Any):
        raise ValueError(f"The embedding type is currently not supported: `{type(key)}`, {key}")


get_rag_embedding = RAGEmbeddingFactory().get_rag_embedding