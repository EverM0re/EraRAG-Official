from Core.Utils.YamlModel import YamlModel


class ChunkConfig(YamlModel):
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    chunk_method: str = "chunking_by_token_size"
