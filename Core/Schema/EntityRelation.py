from dataclasses import dataclass, asdict, field


@dataclass
class Entity:
    entity_name: str  # Primary key for entity
    source_id: str  # Unique identifier of the source from which this node is derived
    entity_type: str = field(default="")  # Entity type
    description: str = field(default="")  # The description of this entity


    @property
    def as_dict(self):
        return asdict(self)


@dataclass
class Relationship:
    """
    Initializes an Edge object with the given attributes.

    Args:
        src_id (str): The name of the entity on the left side of the edge.
        tgt_id (str): The name of the entity on the right side of the edge.
        source_id (str): The unique identifier of the source from which this edge is derived.
        **kwargs: Additional keyword arguments for optional attributes.
            - relation_name (str, optional): The name of the relation. Defaults to an empty string.
            - weight (float, optional): The weight of the edge, used in GraphRAG and LightRAG. Defaults to 0.0.
            - description (str, optional): A description of the edge, used in GraphRAG and LightRAG. Defaults to an empty string.
            - keywords (str, optional): Keywords associated with the edge, used in LightRAG. Defaults to an empty string.
            - rank (int, optional): The rank of the edge, used in LightRAG. Defaults to 0.
    """
    src_id: str  # Name of the entity on the left side of the edge
    tgt_id: str  # Name of the entity on the right side of the edge
    # (src_id, tgt_id), serving as the primary key for one edge.
    source_id: str  # Unique identifier of the source from which this edge is derived
    relation_name: str = field(default="")  # Name of the relation
    weight: float = field(default=0.0)  # Weight of the edge, used in GraphRAG and LightRAG
    description: str = field(default="")  # Description of the edge, used in GraphRAG and LightRAG
    keywords: str = field(default="")  # Keywords associated with the edge, used in LightRAG
    rank: int = field(default=0)  # Rank of the edge, used in LightRAG



    @property
    def as_dict(self):
        return asdict(self)
