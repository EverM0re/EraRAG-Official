from typing import List

from Core.Schema.EntityRelation import Entity, Relationship

class ERGraphSchema:
    nodes: List[Entity]
    edges: List[Relationship]
