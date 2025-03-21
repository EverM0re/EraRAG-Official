from collections import Counter
from typing import List
from Core.Common.Constants import GRAPH_FIELD_SEP


class MergeEntity:

    merge_function = None

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        merged_source_ids = list(set(new_source_ids) | set(existing_source_ids))

        return GRAPH_FIELD_SEP.join(merged_source_ids)

    @staticmethod
    def merge_types(existing_entity_types: List[str], new_entity_types):
        # Use the most frequency entity type as the new entity
        merged_entity_types = existing_entity_types + new_entity_types
        entity_type_counts = Counter(merged_entity_types)
        most_common_type = entity_type_counts.most_common(1)[0][0] if entity_type_counts else ''
        return most_common_type

    @staticmethod
    def merge_descriptions(entity_relationships: List[str], new_descriptions):
        merged_descriptions = list(set(new_descriptions) | set(entity_relationships))
        description = GRAPH_FIELD_SEP.join(sorted(merged_descriptions))
        return description

    @classmethod
    async def merge_info(cls, merge_keys, nodes_data, merge_dict):
        """
        Merge entity information for a specific entity name, including source IDs, entity types, and descriptions.
        If an existing key is present in the data, merge the information; otherwise, use the new insert data.
        """
        if len(nodes_data) == 0:
            return []
        result = []
        if cls.merge_function is None:
            cls.merge_function = {
                "source_id": cls.merge_source_ids,
                "entity_type": cls.merge_types,
                "description": cls.merge_descriptions,
            }

        for merge_key in cls.merge_keys:
            if merge_key in merge_dict:
                result.append(cls.merge_function[merge_key](nodes_data[merge_key], merge_dict[merge_key]))
        if len(result) < len(cls.merge_keys):
            result.extend("" * (len(cls.merge_keys) - len(result)))
        return tuple(result)


class MergeRelationship:
    merge_keys = ["source_id", "weight", "description", "keywords", "relation_name"]
    merge_function = None

    @staticmethod
    def merge_weight(merge_weight, new_weight):
        return sum(new_weight + merge_weight)

    @staticmethod
    def merge_descriptions(entity_relationships, new_descriptions):
        return GRAPH_FIELD_SEP.join(
            sorted(set(new_descriptions + entity_relationships))
        )

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        return GRAPH_FIELD_SEP.join(
            set(new_source_ids + existing_source_ids)
        )

    @staticmethod
    def merge_keywords(keywords: List[str], new_keywords):
        return GRAPH_FIELD_SEP.join(
            set(keywords + new_keywords)
        )

    @staticmethod
    def merge_relation_name(relation_name, new_relation_name):
        return GRAPH_FIELD_SEP.join(
            sorted(set(relation_name + new_relation_name)
                   ))

    @classmethod
    async def merge_info(cls, edges_data, merge_dict):
        """
        Merge entity information for a specific entity name, including source IDs, entity types, and descriptions.
        If an existing key is present in the data, merge the information; otherwise, use the new insert data.
        """
        if len(edges_data) == 0:
            return []
        result = []
        if cls.merge_function is None:
            cls.merge_function = {
                "weight": cls.merge_weight,
                "description": cls.merge_description,
                "source_id": cls.merge_source_ids,
                "keywords": cls.merge_keywords,
                "relation_name": cls.merge_relation_name
            }

        for merge_key in cls.merge_keys:
            if merge_key in merge_dict:
                result.append(cls.merge_function[merge_key](edges_data[merge_key], merge_dict[merge_key]))

        return tuple(result)
