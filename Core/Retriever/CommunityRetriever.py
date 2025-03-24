from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import asyncio
import json
from Core.Common.Utils import truncate_list_by_token_size
from collections import Counter
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Prompt import QueryPrompt


class CommunityRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["from_entity", "from_level"]
        self.type = "community"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="community", method_name="from_entity")
    async def _find_relevant_community_from_entities(self, seed: list[dict]):

        community_reports = self.community.community_reports
        related_communities = []
        for node_d in seed:
            if "clusters" not in node_d:
                continue
            related_communities.extend(json.loads(node_d["clusters"]))
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= self.config.level
        ]

        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        _related_community_datas = await asyncio.gather(
            *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
        )
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1),
            ),
            reverse=True,
        )
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys
        ]

        use_community_reports = truncate_list_by_token_size(
            sorted_community_datas,
            key=lambda x: x["report_string"],
            max_token_size=self.config.local_max_token_for_community_report,
        )
        if self.config.local_community_single_one:
            use_community_reports = use_community_reports[:1]
        return use_community_reports

    @register_retriever_method(type="community", method_name="from_level")
    async def find_relevant_community_by_level(self, seed=None):
        community_schema = self.community.community_schema
        community_schema = {
            k: v for k, v in community_schema.items() if v.level <= self.config.level
        }
        if not len(community_schema):
            return QueryPrompt.FAIL_RESPONSE

        sorted_community_schemas = sorted(
            community_schema.items(),
            key=lambda x: x[1].occurrence,
            reverse=True,
        )

        sorted_community_schemas = sorted_community_schemas[
                                   : self.config.global_max_consider_community
                                   ]
        community_datas = await self.community.community_reports.get_by_ids(  ###
            [k[0] for k in sorted_community_schemas]
        )

        community_datas = [c for c in community_datas if c is not None]
        community_datas = [
            c
            for c in community_datas
            if c["report_json"].get("rating", 0) >= self.config.global_min_community_rating
        ]
        community_datas = sorted(
            community_datas,
            key=lambda x: (x['community_info']['occurrence'], x["report_json"].get("rating", 0)),
            reverse=True,
        )
        logger.info(f"Retrieved {len(community_datas)} communities")
        return community_datas
