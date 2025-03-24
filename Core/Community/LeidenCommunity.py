"""
Please refer to the Nano-GraphRAG: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py
"""
from Core.Community.BaseCommunity import BaseCommunity
from collections import defaultdict
from graspologic.partition import hierarchical_leiden
from Core.Common.Utils import (
    community_report_from_json,
    list_to_quoted_csv_string,
    encode_string_by_tiktoken,
    truncate_list_by_token_size, clean_str
)
from Core.Common.Logger import logger
import asyncio

from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.CommunitySchema import CommunityReportsResult, LeidenInfo
from Core.Prompt import CommunityPrompt
from Core.Community.ClusterFactory import register_community
from Core.Storage.JsonKVStorage import JsonKVStorage


@register_community(name="leiden")
class LeidenCommunity(BaseCommunity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._community_reports: JsonKVStorage = JsonKVStorage(self.namespace, "community_report")
        self._community_node_map: JsonKVStorage = JsonKVStorage(self.namespace, "community_node_map")
        self._communities_schema: dict[str, LeidenInfo] = defaultdict(LeidenInfo)

    @property
    def community_reports(self):
        """Getter method for community_reports."""
        return self._community_reports

    async def clustering(self, largest_cc, max_cluster_size, random_seed):
        await self._clustering(largest_cc, max_cluster_size, random_seed)

    async def _clustering(self, largest_cc, max_cluster_size, random_seed):
        if largest_cc is None:
            logger.warning("No largest connected component found, skipping Leiden clustering; Please check the input graph.")
            return None
        community_mapping = hierarchical_leiden(
            largest_cc,
            max_cluster_size=max_cluster_size,
            random_seed=random_seed,
        )
        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)

        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
          
            node_communities[clean_str(partition.node)].append(
                {"level": level_key, "cluster": str(cluster_id)}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        await self._community_node_map.upsert(node_communities)


    @property
    def community_schema(self):
        return self._communities_schema

    async def _generate_community_report(self, er_graph):
        # Construct the cluster <-> node mapping
        await er_graph.cluster_data_to_subgraphs(self._community_node_map.json_data)
        # Fetch community schema
        self._communities_schema = await er_graph.community_schema()
        if self._communities_schema is None:
            logger.warning("No community schema found, skipping community report generation.")
            return None
        community_keys, community_values = list(self._communities_schema.keys()), list(self._communities_schema.values())
        # Generate reports by community levels
        levels = sorted(set([c.level for c in community_values]), reverse=True)
        logger.info(f"Generating by levels: {levels}")
        community_datas = {}

        for level in levels:
            this_level_community_keys, this_level_community_values = zip(
                *[(k, v) for k, v in zip(community_keys, community_values) if v.level == level]
            )
            this_level_communities_reports = await asyncio.gather(
                *[self._form_single_community_report(er_graph, c, community_datas) for c in this_level_community_values]
            )

            community_datas.update(
                {
                    k: {
                        "report_string": community_report_from_json(r),
                        "report_json": r,
                        **v.as_dict
                    }
                    for k, r, v in
                    zip(this_level_community_keys, this_level_communities_reports, this_level_community_values)
                }
            )

        await self._community_reports.upsert(community_datas)

    async def _form_single_community_report(self, er_graph, community,
                                            already_reports: dict[str, CommunityReportsResult]) -> dict:

        describe = await self._pack_single_community_describe(er_graph, community, already_reports=already_reports)
        prompt = CommunityPrompt.COMMUNITY_REPORT.format(input_text=describe)

        response = await self.llm.aask(prompt, format = "json")
        # data = prase_json_from_response(response)

        return response

    @staticmethod
    async def _pack_single_community_by_sub_communities(

            community,
            max_token_size: int,
            already_reports: dict[str, CommunityReportsResult],
    ):

        """Pack a single community by summarizing its sub-communities."""
        all_sub_communities = [
            already_reports[k] for k in community.sub_communities if k in already_reports
        ]
        all_sub_communities = sorted(
            all_sub_communities, key=lambda x: x["occurrence"], reverse=True
        )
        truncated_sub_communities = truncate_list_by_token_size(
            all_sub_communities,
            key=lambda x: x["report_string"],
            max_token_size=max_token_size,
        )
        sub_fields = ["id", "report", "rating", "importance"]
        sub_communities_describe = list_to_quoted_csv_string(
            [sub_fields]
            + [
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", -1),
                    c["occurrence"],
                ]
                for i, c in enumerate(truncated_sub_communities)
            ]
        )
        already_nodes = set()
        already_edges = set()
        for c in truncated_sub_communities:
            already_nodes.update(c["nodes"])
            already_edges.update([tuple(e) for e in c["edges"]])
        return (
            sub_communities_describe,
            len(encode_string_by_tiktoken(sub_communities_describe)),
            already_nodes,
            already_edges,
        )

    async def _pack_single_community_describe(
            self, er_graph: BaseGraph, community: LeidenInfo, max_token_size: int = 12000,
            already_reports=None
    ) -> str:
        """Generate a detailed description of the community based on its attributes and existing reports."""
        if already_reports is None:
            already_reports = {}
        nodes_in_order = sorted(community.nodes)
        edges_in_order = sorted(community.edges, key=lambda x: x[0] + x[1])

        nodes_data = await asyncio.gather(*[er_graph.get_node(n) for n in nodes_in_order])
        edges_data = await asyncio.gather(*[er_graph.get_edge(src, tgt) for src, tgt in edges_in_order])

        node_fields = ["id", "entity", "type", "description", "degree"]
        edge_fields = ["id", "source", "target", "description", "rank"]

        nodes_list_data = [
            [
                i,
                node_name,
                node_data.get("entity_type", "UNKNOWN"),
                node_data.get("description", "UNKNOWN"),
                await er_graph.node_degree(node_name),
            ]
            for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
        ]
        nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
        )

        edges_list_data = [
            [
                i,
                edge_name[0],
                edge_name[1],
                edge_data.get("description", "UNKNOWN"),
                await er_graph.edge_degree(*edge_name),
            ]
            for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
        ]
        edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
        edges_may_truncate_list_data = truncate_list_by_token_size(
            edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
        )

        truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(edges_list_data) > len(
            edges_may_truncate_list_data)
        report_describe = ""
        need_to_use_sub_communities = truncated and len(community.sub_communities) and len(already_reports)

        if need_to_use_sub_communities or self.enforce_sub_communities:
            logger.info(
                f"Community {community.title} exceeds the limit or force_to_use_sub_communities is True, using sub-communities")
            report_describe, report_size, contain_nodes, contain_edges = await self._pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )

            report_exclude_nodes_list_data = [n for n in nodes_list_data if n[1] not in contain_nodes]
            report_include_nodes_list_data = [n for n in nodes_list_data if n[1] in contain_nodes]
            report_exclude_edges_list_data = [e for e in edges_list_data if (e[1], e[2]) not in contain_edges]
            report_include_edges_list_data = [e for e in edges_list_data if (e[1], e[2]) in contain_edges]

            nodes_may_truncate_list_data = truncate_list_by_token_size(
                report_exclude_nodes_list_data + report_include_nodes_list_data,
                key=lambda x: x[3],
                max_token_size=(max_token_size - report_size) // 2,
            )
            edges_may_truncate_list_data = truncate_list_by_token_size(
                report_exclude_edges_list_data + report_include_edges_list_data,
                key=lambda x: x[3],
                max_token_size=(max_token_size - report_size) // 2,
            )

        nodes_describe = list_to_quoted_csv_string([node_fields] + nodes_may_truncate_list_data)
        edges_describe = list_to_quoted_csv_string([edge_fields] + edges_may_truncate_list_data)

        return f"""-----Reports-----
            ```csv
            {report_describe}
            ```
            -----Entities-----
            ```csv
            {nodes_describe}
            ```
            -----Relationships-----
            ```csv
            {edges_describe}
        ```"""

    async def _load_community_report(self, graph, force) -> bool:

        if force:
            logger.info("☠️ Force to regenerate the community report.")
            return True
        await self._community_reports.load()
        if await self._community_reports.is_empty():
            logger.error("Failed to load community report.")
            return False
        else:
            self._communities_schema = await graph.community_schema()
            logger.info("Successfully loaded community report.")
            return True

    async def _persist_community(self):
        try:
            await self._community_reports.persist()
        except Exception as e:
            logger.exception("❌ Failed to persist community report: {error}.".format(error=e))

    async def _load_cluster_map(self, force):
        if force: return False
        await self._community_node_map.load()
        if await self._community_node_map.is_empty():
            logger.error("❌ Failed to load community <-> node map.")
            return False
        else:
            logger.info("✅ Successfully loaded community <-> node map.")
            return True

    async def _persist_cluster_map(self):
        try:
            await self._community_node_map.persist()
        except Exception as e:
            logger.exception("❌ Failed to persist community <-> node map: {error}.".format(error=e))
