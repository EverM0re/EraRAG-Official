from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from typing import Union
import asyncio
# from torch_geometric.data import Data, InMemoryDataset
from typing import Any, Dict, List, Tuple, no_type_check
from pcst_fast import pcst_fast
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Core.Common.Utils import truncate_str_by_token_size


class GRQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def initialization(self):
        from Core.Common.Constants import GRAPH_FIELD_SEP
        origin_nodes = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY,
                                                                 mode="get_all") # list[dict]
        origin_edges = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION,
                                                                 mode="get_all") # list[dict]
        relations = list(map(lambda x: x["relation_name"].split(sep=GRAPH_FIELD_SEP), origin_edges)) # list[list]
        document_graph_triplets = [] # list[tuple]
        for index, edge in enumerate(origin_edges):
            for rel in relations[index]:
                document_graph_triplets.append((edge["src_id"], rel, edge["tgt_id"]))

        raw_nodes: Dict[str, int] = {}
        raw_edges = []

        for node in origin_nodes:
            if node["entity_name"] not in raw_nodes:
                raw_nodes[node["entity_name"]] = len(raw_nodes)

        for tri in document_graph_triplets:
            h, r, t = tri
            if h not in raw_nodes:
                raw_nodes[h] = len(raw_nodes)
            if t not in raw_nodes:
                raw_nodes[t] = len(raw_nodes)
            raw_edges.append({
                "src": raw_nodes[h],
                "edge_attr": r,
                "dst": raw_nodes[t]
            })

        nodes = pd.DataFrame([{ "node_id": v, "node_attr": k} for k, v in raw_nodes.items()],
                             columns=["node_id", "node_attr"])
        edges = pd.DataFrame(raw_edges,
                             columns=["src", "edge_attr", "dst"])

        nodes.node_attr = nodes.node_attr.fillna("")

        edge_index = torch.tensor([
            edges.src.tolist(),
            edges.dst.tolist(),
        ], dtype=torch.long)

        self.edge_index = edge_index # torch[2, -1]
        self.nodes = nodes # pandas: "node_id": int, "node_attr": str
        self.edges = edges # pandas: "src":int,  "edge_attr":str,  "dst": int
        self.raw_nodes = raw_nodes # dict: key: "node_attr": str, "node_id": int
        self.edges_list = relations # list[str]:  "edge_attr":str



    async def retrieval_via_pcst(
            self,
            query: str,
            topk: int = 3,
            topk_e: int = 3,
            cost_e: float = 0.5,
    ):
        c = 0.01
        if len(self.nodes) == 0 or len(self.edges) == 0:
            desc = self.nodes.to_csv(index=False) + "\n" + self.edges.to_csv(
                index=False,
                columns=["src", "edge_attr", "dst"],
            )
            return desc

        root = -1
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0
        if topk > 0:
            topk = min(topk, len(self.nodes))
            retrieve_entity = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY, mode="vdb", seed=query) # list[dict]
            retrieve_entity_id = torch.tensor(list(map(lambda x: self.raw_nodes[x["entity_name"]], retrieve_entity))) # [0,1,2,..,node_id]
            n_prizes = torch.zeros(len(self.nodes))
            n_prizes[retrieve_entity_id] = torch.arange(topk, 0, -1).float()
        else:
            n_prizes = torch.zeros(len(self.nodes))

        if topk_e > 0:
            topk_e = min(topk_e, len(self.edges))
            retrieve_relations, topk_e_values = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION, mode="vdb", seed=query,
                                                                need_score=True, need_context=False)
            e_prizes = torch.zeros(len(self.edges))
            for i, rel in enumerate(retrieve_relations):
                index = self.edges[
                    (self.edges['src'] == retrieve_relations[i]["src_id"]) &
                    (self.edges['edge_attr'] == retrieve_relations[i]["relation_name"]) &
                    (self.edges['dst'] == retrieve_relations[i]['tgt_id'])
                ].index
                e_prizes[index] = topk_e_values[i]

            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e - k) / sum(indices), last_topk_e_value - c)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
            # reduce the cost of the edges such that at least one edge is selected
            cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
        else:
            e_prizes = torch.zeros(len(self.edges))

        costs = []
        edges = []
        virtual_n_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        for i, (src, dst) in enumerate(self.edge_index.t().numpy()):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = len(self.nodes) + len(virtual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                virtual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
        num_edges = len(edges)
        if len(virtual_costs) > 0:
            costs = np.array(costs + virtual_costs)
            edges = np.array(edges + virtual_edges)

        vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters,
                                    pruning, verbosity_level)

        selected_nodes = vertices[vertices < len(self.nodes)]
        selected_edges = [mapping_e[e] for e in edges if e < num_edges]
        virtual_vertices = vertices[vertices >= len(self.nodes)]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= len(self.nodes)]
            virtual_edges = [mapping_n[i] for i in virtual_vertices]
            selected_edges = np.array(selected_edges + virtual_edges)

        edge_index = self.edge_index[:, selected_edges]
        selected_nodes = np.unique(
            np.concatenate(
                [selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

        n = self.nodes.iloc[selected_nodes]
        e = self.edges.iloc[selected_edges]
        desc = n.to_csv(index=False) + '\n' + e.to_csv(
            index=False, columns=['src', 'edge_attr', 'dst'])

        return desc


    async def _retrieve_relevant_contexts(self, query: str):
        query = f"Question: {query}\nAnswer: "
        desc = await self.retrieval_via_pcst(
            query=query,
            topk=self.config.top_k,
            topk_e=self.config.topk_e,
            cost_e=self.config.cost_e,
        )
        desc = truncate_str_by_token_size(input_str=desc, max_token_size=self.config.max_txt_len)
        return query, desc

    async def query(self, query):
        await self.initialization()

        query, context = await self._retrieve_relevant_contexts(query)
        print(context)
        response = await self.generation_qa(query, context)

        return response

    async def generation_qa(self, query: str, context: str):
        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": context + query}]
        response = await self.llm.aask(msg=messages)
        return response

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE