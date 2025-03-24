from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from Core.Prompt.QueryPrompt import DALK_RERANK_PROMPT, DALK_CONVERT_PROMPT, DALK_STEP_PROMPT, DALK_CONTEXT_PROMPT, DALK_QUERY_PROMPT

class DalkQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _select_most_relative_paths(self, path_list, query):
        if path_list is None: return ""
        # import pdb
        # pdb.set_trace()
        path_str = [(path[0]["src_id"]
                     + ("->" + e["content"] + "-> " + e["tgt_id"]) for edge in path)
                     for path in path_list]
        context = DALK_RERANK_PROMPT.format(graph=path_str, question=query)

        return await self.llm.aask(context)

    async def _select_most_relative_neis(self, nei_list, query):
        # import pdb
        # pdb.set_trace()
        if nei_list is None: return ""
        nei_str_list = ["->".join([e["src_id"], e["content"], e["tgt_id"]])
                        for e in nei_list]
        if len(nei_str_list) > 5:
            nei_str_list = nei_str_list[:-5]
        neighbor_str = "\n".join(nei_str_list)
        context = DALK_RERANK_PROMPT.format(graph=neighbor_str, question=query)

        return await self.llm.aask(context)

    async def _to_natural_language(self, context):
        context = DALK_CONVERT_PROMPT.format(graph=context)
        return await self.llm.aask(context)

    async def _process_info(self, query, paths, neis):
        paths = await self._to_natural_language(paths)
        neis = await self._to_natural_language(neis)
        step = await self.llm.aask(DALK_STEP_PROMPT.format(question=query, paths=paths, neis=neis))
        return paths, neis, step

    async def _retrieve_relevant_contexts(self, query):
        entities = await self.extract_query_entities(query)
        entities = await self._retriever.retrieve_relevant_content(type = Retriever.ENTITY, mode = "link_entity", query_entities = entities)
        entities = [node['entity_name'] for node in entities]
        paths = await self._select_most_relative_paths(
                    await self._retriever.retrieve_relevant_content(type = Retriever.SUBGRAPH, mode = "paths_return_list", seed = entities, cutoff = self.config.k_hop),
                    query
                    )
        neis = await self._select_most_relative_neis(
                    await self._retriever.retrieve_relevant_content(type = Retriever.SUBGRAPH, mode = "neighbors_return_list", seed = entities),
                    query
                    )
        paths, neis, step = await self._process_info(query, paths, neis)
        return DALK_CONTEXT_PROMPT.format(paths=paths, neis=neis, step=step)

    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query)
        response = await self.llm.aask(DALK_QUERY_PROMPT.format(question=query, context=context))
        logger.info(query)
        logger.info(response)
        return response

    async def generation_qa(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        context_str = '\n'.join('{index}: {context}'.format(index=i, context=c) for i, c in enumerate(context, start=1))

        answer = await self.llm.aask(DALK_QUERY_PROMPT.format(question=query, context=context_str))
        return answer

    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
