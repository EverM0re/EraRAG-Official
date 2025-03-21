from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt


class KGPQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _retrieve_relevant_contexts(self, query):
        corpus, candidates_idx = await self._retriever.retrieve_relevant_content(key="description",
                                                                                 type=Retriever.ENTITY, mode="all")
        cur_contexts, idxs = await self._retriever.retrieve_relevant_content(seed=query, corpus=corpus,
                                                                             candidates_idx=candidates_idx,
                                                                             top_k=self.config.top_k // self.config.nei_k,
                                                                             type=Retriever.ENTITY, mode="tf_df")
        contexts = []
        next_reasons = [
            query + '\n' + (await self.llm.aask(QueryPrompt.KGP_REASON_PROMPT.format(question=query, context=context)))
            for context in cur_contexts]

        logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

        visited = []

        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = await self._retriever.retrieve_relevant_content(seed=idx, type=Retriever.ENTITY,
                                                                                 mode="by_neighbors")
            nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
            if nei_candidates_idx == []:
                continue

            next_contexts, next_idxs = await self._retriever.retrieve_relevant_content(seed=next_reason, corpus=corpus,
                                                                                       candidates_idx=nei_candidates_idx,
                                                                                       top_k=self.config.nei_k,
                                                                                       type=Retriever.ENTITY,
                                                                                       mode="tf_df")
            contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_idxs if corpus[_] != corpus[idx]])
            visited.append(idx)
            visited.extend([_ for _ in next_idxs])
        return contexts

    async def generation_qa(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        context_str = '\n'.join('{index}: {context}'.format(index=i, context=c) for i, c in enumerate(context, start=1))

        answer = await self.llm.aask(QueryPrompt.KGP_QUERY_PROMPT.format(question=query, context=context_str))
        return answer

    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
