from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt
from typing import Union


class ToGQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _extend_reasoning_paths_per_depth(self, query: str) -> tuple[bool, list[list[tuple]], list, list, list]:
        from collections import defaultdict
        total_entity_relation_list = []
        total_relations_dict = defaultdict(list)

        ####################### Retrieve relations from entities by agent ############################################
        for index, entity in enumerate(self.topic_entity_candidates):
            if entity != "[FINISH]":
                relations, relations_dict = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION, # aim at retrieving relations
                                                                                            mode="from_entity_by_agent", # method: use information of entity by agent
                                                                                            query=query,
                                                                                            entity=entity,
                                                                                            pre_relations_name=self.pre_relations_name,
                                                                                            pre_head=self.pre_heads[index],
                                                                                            width=self.config.width)
                # merge
                for key, value in relations_dict.items():
                    total_relations_dict[key].extend(value)
                total_entity_relation_list.extend(relations)

        ###################### Retrieve entities from relations by agent #############################################
        flag, candidates_list = await self._retriever.retrieve_relevant_content(
            type=Retriever.ENTITY, # aim at retrieving entity
            mode="from_relation_by_agent", # method: use information of relation by agent
            query=query,
            total_entity_relation_list=total_entity_relation_list,
            total_relations_dict=total_relations_dict,
            width=self.config.width,
        )

        ####################### Assemble relations and entities to obtain paths ########################################
        rels, candidates, tops, heads, scores = map(list, zip(*candidates_list))
        chains= [[(tops[i], rels[i], candidates[i]) for i in range(len(candidates))]]
        return flag, chains, candidates, rels, heads

    async def _retrieve_initialization(self, query: str):
        # find relevant entity seeds
        entities = await self.extract_query_entities(query)
        # link these entity seeds to our actual entities in vector database and our built graph
        entities = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY,
                                                             mode="link_entity",
                                                             query_entities=entities)
        entities = list(map(lambda x: x['entity_name'], entities))
        # define the global variable about reasoning
        self.reasoning_paths_list = []
        self.pre_relations_name = []
        self.pre_heads = [-1] * len(entities)
        self.flag_printed = False
        self.topic_entity_candidates = entities

    async def _retrieve_relevant_contexts(self, query: str, mode: str) -> str:
        if mode == "retrieve":
            flag, chains, candidates, rels, heads = await self._extend_reasoning_paths_per_depth(query)
            # update
            self.pre_relations_name = rels
            self.pre_heads = heads
            self.reasoning_paths_list.append(chains)
            self.topic_entity_candidates = candidates
            # If we can retrieve some paths, then try to response the query using current information.
            if flag:
                from Core.Prompt.TogPrompt import prompt_evaluate
                context = prompt_evaluate + query + "\n"
                chain_prompt = '\n'.join(
                    [', '.join([str(x) for x in chain]) for sublist in self.reasoning_paths_list for chain in sublist])
                context += "\nKnowledge Triplets: " + chain_prompt + 'A: '
                return context
            else:
                return ""

        elif mode == "half_stop":
            from Core.Prompt.TogPrompt import answer_prompt
            context = answer_prompt.format(query + '\n')
            chain_prompt = '\n'.join(
                [', '.join([str(x) for x in chain]) for sublist in self.reasoning_paths_list for chain in sublist])
            context += "\nKnowledge Triplets: " + chain_prompt + 'A: '

            return context
        else:
            raise TypeError("parameter mode must be either retrieve or half_stop!")

    def _encapsulate_answer(self, query, answer, chains):
        dic = {"question": query, "results": answer, "reasoning_chains": chains}
        return dic

    def _is_finish_list(self)->tuple[bool, list]:
        if all(elem == "[FINISH]" for elem in self.topic_entity_candidates):
            return True, []
        else:
            new_lst = [elem for elem in self.topic_entity_candidates if elem != "[FINISH]"]
            return False, new_lst

    async def _half_stop(self, query, depth):
        logger.info("No new knowledge added during search depth %d, stop searching." % depth)
        context = await self._retrieve_relevant_contexts(query=query, mode="half_stop")
        response = await self.generation_qa(mode="half_stop", context=context)
        return response

    async def query(self, query):
        # Initialization and find the entitiy seeds about given query
        await self._retrieve_initialization(query)
        for depth in range(1, self.config.depth + 1):
            context = await self._retrieve_relevant_contexts(query=query, mode="retrieve")
            if context:
                # If the length of context is not zero, we try to answer the question
                is_stop, response = await self.generation_qa(mode="retrieve", context=context)
                if is_stop:
                    # If is_stop is True, we can answer the question without searching further
                    logger.info("ToG stopped at depth %d." % depth)
                    break
                else:
                    # Otherwise, we still need to search
                    logger.info("ToG still not find the answer at depth %d." % depth)
                    all_candidates_are_finish, new_candidates = self._is_finish_list()
                    if all_candidates_are_finish:
                        # If all the item in candidates is [FINISH], then stop
                        response = await self._half_stop(query, depth)
                        break
                    else:
                        # Otherwise, continue searching
                        self.topic_entity_candidates = new_candidates
            else:
                # Otherwise, it means no more new knowledge paths can be found.
                response = await self._half_stop(query, depth)
                break
        # This is the evidence chain.
        #print(self.reasoning_paths_list)
        return response

    async def generation_qa(self, mode: str, context: str) -> Union[tuple[bool, str], str]:
        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": context}]
        response = await self.llm.aask(msg=messages)

        if mode == "retrieve":
            # extract is_stop answer from response
            start_index = response.find("{")
            end_index = response.find("}")
            if start_index != -1 and end_index != -1:
                is_stop = response[start_index + 1:end_index].strip()
            else:
                is_stop = ""

            # determine whether it can stop.
            is_true = lambda text: text.lower().strip().replace(" ", "") == "yes"
            if is_true(is_stop):
                return True, response
            else:
                return False, response
        elif mode == "half_stop":
            return response
        else:
            raise TypeError("parameter mode must be either retrieve or half_stop!")

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE