#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/01 00:57
@File    : BaseLLM.py
@Desc    : Refer to the MetaGPT https://github.com/geekan/MetaGPT/blob/main/metagpt/provider/base_llm.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


from Core.Common.EmbConfig import EmbConfig
from Core.Common.Constants import  USE_CONFIG_TIMEOUT
from Core.Common.Logger import logger



class BaseEmb(ABC):
    """Emb API abstract class, requiring all inheritors to provide a series of standard capabilities"""

    config: EmbConfig   
  
    # OpenAI / Azure / Others
    aclient: Optional[Union[AsyncOpenAI]] = None
    # cost_manager: Optional[CostManager] = None
    model: Optional[str] = None  # deprecated
    # pricing_plan: Optional[str] = None

    @abstractmethod
    def __init__(self, config: EmbConfig):
        pass


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(ConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(
        self, messages: list[dict], stream: bool = False, timeout: int = USE_CONFIG_TIMEOUT, max_tokens = None
    ) -> str:
        """Asynchronous version of completion. Return str. Support stream-print"""
        if stream:
            return await self._achat_completion_stream(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)
        resp = await self._achat_completion(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)
        return self.get_choice_text(resp)

