#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
import os
from loguru import logger as _logger
from Option.Config2 import default_config
_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")

    
    if name:
        log_dir = os.path.join(name, "Logs")
        os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
        log_name = os.path.join(log_dir, f"{formatted_date}.log")
    else:
        log_name = f"Logs/{formatted_date}.log"


    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f"{log_name}", level=logfile_level)
    return _logger


logger = define_log_level(name = os.path.join(default_config.working_dir, default_config.exp_name))


def log_llm_stream(msg):
    _llm_stream_log(msg)


def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func


def _llm_stream_log(msg):
    if _print_level in ["INFO"]:
        print(msg, end="")