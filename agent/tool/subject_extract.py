#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/05 14:55
@Author  : weiyutao
@File    : subject_extract.py
"""


from typing import (
    Any,
    Optional,
    overload
)
from datetime import datetime
from pathlib import Path


from agent.base.base_tool import tool
from agent.tool.info_extract import InfoExtract
from agent.llm_api.base_llm import BaseLLM


@tool
class SubjectExtract(InfoExtract):
    @overload
    def __init__(
        self, 
        llm: Optional[BaseLLM] = None
    ):
        ...
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')