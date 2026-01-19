import json
import os
from typing import List, Dict, Any, Callable
from tool.weather import get_weather, WEATHER_SCHEMA
from tool.google_search import google_search, GOOGLE_SEARCH


class ReactTools:
    """
    React Agent 工具类

    为 ReAct Agent 提供标准化的工具接口
    """

    def __init__(self) -> None:
        # 注册工具
        self._tool_map: Dict[str, Callable] = {
            # FIXME:要注册新的函数注册即可
            "get_weather": get_weather,
            "google_search": google_search,
        }
        # 用于生成prompt
        self.toolConfig = [WEATHER_SCHEMA, GOOGLE_SEARCH]

    def get_available_tools(self) -> List[str]:
        pass

    def get_tool_description(self, tool_name: str) -> str:
        pass
