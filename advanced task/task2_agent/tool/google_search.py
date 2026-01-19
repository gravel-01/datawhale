import json
import os


def google_search(self, search_query: str) -> str:
    """执行谷歌搜索
    Args:
        search_query: 搜索关键词
    Returns:
        格式化的搜索结果字符串
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": search_query})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    pass


GOOGLE_SEARCH = {
    "name_for_human": "谷歌搜索",
    "name_for_model": "google_search",
    "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。",
    "parameters": [
        {
            "name": "search_query",
            "description": "搜索关键词或短语",
            "required": True,
            "schema": {"type": "string"},
        }
    ],
}
