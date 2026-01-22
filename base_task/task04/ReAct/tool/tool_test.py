from tool import *
from google_search import google_search
from weather import get_weather

Tool = ReactTools()

# print(Tool.get_tool_descriptions())
# print(Tool.execute_tool("get_weather", city="上海"))

print(Tool.execute_tool("google_search", search_query="Python编程语言的优缺点"))
