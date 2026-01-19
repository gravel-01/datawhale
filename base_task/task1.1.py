import requests
import re
import os
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# system_prompt init
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 收集用户信息：
当你没有足够的信息能回复用户的最终问题时，你必须在`Action:`字段后使用 `query(answer="...")` 来输出询问用户的问题。

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。

# 反思机制：
如果 Observation 中包含用户的不满意反馈，你必须在下一次 Thought 中首先分析为什么之前的推荐失败了，并在 Action 中体现出策略的改变。

#注意事项：如果遇到网络问题不能查询天气，请礼貌地告知用户，并且继续推荐景点。如果遇到景区无法正常进入，如票售罄等情况，请及时调整推荐方案。

请开始吧！
"""


# LLM init
class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: list) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # 这里换成了带有基于的列表
                stream=False,
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误: 未找到 TAVILY_API_KEY 环境变量。请设置该环境变量以使用旅游景点推荐功能。"
    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}'{weather}'天气下最值得去的旅游景点推荐理由"
    try:
        # include_answer=True 表示返回综合性回答
        responses = tavily.search(
            query=query, search_depth="basic", include_answer=True
        )
        if responses.get("answers"):
            return responses["answers"]

        # 若没有综合性回答,格式化原始结果
        formatted_results = []
        for result in responses.get("result", []):
            formatted_results.append(f"-{result['title']}:{result['content']}")
            # FIXME:打印看看是什么东西
            # print(formatted_results)
        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为你找到一下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


# skills dictionary
skills = {"get_weather": get_weather, "get_attraction": get_attraction}

if __name__ == "__main__":
    # initialize LLM client
    load_dotenv()
    llm = OpenAICompatibleClient(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    # welcome message
    print("欢迎使用智能旅行助手！")

    # initialize user memory
    user_memory = {
        "preference": "喜欢历史文化景点",
        "budget": "中等预算",
        "history_rejections": [],  # 记录用户拒绝过的景点
    }
    user_query = input("\n✨ 请输入您的旅行相关问题 :")
    # initialize chat history
    chat_history = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT
            + f"\n用户发旅行偏好是:{user_memory['preference']}",
        },
        {"role": "user", "content": user_query},
    ]
    # 开始标记
    START_flag = False
    # 记录不满意的次数
    UNSATISFIED_flag = 0
    while True:
        if START_flag:
            user_query = input("\n✨ 请输入您的旅行相关问题 :")
            # 记录观察结果
            chat_history.append({"role": "user", "content": user_query})
            if user_query.lower() in ["exit", "quit", "退出"]:
                print("\n✨ 很高兴为您服务！")
                break
        else:
            START_flag = True

        # interaction loop
        for i in range(100):
            """
            运行的逻辑参考了给的代码模板，但是有改动，具体的逻辑如下：
            - 使用ChatPromptTemplate来定义prompt模板，使用list数据结构来实现聊天记录的存储，实现记录上下文功能
            - 调用LLM，这里我使用的是deepseek的API
            - 解析Action并执行工具，这里沿用了datawhale给的代码模板
              - 执行工具获得结果
              -将工具的观察结果作为用户反馈存入历史记录

            - 核心逻辑初始化一个消息列表 -> 进入循环 ->处理输出格式-> 生成 Prompt -> 获取 AI 回复 -> 获取工具结果 -> 追加到列表 -> 下一轮
            """
            print(f"--- 循环 {i+1} ---\n")

            # 构建提示词
            # 这里返回的是一个Template对象
            prompt_template = ChatPromptTemplate.from_messages(chat_history)

            # 转换为 LangChain 消息对象
            lc_messages = prompt_template.format_messages()

            messages = []
            for msg in lc_messages:
                # 定义角色映射关系
                role_map = {"human": "user", "ai": "assistant", "system": "system"}
                # 如果是 LangChain 的 AIMessage，msg.type 是 'ai'，需转为 'assistant'
                messages.append(
                    {"role": role_map.get(msg.type, "user"), "content": msg.content}
                )

            # 调用LLM生成回应
            response = llm.generate(messages)
            # 处理多余输出的thought-Action
            match = re.search(
                r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
                response,
                re.DOTALL,
            )
            if match:
                truncated = match.group(1).strip()
                if truncated != response.strip():
                    response = truncated
                    print("已截断多余的 Thought-Action 对")
            print(f"模型输出:\n{response}\n")
            # FIXME:
            # print(response)
            chat_history.append({"role": "assistant", "content": response})

            # 解析并执行行动
            action_match = re.search(r"Action: (.*)", response, re.DOTALL)
            if not action_match:
                print("解析错误:模型输出中未找到 Action。")
                break
            action_str = action_match.group(1).strip()

            # 询问用户环节
            if action_str.startswith("query"):
                final_answer = re.search(r'query\(answer="(.*)"\)', action_str).group(1)
                print(f"\n✨ 智能助手: {final_answer}")
                query_data = input("\n请您回答:")
                chat_history.append(
                    {"role": "user", "content": f"Observation: {query_data}"}
                )
                break

            # 最终回答环节
            if action_str.startswith("finish"):
                final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(
                    1
                )
                print(f"\n✨ 智能助手回答: {final_answer}")
                # 实现询问用户是否满意
                feedback = input("\n您对这个建议还满意吗?(满意/不满意)")
                if "不满意" in feedback:
                    UNSATISFIED_flag += 1
                    # 让用户说出原因
                    reason = input("能告诉我不满意的具体原因吗？")
                    # 更新长期记忆
                    user_memory["preference"] += f"推荐时避开这些因素：{reason}"
                    chat_history.append(
                        {"role": "user", "content": f"Observation: {reason}"}
                    )
                    print(
                        f"✨ 已记录您的偏好。下次我会注意，我将为您重新推荐一个景点，若要退出本次对话请回复exit、quit、退出三者其一"
                    )
                    if UNSATISFIED_flag >= 3:
                        new_system_content = (
                            AGENT_SYSTEM_PROMPT
                            + "\n【重要反思】：用户已连续多次不满意！请彻底放弃之前的推荐思路，尝试更独特或更符合用户避雷要求的方案。"
                        )
                        if chat_history[0]["role"] == "system":
                            chat_history[0]["content"] = new_system_content
                else:
                    print("✨😊 太棒了！很高兴能帮到您。")

                break

            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            args_str = re.search(r"\((.*)\)", action_str).group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

            if tool_name in skills:
                observation = skills[tool_name](**kwargs)
            else:
                observation = f"错误:未定义的工具 '{tool_name}'"

            # 记录观察结果
            chat_history.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
