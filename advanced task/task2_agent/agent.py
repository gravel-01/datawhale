import json5
import re
import time
import os
import sys
from dotenv import load_dotenv
from llm import OpenAICompatibleClient

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "tool"))
from tool.tool import *


class ReactAgent:
    def __init__(self, api_key: str = "", url: str = "") -> None:
        self.api_key = api_key
        self.tools = ReactTools()
        self.model = OpenAICompatibleClient(
            model="deepseek-chat",
            api_key=api_key,
            base_url=url,
        )
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示，直接从工具类获取描述"""
        tool_info = []
        for tool in self.tools.toolConfig:
            tool_info.append(
                f"- {tool['name_for_model']}: {tool['description_for_model']}"
            )

        tool_names = list(self.tools._tools_map.keys())

        prompt = f"""现在时间是 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}。你是一位智能助手，可以使用以下工具：

{chr(10).join(tool_info)}

请遵循以下 ReAct 模式：


思考：分析问题和需要使用的工具
行动：选择工具 [{', '.join(tool_names)}] 中的一个
行动输入：提供工具的参数
观察：工具返回的结果

你可以重复以上循环，直到获得足够的信息来回答问题。

最终答案：基于所有信息给出最终答案

开始！"""
        return prompt

    # 解析大模型的回答
    def _parse_action(self, text: str, verbose: bool = False) -> tuple[str, dict]:
        """从文本中解析行动和行动输入"""
        # 更灵活的正则表达式模式
        action_pattern = r"行动[:：]\s*(\w+)"
        action_input_pattern = r"行动输入[:：]\s*({.*?}|\{.*?\}|[^\n]*)"

        action_match = re.search(action_pattern, text, re.IGNORECASE)
        action_input_match = re.search(action_input_pattern, text, re.DOTALL)

        action = action_match.group(1).strip() if action_match else ""
        action_input_str = (
            action_input_match.group(1).strip() if action_input_match else ""
        )

        # 清理和解析JSON
        action_input_dict = {}
        if action_input_str:
            try:
                # 尝试解析为JSON对象
                action_input_str = action_input_str.strip()
                if action_input_str.startswith("{") and action_input_str.endswith("}"):
                    action_input_dict = json5.loads(action_input_str)
                else:
                    # 如果不是JSON格式，尝试解析为简单字符串参数
                    action_input_dict = {"search_query": action_input_str.strip("\"'")}
            except Exception as e:
                if verbose:
                    print(f"[ReAct Agent] 解析参数失败，使用字符串作为搜索查询: {e}")
                action_input_dict = {"search_query": action_input_str.strip("\"'")}

        return action, action_input_dict

    # TODO:这里要改成更加通用的形式
    def _execute_action(self, action: str, action_input: dict) -> str:
        """执行指定的行动，使用解耦后的 tools 管理器"""
        # 检查工具是否存在于我们的注册表中
        if action in self.tools._tools_map:
            try:
                # 动态调用工具函数并传入参数
                # 使用 **action_input 将字典解包为命名参数
                results = self.tools.execute_tool(action, **action_input)
                return f"观察：{results}"
            except Exception as e:
                return f"观察：执行工具 {action} 时出错: {str(e)}"

        return f"观察：未知行动 '{action}'，请尝试从已知工具列表中选择。"

    def _format_response(self, response_text: str) -> str:
        """格式化最终响应"""
        if "最终答案：" in response_text:
            return response_text.split("最终答案：")[-1].strip()
        return response_text

    def run(self, query: str, max_iterations: int = 3, verbose: bool = True) -> str:
        """运行 ReAct Agent

        Args:
            query: 用户查询
            max_iterations: 最大迭代次数
            verbose: 是否显示中间执行过程
        """
        chat_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{query}"},
        ]

        # 绿色ANSI颜色代码
        GREEN = "\033[92m"
        RESET = "\033[0m"

        if verbose:
            print(f"{GREEN}[ReAct Agent] 开始处理问题: {query}{RESET}")

        for iteration in range(max_iterations):
            if verbose:
                print(f"{GREEN}[ReAct Agent] 第 {iteration + 1} 次思考...{RESET}")

            # 获取模型响应
            response = self.model.generate(chat_history)

            if verbose:
                print(f"{GREEN}[ReAct Agent] 模型响应:\n{response}{RESET}")

            chat_history.append({"role": "assistant", "content": response})
            # 解析行动
            action, action_input = self._parse_action(response, verbose=verbose)

            if not action or action == "最终答案" or "最终答案：" in response:
                final_answer = self._format_response(response)
                if verbose:
                    print(f"{GREEN}[ReAct Agent] 任务完成{RESET}")
                return final_answer

            if verbose:
                print(
                    f"{GREEN}[ReAct Agent] 执行行动: {action} | 参数: {action_input}{RESET}"
                )

            # 执行行动
            observation = self._execute_action(action, action_input)

            if verbose:
                print(f"{GREEN}[ReAct Agent] 观察结果:\n{observation}{RESET}")

            # 更新当前文本以继续对话
            chat_history.append({"role": "user", "content": observation})

        # 达到最大迭代次数，返回当前响应
        if verbose:
            print(f"{GREEN}[ReAct Agent] 达到最大迭代次数，返回当前响应{RESET}")
        return self._format_response(response)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1"
    agent = ReactAgent(api_key=api_key, url=url)

    response = agent.run(
        "美国最近一次阅兵的原因有哪些？", max_iterations=3, verbose=True
    )
    print("最终答案：", response)
