import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research_agent.tools.models import ToolCall


class QwenLLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=512):
        return self.generate_messages(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
        )

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """生成聊天回复；tools 会进入模型原生 chat template。

        这与普通 Python 函数调用不同：模型只生成结构化 ToolCall，真正的参数
        校验和执行由 ToolExecutor 完成。模型目录必须包含兼容 tools 的模板。
        """
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_args["tools"] = tools
        text = self.tokenizer.apply_chat_template(messages, **template_args)

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

    def generate_tool_calls(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        *,
        max_calls: int = 12,
        max_new_tokens: int = 1200,
    ) -> list[ToolCall]:
        """让支持工具模板的 Qwen 选择工具，并解析标准/常见 Qwen 格式。"""
        raw = self.generate_messages(
            [{"role": "user", "content": prompt}],
            tools=tools,
            max_new_tokens=max_new_tokens,
        )
        values = []
        for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S):
            try:
                values.append(json.loads(block))
            except json.JSONDecodeError:
                continue

        if not values:
            parsed = self._extract_json(raw)
            if isinstance(parsed, dict):
                values = parsed.get("tool_calls") or [parsed]
            elif isinstance(parsed, list):
                values = parsed

        calls = []
        for value in values:
            if not isinstance(value, dict):
                continue
            try:
                call = ToolCall.from_mapping(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if call.name:
                calls.append(call)
            if len(calls) >= max_calls:
                break
        return calls

    @staticmethod
    def _extract_json(text: str) -> Any:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text or ""):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
        return None
