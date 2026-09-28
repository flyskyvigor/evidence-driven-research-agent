"""不加载模型和网络的确定性替身。"""

from __future__ import annotations

import json


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, max_new_tokens=512):
        if not self.responses:
            raise AssertionError("Fake LLM response queue exhausted")
        value = self.responses.pop(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def generate_tool_calls(self, prompt, tools, **kwargs):
        return []

