import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
