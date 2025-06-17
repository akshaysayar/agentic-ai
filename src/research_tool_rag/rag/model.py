from abc import ABC
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

model_path = Path(__file__).parent.parent / "models" / "legal-bert"
# Loading the model
llm_model = AutoModel.from_pretrained(model_path)

# Loading the tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(model_path)


class LLMModel(ABC):
    def __init__(self, model_path: Path):
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(self, text: str): ...

    def decode(self, encoded_text): ...


class onlineLLMModel(LLMModel):
    def __init__(self, model_path: Path):
        super().__init__(model_path)

    def generate(self, prompt: str, max_length: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class offlineLLMModel(LLMModel):
    def __init__(self, model_path: Path):
        super().__init__(model_path)

    def summarize(self, text: str, max_length: int = 100):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
