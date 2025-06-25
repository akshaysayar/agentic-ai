from abc import ABC
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

model_path = Path(__file__).parent.parent / "models" / "legal-bert"
# Loading the model
llm_model = AutoModel.from_pretrained(model_path)

# Loading the tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(model_path)


# from research_tool_rag.rag.model import llm, OllamaStreamingLLM

import json
from typing import Generator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class OllamaStreamingLLM(LLM):
    model: str = "mistral"
    url: str = "http://localhost:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Simple implementation: request with stream=False and return full text
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[GenerationChunk, None, None]:
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True,
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                if run_manager:
                    run_manager.on_llm_new_token(token)
                yield GenerationChunk(text=token)

    @property
    def _llm_type(self) -> str:
        return "ollama-streaming"


class Mistral(LLM):
    model: str = "mistral"
    url: str = "http://localhost:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    @property
    def _llm_type(self) -> str:
        return "deepseek-local"


llm = Mistral()


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
