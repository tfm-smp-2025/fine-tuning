import os
from langchain_ollama import OllamaLLM
from .types import LLMModel

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

MAX_STEP_TOKENS = int(os.getenv('MAX_STEP_TOKENS', -1))

MODELS_FILE = os.path.join(ROOT_DIR, "infra", "models.txt")
with open(MODELS_FILE) as f:
    MODELS_NAMES = f.read().strip().split("\n")

LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.2))

class OllamaModel(LLMModel):
    """Class to unify invocation of models that are accessed through Ollama."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[OllamaLLM] = None

    def invoke(self, messages: list[str], verbose=True, temperature=None) -> str:
        if self.model is None:
            temperature = temperature or LLM_TEMPERATURE
            self.model = OllamaLLM(
                model=self.model_name, verbose=verbose, temperature=temperature
            )

        result = []
        for chunk in self.model.stream(messages):
            result.append(chunk)
            print(chunk, end='', flush=True)

            if MAX_STEP_TOKENS > 0 and len(result) > MAX_STEP_TOKENS:
              result.append("\n{I think this is enough}\n")
              print(f"\nForcing stop: {result[-1].strip()}", end='', flush=True)
              break
        return ''.join(result)

    def __repr__(self):
        return "{} on Ollama".format(self.model_name)


all_models = [OllamaModel(model_name) for model_name in MODELS_NAMES]
