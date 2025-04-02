import os
from langchain_ollama import OllamaLLM

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

MODELS_FILE = os.path.join(ROOT_DIR, "infra", "models.txt")
with open(MODELS_FILE) as f:
    MODELS_NAMES = f.read().strip().split("\n")


class OllamaModel:
    """Class to unify invocation of models that are accessed through Ollama."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[OllamaLLM] = None

    def invoke(self, text: str, verbose=True) -> str:
        if self.model is None:
            self.model = OllamaLLM(model=self.model_name, verbose=verbose)

        return self.model.invoke(text)

    def __repr__(self):
        return "{} on Ollama".format(self.model_name)

all_models = [OllamaModel(model_name) for model_name in MODELS_NAMES]
