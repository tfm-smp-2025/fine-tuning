import os
from langchain_mistralai import ChatMistralAI
from .types import LLMModel

MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']

MODELS = [
    # See: https://docs.mistral.ai/getting-started/models/models_overview/
    'open-mistral-7b',
]


class MistralModel(LLMModel):
    """Class to unify invocation of models that are accessed through Ollama."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[ChatMistralAI] = None

    def invoke(self, messages: list[str], verbose=True) -> str:
        if self.model is None:
            self.model = ChatMistralAI(model=self.model_name, verbose=verbose)

        input = []
        has_system = False

        for idx, msg in enumerate(messages):
            actor = "user" if idx % 2 == int(has_system) else "ai"
            input.append((actor, msg))

        return self.model.invoke(input=input).content

    def __repr__(self):
        return "{} on API".format(self.model_name)

all_models = [
    MistralModel(model_name) for model_name in MODELS
]
