import logging
import os
from .utils import deindent_text
from langchain_ollama import OllamaLLM

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

MODELS_FILE = os.path.join(ROOT_DIR, "infra", "models.txt")
with open(MODELS_FILE) as f:
    MODELS_NAMES = f.read().split("\n")


class OllamaModel:
    """Class to unify invocation of models that are accessed through Ollama."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[OllamaLLM] = None

    def invoke(self, text: str, verbose=False) -> str:
        if self.model is None:
            self.model = OllamaLLM(model=self.model_name, verbose=verbose)

        return self.model.invoke(text)

    def __repr__(self):
        return "{} on Ollama".format(self.model_name)


class PromptBasedTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using prompts on unmodified LLMs.
    """

    def __init__(self, base_model: OllamaModel):
        self.model = base_model

    def translate(self, query: str) -> str:
        # Naïve test
        query_for_llm = deindent_text(
            f"""
        Translate this natural language query into a SPARQL query:
        
        --- Natural language query ---
        {query}
        --- End of natural language query ---
        """
        )
        logging.debug("Query for LLM: {}".format(query_for_llm))
        return self.model.invoke(query_for_llm)

    def __repr__(self):
        return "{} + prompt".format(self.model)


models = [OllamaModel(model_name) for model_name in MODELS_NAMES]

translators = [PromptBasedTranslator(model) for model in models]
