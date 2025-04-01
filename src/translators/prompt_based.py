import logging
from .utils import deindent_text
from .ollama_model import all_models, OllamaModel

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

translators = [
    PromptBasedTranslator(model) for model in all_models
]