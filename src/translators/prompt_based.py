import logging
from .ollama_model import OllamaModel, all_models
from .utils import deindent_text

class PromptBasedTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using prompts on unmodified LLMs.
    """

    def __init__(self, base_model: OllamaModel):
        self.model = base_model
        self.ontology = None

    def set_ontology(self, ontology):
        self.ontology = ontology

    def translate(self, query: str) -> str:
        # Naïve test
        prefix = ''
        if self.ontology:
            prefix = 'Consider this RDF ontology:\n\n'
            prefix += self.ontology

        query_for_llm = deindent_text(
        f"""
{prefix}

Considering those properties, translate this natural language query into a SPARQL query:

--- Natural language query ---
{query}
--- End of natural language query ---
        """
        )
        logging.debug("Query for LLM: {}".format(query_for_llm))
        return self.model.invoke(query_for_llm)

    def __repr__(self):
        return "{} + prompt".format(self.model)


translators = [PromptBasedTranslator(model) for model in all_models]
