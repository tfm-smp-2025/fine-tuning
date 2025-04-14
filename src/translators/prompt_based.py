import logging
from .types import LLMModel
from .utils import deindent_text

from .ollama_model import all_models as ollama_models
from .mistral_model import all_models as mistral_models
from ..ontology import property_graph_to_rdf

class PromptBasedTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using prompts on unmodified LLMs.
    """

    def __init__(self, base_model: LLMModel):
        self.model = base_model
        self.ontology = None
        self.ontology_description = None

    def set_ontology(self, ontology):
        self.ontology = ontology
        self.ontology_description = property_graph_to_rdf(ontology.get_all_properties_in_graph()).serialize(format='pretty-xml')

    def translate(self, nl_query: str) -> str:
        # Naïve test
        prefix = ''
        if self.ontology_description:
            prefix = 'Consider this RDF ontology:\n\n'
            prefix += self.ontology_description

        query_for_llm = deindent_text(
        f"""
{prefix}

Considering those properties, translate this natural language query into a SPARQL query:

--- Natural language query ---
{nl_query}
--- End of natural language query ---
        """
        )
        logging.debug("Query for LLM: {}".format(query_for_llm))
        return self.model.invoke([query_for_llm])

    def __repr__(self):
        return "{} + prompt".format(self.model)


translators = [PromptBasedTranslator(model) for model in ollama_models + mistral_models]
