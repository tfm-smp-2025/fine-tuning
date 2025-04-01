import logging
from .utils import deindent_text, extract_code_blocks, CodeBlock
from .ollama_model import all_models, OllamaModel
from pydantic import BaseModel

class Entity(BaseModel):
    label: str

class EntityList(BaseModel):
    entities: list[Entity]

class PromptWithSearchTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using prompts + searches on a Knowledge Base.
    """

    def __init__(self, base_model: OllamaModel, kb_url: str):
        self.model = base_model
        self.kb_url = kb_url

    def translate(self, query: str) -> str:
        entities = self._get_entities_in_query(query)
        print("Entities:", entities)

    def _get_entities_in_query(self, query: str) -> list[CodeBlock]:
        query_for_llm = deindent_text(
        f"""
        You will have to translate a natural language query into SPARQL. We'll do this in 3 steps:

        1. Find entities in the query.
        2. Find relations in the query.
        3. Generate the final query.

        This is the first step. Which are the entities that will be needed to translate this natural language query to SPARQL?
        
        --- Natural language query ---
        {query}
        --- End of natural language query ---

        Reason step by step and at the end add a ```json block with a list of `entities`, each one with a `label` property. For example:

        ```json
        {{
            "entities": [
                {{
                    "label": "John Doe"
                }}
            ]
        }}
        """
        )
        logging.debug("Query for LLM: {}".format(query_for_llm))
        result = self.model.invoke(query_for_llm)
        code_blocks = extract_code_blocks(result)
        return code_blocks

    def __repr__(self):
        return "{} + prompt & search".format(self.model)


translators = [
    PromptWithSearchTranslator(model, None) for model in all_models
]
