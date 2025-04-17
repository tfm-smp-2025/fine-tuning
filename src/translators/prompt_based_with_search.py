"""
This module runs a conversation with the model to help it explore the knowledge base. 
This conversation follows these steps:

1. Given natural language query, extract the relevant entities.
2. (Accessing the KG data with word vector embedding distances) find the classes that can be used to resolve each entity.
3. For each class, find the elements in them that are referred to in the conversation.
4. (Via KG) find relevant relationships between the classes or instances.
5. Ask the LLM to generate the SPARQL query.
"""

import json
import logging
from typing import Optional

from pydantic import BaseModel

from ..ontology import property_graph_to_rdf, Ontology

from . import text_embeddings, nlp_utils
from .types import LLMModel
from .ollama_model import all_models as ollama_models
from .mistral_model import all_models as mistral_models
from .utils import deindent_text, extract_code_blocks, CodeBlock, url_to_value, deduplicate
from ..structured_logger import get_context


class Entity(BaseModel):
    label: str

class EntityList(BaseModel):
    entities: list[Entity]

class PromptWithSearchTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using prompts + searches on a Knowledge Base.
    """

    def __init__(self, base_model: LLMModel, kb_url: str):
        self.model = base_model
        self.kb_url = kb_url
        self.ontology: Optional[Ontology] = None
        self.ontology_description = None

    def set_ontology(self, ontology):
        self.ontology = ontology
        self.ontology_description = property_graph_to_rdf(ontology.get_all_properties_in_graph()).serialize(format='pretty-xml')

    def translate(self, nl_query: str) -> str:
        assert self.ontology, \
            "PromptWithSearchTranslator requires access to the KG to be used"

        # 1. Get entities
        messages, entities_str = self._get_entities_in_query(nl_query)
        print("Entities:", entities_str)
        entities = json.loads(entities_str[-1].content)

        # 2. Map entities to potential classes
        entity_mapping = {}
        classes_on_kg = deduplicate(self.ontology.get_classes_in_kg())
        classes_are_types = len(classes_on_kg) == 0
        classes_on_kg = deduplicate([
                type['type']['value']
                for type in self.ontology.get_types_in_kg()
                if ':' in type['type']['value']
            ])

        cleaned_classes = [
            url_to_value(_class)
            for _class in classes_on_kg
        ]
        for entity in entities:
            entity_ranking = text_embeddings.rank_by_similarity(
                entity,
                cleaned_classes,
            )
            cutoff = text_embeddings.cutoff_on_max_difference(
                entity_ranking,
            )
            assert len(cutoff) > 0

            logging.info("Found {} close classes for entity “{}”: “{}”".format(
                len(cutoff), entity, cutoff))

            if len(cutoff) == 1:
                entity_mapping[entity] = {
                    'url': classes_on_kg[cutoff[0].original_index],
                    'name': cutoff[0].text,
                }
            else:
                # TODO: Query LLM?
                entity_mapping[entity] = { 'alternatives': [
                    {
                        'url': classes_on_kg[alt.original_index],
                        'name': alt.text,
                    }
                    for alt in cutoff
                ]}

        # 3. Find singular elements that are inside classes
        messages, elements_in_kg = self._split_singular_and_plural(
            messages,
            entity_mapping,
            nl_query=nl_query,
        )
        
        singulars = elements_in_kg['singular']
        singular_mapping = {}
        for _class in singulars:
            mapping = entity_mapping[_class]
            if 'alternatives' in mapping:
                raise NotImplemented('Alternatives in entity mapping')

            print("Checking instances of:", mapping)

            listing = self.ontology.find_instances_of(mapping['url'])
            cleaned_listing = [
                url_to_value(value)
                for value in listing
            ]
            ranking = text_embeddings.rank_by_similarity(
                _class,
                cleaned_listing,
            )
            cutoff = text_embeddings.cutoff_on_max_difference(
                ranking,
            )
            assert len(cutoff) > 0

            logging.info("Found {} close value for class “{}”: “{}”".format(
                len(cutoff), _class, cutoff))

            if len(cutoff) == 1:
                singular_mapping[_class] = {
                    'url': listing[cutoff[0].original_index],
                    'name': cutoff[0].text,
                }
            else:
                # TODO: Query LLM?
                singular_mapping[_class] = { 'alternatives': [
                    {
                        'url': listing[alt.original_index],
                        'name': alt.text,
                    }
                    for alt in cutoff
                ]}
        
        # 4. Find relations
        logging.info("Checking relations...")
        class_combinations = set()
        class_relations = {}
        for k1 in entity_mapping.keys():
            for k2 in entity_mapping.keys():
                c1 = entity_mapping[k1]['url']
                c2 = entity_mapping[k2]['url']

                if c1 == c2:
                    continue

                if (c1, c2) in class_combinations:
                    continue

                # Find relations
                if classes_are_types:
                    c1_to_c2 = self.ontology.find_relations_between_type_objects(c1, c2)
                else:
                    c1_to_c2 = self.ontology.find_relations_between_class_objects(c1, c2)
                class_relations[(c1, c2)] = c1_to_c2

                if classes_are_types:
                    c2_to_c1 = self.ontology.find_relations_between_type_objects(c2, c1)
                else:
                    c2_to_c1 = self.ontology.find_relations_between_class_objects(c2, c1)
                class_relations[(c2, c1)] = c2_to_c1

        logging.info("Class relations: {}".format(class_relations))

        # 5. Generate SPARQL query
        final_query = self._generate_sparql_query(
            messages,
            nl_query,
            singular_mapping,
            class_relations,
        )

        print("Final query:", final_query)

        # Done
        return final_query

    def _generate_sparql_query(
        self,
        messages,
        nl_query, 
        singular_mapping,
        class_relations,
    ):
        query_for_llm = f'''
Given that the entities being referenced are:

```json
{json.dumps(singular_mapping, indent=4)}
```
'''
        if sum([len(relations) for relations in class_relations.values()]) == 0:
            # Skip this if there's not relation to be used
            logging.warning("No useful relations found")
        else:
            query_for_llm += '\nAnd knowing that the following classes are related:\n'

            for classes, relations in class_relations.items():
                c1, c2 = classes

                if len(relations) == 0:
                    continue  # Ignore this combination
                if len(relations) == 1:
                    via = relations[0]
                else:
                    via = '", "'.format(relations[:-1]) + '" and "' + relations[-1] + '"'

                query_for_llm += f'- ":{url_to_value(c1)}" to ":{url_to_value(c2)}" via "{via}"'

        query_for_llm += "\n\nConsider the type of answer to this natural language query. If it's an item list just do a SELECT, but if it's numeric you might need to use a verb like COUNT(), and if it's boolean you might need to use ASK.\n\nConstruct a SPARQL to solve it on a single query, keep it simple:\n\n"
        query_for_llm += '> ' + nl_query

        logging.info("Query for LLM: {}".format(query_for_llm))

        get_context().log_operation(
            level='INFO',
            message='Query LLM: {}'.format(query_for_llm),
            operation='query_llm_in',
            data={
                'type': 'generate_sparql_query',
                'input': query_for_llm,
            }
        )
        result = self.model.invoke(messages + [query_for_llm])
        get_context().log_operation(
            level='INFO',
            message='LLM response: {}'.format(result),
            operation='query_llm',
            data={
                'type': 'generate_sparql_query',
                'input': query_for_llm,
                'output': result,
            }
        )

        logging.info("Result: {}".format(result))
        code_blocks = extract_code_blocks(result)

        sparql_code_blocks = [ cb for cb in code_blocks if cb.language == 'sparql' ]

        return sparql_code_blocks[-1]


    def _get_entities_in_query(self, query: str) -> list[CodeBlock]:
        prefix = ''
        if self.ontology_description:
            prefix = 'Consider this RDF ontology:\n\n'
            prefix += self.ontology_description

        query_for_llm = deindent_text(
        f"""
Considering those properties, extract the nouns from this natural language query.

> {query}

Let's reason step by step. Identify the nouns on the query, skip the ones that can be solved by a SPARQL verb (ignore, for example, "count" or "number of"), and output a json list like this.

```json
[
    "entity1",
    "entity2",
    ...
    "entityN"
]
```""")

        get_context().log_operation(
            level='INFO',
            message='Query LLM: {}'.format(query_for_llm),
            operation='query_llm_in',
            data={
                'type': 'get_entities_in_query',
                'input': query_for_llm,
            }
        )
        result = self.model.invoke([query_for_llm])
        get_context().log_operation(
            level='INFO',
            message='LLM response: {}'.format(result),
            operation='query_llm',
            data={
                'type': 'get_entities_in_query',
                'input': query_for_llm,
                'output': result,
            }
        )

        code_blocks = extract_code_blocks(result)
        return [
            query_for_llm, result,
        ], [
            cb
            for cb in code_blocks
            if cb.language == 'json'
        ]

    def _split_singular_and_plural(self, messages, entity_mapping, nl_query):
        singulars = []
        plurals = []
        for item in entity_mapping.keys():
            sing = nlp_utils.is_singular(item)    
            get_context().log_operation(
                level='INFO',
                message='Checking if "{}" is singular'.format(item),
                operation='checking_singular_plural',
                data={
                    'input': item,
                    'singular': sing,
                }
            )
            if sing:
                singulars.append(item)
            else:
                plurals.append(item)

        return messages, {'singular': singulars, 'plural': plurals}

    def __repr__(self):
        return "{} + prompt & search".format(self.model)


translators = [
    PromptWithSearchTranslator(model, None) for model in ollama_models + mistral_models
]
