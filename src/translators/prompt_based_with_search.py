"""
This module runs a conversation with the model to help it explore the knowledge base. 
This conversation follows these steps:

1. Given natural language query, extract the relevant entities.
2. (Accessing the KG data with word vector embedding distances) find the classes that can be used to resolve each entity.
3. For each class, find the elements in them that are referred to in the conversation.
4. (Accessing the KG data with word vector embedding distances) find the relations that can be used to resolve each entity.
5. (Via KG) find relevant relationships between the classes or instances.
6. Ask the LLM to generate the SPARQL query.
"""

import tqdm
import json
import logging
from typing import Optional

from pydantic import BaseModel

from ..ontology import property_graph_to_rdf, Ontology
from .. import class_embeddings_caching
from . import text_embeddings, nlp_utils
from .types import LLMModel
from .ollama_model import all_models as ollama_models
from .mistral_model import all_models as mistral_models
from .utils import deindent_text, extract_code_blocks, CodeBlock, url_to_value, deduplicate, deduplicate_on_key, merge_mapping_dicts
from ..structured_logger import get_context


THINGS_URL = "http://www.w3.org/2002/07/owl#Thing"
SKIP_UNLOADED_VECTOR_COLLECTIONS = True

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

        if not text_embeddings.exists_collection(self.ontology.sparql_endpoint, 'classes'):
            classes_on_kg = self.ontology.get_classes_in_kg()
            classes_on_kg = deduplicate_on_key([
                    {
                        'raw': type['type']['value'],
                        'clean': url_to_value(type['type']['value'],),
                    }
                    for type in self.ontology.get_types_in_kg()
                    if ':' in type['type']['value']
                ], key='raw')
            text_embeddings.load_collection(self.ontology.sparql_endpoint, 'classes', classes_on_kg)
            del classes_on_kg

        for entity in tqdm.tqdm(entities, desc='Finding potential classes'):
            candidate_classes = text_embeddings.find_close_in_collection(
                collection_group=self.ontology.sparql_endpoint,
                collection_name='classes',
                reference=entity,
            )
            assert len(candidate_classes) > 0

            logging.info("Found {} close classes for entity “{}”: “{}”".format(
                len(candidate_classes), entity, candidate_classes))

            if len(candidate_classes) == 1:
                entity_mapping[entity] = {
                    'url': candidate_classes[0].raw,
                    'name': candidate_classes[0].clean,
                }
            else:
                entity_mapping[entity] = { 'alternatives': [
                    {
                        'url': alt.raw,
                        'name': alt.clean,
                    }
                    for alt in candidate_classes
                ]}

        # 3. Find singular elements that are inside classes
        messages, elements_in_kg = self._split_singular_and_plural(
            messages,
            entity_mapping,
            nl_query=nl_query,
        )
        
        singulars = elements_in_kg['singular']
        singular_mapping = {}

        things_embeddings = None

        for _class in tqdm.tqdm(singulars, desc='Finding singular elements'):
            mapping = entity_mapping[_class]
            class_instance_candidates = []
            if 'alternatives' not in mapping:
                alts = [mapping]
            else:
                alts = entity_mapping[_class]['alternatives']

            alts.insert(0, { "url": THINGS_URL })

            for alt in alts:
                print("Checking instances of:", alt)

                if not text_embeddings.exists_collection(self.ontology.sparql_endpoint, alt['url']):
                    if alt['url'] != THINGS_URL and SKIP_UNLOADED_VECTOR_COLLECTIONS:
                        continue

                    instances_of = self.ontology.find_instances_of(alt['url'])
                    instances_of = deduplicate_on_key([
                        {
                            'raw': instance,
                            'clean': url_to_value(instance,),
                        }
                        for instance in instances_of
                    ], key='raw')
                    text_embeddings.load_collection(self.ontology.sparql_endpoint, alt['url'], instances_of)
                    del instances_of

                candidate_instances = text_embeddings.find_close_in_collection(
                    collection_group=self.ontology.sparql_endpoint,
                    collection_name=alt['url'],
                    reference=_class,
                )
                assert len(candidate_instances) > 0
                class_instance_candidates.extend([
                        text_embeddings.RankedTerm(
                            clean=term.clean,
                            raw=term.raw,
                            distance=term.distance
                        )
                        for term in candidate_instances
                    ]
                )

                logging.info("Found {} close value for class “{}”: “{}”".format(
                    len(candidate_instances), _class, candidate_instances))

            # 3.5 Apply cutoff across all alternatives
            class_instance_candidates_ranking = sorted(
                class_instance_candidates,
                key=lambda t: t.distance
            )
            prev_class_instance_candidates = class_instance_candidates
            class_instance_candidates = text_embeddings.cutoff_on_max_difference(
                class_instance_candidates_ranking,
            )

            logging.info("Reduced all-instance-candidates for class={}? {} - {} = {}".format(
                _class,
                len(class_instance_candidates),
                len(prev_class_instance_candidates),
                len(class_instance_candidates) - len(prev_class_instance_candidates)
            ))

            if len(class_instance_candidates) == 1:
                singular_mapping[_class] = {
                    'url': class_instance_candidates[0].raw,
                    'name': class_instance_candidates[0].clean,
                }
            else:
                singular_mapping[_class] = { 'alternatives': [
                    {
                        'url': alt.raw,
                        'name': alt.clean,
                    }
                    for alt in class_instance_candidates
                ]}

        # 4. Find relations
        logging.info("Checking relations...")
        relation_mapping = {}
        if not text_embeddings.exists_collection(self.ontology.sparql_endpoint, 'relations'):
            relations_on_kg = deduplicate_on_key([
                    {
                        'raw': type['rel']['value'],
                        'clean': url_to_value(type['rel']['value'],),
                    }
                    for type in self.ontology.get_relation_types_in_kg()
                ], key='raw')
            text_embeddings.load_collection(self.ontology.sparql_endpoint, 'relations', relations_on_kg)
            del relations_on_kg

        for entity in tqdm.tqdm(entities, desc='Finding potential relations'):
            candidate_relations = text_embeddings.find_close_in_collection(
                collection_group=self.ontology.sparql_endpoint,
                collection_name='relations',
                reference=entity,
            )
            assert len(candidate_relations) > 0

            logging.info("Found {} close relations for entity “{}”: “{}”".format(
                len(candidate_relations), entity, candidate_relations))

            candidate_relations = text_embeddings.cutoff_on_max_difference(
                candidate_relations,
            )
            assert len(candidate_relations) > 0

            logging.info("Found {} close relations for entity “{}”: “{}”".format(
                len(candidate_relations), entity, candidate_relations))

            if len(candidate_relations) == 1:
                relation_mapping[entity] = {
                    'url': candidate_relations[0].raw,
                    'name': candidate_relations[0].clean,
                }
            else:
                relation_mapping[entity] = { 'alternatives': [
                    {
                        'url': alt.raw,
                        'name': alt.clean,
                    }
                    for alt in candidate_relations
                ]}

        # 5. Find relations between classes
        logging.info("Checking relations between classes...")
        class_combinations = set()

        all_nodes = set()
        for k_entities in entity_mapping.keys() :
            if 'alternatives' not in entity_mapping[k_entities]:
                k_alts = [entity_mapping[k_entities]]
            else:
                k_alts = entity_mapping[k_entities]['alternatives']

            for k in k_alts:
                all_nodes.add(k['url'])

        for k_entities in singular_mapping.keys() :
            if 'alternatives' not in singular_mapping[k_entities]:
                k_alts = [singular_mapping[k_entities]]
            else:
                k_alts = singular_mapping[k_entities]['alternatives']

            for k in k_alts:
                all_nodes.add(k['url'])

        all_relations = set()
        for k_entities in relation_mapping.keys():
            if 'alternatives' not in relation_mapping[k_entities]:
                k_alts = [relation_mapping[k_entities]]
            else:
                k_alts = relation_mapping[k_entities]['alternatives']

            for k in k_alts:
                all_relations.add(k['url'])

        outgoing_relations_from_nodes = self.ontology.find_relations_outgoing_from_nodes(
            nodes=list(all_nodes),
            predicates=list(all_relations),
        )

        # 6. Generate SPARQL query
        final_query = self._generate_sparql_query(
            messages,
            nl_query,
            mix_mapping(
                merge_mapping_dicts(singular_mapping, entity_mapping),
                relation_mapping,
                outgoing_relations_from_nodes
            ),
        )

        print("Final query:", final_query)

        # Done
        return final_query

    def _generate_sparql_query(
        self,
        messages,
        nl_query, 
        mixed_mapping,
    ):
        query_for_llm = f'''
Given that the entities being referenced are:

```json
{json.dumps(mixed_mapping, indent=4)}
```
'''
        query_for_llm += "\n\nConsider the type of answer to this natural language query. If it's an item list just do a SELECT, but if it's numeric you might need to use a verb like COUNT(), and if it's boolean you might need to use ASK.\n\nConsider what are the necessary relations to solve this query and what are their directions."

        logging.info("Query for LLM (chain of though 1/2): {}".format(query_for_llm))

        get_context().log_operation(
            level='INFO',
            message='Query LLM: {}'.format(query_for_llm),
            operation='query_llm_in',
            data={
                'type': 'generate_sparql_query_cot_1_of_2',
                'input': query_for_llm,
            }
        )
        result = self.model.invoke(messages + [query_for_llm])
        get_context().log_operation(
            level='INFO',
            message='LLM response: {}'.format(result),
            operation='query_llm',
            data={
                'type': 'generate_sparql_query_cot_1_of_2',
                'input': query_for_llm,
                'output': result,
            }
        )

        messages = messages + [query_for_llm, result]

        query_for_llm = 'Construct a SPARQL to solve it on a single query, keep it simple:\n\n'
        query_for_llm += f'> {nl_query}'

        get_context().log_operation(
            level='INFO',
            message='Query LLM: {}'.format(query_for_llm),
            operation='query_llm_in',
            data={
                'type': 'generate_sparql_query_cot_2_of_2',
                'input': query_for_llm,
            }
        )
        
        result = self.model.invoke(messages + [query_for_llm])
        get_context().log_operation(
            level='INFO',
            message='LLM response: {}'.format(result),
            operation='query_llm',
            data={
                'type': 'generate_sparql_query_cot_2_of_2',
                'input': query_for_llm,
                'output': result,
            }
        )

        logging.info("Result: {}".format(result))
        code_blocks = extract_code_blocks(result)

        sparql_code_blocks = [ cb for cb in code_blocks if cb.language == 'sparql' ]

        return sparql_code_blocks[-1]


    def _get_entities_in_query(self, query: str) -> list[CodeBlock]:
        query_for_llm = deindent_text(
        f"""
Extract the nouns from this natural language query.

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


def mix_mapping(node_mapping, relation_mapping, outgoing_relations_from_nodes):
    outgoing_relations_indexed_by_node = {}
    for rel in outgoing_relations_from_nodes:
        if rel.subject not in outgoing_relations_indexed_by_node:
            outgoing_relations_indexed_by_node[rel.subject] = []
        outgoing_relations_indexed_by_node[rel.subject].append(rel.predicate)

    mix = {}
    for k in set(node_mapping.keys()) | set(relation_mapping.keys()):
        options = {'relations': [], 'nodes': []}
        if k in node_mapping:
            if 'alternatives' in node_mapping[k]:
                options['nodes'] = node_mapping[k]['alternatives']
            else:
                options['nodes'] = [node_mapping[k]]
            for node in options['nodes']:
                if node['url'] in outgoing_relations_indexed_by_node:
                    node['outgoing_predicates'] = list(set(outgoing_relations_indexed_by_node[node['url']]))

        if k in relation_mapping:
            if 'alternatives' in relation_mapping[k]:
                options['relations'] = relation_mapping[k]['alternatives']
            else:
                options['relations'] = [relation_mapping[k]]

        mix[k] = options

    return mix


translators = [
    PromptWithSearchTranslator(model, None) for model in ollama_models + mistral_models
]
