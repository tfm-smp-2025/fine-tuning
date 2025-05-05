import os
import logging
from .types import LLMModel
from .utils import deindent_text

from .ollama_model import all_models as ollama_models
from .mistral_model import all_models as mistral_models
from ..ontology import property_graph_to_rdf

from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from ..structured_logger import get_context

LOCAL_COPY_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ), "..", "..", "local_graphs")


def url_to_slug(ds_name: str) -> str:
    """Convert a URL to a format more adequate for directory name."""
    # TODO: De-duplicate with the `pull_datasets` script and `dataset_loader
    return ds_name.lower().replace(":", "_").replace(" ", "_")


class RdflibBasedTranslator:
    """
    Implementation of a class to translate natural language queries into SPARQL queries
    by using Langchain's RDFLib module.
    """

    def __init__(self, base_model: LLMModel):
        self.model = base_model
        self.graph = None
        self.chain = None

    def set_ontology(self, ontology):
        local_copy = os.path.join(
            LOCAL_COPY_DIR,
            url_to_slug(ontology.sparql_server),
            ontology.sparql_endpoint + '.ttl'
        )
        os.makedirs(os.path.dirname(local_copy), exist_ok=True)
        store_kwargs = {}
        if 'KB_PASSWORD' in os.environ:
            store_kwargs = {"auth": ('admin', os.environ['KB_PASSWORD'])}

        self.graph = RdfGraph(
            # source_file=ontology.sparql_server.strip('/') + '/' + ontology.sparql_endpoint,
            query_endpoint=ontology.sparql_server.strip('/') + '/' + ontology.sparql_endpoint,
            standard="rdf",
            local_copy=local_copy,
            store_kwargs=store_kwargs,
        )
        self.chain = GraphSparqlQAChain.from_llm(
            self.model,
            graph=self.graph,
            verbose=True,
            return_sparql_query=True,
        )

    def translate(self, nl_query: str) -> str:
        assert self.chain, "Chain not initialized, call `.set_ontology()`"
        result = self.chain.run(nl_query)
        get_context().log_operation(
            level="INFO",
            message="Rdflib answer: {}".format(result),
            operation="query_llm_in",
            data={
                "type": "rdflib_translate",
                "input": nl_query,
                "result": result,
            },
        )
        return result['sparql_query']

    def __repr__(self):
        return "{} + prompt".format(self.model)


translators = [RdflibBasedTranslator(model) for model in ollama_models + mistral_models]
