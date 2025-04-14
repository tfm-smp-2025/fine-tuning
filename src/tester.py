import logging

import tqdm

from .translators import all_translators
from .datasets import dataset_loader
from .translators.utils import extract_code_blocks, CodeBlock
from .ontology import Ontology, property_graph_to_rdf

def run_test(args):
    """Run some tests over the configured models."""
    sample_size = args.sample
    is_full = args.full

    assert (
        sample_size is not None
    ) ^ is_full, "Expected either --sample SIZE or --full, found: {}".format(args)

    for dataset_name in args.datasets:
        ds = dataset_loader.load_dataset(dataset_name)

        ontology = None
        if args.sparql_server:
            ontology = Ontology(args.sparql_server, ds.sparql_endpoint)

        for translator in all_translators:
            if translator.model.model_name not in args.models:
                logging.debug("({}) SKIPPING, model not selected".format(translator))
                continue

            if ontology:
                translator.set_ontology(ontology)

            logging.info("({}) DATASET: {}".format(translator, ds))

            dataset_counter = 0
            for question in tqdm.tqdm(ds.get_test_data()):
                if question.lang not in ('en', None):
                    logging.debug('SKIPPING question in non-english: {}'.format(question.lang))
                    continue

                dataset_counter += 1
                if sample_size is not None and dataset_counter > sample_size:
                    logging.debug('Closing dataset after {} elements tested'.format(sample_size))

                logging.info("({}) INPUT: {}".format(translator, question.question))

                try:
                    if ontology:
                        expected_result = ontology.run_query(question.answer)
                        logging.info("({}) EXPECTED RESULT: {}".format(translator, expected_result))

                    result = translator.translate(question.question)

                    logging.info("({}) RESULT query: {}".format(translator, result))
                    if isinstance(result, CodeBlock):
                        sparql_code_blocks = [result]
                    else:
                        sparql_code_blocks = [
                            block.content
                            for block in extract_code_blocks(text=result)
                            if block.language.lower() == 'sparql'
                        ]

                    if ontology:
                        logging.info("({}) TESTING query: {}".format(translator, sparql_code_blocks[-1]))
                        translator_result = ontology.run_query(sparql_code_blocks[-1].content)
                        logging.info("({}) TRANSLATOR RESULT: {}".format(translator, translator_result))

                except KeyboardInterrupt:
                    logging.error("Stopping due to Keyboard Interrupt")
                    raise
                except:
                    logging.exception("({}) EXCEPTION".format(translator))
                    # raise
