import logging

import tqdm

from .translators import all_translators
from .datasets import dataset_loader
from .translators.utils import extract_code_blocks, CodeBlock
from .ontology import Ontology, property_graph_to_rdf
from .structured_logger import StructuredLogger, get_logger

def run_test(args):
    """Run some tests over the configured models."""
    sample_size = args.sample
    is_full = args.full

    logger = get_logger()

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
                logging.debug("SKIPPING, model not selected")
                continue

            if ontology:
                translator.set_ontology(ontology)

            dataset_counter = 0
            for question in tqdm.tqdm(ds.get_test_data()):
                with logger.context(
                    "({}) DATASET: {}".format(translator, ds.name),
                    {
                        "translator": {
                            "model_name": translator.model.model_name,
                        },
                        "question": question,
                        "dataset": {
                            "name": ds.name,
                            "sparql_endpoint": ontology.sparql_endpoint if ontology is not None else None,
                        }
                    }
                ) as ctxt:
                    if question.lang not in ('en', None):
                        logging.debug('SKIPPING question in non-english: {}'.format(question.lang))
                        continue

                    dataset_counter += 1
                    if sample_size is not None and dataset_counter > sample_size:
                        logging.debug('Closing dataset after {} elements tested'.format(sample_size))

                    ctxt.log_operation(
                        level='INFO',
                        message='Input: {}'.format(question.question),
                        operation='input_question',
                        data=question.question,
                    )

                    expected_result = None
                    try:
                        if ontology:
                            expected_result = ontology.run_query(question.answer)
                            ctxt.log_operation(
                                level='INFO',
                                message='Expected query result: {}'.format(expected_result),
                                operation='expected_result',
                                data=expected_result,
                            )
                        else:
                            expected_result = None

                        result = translator.translate(question.question)

                        ctxt.log_operation(
                            level='INFO',
                            message='Translated query: {}'.format(result),
                            operation='translated_query',
                            data=result,
                        )

                        if isinstance(result, CodeBlock):
                            sparql_code_blocks = [result]
                        else:
                            sparql_code_blocks = [
                                block.content
                                for block in extract_code_blocks(text=result)
                                if block.language.lower() == 'sparql'
                            ]

                        if ontology:
                            logging.info("TESTING query: {}".format(sparql_code_blocks[-1]))
                            translator_result = ontology.run_query(sparql_code_blocks[-1].content)
                            ctxt.log_operation(
                                level='INFO',
                                message='Translated query result: {}'.format(translator_result),
                                operation='translated_query_result',
                                data=translator_result,
                            )

                            ctxt.log_operation(
                                level='INFO',
                                message="Test result: {}".format(translator_result == expected_result),
                                operation='test_result',
                                data={
                                    'input': question.question,
                                    'expected_query': question.answer,
                                    'expected_result': expected_result,
                                    'found_answer': result,
                                    'found_result': translator_result,
                                    'result': translator_result == expected_result,
                                }
                            )
                        else:
                            ctxt.log_operation(
                                level='INFO',
                                message="Test result: -unknown-",
                                operation='test_result',
                                data={
                                    'input': question.question,
                                    'expected_query': question.answer,
                                    'found_answer': result,
                                    'result': None,
                                }
                            )

                    except KeyboardInterrupt:
                        logging.fatal("Stopping due to Keyboard Interrupt")
                        ctxt.log_operation(
                            level='INFO',
                            message="Test result: cancelled",
                            operation='test_result',
                            data={
                                'input': question.question,
                                'expected_query': question.answer,
                                'expected_result': expected_result,
                                'result': 'cancelled',
                            }
                        )
                        raise
                    except:
                        logging.exception("EXCEPTION".format(translator))
                        ctxt.log_operation(
                            level='INFO',
                            message="Test result: error",
                            operation='test_result',
                            data={
                                'input': question.question,
                                'expected_query': question.answer,
                                'expected_result': expected_result,
                                'result': 'error',
                            }
                        )
