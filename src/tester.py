import logging
import tqdm

from .translators import all_translators
from .datasets import dataset_loader


def run_test(args):
    """Run some tests over the configured models."""
    sample_size = args.sample
    is_full = args.full

    assert (
        sample_size is not None
    ) ^ is_full, "Expected either --sample SIZE or --full, found: {}".format(args)


    for translator in all_translators:
        if translator.model.model_name not in args.models:
            logging.debug("({}) SKIPPING, model not selected".format(translator))
            continue

        for dataset_name in args.datasets:
            ds = dataset_loader.load_dataset(dataset_name)
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
                    result = translator.translate(question.question)

                    logging.info("({}) RESULT query: {}".format(translator, result))

                except KeyboardInterrupt:
                    raise
                except:
                    raise
                    logging.exception("({}) EXCEPTION".format(translator))
