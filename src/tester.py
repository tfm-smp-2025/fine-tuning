import logging

from .translators import all_translators


def run_test(args):
    """Run some tests over the configured models."""
    sample_size = args.sample
    is_full = args.full

    assert (
        sample_size is not None
    ) ^ is_full, "Expected either --sample SIZE or --full, found: {}".format(args)

    TEST_PROMPT = "when was father chris riley born?"
    for translator in all_translators:
        if translator.model.model_name not in args.models:
            logging.info("({}) SKIPPING, model not selected".format(translator))
            continue

        logging.info("({}) INPUT: {}".format(translator, TEST_PROMPT))
        try:
            result = translator.translate(TEST_PROMPT)
            logging.info("({}) RESULT: {}".format(translator, result))
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("({}) EXCEPTION".format(translator))
