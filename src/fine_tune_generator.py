import json
import logging
import subprocess
import tempfile
import tqdm

from .datasets import dataset_loader
from .translators.utils import deindent_text
from .translators.prompt_based_with_search import trainer as fine_tuning_trainer
from .translators.types import FineTuneGenSpecificError
from .structured_logger import get_logger
from .ontology import Ontology

def generate(args):
    """Generate fine-tuning data."""
    trainer = fine_tuning_trainer
    logger = get_logger()

    found_anchor = True # False
    for dataset_name in args.datasets:
        dataset = dataset_loader.load_dataset(dataset_name, rand_seed=args.seed)
        data = {}

        if args.sparql_server:
            ontology = Ontology(args.sparql_server, dataset.sparql_endpoint)
            trainer.set_ontology(ontology)

        for ds_name, ds_data in zip(("train", "test"), dataset.get_split_dataset()):
            sub_dataset = []
            if ds_name != 'test':
                # TODO: Remove to generate testing data
                print("SKIPPING {} dataset".format(ds_name))
                continue

            with logger.context(
                context_name="Fine tuning on {}/{}".format(dataset.name, ds_name,),
                context_params={
                    "flow": "gen-fine-tuning-data",
                    "dataset_name": dataset.name,
                    "split": ds_name,
                }
            ):
                num_rows = len(ds_data)
                for item_idx, item in tqdm.tqdm(enumerate(ds_data), desc=f"{dataset_name} for {ds_name}"):
                    if item.lang not in (None, "en"):
                        logger.debug(
                            "SKIPPING question in non-english: {}".format(item.lang)
                        )
                        continue

                    if item.question == 'List the popular works of the author of Luther: The Calling ?':
                        found_anchor = True
                        continue
                    elif not found_anchor:
                        continue

                    with logger.context(
                        context_name="[Q-{}/{}]Fine tuning on {}/{}".format(
                            item_idx + 1, num_rows,
                            dataset.name, ds_name,
                            ),
                        context_params={
                            "flow": "gen-fine-tuning-data",
                            "dataset_name": dataset.name,
                            "split": ds_name,
                            "question_index": item_idx,
                            "num_questions": num_rows,
                        }
                    ):
                        try:
                            row = trainer.gen_expected_conversation_data(
                                item,
                            )
                        except FineTuneGenSpecificError:
                            logging.exception('Skipping this case')
                            continue

                        if args.split_test:
                            out = args.output
                            if ds_name == "test":
                                out = args.split_test
                            out.write(json.dumps(row) + "\n")
                            out.flush()
                        else:
                            sub_dataset.append(row)
                data[ds_name] = sub_dataset

        if not args.split_test:
            args.output.write(json.dumps(data, indent=4))
        else:
            args.split_test.close()
        args.output.close()
