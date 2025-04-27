import argparse
import logging
import random
import sys
from . import tester
from . import fine_tune_generator
from . import ontology
from .translators.ollama_model import all_models


def get_argparser():
    parser = argparse.ArgumentParser("SPARQL LLM fine-tuner")
    parser.add_argument("--seed", type=int, help="Seed used to for random choices")

    subparser = parser.add_subparsers(help="subcommand help", required=True)

    test_subparser = subparser.add_parser("test")
    set_test = test_subparser.add_mutually_exclusive_group(required=True)
    set_test.add_argument(
        "--sample", type=int, help="Use only a sample of (n) queries to run the test"
    )
    set_test.add_argument(
        "--full", action="store_true", help="Run the test with the full query set"
    )
    test_subparser.add_argument(
        "--sample-offset", type=int,
        default=0,
        help="Skip the first (n) queries from the test sample",
    )

    test_subparser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=[model.model_name for model in all_models],
    )

    test_subparser.add_argument(
        "--datasets", nargs="+", type=str, default=["beastiary", "LC-QuAD 1.0"]
    )
    test_subparser.add_argument(
        "--sparql-server",
        required=False,
        type=str,
        help='The address of the SPARQL server to test the queries on. In the format "http://127.0.0.1:3030".',
    )
    test_subparser.set_defaults(func=tester.run_test)

    fine_tune = subparser.add_parser("gen-fine-tuning-data")
    fine_tune.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=[
            "beastiary",
            "qald-9",
            "qald-10",
            "lc-quad_1.0",
            "vquanda",
            "lc-quad_2.0",
            "webquestions_sp",
        ],
    )

    fine_tune.add_argument(
        "--output",
        required=True,
        help="Name of the `.json` file to be generated",
        type=argparse.FileType("w"),
    )

    fine_tune.add_argument(
        "--split-test",
        required=False,
        help="Generate as two files, this argument will point to the one with the test set",
        type=argparse.FileType("w"),
    )

    fine_tune.set_defaults(func=fine_tune_generator.generate)

    extract_ontology_subparser = subparser.add_parser("extract-ontology")
    extract_ontology_subparser.add_argument(
        "--sparql-server",
        required=True,
        type=str,
        help='The address of the SPARQL server to test the queries on. In the format "http://127.0.0.1:3030".',
    )
    extract_ontology_subparser.add_argument(
        "--sparql-endpoint",
        required=True,
        type=str,
        help='The address of the SPARQL server to test the queries on. In the format "beastiary".',
    )
    extract_ontology_subparser.add_argument(
        "--output",
        required=True,
        help="Name of the `.rdf` file to be generated",
        type=argparse.FileType("w"),
    )
    extract_ontology_subparser.set_defaults(func=ontology.extract_ontology_to_file)

    return parser


def main() -> int:
    parser = get_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.seed is not None:
        random.seed(args.seed)

    if args.func:
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
