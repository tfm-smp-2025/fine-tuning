import argparse
import logging
import random
import sys
from . import tester

def get_argparser():
    parser = argparse.ArgumentParser("SPARQL LLM fine-tuner")
    parser.add_argument('--seed', type=int, help='Seed used to for random choices')
    subparser = parser.add_subparsers(help='subcommand help', required=True)

    test_subparser = subparser.add_parser('test')
    set_test = test_subparser.add_mutually_exclusive_group(required=True)
    set_test.add_argument(
        "--sample",
        type=int,
        help='Use only a sample of (n) queries to run the test'
    )
    set_test.add_argument(
        "--full",
        action='store_true',
        help='Run the test with the full query set'
    )
    test_subparser.set_defaults(func=tester.run_test)
    return parser

def main() -> int:
    parser = get_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.seed is not None:
        random.seed(args.seed)

    if args.func:
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())