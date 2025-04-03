import json
import logging
import subprocess
import tempfile
import tqdm

from .datasets import dataset_loader
from .translators.utils import deindent_text

def format_sparql_query(query: str) -> str:
    """Properly indent a SPARQL query."""
    # TODO: Avoid dependency on external tools, right now
    #  we depend on https://github.com/sparqling/sparql-formatter
    with tempfile.NamedTemporaryFile('wt') as f:
        f.write(query)
        f.flush()
        proc = subprocess.run(
            ["sparql-formatter", f.name],
            stdout=subprocess.PIPE,
        )
        result = proc.stdout.decode()
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError:
            return query
    return result

def generate(args):
    """Generate fine-tuning data."""
    for dataset_name in args.datasets:
        dataset = dataset_loader.load_dataset(dataset_name, rand_seed=args.seed)
        data = {}
        for ds_name, ds_data in zip(('train', 'test'), dataset.get_split_dataset()):
            sub_dataset = []
            for item in tqdm.tqdm(ds_data, desc=f'{dataset_name} for {ds_name}'):
                for variant in item['question']:
                    if variant['language'] != 'en':
                        logging.debug('SKIPPING question in non-english: {}'.format(variant['language']))
                        continue

                    row = {
                        "user": deindent_text(f"""
                        Generate the SPARQL query for this natural language query:

                        --- Natural language query
                        {variant['string']}
                        --- End of natural language query
                        """).strip(),
                        # @TODO@: Properly format SPARQL query
                        "assistant": '```sparql\n' + format_sparql_query(item['query']['sparql']) + '\n```\n'
                    }

                    if args.split_test:
                        out = args.output
                        if ds_name == 'test':
                            out = args.split_test
                        out.write(json.dumps(row) + '\n')
                    else:
                        sub_dataset.append(row)
            data[ds_name] = sub_dataset
        if not args.split_test:
            args.output.write(json.dumps(data, indent=4))