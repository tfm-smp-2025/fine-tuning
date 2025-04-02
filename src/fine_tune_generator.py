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
        proc.check_returncode()
    return result

def generate(args):
    """Generate fine-tuning data."""
    for dataset_name in args.datasets:
        with dataset_loader.load_dataset(dataset_name) as ds:
            for item in tqdm.tqdm(ds.get_train_data()):
                for variant in item['question']:
                    if variant['language'] != 'en':
                        logging.debug('SKIPPING question in non-english: {}'.format(variant['language']))
                        continue

                    args.output.write(json.dumps({
                        "user": deindent_text(f"""
                        Generate the SPARQL query for this natural language query:

                        --- Natural language query
                        {variant['string']}
                        --- End of natural language query
                        """).strip(),
                        # @TODO@: Properly format SPARQL query
                        "assistant": '```sparql\n' + format_sparql_query(item['query']['sparql']) + '\n```\n'
                    }))