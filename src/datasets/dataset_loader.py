import collections
import json
import logging
import os
import random
from typing import Any, Optional

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')

QA = collections.namedtuple('QA', ('question', 'answer', 'lang'))

def dataset_name_to_slug(ds_name: str) -> str:
    """Convert a dataset name to a format more adequate for directory name."""
    # TODO: De-duplicate with the `pull_datasets` script
    assert ":" not in ds_name
    return ds_name.lower().replace(" ", "_")


def to_qa_list(dataset: list | dict) -> list[QA]:
    if isinstance(dataset, dict):
        if 'questions' in dataset:
            dataset = dataset['questions']
        elif 'Questions' in dataset:
            dataset = dataset['Questions']
        else:
            raise NotImplemented('Unknown dict dataset. Keys: {}'.format(dataset.keys()))

    assert isinstance(dataset, list), "Expected a list of data, found: {}".format(type(dataset))

    for row in dataset:
        assert isinstance(row, dict), "Expected dict item, found: {}".format(row)
        if 'corrected_question' in row and 'sparql_query' in row:
            yield QA(
                question=row['corrected_question'],
                answer=row['sparql_query'],
                lang=None,
            )
        elif 'paraphrased_question' in row and 'sparql_dbpedia18' in row:
            yield QA(
                question=row['paraphrased_question'],
                answer=row['sparql_dbpedia18'],
                lang=None,
            )
        elif 'RawQuestion' in row and 'Parses' in row:
            for parse in row['Parses']:
                yield QA(
                    question=row['RawQuestion'],
                    answer=parse['Sparql'],
                    lang=None,
                )
        elif 'question' in row and 'query' in row:
            if isinstance(row['question'], list) and isinstance(row['query'], dict):
                for alt in row['question']:
                    if 'string' not in alt:
                        logging.warn('Skipping empty question: {}'.format(alt))
                        continue

                    assert 'string' in alt and 'language' in alt, \
                        "Expected 'string' and 'language' in alt, found: {}".format(alt)
                    yield QA(
                        question=alt['string'],
                        answer=row['query']['sparql'],
                        lang=alt['language'],
                    )
            elif  isinstance(row['question'], str) and isinstance(row['query'], str):
                yield QA(
                    question=row['question'],
                    answer=row['query'],
                    lang=None,
                )
            else:
                raise NotImplemented('Unknown item type: {}'.format(row))
        else:
            raise NotImplemented('Unknown item type: {}'.format(row))

class DatasetLoader:
    def get_split_dataset(self) -> tuple[list[QA], list[QA]]:
        raise NotImplemented('This is an abstract class. You should use UnifiedDatasetLoader or SplitDatasetLoader.')

    def get_test_data(self) -> list[QA]:
        return self.get_split_dataset()[1]



class UnifiedDatasetLoader(DatasetLoader):
    def __init__(self, name: str, train_split: float=0.8, rand_seed: Optional[int]=None):
        self.name = name
        self.train_split = train_split
        self.rand_seed = rand_seed or random.random()

        root_dir = os.path.join(DATASETS_DIR, dataset_name_to_slug(name))
        assert os.path.isdir(root_dir), "Expected {} to be a directory".format(root_dir)
        self.file_path = os.path.join(root_dir, 'all.json')
        assert os.path.isfile(self.file_path), "Expected {} to be a file".format(self.file_path)
        self.data = None

    def __repr__(self):
        return f"Dataset '{self.name}' ({'not loaded' if self.data is None else str(len(self.data)) + ' entries'})"

    def __load(self):
        with open(self.file_path) as f:
            return json.load(f)['questions']

    def get_split_dataset(self) -> tuple[list[QA], list[QA]]:
        data = list(to_qa_list(self.__load()))

        random.seed(self.rand_seed)
        random.shuffle(data)

        cutoff = int(len(data) * self.train_split)
        return (data[:cutoff], data[cutoff:])


class SplitDatasetLoader(DatasetLoader):
    def __init__(self, name: str):
        self.name = name
        root_dir = os.path.join(DATASETS_DIR, dataset_name_to_slug(name))
        assert os.path.isdir(root_dir), "Expected {} to be a directory".format(root_dir)
        self.test_path = os.path.join(root_dir, 'test.json')
        self.train_path = os.path.join(root_dir, 'train.json')

    def get_split_dataset(self) -> tuple[list[QA], list[QA]]:
        with open(self.train_path) as f:
            train_data = json.load(f)
            train_data = list(to_qa_list(train_data))

        with open(self.test_path) as f:
            test_data = json.load(f)
            test_data = list(to_qa_list(test_data))

        return train_data, test_data


def load_dataset(name, rand_seed: Optional[int]=None) -> DatasetLoader:
    root_dir = os.path.join(DATASETS_DIR, dataset_name_to_slug(name))

    if os.path.exists(os.path.join(root_dir, 'test.json')):
        return SplitDatasetLoader(name)

    return UnifiedDatasetLoader(name, rand_seed=rand_seed)
