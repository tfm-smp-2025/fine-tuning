import json
import os
import random
from typing import Optional

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')

def dataset_name_to_slug(ds_name: str) -> str:
    """Convert a dataset name to a format more adequate for directory name."""
    # TODO: De-duplicate with the `pull_datasets` script
    assert ":" not in ds_name
    return ds_name.lower().replace(" ", "_")


class UnifiedDatasetLoader:
    def __init__(self, name: str, train_split: float=0.8, rand_seed: Optional[int]=None):
        self.name = name
        self.train_split = train_split
        self.rand_seed = rand_seed

        root_dir = os.path.join(DATASETS_DIR, dataset_name_to_slug(name))
        assert os.path.isdir(root_dir), "Expected {} to be a directory".format(root_dir)
        self.file_path = os.path.join(root_dir, 'all.json')
        assert os.path.isfile(self.file_path), "Expected {} to be a file".format(self.file_path)
        self.data = None

    def __repr__(self):
        return f"Dataset '{self.name}' ({'not loaded' if self.data is None else str(len(self.data)) + ' entries'})"

    def __enter__(self):
        self.__load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__unload()

    def __load(self):
        with open(self.file_path) as f:
            self.data = json.load(f)['questions']

    def __unload(self):
        self.data = None

    def get_test_data(self):
        if self.data is None:
            self.__load()

        random.seed(self.rand_seed)
        for it in self.data:
            if random.random() > self.train_split:
                yield it

    def get_train_data(self):
        if self.data is None:
            self.__load()

        random.seed(self.rand_seed)
        for it in self.data:
            if random.random() <= self.train_split:
                yield it


def load_dataset(name):
    # @TODO@ Handle non-unified datasets
    return UnifiedDatasetLoader(name)