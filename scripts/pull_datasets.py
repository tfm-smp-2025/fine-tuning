#!/usr/bin/env python3

import json
import os
import urllib.request
import tempfile
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DATASET_JSON = os.path.join(DATASET_DIR, "datasets.json")


def dataset_name_to_slug(ds_name: str) -> str:
    """Convert a dataset name to a format more adequate for directory name."""
    assert ":" not in ds_name
    return ds_name.lower().replace(" ", "_")


def pull_datasets():
    """Pull all known datasets to the `datasets` directory."""
    with open(DATASET_JSON, "rt") as f:
        dataset_data = json.load(f)

    for dataset_name, dataset_urls in dataset_data.items():
        if dataset_name.startswith("__"):
            # Skip
            continue

        dataset_path = os.path.join(DATASET_DIR, dataset_name_to_slug(dataset_name))

        pull_dataset(dataset_path, dataset_name, dataset_urls)


def pull_dataset(path: str, name: str, urls: dict[str, str]):
    """Download the files contained in a dataset."""
    os.makedirs(path, exist_ok=True)
    if "all" in urls:
        print("{} | Unified dataset...".format(name))
        # All dataset in a single file
        if os.path.exists(
            os.path.join(
                path,
                "all.json",
            )
        ):
            print(f"  ✔ {name} unified dataset present")
        else:
            print(f"  ↓  Downloading {name} dataset")
            if urls["all"].endswith(".json"):
                urllib.request.urlretrieve(urls["all"], os.path.join(path, "all.json"))
            else:
                # Zip file
                zip_file = download_temp_zip_file(urls["all"])
                # Expect a single file inside
                inner_files = zip_file.namelist()

                if "train_subpath" in urls and "test_subpath" in urls:
                    for key in ("train", "test"):
                        with zip_file.open(urls[key + "_subpath"], "r") as finner:
                            with open(os.path.join(path, key + ".json"), "wt") as fouter:
                                fouter.write(finner.read().decode())
                else:
                    assert (
                        len(inner_files) == 1
                    ), "Expected a file inside the zip as no test_subpath and train_subpath were found. Keys: {}; files: {}".format(urls.keys(), inner_files)
                    with zip_file.open(inner_files[0], "r") as finner:
                        with open(os.path.join(path, "all.json"), "wt") as fouter:
                            fouter.write(finner.read().decode())

            print(f"  ✔  {name} dataset ready")

    else:
        # Dataset pre-splitted in train and test
        assert (
            "train" in urls and "test" in urls
        ), "Expected 'train' and 'test' in URLs, found: {}".format(urls.keys())

        print("{} | Split dataset...".format(name))
        for key in "train", "test":
            assert urls[key].endswith(".json")
            if os.path.exists(os.path.join(path, key + ".json")):
                print(f"  ✔  {key.title()} file present")
            else:
                print(f"  ↓  Downloading {key} file")
                urllib.request.urlretrieve(urls[key], os.path.join(path, key + ".json"))
                print(f"  ✔  {key.title()} file ready")
    print()


def download_temp_zip_file(url: str) -> zipfile.ZipFile:
    """Download a Zip file to a temporary location and open it."""
    with tempfile.NamedTemporaryFile(suffix=".zip") as f:
        urllib.request.urlretrieve(
            url,
            f.name,
        )
        return zipfile.ZipFile(f.name)


if __name__ == "__main__":
    pull_datasets()
