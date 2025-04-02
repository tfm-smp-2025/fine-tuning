#!/usr/bin/env python3

import shutil
import json
import os
import urllib.request
import tempfile
import tqdm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
KNOWLEDGE_BASE = os.path.join(DATASET_DIR, "knowledge_bases.json")

def kb_name_to_slug(ds_name: str) -> str:
    """Convert a dataset name to a format more adequate for directory name."""
    assert ":" not in ds_name
    return ds_name.lower().replace(" ", "_").replace("-", "_")


def retrieve_via_temp(url, final_path):
    with tempfile.NamedTemporaryFile() as f:
        urllib.request.urlretrieve(url, f.name)
        shutil.move(f.name, final_path)

        # Create the file so it can be deleted by NamedTemporaryFile.
        #   We could also set it's delete=False, but this way we probably
        #   lean more towards collecting it on case of error (didn't bother
        #   to check that).
        open(f.name, 'w')


def pull_knowledge_bases():
    """Pull all known knowledge bases to the `datasets` directory."""
    with open(KNOWLEDGE_BASE, "rt") as f:
        kb_data = json.load(f)

    for kb_name, kb_metadata in kb_data.items():
        if kb_name.startswith("__"):
            # Skip
            continue

        kb_path = os.path.join(DATASET_DIR, kb_name_to_slug(kb_name))

        pull_knolwedge_base(kb_path, kb_name, kb_metadata, os.path.dirname(kb_path))


def get_save_path(link: dict[str, str], basedir) -> str:
    by_url = link['url'].split('/')[2:]
    return os.path.join(basedir, 'by_url', *by_url)


def pull_knolwedge_base(path: str, name: str, metadata: dict[str, str], basedir: str):
    """Download the files contained in a dataset."""
    os.makedirs(path, exist_ok=True)

    links_path = os.path.join(basedir, metadata['links'])
    with open(links_path) as f:
        links_data = json.load(f)

    print("{}: {} links".format(name, len(links_data)))

    for link in tqdm.tqdm(links_data):
        save_path = get_save_path(link, basedir)
        assert not os.path.isdir(save_path)

        if os.path.exists(save_path):
            print(f"  ✔ Already present: '{link['dataset_name']}' in '{link['lang']}'")
            continue

        print(f"  ↓ Downloading: '{link['dataset_name']}' in '{link['lang']}' to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        retrieve_via_temp(link["url"], save_path)
        print(f"  ✔ Ready: '{link['dataset_name']}' in '{link['lang']}'")

    print()

if __name__ == "__main__":
    pull_knowledge_bases()
