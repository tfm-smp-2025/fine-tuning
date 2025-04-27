from . import caching

import os
import numpy

CACHE_DIR = "src/class_embeddings_caching"


def get_cache_path(cache_group, cache_key):
    return caching.get_cache_path(
        CACHE_DIR, os.path.join(cache_group, cache_key)
    ).replace(".json", ".npy")


def in_cache(cache_group, cache_key):
    return os.path.exists(get_cache_path(cache_group, cache_key))


def get_from_cache(cache_group, cache_key):
    return numpy.load(get_cache_path(cache_group, cache_key))


def put_in_cache(cache_group, cache_key, value):
    cache_path = get_cache_path(cache_group, cache_key)

    os.makedirs(
        os.path.dirname(cache_path),
        exist_ok=True,
    )
    return numpy.save(
        cache_path,
        numpy.array(value),
    )
