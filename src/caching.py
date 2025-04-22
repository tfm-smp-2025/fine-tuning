import os
import json
import hashlib
import base64

CACHE_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'cache'
)

def _args_to_key(args: str) -> str:
    return base64.b64encode(hashlib.sha1(args).digest())

def get_naive_cache_path(cache_dir, cache_key):
    """Get a cache path just based on dir and key."""
    return os.path.join(CACHE_ROOT_DIR, cache_dir, cache_key + '.json')

def get_lighter_cache_path(cache_dir, cache_key):
    """Generates a cache path that distributes the cache-keys in two additional hash-based levels."""
    distributed_key = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(CACHE_ROOT_DIR, cache_dir, distributed_key[:2], distributed_key[2:4], cache_key + '.json')

def get_cache_path(cache_dir, cache_key):
    light_path = get_lighter_cache_path(cache_dir, cache_key)
    if os.path.exists(light_path):
        return light_path

    naive_path = get_naive_cache_path(cache_dir, cache_key)
    if os.path.exists(naive_path):  # If lighter one is not found but naive is, use the naive one
        return naive_path

    # If none is found, prefer the lighter one
    return light_path

def in_cache(cache_dir, cache_key):
    return os.path.exists(get_cache_path(cache_dir, cache_key))

def get_from_cache(cache_dir, cache_key):
    with open(get_cache_path(cache_dir, cache_key), 'rt') as f:
        return json.load(f)

def put_in_cache(cache_dir, cache_key, data):
    cache_path = get_cache_path(cache_dir, cache_key)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wt') as f:
        json.dump(data, f)


def function_cache(cache_dir):
    """Cache the result of a function by a `cache_dir` and it's arguments."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            params = {"args": args, "kwargs": kwargs}
            cache_key = _args_to_key(params)

            if in_cache(cache_dir=cache_dir, cache_key=cache_key):
                return get_from_cache(cache_dir, cache_key)['result']

            result = function(*args, **kwargs)

            put_in_cache(cache_dir, cache_key, {'parameters': params, 'result': result})

            return result
        return wrapper
    return decorator
