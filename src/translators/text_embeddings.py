import collections
import logging
import os
import re
from typing import TypedDict

import numpy
import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5  # Generate a deterministic ID

from .. import caching
from ..structured_logger import get_context

# Originally used BAAI/bge-large-en-v1.5, like the following paper: https://arxiv.org/abs/2410.06062
# Moved to BAAI/bge-small-en-v1.5 for faster tests
MODEL_NAME = "BAAI/bge-small-en-v1.5"

RankedTerm = collections.namedtuple('RankedTerm', ('clean', 'raw', 'distance'))

IMPORT_BATCH_SIZE = 1024
IMPORT_CONCURRENT_REQUESTS = 4

MAX_TERMS_IN_CUTOFF = 10
MAX_ERRORS_IN_IMPORT_BY_BATCH = 10

WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY', 'adminkey')
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
WEAVIATE_HTTP_PORT = int(os.getenv('WEAVIATE_HTTP_PORT', 8080))
WEAVIATE_GRPC_PORT = int(os.getenv('WEAVIATE_GRPC_PORT', 50051))

class IndexedEntries(TypedDict):
    raw: str
    clean: str

class Connection:
    def __init__(self):
        pass

    def _connect(self):
        self.connection = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )

    def _disconnect(self):
        self.connection.close()

    def __enter__(self) -> weaviate.client.WeaviateClient:
        self._connect()
        return self.connection

    def __exit__(self, exc, exc_type, tb):
        self._disconnect()


def _serialize_name(name: str):
    """Convert name to one acceptable for Weaviate."""
    # assert len(name) < 64, \
    #     'Max size for a Weaviate name is 64'
    name = name[-63:]
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def exists_collection(collection_group: str, collection_name: str) -> bool:
    with Connection() as client:
        if not client.collections.exists(_serialize_name(collection_group)):
            return False

        return client.collections.get(
            _serialize_name(collection_group)
        ).tenants.exists(
            _serialize_name(collection_name)
        )


def _create_weaviate_collection(client: weaviate.client.WeaviateClient, name: str):
    client.collections.create(
        _serialize_name(name),
        vectorizer_config=[
            Configure.NamedVectors.text2vec_transformers(
                name="clean_vector",
                source_properties=["clean"],
            ),
            # This makes the indexing of owl#Things take too long
            #  and doesn't add anything of value
            # Configure.NamedVectors.text2vec_transformers(
            #     name="raw_vector",
            #     source_properties=["raw"],
            # ),
        ],
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=False,
        ),
        properties=[
            Property(name="clean", data_type=DataType.TEXT),
            Property(name="raw", data_type=DataType.TEXT),
        ]
    )


def load_collection(collection_group: str, collection_name: str, texts: IndexedEntries):
    with Connection() as client:
        if not client.collections.exists(_serialize_name(collection_group)):
            _create_weaviate_collection(client, _serialize_name(collection_group))

        get_context().log_operation(
            level='INFO',
            message='Loading collection {} with {} elements'.format((collection_group, collection_name), len(texts)),
            operation='load_collection',
            data={
                'collection_group': collection_group,
                'collection_name': collection_name,
                'len_texts': len(texts),
            }
        )

        collection = client.collections.get(_serialize_name(collection_group))

        # Create tenant
        if not collection.tenants.exists(_serialize_name(collection_name)):
            collection.tenants.create(_serialize_name(collection_name))

        collection = collection.with_tenant(_serialize_name(collection_name))

        for _ in collection.iterator():
            # Some element found
            raise Exception(
                'The collection ({}) has already been loaded. Maybe there was a problem building it?'.format(
                    (collection_group, _serialize_name(collection_name)),
            ))

        # Load data
        with collection.batch.fixed_size(batch_size=IMPORT_BATCH_SIZE, concurrent_requests=IMPORT_CONCURRENT_REQUESTS) as batch:
            for data_row in tqdm.tqdm(texts):
                obj_uuid = generate_uuid5(data_row)
                batch.add_object(
                    properties=data_row,
                    uuid=obj_uuid,
                )
                if batch.number_errors > MAX_ERRORS_IN_IMPORT_BY_BATCH:
                    logging.error("Batch import stopped due to excessive errors.")
                    break

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            error_msg = f"Number of failed imports: {len(failed_objects)}"
            error_msg += f"\nFirst failed object: {failed_objects[0]}"
            raise Exception(error_msg)
        else:
            print("No failed objects")


def find_close_in_collection(collection_group: str, collection_name: str, reference: str) -> list[RankedTerm]:
    """
    Sort a list of strings by their embedding distance to a `reference` one.ReferenceError

    Return a sorted list of tuples (string, original_index, cosine distance).
    """
    get_context().log_operation(
        level='DEBUG',
        message='Ranking by similarity to {}'.format(len(reference)),
        operation='rank_by_similarity',
        data={
            'reference': reference,
        }
    )
    with Connection() as client:
        collection = client.collections.get(
            _serialize_name(collection_group)
        ).with_tenant(
            _serialize_name(collection_name)
        )

        response = collection.query.near_text(
            query=reference,
            limit=MAX_TERMS_IN_CUTOFF,
            target_vector="clean_vector",
            return_metadata=MetadataQuery(distance=True)
        )

        return [
            RankedTerm(
                raw=o.properties['raw'],
                clean=o.properties['clean'],
                distance=o.metadata.distance,
            )
            for o in response.objects
        ]
