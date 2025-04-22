import collections
import logging

import numpy
import tqdm
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

from .. import caching
from ..structured_logger import get_context

# Originally used BAAI/bge-large-en-v1.5, like the following paper: https://arxiv.org/abs/2410.06062
# Moved to BAAI/bge-small-en-v1.5 for faster tests
model_name = "BAAI/bge-small-en-v1.5"

RankedTerm = collections.namedtuple('RankedTerm', ('text', 'original_index', 'distance'))

CACHE_DIR = 'src/text_embeddings/' + model_name
MAX_TERMS_ON_CUTOFF = 10

MODEL = None

def _init_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(
            model_name,
        )


def load_text_embeddings(texts: list[str]):
    _init_model()
    text_embeddings = []
    texts_to_embed = []
    for text in tqdm.tqdm(texts, desc='Loading cached embeds'):
        if not caching.in_cache(CACHE_DIR, text):
            texts_to_embed.append(text)
            text_embeddings.append(None)
        else:
            text_embeddings.append(numpy.array(caching.get_from_cache(CACHE_DIR, text)))

    if len(texts_to_embed) > 0:
        new_text_embeddings = MODEL.encode(
            texts_to_embed,
            # normalize_embeddings=True,
            show_progress_bar=True,
        )
    else:
        new_text_embeddings = None

    new_values_idx = 0
    for idx in range(len(texts)):
        if text_embeddings[idx] is None:
            text_embeddings[idx] = new_text_embeddings[new_values_idx]
            caching.put_in_cache(CACHE_DIR, texts[idx], new_text_embeddings[new_values_idx].tolist())

            new_values_idx += 1

    return text_embeddings

def rank_by_similarity(reference: str, texts: list[str], embeddings=None) -> list[RankedTerm]:
    """
    Sort a list of strings by their embedding distance to a `reference` one.ReferenceError

    Return a sorted list of tuples (string, original_index, cosine distance).
    """
    print("\r Ranking...", end='\r', flush=True)
    _init_model()


    get_context().log_operation(
        level='DEBUG',
        message='Ranking by similarity {} terms'.format(len(texts)),
        operation='rank_by_similarity',
        data={
            'reference': reference,
            # 'texts': texts,
        }
    )

    ref_embed = MODEL.encode(
        [reference],
        # normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    if embeddings is None:
        text_embeddings = load_text_embeddings(texts)
    else:
        assert len(embeddings) == len(texts)
        text_embeddings = embeddings

    items = []
    for idx, text in enumerate(texts):
        embedding = text_embeddings[idx]
        items.append(RankedTerm(
            text=text,
            original_index=idx,
            distance=distance.cosine(embedding, ref_embed),
        ))

    return sorted(
        items,
        key=lambda x: x.distance,
    )


def cutoff_on_max_difference(terms: list[RankedTerm]) -> list[RankedTerm]:
    if len(terms) in (0, 1):
        return terms

    distance_differences_proportional_max = None
    distance_differences_proportional_idx = None

    for idx in range(len(terms) - 1):
        distance_difference = terms[idx + 1].distance - terms[idx].distance
        if distance_difference == 0:
            logging.warn('Duplicated terms? {} vs {}'.format(terms[idx], terms[idx + 1]))
        proportion = distance_difference / terms[idx].distance
        
        if (
            (distance_differences_proportional_idx is None)
            or (proportion > distance_differences_proportional_max)
        ):
            distance_differences_proportional_idx = idx
            distance_differences_proportional_max = proportion

    selected_terms = terms[:distance_differences_proportional_idx + 1]

    if len(selected_terms) > MAX_TERMS_ON_CUTOFF:
        logging.warn('Selected {} terms, max cutoff: {}, artificially removing more of them'.format(
            len(selected_terms),
            MAX_TERMS_ON_CUTOFF,
        ))
        selected_terms = selected_terms[:MAX_TERMS_ON_CUTOFF]

    return selected_terms
