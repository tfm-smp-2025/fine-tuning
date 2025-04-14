import collections
import logging

from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

# Using BAAI/bge-large-en-v1.5, like the following paper: https://arxiv.org/abs/2410.06062
model_name = "BAAI/bge-large-en-v1.5"

RankedTerm = collections.namedtuple('RankedTerm', ('text', 'original_index', 'distance'))

def rank_by_similarity(reference: str, texts: list[str]) -> list[RankedTerm]:
    """
    Sort a list of strings by their embedding distance to a `reference` one.ReferenceError

    Return a sorted list of tuples (string, original_index, cosine distance).
    """
    model = SentenceTransformer(
        model_name,
    )

    ref_embed = model.encode(
        [reference],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    text_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

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

    return terms[:distance_differences_proportional_idx + 1]
            