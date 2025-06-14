import spacy
import logging
from ..structured_logger import get_context

NLP_MODEL = None


def load_nlp_model():
    global NLP_MODEL
    NLP_MODEL = spacy.load("en_core_web_md")

def get_entities(text: str) -> list[str]:
    load_nlp_model()

    analysis = list(NLP_MODEL(text))
    result = []
    last_is_noun = False
    for tok in analysis:
        if tok.tag_.startswith('V'):
            last_is_noun = False
            if not NLP_MODEL.vocab[tok.text].is_stop:
                result.append(tok.text)
        elif tok.tag_.startswith('N'):
            if last_is_noun:
                result[-1] = result[-1] + ' ' + tok.text
            else:
                result.append(tok.text)
            last_is_noun = True
        else:
            last_is_noun = False
    return result

def is_singular(text: str) -> bool:
    load_nlp_model()

    analysis = list(NLP_MODEL(text))
    if len(analysis) != 1:
        logging.warning(
            'Expected 1 item from phrase: "{}", found:\n    {}'.format(
                text,
                "\n    ".join(
                    [
                        "; ".join([tok.text, tok.tag_, spacy.explain(tok.tag_)])
                        for tok in analysis
                    ]
                ),
            )
        )

    any_plural = False
    for element in analysis:
        if not element.tag_.startswith("NN"):
            logging.warning(
                'Expected "{}" to be recognized as NN* (noun). Recognized as: {} ({}); from text: {}'.format(
                    element.text,
                    element.tag_,
                    spacy.explain(element.tag_),
                    element,
                )
            )
        else:
            # See relevant POS tags: https://github.com/explosion/spaCy/blob/98a19df91a9f28a5cc208aacd2e56b2a9376bf86/spacy/glossary.py#L74-L77
            if element.tag_ in ("NNPS", "NNS"):
                any_plural = True

    return not any_plural
