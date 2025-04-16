from .prompt_based import translators as prompt_translators
from .prompt_based_with_search import translators as prompt_with_search_translators

all_translators = prompt_translators + prompt_with_search_translators
