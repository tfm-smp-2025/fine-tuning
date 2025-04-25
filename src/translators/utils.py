import collections
import re

from typing import Any

from ..structured_logger import get_context

def deindent_text(text: str) -> str:
    """
    Given a block of text removes the indentation that is common to all non-empty lines.
    """
    line_indentations = [
        len(line) - len(line.strip())
        for line in text.split("\n")
        if len(line.strip()) > 0
    ]
    min_line_indentation = min(line_indentations)
    reindented_lines = [
        line[min(len(line), min_line_indentation) :] for line in text.split("\n")
    ]
    return "\n".join(reindented_lines).strip('\n')


CodeBlock = collections.namedtuple('CodeBlock', ('language', 'content'))
CODE_BLOCK_RE = re.compile(
    r'^\s*```([^\n]*)\s*\n((.|\n)*?)^\s*```\s*$',
    re.DOTALL | re.MULTILINE
)

def extract_code_blocks(text: str) -> list[CodeBlock]:
    result = []
    for block in CODE_BLOCK_RE.findall(text):
        result.append(CodeBlock(language=block[0], content=block[1]))
    get_context().log_operation(
        level='DEBUG',
        message='',
        operation='extract_code_blocks',
        data={
            'input': text,
            'blocks': result,
        }
    )

    return result

def url_to_value(url: str) -> str:
    return url.split('#')[-1].split('/')[-1]

def deduplicate(l: list[str]) -> list[str]:
    return list(set(l))

def deduplicate_on_key(l: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    known_keys = set()
    results = []

    for val in l:
        if val[key] not in known_keys:
            known_keys.add(val[key])
            results.append(val)

    return results

def merge_mapping_dicts(d1, d2):
    """Merges two dictionaries while handling duplicate keys."""
    result = {}
    for k in set(d1.keys()) | set(d2.keys()):
        v1 = d1.get(k)
        v2 = d2.get(k)

        if v1 and not v2:
            result[k] = v1
        elif v2 and not v1:
            result[k] = v2
        else:
            if 'alternatives' in v1:
                v1_alts = v1['alternatives']
            else:
                v1_alts = [v1]

            if 'alternatives' in v2:
                v2_alts = v2['alternatives']
            else:
                v2_alts = [v2]

            alts = []
            known_urls = set()
            for alt in v1_alts + v2_alts:
                if alt['url'] not in known_urls:
                    known_urls.add(alt['url'])
                    alts.append(alt)
            result[k] = { 'alternatives': alts }
    return result