import collections
import re

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
    return "\n".join(reindented_lines)


CodeBlock = collections.namedtuple('CodeBlock', ('language', 'content'))
CODE_BLOCK_RE = re.compile(
    r'^```([^\n]*)\s*\n((.|\n)*?)^```\s*$',
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