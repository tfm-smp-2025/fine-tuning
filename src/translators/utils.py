import collections
import re

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
    r'^```([^\n]*)\n((.|\n)*?)^```$',
    re.DOTALL | re.MULTILINE
)

def extract_code_blocks(text: str) -> list[CodeBlock]:
    print(f"TXT: {text}")
    result = []
    for block in CODE_BLOCK_RE.findall(text):
        print(f"BLOCK: {block}")
        result.append(CodeBlock(language=block[0], content=block[1]))

    return result