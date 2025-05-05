from typing import TypedDict, Union
import re

import rdflib.plugins.sparql.parser

IRI_BLOCK_RE = re.compile(r'<([^>]+)>')

class ParsedNamespaces(TypedDict):
    alias: str
    iri: str


class IRIItem(TypedDict):
    iri: str


class PrefixedItem(TypedDict):
    prefix: str
    name: str


class Variable(TypedDict):
    name: str


TripleItem = Union[IRIItem, PrefixedItem, Variable]


class Triple(TypedDict):
    subject: TripleItem
    predicate: TripleItem
    object: TripleItem


class ParsedQuery(TypedDict):
    namespaces: list[ParsedNamespaces]
    projection: dict
    where: dict

    variables: list


def parse(query: str):
    parsed = rdflib.plugins.sparql.parser.parseQuery(query)

    namespaces = []
    for block in parsed:
        if isinstance(block, list):
            items = block
        else:
            items = [block]

        for item in items:
            if item.name == "PrefixDecl":
                assert (
                    "prefix" in item and "iri" in item
                ), "Expected @PREFIX with `prefix` and `iri` items, found: {}".format(
                    dict(item)
                )

                namespaces.append(
                    {
                        "alias": item["prefix"],
                        "iri": item["iri"],
                    }
                )

            elif item.name == "SelectQuery":
                raise NotImplementedError(
                    "SELECT query not implemented yet: {}".foramt(item)
                )
            else:
                raise NotImplementedError("Unknown query type: {}".foramt(item))


def get_entities(query: str) -> list[str]:
    return IRI_BLOCK_RE.findall(query)


def format_sparql_query(query: str) -> str:
    """Properly indent a SPARQL query."""
    # TODO: Avoid dependency on external tools, right now
    #  we depend on https://github.com/sparqling/sparql-formatter
    with tempfile.NamedTemporaryFile("wt") as f:
        f.write(query)
        f.flush()
        proc = subprocess.run(
            ["sparql-formatter", f.name],
            stdout=subprocess.PIPE,
        )
        result = proc.stdout.decode()
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError:
            return query
    return result
