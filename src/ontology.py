import string
import sys
import logging
from typing import Any, TypedDict, Union

import SPARQLWrapper
import rdflib


def run_query(args, query):
    sparql = SPARQLWrapper.SPARQLWrapper(
        f"{args.sparql_server.strip('/')}/beastiary/sparql"
    )
    sparql.setReturnFormat(SPARQLWrapper.JSON)
    sparql.setQuery(query)

    ret = sparql.queryAndConvert()

    return [
        binding
        for binding in ret['results']['bindings']
    ]


def get_properties_for_class(args, class_uri):
    props = run_query(args, f'''
SELECT DISTINCT ?prop
WHERE {{
  ?s a <{class_uri}> .
  ?s ?prop []
}}''')
    return [
        prop['prop']['value']
        for prop in props
    ]

def get_superclass_of_class(args, class_uri) -> str:
    superclasses = run_query(args, f'''
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?superclass
WHERE {{
  <{class_uri}> rdfs:subClassOf ?superclass
}}''')
    assert len(superclasses) == 1
    return superclasses[0]['superclass']['value']


RDFProperty = Union[str, list[tuple[str, 'RDFProperty']]]

class PropertyGraphClass(TypedDict):
    properties: list[tuple[str, RDFProperty]]
    obj_properties: list[str]
    subclass_of: str

class PropertyGraph(TypedDict):
    concepts: list[str]
    classes: dict[str, PropertyGraphClass]
    annotation_properties: dict[str, list[tuple[str, str]]]
    datatype_properties: dict[str, list[tuple[str, str]]]
    object_properties: dict[str, list[tuple[str, str]]]


def get_data_in_blank_node(args, item_uri, path):
    print(
        f"\r\x1b[0K Inspecting {item_uri.split('#')[-1]} ({' > '.join([sp.split('#')[-1] for sp in path])})...",
        end='\r',
        flush=True,
        file=sys.stderr
    )
    last = f'<{item_uri}>'
    path_filter = []
    for idx, step in enumerate(path):
        nxt = '?' + string.ascii_lowercase[idx]
        if ':' in step:
            step = f'<{step}>'
        path_filter.append(f'    {last} {step} {nxt}.')
        last = nxt

    comb_path_filter = '\n'.join(path_filter)

    query = f'''SELECT DISTINCT ?subprop ?subobj
WHERE {{
{comb_path_filter}
    {last} ?subprop ?subobj
}}'''

    logging.debug(f'''--------
{query}
------''')

    results = run_query(args, query)
    for prop in results:
        if ':' in prop['subobj']['value']:
            yield (prop['subprop']['value'], prop['subobj']['value'])
        else:
            subresults = list(get_data_in_blank_node(args, item_uri, path + [prop['subprop']['value']]))
            yield (prop['subprop']['value'], subresults)


def get_all_data_in_item(args, item_uri) -> list[tuple[str, RDFProperty]]:
    print(
        f"\r\x1b[0K Inspecting {item_uri.split('#')[-1]}...",
        end='\r',
        flush=True,
        file=sys.stderr
    )
    results = run_query(args, f'''SELECT DISTINCT ?prop ?obj
WHERE {{
  <{item_uri}> ?prop ?obj
}}''')
    for prop in results:
        if ':' in prop['obj']['value']:
            yield (prop['prop']['value'], prop['obj']['value'])
        else:
            subresults = list(get_data_in_blank_node(args, item_uri, [prop['prop']['value']]))
            yield (prop['prop']['value'], subresults)


def get_all_properties_in_graph(args) -> PropertyGraph:
    # List all classes
    classes = run_query(args, '''
SELECT DISTINCT ?class
WHERE {
  ?class a <http://www.w3.org/2002/07/owl#Class>
}
    ''')

    annotation_properties = run_query(args, '''
SELECT DISTINCT ?prop
WHERE {
  ?prop a <http://www.w3.org/2002/07/owl#AnnotationProperty>
}
''')

    object_properties = run_query(args, '''
SELECT DISTINCT ?prop
WHERE {
  ?prop a <http://www.w3.org/2002/07/owl#ObjectProperty>
}
''')

    datatype_properties = run_query(args, '''
SELECT DISTINCT ?prop
WHERE {
  ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty>
}
''')

    graph = {
        'annotation_properties': {prop['prop']['value']: {} for prop in annotation_properties },
        'object_properties': {prop['prop']['value']: {} for prop in object_properties },
        'datatype_properties': {prop['prop']['value']: {} for prop in datatype_properties },
        'classes': {},
    }

    for col in (
        'annotation_properties',
        'object_properties',
        'datatype_properties',
    ):
        for prop_name, prop_data in graph[col].items():
            graph[col][prop_name] = list(get_all_data_in_item(args, prop_name))

    for class_uri in [ _class['class']['value'] for _class in classes ]:
        if ':' not in class_uri:
            # Just a placeholder class, not directly implemented
            continue

        graph['classes'][class_uri] = {
            'obj_properties': get_properties_for_class(args, class_uri),
            'properties': list(get_all_data_in_item(args, class_uri)),
            'subclass_of': get_superclass_of_class(args, class_uri),
        }

    return graph


def property_graph_to_rdf(pg: PropertyGraph):
    g = rdflib.Graph()

    def _fill_properties(item_uri, props):
        for predicate, obj in props:
            g.add((
                item_uri,
                rdflib.URIRef(predicate),
                rdflib.URIRef(obj),
            ))

    for col, rdf_type in (
        ('annotation_properties', rdflib.namespace.OWL.AnnotationProperty),
        ('object_properties', rdflib.namespace.OWL.ObjectProperty),
        ('datatype_properties', rdflib.namespace.OWL.DatatypeProperty),
    ):
        for item_name, item_props in pg[col].items():
            item_uri = rdflib.URIRef(item_name)
            g.add((
                item_uri,
                rdflib.namespace.RDF.type,
                rdflib.URIRef(rdf_type),
            ))
            _fill_properties(item_uri, item_props)

    for nsname, nsuri in (
        ('owl', 'http://www.w3.org/2002/07/owl#'),
        ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
        ('xml', 'http://www.w3.org/XML/1998/namespace#'),
        ('xasd', 'http://www.w3.org/2001/XMLSchema#'),
        ('rdfs', 'http://www.w3.org/2000/01/rdf-schema#'),
        ('custom', 'http://codigoparallevar.com/tfm/ontology#')
    ):
        g.bind(nsname, nsuri)

    for class_name, class_data in pg['classes'].items():
        class_uri = rdflib.URIRef(class_name)
        g.add((
            class_uri,
            rdflib.namespace.RDF.type,
            rdflib.URIRef(rdflib.namespace.OWL.Class),
        ))

        g.add((
            class_uri,
            rdflib.namespace.RDFS.subClassOf,
            rdflib.URIRef(class_data['subclass_of']),
        ))

        for prop in class_data['obj_properties']:
            g.add((
                class_uri,
                rdflib.URIRef('http://codigoparallevar.com/tfm/ontology#objectProperty'),
                rdflib.URIRef(prop),
            ))

        _fill_properties(class_uri, class_data['properties'])

    return g

def extract_ontology_to_file(args):
    pg = get_all_properties_in_graph(args)
    rdf = property_graph_to_rdf(pg)
    args.output.write(rdf.serialize(format='pretty-xml'))