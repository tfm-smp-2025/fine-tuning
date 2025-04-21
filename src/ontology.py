import datetime
import time
import string
import sys
import re
import logging
import traceback
from typing import Any, TypedDict, Union

import SPARQLWrapper
import rdflib

from . import caching
from .structured_logger import get_context

RDFProperty = Union[str, list[tuple[str, 'RDFProperty']]]
CACHE_DIR = 'src/ontology'
WAIT_BETWEEN_REMOTE_RETRIES = 3

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


class Ontology:
    def __init__(self, sparql_server, sparql_endpoint):
        self.sparql_server = sparql_server
        self.sparql_endpoint = sparql_endpoint
        self.class_list_cache = None

    def run_query(self, query, quiet=False):
        sparql = SPARQLWrapper.SPARQLWrapper(
            f"{self.sparql_server.strip('/')}/{self.sparql_endpoint}/sparql"
        )

        print("[Querying] {}".format(re.sub(r'\s+', ' ', query)[:40]), end='', flush=True)

        sparql.setReturnFormat(SPARQLWrapper.JSON)
        sparql.setQuery(query)

        try:
            ret = sparql.queryAndConvert()
        except Exception:
            print("\r\x1b[0K", end='')
            get_context().log_operation(
                level='ERROR',
                message='Error SPARQL query: {}'.format(query),
                operation='run_sparql',
                data={
                    'query': query,
                },
                exception = traceback.format_exc(),
            )
            raise

        if not quiet:
            print("\r\x1b[0K", end='')
            get_context().log_operation(
                level='DEBUG',
                message='Running SPARQL query: {}'.format(query),
                operation='run_sparql',
                data={
                    'query': query,
                    'result': ret,
                }
            )

        if 'boolean' in ret:
            return ret['boolean']

        return [
            binding
            for binding in ret['results']['bindings']
        ]

    def iterative_listing(
        self,
        query,
        limit=1000*100,
        offset=0,
        max_retries=5,
        distinct_key=None,
        save_partial=None,
    ):
        known_results = set()
        total_count = 0
        all_results = []
        get_context().log_operation(
            level='DEBUG',
            message='Running long SPARQL query: {}'.format(query),
            operation='run_sparql_iterative_in',
            data={
                'query': query,
            }
        )

        all_times = []
        while True:
            if len(all_times) == 0:
                time_info = '-- no time info --'
            else:
                time_until_now = sum([
                    td.total_seconds() for td in all_times
                ])
                mean_time = datetime.timedelta(seconds=time_until_now / len(all_times))
                by_element = datetime.timedelta(seconds=time_until_now / total_count)
                by_thousand = datetime.timedelta(seconds=(time_until_now * 1000) / total_count)
                by_million = datetime.timedelta(seconds=(time_until_now * 1000 * 1000) / total_count)
                time_info = f'last time: {all_times[-1]}; mean: {mean_time}; {by_element}/item; {by_thousand}/k; {by_million}/M; {len(known_results)} uniques'
            print("\r\x1b[0K {} -> {} ({})".format(offset, offset + limit, time_info), end='\r', flush=True)

            t0 = datetime.datetime.now()

            for retry in range(max_retries):
                try:
                    results = self.run_query(query.format(
                        limit=limit,
                        offset=offset,
                    ), quiet=True)
                    break
                except Exception:
                    msg = 'Error on query {}/{}'.format(retry + 1, max_retries)
                    if retry == max_retries - 1:
                        logging.error(msg)
                        if save_partial is not None:
                            save_partial(all_results)
                        raise
                    else:
                        logging.warn(msg)
                    time.sleep(WAIT_BETWEEN_REMOTE_RETRIES)

            if len(results) == 0:
                break

            if distinct_key is None:
                all_results.extend(results)
            else:
                for r in results:
                    val = r[distinct_key]['value']
                    if val not in known_results:
                        all_results.append(val)
                        known_results.add(val)

            offset += len(results)
            total_count += len(results)
            all_times.append(datetime.datetime.now() - t0)

        get_context().log_operation(
            level='DEBUG',
            message='Completed iterative SPARQL query with {} results: {}'.format(len(all_results), query),
            operation='run_sparql_iterative',
            data={
                'query': query,
                'count': len(all_results),
            }
        )

        return all_results
        

    def get_properties_for_class(self, class_uri):
        props = self.run_query(f'''
    SELECT DISTINCT ?prop
    WHERE {{
    ?s a <{class_uri}> .
    ?s ?prop []
    }}''')
        return [
            prop['prop']['value']
            for prop in props
        ]

    def get_superclass_of_class(self, class_uri) -> str:
        superclasses = self.run_query(f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?superclass
    WHERE {{
    <{class_uri}> rdfs:subClassOf ?superclass
    }}''')
        assert len(superclasses) == 1
        return superclasses[0]['superclass']['value']

    def get_data_in_blank_node(self, item_uri, path):
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

        results = self.run_query(query)
        for prop in results:
            if ':' in prop['subobj']['value']:
                yield (prop['subprop']['value'], prop['subobj']['value'])
            else:
                subresults = list(self.get_data_in_blank_node(item_uri, path + [prop['subprop']['value']]))
                yield (prop['subprop']['value'], subresults)

    def get_all_data_in_item(self, item_uri) -> list[tuple[str, RDFProperty]]:
        print(
            f"\r\x1b[0K Inspecting {item_uri.split('#')[-1]}...",
            end='\r',
            flush=True,
            file=sys.stderr
        )
        results = self.run_query(f'''SELECT DISTINCT ?prop ?obj
    WHERE {{
    <{item_uri}> ?prop ?obj
    }}''')
        for prop in results:
            if ':' in prop['obj']['value']:
                yield (prop['prop']['value'], prop['obj']['value'])
            else:
                subresults = list(self.get_data_in_blank_node(item_uri, [prop['prop']['value']]))
                yield (prop['prop']['value'], subresults)

    def get_classes_in_kg(self):
        if self.class_list_cache is None:
            res = self.run_query('''
        SELECT DISTINCT ?class
        WHERE {
        ?class a <http://www.w3.org/2002/07/owl#Class>
        }
            ''')
            self.class_list_cache = [
                _class['class']['value']
                for _class in res
                if ':' in _class['class']['value']
            ]
        return list(self.class_list_cache)

    def get_types_in_kg(self):
        cache_key = self.sparql_endpoint + '-types'
        if caching.in_cache(CACHE_DIR, cache_key):
            return caching.get_from_cache(CACHE_DIR, cache_key)

        result = self.run_query('''
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?type
        WHERE {
            [] rdf:type ?type
            FILTER (!strstarts(str(?type), "http://dbpedia.org/class/yago/Wikicat")) # Skip internal DBPedia listings
        }
        ''')
        caching.put_in_cache(CACHE_DIR, cache_key, result)
        return result

    def find_instances_of(self, _class):
        res = self.run_query(f'''
    SELECT DISTINCT ?value
    WHERE {{
        ?value a <{_class}>
    }}
        ''')
        return [
            value['value']['value']
            for value in res
            if ':' in value['value']['value']
        ]

    def find_relations_between_class_objects(self, from_class, to_class) -> list[str]:
        res = self.run_query(f'''
    SELECT DISTINCT ?pred
    WHERE {{
        ?o1 rdf:type <{from_class}> .
        ?o2 rdf:type <{to_class}> .
        ?o1 ?pred ?o2 .
    }}
        ''')
        return [
            pred['pred']['value']
            for pred in res
        ]


    def get_all_properties_in_graph(self) -> PropertyGraph:
        # List all classes
        if caching.in_cache(CACHE_DIR, self.sparql_endpoint):
            return caching.get_from_cache(CACHE_DIR, self.sparql_endpoint)

        print("\r\x1b[0K Reading classes...", end='\r', flush=True)
        classes = self.get_classes_in_kg()

        print("\r\x1b[0K Reading types...", end='\r', flush=True)
        types = self.get_types_in_kg()

        print("\r\x1b[0K Reading annotation properties...", end='\r', flush=True)
        annotation_properties = self.run_query('''
    SELECT DISTINCT ?prop
    WHERE {
    ?prop a <http://www.w3.org/2002/07/owl#AnnotationProperty>
    }
    ''')

        print("\r\x1b[0K Reading object properties...", end='\r', flush=True)
        object_properties = self.run_query('''
    SELECT DISTINCT ?prop
    WHERE {
    ?prop a <http://www.w3.org/2002/07/owl#ObjectProperty>
    }
    ''')

        print("\r\x1b[0K Reading datatype properties...", end='\r', flush=True)
        datatype_properties = self.run_query('''
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
            'types': { t['type']['value']: {} for t in types }
        }

        for col in (
            'annotation_properties',
            'object_properties',
            'datatype_properties',
            'types'
        ):
            for prop_name, prop_data in graph[col].items():
                graph[col][prop_name] = list(self.get_all_data_in_item(prop_name))

        print("\r\x1b[0K Introspecting classes...", end='\r', flush=True)
        for class_uri in classes:
            if ':' not in class_uri:
                # Just a placeholder class, not directly implemented
                continue

            graph['classes'][class_uri] = {
                'obj_properties': self.get_properties_for_class(class_uri),
                'properties': list(self.get_all_data_in_item(class_uri)),
                'subclass_of': self.get_superclass_of_class(class_uri),
            }

        caching.put_in_cache(CACHE_DIR, self.sparql_endpoint, graph)

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

    for type_name, type_props in pg['types'].items():
        type_uri = rdflib.URIRef(type_name)
        g.add((
            type_uri,
            rdflib.namespace.RDF.type,
            rdflib.namespace.RDF.type,
        ))
        _fill_properties(type_uri, type_props)

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
    ont = Ontology(args.sparql_server, args.sparql_endpoint)
    pg = ont.get_all_properties_in_graph()
    rdf = property_graph_to_rdf(pg)
    args.output.write(rdf.serialize(format='pretty-xml'))
