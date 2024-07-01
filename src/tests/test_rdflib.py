import bz2
import os
import pathlib
import sys

import networkx as nx
from matplotlib import pyplot as plt
from networkx import dfs_preorder_nodes
from rdflib import Graph, URIRef, Literal, OWL, Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
from rdflib.extras.infixowl import AllClasses
from rdflib.namespace import NamespaceManager
import lemon

def test_create_lexicon():
    graph = Graph()
    # graph.parse("../resources/lexinfo.owl", format='xml')
    # graph.parse("../resources/oils.owl", format='xml')
    rootpath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources")
    graph.parse(os.path.join(rootpath, "lemon.ttl"), format='ttl')
    root = pathlib.Path(os.path.join(rootpath, "lexicon"))  # pathlib.Path("dudes/resources/lexicon/")
    paths = root.rglob("*.ttl")
    # print(list(paths))
    for path in paths:
        graph.parse(path, format='ttl')
    print(len(graph))
    graph.serialize(destination=os.path.join(rootpath, "lexicon.owl"), format="pretty-xml")
    graph.serialize(destination=os.path.join(rootpath, "lexicon.ttl"), format="ttl")

def test_dbpedia_schema():
    graph = Graph()
    # graph.parse("../resources/lexinfo.owl", format='xml')
    # graph.parse("../resources/oils.owl", format='xml')
    rootpath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources")
    graph.parse(os.path.join(rootpath, "dbpedia-schema.owl"), format='xml')

    root = pathlib.Path("/home/dvs23/Downloads/dbpedia/")  # pathlib.Path("dudes/resources/lexicon/")
    paths = root.rglob("*.ttl.bzip2")
    # print(list(paths))
    for path in paths:
        print(path, flush=True)
        with bz2.open(path, "r") as file:
            graph.parse(file, format='ttl')

    print(len(graph))
    digraph = rdflib_to_networkx_digraph(graph)
    digraph.remove_node(URIRef('http://dbpedia.org/ontology/'))
    digraph.remove_node(URIRef('http://dbpedia.org/ontology/data/definitions.ttl'))
    res = nx.shortest_simple_paths(G=digraph, source=URIRef("http://dbpedia.org/ontology/leaderName"), target=URIRef("http://dbpedia.org/ontology/Mayor"))
    i = 0
    for p in res:
        print(p, flush=True)
        i += 1
        if i >= 20:
            break
    print(list(graph.subject_predicates(URIRef("http://dbpedia.org/ontology/leaderName"))))
    print(list(graph.predicate_objects(URIRef("http://dbpedia.org/ontology/leaderName"))))


def test_rdflib():
    print("")
    lemon = Namespace('http://localhost:8000/lemon.owl')
    # lemon = Namespace('http://localhost:8000/lemon.owl')
    namespace_manager = NamespaceManager(Graph())
    namespace_manager.bind('lemon', lemon, override=False)
    namespace_manager.bind('owl', OWL, override=True)
    graph = Graph()
    rootpath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources")
    graph.parse(os.path.join(rootpath, "lemon.ttl"))
    graph.namespace_manager = namespace_manager
    root = pathlib.Path(os.path.join(rootpath, "lexicon"))
    paths = root.rglob("*.ttl")
    # print(list(paths))
    for path in paths:
        graph.parse(path, format='ttl')
    # graph.parse('../resources/lexicon/nouns/NounPPFrame-lexicon-album-of.ttl', format='ttl')
    # graph.parse('../resources/lexicon/verbs/IntransitivePPFrame-lexicon-draft_in.ttl', format='ttl')
    nodes = [graph.qname(str(o)) for o in graph.objects() if isinstance(o, URIRef)]
    nodes2 = [str(o) for o in graph.objects() if isinstance(o, Literal)]
    print(list(graph.subject_objects(predicate=URIRef('http://lemon-model.net/lemon#writtenRep'))))
    lexical_entries = list(graph.subject_predicates(URIRef("http://localhost:8000/lemon.owl#LexicalEntry")))
    # print(lexical_entries)
    print(list(AllClasses(graph)))

    list(graph.predicate_objects(lexical_entries[0][0]))

    digraph = rdflib_to_networkx_digraph(graph)
    pred = list(digraph.predecessors(URIRef('http://localhost:8000/lemon.owl#LexicalEntry')))
    bfsnodes = [s for s in dfs_preorder_nodes(digraph, pred[0])]
    print(bfsnodes)
    subgraph = digraph.subgraph(bfsnodes)

    nx.draw(subgraph, with_labels=True)
    plt.show()

    # with open("test.dot", "w") as file:
    # rdf2dot(graph, file)
    # print(stream.getvalue())


if __name__ == "__main__":
    test_rdflib()
