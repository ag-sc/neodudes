from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

from rdflib import Graph, OWL, URIRef, Literal, RDF
from rdflib.namespace import NamespaceManager, Namespace
from rdflib.plugins.parsers.notation3 import BadSyntax
from rdflib.term import Node, BNode

from dudes import utils
from lemon.csv_to_lemon import csv_to_lemon
from lemon.lemon import *
from lemon.lemon_class_generator import LEMONDataClassGenerator, AttrType
import dudes
from lemon.lexicon import Lexicon
from lemon.namespaces import default_namespaces

error_docs: List[str] = [] + ["_TEST"]
#"Have_seat", "state_in", "largest_country"
#['operate_1', 'create_1', "Develop_3", 'call_1']
#['compose_for_1', 'operate_1', 'call_1', 'create_1', "Develop_3"]


# ['belong_to_10', 'founded_on_3', 'end_in_2', 'locate_in_11', 'produced_in_1', 'compose_for_3',
#                      'discontinue_in_1', 'born_in_7', 'belong_to_15', 'locate_in_6', 'reopen_on_1', 'study_at_1',
#                      'released_on_4', 'operate_1', 'live_in_2', 'relate_to_1', 'associate_with_3', 'locate_in_1',
#                      'released_on_1', 'die_from_1', 'introduced_on_1', 'published_in_3', 'discontinue_on_1',
#                      'build_in_1', 'destroy_in_1', 'die_in_1', 'initially_use_for_1', 'start_in_1', 'name_after_1',
#                      'born_in_2', 'play_in_1', 'grow_in_2', 'stop_acting_on_2', 'locate_in_8', 'belong_to_18',
#                      'belong_to_12', 'complete_in_2', 'associate_with_2', 'belong_to_11', 'born_in_3',
#                      'published_on_1', 'locate_in_7', 'born_in_1', 'play_in_3', 'participate_in_1', 'end_in_5',
#                      'rebuild_on_1', 'to_call_1', 'come_from', 'inaugurate_in_2', 'belong_to_7', 'starr_in_3',
#                      'live_in_3', 'locate_in_2', 'bury_in_2', 'belong_to_1', 'build_in_2', #'written_in',
#                      'originate_in_1', 'Have_seat_1', 'born_in_6', 'flow_into_2', 'die_in_2', 'complete_in_5',
#                      'flow_into', 'released_on_3', 'locate_in_3', 'belong_to_9', 'end_in_3', 'commence_on_2',
#                      'compose_for_2', 'belong_to_14', 'belong_to_8', 'end_in_4', 'rebuild_in_2', 'founded_on_2',
#                      'work_for_1', 'born_in_4', 'build_in_3', 'currently_use_for_1', 'begin_at_2',
#                      'start_acting_on_1', 'die_in_3', 'bury_in_1', 'published_in_1', 'reopen_in_2', 'dissolve_on_1',
#                      'belong_to_17', 'commence_on_1', 'produced_on_3', 'play_in_2', 'establish_in_1', 'open_on_1',
#                      'grow_in', 'die_from_2', 'live_in_4', 'end_in_6', 'flow_into_1', 'locate_in_4',
#                      'to_compose_for_1', 'work_for_2', 'published_on_2', 'complete_in_3', 'locate_in_12',
#                      'fight_in_1', 'border_in_1', 'stand_for_2', 'locate_in_10', 'found_in_5', 'complete_in_4',
#                      'released_on_2', 'associate_with_1', 'published_in_2', 'published_on_3', 'belong_to_3',
#                      'belong_to_4', 'complete_in_1', 'play_for_1', 'form_in_1', 'introduced_in__2', 'fight_in',
#                      'produced_on_1', 'destroy_on_1', 'born_on_1', 'produced_on_2', 'belong_to_6', 'begin_at_1',
#                      'compose_for_1', 'found_in_3', 'end_career_on', 'found_in_4', 'flow_through_2', 'born_in_5',
#                      'open_in_1', 'belong_to_13', 'die_in_4', 'record_in_1', 'call_1', 'draft_in', 'live_in_1',
#                      'end_in_1', 'inaugurate_in_1', 'locate_in_5', 'die_on_1', 'inaugurate_in_3', 'founded_on_1',
#                      'split_up_in_1', 'start_in_2', 'flow_through_1', 'die_in_5', 'locate_in_9', 'to_operate_1']


class LEMONParser(object):
    """Parser to generate LexicalEntry objects from LEMON rdflib graph."""

    def __init__(self,
                 to_parse: List[pathlib.Path],
                 base_uri: str = 'http://localhost:8000/lemon.owl',
                 placeholders: Optional[List[str]] = None,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 list_entries: Optional[List[LexicalEntry]] = None,
                 namespace_manager: Optional[NamespaceManager] = None):
        """
        Create LEMONParser from list of paths to parse (with rdflib to create RDF graph), base URI of LEMON and a list
        of used placeholders.

        :param to_parse: List of paths to, e.g., TTL files to parse with rdflib to create RDF graph
        :type to_parse: List[pathlib.Path]
        :param base_uri: Base URI of LEMON, "#LexicalEntry" is appended to get LexicalEntry URI.
        :type base_uri: str
        :param placeholders: Placeholders whose children are not further examined. By default ["arg1", "arg2"].
        :type placeholders: Optional[List[str]]
        :param namespaces: List of namespaces which should be resolved. Structure: tuple(<name>, Namespace(<uri>)).
        :type namespaces: Optional[Iterable[Tuple[str, Namespace]]]
        :param list_entries: List of LexicalEntry objects extracted from CSV lists.
        :type list_entries: Optional[List[LexicalEntry]]
        """
        if placeholders is None:
            placeholders = ["arg1", "arg2"]
        if namespaces is None:
            namespaces = default_namespaces
        if list_entries is None:
            list_entries = []

        self.base_uri = base_uri
        self.placeholders = placeholders
        self.list_entries = list_entries

        self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=namespace_manager)

        self.graph = Graph()
        self.graph.namespace_manager = self.nsmanager

        rootpath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources")
        logging.info(rootpath)

        self.graph.parse(os.path.join(rootpath, "lemon.ttl"))

        for path in to_parse:
            if any([e in str(path) for e in error_docs]):
                continue
            try:
                self.graph.parse(path)
            except BadSyntax as e:
                logging.warning(f"Invalid ttl syntax for {path}")
            except RuntimeError as e:
                logging.warning(f"Error while parsing {path}: {e}")

        for name, ns in namespaces:#set namespaces again
            self.graph.namespace_manager.bind(name, ns, override=True, replace=True)

        self.cg: LEMONDataClassGenerator = LEMONDataClassGenerator(graph=self.graph, base_uri=self.base_uri)
        self.classtypes: LEMONDataClassGenerator.CLASSTYPES = self.cg.gen_classes()[1]

        # print("Namespaces:", list(self.graph.namespaces()))

    @classmethod
    def from_ttl_dir(cls,
                     ttl_dir: str = os.path.join(
                         os.path.dirname(sys.modules["lemon"].__file__),
                         "resources",
                         "lexicon"
                     ),
                     csv_dir: str = os.path.join(
                         os.path.dirname(sys.modules["lemon"].__file__),
                         "resources",
                         "entityLists"
                     ),
                     base_uri: str = 'http://localhost:8000/lemon.owl',
                     placeholders: Optional[List[str]] = None,
                     namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                     namespace_manager: Optional[NamespaceManager] = None
                     ):
        """
        Create LEMONParser from directory, which is recursively searched for .ttl files, base URI of LEMON and a list
        of used placeholders.

        :param ttl_dir: Directory where .ttl files are searched recursively in. Resulting paths are passed to regular constructor.
        :type ttl_dir: str
        :param csv_dir: Directory where .csv files are searched recursively in. Resulting paths are translated to Lemon entries.
        :type csv_dir: str
        :param base_uri: Base URI of LEMON, "#LexicalEntry" is appended to get LexicalEntry URI.
        :type base_uri: str
        :param placeholders: Placeholders whose children are not further examined. By default ["arg1", "arg2"].
        :type placeholders: Optional[List[str]]
        :return: LEMONParser created from directory of .ttl files
        :rtype: LEMONParser
        """
        csv_root = pathlib.Path(csv_dir)
        list_entries: List[LexicalEntry] = []
        for path in csv_root.rglob("*.csv"):
            list_entries.extend(csv_to_lemon(path=str(path)))

        root = pathlib.Path(ttl_dir)
        paths = list(root.rglob("*.ttl"))
        return cls(to_parse=paths, base_uri=base_uri, placeholders=placeholders, list_entries=list_entries, namespaces=namespaces, namespace_manager=namespace_manager)

    @property
    def entry_nodes(self) -> list[Node]:
        """
        :return: List of LexicalEntry nodes in the current lexicon graph.
        :rtype: list[Node]
        """
        entries = [sp[0] for sp in self.graph.subject_predicates(URIRef(self.base_uri + "#LexicalEntry"))
                   if sp[1] == RDF.type and not str(sp[0]).startswith(self.base_uri)]
        return entries

    @property
    def lexicon(self) -> Lexicon:
        entries = self.parse_nodes(self.entry_nodes)
        return Lexicon(entries, namespace_manager=self.nsmanager)

    def to_dict(self,
                node: Node) -> Dict[str, str | Dict | list]:
        """
        Convert rdflib graph node and its successors to dictionary representation. Keys are predicates, (final) values
        are literals or URIs which are not further resolved.

        **Caution! Only immediate self-loops are ignored. Other circles within the graph lead to infinite recursion!**

        :param node: Node to start graph traversal at.
        :type node: Node
        :return: Dictionary representation of nodes and its successors.
        :rtype: Dict[str, str | Dict | list]
        """
        res: Dict[str, str | Dict | list] = dict()

        for pred, obj in self.graph.predicate_objects(node):
            spred = self.graph.qname(utils.sanitize_url(str(pred)))
            sobj = str(obj) if isinstance(obj, Literal) or isinstance(obj, BNode) else self.graph.qname(utils.sanitize_url(str(obj)))
            if spred == "rdf:type":
                if spred not in res.keys():
                    res[spred] = [sobj]
                else:
                    res[spred].append(sobj)
                continue

            if node == obj:
                continue

            res["lemon:uri"] = str(node)

            to_assign: str | Dict | list

            if isinstance(obj, Literal) or sobj in self.placeholders:
                to_assign = sobj
            else:
                node_dict = self.to_dict(obj)

                if len(node_dict) == 0:
                    to_assign = self.graph.qname(utils.sanitize_url(str(obj)))
                else:
                    to_assign = node_dict

            if spred in res.keys():
                curr = res[spred]
                if isinstance(curr, list):
                    res[spred] = curr + [to_assign]
                else:
                    res[spred] = [curr, to_assign]
            else:
                res[spred] = to_assign

        return res

    def parse_nodes(self, nodes: Iterable[Node]) -> List[LexicalEntry]:
        """
        Parse list of LexicalEntry rdflib graph nodes to LexicalEntry python objects (of the generated LEMON classes).

        :param nodes: List of LexicalEntry rdflib graph nodes.
        :type nodes: Iterable[Node]
        :return: List of LexicalEntry python objects (of the generated LEMON classes).
        :rtype: List[LexicalEntry]
        """
        res = []
        for node in nodes:
            try:
                res.append(self.parse_node(node))
            except Exception as e:
                logging.warning(f"Error while parsing node {node}: {e}")
        return res + self.list_entries
        #return [self.parse_node(node) for node in nodes] + self.list_entries

    def parse_node(self, node: Node) -> LexicalEntry:
        """
        Parse LexicalEntry rdflib graph node to LexicalEntry python object (of the generated LEMON classes).

        :param node: LexicalEntry rdflib graph node.
        :type node: Node
        :return: LexicalEntry python object (of the generated LEMON classes).
        :rtype: LexicalEntry
        """
        node_dict: Dict = self.to_dict(node)  # {"lemon:LexicalEntry": }
        # print(json.dumps(node_dict, sort_keys=True, indent=4))
        return self.parse_dict("LexicalEntry", node_dict=node_dict)

    def parse_dict(self, classname: str, node_dict: Dict) -> LEMON:
        """
        Parse dictionary representation of the given class and return the corresponding LEMON objects.

        :param classname: Name of the class for which an object should be created from the given dictionary representation. **Must correspond to a class in the generated LEMON classes!**
        :type classname: str
        :param node_dict: Dictionary representation of the object to parse.
        :type node_dict: Dict
        :return: The parsed LEMON object.
        :rtype: LEMON
        """

        init_dict: Dict = dict()

        dummy_obj = globals()[classname]()

        for arg in node_dict.keys():
            argclassname = LEMONDataClassGenerator.capitalize(LEMONDataClassGenerator.clean_namespaces(arg))
            varname = LEMONDataClassGenerator.varname(LEMONDataClassGenerator.clean_namespaces(arg))
            attrtype = self.classtypes[classname][varname]
            obj = node_dict[arg]
            match attrtype:
                case AttrType.STR:
                    obj = node_dict[arg]
                case AttrType.ENUM:
                    klass = globals()[argclassname]
                    obj = klass[LEMONDataClassGenerator.clean_namespaces(node_dict[arg]).upper()]
                case AttrType.CLASS:
                    if isinstance(node_dict[arg], list):
                        obj = [self.parse_dict(argclassname, n) for n in node_dict[arg]]
                    else:
                        obj = self.parse_dict(argclassname, node_dict[arg])
                case AttrType.CLASSORSTR:
                    if isinstance(node_dict[arg], str):
                        obj = node_dict[arg]
                    else:
                        if isinstance(node_dict[arg], list):
                            obj = [self.parse_dict(argclassname, n) for n in node_dict[arg]]
                        else:
                            obj = self.parse_dict(argclassname, node_dict[arg])

            if isinstance(getattr(dummy_obj, varname), list):
                if varname not in init_dict.keys():
                    init_dict[varname] = []

                if isinstance(obj, set) or isinstance(obj, list):
                    init_dict[varname].extend(obj)
                else:
                    init_dict[varname].append(obj)
            else:
                if isinstance(obj, set) or isinstance(obj, list):
                    raise RuntimeError("Set where no multiple values are expected!")
                else:
                    init_dict[varname] = obj

        return globals()[classname](**init_dict)
