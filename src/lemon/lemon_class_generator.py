from __future__ import annotations

import logging
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Dict, Optional, Set, Union, List, Tuple

import pydot as pydot
from rdflib import Graph, URIRef, RDF, Literal, BNode
from rdflib.term import Node

from dudes import utils
from dudes.dot import DotFile


class AttrType(Enum):
    """Type of a generated class/attribute."""
    STR = 0,
    """Attribute has type Optional[str] or List[Optional[str]]."""
    CLASS = 1,
    """Attribute has type Optional[ClassName] or List[Optional[ClassName]]."""
    ENUM = 2,
    """Attribute has type Optional[ClassName] or List[Optional[ClassName]], but ClassName is an Enum."""
    CLASSORSTR = 3
    """Attribute has type Optional[Union[ClassName, str]] or List[Optional[Union[ClassName, str]]]."""


class Pruned(Enum):
    """Placeholder which indicates that values have been pruned here (like, e.g., many DBPEDIA entities)."""
    PRUNED = 0


@dataclass(frozen=True)
class LEMONClassDescriptor(object):
    """Collection of necessary information to later generate python classes from this."""
    classname: str
    """Name of the described (non-enum) class (in PascalCase)."""
    const: Tuple[str, ...] = field(default_factory=tuple)
    """Vars with constant values."""
    complex: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    """Vars with complex values."""
    complex_or_const: Tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Vars with complex or constant values."""
    list_const: Tuple[str, ...] = field(default_factory=tuple)
    """Lists of vars with constant values."""
    list_complex: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    """Lists of vars with complex values."""
    list_complex_or_const: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    """Lists of vars with complex or constant values."""


@dataclass(frozen=True)
class LEMONEnumClassDescriptor(object):
    """Collection of necessary information to later generate python enums from this."""
    classname: str
    """Name of the described (enum) class (in PascalCase)."""
    enum_vals: Tuple[str, ...]
    """Different values of the enum (in ALLCAPS)"""


class LEMONDataClassGenerator(object):
    """Generates python classes from LEMON rdf graph."""

    TYPEDICT = Dict[Optional[str], Optional['TYPEDICT']]
    """Lists different predicates that are associated with some subject, and which predicates follow those predicates 
    etc. recursively as long as the value/object of the predicate is not a string literal (or a placeholder)"""

    FLATTYPEDICT = Optional[
        Dict[
            Optional[str],
            Set[Optional[str]]
        ]
    ]
    """Flattened TYPEDICT, lists for every predicate in the graph which different predicates follow it (or None if 
    the predicate is not followed by any predicate, i.e. its object is a string value or a placeholder)"""

    PRUNEDDICT = Optional[
        Dict[
            Optional[str],
            Union[list[Optional[str]], Pruned, None]
        ]
    ]
    """Pruned FLATTYPEDICT, where entries like DBPEDIA links have been replaced by PRUNED."""

    CLASSTYPES = Dict[str, Dict[str, AttrType]]
    """Describes what types the members of a class have. 
    Structure: Classname -> variable name in that class -> type of that variable."""

    LISTVARMAP = Dict[Optional[str], Set[str]]
    """Describes which predicates occur **multiple times** after another predicate, i.e. which have to be stored in a 
    list. Structure: Predicate name (or None if predicate is direct member of LexicalEntry) -> set of predicates which 
    occur multiple times after that predicate. 
    
    Example: When we have a lexical entry which has multiple otherForm successors, we get:: 
        
        {
            None: {otherForm}
        }
    
    Similarly, for subsenses of a Sense we get::
    
        {
            Sense: {subsense}
        }
    
    """

    _lvmap: LISTVARMAP
    _pruned: PRUNEDDICT

    def __init__(self,
                 graph: Graph,
                 base_uri: str = 'http://localhost:8000/lemon.owl',
                 placeholders: Optional[List[str]] = None):
        """
        Create LEMONDataClassGenerator from (LEMON) rdflib graph, the LEMON base uri (used to determine URI of
        LexicalEntry to start with) and list of placeholders (which are not further analyzed, by default
        ["arg1", "arg2"]). Already generates relevant metadata such as the different value dict in the constructor.

        :param graph: LEMON rdflib graph to deduce structure of LEMON from.
        :type graph: Graph
        :param base_uri: Base URI of LEMON, "#LexicalEntry" is appended to get LexicalEntry URI.
        :type base_uri: str
        :param placeholders: Placeholders whose children are not further examined. By default ["arg1", "arg2"].
        :type placeholders: Optional[List[str]]
        """
        if placeholders is None:
            placeholders = ["arg1", "arg2"]
        self.base_uri = base_uri
        self.graph = graph
        self.placeholders = placeholders
        self._refresh_metadata()

    def _refresh_metadata(self):
        dvd, self._lvmap = self._diff_val_dict()
        dvd["lemon:uri"] = None
        flatdvd: LEMONDataClassGenerator.FLATTYPEDICT = self._flatten_dict({"lemon:LexicalEntry": dvd})
        self._pruned = self._prune_flat_dict(flatdvd)
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")

    @property
    def entry_nodes(self) -> list[Node]:
        """
        :return: List of LexicalEntry nodes in the current lexicon graph.
        :rtype: list[Node]
        """
        entries = [sp[0] for sp in self.graph.subject_predicates(URIRef(self.base_uri + "#LexicalEntry"))
                   if sp[1] == RDF.type and not str(sp[0]).startswith(self.base_uri)]
        return entries

    @staticmethod
    def capitalize(n: str) -> str:
        """
        :param n: String to capitalize
        :type n: str
        :return: Makes the first letter of n a capital letter and does not change the remaining string.
        :rtype: str
        """
        if len(n) <= 1:
            return n.upper()
        else:
            return n[0].upper() + n[1:]

    @staticmethod
    def clean_namespaces(n: str) -> str:
        """
        :param n: QName with namespaces, e.g. lexicon:book_by
        :type n: str
        :return: Remove all namespaces of n by removing everything from the beginning up to (and including) the last ":"
        :rtype: str
        """
        return n[n.rfind(":") + 1:]

    @staticmethod
    def varname(n: str) -> str:
        """
        :param n: String in camelCase
        :type n: str
        :return: Converts camelCase to snake_case for usage as variable/attribute name
        :rtype:
        """
        return re.sub(r'(?<!^)(?=[A-Z])', '_', n).lower()

    def _is_const(self, n: str) -> bool:
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")
        return self._pruned[n] is None or self._pruned[n] == Pruned.PRUNED

    def _has_none(self, n: str) -> bool:
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")
        return self._pruned[n] is None or (self._pruned[n] != Pruned.PRUNED and None in self._pruned[n])

    def _is_enum(self, n: str) -> bool:
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")
        tpossible_vals = self._pruned[n]

        if tpossible_vals is None or tpossible_vals == Pruned.PRUNED or None in tpossible_vals or "lemon:uri" in tpossible_vals:
            return False

        tconsts = len([LEMONDataClassGenerator.clean_namespaces(v)
                       for v in tpossible_vals if
                       v is not None and self._is_const(v) and self._pruned[v] != Pruned.PRUNED])
        tclasses = len([LEMONDataClassGenerator.clean_namespaces(v)
                        for v in tpossible_vals if v is not None and not self._is_const(v)])

        return tclasses == 0 and tconsts > 0

    def _is_list(self, ppred: str, pred: str) -> bool:
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")
        if ppred == "lemon:LexicalEntry" and pred in self._lvmap[None]:
            return True
        else:
            return pred in self._lvmap[ppred]

    def _gen_class_descriptors(self) -> tuple[
        List[Union[LEMONClassDescriptor, LEMONEnumClassDescriptor]],
        CLASSTYPES
    ]:
        if self._pruned is None:
            raise RuntimeError("Unexpected None where dict should be!")

        q: Queue[str] = Queue()
        q.put("lemon:LexicalEntry")

        visited: Set[str] = set()

        class_types: LEMONDataClassGenerator.CLASSTYPES = defaultdict(dict)
        class_descs: List[Union[LEMONClassDescriptor, LEMONEnumClassDescriptor]] = []

        while not q.empty():
            name: str = q.get()
            possible_vals = self._pruned[name]
            # print(name, possible_vals)

            if name in visited:
                continue
            if possible_vals == Pruned.PRUNED:
                logging.warning("Reached _pruned, which should have been handled like string!")
                continue

            visited.add(name)

            if possible_vals is None:
                raise RuntimeError("Unexpected None where dict should be -> No lexical entries?")

            consts_ns = [v for v in possible_vals if v is not None and self._is_const(v)]
            consts = [LEMONDataClassGenerator.clean_namespaces(v) for v in consts_ns]
            classes_ns = [v for v in possible_vals if v is not None and not self._is_const(v)]
            classes = [LEMONDataClassGenerator.clean_namespaces(v) for v in classes_ns]

            has_pruned = any([self._pruned[n] == Pruned.PRUNED for n in possible_vals if n is not None])

            classname = LEMONDataClassGenerator.capitalize(LEMONDataClassGenerator.clean_namespaces(name))

            if len(classes) > 0:
                class_descs.append(
                    LEMONClassDescriptor(
                        classname=classname,
                        const=tuple(set([
                            LEMONDataClassGenerator.varname(consts[i])
                            for i in range(len(consts)) if not self._is_list(name, consts_ns[i])
                        ])),
                        complex=tuple(set([
                            (
                                LEMONDataClassGenerator.varname(classes[i]),
                                LEMONDataClassGenerator.capitalize(classes[i])
                            )
                            for i in range(len(classes)) if
                            not self._has_none(classes_ns[i]) and not self._is_list(name, classes_ns[i])
                        ])),
                        complex_or_const=tuple(set([
                            (
                                LEMONDataClassGenerator.varname(classes[i]),
                                LEMONDataClassGenerator.capitalize(classes[i])
                            )
                            for i in range(len(classes)) if
                            self._has_none(classes_ns[i]) and not self._is_list(name, classes_ns[i])
                        ])),
                        list_const=tuple(set([
                            LEMONDataClassGenerator.varname(consts[i])
                            for i in range(len(consts)) if self._is_list(name, consts_ns[i])
                        ])),
                        list_complex=tuple(set([
                            (
                                LEMONDataClassGenerator.varname(classes[i]),
                                LEMONDataClassGenerator.capitalize(classes[i])
                            )
                            for i in range(len(classes)) if
                            not self._has_none(classes_ns[i]) and self._is_list(name, classes_ns[i])
                        ])),
                        list_complex_or_const=tuple(set([
                            (
                                LEMONDataClassGenerator.varname(classes[i]),
                                LEMONDataClassGenerator.capitalize(classes[i])
                            )
                            for i in range(len(classes)) if
                            self._has_none(classes_ns[i]) and self._is_list(name, classes_ns[i])
                        ]))
                    )
                )
                for c in consts:
                    class_types[classname][LEMONDataClassGenerator.varname(c)] = AttrType.STR
                for c in classes_ns:  # use names with namespace here
                    varname = LEMONDataClassGenerator.varname(LEMONDataClassGenerator.clean_namespaces(c))
                    # TODO: Union[Enum, str] useless because str values would have been merged with enum?
                    if self._has_none(c):
                        class_types[classname][varname] = AttrType.CLASSORSTR
                    else:
                        class_types[classname][varname] = AttrType.ENUM if self._is_enum(c) else AttrType.CLASS
                    q.put(c)
            elif len(consts) > 0:
                if has_pruned:
                    class_descs.append(
                        LEMONClassDescriptor(
                            classname=classname,
                            const=tuple(set([
                                LEMONDataClassGenerator.varname(consts[i])
                                for i in range(len(consts)) if not self._is_list(name, consts_ns[i])
                            ])),
                            list_const=tuple(set([
                                LEMONDataClassGenerator.varname(consts[i])
                                for i in range(len(consts)) if self._is_list(name, consts_ns[i])
                            ]))
                        )
                    )
                    for c in consts:
                        varname = LEMONDataClassGenerator.varname(c)
                        class_types[classname][varname] = AttrType.STR
                else:
                    class_descs.append(
                        LEMONEnumClassDescriptor(
                            classname=classname,
                            enum_vals=tuple(set([c.upper() for c in consts]))
                        )
                    )
            else:
                logging.warning("Neither consts nor classes?")
                raise RuntimeError("Neither consts nor classes?")

        return list(dict.fromkeys(class_descs)), class_types

    def gen_classes(self) -> Tuple[str, CLASSTYPES]:
        """
        Generates python classes which resemble the structure of the given LEMON lexicon.
        :return: Tuple cosnisting of the class definitions (as string) and the types of each class member.
        :rtype: Tuple[str, CLASSTYPES]
        """
        res: List[str] = []
        class_descs, class_types = self._gen_class_descriptors()
        for cd in class_descs:
            if isinstance(cd, LEMONClassDescriptor):
                res.append(
                    """
                @dataclass
                class {}(object):
                    # Vars with constant values
                    {}
                    # Vars with complex values
                    {}
                    # Vars with complex or constant values
                    {}
                    # Lists of vars with constant values
                    {}
                    # Lists of vars with complex values
                    {}
                    # Lists of vars with complex or constant values
                    {}                         
                """.format(
                        cd.classname,
                        """
                    """.join(sorted(
                            ["{}: Optional[str] = None".format(c) for c in cd.const]
                        )),
                        """
                    """.join(sorted(
                            ["{}: Optional[{}] = None".format(c1, c2) for c1, c2 in cd.complex]
                        )),
                        """
                    """.join(sorted(
                            ["{}: Optional[Union[{}, str]] = None".format(c1, c2) for c1, c2 in cd.complex_or_const]
                        )),
                        """
                    """.join(sorted(
                            ["{}: list[Optional[str]] = field(default_factory=list)".format(c) for c in cd.list_const]
                        )),
                        """
                    """.join(sorted(
                            ["{}: list[Optional[{}]] = field(default_factory=list)".format(c1, c2) for c1, c2 in
                             cd.list_complex]
                        )),
                        """
                    """.join(
                            sorted(["{}: list[Optional[Union[{}, str]]] = field(default_factory=list)".format(c1, c2)
                                    for c1, c2 in cd.list_complex_or_const]))
                    )
                )
            elif isinstance(cd, LEMONEnumClassDescriptor):
                res.append(
                    """
                class {}(Enum):
                    {}                        
                """.format(
                        cd.classname,
                        """
                    """.join(sorted(["{} = {}".format(list(sorted(cd.enum_vals))[i], i) for i in range(len(cd.enum_vals))]))
                    )
                )
            else:
                raise RuntimeError("Unexpected class found!")

        header = "from __future__ import annotations\nfrom dataclasses import dataclass, field\nfrom enum import " \
                 "Enum\nfrom typing import Union, Optional\n"
        body = "\n".join(sorted(res))
        return header + textwrap.dedent(body) + "\nLEMON = Union[" + ", ".join(
            [cd.classname for cd in class_descs]) + "]", class_types

    def _gen_class_dot(self, skip_consts=False):
        class_descs, class_types = self._gen_class_descriptors()
        graph = pydot.Dot("LEMON", graph_type="digraph")
        for cd in class_descs:
            graph.add_node(pydot.Node(name=cd.classname, label=cd.classname, shape="rectangle"))
        for cd in [cd for cd in class_descs if isinstance(cd, LEMONClassDescriptor)]:
            if not skip_consts:
                for cn in cd.const:
                    graph.add_edge(pydot.Edge(cd.classname, cn, label=""))
            for vn, cn in cd.complex:
                graph.add_edge(pydot.Edge(cd.classname, cn, label=""))
            for vn, cn in cd.complex_or_const:
                graph.add_edge(pydot.Edge(cd.classname, cn, label="", style="dotted"))
            if not skip_consts:
                for cn in cd.list_const:
                    graph.add_edge(pydot.Edge(cd.classname, cn, label="list"))
            for vn, cn in cd.list_complex:
                graph.add_edge(pydot.Edge(cd.classname, cn, label="list"))
            for vn, cn in cd.list_complex_or_const:
                graph.add_edge(pydot.Edge(cd.classname, cn, label="list", style="dotted"))
        # DotFile.runXDot(dotStr=graph.to_string())
        logging.debug(graph.to_string())
        return graph

    def _diff_val_dict(self) -> tuple[TYPEDICT, LISTVARMAP]:
        cdict: LEMONDataClassGenerator.TYPEDICT = dict()
        lvmap: LEMONDataClassGenerator.LISTVARMAP = defaultdict(set)

        for node in self.entry_nodes:
            cdict, lvmap = self._node_diff_val_dict(node=node,
                                                    cdict=cdict,
                                                    lvmap=lvmap,
                                                    path=[])

        return cdict, lvmap

    def get_types(self, node: Node) -> Set[str]:
        """
        :param node: Node to get types for.
        :type node: Node
        :return: Types of the given node, i.e., objects of RDF triples (node, a, <obj>)
        :rtype: Set[str]
        """
        return set([str(obj) for obj in self.graph.objects(subject=node, predicate=RDF.type)])

    def get_prev_preds(self, node: Node) -> Set[str]:
        """
        :param node: Node to get predicates for.
        :type node: Node
        :return: Get predicates of which the given node is an object.
        :rtype: Set[str]
        """
        return set([str(pred) for subj, pred in self.graph.subject_predicates(node)])

    def _node_diff_val_dict(self,
                            node: Node,
                            cdict: TYPEDICT,
                            lvmap: LISTVARMAP,
                            path: list[tuple[str, str]],
                            loop_nodes: Optional[set[str]] = None) -> tuple[TYPEDICT, LISTVARMAP]:
        if loop_nodes is None:
            loop_nodes = set()
        res: LEMONDataClassGenerator.TYPEDICT = cdict
        reslv: LEMONDataClassGenerator.LISTVARMAP = lvmap
        assert res is not None

        snode = str(node) if isinstance(node, Literal) or isinstance(node, BNode) else self.graph.qname(utils.sanitize_url(str(node)))

        # print(node)
        pred_visited: Set[str] = set()
        # sameas: Set[str] = set()
        for pred, obj in self.graph.predicate_objects(node):
            spred = self.graph.qname(utils.sanitize_url(str(pred)))
            sobj = str(obj) if isinstance(obj, Literal) or isinstance(obj, BNode) else self.graph.qname(utils.sanitize_url(str(obj)))

            # if spred == "rdf:type":
            #    sameas.add(sobj)

            if "ns1" in sobj:
                logging.warning(sobj + " " + str(obj))

            if spred == "rdf:type":
                if spred not in res.keys():
                    res[spred] = dict()
                res[spred][sobj] = None
                if len(path) > 0:
                    reslv[path[-1][1]].add(spred)
                else:
                    reslv[None].add(spred)

            if spred[:3] in ["rdf"]: # , "owl"
                continue

            if spred in pred_visited:
                if len(path) > 0:
                    if 'lemon:subjOfProp' == path[-1][1] and spred == 'lemon:marker':
                        pass
                    reslv[path[-1][1]].add(spred)
                else:
                    reslv[None].add(spred)

            pred_visited.add(spred)

            if spred not in res.keys():
                res[spred] = dict()

            if obj == node or sobj in loop_nodes:
                continue
            elif sobj in [sn for sn, sp in path]:
                res[spred][sobj] = None
                loop_nodes.add(sobj)
                logging.debug("Loop detected: {} {}, enforcing string value.".format(spred, sobj))
                continue

            if isinstance(obj, Literal) or sobj in self.placeholders:
                res[spred][sobj] = None
            else:
                node_dict, reslv = self._node_diff_val_dict(node=obj,
                                                            cdict=dict(),
                                                            lvmap=reslv,
                                                            path=path + [(snode, spred)],
                                                            loop_nodes=loop_nodes)
                if len(node_dict) == 0:
                    res[spred][sobj] = None
                else:
                    res[spred] = self._unify_dicts(res[spred], node_dict)
                    res[spred]["lemon:uri"] = None

        # if 'http://localhost:8000/lemon.owl#Form' in self.get_classes(node) and len(reslv[snode]) > 0:
        #    print("Bad:", str(node))

        return res, reslv

    @staticmethod
    def _unify_dicts(a: TYPEDICT | None, b: TYPEDICT | None) -> TYPEDICT | None:
        res: LEMONDataClassGenerator.TYPEDICT
        if a is None and b is None:
            return None
        elif a is None and b is not None:
            res = dict(b)
            res[None] = None
            return res
        elif a is not None and b is None:
            res = dict(a)
            res[None] = None
            return res
        elif a is not None and b is not None:
            res = dict(b)  # TODO: handle if a or b are None
            for k, v in a.items():
                if k in res.keys():
                    if res[k] == a[k]:
                        res[k] = v
                    else:
                        res[k] = LEMONDataClassGenerator._unify_dicts(a[k], res[k])
                else:
                    res[k] = v
            return res
        else:
            raise RuntimeError("Should never happen")

    def _flatten_dict(self, cdict: TYPEDICT | None) -> FLATTYPEDICT:
        if cdict is None:
            return None
        else:
            res: LEMONDataClassGenerator.FLATTYPEDICT = dict()
            assert res is not None
            for k, v in cdict.items():
                #if k == 'lemon:phraseRoot':
                #    pass
                if k not in res.keys():
                    res[k] = set()

                tres = self._flatten_dict(cdict[k])
                if v is None or tres is None:
                    res[k].add(None)
                else:
                    for k2 in v.keys():
                        res[k].add(k2)
                    for k2, v2 in tres.items():
                        if k2 not in res.keys():
                            res[k2] = set()
                        res[k2].update(v2)
            return res

    @staticmethod
    def _prune_flat_dict(flat_dict: FLATTYPEDICT,
                         prune_prefixes=None
                         ) -> PRUNEDDICT:
        if prune_prefixes is None:
            prune_prefixes = ["dbpedia", "default1", "local", "dbo", "dbp", "dbr", "lexicon"]#"owl,
        if flat_dict is None:
            return None
        pruned: LEMONDataClassGenerator.PRUNEDDICT = dict()
        if pruned is None or pruned == Pruned.PRUNED:
            raise RuntimeError("Invalid pruned dict!")

        pruned["rdf:type"] = Pruned.PRUNED

        for k, v in flat_dict.items():
            if ":" not in str(k):  # todo: prune also dbpedia: -> and what is ns1??
                continue
            elif len(v) == 1 and list(v)[0] is None:
                pruned[k] = None
            else:
                pruned_vals = set([n for n in v for pref in prune_prefixes if str(n).startswith(pref)])
                if len(pruned_vals) > 0:
                    if any([flat_dict.get(n) is not None and len(flat_dict.get(n)) - 1 > (1 if None in flat_dict.get(n) else 0) for n in v if ":" in str(n)]):
                        # some values have successors, treat like CLASSORSTR instead
                        res = [n for n in v if ":" in str(n) and n not in pruned_vals]
                        res.append(None)
                        pruned[k] = res
                    else:
                        pruned[k] = Pruned.PRUNED  # no enum when it could contain now ignored stuff like dbpedia
                else:
                    res = [n for n in v if ":" in str(n)]
                    # as an pruning directly leads to str type, we can omit this: and not any([str(n).startswith(pref) for pref in prune_prefixes])
                    if len(res) == 0:
                        pruned[k] = None
                    else:
                        if any([":" not in str(n) for n in v]):
                            res.append(None)
                        pruned[k] = res

        pruned["rdf:type"] = Pruned.PRUNED

        return pruned
