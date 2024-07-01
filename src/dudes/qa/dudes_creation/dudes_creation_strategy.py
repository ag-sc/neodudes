import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Iterable

from rdflib import Namespace, Graph
from rdflib.namespace import NamespaceManager

from treelib import Node, Tree

from dudes import consts, utils
from dudes.dudes import DUDES
from lemon.lemon import LexicalEntry, Sense, Reference, PartOfSpeech
from lemon.lexicon import Lexicon
from lemon.namespaces import default_namespaces


class DUDESCreationStrategy(ABC):
    @abstractmethod
    def node_to_dudes(self, node: Node, tree: Tree, next_var_id: int) -> Tuple[List[DUDES], int]:
        pass


class FixedTermsStrategy(DUDESCreationStrategy):
    def node_to_dudes(self, node: Node, tree: Tree, next_var_id: int) -> Tuple[List[DUDES], int]:
        res = []
        if utils.any_in_list(node.data.token.text_, consts.top_strong_keywords, lower=True):
            res = [
                DUDES.top(order_val='Degree.STRONG', count=1, var="x" + str(next_var_id), label="l" + str(next_var_id))
            ]
            next_var_id += 1
        elif utils.any_in_list(node.data.token.text_, consts.top_weak_keywords, lower=True):
            res = [
                DUDES.top(order_val='Degree.WEAK', count=1, var="x" + str(next_var_id), label="l" + str(next_var_id))
            ]
            next_var_id += 1
        elif utils.any_in_list(node.data.token.text_, ["in"], lower=True):
            res = [
                DUDES.binary(rel_name="dbo:location",
                             var_1="x" + str(next_var_id + 1),
                             var_2="x" + str(next_var_id + 2),
                             label="l" + str(next_var_id),
                             var_2_markers=["in"]),
                DUDES.binary(rel_name="dbo:locatedInArea",
                             var_1="x" + str(next_var_id + 1),
                             var_2="x" + str(next_var_id + 2),
                             label="l" + str(next_var_id),
                             var_2_markers=["in"]),
            ]
            next_var_id += 3
        elif utils.any_in_list(node.data.token.text_, [v + " than" for v in consts.comp_gt_keywords] + consts.comp_gt_keywords_no_than, lower=True):
            res = [
                DUDES.comp(num_var="x" + str(next_var_id + 1), order_val='Degree.STRONG',
                           var="x" + str(next_var_id + 2), label="l" + str(next_var_id)),
                DUDES.countcomp(num_var="x" + str(next_var_id + 1), order_val='Degree.STRONG',
                                var="x" + str(next_var_id + 2), group_var="x" + str(next_var_id + 3),
                                label="l" + str(next_var_id)),
            ]
            next_var_id += 4
        elif utils.any_in_list(node.data.token.text_, [v + " than" for v in consts.comp_lt_keywords] + consts.comp_lt_keywords_no_than, lower=True):
            res = [
                DUDES.comp(num_var="x" + str(next_var_id + 1), order_val='Degree.WEAK',
                           var="x" + str(next_var_id + 2), label="l" + str(next_var_id)),
                DUDES.countcomp(num_var="x" + str(next_var_id + 1), order_val='Degree.WEAK',
                                var="x" + str(next_var_id + 2), group_var="x" + str(next_var_id + 3),
                                label="l" + str(next_var_id)),
            ]
            next_var_id += 4
        elif utils.any_in_list(node.data.token.text_, consts.some_relation_words, lower=True):
            res = [
                DUDES.binary(rel_name="local:with",
                             var_1="x" + str(next_var_id + 1),
                             var_2="x" + str(next_var_id + 2),
                             label="l" + str(next_var_id)),
                DUDES.binary(rel_name="local:rwith",
                             var_1="x" + str(next_var_id + 1),
                             var_2="x" + str(next_var_id + 2),
                             label="l" + str(next_var_id))
            ]
            next_var_id += 3

        #additional TODO: others also additional not elif?
        if node.data.token.pos_ == "NUM" or any([t.pos_ == "NUM" for t in node.data.token.merged_tokens]):
            num_tokens = [t for t in [node.data.token] + node.data.token.merged_tokens if t.pos_ == "NUM"]

            res += [
                DUDES.equality(text=nt.text, var="x" + str(next_var_id), label="l" + str(next_var_id))
                for nt in num_tokens
            ]
            next_var_id += 1
        # elif node.data.token.text_.lower() in ["and"]:
        #     res = [
        #         DUDES.binary(rel_name="local:and",
        #                      var_1="x" + str(next_var_id + 1),
        #                      var_2="x" + str(next_var_id + 2),
        #                      label="l" + str(next_var_id))
        #     ]
        #     next_var_id += 3

        return res, next_var_id

class EntityURIStrategy(DUDESCreationStrategy):
    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 namespace_manager: Optional[NamespaceManager] = None):
        self.nsmanager = utils.create_namespace_manager(namespaces, namespace_manager)

    def node_to_dudes(self, node: Node, tree: Tree, next_var_id: int) -> Tuple[List[DUDES], int]:
        res = []
        if len(node.data.token.candidate_uris) > 0:
            # if any([self.nsmanager.qname(utils.sanitize_url(cand)).lower().startswith("ns1:") for cand in set(node.data.token.candidate_uris)]):
            #     pass
            res = [
                DUDES.equality(self.nsmanager.qname(utils.sanitize_url(cand)),
                               var="x" + str(next_var_id),
                               label="l" + str(next_var_id))
                for cand in utils.make_distinct_ordered(node.data.token.candidate_uris)
            ]
        return res, next_var_id + 1

class LemonStrategy(DUDESCreationStrategy):
    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 namespace_manager: Optional[NamespaceManager] = None):
        self.nsmanager = utils.create_namespace_manager(namespaces, namespace_manager)

    def node_to_dudes(self, node: Node, tree: Tree, next_var_id: int) -> Tuple[List[DUDES], int]:
        res: List[List[DUDES]] = []
        for entry in node.data.lex_candidates:
            try:
                senses = entry.sense if entry.sense is not None and len(entry.sense) > 0 else [None]
                for sense in senses:
                    res.append(
                        self._entry_to_dudes(entry=entry, node=node, tree=tree, sense=sense,
                                             var="x" + str(next_var_id), label="l" + str(next_var_id))
                    )
            except RuntimeError as e:
                logging.warning("entry_to_dudes failed:", e)
        return list(utils.roundrobin(*res)), next_var_id + 1

    def _entry_to_dudes(self,
                       entry: LexicalEntry,
                       node: Node,
                       tree: Tree,
                       sense: Optional[Sense] = None,
                       var: str = "x",
                       label: str = "l") -> List[DUDES]:
        """
        Create DUDES from a LexicalEntry and a tree node based on POS tags and lexinfo frames.
        Mostly adapted from `Java implementation <https://github.com/ag-sc/DUDES/blob/master/src/main/java/de/citec/sc/dudes/lemon/Lexicon2DUDES.java>`_.

        :param entry: LexicalEntry to generate DUDES from.
        :type entry: LexicalEntry
        :param node: Corresponding Tree node to generate DUDES for.
        :type node: Node
        :param tree: Tree which node is part of (to determine children etc.).
        :type tree: Tree[DUDESTreeData]
        :param sense: Sense of lexical entry to use. If None is given, choose entry.sense[0] if existing
        :type sense: Optional[Sense]
        :param var: Variable of the created DUDES. Might be slightly varied if more than one variable is necessary.
        :type var: str
        :param label: (Main-) label of the created DUDES.
        :type label: str
        :return: DUDES corresponding to the given LexicalEntry and node.
        :rtype: DUDES
        """
        assert entry.uri is not None
        qname: str = self.nsmanager.qname(utils.sanitize_url(entry.uri))
        # if qname.lower().startswith("ns1:"):
        #     pass

        fixed_qname: List[str] = []
        fixed_object: List[str] = []

        try:
            if sense is None:
                sense = entry.sense[0]
        except:
            logging.warning("No sense in entry!")
            pass

        try:
            # Try to get reference and just fail if it doesn't exist as expected
            refs = Lexicon.get_attr_by_name(sense, "reference")
            if len(refs) > 0 and isinstance(refs[0], List) and len(refs[0]) > 0:
                if all([isinstance(r, str) for r in refs[0]]):
                    if len(refs[0]) > 1:
                        logging.warning("Multiple string references in entry!")
                    qname = refs[0][0]
                elif all([isinstance(r, Reference) for r in refs[0]]):
                    for r in refs[0]:
                        if r.on_property is not None and r.has_value is not None:
                            fixed_qname.append(r.on_property)
                            fixed_object.append(r.has_value)
                    # assert isinstance(qname, str)
                    # assert isinstance(fixed_object, str)

            # if isinstance(entry.sense[0].reference[0], str):
            #     qname = entry.sense[0].reference[0]
            # else:
            #     logging.warning("Reference not of type string!")
        except:
            pass

        if len(fixed_object) > 0:
            return [DUDES.binary_with_fixed_objs(rel_name=fixed_qname,
                                                 obj_val=fixed_object,
                                                 var=var,
                                                 var_obj=var + "-fixed",
                                                 label=label)]

        frames = []
        if entry.syn_behavior is not None:
            frames = [f for sb in entry.syn_behavior for f in sb.type if "Frame" in f]

        subj_of_prop = set(Lexicon.get_uri_by_name(sense, "subj_of_prop"))
        obj_of_prop = set(Lexicon.get_uri_by_name(sense, "obj_of_prop"))

        property_domain = set(Lexicon.get_attr_by_name(sense, "property_domain"))
        property_range = set(Lexicon.get_attr_by_name(sense, "property_range"))
        assert len(property_range) == len(property_domain)

        match entry.part_of_speech:
            case PartOfSpeech.NOUN:
                # assert node.data.token.pos_ in ["NOUN", "PROPN"]
                if len(frames) == 0:
                    if node.data.token.pos_ == "NOUN":
                        return [DUDES.unary(rel_name=qname, var=var, label=label)]
                    elif node.data.token.pos_ == "PROPN":
                        return [DUDES.equality(text=qname, var=var, label=label)]
                    else:
                        logging.warning("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                                                                                    entry.part_of_speech))
                        return [DUDES.equality(text=qname, var=var, label=label)]
                        # raise RuntimeError("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                        #                                                                entry.part_of_speech))
                else:
                    match frames[0]:
                        case "lexinfo:NounPPFrame":
                            prep_adj = set(Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_adjunct"))

                            main_vars = [var + "-0", var + "-1"]
                            var_1_markers = None
                            var_2_markers = None

                            markers = Lexicon.get_markers(entry=entry)

                            #if (len(prep_adj.intersection(subj_of_prop)) > 0
                            #        and len(prep_adj.intersection(obj_of_prop)) > 0):
                            #    raise RuntimeError("Main variable ambiguous, both subject and object can be substituted"
                            #                       " by the prepositional adjunct!")
                            if len(prep_adj.intersection(subj_of_prop)) > 0:
                                main_vars = [var + "-1", var + "-0"]
                                if len(markers) > 0:
                                    var_1_markers = [Lexicon.marker_to_str(m) for m in markers]
                            #elif len(prep_adj.intersection(obj_of_prop)) > 0:
                            #    main_var = var + "-0"
                            #    if len(markers) > 0:
                            #        var_2_markers = [Lexicon.marker_to_str(m) for m in markers]
                            else:
                                logging.warning(
                                    "Prepositional adjunct neither subject nor object of property? Fallback to var-0, var-1 order " + entry.uri)
                                #main_var = var + "-0"

                            if len(property_domain) == 0 or len(property_range) == 0:
                                return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                     label=label, main_var=main_var,
                                                     var_1_markers=var_1_markers, var_2_markers=var_2_markers) for main_var in main_vars]
                            else:
                                return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                     label=label, main_var=main_var,
                                                     var_1_markers=var_1_markers, var_2_markers=var_2_markers)
                                        for main_var in main_vars] + [
                                    DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=main_var, domain=dom, range=ran,
                                                 domain_var=var + "-2", range_var=var + "-3",
                                                 var_1_markers=var_1_markers, var_2_markers=var_2_markers)
                                    for dom, ran in zip(property_domain, property_range) for main_var in main_vars
                                ]

                        # case "lexinfo:NounPredicateFrame":
                        #     return DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                        #                         label=label)
                        case _:
                            raise RuntimeError("Unknown frame {}".format(frames[0]))

            case PartOfSpeech.ADJECTIVE:
                if len(frames) == 0:
                    if node.data.token.pos_ != "ADJ":
                        logging.warning("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                                                                                    entry.part_of_speech))
                        # raise RuntimeError("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                        #                                                                entry.part_of_speech))

                    return [DUDES.unary(rel_name=qname, var=var, label=label)]

                else:
                    match frames[0]:
                        # "JJR": "adjective, comparative",
                        # "RBR": "adverb, comparative",
                        # "JJS": "adjective, superlative",
                        # "RBS": "adverb, superlative",
                        case "lexinfo:AdjectiveSuperlativeFrame":
                            boundto = Lexicon.get_uri_by_name(sense, "bound_to")
                            degree = Lexicon.get_attr_by_name(sense, "degree")
                            assert len(set(degree)) == 1 and len(set(boundto)) <= 1

                            pos_tags = [node.data.token.tag_] + [t.tag_ for t in node.data.token.merged_tokens]

                            # if node.data.token.tag_ not in ["JJS", "RBS"]:
                            #     print("Superlative frame for non-superlative adjective!")
                            #     return DUDES.binary(rel_name=boundto[0], var_1=var + "-0", var_2=var + "-1",
                            #                         label=label, main_var=None)
                            # raise RuntimeError("Superlative frame for non-superlative adjective!")
                            if len(set(boundto)) == 1:
                                if any([tag in ["JJS", "RBS"] for tag in pos_tags]):
                                    return [DUDES.top_bound(count=1,
                                                            var=var + "-0",
                                                            order_val=str(degree[0]),
                                                            label=label,
                                                            bound_to=boundto[0],
                                                            bound_to_var=var + "-1")]
                                elif any([tag in ["JJR", "RBR"] for tag in pos_tags]):
                                    return [
                                        DUDES.comp_bound(num_var=var + "-0", var=var + "-1", order_val=str(degree[0]),
                                                         label=label, bound_to=boundto[0], bound_to_var=var + "-2"),
                                        DUDES.countcomp_bound(num_var=var + "-0", var=var + "-1",
                                                              order_val=str(degree[0]), label=label,
                                                              bound_to=boundto[0], bound_to_var=var + "-2"),
                                        DUDES.comp_bound(num_var=var + "-0", var=var + "-1", order_val=str(degree[0]),
                                                         label=label, bound_to=boundto[0], bound_to_var=var + "-2",
                                                         bind_other=True),
                                        DUDES.countcomp_bound(num_var=var + "-0", var=var + "-1",
                                                              order_val=str(degree[0]), label=label,
                                                              bound_to=boundto[0], bound_to_var=var + "-2",
                                                              bind_other=True),
                                        DUDES.comp_bound(num_var=var + "-0", var=var + "-1", order_val=str(degree[0]),
                                                         label=label, bound_to=boundto[0], bound_to_var=var + "-2",
                                                         bound_to_var2=var + "-3"),
                                        DUDES.countcomp_bound(num_var=var + "-0", var=var + "-1",
                                                              order_val=str(degree[0]), label=label,
                                                              bound_to=boundto[0], bound_to_var=var + "-2",
                                                              bound_to_var2=var + "-3")

                                    ]
                                else:
                                    logging.warning("Neither superlative nor comparative adjective, fallback to bound_to!")
                                    return [DUDES.binary(rel_name=boundto[0], var_1=var + "-0", var_2=var + "-1",
                                                         label=label, main_var=None)]
                            else:
                                if any([tag in ["JJS", "RBS"] for tag in pos_tags]):
                                    return [DUDES.top(order_val=str(degree[0]),
                                                      count=1,
                                                      var=var,
                                                      label=label)]
                                elif any([tag in ["JJR", "RBR"] for tag in pos_tags]):
                                    return [
                                        DUDES.comp(num_var=var + "-0", order_val=str(degree[0]), var=var + "-1",
                                                   label=label),
                                        DUDES.countcomp(num_var=var + "-0", order_val=str(degree[0]), var=var + "-1",
                                                        group_var=var + "-2", label=label)
                                    ]
                                else:
                                    logging.warning("Neither superlative nor comparative adjective and no bound_to, fallback to qname!")
                                    return [DUDES.unary(rel_name=qname, var=var, label=label)]

                        # case "lexinfo:AdjectivePredicateFrame":
                        #     pass
                        # case "lexinfo:AdjectiveAttributiveFrame":
                        #     pass
                        case _:
                            raise RuntimeError("Unknown frame {}".format(frames[0]))
            case PartOfSpeech.VERB:
                # e.g. "married" in "is married to" gets recognized as adjective although the whole thing is verb-ish
                if node.data.token.pos_ != "VERB":  # or node.data.token.pos_ == "ADJ":
                    logging.warning("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                                                                                entry.part_of_speech))
                    # raise RuntimeError("Unexpected POS mismatch: {} vs. {}".format(node.data.token.pos_,
                    #                                                               entry.part_of_speech))
                if len(frames) == 0:
                    logging.warning("No frames although one was expected, fallback to default behavior!")
                    match len(list(tree.children(node.identifier))):
                        case 0:
                            raise RuntimeError("No full sentence!")
                        case 1:
                            return [DUDES.unary(rel_name=qname, var=var, label=label)]
                        case 2:
                            return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=None)]
                        case 3:
                            return [DUDES.ternary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                  var_3=var + "-2", label=label, main_var=None)]
                        case _:
                            raise RuntimeError("Too many successors for single verb!")
                elif len(frames) > 1:
                    logging.warning("Multiple frames given, choosing first one!")
                    frames = [frames[0]]

                match frames[0]:  # TODO: alternatively qname(sense.uri) instead of qname of lexical entry?
                    case "lexinfo:IntransitivePPFrame":
                        prep_adj = set(Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_adjunct"))

                        var_1_markers = None
                        var_2_markers = None

                        markers = Lexicon.get_markers(entry=entry)
                        if len(markers) > 0:
                            if (len(prep_adj.intersection(subj_of_prop)) > 0
                                    and len(prep_adj.intersection(obj_of_prop)) > 0):
                                logging.warning("Both subject and object can be prepositional adjunct for markers!")
                            elif len(prep_adj.intersection(subj_of_prop)) > 0:
                                var_1_markers = [Lexicon.marker_to_str(m) for m in markers]
                            elif len(prep_adj.intersection(obj_of_prop)) > 0:
                                var_2_markers = [Lexicon.marker_to_str(m) for m in markers]
                            else:
                                logging.warning("Prepositional adjunct neither subject nor object of property? "
                                                + entry.uri)

                        if len(property_domain) == 0 or len(property_range) == 0:
                            return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=None,
                                                 var_1_markers=var_1_markers, var_2_markers=var_2_markers)]
                        else:
                            return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=None,
                                                 var_1_markers=var_1_markers, var_2_markers=var_2_markers)] + [
                                DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                             label=label, main_var=None, domain=dom, range=ran,
                                             domain_var=var + "-2", range_var=var + "-3",
                                             var_1_markers=var_1_markers, var_2_markers=var_2_markers)
                                for dom, ran in zip(property_domain, property_range)
                            ]
                    case "lexinfo:TransitiveFrame":
                        if len(property_domain) == 0 or len(property_range) == 0:
                            return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=None)]
                        else:
                            return [DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                                 label=label, main_var=None)] + [
                                DUDES.binary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                             label=label, main_var=None, domain=dom, range=ran,
                                             domain_var=var + "-2", range_var=var + "-3")
                                for dom, ran in zip(property_domain, property_range)
                            ]
                    case "lexinfo:TransitivePPFrame":
                        if len(property_domain) > 0 and len(property_range) > 0:
                            logging.warning("Property and domain specification not implemented for TransitivePPFrame!")
                        if len(Lexicon.get_markers(entry=entry)) > 0:
                            logging.warning("Markers not implemented for TransitivePPFrame!")
                        return [DUDES.ternary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                              var_3=var + "-2", label=label, main_var=None)]
                    case "lexinfo:DitransitiveFrame":
                        if len(property_domain) > 0 and len(property_range) > 0:
                            logging.warning("Property and domain specification not implemented for DitransitiveFrame!")
                        return [DUDES.ternary(rel_name=qname, var_1=var + "-0", var_2=var + "-1",
                                              var_3=var + "-2", label=label, main_var=None)]
                    case _:
                        raise RuntimeError("Unknown frame {}".format(frames[0]))

                    # TODO: Other frames -> is there a list that is not the turtle specification?
        raise RuntimeError("Unknown error, could not create DUDES from entry!")

class RawStrategy(DUDESCreationStrategy):
    def node_to_dudes(self, node: Node, tree: Tree, next_var_id: int) -> Tuple[List[DUDES], int]:
        res = [
            RawStrategy._atomic_dudes(tree=tree, node=node, var="x" + str(next_var_id), label="l" + str(next_var_id))
        ]
        return res, next_var_id + 1

    @staticmethod
    def _atomic_dudes(tree: Tree,
                      node: Node,
                      var: str = "x",
                      label: str = "l") -> DUDES:
        name = node.data.token.text
        # if len(node.data.token.ent_kb_id_) > 0 and len(node.data.token.ent_kb_id_[0]) > 0:
        #     name = self.nsmanager.qname(node.data.token.ent_kb_id_[0])
        #     return DUDES.equality(text=name, var=var, label=label)

        def lower(n: str) -> str:  # TODO: could be obsolete now
            return n.lower() #n if len(node.data.token.ent_kb_id_) > 0 and len(node.data.token.ent_kb_id_[0]) > 0 else n.lower()

        match node.data.token.pos_:
            case "ADJ":  # TODO: lower? what if dbpedia entry?
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "ADP":
                # treat like adverb without an ontology -> walks *into* -> into describes walks further
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "ADV":
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "AUX":
                # treat like adverb without an ontology -> *has* walked -> has is part of verb, describes walk further
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "CONJ":
                pass
            case "CCONJ":
                pass
            case "SCONJ":
                pass
            case "DET":
                # could maybe be left out/merged?
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "INTJ":
                pass
            case "NOUN":
                # TODO: alternatively treat more like an adjective? fox -> noun -> fox(x) instead of x = fox?
                # return DUDES.pnoun(noun=name, var=var, label=label)
                return DUDES.unary(rel_name=lower(name), var=var, label=label)
            case "NUM":
                # TODO: alternatively use Int(var) as variables, would allow comparisons etc. in Z3
                # Caution! Also sth. like "twenty-three"! -> numerizer
                return DUDES.equality(text=lower(name), var=var, label=label)
            case "PART":
                # TODO: "not" etc. could be used to create duplex conditions
                return DUDES.unary(rel_name=lower(name), var=var, label=label)  # allowed_pos=[],
            case "PRON":  # TODO: x = he or he(x)?
                return DUDES.unary(rel_name=lower(name), var=var, label=label)  # allowed_pos=[],
                # return DUDES.noun(noun=node.data.token.text, var=var, label=label)
            case "PROPN":
                return DUDES.equality(text=name, var=var, label=label)
            case "VERB":
                match len(list(tree.children(node.identifier))):
                    case 0:
                        logging.warning("No full sentence! nevertheless fallback to unary predicate!")
                        return DUDES.unary(rel_name=lower(name), var=var, label=label)
                        #raise RuntimeError("No full sentence!")
                    case 1:
                        return DUDES.unary(rel_name=lower(name), var=var, label=label)
                    case 2:
                        return DUDES.binary(rel_name=lower(name), var_1=var + "-0", var_2=var + "-1",
                                            label=label, main_var=None)
                    # case 3:
                    #     return DUDES.ternary(rel_name=lower(name), var_1=var + "-0", var_2=var + "-1",
                    #                          var_3=var + "-2", label=label, main_var=None)
                    case _:
                        logging.warning("Too many successors for single verb! Fallback to binary predicate!")
                        return DUDES.binary(rel_name=lower(name), var_1=var + "-0", var_2=var + "-1",
                                            label=label, main_var=None)
                        #raise RuntimeError("Too many successors for single verb!")
            case _:
                logging.error("No default atomic DUDES available for POS type " + node.data.token.pos_)
                #raise RuntimeError("No default atomic DUDES available for POS type " + node.data.token.pos_)
        logging.warning("Fallback to unary predicate, no default DUDES defined for POS tag yet!")
        return DUDES.unary(rel_name=lower(name), var=var, label=label)