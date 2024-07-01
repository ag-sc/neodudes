import copy
import itertools
import logging
from abc import abstractmethod, ABC
from typing import Tuple, List, Optional, Set

from treelib import Node, Tree  # type: ignore

from dudes.dudes import DUDES, SelectionPair
from dudes.duplex_condition import Quantifier, DuplexCondition, CDUDES
from lemon.lemon import LexicalEntry
from lemon.lexicon import Lexicon


class DUDESComposeStrategy(ABC):
    @abstractmethod
    def compose_dudes(self,
                      node: Node,
                      node_dudes: CDUDES,
                      parent: Node,
                      parent_dudes: CDUDES,
                      tree: Tree,
                      dudes_composer) -> List[CDUDES]:
        pass


class DUDESComposer:
    def __init__(self, strategies: List[DUDESComposeStrategy], candidate_limit: int = 10):
        self.strategies = strategies
        self.candidate_limit = candidate_limit

    @classmethod
    def default(cls, candidate_limit: int = 10):
        return cls([
            ConjDuplexConditionComposeStrategy(),
            LemonDUDESComposeStrategy(),
            BasicDUDESComposeStrategy()
        ], candidate_limit=candidate_limit)

    def compose_dudes(self,
                      node: Node,
                      node_dudes: CDUDES,
                      parent: Node,
                      parent_dudes: CDUDES,
                      tree: Tree) -> List[CDUDES]:
        res: List[CDUDES] = []
        for strategy in self.strategies:
            if len(res) == 0:
                res = strategy.compose_dudes(node, node_dudes, parent, parent_dudes, tree, dudes_composer=self)
            else:
                break
        if len(res) > self.candidate_limit:
            logging.warning(f"More than {self.candidate_limit} candidates returned, discarding {len(res)-self.candidate_limit}!")
        return res[:self.candidate_limit]

    @staticmethod
    def compose_prepared(possible_merges: List[Tuple[CDUDES, Optional[SelectionPair], CDUDES]],
                         node: Node,
                         parent: Node) -> List[CDUDES]:
        # possible_merges = self.merge_preparer.compose_dudes(node=node, node_dudes=node_dudes, parent=parent,
        #                                                     parent_dudes=parent_dudes)
        result_dudes: List[CDUDES] = []

        other: CDUDES
        sp: Optional[SelectionPair]
        merge_dudes: CDUDES
        for merge_dudes, sp, other in possible_merges:
            merge_dudes = copy.deepcopy(merge_dudes)
            sp = copy.deepcopy(sp)
            other = copy.deepcopy(other)
            # node_dudes = copy.deepcopy(node.data.dudes_candidates[node_dudes_idx])
            if sp is not None and sp.markers is not None and not (
                    node.data.lmarker in sp.lmarkers
                    or parent.data.lmarker in sp.lmarkers
            ):
                logging.warning("Marker mismatch! Fallback to unifying merge.")

                # node_dudes2 = copy.deepcopy(node_dudes)

                merge_dudes2 = copy.deepcopy(merge_dudes)
                other2 = copy.deepcopy(other)

                res2 = merge_dudes2.merge(other=other2, sp=None)
                result_dudes.append(res2)

            res = merge_dudes.merge(other=other, sp=sp)
            result_dudes.append(res)
        return result_dudes


class ConjDuplexConditionComposeStrategy(DUDESComposeStrategy):

    @staticmethod
    def conj_to_quantifier(conj: str):
        if conj.lower() == "and":
            return Quantifier.AND
        elif conj.lower() == "or":
            return Quantifier.OR
        else:
            raise ValueError(f"Unsupported conjunction: {conj}")

    def compose_dudes(self,
                      node: Node,
                      node_dudes: CDUDES,
                      parent: Node,
                      parent_dudes: CDUDES,
                      tree: Tree,
                      dudes_composer: DUDESComposer) -> List[CDUDES]:
        if node.data.token.dep_ in ["conj"]:
            cconj_children = [
                c for c in tree.children(parent.identifier)
                if c.data.token.pos_ in ["CCONJ"] and c.data.token.text.lower() in ["and", "or"]
            ]
            merged_cconj = [
                c for c in parent.data.token.merged_tokens
                if c.pos_ in ["CCONJ"] and c.text.lower() in ["and", "or"]
            ]

            # either conj is child and is/will be merged into parent
            if len(cconj_children) + len(merged_cconj) > 0 or isinstance(node_dudes, DuplexCondition):  # otherwise the duplex condition would have an empty quantifier
                # TODO: what happens with selection pairs?
                # TODO: child merges always go to old parent, when duplex condition is child parent is copied and applied to both sides?
                quantifier_cands = {self.conj_to_quantifier(c.data.token.text.lower()) for c in (cconj_children + merged_cconj)}
                if len(quantifier_cands) == 0 and isinstance(node_dudes, DuplexCondition) and node_dudes.quantifier is not None:
                    quantifier_cands = {
                        node_dudes.quantifier
                    }
                assert len(quantifier_cands) > 0

                res: List[CDUDES] = []

                for qc in quantifier_cands:
                    node_dudes_copy = copy.deepcopy(node_dudes)
                    parent_dudes_copy = copy.deepcopy(parent_dudes)
                    dc_dudes: DuplexCondition = DuplexCondition(quantifier=qc, restrictor=parent_dudes_copy, scope=node_dudes_copy)
                    dc_dudes.variable = dc_dudes.distinctify_and_unify_main_vars()
                    dc_dudes.refresh_pred_var_dict()
                    res.append(dc_dudes)

                return res
            else:
                return []
        elif node.data.token.pos_ in ["CCONJ"]:
            return [parent_dudes]
        elif isinstance(node_dudes, DuplexCondition) and isinstance(parent_dudes, DUDES):
            node_dudes2: DuplexCondition
            parent_dudes_left: DUDES
            parent_dudes_right: DUDES

            if isinstance(node_dudes, DuplexCondition) and isinstance(parent_dudes, DUDES):
                node_dudes2 = copy.deepcopy(node_dudes)
                parent_dudes_left = copy.deepcopy(parent_dudes)
                parent_dudes_right = copy.deepcopy(parent_dudes)
            elif isinstance(node_dudes, DUDES) and isinstance(parent_dudes, DuplexCondition):
                node_dudes2 = copy.deepcopy(parent_dudes)
                parent_dudes_left = copy.deepcopy(node_dudes)
                parent_dudes_right = copy.deepcopy(node_dudes)
            else:
                raise RuntimeError("This should not happen!")
            restr_dudes_cands = []
            if node_dudes2.restrictor is not None:
                restr_dudes_cands = dudes_composer.compose_dudes(node, node_dudes2.restrictor, parent,
                                                                 parent_dudes_left, tree)

            scope_dudes_cands = []
            if node_dudes2.scope is not None:
                scope_dudes_cands = dudes_composer.compose_dudes(node, node_dudes2.scope, parent, parent_dudes_right,
                                                                 tree)

            res = []
            for rd, sd in itertools.product(restr_dudes_cands, scope_dudes_cands):
                dc_dudes = DuplexCondition(quantifier=node_dudes2.quantifier, variable=node_dudes2.variable,
                                           restrictor=rd, scope=sd)
                dc_dudes.refresh_pred_var_dict()
                res.append(dc_dudes)
            return res
        elif isinstance(node_dudes, DUDES) and isinstance(parent_dudes, DuplexCondition):
            parent_dudes2 = copy.deepcopy(parent_dudes)
            restr_dudes_cands = []
            if parent_dudes2.restrictor is not None:
                node_dudes_copy = copy.deepcopy(node_dudes)
                restr_dudes_cands = dudes_composer.compose_dudes(node, parent_dudes2.restrictor, parent,
                                                                 node_dudes_copy, tree)
            res = []
            for rc in restr_dudes_cands:
                parent_dudes_cand = copy.deepcopy(parent_dudes)
                parent_dudes_cand.restrictor = rc
                res.append(parent_dudes_cand)
            return res
        else:
            return []
        # TODO: deal with conj dependency tag creating duplex condition ANDing/ORing parent and child DUDES
        # TODO: deal with CONJ POS tag for conjunctions and/or setting operator in parent duplex conditions
        # TODO: deal with child duplex with operator into parent duplex without operator or CONJ child, copying the child's operator


class LemonDUDESComposeStrategy(DUDESComposeStrategy):
    def compose_dudes(self,
                      node: Node,
                      node_dudes: CDUDES,
                      parent: Node,
                      parent_dudes: CDUDES,
                      tree: Tree,
                      dudes_composer: DUDESComposer) -> List[CDUDES]:
        possible_merges = self.compose_dudes_prepare(node, node_dudes, parent, parent_dudes)
        return DUDESComposer.compose_prepared(possible_merges, node, parent)

    @staticmethod
    def compose_dudes_prepare(node: Node,
                              node_dudes: DUDES,
                              parent: Node,
                              parent_dudes: DUDES) -> List[Tuple[DUDES, Optional[SelectionPair], DUDES]]:
        """
        Apply some heuristics to find the best way to merge two nodes. In many edge cases falls back to BasicOntology
        behavior. Applied heuristics:

        * In any case, if the attempt fails, the BasicOntology fallback is called and might make a different decision.
        * When the child is, e.g., a modifier of some kind, merge parent into child. More precisely, when the dependency
          tag is one of these: ["advmod", "amod", "attr", "infmod", "meta", "neg", "nmod", "nn", "npadvmod", "nounmod",
          "npmod", "num", "nummod", "poss", "possessive", "prep", "quantmod", "hmod", "aux", "auxpass", "det"]
        * Otherwise, try to merge child into parent.
            * Cases no and one selection pair are trivial, just merge.
            * For two selection pairs differentiate whether the current node to merge is an object or a subject
                * Subjects are preferred for the first variable/position in the functions of the corresponding DUDES
                * Objects are preferred for the second variable/position
            * For more than two variables or non-matching numbers of variables etc. fallback to BasicOntology

        :param node_dudes: DUDES of node to use for merge
        :type node_dudes: DUDES
        :param parent_dudes: DUDES of parent to use for merge
        :type parent_dudes: DUDES
        :param node: Child node to consider for merge.
        :type node: Node
        :param parent: Parent node to consider for merge.
        :type parent: Node
        :return: DUDES to merge into, SelectionPair to use for merge and DUDES to 'consume'/merge.
        :rtype: Tuple[DUDES, Optional[SelectionPair], DUDES]
        """
        res: List[Tuple[DUDES, Optional[SelectionPair], DUDES]] = []

        merge_dudes: DUDES
        sp: Optional[SelectionPair]
        other: DUDES
        node_dep = node.data.token.dep_
        parent_dep = parent.data.token.dep_

        parent_into_child = [
            "acl", "advcl", "advmod", "amod", "appos", "attr", "infmod", "meta", "neg", "nmod", "nn", "npadvmod",
            "nounmod", "npmod", "num", "number", "nummod", "partmod", "poss", "possessive", "prep", "quantmod", "rcmod",
            "relcl",
            "hmod",  # ??
            # if determiner or auxiliaries are not filtered out,
            # we also need to replace the parent into the child dudes
            "aux", "auxpass", "det",
        ]  # TODO: clauses?? "advcl"
        try:
            if node_dudes.main_variable is None and parent_dudes.main_variable is None:
                # merge_dudes = parent_dudes
                # sp = None
                # other = node_dudes
                res.append((parent_dudes, None, node_dudes))
                logging.warning("Main variable of both DUDES is None!")
                # return merge_dudes, sp, other
            elif any([node.data.lmarker in tsp.lmarkers for tsp in parent_dudes.selection_pairs]):
                cand_sps = [tsp for tsp in parent_dudes.selection_pairs if node.data.lmarker in tsp.lmarkers]
                # if len(cand_sps) > 0:
                for cand_sp in cand_sps:
                    # sp = cand_sps[0]
                    # merge_dudes = parent_dudes
                    # other = node_dudes
                    res.append((parent_dudes, cand_sp, node_dudes))
                # else:
                #     raise RuntimeError("Matching selection pair vanished? Should never occur!")
            elif any([parent.data.lmarker in tsp.lmarkers for tsp in node_dudes.selection_pairs]):
                cand_sps = [tsp for tsp in node_dudes.selection_pairs if parent.data.lmarker in tsp.lmarkers]
                for cand_sp in cand_sps:
                    # sp = cand_sps[0]
                    # merge_dudes = node_dudes
                    # other = parent_dudes
                    res.append((node_dudes, cand_sp, parent_dudes))
                # else:
                #     raise RuntimeError("Matching selection pair vanished? Should never occur!")
            elif ((node_dep in parent_into_child or node_dudes.main_variable is None)
                  and parent_dudes.main_variable is not None):  # merge parent into child dudes
                if len(node_dudes.selection_pairs) > 0:
                    for cand_sp in node_dudes.selection_pairs:
                        # merge_dudes = node_dudes
                        # sp = node_dudes.selection_pairs[0]
                        # other = parent_dudes
                        if len(node_dudes.selection_pairs) == 1:
                            node_dudes.initial_pred_var_dict = parent_dudes.initial_pred_var_dict
                        else:
                            logging.warning(
                                "More than one selection pair although just one was expected for modifiers!")
                        res.append((node_dudes, cand_sp, parent_dudes))
                else:
                    logging.warning(
                        "No selection pair although dependency tag is modifier-like, fallback to basic default behavior")
                    # res.extend(super().prepare_merge(node, node_dudes, parent, parent_dudes))

            else:  # merge child into parent dudes
                # merge_dudes = parent_dudes
                # other = node_dudes
                if len(parent_dudes.selection_pairs) == 0:
                    # sp = None
                    logging.warning("Merging without selection pair!")
                    res.append((parent_dudes, None, node_dudes))
                elif len(parent_dudes.selection_pairs) == 1:
                    # sp = parent_dudes.selection_pairs[0]
                    res.append((parent_dudes, parent_dudes.selection_pairs[0], node_dudes))
                elif len(parent_dudes.selection_pairs) == 2:
                    # TODO: Lexicon based matching subjOfProp objOfProp with subject, prepositionalAdjunct etc.
                    # TODO: new default behavior subjects preferred first parameter,
                    # objects preferred second fallback first, pobj etc. third fallback second fallback first

                    pvd = {k: v for k, v in parent_dudes.initial_pred_var_dict.items() if not k.startswith("local:")}
                    if len(pvd.values()) > 1 or len(pvd.values()) == 0:
                        raise RuntimeWarning(
                            "Warning: Initial DUDES property could not be identified, fallback to basic default behavior")
                    else:
                        candidate_var_orders = list(pvd.values())[0]
                        if len(candidate_var_orders) > 1 or len(candidate_var_orders) == 0:
                            raise RuntimeWarning(
                                "Warning: No candidate variable order, fallback to basic default behavior")
                        else:
                            var_order = candidate_var_orders[0]
                            match len(var_order):
                                case 0:
                                    raise RuntimeWarning(
                                        "Warning: Variable order empty, fallback to basic default behavior")
                                case 1:
                                    raise RuntimeWarning(
                                        "Warning: Just a single variable but multiple selection pairs, fallback to "
                                        "basic default behavior")
                                case 2:
                                    sp_cands = []
                                    try:
                                        # raise RuntimeError("")#TODO:
                                        if len(parent.data.lex_candidates) == 0:
                                            raise RuntimeError("No lexical entry candidates!")

                                        entry: LexicalEntry = parent.data.lex_candidates[0]
                                        subj_of_prop = set(Lexicon.get_uri_by_name(entry.sense, "subj_of_prop"))
                                        obj_of_prop = set(Lexicon.get_uri_by_name(entry.sense, "obj_of_prop"))

                                        if len(subj_of_prop) == 0 and len(obj_of_prop) == 0:
                                            raise RuntimeError("No subj_of_prop and obj_of_prop specified in entry!")

                                        syn_uris: Set = set()  # TODO: also implement this disambiguation for modifiers to match variables correctly?
                                        if node.data.token.dep_ in ["csubj", "csubjpass", "nsubj", "nsubjpass"]:
                                            syn_uris = set(Lexicon.get_uri_by_name(entry.syn_behavior, "subject"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "copulative_subject"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "copulative_arg"))
                                        elif node.data.token.dep_ in ["dobj"]:
                                            syn_uris = set(Lexicon.get_uri_by_name(entry.syn_behavior, "direct_object"))
                                        elif node.data.token.dep_ in ["pobj"]:
                                            syn_uris = set(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_object"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_adjunct"))
                                        elif node.data.token.dep_ in ["iobj",
                                                                      "obj"]:  # TODO: isn't there a predicate for indirect objects?
                                            syn_uris = set(Lexicon.get_uri_by_name(entry.syn_behavior, "direct_object"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_object"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "prepositional_adjunct"))
                                        else:
                                            syn_uris = set(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "attributive_arg"))
                                            syn_uris.update(
                                                Lexicon.get_uri_by_name(entry.syn_behavior, "possessive_adjunct"))

                                        if len(syn_uris) == 0:
                                            raise RuntimeError("No syntactic anchors in syn_behavior!")
                                        else:
                                            subj_inter = subj_of_prop.intersection(syn_uris)
                                            obj_inter = obj_of_prop.intersection(syn_uris)

                                            if len(subj_inter) == len(obj_inter):  # also covers case that both is 0!
                                                raise RuntimeError("Equally many subject and object candidates!")
                                            elif len(subj_inter) > len(obj_inter):
                                                sp_cands = [sp for sp in parent_dudes.selection_pairs if
                                                            sp.variable == var_order[0]]
                                                # still add other candidates with lower prio!
                                                sp_cands += [sp for sp in parent_dudes.selection_pairs if
                                                             sp.variable != var_order[0]]
                                            else:
                                                sp_cands = [sp for sp in parent_dudes.selection_pairs if
                                                            sp.variable == var_order[1]]
                                                # still add other candidates with lower prio!
                                                sp_cands += [sp for sp in parent_dudes.selection_pairs if
                                                             sp.variable != var_order[1]]

                                    except Exception as e:  # Try to use subjOfProp, objOfProp and fallback to this
                                        logging.warning(f"Usage of subjOfProp/objOfProp failed: {e}")
                                        sp_cands = parent_dudes.selection_pairs
                                        # # if node is subject, then first var's SP, otherwise second
                                        # if node.data.token.dep_ in ["csubj", "csubjpass", "nsubj", "nsubjpass"]:
                                        #     sp_cands = [sp for sp in parent_dudes.selection_pairs if
                                        #                 sp.variable == var_order[0]]
                                        # # for objects use second variable's SP
                                        # elif node.data.token.dep_ in ["dobj", "iobj", "obj", "pobj"]:
                                        #     sp_cands = [sp for sp in parent_dudes.selection_pairs if
                                        #                 sp.variable == var_order[1]]
                                        # else:  # otherwise default to first SP in list
                                        #     if len(parent_dudes.selection_pairs) > 0:
                                        #         sp_cands = [parent_dudes.selection_pairs[0]]
                                        #     else:
                                        #         sp_cands = []

                                    for cand_sp in sp_cands:
                                        res.append((parent_dudes, cand_sp, node_dudes))
                                    # if len(sp_cands) > 1 or len(sp_cands) == 0:
                                    #     raise RuntimeWarning(
                                    #         "Warning: Multiple or no selection pairs possible, fallback to basic "
                                    #         "default behavior")
                                    # else:
                                    #     sp = sp_cands[0]
                                case _:
                                    raise RuntimeWarning(
                                        "Warning: More than two variables not supported yet, fallback to basic default behavior")
                else:
                    for cand_sp in parent_dudes.selection_pairs:
                        res.append((parent_dudes, cand_sp, node_dudes))
                    # raise RuntimeWarning(
                    #     "Warning: More than two selection pairs not supported yet, fallback to basic default behavior")


        except RuntimeWarning as e:
            logging.warning(e)
            # res.extend(super().prepare_merge(node, node_dudes, parent, parent_dudes))
        except Exception as e:
            raise e

        return res  # res[:1]


class BasicDUDESComposeStrategy(DUDESComposeStrategy):
    def compose_dudes(self,
                      node: Node,
                      node_dudes: CDUDES,
                      parent: Node,
                      parent_dudes: CDUDES,
                      tree: Tree,
                      dudes_composer: DUDESComposer) -> List[CDUDES]:
        possible_merges = self.compose_dudes_prepare(node, node_dudes, parent, parent_dudes)
        return DUDESComposer.compose_prepared(possible_merges, node, parent)

    @staticmethod
    def compose_dudes_prepare(node: Node,
                              node_dudes: DUDES,
                              parent: Node,
                              parent_dudes: DUDES) -> List[Tuple[DUDES, Optional[SelectionPair], DUDES]]:
        """
        Decides which node to merge into which using very simple heuristics: When both nodes do not have selection pairs
        (should not happen), just unify them. When only one node has SelectionPairs, then use the first SelectionPair
        of that node. Otherwise, merge parent into child using first SelectionPair of child (as children should run out
        of SelectionPairs earlier, but this can also be not the case, e.g., when the child has some variables which are
        not determined by the sentences).

        :param node_dudes: DUDES of node to use for merge
        :type node_dudes: DUDES
        :param parent_dudes: DUDES of parent to use for merge
        :type parent_dudes: DUDES
        :param node: Child node to consider for merge.
        :type node: Node
        :param parent: Parent node to consider for merge.
        :type parent: Node
        :return: DUDES to merge into, SelectionPair to use for merge and DUDES to 'consume'/merge.
        :rtype: Tuple[DUDES, Optional[SelectionPair], DUDES]
        """
        merge_dudes: DUDES
        sp: Optional[SelectionPair]
        other: DUDES

        # TODO: Ugly, refactor!
        if node_dudes.main_variable is None and parent_dudes.main_variable is None:
            logging.warning("No merge possible, both DUDES have empty main variable!")
            return [(parent_dudes, None, node_dudes)]
        elif node_dudes.main_variable is None and parent_dudes.main_variable is not None:
            if len(node_dudes.selection_pairs) > 1:
                logging.warning("More than one selection pair possible!")
            return [
                (node_dudes, node_dudes.selection_pairs[i], parent_dudes)
                for i in range(len(node_dudes.selection_pairs))
            ] if len(node_dudes.selection_pairs) > 0 else [(node_dudes, None, parent_dudes)]
        elif node_dudes.main_variable is not None and parent_dudes.main_variable is None:
            if len(parent_dudes.selection_pairs) > 1:
                logging.warning("More than one selection pair possible!")
            return [
                (parent_dudes, parent_dudes.selection_pairs[i], node_dudes)
                for i in range(len(parent_dudes.selection_pairs))
            ] if len(parent_dudes.selection_pairs) > 0 else [(parent_dudes, None, node_dudes)]

        if len(node_dudes.selection_pairs) == 0 and len(parent_dudes.selection_pairs) == 0:
            return [(parent_dudes, None, node_dudes)]
        else:
            return [
                (node_dudes, node_dudes.selection_pairs[i], parent_dudes)
                for i in range(len(node_dudes.selection_pairs))
            ] + [
                (parent_dudes, parent_dudes.selection_pairs[i], node_dudes)
                for i in range(len(parent_dudes.selection_pairs))
            ]
