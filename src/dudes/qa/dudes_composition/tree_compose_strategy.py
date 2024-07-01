import copy
import itertools
import logging
import os
import sys
from abc import ABC, abstractmethod
from queue import Queue
from typing import List, Dict, Optional, Generator, Iterable, Set, Tuple

import xxhash
import z3  # type: ignore
from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from sqlitedict import SqliteDict
from treelib import Tree, Node  # type: ignore

import graphviz as gv  # type: ignore

from dudes import utils
from dudes.dudes import DUDES, SelectionPair
from dudes.dot import DotFile
from dudes.dudes_tree import DUDESTreeCombData
from dudes.qa.dudes_composition.dudes_composer import DUDESComposer
import lemon


class TreeComposeStrategy(ABC):
    @abstractmethod
    def compose_tree(self, tree: Tree) -> Iterable[DUDES]:
        pass


class BottomUpTreeComposeStrategy(TreeComposeStrategy):

    def __init__(self, dudes_composer: DUDESComposer):
        self.dudes_composer = dudes_composer

    @staticmethod
    def _tree_node_order(tree: Tree):
        vars: Dict[int, z3.ExprRef] = dict()
        vars_to_id: Dict[z3.ExprRef, int] = dict()

        if len(tree.nodes.keys()) < 2:
            return []

        for id in tree.nodes.keys():
            v = z3.Int("v" + str(id))
            vars[id] = v
            vars_to_id[v] = id

        conds = set()
        for node in tree.nodes.values():
            for child in tree.children(node.identifier):
                conds.add(
                    vars[node.identifier] > vars[child.identifier]
                )

            with_lex = [child.identifier for child in tree.children(node.identifier)
                        if len(child.data.lex_candidates) > 0]
            without_lex = [child.identifier for child in tree.children(node.identifier)
                           if len(child.data.lex_candidates) == 0]

            for wl, wol in itertools.product(with_lex, without_lex):
                conds.add(
                    vars[wol] > vars[wl]
                )

        logging.debug(conds)

        solv = z3.Solver()
        for form in conds:
            solv.add(form)
        if solv.check() == z3.sat:
            model = solv.model()
            var_order = sorted([(model[v].as_long(), -vars_to_id[v]) for v in vars.values()])

            return [-v for n, v in var_order]
        else:
            raise RuntimeError("Formula could not be solved!")

    def compose_tree(self, tree: Tree) -> Iterable[DUDES]:
        """
        BFS-traverse the tree bottom-up and merge children and parent nodes, considering all combinations of DUDES
        candidates on the way. That means, child nodes are merged and removed such that only one node remains in the
        end. That remaining node contains the different possible DUDES which represents the meaning of the whole
        sentence.

        :param tree: Tree to merge DUDES in
        :type tree: Tree[DUDESTreeCombData]
        :param debug: Print debug infos such as intermediate DUDES on the way.
        :type debug: bool
        """
        q: Queue[int] = Queue()
        # leaves = list(reversed(list(tree.leaves())))
        # if debug:
        #    print("Leaves", leaves)
        # for cl in leaves:
        node_order = self._tree_node_order(tree)
        for id in node_order:
            q.put(id)

        # TODO: completely pre-calculate queue content and ensure nodes with lex_candidates are merged before ones
        # without, or in the long term for nodes in which a "delay merge" flag is set/no requirements are made
        # -> SMT solver for requirements?

        while not q.empty():
            nodeid: int = q.get()

            if tree.get_node(nodeid) is None:
                continue

            node: Node = tree[nodeid]
            if node.is_leaf() and not node.is_root():
                parent = tree.parent(node.identifier)

                result_dudes = []

                for node_dudes_idx, parent_dudes_idx in itertools.product(
                        range(len(node.data.dudes_candidates)),
                        range(len(parent.data.dudes_candidates))
                ):
                    node_dudes = copy.deepcopy(node.data.dudes_candidates[node_dudes_idx])
                    parent_dudes = copy.deepcopy(parent.data.dudes_candidates[parent_dudes_idx])

                    curr_res_dudes = self.dudes_composer.compose_dudes(node=node, node_dudes=node_dudes, parent=parent,
                                                                       parent_dudes=parent_dudes, tree=tree)

                    result_dudes.extend(curr_res_dudes)

                    # other: DUDES
                    # sp: Optional[SelectionPair]
                    # merge_dudes: DUDES
                    # for merge_dudes, sp, other in possible_merges:
                    #     merge_dudes = copy.deepcopy(merge_dudes)
                    #     sp = copy.deepcopy(sp)
                    #     other = copy.deepcopy(other)
                    #     # node_dudes = copy.deepcopy(node.data.dudes_candidates[node_dudes_idx])
                    #     if sp is not None and sp.markers is not None and not (
                    #             node.data.lmarker in sp.lmarkers
                    #             or parent.data.lmarker in sp.lmarkers
                    #     ):
                    #         logging.warning("Marker mismatch! Fallback to unifying merge.")
                    #
                    #         # node_dudes2 = copy.deepcopy(node_dudes)
                    #
                    #         merge_dudes2 = copy.deepcopy(merge_dudes)
                    #         other2 = copy.deepcopy(other)
                    #
                    #         res2 = merge_dudes2.merge(other=other2, sp=None)
                    #         result_dudes.append(res2)
                    #
                    #     res = merge_dudes.merge(other=other, sp=sp)
                    #     result_dudes.append(res)

                parent.data.dudes_candidates = result_dudes

                # q.put(parent.identifier)
                tree.remove_node(node.identifier)
            elif not node.is_root():
                raise RuntimeError("Node is no leaf!")

        final_nodes = list(tree.nodes.values())
        assert len(final_nodes) == 1
        return final_nodes[0].data.dudes_candidates

class YieldTreeComposeStrategy(BottomUpTreeComposeStrategy):
    def __init__(self,
                 dudes_composer: DUDESComposer,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 namespace_manager: Optional[NamespaceManager] = None,
                 entity_predicates_path: Optional[str] = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "entity_predicates.sqlite")):
        super().__init__(dudes_composer=dudes_composer)
        self.nsmanager = utils.create_namespace_manager(namespaces, namespace_manager)
        # outer_stack=False for efficiency
        self.entity_predicates = SqliteDict(entity_predicates_path, autocommit=True, outer_stack=False)


    def check_entity_predicate_compatibility(self, entities: Set[str], predicates: Set[str]) -> bool:
        if len(entities) == 0 or len(predicates) == 0:
            return False

        predicates_long: Set[str] = set()
        for pred in predicates:
            try:
                pred_long = str(self.nsmanager.expand_curie(pred))
                predicates_long.add(pred_long)
            except Exception as e:
                logging.debug(f"Error in entity-predicate check: {e} {pred}")
        if len(predicates_long) == 0:
            predicates_long = predicates
            logging.debug(f"Could not expand predicates: {predicates}, fallback to given list!")

        if len(predicates_long) > 1 and "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in predicates_long: #TODO: useful?
            predicates_long.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

        entity_found = False
        entity_likely_valid = False
        for entity in entities:
            try:
                entity_long = str(self.nsmanager.expand_curie(entity))
                entity_likely_valid = True
                if entity_long in self.entity_predicates:
                    entity_found = True
                    supported_preds: Set[str] = self.entity_predicates[entity_long]
                    if not predicates_long.isdisjoint(supported_preds):
                        return True
            except Exception as e:
                logging.debug(f"Error expanding entity: {e} {entity}")
                if "http://" in entity or "https://" in entity:
                    entity_likely_valid = True
                if entity in self.entity_predicates:
                    entity_found = True
                    supported_preds = self.entity_predicates[entity]
                    if not predicates_long.isdisjoint(supported_preds):
                        return True
        # if no entity is found in DB but there are some, this might be due to empty/incomplete DB, continue
        return (not entity_found) and entity_likely_valid

    def compose_tree(self, tree: Tree) -> Generator[DUDES, None, None]:
        node_order = self._tree_node_order(tree)
        if len(node_order) == 0:
            return

        candidates: List[List[Optional[List[DUDES]]]] = [
            [[dds] for dds in tree.get_node(i).data.dudes_candidates] if tree.get_node(i) is not None else [None]
            for i in range(max(node_order)+1)
        ]

        generated_dudes: Set[int] = set()
        xh = xxhash.xxh32()

        def hash_dudes(dudes: DUDES) -> int:
            return utils.hash_str(str(dudes), xh)

        for cand_tree_vals_tuple in utils.diagonal_generator(candidates): #itertools.product(*candidates):
            cand_tree_vals = list(cand_tree_vals_tuple)
            all_dudes_list: List[DUDES] = sum([l for l in cand_tree_vals if l is not None], [])
            cand_entities = set().union(*[d.entities for d in all_dudes_list])
            cand_predicates = set().union(*[d.predicates for d in all_dudes_list])
            if not self.check_entity_predicate_compatibility(cand_entities, cand_predicates):
                logging.debug(f"Skipping dudes due to entity-predicate mismatch: {cand_entities} {cand_predicates}")
                continue

            q: Queue[int] = Queue()
            for id in node_order:
                q.put(id)

            while not q.empty():
                nodeid: int = q.get()

                if tree.get_node(nodeid) is None or cand_tree_vals[nodeid] is None:
                    continue

                node: Node = tree[nodeid]
                if not node.is_root():
                    parent = tree.parent(node.identifier)
                    composed_dudes = []
                    for node_dudes_orig, parent_dudes_orig in itertools.product(cand_tree_vals[nodeid], cand_tree_vals[parent.identifier]):
                        node_dudes = copy.deepcopy(node_dudes_orig)
                        parent_dudes = copy.deepcopy(parent_dudes_orig)

                        try:
                            curr_res_dudes = self.dudes_composer.compose_dudes(node=node, node_dudes=node_dudes, parent=parent,
                                                                               parent_dudes=parent_dudes, tree=tree)
                            composed_dudes.extend(curr_res_dudes)
                        except Exception as e:
                            logging.error(f"Error in composition: {e}")
                            continue
                    cand_tree_vals[parent.identifier] = composed_dudes
                    cand_tree_vals[nodeid] = None
            yield from (dds for dds in cand_tree_vals[tree.root] if hash_dudes(dds) not in generated_dudes)
            generated_dudes.update([hash_dudes(dds) for dds in cand_tree_vals[tree.root]])

# class BottomUpComposeDebugStrategy(BottomUpTreeComposeStrategy):
#     def __init__(self, dudes_composer: DUDESComposer):
#         super().__init__(dudes_composer=dudes_composer)
#         self.debug = True
#
#     @staticmethod
#     def _create_single_node_tree(node: Node, dudes: DUDES) -> Tree:
#         tree: Tree = Tree()
#         data: DUDESTreeCombData = copy.deepcopy(node.data)
#         data.dudes_candidates = [copy.deepcopy(dudes)]  # , dudes=dudes))
#         node = tree.create_node(tag=node.tag,
#                                 identifier=node.identifier,
#                                 data=data)
#         return tree
#
#     @staticmethod
#     def _node_to_label(node: Node):
#         return gv.nohtml(
#             node.data.token.text_ + " " + node.data.token.pos_ + " " + node.data.token.dep_ + "\\l" +
#             str(node.data.dudes_candidates[0]).replace("\n", "\\l")
#         )
#
#     @staticmethod
#     def _tree_to_dot(tree: Tree,
#                      graph: Optional[gv.Digraph] = None,
#                      node_name_postfix: str = "",
#                      merged: Optional[Node] = None,
#                      merged_into: Optional[Node] = None,
#                      selection_pair: Optional[SelectionPair] = None):
#         def node_name(nid):
#             return "v" + str(nid) + node_name_postfix
#
#         g = gv.Digraph()
#         if graph is not None:
#             g = graph
#         q: Queue = Queue()
#         root_node: Node = tree.get_node(tree.root)
#         g.node(name=node_name(tree.root),
#                label=BottomUpComposeDebugStrategy._node_to_label(root_node),
#                shape="box",
#                color="red" if root_node in [merged, merged_into] else "black")
#         q.put(tree.root)
#
#         while not q.empty():
#             nid = q.get()
#             for child in tree.children(nid):
#                 g.node(name=node_name(child.identifier),
#                        label=BottomUpComposeDebugStrategy._node_to_label(child),
#                        shape="box",
#                        color="red" if child in [merged, merged_into] else "black")
#                 g.edge(node_name(nid), node_name(child.identifier))
#                 q.put(child.identifier)
#
#         if merged is not None and merged_into is not None:
#             g.edge(node_name(merged.identifier),
#                    node_name(merged_into.identifier),
#                    label="Merge" + (":\\l" + str(selection_pair) if selection_pair is not None else ""),
#                    color="red")
#
#         # DotFile.runXDot(str(g))
#         return g
#
#     def compose_tree(self, tree: Tree) -> List[DUDES]:
#         """
#         BFS-traverse the tree bottom-up and merge children and parent nodes, considering all combinations of DUDES
#         candidates on the way. That means, child nodes are merged and removed such that only one node remains in the
#         end. That remaining node contains the different possible DUDES which represents the meaning of the whole
#         sentence.
#
#         :param tree: Tree to merge DUDES in
#         :type tree: Tree[DUDESTreeCombData]
#         :param debug: Print debug infos such as intermediate DUDES on the way.
#         :type debug: bool
#         """
#
#         for tnode in tree.nodes.values():
#             tnode.data.tree_histories = [
#                 [self._create_single_node_tree(node=tnode, dudes=dudes)] for dudes in tnode.data.dudes_candidates
#             ] if tnode.is_leaf() else []
#
#         q: Queue[int] = Queue()
#         # leaves = list(reversed(list(tree.leaves())))
#         # if debug:
#         #    print("Leaves", leaves)
#         # for cl in leaves:
#         node_order = self._tree_node_order(tree)
#         if self.debug:
#             logging.debug(f"Node order {node_order}")
#         for id in node_order:
#             q.put(id)
#
#         # TODO: completely pre-calculate queue content and ensure nodes with lex_candidates are merged before ones
#         # without, or in the long term for nodes in which a "delay merge" flag is set/no requirements are made
#         # -> SMT solver for requirements?
#
#         while not q.empty():
#             nodeid: int = q.get()
#
#             if tree.get_node(nodeid) is None:
#                 continue
#
#             node: Node = tree[nodeid]
#             if node.is_leaf() and not node.is_root():
#                 parent = tree.parent(node.identifier)
#
#                 result_dudes = []
#                 result_histories = []
#
#                 if self.debug:
#                     logging.debug(f"### Proceeding {node.tag}")
#
#                 for node_dudes_idx, parent_dudes_idx in itertools.product(
#                         range(len(node.data.dudes_candidates)),
#                         range(len(parent.data.dudes_candidates))
#                 ):
#                     node_dudes = copy.deepcopy(node.data.dudes_candidates[node_dudes_idx])
#                     parent_dudes = copy.deepcopy(parent.data.dudes_candidates[parent_dudes_idx])
#
#                     possible_merges = self.merge_preparer.compose_dudes(node=node, node_dudes=node_dudes, parent=parent,
#                                                                         parent_dudes=parent_dudes, tree=tree)
#
#                     other: DUDES
#                     sp: Optional[SelectionPair]
#                     merge_dudes: DUDES
#                     for merge_dudes, sp, other in possible_merges:
#                         merge_dudes = copy.deepcopy(merge_dudes)
#                         sp = copy.deepcopy(sp)
#                         other = copy.deepcopy(other)
#                         # node_dudes = copy.deepcopy(node.data.dudes_candidates[node_dudes_idx])
#                         parent_dudes = copy.deepcopy(
#                             parent.data.dudes_candidates[parent_dudes_idx]) if self.debug else None
#                         if sp is not None and sp.markers is not None and not (
#                                 node.data.lmarker in sp.lmarkers
#                                 or parent.data.lmarker in sp.lmarkers
#                         ):
#                             logging.warning("Marker mismatch! Fallback to unifying merge.")
#
#                             # node_dudes2 = copy.deepcopy(node_dudes)
#
#                             merge_dudes2 = copy.deepcopy(merge_dudes)
#                             other2 = copy.deepcopy(other)
#
#                             new_history2 = []
#                             if self.debug:
#                                 parent_dudes2 = copy.deepcopy(parent.data.dudes_candidates[parent_dudes_idx])
#                                 sp2 = copy.deepcopy(sp)
#                                 logging.debug("Merging: ")
#                                 logging.debug(str(other2))
#                                 logging.debug("into: ")
#                                 logging.debug(str(merge_dudes2))
#                                 logging.debug(f"with: {sp2}")
#                                 logging.debug("Tree:")
#                                 tree.show(line_type="ascii-em", data_property="desc")
#
#                                 for htree2 in node.data.tree_histories[node_dudes_idx]:
#                                     new_tree2 = self._create_single_node_tree(parent, parent_dudes2)
#                                     new_tree2.paste(new_tree2.root, htree2)
#                                     new_history2.append(new_tree2)
#
#                             res2 = merge_dudes2.merge(other=other2, sp=None)
#                             result_dudes.append(res2)
#
#                             if self.debug:
#                                 new_history2.append(self._create_single_node_tree(parent, res2))
#                                 result_histories.append((parent_dudes_idx, new_history2))
#                                 logging.debug("Result: ")
#                                 logging.debug(str(res2))
#
#                         new_history = []
#                         if self.debug:
#                             logging.debug("Merging: ")
#                             logging.debug(str(other))
#                             logging.debug("into: ")
#                             logging.debug(str(merge_dudes))
#                             logging.debug(f"with: {sp}")
#                             logging.debug("Tree:")
#                             tree.show(line_type="ascii-em", data_property="desc")
#
#                             for htree in node.data.tree_histories[node_dudes_idx]:
#                                 new_tree = self._create_single_node_tree(parent, parent_dudes)
#                                 new_tree.paste(new_tree.root, copy.deepcopy(htree))
#                                 new_history.append(new_tree)
#
#                         res = merge_dudes.merge(other=other, sp=sp)
#                         result_dudes.append(res)
#
#                         if self.debug:
#                             new_history.append(self._create_single_node_tree(parent, res))
#                             result_histories.append((parent_dudes_idx, new_history))
#                             logging.debug("Result: ")
#                             logging.debug(str(res))
#
#                 parent.data.dudes_candidates = result_dudes
#                 if self.debug:
#                     if len(parent.data.tree_histories) == 0:
#                         parent.data.tree_histories = [p[1] for p in result_histories]
#                     else:
#                         new_parent_hists = []
#                         for parent_idx, hist in result_histories:
#                             assert len(hist) > 1  # at least one merge operation
#                             par_hist = copy.deepcopy(parent.data.tree_histories[parent_idx])
#                             app = hist[0]
#                             for child in app.children(app.root):
#                                 for i in range(len(par_hist)):
#                                     par_hist[i].paste(par_hist[i].root, copy.deepcopy(app.subtree(child.identifier)))
#                             par_hist = par_hist + copy.deepcopy(hist[1:])
#                             new_parent_hists.append(par_hist)
#                         parent.data.tree_histories = new_parent_hists
#                         # TODO copies may be necessary, result should be list comprehension with copies of parent instance for every child dudes instance
#
#                 # q.put(parent.identifier)
#                 tree.remove_node(node.identifier)
#             elif not node.is_root():
#                 raise RuntimeError("Node is no leaf!")
#
#         if self.debug:
#             num_hist = 1
#             tree_root = tree.get_node(tree.root)
#             # dist_histories = utils.make_distinct(tree_root.data.tree_histories)
#             # dist_histories = list(dict.fromkeys([tuple(x) for x in tree_root.data.tree_histories]))
#             for hist in tree_root.data.tree_histories:
#                 idx = 0
#                 g = gv.Digraph("History{}".format(num_hist))
#                 for htree in hist:
#                     with g.subgraph(name="cluster_{}".format(idx)) as c:
#                         c.attr(color='white', margin="100.0")
#                         BottomUpComposeDebugStrategy._tree_to_dot(tree=htree, graph=c, node_name_postfix="c" + str(idx))
#                         idx += 1
#                 g.format = 'pdf'
#                 g.render(directory="./")
#                 DotFile.runXDot(str(g), run=False if num_hist < len(tree_root.data.tree_histories) else True)
#                 num_hist += 1
#
#         final_nodes = list(tree.nodes.values())
#         assert len(final_nodes) == 1
#         return final_nodes[0].data.dudes_candidates
