from abc import ABC, abstractmethod
from typing import Tuple, Any

from treelib import Tree

from dudes import consts, utils


class TreeScorer(ABC):
    @abstractmethod
    def score(self, tree: Tree) -> Any:
        pass


class StrictFirstScorer(TreeScorer):

    @staticmethod
    def has_lexicon_entry(data, strict_only=False):
        return len(data.lex_candidates) > 0 and (data.are_strict_candidates or not strict_only)

    @staticmethod
    def has_uri(data, strict_only=False):
        return len(data.token.candidate_uris) > 0 and (data.token.are_strict_candidate_uris or not strict_only)

    @staticmethod
    def is_special_word(data):
        return all([tok.text.lower() in consts.special_words for tok in data.token.merged_tokens + [data.token.main_token]])

    @staticmethod
    def is_number(data):
        return data.token.main_token.pos_ == "NUM" and len(data.token.merged_tokens) == 0

    @staticmethod
    def has_in(data):
        # these words cannot be added to special words without disturbing behavior outside this function
        return utils.any_in_list(data.token.text_, ["in"], lower=True)

    @staticmethod
    def get_multiplier(data, strict_only=False):
        if StrictFirstScorer.has_lexicon_entry(data, strict_only):
            return 1.0
        elif StrictFirstScorer.is_number(data):
            return 0.9
        elif StrictFirstScorer.has_uri(data, strict_only):
            return 0.9
        elif StrictFirstScorer.is_special_word(data):
            return 0.8
        elif StrictFirstScorer.has_in(data):
            return 0.8
        return 0.0

    @staticmethod
    def get_match_type(data, strict_only=False):
        if StrictFirstScorer.has_lexicon_entry(data, strict_only):
            return "lex"
        elif StrictFirstScorer.is_number(data):
            return "num"
        elif StrictFirstScorer.has_uri(data, strict_only):
            return "uri"
        elif StrictFirstScorer.is_special_word(data):
            return "special"
        elif StrictFirstScorer.has_in(data):
            return "in"
        return ""

    @staticmethod
    def tree_score(tree: Tree, strict_only=False) -> float:
        total = 0.0  # len(tree.nodes)
        found = 0.0

        for n in tree.nodes:
            node = tree.get_node(n)
            weight = 1 + len(node.data.token.merged_tokens)
            total += weight
            found += weight * StrictFirstScorer.get_multiplier(node.data, strict_only)

        if total == 0:
            return 0.0
        else:
            return found / total

    @staticmethod
    def tree_total(tree: Tree) -> int:
        total = 0
        for n in tree.nodes:
            node = tree.get_node(n)
            total += 1 + len(node.data.token.merged_tokens)
        return total

    def score(self, tree: Tree) -> Tuple[float, float]:
        return self.tree_score(tree, strict_only=True), self.tree_score(tree)

class FewNodesStrictFirstScorer(StrictFirstScorer):
    def score(self, tree: Tree) -> Tuple[float, float, int]:
        return self.tree_score(tree, strict_only=True), self.tree_score(tree), -len(tree.nodes)

class WeightedAvgStrictFirstScorer(StrictFirstScorer):
    def score(self, tree: Tree) -> float:
        strict_score = self.tree_score(tree, strict_only=True)
        nonstrict_score = self.tree_score(tree)
        nodenum_score = 1.0-(len(tree.nodes)/self.tree_total(tree))

        return (3*strict_score + nonstrict_score + 2.0*nodenum_score) / 6.0
