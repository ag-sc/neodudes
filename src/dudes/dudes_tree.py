from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import treelib  # type: ignore

from dudes.dudes import DUDES
from dudes.dudes_token import DUDESToken
from dudes.qa.tree_merging.tree_scorer import StrictFirstScorer
from lemon.lemon import LexicalEntry


@dataclass
class DUDESTreeCombData(object):
    """Data class for DUDESParser tree nodes"""
    token: DUDESToken
    """DUDESToken of the tree node, possibly with merged tokens included."""
    lex_candidates: List[LexicalEntry]
    """Candidates for LexicalEntries for the node/token."""
    are_strict_candidates: bool = True
    """Whether candidates are strict or fallback was needed."""
    dudes_candidates: List[DUDES] = field(default_factory=list)
    """Candidate DUDES if there are different combinations when multiple lexical entries are present."""
    tree_histories: List[List[Tree]] = field(default_factory=list)
    """Tree merge history associated with candidate DUDES."""
    marker: Optional[str] = None
    """ADP marker of pobj nodes."""

    @property
    def desc(self) -> str:
        """
        :return: More expressive textual representation of the node/token for printing: <text>\_<pos\_>\_<dep\_>\_<i>.
        :rtype: str
        """
        res = f"{self.token.text}_{self.token.pos_}_{self.token.dep_}_{self.token.tag_}"#_{self.token.i}
        strict_score = (1 + len(self.token.merged_tokens)) * StrictFirstScorer.get_multiplier(self, strict_only=True)
        strict_type = StrictFirstScorer.get_match_type(self, strict_only=True)
        nonstrict_score = (1 + len(self.token.merged_tokens)) * StrictFirstScorer.get_multiplier(self, strict_only=False)
        nonstrict_type = StrictFirstScorer.get_match_type(self, strict_only=False)
        res += f"_weight_{strict_score}-{strict_type}_{nonstrict_score}-{nonstrict_type}"
        if self.marker is not None:
            res += f"_marker_{self.marker}"

        return res

    @property
    def lmarker(self):
        return self.marker.lower() if self.marker is not None else None

    def __repr__(self):
        return self.desc
        # return json.dumps({
        #     "token": str(self.token),
        #     "lex_candidates": self.lex_candidates,
        #     "strict_candidates": self.are_strict_candidates,
        #     "dudes_candidates": self.dudes_candidates,
        #     #"tree_histories": self.tree_histories,
        #     "marker": self.marker,
        # }, sort_keys=True)

    def __str__(self):
        return self.desc
        # return json.dumps({
        #     "token": str(self.token),
        #     "lex_candidates": self.lex_candidates,
        #     "strict_candidates": self.are_strict_candidates,
        #     "dudes_candidates": self.dudes_candidates,
        #     # "tree_histories": self.tree_histories,
        #     "marker": self.marker,
        # }, sort_keys=True)
        #return json.dumps(self, default=lambda o: {k: v for k, v in o.__dict__.items() if str(type(v)) not in ["<class 'spacy.tokens.token.Token'>"]} if hasattr(o, "__dict__") else dict(), sort_keys=True)

    # @classmethod
    # def from_tree_data(cls, td: DUDESTreeData):
    #     return cls(token=td.token,
    #                lex_candidates=td.lex_candidates,
    #                dudes_candidates=[td.dudes] if td.dudes is not None else [],
    #                marker=td.marker)


class Tree(treelib.Tree):
    def __eq__(self, other):
        return self.tree_to_json(self) == self.tree_to_json(other)

    def __ne__(self, other):
        return self.tree_to_json(self) != self.tree_to_json(other)

    def __hash__(self):
        return hash(self.tree_to_json(self))

    # @staticmethod
    # def _remove_circular_refs(ob, _seen=None): # https://stackoverflow.com/questions/44777369/remove-circular-references-in-dicts-lists-tuples
    #     if _seen is None:
    #         _seen = set()
    #     if id(ob) in _seen:
    #         # circular reference, remove it.
    #         return None
    #     _seen.add(id(ob))
    #     res = ob
    #     if isinstance(ob, dict):
    #         res = {
    #             Tree._remove_circular_refs(k, _seen): Tree._remove_circular_refs(v, _seen)
    #             for k, v in ob.items()}
    #     elif isinstance(ob, (list, tuple, set, frozenset)):
    #         res = type(ob)(Tree._remove_circular_refs(v, _seen) for v in ob)
    #     # remove id again; only *nested* references count
    #     _seen.remove(id(ob))
    #     return res

    @staticmethod
    def tree_to_json(t):
        tdict = t.to_dict(with_data=True, sort=True)
        return json.dumps(tdict, default=lambda o: {
            k: str(v) for k, v in o.__dict__.items()
            if str(type(v)) not in ["<class 'spacy.tokens.token.Token'>"] and
               k not in ["main_token", "merged_tokens"]
        } if hasattr(o, "__dict__") else None, sort_keys=True)
