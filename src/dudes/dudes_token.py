# from dataclasses import dataclass
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple, Any
import re

from spacy.tokens import Token
from stanza.models.common.doc import Word

from dudes import consts


@dataclass(frozen=True)
class TokenWrapper:
    pos_: str
    """pos\_ value of the initial token."""
    dep_: str
    """dep\_ value of the initial token."""
    tag_: str
    """tag\_ value of the initial token."""
    doc: Any
    whitespace_: str
    text: str
    """text value of the initial token."""
    i: int
    """The index of the initial token within the parent document."""
    idx: int
    """The character offset of the initial token within the parent document."""
    is_stop: bool
    """Whether the token is a stop word."""
    ent_kb_id_: Tuple[str, ...]
    """From DBPEDIA spotlight, either link to DBPEDIA entity or empty string if token is not recognized as part of 
    such an entity."""

    @property
    def data(self):
        return self

    @property
    def token(self):
        return self


    orig_token: Token | Word

    @classmethod
    def from_token(cls, token: Token | Word):
        if isinstance(token, Token):
            return cls(
                pos_=token.pos_,
                dep_=token.dep_,
                tag_=token.tag_,
                doc=token.doc,
                whitespace_=token.whitespace_,
                text=token.text,
                i=token.i,
                idx=token.idx,
                is_stop=token.is_stop,
                ent_kb_id_=tuple([token.ent_kb_id_] if len(token.ent_kb_id_) > 0 else []),
                orig_token=token
            )
        elif isinstance(token, Word):
            tok_start_char = token.start_char
            whitespace_ = ""
            tok_idx = token.sent.words.index(token)

            if token.end_char is not None:
                if tok_idx < len(token.sent.words) - 1 and token.sent.words[tok_idx + 1].start_char is not None:
                    whitespace_ = " " * (token.sent.words[tok_idx + 1].start_char - token.end_char)
                else:
                    whitespace_ = ""  # last token in sentence
            else:
                succ_none_tokens = []
                last_end = 0
                for i in range(1, tok_idx+1):
                    if token.sent.words[tok_idx-i].start_char is not None:
                        last_end = token.sent.words[tok_idx-i].end_char
                        break
                    # else:
                    #     none_tokens.append(token.sent.words[tok_idx-i])
                first_start = None
                for i in range(1, len(token.sent.words) - tok_idx):
                    if token.sent.words[tok_idx+i].end_char is not None:
                        first_start = token.sent.words[tok_idx+i].start_char
                        break
                    else:
                        succ_none_tokens.append(token.sent.words[tok_idx+i])
                tok_start_char = token.sent.text.rfind(token.text, last_end, first_start)
                # successor None nodes should mean no space, but otherwise this should also determine it correctly
                if len(succ_none_tokens) > 0:
                    first_start = token.sent.text.rfind(succ_none_tokens[0].text, last_end, first_start)

                tok_end_char = tok_start_char + len(token.text)
                whitespace_ = " " * len(token.sent.text[tok_end_char:first_start])

            return cls(
                pos_=token.upos,
                dep_=token.deprel.replace(":", ""),
                tag_=token.xpos,
                whitespace_=whitespace_,
                doc=tuple([w for w in token.sent.words]),
                text=token.text,
                i=token.id,
                idx=tok_start_char,#token.start_char,
                is_stop=False,
                ent_kb_id_=tuple(),
                orig_token=token
            )

# @dataclass
class DUDESToken(object):
    """Wrapper class for spaCy tokens, supporting merges with other tokens."""
    # text: str
    # """Initially token.text, after merge(s) the combined text of the initial and all merged tokens."""
    pos_: str
    """pos\_ value of the initial token."""
    dep_: str
    """dep\_ value of the initial token."""
    tag_: str
    """tag\_ value of the initial token."""
    i: int
    """The index of the initial token within the parent document."""
    idx: int
    """The character offset of the initial token within the parent document."""
    main_token: TokenWrapper
    """Initial spaCy token used for object creation."""
    ent_kb_id_: List[str]
    """From DBPEDIA spotlight, either link to DBPEDIA entity or empty string if token is not recognized as part of 
    such an entity."""
    merged_tokens: list[TokenWrapper]
    """List of spaCy tokens which have been merged into this token."""
    tagger_kb_ids: list[str]
    """List of URIs from the trie tagger."""
    candidate_uris: list[str]
    """Set of entity URIs chosen as candidates for this token."""
    are_strict_candidate_uris: bool
    """Whether the candidate URIs are strict candidates, i.e. whether they perfectly match the string or not."""

    def __init__(self, token: Token | Word):
        """
        Create DUDESToken wrapper object from spaCy token.

        :param token: spaCy token to create DUDESToken from.
        :type token: Token
        """
        if isinstance(token, Token) or isinstance(token, Word):
            wrapped_token = TokenWrapper.from_token(token)
            self.pos_ = wrapped_token.pos_
            self.dep_ = wrapped_token.dep_
            self.tag_ = wrapped_token.tag_
            self.i = wrapped_token.i
            self.idx = wrapped_token.idx
            self.is_stop = wrapped_token.is_stop
            self.main_token = wrapped_token
            self.ent_kb_id_ = list(wrapped_token.ent_kb_id_)

        self.merged_tokens = []
        self.tagger_kb_ids = []
        self.candidate_uris = []
        self.are_strict_candidate_uris = True


    @staticmethod
    def _token_string_reconstruction(tokens: List[TokenWrapper]) -> str:
        """
        Reconstruct the original string from a list of DUDESTokens.

        :param tokens: List of DUDESTokens to reconstruct the original string from.
        :type tokens: List[DUDESToken]
        :return: Reconstructed original string.
        :rtype: str
        """
        tokens = sorted(tokens, key=lambda x: x.idx)
        idx_offset = tokens[0].idx
        excl_end_idx = tokens[-1].idx + len(tokens[-1].text)

        rstr = " " * (excl_end_idx - idx_offset)
        for t in tokens:
            rstr = rstr[:t.idx - idx_offset] + t.text + rstr[t.idx - idx_offset + len(t.text):]

        return rstr

    @staticmethod
    def reconstruct_str(tokens) -> str:
        return "".join([i.text + i.whitespace_ if i != tokens[-1] else i.text for i in tokens])

    @property
    def text_(self):
        all_tokens = self.merged_tokens + [self.main_token]
        all_tokens.sort(key=lambda x: x.idx)

        res = [self.reconstruct_str(all_tokens)]

        if len(all_tokens) > 1:  # or res[0].is_stop):
            first_unnecessary = (all_tokens[0].dep_ in ["det"] or all_tokens[0].pos_ in ["ADP", "PART", "AUX"])
            last_unnecessary = (all_tokens[-1].dep_ in ["det"] or all_tokens[-1].pos_ in ["ADP", "PART", "AUX"])
            if first_unnecessary:
                res = [self.reconstruct_str(all_tokens[1:])] + res
            else:
                res = res + [self.reconstruct_str(all_tokens[1:])]
            if last_unnecessary:
                res = [self.reconstruct_str(all_tokens[:-1])] + res
            else:
                res = res + [self.reconstruct_str(all_tokens[:-1])]

            if len(res) > 2:
                if first_unnecessary and last_unnecessary:
                    res = [self.reconstruct_str(all_tokens[1:-1])] + res
                else:
                    res = res + [self.reconstruct_str(all_tokens[1:-1])]
        return res
        #         res = res[1:]

        # if all([tok.text in ["than"] + consts.comp_keywords for tok in all_tokens]): # or tok.tag_ in ["JJR", "RBR"]
        #     res = all_tokens
        # else:
        #     res = all_tokens
        #     if len(res) > 1 and (res[0].dep_ in ["det"] or res[0].pos_ in ["ADP", "PART", "AUX"]):# or res[0].is_stop):
        #         res = res[1:]
        #     if len(res) > 1 and (res[-1].dep_ in ["det"] or res[-1].pos_ in ["ADP", "PART", "AUX"]):# or res[-1].is_stop):
        #         res = res[:-1]
        #
        # if len(res) == 0:
        #     res = all_tokens  # [t for t in [token for token in all_tokens]]
        #
        # assert len(res) > 0
        #
        # res.sort(key=lambda x: x.idx)
        # idx_offset = res[0].idx
        # excl_end_idx = res[-1].idx + len(res[-1].text)
        #
        # rstr = " " * (excl_end_idx - idx_offset)
        # for t in res:
        #     rstr = rstr[:t.idx - idx_offset] + t.text + rstr[t.idx - idx_offset + len(t.text):]
        #
        # rstr = re.sub(' +', ' ', rstr.strip())
        #
        # return rstr  # " ".join([t.text for t in res])

    @property
    def text(self):
        """Initially token.text, after merge(s) the combined text of the initial and all merged tokens."""
        res = self.merged_tokens + [self.main_token]
        assert len(res) > 0

        res = sorted(res, key=lambda x: x.i)
        idx_offset = res[0].idx
        excl_end_idx = res[-1].idx + len(res[-1].text)

        rstr = " " * (excl_end_idx - idx_offset)
        for t in res:
            rstr = rstr[:t.idx - idx_offset] + t.text + rstr[t.idx - idx_offset + len(t.text):]

        rstr = re.sub(' +', '_', rstr.strip())

        return rstr  # " ".join([t.text for t in res])

    def merge(self, token: DUDESToken) -> None:
        """
        Merge single other DUDESToken into this one and combine text representations in correct order.

        :param token: Other DUDESToken to merge into this one.
        :type token: DUDESToken
        """
        # if token.i < self.i:
        #     self.text = token.text + "_" + self.text
        # else:
        #     self.text = self.text + "_" + token.text

        self.merged_tokens.append(token.main_token)
        self.merged_tokens.extend(token.merged_tokens)

    def merge_all(self, tokens: Iterable[DUDESToken]) -> None:
        """
        Merge all given DUDESTokens into this one and combine text representations in correct order.

        :param tokens: Other DUDESTokens to merge into this one.
        :type tokens: Iterable[DUDESToken]
        """
        # tokens = list(tokens)
        # before_self = sorted([token for token in tokens if token.i < self.i], key=lambda x: -x.i)
        # # Marcus merged with [F., Gary] -> F. Marcus merged with [Gary] -> Gary F. Marcus
        # after_self = sorted([token for token in tokens if token.i > self.i], key=lambda x: x.i)
        # # after main token the other way around
        #
        # stokens = before_self + after_self

        for token in tokens:
            self.merge(token=token)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['main_token', 'merged_tokens']:
                setattr(result, k, copy.deepcopy(v, memo))
            elif k == 'main_token':
                setattr(result, k, v)
            else:  # 'merged_tokens'
                setattr(result, k, v.copy())
        return result

    def __repr__(self):
        return json.dumps({
            "pos_": self.pos_,
            "dep_": self.dep_,
            "tag_": self.tag_,
            "i": self.i,
            "idx": self.idx,
            #"main_token": self.main_token,
            "ent_kb_id_": self.ent_kb_id_,
            #"merged_tokens": self.merged_tokens,
            "tagger_kb_ids": self.tagger_kb_ids,
        }, sort_keys=True)

    def __str__(self):
        return json.dumps({
            "pos_": self.pos_,
            "dep_": self.dep_,
            "tag_": self.tag_,
            "i": self.i,
            "idx": self.idx,
            # "main_token": self.main_token,
            "ent_kb_id_": self.ent_kb_id_,
            # "merged_tokens": self.merged_tokens,
            "tagger_kb_ids": self.tagger_kb_ids,
        }, sort_keys=True)
