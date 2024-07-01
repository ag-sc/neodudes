import copy
import logging
from collections import defaultdict
from functools import lru_cache
from typing import Iterable, List, Dict, Optional, Union, Any, Tuple

import z3  # type: ignore
from SPARQLBurger.SPARQLQueryBuilder import SPARQLSelectQuery, SPARQLGraphPattern  # type: ignore
from SPARQLBurger.SPARQLSyntaxTerms import Prefix, Triple  # type: ignore
from rdflib import Graph
from rdflib.namespace import NamespaceManager, Namespace
from spacy.tokens import Token

from dudes import utils, consts
from dudes.dudes_token import DUDESToken
from lemon.lemon import LexicalEntry, PartOfSpeech, Marker
from lemon.namespaces import default_namespaces


class Lexicon(object):
    """Lexicon class storing a collection of LexicalEntry objects."""

    # MARKERCLASSES = Union[Sense, LexicalEntry]

    def __init__(self,
                 entries: Iterable[LexicalEntry],
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 namespace_manager: Optional[NamespaceManager] = None):
        """
        Create Lexicon from list of LexicalEntries.

        :param entries: List of LexicalEntries to create Lexicon from.
        :type entries: Iterable[LexicalEntry]
        :param namespaces: List of namespaces which should be resolved. Structure: tuple(<name>, Namespace(<uri>)).
        :type namespaces: Optional[Iterable[Tuple[str, Namespace]]]
        """
        self.nsmanager = utils.create_namespace_manager(namespaces, namespace_manager)

        self.entries: List[LexicalEntry] = list(entries)
        self.written_reps: Dict[str, List[LexicalEntry]] = defaultdict(list)
        self.written_reps_lower: Dict[str, List[LexicalEntry]] = defaultdict(list)
        self._refresh_written_reps()
        self.uris: Dict[str, List[LexicalEntry]] = defaultdict(list)
        self._refresh_uris()

    @staticmethod
    def _get_attr_path(obj, attr_path: str):
        for attr in attr_path.split("."):
            try:
                if getattr(obj, attr) is not None:
                    obj = getattr(obj, attr)
                else:
                    return None
            except AttributeError:
                return None
        return obj

    @staticmethod
    def get_attr_by_name(obj, name: str) -> List:
        res: List = []
        if obj is None:
            return res

        if isinstance(obj, list):
            for listval in obj:
                tres = Lexicon.get_attr_by_name(listval, name)
                res.extend(tres)

        if "__dict__" not in dir(obj):
            return res
        elif name in vars(obj).keys():
            res.append(getattr(obj, name))
        for vname, vval in vars(obj).items():
            if isinstance(vval, list):  # Iterable too generic, strings are also iterable!
                for listval in vval:
                    tres = Lexicon.get_attr_by_name(listval, name)
                    res.extend(tres)
            else:
                tres = Lexicon.get_attr_by_name(vval, name)
                res.extend(tres)
        return res

    @staticmethod
    def get_attr_and_parent_by_name(obj, name: str) -> List:
        res: List = []
        if obj is None:
            return res

        if isinstance(obj, list):
            for listval in obj:
                tres = Lexicon.get_attr_and_parent_by_name(listval, name)
                res.extend(tres)

        if "__dict__" not in dir(obj):
            return res
        elif name in vars(obj).keys():
            res.append((obj, getattr(obj, name)))  # (parent, attr)
        for vname, vval in vars(obj).items():
            if isinstance(vval, list):  # Iterable too generic, strings are also iterable!
                for listval in vval:
                    tres = Lexicon.get_attr_and_parent_by_name(listval, name)
                    res.extend(tres)
            else:
                tres = Lexicon.get_attr_and_parent_by_name(vval, name)
                res.extend(tres)
        return res

    @staticmethod
    def get_uri_by_name(obj, name: Union[str, Iterable[str]]) -> List[str]:
        res: List[str] = []
        if obj is None:
            return res

        if isinstance(name, str):
            name = [name]

        for attr_name in name:
            for v in Lexicon.get_attr_by_name(obj, attr_name):
                if isinstance(v, str):
                    res.append(v)
                elif isinstance(v, Iterable) and all(isinstance(i, str) for i in v):
                    res.extend(v)
                else:
                    uris = Lexicon.get_attr_by_name(v, "uri")
                    for uri in uris:
                        if isinstance(uri, str):
                            res.append(uri)
                        else:
                            logging.warning("URI" + uri + " is not a string, ignoring!")

        return res

    @staticmethod
    def get_uris_of_entry(entry: LexicalEntry) -> List[str]:
        return Lexicon.get_uri_by_name(entry, consts.uri_attrs)

    def get_form_written_reps_of_entry(entry: LexicalEntry) -> List[str]:
        res = []
        for n in Lexicon.get_attr_by_name(entry.canonical_form, "written_rep"):
            if n is None:
                continue
            else:
                if not isinstance(n, str) or len(n) > 0:
                    res.append(n)
        for n in Lexicon.get_attr_by_name(entry.other_form, "written_rep"):
            if n is None:
                continue
            else:
                if not isinstance(n, str) or len(n) > 0:
                    res.append(n)
        return res

    @staticmethod
    def get_written_reps_of_entry(entry: LexicalEntry) -> List[str]:
        res = []
        for n in Lexicon.get_attr_by_name(entry, "written_rep"):
            if n is None:
                continue
            else:
                if not isinstance(n, str) or len(n) > 0:
                    res.append(n)
        return res

    def _refresh_written_reps(self):
        self.written_reps = self._get_entries_by_name("written_rep")
        for k, v in self.written_reps.items():
            self.written_reps_lower[k.lower()] += v

    def _refresh_uris(self):
        self.uris = self._get_entries_by_name("uri")

    def _get_entries_by_name(self, name: str) -> Dict[Any, List[LexicalEntry]]:
        res = defaultdict(list)
        for entry in self.entries:
            for n in Lexicon.get_attr_by_name(entry, name):
                if n is None:
                    continue
                else:
                    if not isinstance(n, str) or len(n) > 0:
                        res[n].append(entry)
        return res

    def _get_candidates(self, token: Union[DUDESToken, Token], strict: bool = True):
        candidates: List[LexicalEntry] = []
        #written_reps = self.written_reps
        #if not strict:
        #    written_reps = self.written_reps_lower

        if isinstance(token, DUDESToken):
            if not strict:
                candidates = sum([self.written_reps.get(t, []) for t in token.text_], [])
                candidates += sum([self.written_reps_lower.get(t.lower(), []) for t in token.text_], [])
            else:
                candidates = self.written_reps.get(token.text_[0], []) + self.written_reps_lower.get(token.text_[0].lower(), [])
                candidates += self.written_reps_lower.get(token.text_[0].lower(), [])

        if len(candidates) == 0 and not strict:
            candidates = self.written_reps.get(token.text, []) + self.written_reps_lower.get(token.text.lower(), [])

        return utils.make_distinct(candidates)

    @lru_cache(maxsize=128)
    def find_entry(self, token: Union[DUDESToken, Token], strict: bool = False) -> Tuple[List[LexicalEntry], bool]:
        """
        Find candidate LexicalEntries which might correspond to the given DUDESToken or spaCy token.

        :param token: Token used to find corresponding LexicalEntry and do some disambiguation.
        :type token: Union[DUDESToken, Token]
        :param strict: Use only strict candidate fetching if true.
        :type strict: bool
        :return: List of LexicalEntries which are most likely to correspond to the given token and bool signaling whether strict candidates were successful or not.
        :rtype: Tuple[List[LexicalEntry], bool]
        """
        # TODO: use auxiliaries for disambiguation?
        # if token.text not in self.written_reps.keys() and token.text.lower() not in self.written_reps.keys():
        #    return []
        candidates = self._get_candidates(token=token, strict=True)
        is_strict = True

        if len(candidates) == 0 and not strict:
            candidates = self._get_candidates(token=token, strict=False)
            is_strict = False

        if len(candidates) == 0:
            return [], is_strict
        elif len(candidates) == 1:
            return utils.make_distinct(candidates), is_strict
        else:
            old_candidates = copy.copy(candidates)
            match token.pos_:  # currently, we don't have entries for all relevant word types...
                case "ADJ":
                    candidates = [cand for cand in candidates if cand.part_of_speech == PartOfSpeech.ADJECTIVE]
                case "ADP":
                    candidates = [cand for cand in candidates if cand.part_of_speech == PartOfSpeech.PREPOSITION]
                case "ADV":
                    pass
                case "NOUN":
                    candidates = [cand for cand in candidates if cand.part_of_speech == PartOfSpeech.NOUN]
                case "PRON":
                    pass
                case "PROPN":
                    pass
                case "VERB":
                    candidates = [cand for cand in candidates if cand.part_of_speech == PartOfSpeech.VERB]

            if len(candidates) == 0:  # instead of filtering out everything just return ambiguous result
                candidates = old_candidates

            if len(candidates) > 0:
                logging.warning("Disambiguation of lexical entry failed.")
            return utils.make_distinct(candidates), is_strict

    def find_by_uri(self, uri: str) -> List[LexicalEntry]:
        """
        Find all LexicalEntries with the given URI. Should be unique, i.e. only one entry returned, but not always is.

        :param uri: URI to search LexicalEntries for.
        :type uri: str
        :return: List of LexicalEntries which have the given URI.
        :rtype: List[LexicalEntry]
        """
        return self.uris[uri]

    def find_by_qname(self, qname: str) -> List[LexicalEntry]:
        """
        Find all LexicalEntries with the given URI. Should be unique, i.e. only one entry returned, but not always is.

        :param qname: QName to search LexicalEntries for. Is expanded using the given namespace manager domains.
        :type qname: str
        :return: List of LexicalEntries which have the given URI.
        :rtype: List[LexicalEntry]
        """
        return self.uris[self.nsmanager.expand_curie(qname)]

    @staticmethod
    def marker_to_str(marker: Marker) -> str:
        if isinstance(marker, str):
            return marker
        elif isinstance(marker, Marker):
            return marker.canonical_form.written_rep
        else:
            raise RuntimeError("Invalid marker type!")
    @staticmethod
    def get_markers(entry: LexicalEntry) -> List[Marker]:  # tuple[Marker, MARKERCLASSES]
        """
        Get all instances of the Marker class in a given lexicalEntry.

        :param entry: Entry to get marker instances for.
        :type entry: LexicalEntry
        :return: List of Marker objects which are part of the given LexicalEntry.
        :rtype: List[Marker]
        """
        return Lexicon.get_attr_by_name(entry, "marker")
        # markers: List[Marker] = []
        #
        # try:
        #     markers.append((entry.syn_behavior.prepositional_adjunct.marker, entry))
        # except AttributeError:
        #     pass
        #
        # for sense in entry.sense:
        #     try:
        #         markers.append((sense.subj_of_prop.marker, sense))
        #     except AttributeError:
        #         pass
        #     try:
        #         markers.append((sense.obj_of_prop.marker, sense))
        #     except AttributeError:
        #         pass
        #
        # return markers
