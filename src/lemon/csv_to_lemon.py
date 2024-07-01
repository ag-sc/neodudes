import csv
from pathlib import Path
from typing import List, Any

from lemon.lemon import *

class csv_to_lemon(object):
    def __init__(self, path: str, uri_num=0):
        self.uri_num = uri_num
        self.base_uri = "http://localhost:8000/#list-{}".format(Path(path).stem)
        self.csv_file = open(path)
        self.csv_dict = csv.DictReader(self.csv_file, delimiter=',')
        assert self.csv_dict.fieldnames is not None

    def _next_uri(self):
        res = self.base_uri + str(self.uri_num)
        self.uri_num += 1
        return res

    def _adjective(self, written_rep: str, on_property: str, has_value: str):
        entry = LexicalEntry(uri=self._next_uri())
        entry.part_of_speech = PartOfSpeech.ADJECTIVE
        entry.canonical_form = CanonicalForm(uri=self._next_uri(), written_rep=written_rep)
        placeholder_uri = "http://localhost:8000/#arg1"  # self._next_uri()
        entry.syn_behavior = [
            SynBehavior(uri=self._next_uri(),
                        copulative_subject=placeholder_uri,
                        type=["lexinfo:AdjectivePredicateFrame"])
        ]
        ref = Reference(uri=self._next_uri(),
                        type=["owl:Restriction"],
                        on_property=on_property,
                        has_value=has_value)
        entry.sense = [
            Sense(uri=self._next_uri(),
                  type=["lemon:LexicalSense"],
                  reference=[ref],
                  is_a=[placeholder_uri])
        ]
        return entry

    def _category_noun(self, written_rep: str, on_property: str, has_value: str):
        entry = LexicalEntry(uri=self._next_uri())
        entry.part_of_speech = PartOfSpeech.NOUN
        entry.canonical_form = CanonicalForm(uri=self._next_uri(), written_rep=written_rep)
        placeholder_uri = "http://localhost:8000/#arg1"  # self._next_uri()
        entry.syn_behavior = [
            SynBehavior(uri=self._next_uri(),
                        copulative_subject=placeholder_uri,
                        type=["lexinfo:NounPredicateFrame"])
        ]
        ref = Reference(uri=self._next_uri(),
                        type=["owl:Restriction"],
                        on_property=on_property,
                        has_value=has_value)
        entry.sense = [
            Sense(uri=self._next_uri(),
                  type=["lemon:LexicalSense"],
                  reference=[ref],
                  is_a=[placeholder_uri])
        ]
        return entry

    def _noun(self, singular: str, plural: str, on_property: str, has_value: str):
        entry = self._category_noun(written_rep=singular, on_property=on_property, has_value=has_value)
        entry.other_form = [
            OtherForm(uri=self._next_uri(), written_rep=singular, number=Number.SINGULAR),
            OtherForm(uri=self._next_uri(), written_rep=plural, number=Number.PLURAL)
        ]
        return entry

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        row: Any
        property_uri_cands: List[str]
        property_uri: str
        value_cols: List[str]

        while True:
            row = next(self.csv_dict)
            assert self.csv_dict.fieldnames is not None
            property_uri_cands = [name for name in self.csv_dict.fieldnames if name.startswith("http")]
            assert len(property_uri_cands) == 1
            property_uri = property_uri_cands[0]

            value_cols = [vc for vc in set(self.csv_dict.fieldnames).difference({"count", property_uri})]
            assert len(value_cols) >= 1

            if not any([len(row[val]) == 0 for val in value_cols]):
                break

        entry = LexicalEntry(uri=self._next_uri())
        if "adjective" in value_cols:
            assert len(value_cols) == 1
            return self._adjective(written_rep=row["adjective"], on_property=property_uri, has_value=row[property_uri])
        elif "noun" in value_cols:
            assert len(value_cols) == 1
            return self._category_noun(written_rep=row["noun"], on_property=property_uri, has_value=row[property_uri])
        elif "singular" in value_cols and "plural" in value_cols:
            assert len(value_cols) == 2
            return self._noun(singular=row["singular"], plural=row["plural"],
                              on_property=property_uri, has_value=row[property_uri])
        # assert len(value_cols) == 1
        # entry.part_of_speech = PartOfSpeech.NOUN
        # entry.canonical_form = CanonicalForm(uri=next_uri(), written_rep=row["noun"])
        # placeholder_uri = "http://localhost:8000/#arg1"#next_uri()
        # entry.syn_behavior = [
        #     SynBehavior(uri=next_uri(),
        #                 copulative_subject=placeholder_uri,
        #                 type=["lexinfo:AdjectivePredicateFrame"])
        # ]
        # ref = Reference(uri=next_uri(),
        #                 type=["owl:Restriction"],
        #                 on_property=property_uri,
        #                 has_value=row[property_uri])
        # entry.sense = [
        #     Sense(uri=next_uri(),
        #           type=["lemon:LexicalSense"],
        #           reference=ref,
        #           is_a=[placeholder_uri])
        # ]
        # res.append(entry)
        #raise StopIteration()
