from pprint import pprint
from typing import Optional

from lemon.lemon import LexicalEntry
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon

import re

prefix_re = re.compile(r"PREFIX\s+([a-zA-Z0-9]+):\s+<([^>]+)>\s+")  # PREFIX res: <http://dbpedia.org/resource/>

def entry_to_str(entry: LexicalEntry):
    argnum = 0
    args = dict()
    assert entry.canonical_form is not None
    res = f"Lexical Entry for {entry.canonical_form.written_rep}:\n"
    res += f"Part of speech: {entry.part_of_speech}\n" if entry.part_of_speech is not None else ""
    res += f"Canonical form: {entry.canonical_form.written_rep}"
    res += f" {entry.canonical_form.verb_form_mood}\n" if entry.canonical_form.verb_form_mood is not None else "\n"
    if len(entry.other_form) > 0:
        res += f"Other forms:\n"
        for f in entry.other_form:
            if f is not None:
                res += f"{f.written_rep}"
                res += f" {f.person}" if f.person is not None else ""
                res += f" {f.number}" if f.number is not None else ""
                res += f" {f.tense}" if f.tense is not None else ""
                res += "\n"
    if len(entry.sense) > 0:
        res += f"Sense:\n"
        for s in entry.sense:
            if s is not None and s.condition is not None:
                res += f"Condition:\n"
                if isinstance(s.condition, str):
                    res += s.condition
                else:
                    res += f"Property domain: {s.condition.property_domain}\n" if s.condition.property_domain is not None else ""
                    res += f"Property range: {s.condition.property_range}\n" if s.condition.property_range is not None else ""

                for r in s.reference:
                    if isinstance(r, str):
                        res += f"Reference: {r}\n"
                    elif r is not None:
                        res += f"Reference: {r.uri}\n" if r.uri is not None else ""
                        res += f"Bound to: {r.bound_to}\n" if r.bound_to is not None else ""
                        res += f"Has value: {r.has_value}\n" if r.has_value is not None else ""
                        res += f"Degree: {r.degree}\n" if r.degree is not None else ""
                        res += f"On property: {r.on_property}\n" if r.on_property is not None else ""

                if isinstance(s.subj_of_prop, str):
                    if s.subj_of_prop not in args:
                        args[s.subj_of_prop] = f"arg{argnum}"
                        argnum += 1
                    res += f"Subject of property: {args[s.subj_of_prop]}\n"
                elif s.subj_of_prop is not None and not isinstance(s.subj_of_prop, str):
                    res += f"Subject of property: "
                    if s.subj_of_prop.uri is not None and s.subj_of_prop.uri not in args:
                        args[s.subj_of_prop.uri] = f"arg{argnum}"
                        argnum += 1

                    res += f"{args[s.subj_of_prop.uri]}" if s.subj_of_prop.uri is not None else ""
                    res += f" Marker: {s.subj_of_prop.marker.canonical_form.written_rep}" if s.subj_of_prop.marker is not None and s.subj_of_prop.marker.canonical_form is not None else ""
                    res += "\n"

                if isinstance(s.obj_of_prop, str):
                    if s.obj_of_prop not in args:
                        args[s.obj_of_prop] = f"arg{argnum}"
                        argnum += 1
                    res += f"Object of property: {args[s.obj_of_prop]}\n"
                elif s.obj_of_prop is not None and not isinstance(s.obj_of_prop, str):
                    res += f"Object of property: "
                    if s.obj_of_prop.uri is not None and s.obj_of_prop.uri not in args:
                        args[s.obj_of_prop.uri] = f"arg{argnum}"
                        argnum += 1
                    res += f"{args[s.obj_of_prop.uri]}" if s.obj_of_prop.uri is not None else ""
                    res += f" Marker: {s.obj_of_prop.marker.canonical_form.written_rep}" if s.obj_of_prop.marker is not None and s.obj_of_prop.marker.canonical_form is not None else ""
                    res += "\n"

    if len(entry.syn_behavior) > 0:
        for b in entry.syn_behavior:
            if isinstance(b, str):
                res += f"Syntactic behavior: {b}\n"
            elif b is not None and not isinstance(b, str):
                res += f"Syntactic behavior: \n"
                if b.attributive_arg is not None:
                    if b.attributive_arg not in args:
                        args[b.attributive_arg] = f"arg{argnum}"
                        argnum += 1
                    res += f"Attributive argument: {args[b.attributive_arg]}\n"
                if b.copulative_arg is not None:
                    if b.copulative_arg not in args:
                        args[b.copulative_arg] = f"arg{argnum}"
                        argnum += 1
                    res += f"Copulative argument: {args[b.copulative_arg]}\n"
                if b.copulative_subject is not None:
                    if b.copulative_subject not in args:
                        args[b.copulative_subject] = f"arg{argnum}"
                        argnum += 1
                    res += f"Copulative subject: {args[b.copulative_subject]}\n"
                if b.prepositional_adjunct is not None:
                    if b.prepositional_adjunct.uri is not None and b.prepositional_adjunct.uri not in args:
                        args[b.prepositional_adjunct.uri] = f"arg{argnum}"
                        argnum += 1
                    res += f"Prepositional adjunct: {args[b.prepositional_adjunct.uri]} " if b.prepositional_adjunct is not None and b.prepositional_adjunct.uri is not None else ""
                    res += f" Marker: {b.prepositional_adjunct.marker.canonical_form.written_rep}\n" if b.prepositional_adjunct.marker is not None and b.prepositional_adjunct.marker.canonical_form is not None else "\n"
                if b.direct_object is not None:
                    if isinstance(b.direct_object, str):
                        if b.direct_object not in args:
                            args[b.direct_object] = f"arg{argnum}"
                            argnum += 1
                        res += f"Direct object: {args[b.direct_object]}\n"
                    elif b.direct_object is not None and not isinstance(b.direct_object, str):
                        if b.direct_object.uri is not None and b.direct_object.uri not in args:
                            args[b.direct_object.uri] = f"arg{argnum}"
                            argnum += 1
                        res += "Direct object: "
                        res += f"{args[b.direct_object.uri]} " if b.direct_object.uri is not None else ""
                        res += f" Marker: {b.direct_object.marker.canonical_form.written_rep}\n" if b.direct_object.marker is not None and b.direct_object.marker.canonical_form is not None else "\n"
                if b.subject is not None:
                    if isinstance(b.subject, str):
                        if b.subject not in args:
                            args[b.subject] = f"arg{argnum}"
                            argnum += 1
                        res += f"Subject: {args[b.subject]}\n"
                    elif b.subject is not None and not isinstance(b.subject, str):
                        if b.subject.uri is not None and b.subject.uri not in args:
                            args[b.subject.uri] = f"arg{argnum}"
                            argnum += 1
                        res += "Subject: "
                        res += f"{args[b.subject.uri]} " if b.subject.uri is not None else ""
                        res += f" Marker: {b.subject.marker.canonical_form.written_rep}\n" if b.subject.marker is not None and b.subject.marker.canonical_form is not None else "\n"

    return res + "\n\n"



def get_entries_for_question(question: str, query: str, lex: Optional[Lexicon] = None):
    if lex is None:
        lex = LEMONParser.from_ttl_dir().lexicon

    matches = re.findall(prefix_re, query)
    for match in matches:
        lex.nsmanager.bind(match[0], match[1], override=True, replace=True)

    relevant = []
    entry: LexicalEntry
    for entry in lex.entries:
        written = {w.lower() for w in Lexicon.get_form_written_reps_of_entry(entry)}
        uris = {u.lower() for u in Lexicon.get_uris_of_entry(entry)}
        qnames = set()
        for u in Lexicon.get_uris_of_entry(entry):
            try:
                qnames.add(lex.nsmanager.qname(u).lower())
            except ValueError:
                pass
        calc_uris = set()
        for u in Lexicon.get_uris_of_entry(entry):
            try:
                calc_uris.add(lex.nsmanager.expand_curie(u).lower())
            except ValueError:
                pass
        if any([w in question.lower() for w in written]) and (
                any([u in query.lower() for u in uris]) or
                any([q in query.lower() for q in qnames]) or
                any([u in query.lower() for u in calc_uris])
        ):
            relevant.append(entry)
    return relevant

if __name__ == "__main__":
    #rel_entries = get_entries_for_question("What is the time zone of Salt Lake City?", "PREFIX res: <http://dbpedia.org/resource/> PREFIX dbp: <http://dbpedia.org/property/> SELECT DISTINCT ?uri WHERE { res:Salt_Lake_City <http://dbpedia.org/ontology/timeZone> ?uri }")
    rel_entries = get_entries_for_question("birth name", "dbr:Angela_Merkel dbo:birthName ?uri")
    rel_entries_str = {entry_to_str(e) for e in rel_entries}
    for entry in rel_entries_str:
        print(entry)

