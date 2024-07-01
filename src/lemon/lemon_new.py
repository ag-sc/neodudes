from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional

@dataclass
class CanonicalForm(object):
    # Vars with constant values
    uri: Optional[str] = None
    written_rep: Optional[str] = None
    # Vars with complex values
    verb_form_mood: Optional[VerbFormMood] = None
    # Vars with complex or constant values

    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class Condition(object):
    # Vars with constant values
    property_domain: Optional[str] = None
    property_range: Optional[str] = None
    uri: Optional[str] = None
    # Vars with complex values

    # Vars with complex or constant values

    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class DirectObject(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    marker: Optional[Marker] = None
    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class LexicalEntry(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    canonical_form: Optional[CanonicalForm] = None
    part_of_speech: Optional[PartOfSpeech] = None
    # Vars with complex or constant values

    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values
    other_form: list[Optional[OtherForm]] = field(default_factory=list)
    sense: list[Optional[Sense]] = field(default_factory=list)
    # Lists of vars with complex or constant values
    syn_behavior: list[Optional[Union[SynBehavior, str]]] = field(default_factory=list)                         


@dataclass
class Marker(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    canonical_form: Optional[CanonicalForm] = None
    part_of_speech: Optional[PartOfSpeech] = None
    # Vars with complex or constant values

    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class ObjOfProp(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    marker: Optional[Marker] = None
    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class OnProperty(object):
    # Vars with constant values
    type: Optional[str] = None
    # Vars with complex values

    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class OtherForm(object):
    # Vars with constant values
    uri: Optional[str] = None
    written_rep: Optional[str] = None
    # Vars with complex values
    number: Optional[Number] = None
    person: Optional[Person] = None
    tense: Optional[Tense] = None
    # Vars with complex or constant values

    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class PrepositionalAdjunct(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    marker: Optional[Marker] = None
    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class Reference(object):
    # Vars with constant values
    bound_to: Optional[str] = None
    has_value: Optional[str] = None
    label: Optional[str] = None
    uri: Optional[str] = None
    # Vars with complex values
    degree: Optional[Degree] = None
    # Vars with complex or constant values
    on_property: Optional[Union[OnProperty, str]] = None
    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class Sense(object):
    # Vars with constant values
    onto_mapping: Optional[str] = None
    uri: Optional[str] = None
    # Vars with complex values

    # Vars with complex or constant values
    condition: Optional[Union[Condition, str]] = None
    obj_of_prop: Optional[Union[ObjOfProp, str]] = None
    subj_of_prop: Optional[Union[SubjOfProp, str]] = None
    # Lists of vars with constant values
    is_a: list[Optional[str]] = field(default_factory=list)
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values
    reference: list[Optional[Union[Reference, str]]] = field(default_factory=list)                         


@dataclass
class SubjOfProp(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    marker: Optional[Marker] = None
    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class Subject(object):
    # Vars with constant values
    uri: Optional[str] = None
    # Vars with complex values
    marker: Optional[Marker] = None
    # Vars with complex or constant values

    # Lists of vars with constant values

    # Lists of vars with complex values

    # Lists of vars with complex or constant values



@dataclass
class SynBehavior(object):
    # Vars with constant values
    attributive_arg: Optional[str] = None
    copulative_arg: Optional[str] = None
    copulative_subject: Optional[str] = None
    uri: Optional[str] = None
    # Vars with complex values
    prepositional_adjunct: Optional[PrepositionalAdjunct] = None
    # Vars with complex or constant values
    direct_object: Optional[Union[DirectObject, str]] = None
    subject: Optional[Union[Subject, str]] = None
    # Lists of vars with constant values
    type: list[Optional[str]] = field(default_factory=list)
    # Lists of vars with complex values

    # Lists of vars with complex or constant values



class Degree(Enum):
    HIGH = 0
    LOW = 1
    STRONG = 2                        


class Number(Enum):
    PLURAL = 0
    SINGULAR = 1                        


class PartOfSpeech(Enum):
    ADJECTIVE = 0
    NOUN = 1
    PREPOSITION = 2
    VERB = 3                        


class Person(Enum):
    SECONDPERSON = 0
    THIRDPERSON = 1                        


class Tense(Enum):
    PAST = 0
    PERFECT = 1
    PRESENT = 2                        


class VerbFormMood(Enum):
    INFINITIVE = 0                        

LEMON = Union[LexicalEntry, CanonicalForm, SynBehavior, PartOfSpeech, OtherForm, Sense, VerbFormMood, PrepositionalAdjunct, DirectObject, Subject, Person, Number, Tense, Condition, Reference, ObjOfProp, SubjOfProp, Marker, Degree, OnProperty]