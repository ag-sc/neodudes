from typing import Tuple, List

from rdflib import OWL
from rdflib.namespace import Namespace

default_namespaces: List[Tuple[str, Namespace]] = [
    ('lemon', Namespace('http://localhost:8000/lemon.owl#')),
    #('lemon', Namespace('http://localhost:8000/lemon.ttl')),
    #('lemon', Namespace('http://lemon-model.net/lemon')),
    ('owl', OWL),
    ('dbo', Namespace("http://dbpedia.org/ontology/")),
    ('dbp', Namespace("http://dbpedia.org/property/")),
    ('dbr', Namespace("http://dbpedia.org/resource/")),
    ('dbc', Namespace("http://dbpedia.org/resource/Category:")),
    ('dct', Namespace("http://purl.org/dc/terms/")),
    ('oils', Namespace("http://localhost:8000/oils.owl/")),
    ('local', Namespace('http://localhost:8000/#')),
    ('lexicon', Namespace('http://localhost:8000/lexicon#')),
    ('yago', Namespace('http://dbpedia.org/class/yago/')),
    ('yago-res', Namespace('https://yago-knowledge.org/resource/')),
]