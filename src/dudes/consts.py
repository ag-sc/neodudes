import os

#spacy_model = "en_core_web_lg"
spacy_model = "en_core_web_trf" # only used in tests

fp_ratio = 10.0

#dbpedia_spotlight_endpoint = None
dbpedia_spotlight_endpoint = os.getenv('DBPEDIA_SPOTLIGHT_ENDPOINT', 'http://localhost:2222/rest')

#sparql_endpoint = "http://dbpedia.org/sparql"
sparql_endpoint = os.getenv('DBPEDIA_ENDPOINT', "http://localhost:8890/sparql")
#sparql_endpoint = http://client.linkeddatafragments.org/#datasources=http%3A%2F%2Ffragments.dbpedia.org%2F2016-04%2Fen

rpc_host = os.getenv('RPC_HOST', "localhost")
rpc_port = int(os.getenv('RPC_PORT', 8042))
rpc_threads = int(os.getenv('RPC_THREADS', 20))

question_words = {"who", "which", "when", "where", "what", "why", "whom", "whose", "how"}

some_relation_words = {"with", "have"}


count_keywords = [["how", "many"], ["how", "often"]]

ask_keywords = ["is", "are", "does", "do", "did", "was", "were", "has", "have", "had", "can", "could", "will", "would",
                "shall", "should", "may", "might", "must"]

comp_gt_keywords = ["more", "greater", "later"]
comp_gt_keywords_no_than = ["after"]
comp_lt_keywords = ["less", "fewer", "earlier"]
comp_lt_keywords_no_than = ["before"]

comp_keywords = comp_gt_keywords + comp_gt_keywords_no_than + comp_lt_keywords + comp_lt_keywords_no_than

top_strong_keywords = ["most", "first", "1st"]
top_weak_keywords = ["least", "last"]

conj_keywords = ["and", "or"]

than_keywords = ["than"]

pronouns = ["i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
            "that", "those", "these", "this", "they", "them", "their", "theirs", "who", "whom", "whose", "which",
            "what", "where", "when", "why", "how"]

special_words = set.union(*(
    question_words,
    some_relation_words,
    sum(count_keywords, []),
    ask_keywords,
    comp_gt_keywords,
    comp_lt_keywords,
    comp_gt_keywords_no_than,
    comp_lt_keywords_no_than,
    top_strong_keywords,
    top_weak_keywords,
    conj_keywords,
    than_keywords,
    pronouns
))

uri_attrs = ["reference", "bound_to", "has_value", "degree", "on_property", "property_domain", "property_range"]

#question_words.union(some_relation_words).union(["most", "least", "many"])
