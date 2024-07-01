import urllib
from typing import Optional, Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON

from dudes import consts


class SPARQLEndpoint:
    def __init__(self, endpoint: str = consts.sparql_endpoint, cache: Optional[Dict[str, Dict[str, Any]]] = None):
        self.endpoint = endpoint
        self.cache = cache
        endpoint_up = urllib.request.urlopen(self.endpoint).getcode()
        if endpoint_up != 200:
            raise RuntimeError(f"SPARQL endpoint {self.endpoint} is not up")
    def get_results_query(self, query: str):
        if self.cache is not None and self.endpoint in self.cache.keys() and query in self.cache[self.endpoint].keys():
            return self.cache[self.endpoint][query]

        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        ret = None
        # try:
        ret = sparql.queryAndConvert()
        if "results" in ret.keys() and "bindings" in ret["results"].keys():
            ret = ret["results"]["bindings"]
        elif "boolean" in ret.keys():
            ret = [ret["boolean"]]

        if self.cache is not None and ret is not None:
            if self.endpoint not in self.cache.keys():
                self.cache[self.endpoint] = dict()
            self.cache[self.endpoint][query] = ret

        # print(ret)

        if ret is None:
            ret = []

        # for r in ret["results"]["bindings"]:
        #    print(r)
        return ret
        # except Exception as e:
        #    print(e)