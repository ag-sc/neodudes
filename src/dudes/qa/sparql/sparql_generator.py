import copy
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Iterable

from SPARQLBurger.SPARQLQueryBuilder import SPARQLSelectQuery, SPARQLGraphPattern
from SPARQLBurger.SPARQLSyntaxTerms import Prefix, Filter, GroupBy, Having, Triple

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.namespace import NamespaceManager, Namespace
from z3 import z3

from dudes import consts, utils
from dudes.consts import ask_keywords, count_keywords

from dudes.dudes import DUDES
from dudes.duplex_condition import DuplexCondition, CDUDES, Quantifier
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint


@dataclass
class OrderByData(object):
    var: str
    """Variable to order by, e.g. ?v42."""
    direction: str
    """Either ASC (ascending) or DESC (descending) depending on specified oils value."""
    limit: int
    """Specified LIMIT number."""

    @property
    def order_str(self):
        return "ORDER BY {}({}) LIMIT {}".format(self.direction, self.var, str(self.limit))


@dataclass
class FilterData(object):
    var: str
    """Variable to filter by, e.g. ?v42."""
    operator: str
    """Either > for Degree.STRONG or < for Degree.WEAK."""
    num: str
    """Number part of comparison."""
    bound: Optional[str] = None
    """Group by part of comparison if count == True."""
    count: bool = False

    @property
    def filter_str(self):
        if self.count:
            assert self.bound is not None
            return self.bound, f"COUNT({self.var}) {self.operator} {self.num}"
        else:
            return f"{self.var} {self.operator} {self.num}"


class SPARQLGenerator(ABC):
    endpoint: str
    cache: Optional[Dict[str, Dict[str, Any]]]

    # Alternative: http://client.linkeddatafragments.org/#datasources=http%3A%2F%2Ffragments.dbpedia.org%2F2016-04%2Fen
    def __init__(self, endpoint: str = "http://dbpedia.org/sparql", cache: Optional[Dict[str, Dict[str, Any]]] = None):
        #self.endpoint = endpoint
        #self.cache = cache
        self.sparql_endpoint = SPARQLEndpoint(endpoint=endpoint, cache=cache)

    @abstractmethod
    def to_sparql(self, query: str, dudes: DUDES, skip_unrecognized: bool = True, include_redundant: bool = False) -> \
    List[str]:
        pass

    def get_results_query(self, query: str):
        return self.sparql_endpoint.get_results_query(query)
        # if self.cache is not None and self.endpoint in self.cache.keys() and query in self.cache[self.endpoint].keys():
        #     return self.cache[self.endpoint][query]
        #
        # sparql = SPARQLWrapper(self.endpoint)
        # sparql.setQuery(query)
        # sparql.setReturnFormat(JSON)
        #
        # ret = None
        # # try:
        # ret = sparql.queryAndConvert()
        # if "results" in ret.keys() and "bindings" in ret["results"].keys():
        #     ret = ret["results"]["bindings"]
        # elif "boolean" in ret.keys():
        #     ret = [ret["boolean"]]
        #
        # if self.cache is not None and ret is not None:
        #     if self.endpoint not in self.cache.keys():
        #         self.cache[self.endpoint] = dict()
        #     self.cache[self.endpoint][query] = ret
        #
        # # print(ret)
        #
        # if ret is None:
        #     ret = []
        #
        # # for r in ret["results"]["bindings"]:
        # #    print(r)
        # return ret
        # # except Exception as e:
        # #    print(e)

    # def get_results_dudes(self, query: str, dudes: DUDES, skip_unrecognized=True):
    #     return [self.get_results_query(q)
    #             for q in self.to_sparql(query=query, dudes=dudes, skip_unrecognized=skip_unrecognized)]


class BasicSPARQLGenerator(SPARQLGenerator):

    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None,
                 endpoint: str = "http://dbpedia.org/sparql",
                 cache: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(endpoint=endpoint, cache=cache)
        # self.lexicon = lexicon
        self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=nsmanager)
        self.next_var_id = 0
        self.unicodere = re.compile(r'\\u\{(.*?)}')

    def _get_new_var(self):
        var = "?v" + str(self.next_var_id)
        self.next_var_id += 1
        return var

    @staticmethod
    def _unify_duplex(dudes: Optional[CDUDES]):
        if dudes is None:
            return None
        elif isinstance(dudes, DUDES):
            return dudes
        elif isinstance(dudes, DuplexCondition):
            dudes.refresh_pred_var_dict()
            dudes.distinctify_and_unify_main_vars()
            assert dudes.quantifier == Quantifier.AND
            restr = BasicSPARQLGenerator._unify_duplex(dudes.restrictor)
            scope = BasicSPARQLGenerator._unify_duplex(dudes.scope)
            if isinstance(restr, DUDES) and isinstance(scope, DUDES):
                return restr.union(scope)
            elif isinstance(restr, DUDES):
                return restr
            elif isinstance(scope, DUDES):
                return scope
            else:
                return None

    def to_sparql(self, query: str, dudes: CDUDES, skip_unrecognized=True, include_redundant: bool = False) -> List[str]:
        if isinstance(dudes, DuplexCondition):
            assert dudes.quantifier == Quantifier.AND
            dudes = self._unify_duplex(dudes)

        self.next_var_id = 0

        clean_query = re.sub(' +', ' ', query.lower().strip())
        is_ask_query = any([clean_query.startswith(ak) for ak in ask_keywords])
        if is_ask_query:
            include_redundant = True

        sparql_query = SPARQLSelectQuery(distinct=True)  # , limit=4267+1)
        # TODO: derive limit automatically from training data!

        for name, uri in self.nsmanager.namespaces():
            sparql_query.add_prefix(
                prefix=Prefix(prefix=name, namespace=str(uri))
            )

        var_map = dict()
        for v in dudes.all_variables:
            var_map[str(v)] = self._get_new_var()

        variables = [var_map[str(v)] for v in dudes.unassigned_variables]

        predicates = set(dudes.pred_var_dict.keys())

        found_wh_words = consts.question_words.intersection(predicates)
        if len(found_wh_words) > 1:
            logging.warning("Found more than one question word, adding all variables: " + str(found_wh_words))

        #TODO: reactivate, more sophisticated selection var determination
        #if len(found_wh_words) == 1:
        #    variables = [var_map[str(v)] for vl in dudes.pred_var_dict[list(found_wh_words)[0]] for v in vl]

        # Create a graph pattern to use for the WHERE part and add some triples
        where_pattern = SPARQLGraphPattern()
        triples, orderdata, filterdata = self._to_triples(dudes=dudes,
                                                          var_map=var_map,
                                                          skip_unrecognized=skip_unrecognized,
                                                          include_redundant=include_redundant)
        where_pattern.add_triples(
            triples=triples
        )



        existing_exprs = set([str(t.subject) for t in triples]
                             + [str(t.predicate) for t in triples]
                             + [str(t.object) for t in triples])

        select_vars = [v for v in variables if v in existing_exprs]

        # found_count_words = count_keywords.intersection(predicates)
        # if len(found_count_words) > 0:
        # if len(found_wh_words) == 1:
        #     variables = [var_map[str(v)] for vl in dudes.pred_var_dict[list(found_count_words)[0]] for v in vl]
        # else:
        if any([all([w in predicates for w in ck]) for ck in count_keywords]):
            select_vars = ["COUNT(DISTINCT {})".format(v) for v in select_vars]

        # sparql_query.add_variables(variables=select_vars)

        res_queries: List[str] = []

        orig_query = copy.deepcopy(sparql_query)
        orig_where_pattern = copy.deepcopy(where_pattern)

        if len(select_vars) == 0:
            select_vars = [select_vars]

        if len(orderdata) == 0:
            orderdata = [None]
        for ov in orderdata:
            for sv in select_vars:
                sparql_query = copy.deepcopy(orig_query)
                where_pattern = copy.deepcopy(orig_where_pattern)
                sparql_query.add_variables(sv if isinstance(sv, List) else [sv])

                havings = []

                for fd in filterdata:
                    if not fd.count:
                        where_pattern.add_filter(
                            Filter(expression=fd.filter_str)
                        )
                    else:
                        gb, fexpr = fd.filter_str
                        if len(select_vars) == 1:#TODO: always use sv here?
                            gb = select_vars[0]
                        sparql_query.add_group_by(GroupBy(variables=[gb]))
                        havings.append(Having(expression=fexpr))

                # Group the results by age
                # sparql_query.add_group_by(
                #     group=GroupBy(
                #         variables=["?age"]
                #     )
                # )

                # print(sparql_query.get_text())

                # Set this graph pattern to the WHERE part
                sparql_query.set_where_pattern(graph_pattern=where_pattern)

                res_sparql = sparql_query.get_text()
                res_sparql += " " + " ".join([h.get_text() for h in havings])
                res_sparql += (" " + ov.order_str if ov is not None else "")
                res_sparql = re.sub(r"SELECT .*?WHERE", "ASK WHERE", res_sparql,
                                    flags=re.DOTALL) if is_ask_query else res_sparql

                res_queries.append(res_sparql)
        return res_queries

    def _fix_unicode(self, val: str) -> str:
        unicodes = set(self.unicodere.findall(val))

        for u in unicodes:
            val = val.replace("\\u{" + u + "}", chr(int(u, 16)))

        return val

    def _to_triples(self,
                    dudes: DUDES,
                    var_map: Dict[str, str],
                    skip_unrecognized: bool = True,
                    include_redundant: bool = False) -> Tuple[List[Triple], List[OrderByData], List[FilterData]]:
        pvd = dudes.pred_var_dict
        # vpd = dudes.var_pred_dict
        model = dudes.get_model()
        triples = []
        orderdata = []
        filterdata = []
        unassigned_variables = dudes.unassigned_variables

        def var_or_value(n: z3.ExprRef) -> str:
            if n in unassigned_variables:
                return var_map[str(n)]
            else:
                val = self._fix_unicode(str(model[n]))

                if ":" in val:
                    nsval = utils.rem_quotes(val)
                    try:
                        if "ns1" in nsval:
                            logging.warning("ns1 in value: " + nsval)
                        return "<{}>".format(self.nsmanager.expand_curie(nsval) if not nsval.startswith("http")
                                             else nsval)
                    except ValueError:
                        return nsval
                else:
                    return val
                # TODO: better heuristic for "not a recognized entity"?

        for pred, vars in pvd.items():
            if ":" not in pred and skip_unrecognized:  # TODO: better heuristic for "not a recognized entity"?
                continue

            vars = [list(x) for x in set(tuple(x) for x in vars)]
            pred = self._fix_unicode(pred)

            for var_order in vars:
                if pred.lower() == "local:top":
                    #assert orderdata is None
                    assert len(var_order) == 3
                    direction: Optional[str] = None
                    match str(var_order[2]):
                        case '"Degree.HIGH"':
                            direction = "DESC"
                        case '"Degree.STRONG"':
                            direction = "DESC"
                        case '"Degree.LOW"':
                            direction = "ASC"
                        case '"Degree.WEAK"':
                            direction = "ASC"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[2]))

                    assert direction is not None
                    orderdata.append(OrderByData(
                        var=var_or_value(var_order[1]),
                        limit=int(utils.rem_quotes(str(var_order[0]), all_border_quotes=True)),
                        direction=direction))
                elif pred.lower() == "local:comp":  # TODO: Switch var0 and var1 if var0 is number literal etc.
                    assert len(var_order) == 3
                    operator: Optional[str] = None
                    match str(var_order[2]):
                        case '"Degree.HIGH"':
                            operator = ">"
                        case '"Degree.STRONG"':
                            operator = ">"
                        case '"Degree.LOW"':
                            operator = "<"
                        case '"Degree.WEAK"':
                            operator = "<"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[2]))

                    var1 = var_or_value(var_order[0])
                    var2 = var_or_value(var_order[1])

                    if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                            and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                        tvar = var2
                        var2 = var1
                        var1 = tvar
                        # operator = "<" if operator == ">" else ">"

                    assert operator is not None
                    filterdata.append(FilterData(
                        var=var1,
                        operator=operator,
                        num=utils.rem_quotes(var2, all_border_quotes=True),
                        count=False,
                    ))
                elif pred.lower() == "local:countcomp":
                    assert len(var_order) == 4
                    operator: Optional[str] = None
                    match str(var_order[3]):
                        case '"Degree.HIGH"':
                            operator = ">"
                        case '"Degree.STRONG"':
                            operator = ">"
                        case '"Degree.LOW"':
                            operator = "<"
                        case '"Degree.WEAK"':
                            operator = "<"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[3]))

                    var1 = var_or_value(var_order[0])
                    var2 = var_or_value(var_order[1])
                    var3 = var_or_value(var_order[2])

                    if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                            and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                        tvar = var2
                        var2 = var1
                        var1 = tvar
                        # operator = "<" if operator == ">" else ">"

                    assert operator is not None
                    filterdata.append(FilterData(
                        var=var1,
                        operator=operator,
                        num=utils.rem_quotes(var2, all_border_quotes=True),
                        count=True,
                        bound=var3
                    ))
                elif pred.lower() == "local:with":  # self._get_new_var()
                    triples.append(  # alternative: rdf:type dbo:class
                        Triple(subject=var_or_value(var_order[0]),
                               predicate=self._get_new_var(),
                               object=var_or_value(var_order[1]))
                    )
                elif pred.lower() == "local:rwith":  # self._get_new_var()
                    triples.append(  # alternative: rdf:type dbo:class
                        Triple(subject=var_or_value(var_order[1]),
                               predicate=self._get_new_var(),
                               object=var_or_value(var_order[0]))
                    )
                elif len(var_order) == 1:
                    if var_order[0] in dudes.unassigned_variables or include_redundant:
                        triples.append(  # alternative: rdf:type dbo:class
                            Triple(subject=var_or_value(var_order[0]), predicate="rdf:type", object=pred)
                        )
                elif len(var_order) == 2:
                    unassigned = [v for v in var_order if v in dudes.unassigned_variables]
                    # used = [v for v in var_order if v in dudes.assigned_variables or len(vpd[v]) > 1]
                    if len(unassigned) > 0 or include_redundant:  # and len(used) > 0:
                        triples.append(
                            Triple(subject=var_or_value(var_order[0]),
                                   predicate="<{}>".format(
                                       self.nsmanager.expand_curie(pred) if not pred.startswith("http")
                                       else pred
                                   ),
                                   object=var_or_value(var_order[1]))
                        )
                else:
                    logging.warning("Invalid number of arguments, skipping {}{}".format(pred, str(var_order)))

        vpd = defaultdict(set)
        for t in triples:
            if "?" in t.subject:
                vpd[t.subject].add(t.predicate)
            if "?" in t.object:
                vpd[t.object].add(t.predicate)

        res_triples = []

        for t in triples:
            # Either values/entities (i.e., no ?var) or at least one variable is further restricted by another triple
            # or redundant triples are included
            if ("?" not in t.subject
                    or "?" not in t.object
                    or len(vpd[t.subject]) > 1
                    or len(vpd[t.object]) > 1
                    or include_redundant):
                res_triples.append(t)

        return res_triples, orderdata, filterdata
