from __future__ import annotations

from collections import defaultdict
from queue import Queue
from typing import Set, Union, Dict, List, Optional

import z3  # type: ignore


class DRS(object):
    """Discourse Representation Structure (DRS)"""
    variables: Set[z3.ExprRef] = set()
    """Variables of the DRS, especially all variables used in conditions of the DRS."""
    conditions: Set[z3.ExprRef] = set()
    """Conditions of the DRS, e.g. expressions like x1 == "Gary_Marcus" or lexicon:old(x1) == True."""
    #declarations: Set[z3.ExprRef] = set()
    #"""Z3 function declarations used in conditions"""

    def __init__(self,
                 variables: Optional[Set[z3.ExprRef]] = None,
                 conditions: Optional[Set[z3.ExprRef]] = None) -> None:
        """
        Creates a Discourse Representation Structure (DRS).

        :param variables: Variables of the DRS, especially all variables used in conditions of the DRS.
        :type variables: Set[z3.ExprRef]
        :param conditions: Conditions of the DRS, e.g. expressions like x1 == "Gary_Marcus" or lexicon:old(x1) == True.
        :type conditions: Set[z3.ExprRef]
        """
        super().__init__()
        assert (variables is not None and all([z3.z3util.is_const(v) for v in variables])) or variables is None
        self.variables = variables if variables is not None else set()
        assert (conditions is not None
                and all([x in self.variables for c in conditions for x in z3.z3util.get_vars(c)])) or conditions is None
        self.conditions = conditions if conditions is not None else set()
        pvd: Dict[str, List[List[z3.ExprRef]]] = defaultdict(list)
        for cond in self.conditions:
            tres = DRS._get_pred_var_dict(cond)

            for k, v in tres.items():
                if k in pvd.keys():
                    pvd[k].extend(tres[k])
                else:
                    pvd[k] = tres[k]
        self.pred_var_dict = pvd
        #self.declarations = declarations if declarations is not None else set()

    @property
    def all_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DRS.
        :rtype: Set[z3.ExprRef]
        """
        # gen_all_vars = set(self.variables).union(*[set(z3.z3util.get_vars(x)) for x in self.conditions])
        # assert self.variables == gen_all_vars
        # if self.variables != gen_all_vars:
        #     print("Error: !!!!ALL VARS NOT EQUAL TO GEN ALL VARS!!!!")
        return self.variables

    @property
    def pred_var_dict_calculated(self) -> Dict[str, List[List[z3.ExprRef]]]:
        """
        Get a dictionary mapping from function names which occur in conditions to all different parameters they occur
        with in conditions. So::

            conditions = [
                lexicon:read(x1, x2) == True,
                lexicon:read(x3, x2) == True,
            ]

        (representation simplified) becomes::

            {
                "lexicon:read": [ [x1, x2], [x3, x2] ]
            }

        :return: Dictionary mapping from function names to all occurring parameter lists.
        :rtype: Dict[str, List[List[z3.ExprRef]]]
        """
        res: Dict[str, List[List[z3.ExprRef]]] = defaultdict(list)
        for cond in self.conditions:
            tres = DRS._get_pred_var_dict(cond)

            for k, v in tres.items():
                if k in res.keys():
                    res[k].extend(tres[k])
                else:
                    res[k] = tres[k]
        return res

    @staticmethod
    def _get_pred_var_dict(root: z3.ExprRef) -> Dict[str, List[List[z3.ExprRef]]]:
        q: Queue[z3.ExprRef] = Queue()
        q.put(root)

        res: Dict[str, List[List[z3.ExprRef]]] = defaultdict(list)

        while not q.empty():
            expr = q.get()

            if z3.is_app(expr) and expr.num_args() > 0:
                if expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    # if expr.decl().name() in res.keys():
                    #    print("Warning: {} is applied multiple times, old variables are overwritten!")
                    res[expr.decl().name()].append(expr.children())
                    # there should not be complex constructs in application -> other option would be to ignore those...
                    assert (set(expr.children()) == set(z3.z3util.get_vars(expr)).union(
                        [c for c in expr.children() if hasattr(c, "is_string_value") and c.is_string_value()]
                    ))
                else:
                    for c in expr.children():
                        q.put(c)

        return res

    @property
    def assigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DRS which are assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """
        vars = set()
        for cond in self.conditions:
            vars.update(self._eq_exprs(cond))
        return set([x for x in self.all_variables if x in vars])

    @property
    def unassigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DRS which are **not** assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """

        return set([x for x in self.all_variables if x not in self.assigned_variables])

    @staticmethod
    def _eq_exprs(root: z3.ExprRef) -> Set[z3.ExprRef]:
        q: Queue[z3.ExprRef] = Queue()
        q.put(root)

        res: Set[z3.ExprRef] = set()

        while not q.empty():
            expr = q.get()

            if z3.is_app(expr) and expr.num_args() > 0:
                if expr.decl().kind() == z3.Z3_OP_EQ:
                    # if expr.decl().name() in res.keys():
                    #    print("Warning: {} is applied multiple times, old variables are overwritten!")
                    res.update(expr.children())
                else:
                    for c in expr.children():
                        q.put(c)

        return res

    def replace(self, old: z3.ExprRef, new: z3.ExprRef) -> DRS:
        """
        Replace every occurrence of expression old by expression new. Can be used, e.g., for variable renaming.

        :param old: Expression to search for/to replace.
        :type old: z3.ExprRef
        :param new: Expression to replace occurrences of old with.
        :type new: z3.ExprRef
        :return: Returns changed DRS.
        :rtype: DRS
        """
        self.variables = {z3.substitute(x, (old, new)) for x in self.variables}
        self.conditions = {z3.substitute(x, (old, new)) for x in self.conditions}
        self.pred_var_dict = {k: [[z3.substitute(x, (old, new)) for x in vl] for vl in v] for k, v in self.pred_var_dict.items()}
        return self

    #
    def union(self, other: DRS) -> DRS:
        """
        Unify this object with another DRS or Duplex Condition.
        **Caution! The returned object is the unified one, not necessarily this object!**

        :param other: Other DRS or Duplex Condition to unify.
        :type other: CDRS
        :return: Unified DRS/Duplex Condition.
        :rtype: CDRS
        """
        if isinstance(other, DRS):
            self.variables.update(other.variables)
            self.conditions.update(other.conditions)
            for k, v in other.pred_var_dict.items():
                if k in self.pred_var_dict.keys():
                    self.pred_var_dict[k].extend(v)
                else:
                    self.pred_var_dict[k] = v
            return self
        else:
            return other.union(other=self)  # if other is duplex condition, unify self into duplex condition

    def __repr__(self):
        return "<DRS (variables: {{ {} }}, conditions: {{ {} }})>".format(", ".join([str(x) for x in self.variables]),
                                                                          ", ".join([str(x) for x in self.conditions]))

    def _old_str(self):
        return """<DRS 
                    variables: {{ {} }}, 
                    conditions: {{ {} }}
                    >
        """.format(", ".join([str(x) for x in self.variables]),
                   """
                        """.join([str(x) for x in self.conditions]))

    def __str__(self):
        return """[{}]
            {}
        """.format(", ".join([str(x) for x in self.variables]),
                   """
            """.join([str(x) for x in self.conditions]))

    # def __or__(self, other) -> Disjunction:
    #    return Disjunction(self, other)

    # def __invert__(self) -> Negation:
    #    return Negation(self)



