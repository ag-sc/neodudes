from __future__ import annotations

import copy
from collections import defaultdict

from enum import Enum
from typing import Optional, Dict, List, Set, Union

import z3  # type: ignore

from dudes.drs import DRS
from dudes.dudes import DUDES, SelectionPair


class Quantifier(Enum):
    """Quantifier of Duplex Condition"""
    NOT = 1
    OR = 2
    ALL = 3
    NO = 4
    MOST = 5
    AND = 6


class DuplexCondition(object):
    """Duplex Condition"""
    restrictor: Optional[CDUDES]  # left-hand side
    """Restrictor/left-hand side of the Duplex Condition."""
    scope: Optional[CDUDES]  # right-hand side
    """Scope/right-hand side of the Duplex Condition."""
    quantifier: Optional[Quantifier]
    """Quantifier of the DuplexCondition, chosen from enum Quantifier."""
    variable: Optional[z3.ExprRef]
    """Quantified variable (if any)."""

    def __init__(self,
                 quantifier: Optional[Quantifier] = None,
                 variable: Optional[z3.ExprRef] = None,
                 restrictor: Optional[CDUDES] = None,
                 scope: Optional[CDUDES] = None) -> None:
        """
        Creates a raw Duplex Condition.

        :param quantifier: Quantifier of the DuplexCondition, chosen from enum Quantifier.
        :type quantifier: Quantifier
        :param variable: Quantified variable (if any).
        :type variable: Optional[z3.ExprRef]
        :param restrictor: Restrictor/left-hand side of the Duplex Condition.
        :type restrictor: Optional[CDRS]
        :param scope: Scope/right-hand side of the Duplex Condition.
        :type scope: Optional[CDRS]
        """
        super().__init__()
        #assert variable is None or z3.z3util.is_var(variable)
        self.quantifier = quantifier
        self.variable = variable
        self.restrictor = restrictor
        self.scope = scope

    @property
    def entities(self) -> Set[str]:
        """
        :return: All entities used in the Duplex Condition.
        :rtype: Set[str]
        """
        return set().union(*[x.entities for x in [self.restrictor, self.scope] if x is not None])

    @property
    def predicates(self) -> Set[str]:
        """
        :return: All properties used in the Duplex Condition.
        :rtype: Set[str]
        """
        return set().union(*[x.predicates for x in [self.restrictor, self.scope] if x is not None])

    def refresh_pred_var_dict(self):
        if self.restrictor is not None and isinstance(self.restrictor, DuplexCondition):
            self.restrictor.refresh_pred_var_dict()
        if self.scope is not None and isinstance(self.scope, DuplexCondition):
            self.scope.refresh_pred_var_dict()
        self.pred_var_dict = self.pred_var_dict_calculated

    @property
    def all_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the Duplex Condition.
        :rtype: Set[z3.ExprRef]
        """
        return set(
            [self.variable] if self.variable is not None else []
        ).union(*[x.all_variables for x in [self.restrictor, self.scope] if x is not None])

    @property
    def assigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DRS which are assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """
        return set().union(*[x.assigned_variables for x in [self.restrictor, self.scope] if x is not None])

    @property
    def unassigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DRS which are **not** assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """
        return set([x for x in self.all_variables if x not in self.assigned_variables])

    @property
    def pred_var_dict_calculated(self) -> Dict[str, List[List[z3.ExprRef]]]:
        """
        Get a dictionary mapping from function names which occur in conditions of scope or restrictor to all different
        parameters they occur with in conditions. So::

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
        for d in [x for x in [self.restrictor, self.scope] if x is not None]:
            tres = d.pred_var_dict

            for k, v in tres.items():
                if k in res.keys():
                    res[k].extend(tres[k])
                else:
                    res[k] = tres[k]
        return res

    @property
    def all_labels(self) -> Set[str]:
        res: Set[str] = set()
        if self.restrictor is not None:
            res = res.union(self.restrictor.all_labels)
        if self.scope is not None:
            res = res.union(self.scope.all_labels)
        return res

    def make_labels_disjoint_to(self, other: CDUDES) -> None:
        if self.restrictor is not None:
            self.restrictor.make_labels_disjoint_to(other=other)
        if self.scope is not None:
            self.scope.make_labels_disjoint_to(other=other)

    def make_vars_disjoint_to(self, other: CDUDES) -> None:
        if self.restrictor is not None:
            self.restrictor.make_vars_disjoint_to(other=other)
        if self.scope is not None:
            self.scope.make_vars_disjoint_to(other=other)

    def distinctify_and_unify_main_vars(self, new_main_var: Optional[str] = None):
        if self.restrictor is not None and self.scope is not None:
            self.restrictor.make_vars_disjoint_to(other=self.scope)
            self.restrictor.make_labels_disjoint_to(other=self.scope)

            if new_main_var is None:
                all_vars: Set[str] = set([str(x) for x in self.all_variables])
                idx = len(all_vars)
                while "x" + str(idx) in all_vars:
                    idx += 1
                new_main_var = "x" + str(idx)

            new_var = z3.Const(new_main_var, z3.StringSort())

            if isinstance(self.restrictor, DuplexCondition):
                self.restrictor.distinctify_and_unify_main_vars(new_main_var=new_main_var)
            elif isinstance(self.restrictor, DUDES):
                if self.restrictor.main_variable is not None:
                    self.restrictor = self.restrictor.replace(old=self.restrictor.main_variable, new=new_var)
                elif len(self.restrictor.selection_pairs) > 0:
                    self.restrictor = self.restrictor.replace(old=self.restrictor.selection_pairs[0].variable, new=new_var)
                else:
                    print("No variable to unify!")
            else:
                raise RuntimeError("Unknown type of restrictor!")

            if isinstance(self.scope, DuplexCondition):
                self.scope.distinctify_and_unify_main_vars()
            elif isinstance(self.scope, DUDES):
                if self.scope.main_variable is not None:
                    self.scope = self.scope.replace(old=self.scope.main_variable, new=new_var)
                elif len(self.scope.selection_pairs) > 0:
                    self.scope = self.scope.replace(old=self.scope.selection_pairs[0].variable, new=new_var)
                else:
                    print("No variable to unify!")
            else:
                raise RuntimeError("Unknown type of scope!")

            return new_var
    def replace(self, old: z3.ExprRef, new: z3.ExprRef) -> DuplexCondition:
        """
        Replace every occurrence of expression old by expression new. Can be used, e.g., for variable renaming.

        :param old: Expression to search for/to replace.
        :type old: z3.ExprRef
        :param new: Expression to replace occurrences of old with.
        :type new: z3.ExprRef
        :return: Returns changed DRS.
        :rtype: DuplexCondition
        """
        self.variable = z3.substitute(self.variable, (old, new))
        if self.restrictor is not None:
            self.restrictor.replace(old=old, new=new)
        if self.scope is not None:
            self.scope.replace(old=old, new=new)
        self.pred_var_dict = {k: [[z3.substitute(x, (old, new)) for x in vl] for vl in v] for k, v in self.pred_var_dict.items()}
        return self

    def union(self, other: CDUDES) -> CDUDES:
        """
        Unify this object with another DRS or Duplex Condition.
        **Caution! The returned object is the unified one, not necessarily this object!**

        :param other: Other DRS or Duplex Condition to unify.
        :type other: CDRS
        :return: Unified DRS/Duplex Condition.
        :rtype: CDRS
        """
        if isinstance(other, DUDES):
            if self.restrictor is not None:
                rother = copy.deepcopy(other)
                self.restrictor.union(other=rother)
            if self.scope is not None:
                sother = copy.deepcopy(other)
                self.scope.union(other=sother)

            for k, v in other.pred_var_dict.items():
                if k in self.pred_var_dict.keys():
                    self.pred_var_dict[k].extend(v)
                else:
                    self.pred_var_dict[k] = v

            return self
        else:
            raise RuntimeError("Union of two duplex conditions is undefined!")

    def merge(self, other: CDUDES, sp: Optional[SelectionPair] = None) -> CDUDES:
        assert sp in self.selection_pairs or sp in other.selection_pairs or sp is None
        assert isinstance(other, DUDES)
        if sp is None:
            if self.restrictor is not None:
                self.restrictor = self.restrictor.merge(other=copy.deepcopy(other), sp=copy.deepcopy(sp))
            if self.scope is not None:
                self.scope = self.scope.merge(other=copy.deepcopy(other), sp=copy.deepcopy(sp))
        elif sp in self.selection_pairs:
            if self.restrictor is not None and sp in self.restrictor.selection_pairs:
                if isinstance(self.restrictor, DuplexCondition):
                    self.restrictor = self.restrictor.merge(other=other, sp=sp)
                elif isinstance(self.restrictor, DUDES):
                    return self.apply_to(other=other, sp=sp)
                else:
                    raise RuntimeError("Unknown type of restrictor!")
            elif self.scope is not None and sp in self.scope.selection_pairs:
                if isinstance(self.scope, DuplexCondition):
                    self.scope = self.scope.merge(other=other, sp=sp)
                elif isinstance(self.scope, DUDES):
                    return self.apply_to(other=other, sp=sp)
                else:
                    raise RuntimeError("Unknown type of scope!")
        elif sp in other.selection_pairs:
            raise RuntimeError("Merging Duplex Condition into DUDES not supported yet")
        else:
            raise RuntimeError("Selection pair in neither of both DUDES and not none!")
        return self

    def merge_duplex(self, other: CDUDES, quantifier: Quantifier) -> DuplexCondition:
        return DuplexCondition(quantifier=quantifier, restrictor=self, scope=other)

    def apply_to(self, other: DUDES, sp: SelectionPair) -> CDUDES:
        assert sp in self.selection_pairs
        if self.restrictor is not None and sp in self.restrictor.selection_pairs:
            self.restrictor.apply_to(other=other, sp=sp)
        elif self.scope is not None and sp in self.scope.selection_pairs:
            self.scope.apply_to(other=other, sp=sp)
        else:
            raise RuntimeError("Selection pair in neither of both DUDES!")
        return self

    @property
    def selection_pairs(self) -> List[SelectionPair]:
        return ((self.restrictor.selection_pairs if self.restrictor is not None else []) +
                (self.scope.selection_pairs if self.scope is not None else []))

    def __repr__(self):
        return "<DuplexCondition [{}] {}{} [{}]>".format(
            str(self.restrictor) if self.restrictor is not None else "",
            self.quantifier.name,
            " " + str(self.variable) if self.variable is not None else "",
            str(self.scope) if self.scope is not None else ""
        )


class Negation(DuplexCondition):
    """'Negation' Duplex Condition"""
    def __init__(self, cdrs: DUDES) -> None:
        """
        Creates a Duplex Condition with quantifier NOT. CDRS us stored in restrictor variable.

        :param cdrs: CDRS which is to be negated by the resulting condition.
        :type cdrs: CDRS
        """
        super().__init__(Quantifier.NOT, restrictor=cdrs)

    def __repr__(self):
        return "<DuplexCondition NOT [{}]>".format(
            str(self.restrictor) if self.restrictor is not None else ""
        )


class Conjunction(DuplexCondition):
    """'Disjunction' Duplex Condition"""

    def __init__(self, left: DRS, right: DRS) -> None:
        """
        Creates a Duplex Condition with quantifier OR.

        :param left: Restrictor/left-hand side DRS.
        :type left: DRS
        :param right: Scope/right-hand side DRS.
        :type right: DRS
        """
        super().__init__(Quantifier.AND, restrictor=left, scope=right)

    def __repr__(self):
        return "<DuplexCondition [{}] AND [{}]>".format(
            str(self.restrictor) if self.restrictor is not None else "",
            str(self.scope) if self.scope is not None else ""
        )


class Disjunction(DuplexCondition):
    """'Disjunction' Duplex Condition"""

    def __init__(self, left: DRS, right: DRS) -> None:
        """
        Creates a Duplex Condition with quantifier OR.

        :param left: Restrictor/left-hand side DRS.
        :type left: DRS
        :param right: Scope/right-hand side DRS.
        :type right: DRS
        """
        super().__init__(Quantifier.OR, restrictor=left, scope=right)

    def __repr__(self):
        return "<DuplexCondition [{}] OR [{}]>".format(
            str(self.restrictor) if self.restrictor is not None else "",
            str(self.scope) if self.scope is not None else ""
        )


class All(DuplexCondition):
    """'All' Duplex Condition"""

    def __init__(self, left: DRS, variable: z3.ExprRef, right: DRS) -> None:
        """
        Creates a Duplex Condition with quantifier ALL.

        Example::

            [Player(y)] ALL y [live(y)]
            "All players live"

        Then::

            left = DRS with condition Player(y) == True
            variable = Z3 constant/"variable" y
            right = DRS with condition live(y) == True


        :param left: Restrictor/left-hand side DRS.
        :type left: DRS
        :param variable: Quantifier variable
        :type variable: z3.ExprRef
        :param right: Scope/right-hand side DRS.
        :type right: DRS
        """
        super().__init__(Quantifier.ALL, variable=variable, restrictor=left, scope=right)


class No(DuplexCondition):
    """'No' Duplex Condition"""

    def __init__(self, left: DRS, variable: z3.ExprRef, right: DRS) -> None:
        """
        Creates a Duplex Condition with quantifier NO.

        Example::

            [Player(y)] NO y [live(y)]
            "No players live"

        Then::

            left = DRS with condition Player(y) == True
            variable = Z3 constant/"variable" y
            right = DRS with condition live(y) == True


        :param left: Restrictor/left-hand side DRS.
        :type left: DRS
        :param variable: Quantifier variable
        :type variable: z3.ExprRef
        :param right: Scope/right-hand side DRS.
        :type right: DRS
        """
        super().__init__(Quantifier.NO, variable=variable, restrictor=left, scope=right)


class Most(DuplexCondition):
    """'Most' Duplex Condition"""

    def __init__(self, left: DRS, variable: z3.ExprRef, right: DRS) -> None:
        """
        Creates a Duplex Condition with quantifier MOST.

        Example::

            [Player(y)] MOST y [live(y)]
            "Most players live"

        Then::

            left = DRS with condition Player(y) == True
            variable = Z3 constant/"variable" y
            right = DRS with condition live(y) == True


        :param left: Restrictor/left-hand side DRS.
        :type left: DRS
        :param variable: Quantifier variable
        :type variable: z3.ExprRef
        :param right: Scope/right-hand side DRS.
        :type right: DRS
        """
        super().__init__(Quantifier.MOST, variable=variable, restrictor=left, scope=right)


CDRS = Union[DRS, DuplexCondition]
"""Complex Discourse Representation Structure type, union of DRS and Duplex Condition"""
CDUDES = Union[DUDES, DuplexCondition]
