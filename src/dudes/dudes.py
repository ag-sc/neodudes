from __future__ import annotations

import logging
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, List, Optional, TYPE_CHECKING

from z3 import ModelRef  # type: ignore

from dudes.drs import DRS
import z3  # type: ignore

if TYPE_CHECKING:
    from dudes.duplex_condition import Quantifier, DuplexCondition, CDRS, CDUDES


class DUDES(object):
    """Dependency-based Underspeciﬁed Discourse Representation Structures (DUDES)"""
    main_variable: z3.ExprRef
    """Main variable of the DUDES."""
    main_label: str
    """Main label of the DUDES, determining main DRS."""
    drs: Dict[str, CDRS]
    """DRS dictionary, mapping from label to DRS object."""
    selection_pairs: List[SelectionPair]
    """Selection pairs of the DUDES."""
    sub_relations: Set[SubRelation]
    """Subordination relations of the DUDES, determining the order in which substitutions are allowed to be made. 
    Currently unused."""

    initial_pred_var_dict: Dict[str, List[List[z3.ExprRef]]]
    """Value of pred_var_dict at time of object creation. Useful to differentiate what has been added by merging and 
    what the functions of the original "atomic" DUDES were."""

    entities: Set[str]
    predicates: Set[str]

    _assigned_vars: Set[z3.ExprRef]
    _all_vars: Set[z3.ExprRef]



    def __init__(self,
                 main_variable: Optional[z3.ExprRef],
                 main_label: str,
                 drs: Dict[str, CDRS],
                 selection_pairs: List[SelectionPair],
                 sub_relations: Set[SubRelation],
                 entities: Optional[Set[str]] = None,
                 predicates: Optional[Set[str]] = None) -> None:
        """
        Creates a Dependency-based Underspeciﬁed Discourse Representation Structures (DUDES).

        :param main_variable: Main variable of the DUDES.
        :type main_variable: z3.ExprRef
        :param main_label: Main label of the DUDES, determining main DRS.
        :type main_label: str
        :param drs: DRS dictionary, mapping from label to DRS object.
        :type drs: Dict[str, CDRS]
        :param selection_pairs: Selection pairs of the DUDES.
        :type selection_pairs: List[SelectionPair]
        :param sub_relations: Subordination relations of the DUDES, determining the order in which substitutions are allowed to be made. Currently unused.
        :type sub_relations: Set[SubRelation]
        """
        super().__init__()
        assert main_variable is None or z3.z3util.is_const(main_variable)
        assert main_label in drs.keys()
        assert main_variable is None or main_variable in drs[main_label].all_variables  # alternatively in any drs?
        # TODO: asserts for selection pairs and sub rels
        self.main_variable = main_variable
        self.main_label = main_label
        self.drs = drs
        self.selection_pairs = selection_pairs
        self.sub_relations = sub_relations
        self.initial_pred_var_dict = self.pred_var_dict
        self.entities = entities if entities is not None else set()
        self.predicates = predicates if predicates is not None else set()

        self._all_vars = (set() if self.main_variable is None else {self.main_variable}).union(
            *[x.all_variables for x in self.drs.values()]
        ).union(
            *[x.all_variables for x in self.selection_pairs]
        )

        self._assigned_vars: Set[z3.ExprRef] = set()
        for d in self.drs.values():
            self._assigned_vars.update(d.assigned_variables)

    @property
    def all_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DUDES.
        :rtype: Set[z3.ExprRef]
        """
        # res = (set() if self.main_variable is None else {self.main_variable}).union(
        #     *[x.all_variables for x in self.drs.values()]
        # ).union(
        #     *[x.all_variables for x in self.selection_pairs]
        # )
        # assert self._all_vars == res
        # return res
        return self._all_vars

    @property
    def all_variable_names(self) -> Set[str]:
        """
        :return: All variable names used in the DUDES.
        :rtype: Set[str]
        """
        return set([str(x) for x in self.all_variables])

    @property
    def assigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DUDES which are assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """
        # res: Set[z3.ExprRef] = set()
        # for d in self.drs.values():
        #     res.update(d.assigned_variables)
        #
        # assert self._assigned_vars == res
        #
        # return res

        return self._assigned_vars

        # model = self.get_model()
        # assigned_var_names = set([x.name() for x in model.decls()])
        # return set([x for x in self.all_variables if str(x) in assigned_var_names])
        # [x.name() for x in model.decls() if x.name() in res_dudes.all_variable_names]

    @property
    def unassigned_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the DUDES which are **not** assigned a value with x == "...".
        :rtype: Set[z3.ExprRef]
        """

        return self.all_variables.difference(self.assigned_variables)

    @property
    def pred_var_dict(self) -> Dict[str, List[List[z3.ExprRef]]]:
        """
        Get a dictionary mapping from function names which occur in any DRS to all different parameters they occur
        with. So::

            {
                "l1": <DRS
                    variables = [x1, x2, x3]
                    conditions = [
                        lexicon:read(x1, x2) == True,
                        lexicon:read(x3, x2) == True,
                    ]
                >
            }

        (representation simplified) becomes::

            {
                "lexicon:read": [ [x1, x2], [x3, x2] ]
            }

        :return: Dictionary mapping from function names to all occurring parameter lists.
        :rtype: Dict[str, List[List[z3.ExprRef]]]
        """
        res: Dict[str, List[List[z3.ExprRef]]] = defaultdict(list)
        for d in self.drs.values():
            tres = d.pred_var_dict
            #assert tres == d._pvd

            for k, v in tres.items():
                if k in res.keys():
                    res[k].extend(tres[k])
                else:
                    res[k] = tres[k]
        return res

    @property
    def pred_var_dict_calculated(self) -> Dict[str, List[List[z3.ExprRef]]]:
        res: Dict[str, List[List[z3.ExprRef]]] = defaultdict(list)
        for d in self.drs.values():
            tres = d.pred_var_dict_calculated
            # assert tres == d._pvd

            for k, v in tres.items():
                if k in res.keys():
                    res[k].extend(tres[k])
                else:
                    res[k] = tres[k]
        return res

    @property
    def var_pred_dict(self) -> Dict[z3.ExprRef, Set[str]]:
        res: Dict[z3.ExprRef, Set[str]] = defaultdict(set)
        for d in self.drs.values():
            tres = d.pred_var_dict

            for k, vll in tres.items():
                for vl in vll:
                    for v in vl:
                        res[v].add(k)
        return res

    @property
    def var_pred_dict_calculated(self) -> Dict[z3.ExprRef, Set[str]]:
        res: Dict[z3.ExprRef, Set[str]] = defaultdict(set)
        for d in self.drs.values():
            tres = d.pred_var_dict_calculated

            for k, vll in tres.items():
                for vl in vll:
                    for v in vl:
                        res[v].add(k)
        return res

    @property
    def all_labels(self) -> Set[str]:
        """
        :return: All labels used in the DUDES.
        :rtype: Set[str]
        """
        return {self.main_label}.union(self.drs.keys()).union(
            [sp.label for sp in self.selection_pairs]
        ).union(
            [sr.left for sr in self.sub_relations if sr.left is not None]
        ).union(
            [sr.right for sr in self.sub_relations if sr.right is not None]
        )

    def replace(self, old: z3.ExprRef, new: z3.ExprRef) -> DUDES:
        """
        Replace every occurrence of expression old by expression new. Can be used, e.g., for variable renaming.

        :param old: Expression to search for/to replace.
        :type old: z3.ExprRef
        :param new: Expression to replace occurrences of old with.
        :type new: z3.ExprRef
        :return: Returns changed DUDES.
        :rtype: DUDES
        """
        if self.main_variable is not None:  # TODO: maybe just if is not None substitute else do nothing?
            self.main_variable = z3.substitute(self.main_variable, (old, new))
        for k, v in self.drs.items():
            self.drs[k] = v.replace(old, new)
        self.selection_pairs = [x.replace(old, new) for x in self.selection_pairs]

        if old in self._all_vars:
            self._all_vars.remove(old)
            self._all_vars.add(new)

        if old in self._assigned_vars:
            self._assigned_vars.remove(old)
            self._assigned_vars.add(new)

        return self

    def replace_label(self, old: str, new: str) -> DUDES:
        """
        Replace every occurrence of label old by label new. Can be used, e.g., for label renaming.

        :param old: Label to search for/to replace.
        :type old: str
        :param new: Label to replace occurrences of label old with.
        :type new: str
        :return: Returns changed DUDES.
        :rtype: DUDES
        """
        assert new not in self.drs.keys()
        self.main_label = new if self.main_label == old else self.main_label
        if old in self.drs.keys():
            temp = self.drs.get(old)
            self.drs.pop(old)
            if temp is not None:
                self.drs[new] = temp
        self.selection_pairs = [x.replace_label(old, new) for x in self.selection_pairs]
        self.sub_relations = {x.replace_label(old, new) for x in self.sub_relations}
        return self

    def make_labels_disjoint_to(self, other: CDUDES) -> None:
        """
        Make own labels disjoint to the labels of the given DUDES other, i.e., replace intersecting labels.

        :param other: Other DUDES to make labels disjoint to.
        :type other: DUDES
        """
        own_labels: Set[str] = set([str(x) for x in self.all_labels])
        other_labels: Set[str] = set([str(x) for x in other.all_labels])
        all_labels: Set[str] = own_labels.union(other_labels)
        intersect_labels: Set[str] = own_labels.intersection(other_labels)
        if len(intersect_labels) > 0:
            fresh_labels: list[str] = []
            idx = 0
            while len(fresh_labels) < len(intersect_labels):
                if "l" + str(idx) not in all_labels:
                    fresh_labels.append("l" + str(idx))
                idx += 1
            for old, new in zip(intersect_labels, fresh_labels):
                self.replace_label(old, new)

    def make_vars_disjoint_to(self, other: CDUDES) -> None:
        """
        Make own variables disjoint to the variables of the given DUDES other, i.e., replace intersecting variables.

        :param other: Other DUDES to make variables disjoint to.
        :type other: DUDES
        """
        own_vars: Set[str] = set([str(x) for x in self.all_variables])
        other_vars: Set[str] = set([str(x) for x in other.all_variables])
        all_vars: Set[str] = own_vars.union(other_vars)
        intersect_vars: Set[str] = own_vars.intersection(other_vars)
        if len(intersect_vars) > 0:
            fresh_vars: list[str] = []
            idx = 0
            while len(fresh_vars) < len(intersect_vars):
                if "x" + str(idx) not in all_vars:
                    fresh_vars.append("x" + str(idx))
                idx += 1
            for old, new in zip(intersect_vars, fresh_vars):
                self.replace(z3.String(old), z3.String(new))

    def merge(self, other: CDUDES, sp: Optional[SelectionPair] = None) -> CDUDES:
        """
        Merge this DUDES with other DUDES using given selection pair and return the resulting merged DUDES.
        Makes variables and labels disjoint before merging.
        **Caution! Either this DUDES or other is changed, so the only reliable way to get the result is
        to use returned value!**

        :param other: Other DUDES to merge with.
        :type other: DUDES
        :param sp: Selection pair to use for merge. Must be either part of this DUDES or the other (or None)!
        :type sp: Optional[SelectionPair]
        :return: Merged DUDES.
        :rtype: DUDES
        """
        assert sp in self.selection_pairs or sp in other.selection_pairs or sp is None
        if isinstance(other, DUDES):
            if sp is None:
                self.make_vars_disjoint_to(other=other)
                self.make_labels_disjoint_to(other=other)
                return self.union(other=other, unify=True)
            elif sp in self.selection_pairs:
                other.make_vars_disjoint_to(other=self)
                other.make_labels_disjoint_to(other=self)
                return self.apply_to(other=other, sp=sp)
            elif sp in other.selection_pairs:
                self.make_vars_disjoint_to(other=other)
                self.make_labels_disjoint_to(other=other)
                return other.apply_to(other=self, sp=sp)
            else:
                raise RuntimeError("Selection pair in neither of both DUDES and not none!")
        elif isinstance(other, DuplexCondition):
            return other.merge(other=self, sp=sp)
        else:
            raise RuntimeError("Unsupported type for other!")

        # other.replace(old=other.main_variable, new=sp.variable)
        # self.selection_pairs.discard(sp)
        # self.drs.update(other.drs) #TODO: ?? Join into main label somehow?
        # self.selection_pairs.update(other.selection_pairs) #TODO: necessary?
        # self.sub_relations.union(other.sub_relations) #TODO: necessary?

    def apply_to(self, other: CDUDES, sp: SelectionPair) -> CDUDES:
        """
        Merge other DUDES into this one using given selection pair. **Neither labels nor variables are made disjoint!**
        The changed DUDES is always this one and not other.

        :param other: Other DUDES to merge into this one.
        :type other: DUDES
        :param sp: Selection pair to use for merge. Must be part of this DUDES (and **not** None).
        :type sp: SelectionPair
        :return: Merged DUDES.
        :rtype: DUDES
        """
        assert other.main_variable is not None
        assert sp in self.selection_pairs
        self.selection_pairs.remove(sp)
        # if len(self.selection_pairs) > 0:
        #     self.initial_pred_var_dict = other.initial_pred_var_dict

        self.replace(old=sp.variable, new=other.main_variable)
        self.replace_label(old=sp.label, new=other.main_label)  # TODO: make labels disjoint too!
        return self.union(other=other, unify=False)

    def union(self, other: DUDES, unify: bool = True) -> DUDES:
        """
        Build union of this DUDES and DUDES other. Always changes this DUDES, not other.

        :param other: Other DUDES to build union with.
        :type other: DUDES
        :param unify: Replace main variable and label in other with main variable and label of this DUDES?
        :type unify: bool
        :return: Returns unified DUDES.
        :rtype: DUDES
        """
        if unify:
            other.replace_label(old=other.main_label, new=self.main_label)
            # if other.main_variable is not None and self.main_variable is not None: #TODO: probably causes more harm than good at the moment
            #     other.replace(old=other.main_variable, new=self.main_variable)

        intersec_labels = set(self.drs.keys()).intersection(other.drs.keys())
        for k, v in other.drs.items():
            if k in intersec_labels:
                self.drs[k] = self.drs[k].union(other=v)
            else:
                self.drs[k] = v

        self._all_vars.update(other._all_vars)
        self._assigned_vars.update(other._assigned_vars)

        self.selection_pairs.extend(other.selection_pairs)
        self.sub_relations.update(other.sub_relations)
        self.entities.update(other.entities)
        self.predicates.update(other.predicates)
        return self

    def get_model(self) -> ModelRef:
        solv = z3.Solver()
        form = self.formula_body
        # print(form)
        solv.add(form)
        if solv.check() == z3.sat:
            return solv.model()
        else:
            logging.error(f"Failed formula: {form}")
            raise RuntimeError("Formula could not be solved!")

    @property
    def formula_body(self) -> z3.ExprRef:
        expr: z3.ExprRef = z3.BoolVal(True)
        for l, drs in self.drs.items():
            if isinstance(drs, DRS):
                expr = z3.And(*([expr] + list(drs.conditions)))
            else:
                raise RuntimeError("Duplex conditions not supported yet!")

        return expr

    @property
    def formula(self) -> z3.ExprRef:
        expr: z3.ExprRef = self.formula_body

        for var in self.all_variables:
            expr = z3.Exists(var, expr)

        return expr

    @property
    def str_formula(self) -> str:

        and_conds: List[str] = []
        for l, drs in self.drs.items():
            if isinstance(drs, DRS):
                for cond in drs.conditions:
                    and_conds.append(str(cond))
            else:
                raise RuntimeError("Duplex conditions not supported yet!")

        expr: str = "∃" + ": ∃".join([str(var) for var in self.all_variables]) + ": " + " ∧ ".join(and_conds)

        return expr

    @classmethod
    def empty(cls, label: str = "l"):
        """
        Build empty DUDES.

        :return: Empty DUDES.
        :rtype: DUDES
        """
        drs = DRS(variables=set(), conditions=set())
        dudes = cls(main_variable=None,
                    main_label=label,
                    drs={label: drs},
                    selection_pairs=list(),
                    sub_relations=set())
        return dudes

    @classmethod
    def equality(cls, text: str, var: str = "x", label: str = "l"):
        """
        Build DUDES with DRS with single equality condition of the form var == "text".

        :param text: String value that will be used to build an equality condition with var.
        :type text: str
        :param var: (Main) variable name to also use for the DRS condition.
        :type var: str
        :param label: (Main) label to also use for created DRS.
        :type label: str
        :return: Equality DUDES.
        :rtype: DUDES
        """
        cond = z3.String(var) == z3.StringVal(text)
        drs = DRS(variables={z3.String(var)}, conditions={cond})
        dudes = cls(main_variable=z3.String(var),
                    main_label=label,
                    drs={label: drs},
                    selection_pairs=list(),
                    sub_relations=set(),
                    entities={text})
        return dudes

    @classmethod
    def unary(cls,
              rel_name: str,
              var: str = "x",
              label: str = "l"):
        """
        Build DUDES with DRS with single relation condition of the form text(var) == True.

        :param rel_name: Relation name that will be used to build a relation condition with var.
        :type rel_name: str
        :param var: (Main) variable name to also use for the DRS condition.
        :type var: str
        :param label: (Main) label to also use for created DRS.
        :type label: str
        :return: Unary DUDES.
        :rtype: DUDES
        """
        f = z3.Function(rel_name, z3.StringSort(), z3.BoolSort())
        x = z3.String(var)
        cond = (f(x) == True)
        drs1 = DRS(variables={x}, conditions={cond})
        dudes = cls(main_variable=x,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label)],
                    sub_relations=set(),
                    predicates={rel_name})
        return dudes

    @classmethod
    def binary(cls,
               rel_name: str,
               var_1: str = "x",
               var_2: str = "y",
               label: str = "l",
               main_var: Optional[str] = None,
               var_1_markers: Optional[List[str]] = None,
               var_2_markers: Optional[List[str]] = None,
               domain: Optional[str] = None,
               domain_var: Optional[str] = None,
               range: Optional[str] = None,
               range_var: Optional[str] = None):
        """
        Build DUDES with DRS with single relation condition of the form text(var_1, var_2) == True.

        :param rel_name: Relation name that will be used to build a relation condition with var_1 and var_2.
        :type rel_name: str
        :param var_1: (Main) variable name to also use for the DRS condition.
        :type var_1: str
        :param var_2: Additional variable name to use for the DRS condition.
        :type var_2: str
        :param label: (Main) label to also use for created DRS.
        :type label: str
        :param main_var: The main variable of the DUDES. Either (value of) var_1, var_2 or None.
        :type main_var: Optional[str]
        :return: Binary DUDES.
        :rtype: DUDES
        """
        # if main_var is None:
        #    main_var = var_1
        if main_var not in [var_1, var_2, None]:
            raise RuntimeError("Main variable is neither var_1 nor var_2 nor None!")
        preds = {rel_name}
        ents = set()

        f = z3.Function(rel_name, z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.String(var_1)
        y = z3.String(var_2)
        mv = None if main_var is None else z3.String(main_var)
        vars = {x, y}
        conds = set()
        cond = (f(x, y) == True)
        conds.add(cond)

        if domain is not None and domain_var is not None:
            d = z3.Function("rdf:type", z3.StringSort(), z3.StringSort(), z3.BoolSort())
            dv = z3.String(domain_var)
            cond2 = (d(x, dv) == True)
            cond2eq = (dv == z3.StringVal(domain))
            conds.add(cond2)
            conds.add(cond2eq)
            vars.add(dv)
            preds.add("rdf:type")
            ents.add(domain)

        if range is not None and range_var is not None:
            r = z3.Function("rdf:type", z3.StringSort(), z3.StringSort(), z3.BoolSort())
            rv = z3.String(range_var)
            cond3 = (r(y, rv) == True)
            cond3eq = (rv == z3.StringVal(range))
            conds.add(cond3)
            conds.add(cond3eq)
            vars.add(rv)
            preds.add("rdf:type")
            ents.add(range)

        drs1 = DRS(variables=vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label, markers=var_1_markers),
                                     SelectionPair(variable=y, label=label, markers=var_2_markers)],
                    sub_relations=set(),
                    predicates=preds,
                    entities=ents)
        return dudes

    @classmethod
    def binary_with_fixed_objs(cls,
                               rel_name: List[str],
                               obj_val: List[str],
                               var: str = "x",
                               var_obj: str = "y",
                               label: str = "l"):
        """
        Build DUDES with DRS with single relation condition of the form text(var_1, var_2) == True.

        :param rel_name: Relation name that will be used to build a relation condition with var_1 and var_2.
        :type rel_name: str
        :param obj_val: Value set for the object of the property/relation.
        :type obj_val: str
        :param var: (Main) variable name to also use for the DRS condition.
        :type var: str
        :param var_obj: Variable name to use for defining the value of the property object.
        :type var_obj: str
        :param label: (Main) label to also use for created DRS.
        :type label: str
        :return: Binary DUDES.
        :rtype: DUDES
        """
        x = z3.String(var)
        conds: Set[z3.ExprRef] = set()
        vars: Set[z3.ExprRef] = {x}

        for i in range(len(rel_name)):
            f = z3.Function(rel_name[i], z3.StringSort(), z3.StringSort(), z3.BoolSort())
            y = z3.String(var_obj+str(i))
            cond = (f(x, y) == True)
            cond2 = (y == z3.StringVal(obj_val[i]))
            conds.add(cond)
            conds.add(cond2)
            vars.add(y)
        drs1 = DRS(variables=vars, conditions=conds)
        dudes = cls(main_variable=x,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label)],
                    sub_relations=set(),
                    predicates=set(rel_name),
                    entities=set(obj_val))
        return dudes

    @classmethod
    def ternary(cls,
                rel_name: str,
                var_1: str = "x",
                var_2: str = "y",
                var_3: str = "z",
                label: str = "l",
                main_var: Optional[str] = None):
        """
        Build DUDES with DRS with single relation condition of the form text(var_1, var_2, var_3) == True.

        :param rel_name: Relation name that will be used to build a relation condition with var_1, var_2 and var_3.
        :type rel_name: str
        :param var_1: (Main) variable name to also use for the DRS condition.
        :type var_1: str
        :param var_2: Additional variable name to use for the DRS condition.
        :type var_2: str
        :param var_3: Additional variable name to use for the DRS condition.
        :type var_3: str
        :param label: (Main) label to also use for created DRS.
        :type label: str
        :param main_var: The main variable of the DUDES. Either (value of) var_1, var_2 or None.
        :type main_var: Optional[str]
        :return: Ternary DUDES.
        :rtype: DUDES
        """
        f = z3.Function(rel_name, z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.String(var_1)
        y = z3.String(var_2)
        z = z3.String(var_3)
        mv = None if main_var is None else z3.String(main_var)
        cond = (f(x, y, z) == True)
        drs1 = DRS(variables={x, y, z}, conditions={cond})
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label),
                                     SelectionPair(variable=y, label=label),
                                     SelectionPair(variable=z, label=label)],
                    sub_relations=set(),
                    predicates={rel_name})
        return dudes

    @classmethod
    def top_bound(cls,
                  order_val: str,
                  bound_to: str,
                  bound_to_var: str,
                  count: int = 1,
                  label: str = "l",
                  var: str = "x"):
        f = z3.Function("local:top", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.BoolSort())
        g = z3.Function(bound_to, z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.StringVal(str(count))
        y = z3.String(var)
        z = z3.StringVal(order_val)
        bv = z3.String(bound_to_var)
        mv = y
        cond = (f(x, bv, z) == True)
        cond2 = (g(y, bv) == True)
        conds = {cond, cond2}
        all_vars = {y, bv}

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=y, label=label), SelectionPair(variable=bv, label=label)],
                    sub_relations=set(),
                    predicates={bound_to})
        return dudes

    @classmethod
    def top(cls,
            order_val: str,
            count: int = 1,
            label: str = "l",
            var: str = "x"):
        f = z3.Function("local:top", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.StringVal(str(count))
        y = z3.String(var)
        z = z3.StringVal(order_val)
        mv = y
        cond = (f(x, y, z) == True)
        conds = {cond}
        all_vars = {y}

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=y, label=label)],
                    sub_relations=set())
        return dudes

    @classmethod
    def comp_bound(cls,
                   order_val: str,
                   bound_to: str,
                   bound_to_var: str,
                   bound_to_var2: Optional[str] = None,
                   num_var: str = "n",
                   label: str = "l",
                   var: str = "x",
                   bind_other: bool = False):
        f = z3.Function("local:comp", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.BoolSort())
        g = z3.Function(bound_to, z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.String(num_var)
        y = z3.String(var)
        z = z3.StringVal(order_val)
        bv = z3.String(bound_to_var)
        mv = y
        all_vars = {x, y, bv}
        sps = [SelectionPair(variable=x, label=label), SelectionPair(variable=y, label=label, markers=["than"]),
               SelectionPair(variable=bv, label=label)]

        conds = set()

        if bind_other and bound_to_var2 is None:
            cond2 = (g(x, bv) == True)
            conds.add(cond2)
            cond = (f(bv, y, z) == True)
            conds.add(cond)
        else:
            cond2 = (g(y, bv) == True)
            conds.add(cond2)

        if bound_to_var2 is not None:
            bv2 = z3.String(bound_to_var2)
            cond = (f(bv2, bv, z) == True)
            cond3 = (g(x, bv2) == True)
            all_vars.add(bv2)
            conds.add(cond)
            conds.add(cond3)
            sps.append(SelectionPair(variable=bv2, label=label))
        elif not bind_other:
            # sps[1] = SelectionPair(variable=y, label=label, markers=["than"])
            cond = (f(x, bv, z) == True)
            conds.add(cond)

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=sps,
                    sub_relations=set(),
                    predicates={bound_to})
        return dudes

    @classmethod
    def comp(cls,
             order_val: str,
             num_var: str = "n",
             label: str = "l",
             var: str = "x"):
        f = z3.Function("local:comp", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.String(num_var)
        y = z3.String(var)
        z = z3.StringVal(order_val)
        mv = y
        cond = (f(x, y, z) == True)
        conds = {cond}
        all_vars = {x, y}

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label),
                                     SelectionPair(variable=y, label=label, markers=["than"])],
                    sub_relations=set())
        return dudes

    @classmethod
    def countcomp_bound(cls,
                        order_val: str,
                        bound_to: str,
                        bound_to_var: str,
                        bound_to_var2: Optional[str] = None,
                        num_var: str = "n",
                        label: str = "l",
                        var: str = "x",
                        bind_other: bool = False):
        f = z3.Function("local:countcomp", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.StringSort(),
                        z3.BoolSort())
        g = z3.Function(bound_to, z3.StringSort(), z3.StringSort(), z3.BoolSort())
        x = z3.String(num_var)
        y = z3.String(var)
        z = z3.StringVal(order_val)
        bv = z3.String(bound_to_var)
        mv = y

        all_vars = {x, y, bv}
        sps = [SelectionPair(variable=x, label=label), SelectionPair(variable=y, label=label, markers=["than"]),
               SelectionPair(variable=bv, label=label)]

        conds = set()

        if bind_other and bound_to_var2 is None:
            cond2 = (g(x, bv) == True)
            conds.add(cond2)
            cond = (f(bv, y, x, z) == True)  # cond = (f(bv, y, z) == True)
            conds.add(cond)
        else:
            cond2 = (g(y, bv) == True)
            conds.add(cond2)

        if bound_to_var2 is not None:
            bv2 = z3.String(bound_to_var2)
            cond = (f(bv2, bv, y, z) == True)
            cond3 = (g(x, bv2) == True)
            all_vars.add(bv2)
            conds.add(cond)
            conds.add(cond3)
            sps.append(SelectionPair(variable=bv2, label=label))
        elif not bind_other:
            # sps[1] = SelectionPair(variable=y, label=label, markers=["than"])
            cond = (f(x, bv, y, z) == True)
            conds.add(cond)

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=sps,
                    sub_relations=set(),
                    predicates={bound_to})
        return dudes

    @classmethod
    def countcomp(cls,
                  order_val: str,
                  num_var: str = "n",
                  group_var: str = "g",
                  label: str = "l",
                  var: str = "x"):
        f = z3.Function("local:countcomp", z3.StringSort(), z3.StringSort(), z3.StringSort(), z3.StringSort(),
                        z3.BoolSort())
        x = z3.String(num_var)
        y = z3.String(var)
        g = z3.String(group_var)
        z = z3.StringVal(order_val)
        mv = y
        cond = (f(x, y, g, z) == True)
        conds = {cond}
        all_vars = {x, y, g}

        drs1 = DRS(variables=all_vars, conditions=conds)
        dudes = cls(main_variable=mv,
                    main_label=label,
                    drs={label: drs1},
                    selection_pairs=[SelectionPair(variable=x, label=label),
                                     SelectionPair(variable=y, label=label, markers=["than"]),
                                     SelectionPair(variable=g, label=label)],
                    sub_relations=set())
        return dudes

    def __repr__(self):
        return "<DUDES [main_var: {}, main_label: {}] drs: [{}], selection_pairs: [{}], sub_relations: [{}]>".format(
            str(self.main_variable),
            self.main_label,
            (", ".join(["{}: {}".format(k, str(v)) for k, v in self.drs.items()])),
            (", ".join([str(x) for x in self.selection_pairs])),
            (", ".join([str(x) for x in self.sub_relations])),
        )

    def __str__(self):
        return textwrap.dedent("""
            [{} | {}]
            {}
            ---- 
            {}
            ---- 
            {}
            --------
        """.format(
            str(self.main_variable),
            self.main_label,
            ("""
            """.join([
                """{}: {{
            {}    }}""".format(k, str(v)) for k, v in self.drs.items()])),
            ("""
            """.join([str(x) for x in self.selection_pairs])),
            ("""
            """.join([str(x) for x in self.sub_relations])),
        ))

    def _str_old(self):
        return textwrap.dedent("""
            <DUDES 
                [main_var: {}, main_label: {}] 
                drs: [
                    {}
                ], 
                selection_pairs: [
                    {}
                ], 
                sub_relations: [
                    {}
                ]
            >
        """.format(
            str(self.main_variable),
            self.main_label,
            ("""
                """.join([
                """{}:
                    {}""".format(k, str(v)) for k, v in self.drs.items()])),
            ("""
                    """.join([str(x) for x in self.selection_pairs])),
            ("""
                    """.join([str(x) for x in self.sub_relations])),
        ))


@dataclass(frozen=True)
class SelectionPair(object):
    """Selection pair used for merging DUDES in a specific way."""
    # tokenValidator: function[] #or z3 function?
    # allowed_pos: frozenset[
    #    str]  # TODO: exchange with dep_ -> subj obj is relevant, not POS! -> except adjectives, there POS more meaningful...
    # TODO: best compromise probably a set of tuples (pos_, dep_) with None for "don't care" in either position
    variable: z3.ExprRef
    """Specifies variable which is replaced by main variable of other DUDES on merge."""
    label: str
    """Specifies label which is replaced by main label of other DUDES on merge."""
    markers: Optional[List[str]] = None
    """List of markers that are used to identify the correct pobj for this selection pair."""

    @property
    def all_variables(self) -> Set[z3.ExprRef]:
        """
        :return: All variables used in the selection pair.
        :rtype: Set[z3.ExprRef]
        """
        return {self.variable}

    @property
    def lmarkers(self) -> List[str]:
        return [m.lower() for m in self.markers] if self.markers is not None else []

    def replace_label(self, old: str, new: str) -> SelectionPair:
        """
        Replace every occurrence of label old by label new. Can be used, e.g., for label renaming.

        :param old: Label to search for/to replace.
        :type old: str
        :param new: Label to replace occurrences of label old with.
        :type new: str
        :return: Returns changed SelectionPair.
        :rtype: SelectionPair
        """
        if old == self.label:
            return SelectionPair(variable=self.variable, label=new)
        else:
            return self

    def replace(self, old: z3.ExprRef, new: z3.ExprRef) -> SelectionPair:
        """
        Replace every occurrence of expression old by expression new. Can be used, e.g., for variable renaming.

        :param old: Expression to search for/to replace.
        :type old: z3.ExprRef
        :param new: Expression to replace occurrences of old with.
        :type new: z3.ExprRef
        :return: Returns changed SelectionPair.
        :rtype: SelectionPair
        """
        variable = z3.substitute(self.variable, (old, new))
        return SelectionPair(variable=variable, label=self.label)

    def __repr__(self):
        return "<SelectionPair (var: {}, label: {}{})>".format(  # token: {},
            # "{" + (", ".join([x for x in self.allowed_pos])) + "}",
            self.variable,
            self.label,
            "" if self.markers is None else ", markers: " + str(self.markers)
        )

    def __str__(self):
        return "({}, {}{})".format(  # token: {},
            # "{" + (", ".join([x for x in self.allowed_pos])) + "}",
            self.variable,
            self.label,
            "" if self.markers is None else ", " + str(self.markers)
        )


@dataclass(frozen=True)
class SubRelation(object):
    """Subordination relation defining the order in which merging DUDES is possible."""

    class Type(Enum):
        """Type of a subordination relation argument."""
        REGULAR = 0
        """A regular label used in the relation."""
        TOP = 1
        """The top element, i.e., the largest element of the order. No label stored in this case."""
        BOTTOM = 2
        """The bottom element, i.e., the smallest element of the order. No label stored in this case."""
        RESTRICTOR = 3
        """Referencing the restrictor of the Duplex Condition referred to by the label."""
        SCOPE = 4
        """Referencing the scope of the Duplex Condition referred to by the corresponding label."""

    left: Optional[str]
    """Left-hand side argument label of the relation."""
    right: Optional[str]
    """Right-hand side argument label of the relation."""
    leftType: Type = Type.REGULAR
    """Type of the left-hand side argument of the relation."""
    rightType: Type = Type.REGULAR
    """Type of the right-hand side argument of the relation."""

    def replace_label(self, old: str, new: str) -> SubRelation:
        """
        Replace every occurrence of label old by label new. Can be used, e.g., for label renaming.

        :param old: Label to search for/to replace.
        :type old: str
        :param new: Label to replace occurrences of label old with.
        :type new: str
        :return: Returns changed SubRelation.
        :rtype: SubRelation
        """
        if old == self.left or old == self.right:
            return SubRelation(left=new if old == self.left else self.left,
                               right=new if old == self.right else self.right,
                               leftType=self.leftType,
                               rightType=self.rightType)
        else:
            return self

    @staticmethod
    def _tostr(e: Optional[str], et: Type) -> str:
        match et:
            case et.REGULAR:
                return e
            case et.TOP:
                return "⊤"
            case et.BOTTOM:
                return "⊥"
            case et.RESTRICTOR:
                return "res(" + e + ")"
            case et.SCOPE:
                return "scope(" + e + ")"
            case _:
                raise RuntimeError("Unknown enum element!")

    def __repr__(self):
        return SubRelation._tostr(self.left, self.leftType) + " < " + SubRelation._tostr(self.right, self.rightType)

    def __str__(self):
        return self.__repr__()
