from z3 import *
import z3

def test_z3():
    x, y = Bools('x y')
    print(And(*[True, True, True, False]))
    expr = And(x, Or(y, And(x, y)))
    print(expr)
    expr2 = substitute(expr, (x, BoolVal(True)))
    print(expr2)
    print(simplify(expr2))
    print(z3util.get_vars(expr2))
    s = SetAdd(SetAdd(EmptySet(StringSort()), StringVal("Mary")),
               z3.StringVal("Paul"))  # "s", [z3.StringVal("Peter"), z3.StringVal("Paul"), z3.StringVal("Mary")]
    x = String("x")
    expr3 = And(IsMember(x, s), x != StringVal("Mary"))  # implicitely existentially quantified!
    solv = Solver()
    solv.add(expr3)
    res = solv.check()  # solve(expr3)
    print(res)
    print(solv.model())
    assert True == True


def test_z3_functions():
    solv = Solver()
    es, vals = EnumSort('Tokens', ('Peter', 'Paul', 'Mary'))
    f = Function("http://dbpedia.org/ontology/nationality", es, es, es, BoolSort())
    # f = Function("f", StringSort(), BoolSort())
    # solv.add(f)
    ex1 = f(vals[0], vals[1], vals[2]) == False
    solv.add(ex1)
    ex2 = f(vals[0], vals[1], vals[0]) == False
    solv.add(ex2)
    print(z3.is_app_of(ex1, z3.Z3_OP_AND))
    print(z3.is_app_of(ex1, z3.Z3_OP_EQ))
    print(ex1.children()[0].decl().name())
    print(f.kind() == z3.Z3_OP_UNINTERPRETED)
    # solv.add(f(vals[2]) == False)
    solv.add(f(Const("x", es)) == True)
    print(solv.check())
    print(solv.model())
