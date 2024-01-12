import logging
from tyche.individuals import Individual as Individual, TycheConceptField, TycheRuleField
from tyche.language import Concept, AsLikelyAs
from tyche.solvers import TycheMathematicaSolver

class A(Individual):
    p: TycheConceptField
    a: TycheConceptField
    b: TycheConceptField

    a_rule: TycheRuleField = Concept("a").as_likely_as(Concept("p") & (~Concept("p")))
    b_rule: TycheRuleField = Concept("b").as_likely_as(Concept("p") | (~Concept("p")))

class B(A):
    a_b_rule: TycheRuleField = AsLikelyAs("a", "b")

print(f"{A.__name__} rule names: " + str(Individual.get_rule_names(A)))

print()

print(f"{B.__name__} rule names: " + str(Individual.get_rule_names(B)))

print()

print(f"Satisfiability equations for {A.__name__}:")
print(Individual.get_satisfiability_equations(A, simplify=False))
print()
print(f"Simplified satisfiability equations for {A.__name__}:")
print(Individual.get_satisfiability_equations(A, simplify=True))

print()

print(f"Satisfiability equations for {B.__name__}:")
print(Individual.get_satisfiability_equations(B, simplify=False))
print()
print(f"Simplified satisfiability equations for {B.__name__}:")
print(Individual.get_satisfiability_equations(B, simplify=True))

print()

logging.disable(logging.WARNING)

with Individual.set_solver(TycheMathematicaSolver(evaluation_timeout_s=0)):
    print(f"Satisfiability of {A.__name__}: " + ("satisfiable" if Individual.are_rules_satisfiable(A) else "unsatisfiable"))
    print(f"Satisfiability of {B.__name__}: " + ("satisfiable" if Individual.are_rules_satisfiable(B) else "unsatisfiable"))

    print(f"Consistent example for {A.__name__}: " + str(Individual.get_consistent_example(A)))
    print(f"Consistent example for {B.__name__}: " + str(Individual.get_consistent_example(B)))
