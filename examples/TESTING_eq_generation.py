from tyche.individuals import Individual as Individual, TycheConceptField, TycheRuleField
from tyche.language import Concept, NoLikelierThan
from tyche.solvers import TycheMathematicaSolver

class A(Individual):
    a: TycheConceptField
    a_sq: TycheConceptField

    a_squared_eq_a_sq: TycheRuleField = (Concept("a") & Concept("a")).as_likely_as("a_sq")
    a_sq_lt_a: TycheRuleField = Concept("a_sq").no_likelier_than("a")

class B(A):
    a_lt_0_5: TycheRuleField = Concept("a").no_likelier_than(0.5)

class C(B):
    a_sq_gt_0_25: TycheRuleField = NoLikelierThan(0.25, "a_sq")

print(f"{A.__name__} rule names: " + str(Individual.get_rule_names(A)))
example_rule_name = "a_squared_eq_a_sq"
print(f"{A.__name__}'s value for rule '{example_rule_name}': " + str(Individual.get_class_rule(A, example_rule_name)))

print()

print(f"{B.__name__} rule names: " + str(Individual.get_rule_names(B)))

print()

print(f"Satisfiability equations for {A.__name__}:")
print(Individual.get_satisfiability_equations(A, simplify=False))
print()
print(f"Simplified satisfiability equations for {A.__name__}:")
print(Individual.get_satisfiability_equations(A, simplify=True))

print()

with Individual.set_solver(TycheMathematicaSolver()):
    print(f"Satisfiability of {A.__name__}: " + ("satisfiable" if Individual.are_rules_satisfiable(A) else "unsatisfiable"))
    print(f"Satisfiability of {B.__name__}: " + ("satisfiable" if Individual.are_rules_satisfiable(B) else "unsatisfiable"))
    print(f"Satisfiability of {C.__name__}: " + ("satisfiable" if Individual.are_rules_satisfiable(C) else "unsatisfiable"))
