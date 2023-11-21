from __future__ import annotations
from math import isnan
from typing import Callable, Dict, Final, NewType, Optional, Type, TypeVar, Union, cast

from tyche.language import CompatibleWithADLNode, ADLNode, Atom, Concept, TycheContext as _TycheContext, TycheLanguageException
from tyche.individuals import Individual as _Individual, IndividualPropertyDecorator, SelfType_IndividualPropertyDecorator, TycheAccessorStore, TycheIndividualsException
from tyche.references import GuardedSymbolReference, SymbolReference

# ! additions via overrides to 'language'

class TycheContext(_TycheContext):
    def get_rule(self, symbol: str) -> RuleValue:
        raise NotImplementedError("get_rule is unimplemented for " + type(self).__name__)

# TODO CompatibleWithADLNode: type = NewType("CompatibleWithADLNode", Union['ADLNode', str]) # type: ignore


# ! additions via overrides to 'individuals'

class Individual(_Individual):
    rules: TycheAccessorStore

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.rules = TycheRuleDecorator.get(type(self))

    def eval_rule(self, rule: CompatibleWithRule) -> RuleValue:
        return Rule.cast(rule).direct_eval(self)
    
    def check_rule(self, rule: CompatibleWithRule, *, epsilon: Optional[float] = None) -> bool:
        """
        TODO description
        If not specified, uses the default epsilon value for the rule.
        `epsilon` should be a valid float between 0 and 1 (inclusive).
        """
        if epsilon is not None:
            epsilon = float(epsilon)
            if epsilon < 0 or epsilon > 1 or isnan(epsilon):
                raise ValueError(f"Expected epsilon to be between 0 and 1 (inclusive), but got '{epsilon}'")
        
        return self.eval_rule(rule).direct_eval(self, epsilon=epsilon)
    
    # ? should probably rename to 'is_consistent' or something
    def check_rules(self, *, epsilon: Optional[float | Dict[CompatibleWithRule, float | None]] = None) -> bool:
        """
        TODO description
        Runs `check_rule` on every rule.
        """
        def get_epsilon(rule: CompatibleWithRule) -> float:
            if isinstance(epsilon, Dict):
                return epsilon.get(rule)
            return epsilon
        
        for rule_symbol in self.rules.all_symbols:
            if not self.check_rule(rule_symbol, epsilon=get_epsilon(rule_symbol)):
                return False
            
        return True
    
    @staticmethod
    def describe(obj_type: Type['Individual']) -> str:
        """ Returns a string describing the concepts, roles, and rules of the given Individual type. """
        atoms = sorted(list(Individual.get_concept_names(obj_type)))
        roles = sorted(list(Individual.get_role_names(obj_type)))
        rules = sorted(list(Individual.get_rule_names(obj_type)))
        return f"{obj_type.__name__} {{atoms={atoms}, roles={roles}, rules={rules}}}"

    @staticmethod
    def get_rule_names(obj_type: Type['Individual']) -> set[str]:
        """ Returns all the rule names of the given Individual type. """
        return TycheRuleDecorator.get(obj_type).all_symbols

    @classmethod
    def coerce_rule_value(cls: type, value: any) -> RuleValue:
        if isinstance(value, RuleValue):
            return value

        raise TycheIndividualsException(
            f"Error in {cls.__name__}: Rule values must be of type "
            f"{type(RuleValue).__name__}, not {type(value).__name__}"
        )

    def get_rule(self, symbol: str) -> RuleValue:
        value = self.rules.get(self, symbol)
        return self.coerce_rule_value(value)

    def get_rule_reference(self, symbol: str) -> SymbolReference[RuleValue]:
        ref = self.rules.get_reference(symbol)
        coerced_ref = GuardedSymbolReference(ref, self.coerce_rule_value, self.coerce_rule_value)
        return coerced_ref.bake(self)

    def to_str(self, *, detail_lvl: int = 1, indent_lvl: int = 0):
        if detail_lvl <= 0:
            return self.name if self.name is not None else f"<{type(self).__name__}>"

        sub_detail_lvl = detail_lvl - 1

        # Key-values of concepts.
        concept_values = [f"{sym}={self.get_concept(sym):.3f}" for sym in self.concepts.all_symbols]
        concept_values.sort()

        # We don't want to list out the entirety of the roles.
        role_values = []
        for role_symbol in self.roles.all_symbols:
            role = self.get_role(role_symbol)
            role_values.append(f"{role_symbol}={role.to_str(detail_lvl=sub_detail_lvl, indent_lvl=indent_lvl)}")
        role_values.sort()

        # Key-values of rules.
        rule_values = []
        for rule_symbol in self.rules.all_symbols:
            rule = self.get_rule(rule_symbol)
            rule_values.append(f"{rule_symbol}={rule.to_str(detail_lvl=sub_detail_lvl, tyche_context=self)}")
        rule_values.sort()

        name = self.name if self.name is not None else ""
        key_values = ", ".join(concept_values + role_values + rule_values)
        return f"{name}({key_values})"


# ! additions to 'language'

CompatibleWithRule: type = NewType("CompatibleWithRule", Union['Rule', str]) # type: ignore

class RuleValue:
    """
    TODO description
    Base class for ADL T-Book rules, represented as an (in)equality between two ADLNodes.
    """

    # 1e-12 was chosen as values are (inclusively) between 0 and 1, and 0.1 + 0.2 gives an error of about 4e-17
    # ? should there be a non-global way to define this default value, e.g. instead per TycheContext
    DEFAULT_EPSILON = 1e-12

    def __init__(self, LHS: CompatibleWithADLNode, RHS: CompatibleWithADLNode, epsilon: float, relation_symbol: str) -> None:
        """
        `epsilon` should be a valid float between 0 and 1 (inclusive).
        """
        self.LHS = ADLNode.cast(LHS)
        self.RHS = ADLNode.cast(RHS)

        self.epsilon = float(epsilon)
        if self.epsilon < 0 or self.epsilon > 1 or isnan(self.epsilon):
            raise ValueError(f"Expected epsilon to be between 0 and 1 (inclusive), but got '{self.epsilon}'")
        
        self.relation_symbol = str(relation_symbol)

    def direct_eval(self, context: TycheContext, *, epsilon: Optional[float] = None) -> bool:
        raise NotImplementedError("direct_eval is unimplemented for " + type(self).__name__)
    
    def __str__(self) -> str:
        return self.to_str()
    
    # ? TODO indent level
    def to_str(self, *, detail_lvl: int = 1, tyche_context: Optional[TycheContext] = None) -> str:
        if detail_lvl <= 0:
            return f"<{type(self).__name__}>"

        lhs_str = str(self.LHS)
        rhs_str = str(self.RHS)

        if detail_lvl > 1 and tyche_context is not None:
            lhs_str = f"{tyche_context.eval(self.LHS):.3f} = {lhs_str}"
            rhs_str = f"{rhs_str} = {tyche_context.eval(self.RHS):.3f}"

        return f"({lhs_str} {self.relation_symbol} {rhs_str})"

class Rule:
    """
    TODO description
    Represents rules restricting possible values (ADL T-Book rules).
    """
    def __init__(self, symbol: str, *, special_symbol: bool = False) -> None:
        if not special_symbol:
            Concept.check_symbol(symbol, symbol_type_name=type(self).__name__)
        
        self.symbol = symbol

    @staticmethod
    def cast(rule: CompatibleWithRule) -> Rule:
        if isinstance(rule, Rule):
            return rule
        elif isinstance(rule, str):
            return Rule(rule)
        else:
            raise TycheLanguageException("Incompatible rule type {}".format(type(rule).__name__))

    def __str__(self):
        return self.symbol
    
    def __hash__(self) -> int:
        return str(self).__hash__()
    
    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.symbol == cast('Rule', other).symbol
    
    def direct_eval(self, context: TycheContext) -> RuleValue:
        return context.get_rule(self.symbol)
    
    # ? TODO eval_reference

class AsLikelyAs(RuleValue):
    def __init__(self, LHS: CompatibleWithADLNode, RHS: CompatibleWithADLNode,
                 *, epsilon: float = RuleValue.DEFAULT_EPSILON) -> None:
        super().__init__(LHS, RHS, epsilon, "\u2248") # ≈
    
    def direct_eval(self, context: TycheContext, *, epsilon: Optional[float] = None) -> bool:
        lhs_value = context.eval(self.LHS)
        rhs_value = context.eval(self.RHS)

        if epsilon is None:
            epsilon = self.epsilon
        
        return abs(lhs_value - rhs_value) < epsilon

class NoLikelierThan(RuleValue):
    def __init__(self, LHS: CompatibleWithADLNode, RHS: CompatibleWithADLNode,
                 *, epsilon: float = RuleValue.DEFAULT_EPSILON, free_variable: Optional[FreeVariable] = None) -> None:
        super().__init__(LHS, RHS, epsilon, "\u227C") # ≼

        if free_variable is None:
            free_variable = FreeVariable()
        self.free_variable = free_variable

    def direct_eval(self, context: TycheContext, *, epsilon: Optional[float] = None) -> bool:
        lhs_value = context.eval(self.LHS)
        rhs_value = context.eval(self.RHS)

        if epsilon is None:
            epsilon = self.epsilon
        
        return lhs_value < rhs_value + epsilon / 2
    
    # TODO override generation of equation to include the free variable

# * problematic with evaluating rules if allowed to be general - should only be used at
# * the top level of a rule, and the rule should have custom evaluation to avoid trying
# * to eval the free variable
class FreeVariable(Atom):
    global_variable_count = 0

    """
    TODO description
    Used by rules to represent `a <= b` as `a * _free_var = b` (i.e. as an equality).
    Cannot be evaluated.
    """
    def __init__(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            symbol = str(FreeVariable.global_variable_count)
            FreeVariable.global_variable_count += 1
        
        super().__init__(f"_free_variable${symbol}", special_symbol=True)

    def direct_eval(self, context: TycheContext) -> float:
        raise TycheLanguageException(f"Instances of {type(self).__name__} cannot be evaluated")

# ! additions to 'individuals'

TycheRuleValue = TypeVar("TycheRuleValue", bound=RuleValue)
TycheRuleField = TypeVar("TycheRuleField", bound=RuleValue)

class TycheRuleDecorator(IndividualPropertyDecorator[TycheRuleValue, None]):
    """
    Not strictly used, but defined to make implementing fields easier,
    and in case method support is added later.
    """
    field_type_hint: Final[type] = TycheRuleField

    def __init__(
            self: SelfType_IndividualPropertyDecorator,
            fn: Callable[[], TycheRuleValue],
            *, symbol: Optional[str] = None):

        super().__init__("rule", fn, symbol=symbol)

    @staticmethod
    def get(obj_type: Type['Individual']) -> TycheAccessorStore:
        return TycheAccessorStore.get_or_populate_for(
            obj_type, TycheRuleDecorator, TycheRuleDecorator.field_type_hint, "rule"
        )
