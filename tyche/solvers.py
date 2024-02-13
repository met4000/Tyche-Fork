"""
This module provides solvers that are required
by individuals for certain calculations (e.g. satisfiability).
"""
from __future__ import annotations
from concurrent.futures import Future
from dataclasses import dataclass
import logging
from typing import Any, Callable, Literal, Optional, TypeVar

from sympy import Expr as SympyExpr
from sympy.parsing.mathematica import parse_mathematica
from wolframclient.evaluation import WolframLanguageSession, WolframKernelEvaluationResult
from wolframclient.language import wlexpr, WLInputExpression
from wolframclient.language.expression import WLSymbol


ResultType = TypeVar('ResultType')
OutputType = TypeVar('OutputType')

class TycheSolversException(Exception):
    """
    An exception type that is thrown when errors occur in
    the construction or use of equation solvers.
    """
    def __init__(self, message: str):
        self.message = "TycheSolversException: " + message

@dataclass
class Equations:
    """
    TODO description
    
    Represents a set of satisfiability equations, and the variables in them.
    """
    equations: list[str]
    variables: list[str]

    def __iter__(self):
        return iter((self.equations, self.variables))

class TycheEquationSolver:
    """
    An interface for an equation solver.

    Requires at least :meth:`~.example_exprs_solution_future` to be
    implemented, and :meth:`~.__exit__` to be overwritten.
    """

    def __enter__(self) -> TycheEquationSolver:
        return self
    
    def __exit__(self):
        raise NotImplementedError("__exit__ is unimplemented for " + type(self).__name__)
    
    
    def are_exprs_satisfiable_future(self, exprs: list[str], vars: list[str]) -> Future[bool | None]:
        """
        Returns None if the answer was not found (e.g. timeout, or not high enough precision),
        Otherwise, returns True if there exists some solution to the given expressions
        over the given variables, and False if no solution exists.

        Has a default implementation using :meth:`~.example_exprs_solution_future`.
        """
        solution_future = self.example_exprs_solution_future(exprs, vars)

        future: Future[bool] = Future()
        def fn(finished_future: Future[tuple[bool | None, Any]]):
            solution_out = finished_future.result()
            solution_satisfiable = solution_out[0]
            future.set_result(solution_satisfiable)
        
        solution_future.add_done_callback(fn)
        return future
    
    def are_exprs_satisfiable(self, exprs: list[str], vars: list[str]) -> bool | None:
        """
        Returns None if the answer was not found (e.g. timeout, or not high enough precision),
        Otherwise, returns True if there exists some solution to the given expressions
        over the given variables, and False if no solution exists.

        Has a default implementation using :meth:`~.are_exprs_satisfiable_future`.
        """
        return self.are_exprs_satisfiable_future(exprs, vars).result()
    
    
    def example_exprs_solution_future(self, exprs: list[str], vars: list[str]) -> Future[tuple[Literal[False] | None, None] | tuple[Literal[True], dict[str, SympyExpr]]]:
        """
        Returns two values; the first will be None if no solution was found (e.g. timeout, or not enough precision),
        False if no solution exists, and True if a solution was found.
        If a solution was found, the second value will be a solution to the given expressions over the given variables.
        """
        raise NotImplementedError("example_exprs_solution_future is unimplemented for " + type(self).__name__)
    
    def example_exprs_solution(self, exprs: list[str], vars: list[str]) -> tuple[Literal[False] | None, None] | tuple[Literal[True], dict[str, SympyExpr]]:
        """
        Returns two values; the first will be None if no solution was found (e.g. timeout, or not enough precision),
        False if no solution exists, and True if a solution was found.
        If a solution was found, the second value will be a solution to the given expressions over the given variables.

        Has a default implementation using :meth:`~.example_exprs_solution_future`.
        """
        return self.example_exprs_solution_future(exprs, vars).result()


class TycheMathematicaSolver(TycheEquationSolver):
    """
    TODO documentation

    During normal operation, the calculations of this solver
    may cause the Mathematica Kernel to log warnings using
    the :mod:`logging` module. These logs can be prevented
    with `logging.disable(logging.WARNING)`.
    
    See https://reference.wolfram.com/language/WolframClientForPython/docpages/advanced_usages.html#logging
    for more information about logging.
    """
    def __init__(self, *,
                 session: Optional[WolframLanguageSession] = None,
                 kernel_location: Optional[str] = None,
                 connection_timeout_s: Optional[float] = None,
                 evaluation_timeout_s: Optional[float] = None,
                 kernel_loglevel: int = logging.NOTSET,
                 logging_disable_warnings: bool = False
    ) -> None:
        super().__init__()

        self.session = session
        self.kernel_location = kernel_location
        self.connection_timeout_s = connection_timeout_s
        self.evaluation_timeout_s = evaluation_timeout_s
        self.kernel_loglevel = kernel_loglevel

        if logging_disable_warnings:
            logging.disable(logging.WARNING)
    
    def __enter__(self):
        self.restart_mathematica_session()
        return self

    def __exit__(self, type, value, traceback):
        return self.session.__exit__(type, value, traceback)

    def set_mathematica_settings(self, *,
                                 kernel_location: Optional[str] = None,
                                 connection_timeout_s: Optional[float] = None,
                                 kernel_loglevel: Optional[int] = None,
    ):
        """
        Updates the stored settings used for Mathematica.
        Does not restart the current Mathematica session;
        call :meth:`~.restart_mathematica_session` explicitly.
        """
        if kernel_location is not None:
            self.kernel_location = kernel_location

        if connection_timeout_s is not None:
            self.connection_timeout_s = connection_timeout_s

        if kernel_loglevel is not None:
            self.kernel_loglevel = kernel_loglevel

    def restart_mathematica_session(self):
        """
        Starts or re-starts the Mathematica session.
        """
        if self.session is not None:
            self.session.terminate()
        self.session = WolframLanguageSession(kernel=self.kernel_location, kernel_loglevel=self.kernel_loglevel)
        self.session.start(block=True, timeout=self.connection_timeout_s)

    def get_mathematica_session(self) -> WolframLanguageSession:
        if self.session is None or not self.session.started:
            self.restart_mathematica_session()
        return self.session
    
    def exec_mathematica_query(self, solver_in: WLInputExpression, result_type: type[ResultType], *, result_transformer: Callable[[ResultType], OutputType] = lambda v: v) -> Future[OutputType]:
        solver_future = self.get_mathematica_session().evaluate_wrap_future(solver_in, timeout=self.evaluation_timeout_s)

        future: Future[bool] = Future()
        def fn(finished_future: Future[WolframKernelEvaluationResult]):
            # ! TODO return `None` if timeout

            solver_out = finished_future.result()
            solver_result = solver_out.result
            if not isinstance(solver_result, result_type):
                raise TycheSolversException(
                    f"Error in {type(self).__name__}: Equation solver result in unknown format"
                )
            transformed_result = result_transformer(solver_result)
            future.set_result(transformed_result)
        
        solver_future.add_done_callback(fn)
        return future
    
    def are_exprs_satisfiable_future(self, exprs: list[str], vars: list[str]) -> Future[bool | None]:
        exprs_str = " && ".join([f"({expr})" for expr in exprs])
        vars_str = f'{{{", ".join([f"({var})" for var in vars])}}}'

        solver_in = wlexpr(f"Resolve[Exists[{vars_str}, {exprs_str}], Reals]")
        return self.exec_mathematica_query(solver_in, bool)
    
    def example_exprs_solution_future(self, exprs: list[str], vars: list[str]) -> Future[tuple[Literal[False] | None, None] | tuple[Literal[True], dict[str, SympyExpr]]]:
        exprs_str = " && ".join([f"({expr})" for expr in exprs])
        vars_str = f'{{{", ".join([f"({var})" for var in vars])}}}'

        solver_in = wlexpr(f"FindInstance[{exprs_str}, {vars_str}, Reals] /. (var_ -> value_) :> (var -> ToString[value, InputForm])")
        def fn(wrapped_result: tuple[tuple[WLSymbol, ...]]) -> tuple[Literal[False] | None, None] | tuple[True, dict[str, SympyExpr]]:
            # ! TODO handle returning `None, None`

            if not len(wrapped_result) > 0:
                return False, None
            
            rules = wrapped_result[0]

            vars_set = set(vars)
            solution: dict[str, SympyExpr] = {}
            for rule in rules:
                full_name = str(rule[0])
                if not isinstance(full_name, str):
                    raise TycheSolversException(
                        f"Error in {type(self).__name__}: Equation solution in unknown format; expected str, but found {type(full_name)}"
                    )
                name = full_name.removeprefix("Global`")

                value: SympyExpr = parse_mathematica(rule[1])
                if not isinstance(value, SympyExpr):
                    raise TycheSolversException(
                        f"Error in {type(self).__name__}: Equation solution in unknown format; expected Sympy Expr, but found {type(value)}"
                    )
                
                if name not in vars_set:
                    raise TycheSolversException(
                        f"Error in {type(self).__name__}: Equation solution contains unknown variable '{name}'"
                    )
                vars_set.remove(name)

                solution[name] = value
            
            if len(vars_set) > 0:
                raise TycheSolversException(
                    f"Error in {type(self).__name__}: Equation solution missing values for {len(vars_set)} variable(s)"
                )

            return True, solution

        return self.exec_mathematica_query(solver_in, tuple, result_transformer=fn)
