from __future__ import annotations
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Literal, TypeVar

from sympy import Expr as SympyExpr


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
    
    def __exit__(self, type, value, traceback):
        raise NotImplementedError("__exit__ is unimplemented for " + type(self).__name__)
    

    @staticmethod
    def wrap_variable(var: str) -> str:
        """
        Passed in during equation generation and used to wrap variable names as
        a given solver may require (e.g. `Var["varname_with_special_chars!"]`).

        Defaults to no-op. Should be overwritten by a solver if required.
        """
        return var
    
    
    def are_exprs_satisfiable_future(self, exprs: list[str], vars: list[str]) -> Future[bool | None]:
        """
        Returns None if the answer was not found (e.g. timeout, or not high enough precision),
        Otherwise, returns True if there exists some solution to the given expressions
        over the given variables, and False if no solution exists.

        Has a default implementation using :meth:`~.example_exprs_solution_future`.
        """
        solution_future = self.example_exprs_solution_future(exprs, vars)

        future: Future[bool | None] = Future()
        def fn(finished_future: Future[tuple[Literal[False] | None, None] | tuple[Literal[True], dict[str, SympyExpr]]]):
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
