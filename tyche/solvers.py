from __future__ import annotations
from concurrent.futures import Future
import logging
from typing import List, Optional

from wolframclient.evaluation import WolframLanguageSession, WolframKernelEvaluationResult
from wolframclient.language import wlexpr

class TycheSolversException(Exception):
    """
    An exception type that is thrown when errors occur in
    the construction or use of equation solvers.
    """
    def __init__(self, message: str):
        self.message = "TycheSolversException: " + message

class TycheEquationSolver:
    def __enter__(self) -> TycheEquationSolver:
        return self
    
    def __exit__(self):
        raise NotImplementedError("__exit__ is unimplemented for " + type(self).__name__)
    
    def are_exprs_satisfiable_future(self, exprs: List[str], vars: List[str]) -> Future[bool]:
        """
        Should return True iff there exists some solution to
        the given expressions over the given variables.
        """
        raise NotImplementedError("__exit__ is unimplemented for " + type(self).__name__)
    
    def are_exprs_satisfiable(self, exprs: List[str], vars: List[str]) -> bool:
        return self.are_exprs_satisfiable_future(exprs, vars).result()

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
    ) -> None:
        super().__init__()

        self.session = session
        self.kernel_location = kernel_location
        self.connection_timeout_s = connection_timeout_s
        self.evaluation_timeout_s = evaluation_timeout_s
        self.kernel_loglevel = kernel_loglevel
    
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
    
    def are_exprs_satisfiable_future(self, exprs: List[str], vars: List[str]) -> Future[bool]:
        exprs_str = " && ".join([f"({expr})" for expr in exprs])
        vars_str = f'{{{", ".join([f"({var})" for var in vars])}}}'

        solver_in = wlexpr(f"Resolve[Exists[{vars_str}, {exprs_str}], Reals]")
        solver_future = self.get_mathematica_session().evaluate_wrap_future(solver_in, timeout=self.evaluation_timeout_s)

        future: Future[bool] = Future()
        def fn(finished_future: Future):
            solver_out: WolframKernelEvaluationResult = finished_future.result()
            solver_result = solver_out.result
            if not isinstance(solver_result, bool):
                raise TycheSolversException(
                    f"Error in {type(self).__name__}: Equation solver result in unknown format"
                )
            future.set_result(bool(solver_result))
        
        solver_future.add_done_callback(fn)
        return future
