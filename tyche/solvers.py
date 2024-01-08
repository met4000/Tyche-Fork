from __future__ import annotations
from typing import List, Optional

from wolframclient.evaluation import WolframLanguageSession
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
    
    def are_exprs_satisfiable(self, exprs: List[str], vars: List[str]) -> bool:
        """
        Should return True iff there exists some solution to
        the given expressions over the given variables.
        """
        raise NotImplementedError("__exit__ is unimplemented for " + type(self).__name__)

# todo use 'future' versions for computations
class TycheMathematicaSolver(TycheEquationSolver):
    def __init__(self, *, session: Optional[WolframLanguageSession] = None, kernel_location: Optional[str] = None, connection_timeout_s: Optional[float] = 5.0) -> None:
        super().__init__()

        self.session = session
        self.kernel_location = kernel_location
        self.connection_timeout_s = connection_timeout_s
    
    def __enter__(self):
        self.restart_mathematica_session()
        return self

    def __exit__(self, type, value, traceback):
        self.session.__exit__(type, value, traceback)
        pass

    def set_mathematica_settings(self, *, kernel_location: Optional[str], connection_timeout_s: Optional[float]):
        """
        Updates the stored settings used for Mathematica.
        Does not restart the current Mathematica session;
        call :func:`~restart_mathematica_session` explicitly.
        """
        if kernel_location is not None:
            self.kernel_location = kernel_location

        if connection_timeout_s is not None:
            self.connection_timeout_s = connection_timeout_s

    def restart_mathematica_session(self):
        """
        Starts or re-starts the Mathematica session.
        """
        if self.session is not None:
            self.session.terminate()
        self.session = WolframLanguageSession(kernel=self.kernel_location)
        self.session.start(block=True, timeout=self.connection_timeout_s)

    def get_mathematica_session(self) -> WolframLanguageSession:
        if self.session is None or not self.session.ensure_started():
            self.restart_mathematica_session()
        return self.session
    
    def are_exprs_satisfiable(self, exprs: List[str], vars: List[str]) -> bool:
        exprs_str = " && ".join([f"({expr})" for expr in exprs])
        vars_str = f'{{{", ".join([f"({var})" for var in vars])}}}'

        solver_in = wlexpr(f"Resolve[Exists[{vars_str}, {exprs_str}], Reals]")
        solver_out = self.get_mathematica_session().evaluate(solver_in)
        if not isinstance(solver_out, bool):
            raise TycheSolversException(
                f"Error in {type(self).__name__}: Equation solver result in unknown format"
            )
        
        return solver_out
