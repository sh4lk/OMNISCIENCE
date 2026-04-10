"""SageMath Bridge Module.

Provides an interface to SageMath for computations that SymPy cannot
handle efficiently:
  - Gröbner bases over large finite fields
  - Elliptic curve order computation (Schoof's algorithm)
  - Number field / ideal arithmetic
  - Advanced lattice operations (fplll integration)
  - Polynomial factoring over finite fields

Communication is via subprocess (SageMath runs as a separate process)
with JSON serialization for data exchange.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any

from omniscience.core.types import SolverResult, SolverStatus

log = logging.getLogger(__name__)


class SageBridge:
    """Interface to SageMath for advanced computations."""

    def __init__(self, sage_path: str | None = None):
        self._sage = sage_path or self._find_sage()
        self._available = self._sage is not None

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------ #
    #  Sage discovery                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_sage() -> str | None:
        """Locate SageMath binary."""
        candidates = [
            "sage",
            "/usr/bin/sage",
            "/usr/local/bin/sage",
            os.path.expanduser("~/sage/sage"),
            os.path.expanduser("~/SageMath/sage"),
            "/opt/sage/sage",
            "/Applications/Sage.app/Contents/Resources/sage/sage",
        ]
        for path in candidates:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    log.info("Found SageMath at %s: %s", path, result.stdout.strip())
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    # ------------------------------------------------------------------ #
    #  Core execution                                                     #
    # ------------------------------------------------------------------ #

    def execute(self, sage_code: str, timeout: float = 300.0) -> dict[str, Any]:
        """Execute SageMath code and return parsed JSON result.

        The sage_code must print a JSON object as its last output line.
        """
        if not self._available:
            return {"error": "SageMath not available"}

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sage", delete=False) as f:
            f.write(sage_code)
            script_path = f.name

        try:
            result = subprocess.run(
                [self._sage, script_path],
                capture_output=True, text=True,
                timeout=timeout,
            )
            os.unlink(script_path)

            if result.returncode != 0:
                log.warning("Sage script failed: %s", result.stderr[:500])
                return {"error": result.stderr[:500]}

            # Parse last JSON line from output
            lines = result.stdout.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") or line.startswith("["):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

            return {"error": "No JSON output from Sage", "stdout": result.stdout[:1000]}

        except subprocess.TimeoutExpired:
            os.unlink(script_path)
            return {"error": "Sage script timed out"}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  High-level operations                                              #
    # ------------------------------------------------------------------ #

    def groebner_basis(
        self, polynomials: list[str], variables: list[str],
        modulus: int, order: str = "lex",
    ) -> dict[str, Any]:
        """Compute Gröbner basis of a polynomial ideal over F_p."""
        var_str = ", ".join(variables)
        poly_str = ", ".join(polynomials)
        sage_code = f"""
import json
R.<{var_str}> = PolynomialRing(GF({modulus}), order='{order}')
I = ideal([{poly_str}])
G = I.groebner_basis()
basis_str = [str(g) for g in G]

# Try to extract solutions
try:
    V = I.variety()
    solutions = [{{str(k): int(v) for k, v in sol.items()}} for sol in V]
except:
    solutions = []

print(json.dumps({{"basis": basis_str, "solutions": solutions}}))
"""
        return self.execute(sage_code)

    def factor_integer(self, n: int) -> dict[str, Any]:
        """Factor an integer using Sage's best algorithms."""
        sage_code = f"""
import json
n = {n}
F = factor(n)
factors = [(int(p), int(e)) for p, e in F]
print(json.dumps({{"factors": factors, "n": {n}}}))
"""
        return self.execute(sage_code)

    def ec_order(self, a: int, b: int, p: int) -> dict[str, Any]:
        """Compute the order of E: y² = x³ + ax + b over F_p."""
        sage_code = f"""
import json
E = EllipticCurve(GF({p}), [{a}, {b}])
order = int(E.order())
group_structure = [int(x) for x in E.abelian_group().invariants()]
print(json.dumps({{"order": order, "group_structure": group_structure}}))
"""
        return self.execute(sage_code)

    def ec_discrete_log(
        self, a: int, b: int, p: int,
        Px: int, Py: int, Qx: int, Qy: int,
    ) -> dict[str, Any]:
        """Compute discrete log Q = [n]P on E(F_p)."""
        sage_code = f"""
import json
E = EllipticCurve(GF({p}), [{a}, {b}])
P = E({Px}, {Py})
Q = E({Qx}, {Qy})
try:
    n = int(P.discrete_log(Q))
    print(json.dumps({{"n": n, "success": True}}))
except:
    print(json.dumps({{"success": False, "error": "discrete_log failed"}}))
"""
        return self.execute(sage_code)

    def lll_reduce(self, matrix: list[list[int]]) -> dict[str, Any]:
        """LLL reduction via Sage's fplll backend."""
        matrix_str = str(matrix)
        sage_code = f"""
import json
M = matrix(ZZ, {matrix_str})
L = M.LLL()
result = [[int(x) for x in row] for row in L]
print(json.dumps({{"reduced": result}}))
"""
        return self.execute(sage_code)

    def solve_dlog(self, g: int, h: int, p: int) -> dict[str, Any]:
        """Discrete logarithm in Z_p* using Sage."""
        sage_code = f"""
import json
F = GF({p})
g = F({g})
h = F({h})
try:
    x = int(discrete_log(h, g))
    print(json.dumps({{"x": x, "success": True}}))
except:
    print(json.dumps({{"success": False}}))
"""
        return self.execute(sage_code)

    def coppersmith_small_roots(
        self, poly_str: str, modulus: int, beta: float = 1.0, epsilon: float = 0.05,
    ) -> dict[str, Any]:
        """Find small roots of a univariate polynomial mod N using Coppersmith."""
        sage_code = f"""
import json
N = {modulus}
P.<x> = PolynomialRing(Zmod(N))
f = {poly_str}
try:
    roots = f.small_roots(beta={beta}, epsilon={epsilon})
    result = [int(r) for r in roots]
    print(json.dumps({{"roots": result, "success": True}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
        return self.execute(sage_code)
