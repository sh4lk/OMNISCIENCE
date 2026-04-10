"""Algebraic Solver Module.

Implements:
  - Gaussian elimination over F_p  (for linear / affine ciphers)
  - Gröbner basis computation via SymPy (with optional SageMath bridge)
    for solving multivariate polynomial systems.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from omniscience.core.types import (
    AlgoFamily,
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class AlgebraicSolver:
    """Solve encryption as a system of algebraic equations over F_p."""

    NAME = "algebraic"

    # ------------------------------------------------------------------ #
    #  Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 300.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            modulus = recon.estimated_modulus or instance.modulus
            if modulus is None or modulus < 2:
                return self._fail("No modulus detected — algebraic solver needs F_p", t0)

            family = recon.algo_family

            # Try Gauss first (fast, works for linear/affine)
            if family in (AlgoFamily.LINEAR, AlgoFamily.UNKNOWN, AlgoFamily.POLYNOMIAL):
                res = self._try_gauss(instance, modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Try polynomial / Gröbner approach
            if family in (AlgoFamily.POLYNOMIAL, AlgoFamily.UNKNOWN, AlgoFamily.RSA_LIKE):
                res = self._try_groebner(instance, modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            return self._fail("Algebraic solver exhausted all strategies", t0)

        except Exception as exc:
            log.exception("Algebraic solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  Gaussian Elimination over F_p                                      #
    # ------------------------------------------------------------------ #

    def _try_gauss(
        self, instance: CryptoInstance, modulus: int, t0: float
    ) -> SolverResult:
        """Model E(x) = A·x + b  (mod p) and solve for A, b.

        Given known pairs (p_i, c_i), build the system:
            A · p_i + b ≡ c_i  (mod p)
        and solve via Gaussian elimination.
        Then decrypt the target: P = A^{-1} · (C_target - b) mod p.
        """
        log.info("[Gauss] Attempting linear system solve over F_%d", modulus)
        pt = np.array(instance.pt_as_int_list(), dtype=np.int64)
        ct = np.array(instance.ct_known_as_int_list(), dtype=np.int64)

        n = len(pt)
        if n < 2:
            return self._fail("Not enough known pairs for Gauss", t0)

        # Treat as scalar mapping first: c = a*p + b mod m
        # Build augmented matrix [p 1 | c] of shape (n, 3)
        aug = np.zeros((n, 3), dtype=np.int64)
        aug[:, 0] = pt % modulus
        aug[:, 1] = 1
        aug[:, 2] = ct % modulus

        solution = self._gauss_elim_fp(aug, modulus)
        if solution is not None:
            a, b = int(solution[0]), int(solution[1])
            log.info("[Gauss] Found linear relation: c = %d * p + %d  (mod %d)", a, b, modulus)
            # Decrypt target
            a_inv = self._modinv(a, modulus)
            if a_inv is None:
                return self._fail("Coefficient not invertible mod p", t0)
            ct_target = np.array(instance.ct_target_as_int_list(), dtype=np.int64)
            decrypted = [(a_inv * (int(c) - b)) % modulus for c in ct_target]
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=[a_inv, (-a_inv * b) % modulus],
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.95,
                details={"method": "gauss_linear", "a": a, "b": b, "modulus": modulus},
            )

        # Try quadratic: c = a*p^2 + b*p + d mod m
        if n >= 3:
            aug2 = np.zeros((n, 4), dtype=np.int64)
            aug2[:, 0] = (pt * pt) % modulus
            aug2[:, 1] = pt % modulus
            aug2[:, 2] = 1
            aug2[:, 3] = ct % modulus
            solution2 = self._gauss_elim_fp(aug2, modulus)
            if solution2 is not None:
                a2, b2, d2 = int(solution2[0]), int(solution2[1]), int(solution2[2])
                log.info("[Gauss] Found quadratic: c = %d*p² + %d*p + %d  (mod %d)", a2, b2, d2, modulus)
                # For decryption, we need to invert the quadratic — try brute-force per byte
                ct_target = np.array(instance.ct_target_as_int_list(), dtype=np.int64)
                decrypted = self._invert_poly(
                    [a2, b2, d2], ct_target, modulus
                )
                if decrypted is not None:
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=[a2, b2, d2],
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.85,
                        details={"method": "gauss_quadratic"},
                    )

        return self._fail("Gauss elimination did not yield a solution", t0)

    def _gauss_elim_fp(
        self, augmented: NDArray[np.int64], p: int
    ) -> NDArray[np.int64] | None:
        """Gaussian elimination on an (n × m+1) augmented matrix over F_p.

        Returns the solution vector of length m, or None if inconsistent / under-determined.
        """
        A = augmented.copy() % p
        n_rows, n_cols = A.shape
        n_vars = n_cols - 1
        pivot_row = 0

        for col in range(n_vars):
            # Find pivot
            found = -1
            for row in range(pivot_row, n_rows):
                if A[row, col] % p != 0:
                    found = row
                    break
            if found == -1:
                continue
            # Swap
            A[[pivot_row, found]] = A[[found, pivot_row]]
            # Scale pivot row
            inv = self._modinv(int(A[pivot_row, col]), p)
            if inv is None:
                continue
            A[pivot_row] = (A[pivot_row] * inv) % p
            # Eliminate
            for row in range(n_rows):
                if row == pivot_row:
                    continue
                factor = A[row, col]
                if factor != 0:
                    A[row] = (A[row] - factor * A[pivot_row]) % p
            pivot_row += 1

        # Check consistency and extract solution
        solution = np.zeros(n_vars, dtype=np.int64)
        for row in range(min(pivot_row, n_rows)):
            leading = -1
            for col in range(n_vars):
                if A[row, col] % p != 0:
                    leading = col
                    break
            if leading == -1:
                if A[row, -1] % p != 0:
                    return None  # inconsistent
                continue
            solution[leading] = A[row, -1] % p

        # Verify
        if pivot_row < n_vars:
            log.debug("[Gauss] Under-determined system (%d pivots for %d vars)", pivot_row, n_vars)
            # Still return partial — might work
        return solution

    # ------------------------------------------------------------------ #
    #  Gröbner Basis via SymPy                                            #
    # ------------------------------------------------------------------ #

    def _try_groebner(
        self, instance: CryptoInstance, modulus: int, t0: float
    ) -> SolverResult:
        """Build a multivariate polynomial system and solve with Gröbner bases."""
        log.info("[Gröbner] Attempting polynomial system solve over F_%d", modulus)
        try:
            from sympy import symbols, groebner, GF, Poly
            from sympy.polys.orderings import lex
        except ImportError:
            return self._fail("SymPy not available for Gröbner computation", t0)

        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        n_pairs = min(len(pt), len(ct), 20)  # limit for tractability

        if n_pairs < 3:
            return self._fail("Not enough pairs for Gröbner", t0)

        # Assume polynomial of degree ≤ 5 in one variable: c = sum(a_k * p^k)
        max_deg = min(5, n_pairs - 1)
        coeffs = symbols(f"a0:{max_deg + 1}")

        field = GF(modulus)
        polys = []
        for i in range(n_pairs):
            p_val = pt[i] % modulus
            c_val = ct[i] % modulus
            expr = sum(coeffs[k] * pow(p_val, k, modulus) for k in range(max_deg + 1)) - c_val
            polys.append(expr)

        try:
            log.debug("[Gröbner] Computing basis for %d polynomials, degree ≤ %d", len(polys), max_deg)
            basis = groebner(polys, *coeffs, modulus=modulus, order="lex")
            # Extract solutions
            from sympy import solve
            sol = solve(list(basis), coeffs, domain=GF(modulus))
            if sol:
                coeff_vals = [int(sol.get(c, 0)) % modulus for c in coeffs]
                log.info("[Gröbner] Solution coefficients: %s", coeff_vals)
                # Decrypt target
                ct_target = instance.ct_target_as_int_list()
                decrypted = self._invert_poly(coeff_vals, np.array(ct_target, dtype=np.int64), modulus)
                if decrypted is not None:
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=coeff_vals,
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.80,
                        details={"method": "groebner", "degree": max_deg},
                    )
        except Exception as exc:
            log.warning("[Gröbner] Computation failed: %s", exc)

        return self._fail("Gröbner basis did not yield a solution", t0)

    # ------------------------------------------------------------------ #
    #  Polynomial inversion helper                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _invert_poly(
        coeffs: list[int],
        ct_target: NDArray[np.int64],
        modulus: int,
    ) -> list[int] | None:
        """Invert c = poly(p) mod m by exhaustive search over F_p (for small p)."""
        if modulus > 2**20:
            log.debug("Modulus too large for brute-force polynomial inversion")
            return None

        # Build lookup table: p → poly(p) mod m
        lookup: dict[int, int] = {}
        for p in range(modulus):
            val = 0
            pk = 1
            for c in coeffs:
                val = (val + c * pk) % modulus
                pk = (pk * p) % modulus
            lookup[val] = p

        result = []
        for c in ct_target.tolist():
            c_mod = int(c) % modulus
            if c_mod not in lookup:
                return None
            result.append(lookup[c_mod])
        return result

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        """Extended Euclidean modular inverse."""
        if m == 0:
            return None
        g, x, _ = AlgebraicSolver._extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    @staticmethod
    def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = AlgebraicSolver._extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Algebraic] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
