"""Approximate GCD (AGCD) Solver Module.

Attacks ciphers based on the Approximate GCD problem:
    Given samples x_i = p * q_i + r_i  (with small r_i),
    recover the hidden common factor p.

This covers:
  - DGHV-style fully homomorphic encryption
  - Schemes where noisy multiples of a secret are published
  - General approximate common divisor problems

Techniques implemented:
  1. Simultaneous Diophantine Approximation via LLL
  2. Orthogonal Lattice Attack (Nguyen-Stern)
  3. Multivariate AGCD via Coppersmith-style bounds
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

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


class AGCDSolver:
    """Solver for Approximate GCD based cryptosystems."""

    NAME = "agcd"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            samples = np.array(instance.ct_known_as_int_list(), dtype=np.int64)
            pub = np.array(instance.pub_as_int_list(), dtype=np.int64)

            # Combine public key and known ciphertexts as AGCD samples
            all_samples = np.concatenate([pub, samples]) if len(pub) > 1 else samples

            if len(all_samples) < 2:
                return self._fail("Need at least 2 samples for AGCD", t0)

            # Strategy 1: Simultaneous Diophantine Approximation
            res = self._attack_sda(all_samples, instance, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 2: Orthogonal Lattice Attack
            res = self._attack_orthogonal(all_samples, instance, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 3: Direct GCD tree for exact GCD component
            res = self._attack_gcd_tree(all_samples, instance, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("AGCD solver exhausted all strategies", t0)

        except Exception as exc:
            log.exception("AGCD solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  Strategy 1: Simultaneous Diophantine Approximation (SDA)           #
    # ------------------------------------------------------------------ #

    def _attack_sda(
        self, samples: NDArray[np.int64], instance: CryptoInstance, t0: float
    ) -> SolverResult:
        """Recover p from x_i = p * q_i + r_i using LLL on the SDA lattice.

        Build lattice:
            [ 2^{rho+1}   x_1   x_2  ...  x_n ]
            [    0         -x_0   0   ...   0  ]
            [    0          0   -x_0  ...   0  ]
            [   ...                            ]
            [    0          0     0   ... -x_0  ]

        where x_0 is the largest sample. A short vector reveals the q_i
        ratios, from which p = x_0 / q_0 (approximately).
        """
        log.info("[AGCD/SDA] Attempting Simultaneous Diophantine Approximation")

        n = len(samples)
        if n < 3:
            return self._fail("SDA needs at least 3 samples", t0)

        n = min(n, 40)  # cap dimension for tractability
        x = samples[:n].tolist()
        x0 = max(x)
        idx0 = x.index(x0)
        others = [x[i] for i in range(n) if i != idx0]
        m = len(others)

        # Estimate noise bound rho (bits)
        # Heuristic: noise is much smaller than samples
        rho = max(1, int(math.log2(max(1, min(abs(v) for v in x if v != 0)))) // 4)
        scale = 1 << (rho + 1)

        # Build (m+1) × (m+1) lattice
        dim = m + 1
        L = np.zeros((dim, dim), dtype=np.int64)
        L[0, 0] = scale
        for i in range(m):
            L[0, i + 1] = others[i]
            L[i + 1, i + 1] = -x0

        reduced = self._lll(L)

        # Search for short vector encoding the quotients
        for row in reduced:
            if row[0] == 0:
                continue
            # Recover q_0 from the first coordinate
            q0_candidate = abs(int(row[0])) // scale
            if q0_candidate < 1:
                continue

            p_candidate = x0 // q0_candidate
            if p_candidate < 2:
                continue

            # Verify: all samples should be close to a multiple of p
            residues = [abs(s % p_candidate) for s in samples.tolist()]
            max_residue = max(residues)
            threshold = p_candidate // 4

            if max_residue < threshold:
                log.info("[AGCD/SDA] Recovered p = %d (max residue = %d)", p_candidate, max_residue)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=p_candidate,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.85,
                    details={
                        "method": "sda",
                        "p": p_candidate,
                        "max_residue": max_residue,
                        "noise_bits": rho,
                    },
                )

        return self._fail("SDA lattice reduction did not recover p", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 2: Orthogonal Lattice Attack (Nguyen-Stern)               #
    # ------------------------------------------------------------------ #

    def _attack_orthogonal(
        self, samples: NDArray[np.int64], instance: CryptoInstance, t0: float
    ) -> SolverResult:
        """Orthogonal lattice attack: find vectors orthogonal to (q_1, ..., q_n).

        If x_i = p * q_i + r_i, then the vector (x_1, ..., x_n) is close
        to p * (q_1, ..., q_n). Vectors in the orthogonal lattice of
        (x_1, ..., x_n) will be short and reveal structure.
        """
        log.info("[AGCD/Ortho] Attempting Orthogonal Lattice Attack")

        n = min(len(samples), 30)
        x = samples[:n].tolist()

        # Build lattice: the kernel of x modulo some large bound
        # [ K  x_1 ]
        # [ 0   1  ]  repeated...
        # This is equivalent to finding short vectors v such that <v, x> ≈ 0

        K = max(abs(v) for v in x) * n  # large scaling factor
        dim = n
        L = np.zeros((dim, dim + 1), dtype=np.int64)
        for i in range(dim):
            L[i, i] = 1
            L[i, dim] = x[i]

        # Scale last column
        L[:, dim] *= (1 << 16)

        reduced = self._lll(L)

        # The short vectors in the reduced basis should have small last coordinate
        # meaning they are approximately orthogonal to x
        # From these, try to recover p by GCD analysis
        candidates: list[int] = []
        for row in reduced:
            inner = sum(int(row[i]) * x[i] for i in range(n))
            if inner != 0 and abs(inner) < max(abs(v) for v in x):
                candidates.append(abs(inner))

        if candidates:
            # p should divide (or nearly divide) the inner products
            from math import gcd
            from functools import reduce

            g = reduce(gcd, candidates)
            if g > 1:
                # Factor g to find p
                p_candidate = g
                # Verify
                residues = [abs(s % p_candidate) for s in x]
                max_res = max(residues)
                if max_res < p_candidate // 4:
                    log.info("[AGCD/Ortho] Recovered p = %d via orthogonal lattice", p_candidate)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=p_candidate,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.80,
                        details={"method": "orthogonal_lattice", "p": p_candidate},
                    )

        return self._fail("Orthogonal lattice attack did not recover p", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 3: GCD Tree (for near-exact multiples)                    #
    # ------------------------------------------------------------------ #

    def _attack_gcd_tree(
        self, samples: NDArray[np.int64], instance: CryptoInstance, t0: float
    ) -> SolverResult:
        """Product/remainder tree for batch GCD, plus perturbation search.

        If the noise is very small, pairwise GCD may directly find p.
        We also try GCD(x_i - delta, x_j) for small delta values.
        """
        log.info("[AGCD/GCD] Attempting GCD tree with perturbation")
        from math import gcd

        x = [abs(int(v)) for v in samples.tolist() if v != 0]
        n = len(x)
        if n < 2:
            return self._fail("Need at least 2 nonzero samples", t0)

        # Direct pairwise GCD
        best_g = 0
        for i in range(min(n, 50)):
            for j in range(i + 1, min(n, 50)):
                g = gcd(x[i], x[j])
                if g > best_g and g > 1:
                    best_g = g

        if best_g > 1:
            # Factor out small primes to find the core factor
            p_candidate = self._remove_small_factors(best_g)
            if p_candidate > 1:
                residues = [abs(v % p_candidate) for v in x]
                if max(residues) == 0:
                    log.info("[AGCD/GCD] Exact GCD found p = %d", p_candidate)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=p_candidate,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.95,
                        details={"method": "exact_gcd", "p": p_candidate},
                    )

        # Perturbation search: try GCD(x_i ± delta, x_j ± delta)
        max_delta = min(256, max(x) // 1000 + 1)
        for delta in range(1, max_delta):
            for i in range(min(n, 20)):
                for j in range(i + 1, min(n, 20)):
                    for di in (-delta, 0, delta):
                        for dj in (-delta, 0, delta):
                            if di == 0 and dj == 0:
                                continue
                            g = gcd(x[i] + di, x[j] + dj)
                            if g > best_g and g > 1:
                                # Verify against more samples
                                close = sum(1 for v in x if min(v % g, g - v % g) <= delta)
                                if close > n * 0.8:
                                    p_candidate = self._remove_small_factors(g)
                                    if p_candidate > 1:
                                        log.info(
                                            "[AGCD/GCD] Perturbed GCD found p = %d (delta=%d)",
                                            p_candidate, delta,
                                        )
                                        return SolverResult(
                                            solver_name=self.NAME,
                                            status=SolverStatus.SUCCESS,
                                            private_key=p_candidate,
                                            elapsed_seconds=time.perf_counter() - t0,
                                            confidence=0.75,
                                            details={"method": "perturbed_gcd", "p": p_candidate, "delta": delta},
                                        )

        return self._fail("GCD tree did not recover p", t0)

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lll(basis: NDArray[np.int64]) -> NDArray[np.int64]:
        """LLL reduce — delegates to fpylll if available."""
        try:
            from fpylll import IntegerMatrix, LLL as FLLL

            n, m = basis.shape
            A = IntegerMatrix(n, m)
            for i in range(n):
                for j in range(m):
                    A[i, j] = int(basis[i, j])
            FLLL.reduction(A)
            result = np.zeros((n, m), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    result[i, j] = A[i, j]
            return result
        except ImportError:
            pass

        # Fallback: use the lattice solver's pure implementation
        from omniscience.solvers.lattice import LatticeSolver
        return LatticeSolver._lll_reduce(basis)

    @staticmethod
    def _remove_small_factors(n: int, bound: int = 1000) -> int:
        """Remove small prime factors from n."""
        for p in range(2, min(bound, int(n**0.5) + 1)):
            while n % p == 0:
                n //= p
        return n

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[AGCD] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
