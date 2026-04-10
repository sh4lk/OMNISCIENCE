"""Lattice Reduction Solver Module.

Implements LLL and BKZ attacks for:
  - Knapsack / subset-sum based ciphers
  - LWE (Learning With Errors) based schemes
  - Hidden number problem variants
  - General lattice-based key recovery
"""

from __future__ import annotations

import logging
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


class LatticeSolver:
    """Lattice-based cryptanalysis via LLL / BKZ reduction."""

    NAME = "lattice"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            modulus = recon.estimated_modulus or instance.modulus

            # Try knapsack attack
            res = self._attack_knapsack(instance, modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Try LWE attack
            if recon.lattice_structure_detected or recon.algo_family in (
                AlgoFamily.LWE_BASED,
                AlgoFamily.LATTICE_BASED,
                AlgoFamily.UNKNOWN,
            ):
                res = self._attack_lwe(instance, modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Try general hidden number problem
            res = self._attack_hnp(instance, modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Lattice solver exhausted all strategies", t0)

        except Exception as exc:
            log.exception("Lattice solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  LLL Implementation (pure NumPy fallback)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lll_reduce(basis: NDArray[np.int64], delta: float = 0.75) -> NDArray[np.int64]:
        """Lenstra–Lenstra–Lovász lattice basis reduction.

        Uses the classical LLL algorithm with Gram-Schmidt orthogonalization.
        For production use with large dimensions, fpylll is preferred (see _lll_fpylll).
        """
        B = basis.astype(np.float64).copy()
        n = B.shape[0]

        def gram_schmidt(B: NDArray) -> tuple[NDArray, NDArray]:
            n = B.shape[0]
            Q = np.zeros_like(B)
            mu = np.zeros((n, n))
            for i in range(n):
                Q[i] = B[i].copy()
                for j in range(i):
                    denom = np.dot(Q[j], Q[j])
                    if denom < 1e-10:
                        mu[i, j] = 0.0
                    else:
                        mu[i, j] = np.dot(B[i], Q[j]) / denom
                    Q[i] -= mu[i, j] * Q[j]
            return Q, mu

        k = 1
        while k < n:
            Q, mu = gram_schmidt(B)
            # Size-reduce B[k]
            for j in range(k - 1, -1, -1):
                if abs(mu[k, j]) > 0.5:
                    r = round(mu[k, j])
                    B[k] -= r * B[j]
                    Q, mu = gram_schmidt(B)

            # Lovász condition
            lhs = np.dot(Q[k], Q[k])
            rhs = (delta - mu[k, k - 1] ** 2) * np.dot(Q[k - 1], Q[k - 1])
            if lhs >= rhs:
                k += 1
            else:
                B[[k, k - 1]] = B[[k - 1, k]]
                k = max(k - 1, 1)

        return np.rint(B).astype(np.int64)

    @staticmethod
    def _lll_fpylll(basis: NDArray[np.int64], block_size: int = 20) -> NDArray[np.int64] | None:
        """LLL/BKZ reduction via fpylll (much faster for large dimensions)."""
        try:
            from fpylll import IntegerMatrix, LLL, BKZ
            from fpylll.algorithms.bkz2 import BKZReduction

            n, m = basis.shape
            A = IntegerMatrix(n, m)
            for i in range(n):
                for j in range(m):
                    A[i, j] = int(basis[i, j])

            # LLL first
            LLL.reduction(A)

            # BKZ if block_size > 2
            if block_size > 2 and n > 4:
                params = BKZ.Param(block_size=min(block_size, n), strategies=BKZ.DEFAULT_STRATEGY)
                BKZReduction(A)(params)

            result = np.zeros((n, m), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    result[i, j] = A[i, j]
            return result
        except ImportError:
            log.debug("fpylll not available, falling back to pure NumPy LLL")
            return None
        except Exception as exc:
            log.warning("fpylll reduction failed: %s", exc)
            return None

    def _reduce(self, basis: NDArray[np.int64], block_size: int = 20) -> NDArray[np.int64]:
        """Try fpylll first, fall back to pure NumPy."""
        result = self._lll_fpylll(basis, block_size)
        if result is not None:
            return result
        return self._lll_reduce(basis)

    # ------------------------------------------------------------------ #
    #  Knapsack / Subset-Sum Attack                                       #
    # ------------------------------------------------------------------ #

    def _attack_knapsack(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """CJLOSS lattice attack on low-density knapsack ciphers.

        Given public key w = [w_1, ..., w_n] and ciphertext s = sum(x_i * w_i),
        build the lattice:
            | I_n   0 |
            | w^T   -s|
        and reduce with LLL. A short vector encodes the secret bits x_i.
        """
        log.info("[Lattice/Knapsack] Attempting CJLOSS attack")
        pub = np.array(instance.pub_as_int_list(), dtype=np.int64)
        n = len(pub)
        if n < 2 or n > 256:
            return self._fail("Public key dimension unsuitable for knapsack attack", t0)

        ct_target = np.array(instance.ct_target_as_int_list(), dtype=np.int64)
        if len(ct_target) == 0:
            return self._fail("No target ciphertext", t0)

        # Assume ciphertext is a single value (sum)
        target_sum = int(ct_target[0]) if len(ct_target) == 1 else int(np.sum(ct_target))

        # Build CJLOSS lattice of dimension (n+1) × (n+1)
        N = max(abs(target_sum), int(np.max(np.abs(pub))), 1)
        scale = N * n  # scaling factor for the last column

        L = np.zeros((n + 1, n + 1), dtype=np.int64)
        # Identity block
        for i in range(n):
            L[i, i] = 2
        # Public key row
        L[n, :n] = pub
        L[n, n] = -target_sum

        # Scale last column to force the target equation
        L[:, n] *= scale

        log.debug("[Lattice/Knapsack] Reducing %d×%d lattice", n + 1, n + 1)
        reduced = self._reduce(L, block_size=min(30, n))

        # Search for a short vector with binary-like entries
        for row in reduced:
            candidate = row[:n]
            # Check if entries are in {0, 1} (or {-1, 0, 1} after centering)
            bits_01 = all(v in (0, 2) for v in candidate)
            bits_pm1 = all(v in (-1, 0, 1) for v in candidate)

            if bits_01:
                x = (candidate // 2).tolist()
                check = sum(x[i] * int(pub[i]) for i in range(n))
                if check == target_sum:
                    log.info("[Lattice/Knapsack] Found secret bits: %s", x)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=x,
                        decrypted=x,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={"method": "knapsack_cjloss"},
                    )
            elif bits_pm1:
                x = candidate.tolist()
                check = sum(x[i] * int(pub[i]) for i in range(n))
                if check == target_sum or check == -target_sum:
                    log.info("[Lattice/Knapsack] Found secret vector: %s", x)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=x,
                        decrypted=x,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.85,
                        details={"method": "knapsack_cjloss_signed"},
                    )

        return self._fail("Knapsack LLL did not find binary solution", t0)

    # ------------------------------------------------------------------ #
    #  LWE Attack (Kannan's embedding)                                    #
    # ------------------------------------------------------------------ #

    def _attack_lwe(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """Attack LWE-based encryption using Kannan's embedding technique.

        For LWE: b = A·s + e (mod q), recover secret s given (A, b).
        We interpret:
          - public_key encodes matrix A (flattened)
          - ciphertext encodes vector b
        """
        log.info("[Lattice/LWE] Attempting Kannan embedding attack")
        if modulus is None or modulus < 2:
            return self._fail("LWE attack requires a modulus", t0)

        pub = np.array(instance.pub_as_int_list(), dtype=np.int64) % modulus
        ct = np.array(instance.ct_known_as_int_list(), dtype=np.int64) % modulus

        m = len(ct)  # number of LWE samples
        if m < 2:
            return self._fail("Not enough samples for LWE attack", t0)

        # Try to infer dimension n from public key size
        if len(pub) % m != 0:
            # Try square matrix
            n = int(np.sqrt(len(pub)))
            if n * n != len(pub):
                return self._fail("Cannot reshape public key into matrix", t0)
            A = pub[:n * n].reshape(n, n) % modulus
            b = ct[:n] % modulus
            m, n = A.shape
        else:
            n = len(pub) // m
            A = pub[:m * n].reshape(m, n) % modulus
            b = ct[:m] % modulus

        if n > 80:
            return self._fail(f"LWE dimension {n} too large for direct lattice attack", t0)

        # Kannan embedding: build (m+1) × (n+1) lattice
        # [  q*I_m   0  ]
        # [   A^T    1  ]
        # [   b      0  ]  <- target row
        dim = n + 1
        L = np.zeros((m + n + 1, m + n + 1), dtype=np.int64)

        # q * I_m block
        for i in range(m):
            L[i, i] = modulus

        # A^T block
        for i in range(n):
            for j in range(m):
                L[m + i, j] = int(A[j, i])
            L[m + i, m + i] = 1

        # Target row
        for j in range(m):
            L[m + n, j] = int(b[j])

        log.debug("[Lattice/LWE] Reducing %d×%d lattice (q=%d)", L.shape[0], L.shape[1], modulus)
        reduced = self._reduce(L, block_size=min(25, dim))

        # The shortest vector should encode the error vector + secret
        for row in reduced:
            # Check if this could be the error vector (small entries)
            candidate_e = row[:m]
            if np.max(np.abs(candidate_e)) < modulus // 4:
                # Recover secret: s = A^{-1} * (b - e) mod q
                try:
                    from sympy import Matrix
                    A_sym = Matrix(A.tolist())
                    b_vec = Matrix([(int(b[j]) - int(candidate_e[j])) % modulus for j in range(min(m, n))]).T
                    # Use pseudo-inverse approach
                    if m == n:
                        A_inv = A_sym.inv_mod(modulus)
                        s = (A_inv * b_vec.T) % modulus
                        secret = [int(s[i]) % modulus for i in range(n)]
                        log.info("[Lattice/LWE] Recovered secret vector")
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=secret,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.75,
                            details={"method": "lwe_kannan", "dimension": n},
                        )
                except Exception as exc:
                    log.debug("[Lattice/LWE] Matrix inversion failed: %s", exc)

        return self._fail("LWE lattice reduction did not yield small error vector", t0)

    # ------------------------------------------------------------------ #
    #  Hidden Number Problem                                              #
    # ------------------------------------------------------------------ #

    def _attack_hnp(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """Hidden Number Problem: recover x given partial information about t_i*x mod p.

        This covers schemes where ciphertext leaks MSBs of a secret multiple.
        """
        log.info("[Lattice/HNP] Attempting HNP lattice attack")
        if modulus is None or modulus < 2:
            return self._fail("HNP requires modulus", t0)

        pt = np.array(instance.pt_as_int_list(), dtype=np.int64) % modulus
        ct = np.array(instance.ct_known_as_int_list(), dtype=np.int64) % modulus
        n = min(len(pt), len(ct))
        if n < 3:
            return self._fail("Not enough pairs for HNP", t0)
        n = min(n, 50)  # cap dimension

        # Build HNP lattice:
        # [ p  0  0 ... 0  0 ]
        # [ 0  p  0 ... 0  0 ]
        # [ t1 t2 t3... tn B/p]
        # [ c1 c2 c3... cn 0 ]
        B_scale = modulus
        dim = n + 1
        L = np.zeros((dim, dim), dtype=np.int64)
        for i in range(n - 1):
            L[i, i] = modulus
        for i in range(n - 1):
            L[n - 1, i] = int(pt[i])
        L[n - 1, n - 1] = 1
        L[n, :n - 1] = ct[:n - 1]

        reduced = self._reduce(L, block_size=min(20, dim))

        # Check shortest vectors for the secret
        for row in reduced:
            if row[-1] != 0:
                candidate = abs(int(row[-1])) % modulus
                if candidate > 0:
                    # Verify: does t_i * candidate ≈ c_i (mod p)?
                    errors = [(int(pt[i]) * candidate - int(ct[i])) % modulus for i in range(n)]
                    max_err = max(min(e, modulus - e) for e in errors)
                    if max_err < modulus // 4:
                        log.info("[Lattice/HNP] Recovered hidden number: %d", candidate)
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=candidate,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.70,
                            details={"method": "hnp", "max_error": max_err},
                        )

        return self._fail("HNP lattice attack did not recover hidden number", t0)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Lattice] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
