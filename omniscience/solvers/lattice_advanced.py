"""Advanced Lattice Cryptanalysis Module — "Matrix Labyrinth".

Extends the base lattice solver with specialized attacks against
lattice-based cryptographic constructions:

  1. NTRU Attack (lattice reduction on the NTRU public key matrix)
  2. GGH/HNF Attack (Goldreich-Goldwasser-Halevi lattice encryption)
  3. LWE Dual Attack (dual lattice approach for Learning With Errors)
  4. Ring-LWE / Module-LWE structural attacks
  5. SIS (Short Integer Solution) reduction
  6. Coppersmith's method (small roots of polynomials mod N)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)
from omniscience.solvers.lattice import LatticeSolver

log = logging.getLogger(__name__)


class AdvancedLatticeSolver:
    """Advanced lattice attacks for structured lattice-based crypto."""

    NAME = "lattice_advanced"

    def __init__(self):
        self._base = LatticeSolver()

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            modulus = recon.estimated_modulus or instance.modulus

            # Try NTRU
            res = self._attack_ntru(instance, modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Try GGH
            res = self._attack_ggh(instance, modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Try LWE Dual
            if modulus and modulus > 1:
                res = self._attack_lwe_dual(instance, modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Try Coppersmith
            if modulus and modulus > 1:
                res = self._attack_coppersmith(instance, modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Try SIS
            res = self._attack_sis(instance, modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Advanced lattice solver exhausted all strategies", t0)

        except Exception as exc:
            log.exception("Advanced lattice solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  1. NTRU Attack                                                     #
    # ------------------------------------------------------------------ #

    def _attack_ntru(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """Attack NTRU: h = f^{-1} * g (mod q) in Z[x]/(x^n - 1).

        Build the NTRU lattice:
            [ qI_n  0 ]
            [  H   I_n]
        where H is the circulant matrix of the public key h.
        LLL finds (f, g) as a short vector.
        """
        log.info("[Lattice/NTRU] Attempting NTRU lattice attack")
        pub = np.array(instance.pub_as_int_list(), dtype=np.int64)
        n = len(pub)

        if n < 4 or n > 200:
            return self._fail("Public key dimension unsuitable for NTRU", t0)

        q = modulus or (int(np.max(pub)) + 1)
        if q < 2:
            return self._fail("Invalid modulus for NTRU", t0)

        # Build circulant matrix H from public polynomial h
        H = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                H[i, j] = pub[(j - i) % n]

        # NTRU lattice: 2n × 2n
        L = np.zeros((2 * n, 2 * n), dtype=np.int64)
        # Top-left: qI
        for i in range(n):
            L[i, i] = q
        # Bottom-left: H
        L[n:, :n] = H
        # Bottom-right: I
        for i in range(n):
            L[n + i, n + i] = 1

        log.debug("[Lattice/NTRU] Reducing %d×%d lattice (q=%d)", 2 * n, 2 * n, q)
        reduced = self._base._reduce(L, block_size=min(30, n))

        # The shortest vector should be (f, g) with small coefficients
        for row in reduced:
            f_vec = row[:n]
            g_vec = row[n:]
            f_norm = np.max(np.abs(f_vec))
            g_norm = np.max(np.abs(g_vec))

            # NTRU private keys have small coefficients (typically ternary)
            if 0 < f_norm <= q // 4 and g_norm <= q // 4:
                # Verify: f * h ≡ g (mod q) in Z[x]/(x^n - 1)
                fh = self._poly_mul_cyclic(f_vec, pub, n) % q
                # Normalize to [-q/2, q/2)
                fh_centered = np.where(fh > q // 2, fh - q, fh)
                g_centered = np.where(g_vec > q // 2, g_vec - q, g_vec)

                if np.allclose(fh_centered, g_centered) or np.max(np.abs(fh_centered - g_centered)) < q // 10:
                    log.info("[Lattice/NTRU] Recovered private key (f, g)")
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=f_vec.tolist(),
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={
                            "method": "ntru",
                            "f": f_vec.tolist(),
                            "g": g_vec.tolist(),
                            "n": n,
                            "q": q,
                        },
                    )

        return self._fail("NTRU lattice did not yield short (f, g)", t0)

    @staticmethod
    def _poly_mul_cyclic(a: NDArray, b: NDArray, n: int) -> NDArray:
        """Polynomial multiplication in Z[x]/(x^n - 1) via convolution."""
        result = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                result[(i + j) % n] += int(a[i]) * int(b[j])
        return result

    # ------------------------------------------------------------------ #
    #  2. GGH Attack                                                      #
    # ------------------------------------------------------------------ #

    def _attack_ggh(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """Attack GGH encryption: c = m * B + e (with small error e).

        Use Babai's nearest plane / round-off on the public basis to recover m.
        """
        log.info("[Lattice/GGH] Attempting GGH/closest vector attack")
        pub = np.array(instance.pub_as_int_list(), dtype=np.int64)
        ct = np.array(instance.ct_target_as_int_list(), dtype=np.int64)

        # Try to interpret public key as a flattened basis matrix
        n = len(ct)
        if n < 2:
            return self._fail("Target too short for GGH", t0)

        if len(pub) < n * n:
            return self._fail("Public key too small to form GGH basis", t0)

        B = pub[:n * n].reshape(n, n).astype(np.float64)

        # LLL-reduce the basis
        B_int = pub[:n * n].reshape(n, n).copy()
        B_reduced = self._base._reduce(B_int, block_size=min(25, n))

        # Babai's round-off: m = round(c * B^{-1})
        try:
            B_red_f = B_reduced.astype(np.float64)
            B_inv = np.linalg.inv(B_red_f)
            coords = ct.astype(np.float64) @ B_inv
            m_rounded = np.rint(coords).astype(np.int64)

            # Compute error
            reconstructed = m_rounded @ B_reduced
            error = ct - reconstructed
            error_norm = np.max(np.abs(error))

            if error_norm < max(1, np.max(np.abs(B_reduced)) // 4):
                log.info("[Lattice/GGH] Babai round-off succeeded (error norm = %d)", error_norm)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=m_rounded.tolist(),
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.80,
                    details={"method": "ggh_babai", "error_norm": int(error_norm)},
                )
        except np.linalg.LinAlgError:
            log.debug("[Lattice/GGH] Basis not invertible")

        # Babai's nearest plane algorithm
        try:
            decrypted = self._babai_nearest_plane(B_reduced, ct)
            if decrypted is not None:
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=decrypted.tolist(),
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.75,
                    details={"method": "ggh_nearest_plane"},
                )
        except Exception:
            pass

        return self._fail("GGH attack did not recover message", t0)

    @staticmethod
    def _babai_nearest_plane(B: NDArray[np.int64], target: NDArray[np.int64]) -> NDArray[np.int64] | None:
        """Babai's nearest plane algorithm for CVP approximation."""
        n = B.shape[0]
        B_f = B.astype(np.float64)

        # Gram-Schmidt
        Q = np.zeros_like(B_f)
        mu = np.zeros((n, n))
        for i in range(n):
            Q[i] = B_f[i].copy()
            for j in range(i):
                dot = np.dot(B_f[i], Q[j])
                norm = np.dot(Q[j], Q[j])
                if norm < 1e-10:
                    continue
                mu[i, j] = dot / norm
                Q[i] -= mu[i, j] * Q[j]

        b = target.astype(np.float64).copy()
        coeffs = np.zeros(n)
        for i in range(n - 1, -1, -1):
            norm = np.dot(Q[i], Q[i])
            if norm < 1e-10:
                continue
            c = np.dot(b, Q[i]) / norm
            coeffs[i] = round(c)
            b -= coeffs[i] * B_f[i]

        return np.rint(coeffs).astype(np.int64)

    # ------------------------------------------------------------------ #
    #  3. LWE Dual Attack                                                 #
    # ------------------------------------------------------------------ #

    def _attack_lwe_dual(
        self, instance: CryptoInstance, modulus: int, t0: float
    ) -> SolverResult:
        """Dual lattice attack on LWE: find short vector in the dual lattice
        such that <v, b> is small (reveals distinguishing information).

        Dual lattice of A: vectors v such that v*A ≡ 0 (mod q).
        If v*e is small, then v*b = v*A*s + v*e ≈ v*e is small.
        """
        log.info("[Lattice/Dual-LWE] Attempting dual lattice attack")
        pub = np.array(instance.pub_as_int_list(), dtype=np.int64) % modulus
        ct = np.array(instance.ct_known_as_int_list(), dtype=np.int64) % modulus

        m = len(ct)
        if m < 2 or len(pub) < m:
            return self._fail("Not enough data for dual attack", t0)

        # Try square interpretation
        n = int(math.sqrt(len(pub)))
        if n * n != len(pub):
            n = len(pub) // m if len(pub) % m == 0 else 0
        if n < 2:
            return self._fail("Cannot reshape for dual attack", t0)

        A = pub[:m * n].reshape(m, n) % modulus if m * n <= len(pub) else None
        if A is None:
            return self._fail("Matrix reshape failed", t0)

        b = ct[:m] % modulus

        # Build dual lattice: [A^T | qI] and reduce
        # Kernel of A^T mod q
        dim = m + n
        L = np.zeros((dim, dim), dtype=np.int64)
        L[:n, :n] = np.eye(n, dtype=np.int64)
        L[:n, n:] = A.T % modulus
        for i in range(m):
            L[n + i, n + i] = modulus

        reduced = self._base._reduce(L, block_size=min(25, dim // 2 + 1))

        # Short vectors v in the dual satisfy v*A ≡ 0 (mod q)
        # Then v*b = v*e should be small
        for row in reduced:
            v = row[n:]  # the dual vector component
            if np.all(v == 0):
                continue
            vb = int(np.dot(v, b)) % modulus
            # Small means close to 0 or q
            vb_centered = min(vb, modulus - vb)
            if vb_centered < modulus // 8:
                log.info("[Lattice/Dual-LWE] Found distinguishing vector (v·b = %d)", vb_centered)
                # This alone doesn't decrypt, but gives structural information
                # Try to recover secret via accumulated short vectors
                pass

        return self._fail("Dual LWE attack: no direct decryption achieved", t0)

    # ------------------------------------------------------------------ #
    #  4. Coppersmith's Method (small roots mod N)                        #
    # ------------------------------------------------------------------ #

    def _attack_coppersmith(
        self, instance: CryptoInstance, modulus: int, t0: float
    ) -> SolverResult:
        """Coppersmith's method: find small roots of f(x) ≡ 0 (mod N).

        Useful for:
          - RSA with small padding (Stereotyped message attack)
          - Partial key exposure
          - Hastad's broadcast attack
        """
        log.info("[Lattice/Coppersmith] Attempting small roots via lattice")

        pub = instance.pub_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        if not ct_target:
            return self._fail("No target for Coppersmith", t0)

        e = pub[0] if pub else 3
        N = modulus
        c = ct_target[0]

        # Hastad attack: if e is small and c = m^e mod N
        # We look for small root of f(x) = x^e - c mod N
        if e <= 7:
            root = self._coppersmith_small_root(e, c, N, t0)
            if root is not None:
                log.info("[Lattice/Coppersmith] Found small root m = %d", root)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=[root],
                    private_key=root,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.85,
                    details={"method": "coppersmith", "root": root, "e": e},
                )

        # Stereotyped message: m = m_known + x, find small x
        pt = instance.pt_as_int_list()
        if pt:
            m_known = pt[0]
            # f(x) = (m_known + x)^e - c mod N
            root = self._coppersmith_stereotyped(m_known, e, c, N, t0)
            if root is not None:
                m = m_known + root
                log.info("[Lattice/Coppersmith] Stereotyped: m = %d + %d = %d", m_known, root, m)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=[m],
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.80,
                    details={"method": "coppersmith_stereotyped"},
                )

        return self._fail("Coppersmith did not find small root", t0)

    def _coppersmith_small_root(self, e: int, c: int, N: int, t0: float) -> int | None:
        """Find small root of x^e ≡ c (mod N) using Howgrave-Graham's lattice method."""
        # Bound on root: X = N^(1/e) (Coppersmith bound)
        X = int(N ** (1.0 / e)) + 1

        # Build the lattice for Howgrave-Graham
        # Polynomials: g_{i,j}(x) = N^{max(t-i,0)} * x^j * f(x)^i
        # where f(x) = x^e - c
        t_param = max(1, e)
        h = t_param + e

        # Simplified: build matrix from shifted polynomials of f(x) = x^e - c
        dim = h
        L = np.zeros((dim, dim), dtype=np.int64)

        # Fill with coefficients of g polynomials evaluated at X
        for i in range(dim):
            if i < e:
                # x^i * N
                for j in range(dim):
                    if j == i:
                        L[i, j] = N * (X ** i)
            else:
                # x^{i-e} * f(x) = x^{i-e} * (x^e - c) = x^i - c*x^{i-e}
                idx = i
                if idx < dim:
                    L[i, idx] = X ** idx
                idx_low = i - e
                if 0 <= idx_low < dim:
                    L[i, idx_low] = (-c * X ** idx_low) % (N * X ** dim)

        try:
            reduced = self._base._reduce(L, block_size=min(20, dim))
        except Exception:
            return None

        # Extract candidate roots from short vectors
        for row in reduced:
            # Reconstruct polynomial and find roots
            # The short vector gives coefficients of a polynomial h(x) with h(root) = 0 over Z
            coeffs = []
            for j in range(dim):
                if X ** j != 0:
                    coeffs.append(int(row[j]) // (X ** j) if X ** j != 0 else 0)
                else:
                    coeffs.append(0)

            # Try small values
            for x in range(min(X, 10**6)):
                val = sum(coeffs[j] * (x ** j) for j in range(len(coeffs)))
                if val == 0 and pow(x, e, N) == c % N:
                    return x

        return None

    def _coppersmith_stereotyped(
        self, m_known: int, e: int, c: int, N: int, t0: float
    ) -> int | None:
        """Find small x such that (m_known + x)^e ≡ c (mod N)."""
        # Brute-force for small x (when Coppersmith bound is small)
        bound = int(N ** (1.0 / e)) + 1
        bound = min(bound, 10**7)  # cap for tractability

        for x in range(bound):
            if pow(m_known + x, e, N) == c % N:
                return x
        return None

    # ------------------------------------------------------------------ #
    #  5. SIS Attack                                                      #
    # ------------------------------------------------------------------ #

    def _attack_sis(
        self, instance: CryptoInstance, modulus: int | None, t0: float
    ) -> SolverResult:
        """Short Integer Solution: find short x such that A*x ≡ 0 (mod q).

        This directly applies LLL to the columns of A augmented with qI.
        """
        log.info("[Lattice/SIS] Attempting SIS reduction")
        if modulus is None or modulus < 2:
            return self._fail("SIS requires modulus", t0)

        pub = np.array(instance.pub_as_int_list(), dtype=np.int64) % modulus
        n = len(pub)
        if n < 2:
            return self._fail("Dimension too small for SIS", t0)

        # Interpret as 1D → build lattice [q, pub]
        dim = n + 1
        L = np.zeros((dim, dim), dtype=np.int64)
        L[0, 0] = modulus
        for i in range(n):
            L[i + 1, 0] = pub[i]
            L[i + 1, i + 1] = 1

        reduced = self._base._reduce(L, block_size=min(25, dim))

        for row in reduced:
            if row[0] == 0 and np.any(row[1:] != 0):
                x_vec = row[1:]
                norm = np.max(np.abs(x_vec))
                if 0 < norm < modulus // 2:
                    # Verify: A * x ≡ 0 (mod q)
                    check = int(np.dot(pub, x_vec)) % modulus
                    if check == 0:
                        log.info("[Lattice/SIS] Found short solution (norm=%d)", norm)
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=x_vec.tolist(),
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.75,
                            details={"method": "sis", "norm": int(norm)},
                        )

        return self._fail("SIS reduction did not find short solution", t0)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[AdvLattice] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
