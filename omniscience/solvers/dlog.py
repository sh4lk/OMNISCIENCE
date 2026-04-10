"""Discrete Logarithm Solver Module.

Given g, h, p such that g^x ≡ h (mod p), find x.

Implements:
  1. Baby-step Giant-step (BSGS) — O(√p) time and space
  2. Pohlig-Hellman — when p-1 has small factors
  3. Pollard's rho for DLP — O(√p) time, O(1) space
  4. Index Calculus — sub-exponential for Z_p*
  5. Silver-Pohlig-Hellman combined attack
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class DLogSolver:
    """Discrete logarithm solver for DLP-based cryptosystems."""

    NAME = "dlog"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            modulus = recon.estimated_modulus or instance.modulus
            if modulus is None or modulus < 3:
                return self._fail("DLog requires a prime modulus", t0)

            pub = instance.pub_as_int_list()
            pt = instance.pt_as_int_list()
            ct = instance.ct_known_as_int_list()

            # Infer DLP parameters:
            # g = generator (from public key or plaintext)
            # h = g^x mod p (from ciphertext or public key)
            g, h = self._infer_dlp_params(pub, pt, ct, modulus)
            if g is None:
                return self._fail("Could not infer DLP parameters (g, h)", t0)

            log.info("[DLog] g=%d, h=%d, p=%d (%d bits)", g, h, modulus, modulus.bit_length())

            # Try methods in order of efficiency
            for method_name, method in [
                ("pohlig_hellman", lambda: self._pohlig_hellman(g, h, modulus, timeout=min(120, timeout / 3))),
                ("bsgs", lambda: self._bsgs(g, h, modulus, timeout=min(120, timeout / 3))),
                ("pollard_rho", lambda: self._pollard_rho_dlog(g, h, modulus, timeout=min(120, timeout / 3))),
                ("index_calculus", lambda: self._index_calculus(g, h, modulus, timeout=min(180, timeout / 2))),
            ]:
                if time.perf_counter() - t0 > timeout:
                    break
                log.debug("[DLog/%s] Starting...", method_name)
                x = method()
                if x is not None and pow(g, x, modulus) == h:
                    log.info("[DLog/%s] Found x = %d", method_name, x)
                    # Decrypt targets
                    ct_target = instance.ct_target_as_int_list()
                    decrypted = self._decrypt_with_dlog(x, g, ct_target, modulus, instance)

                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=x,
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.95,
                        details={"method": method_name, "x": x, "g": g, "modulus": modulus},
                    )

            return self._fail("All DLog methods failed", t0)

        except Exception as exc:
            log.exception("DLog solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  1. Baby-step Giant-step                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bsgs(g: int, h: int, p: int, order: int | None = None, timeout: float = 120.0) -> int | None:
        """Baby-step Giant-step in O(√n) time/space."""
        t0 = time.perf_counter()
        n = order or (p - 1)
        m = math.isqrt(n) + 1

        # Limit memory usage — only feasible for order ≤ ~2^32
        if m > 2**26:
            log.debug("[DLog/BSGS] Order too large (m=%d), skipping", m)
            return None

        # Baby step: table[g^j mod p] = j
        table: dict[int, int] = {}
        power = 1
        for j in range(m):
            if time.perf_counter() - t0 > timeout:
                return None
            table[power] = j
            power = power * g % p

        # Giant step: g^{-m} mod p
        factor = pow(g, p - 1 - m, p)  # g^{-m} mod p
        gamma = h
        for i in range(m):
            if time.perf_counter() - t0 > timeout:
                return None
            if gamma in table:
                x = i * m + table[gamma]
                return x % n if order else x
            gamma = gamma * factor % p

        return None

    # ------------------------------------------------------------------ #
    #  2. Pohlig-Hellman                                                  #
    # ------------------------------------------------------------------ #

    def _pohlig_hellman(self, g: int, h: int, p: int, timeout: float = 120.0) -> int | None:
        """Pohlig-Hellman: reduce DLP to small subgroup DLPs when p-1 is smooth."""
        t0 = time.perf_counter()
        n = p - 1
        factors = self._factor_small(n)

        if not factors:
            return None

        # Check smoothness — if largest factor is too big, skip
        max_prime = max(factors.keys())
        if max_prime > 2**28:
            log.debug("[DLog/PH] p-1 not smooth enough (largest prime factor = %d)", max_prime)
            # Still try with partial information if enough small factors
            if sum(pe for pe in factors.values()) < n.bit_length() // 2:
                return None

        residues: list[tuple[int, int]] = []  # (x_i mod q_i^e_i, q_i^e_i)

        for q, e in factors.items():
            if time.perf_counter() - t0 > timeout:
                return None
            qe = q ** e
            # Compute DLP in subgroup of order q^e
            g_sub = pow(g, n // qe, p)
            h_sub = pow(h, n // qe, p)

            # Solve in subgroup using BSGS
            if q <= 2**24:
                x_sub = self._bsgs(g_sub, h_sub, p, order=qe, timeout=min(30, timeout / len(factors)))
            else:
                x_sub = self._pollard_rho_dlog(g_sub, h_sub, p, order=qe, timeout=min(30, timeout / len(factors)))

            if x_sub is None:
                continue

            residues.append((x_sub % qe, qe))

        if not residues:
            return None

        # CRT reconstruction
        return self._crt(residues)

    # ------------------------------------------------------------------ #
    #  3. Pollard's rho for DLP                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pollard_rho_dlog(
        g: int, h: int, p: int, order: int | None = None, timeout: float = 120.0
    ) -> int | None:
        """Pollard's rho algorithm for DLP using Floyd's cycle detection."""
        t0 = time.perf_counter()
        n = order or (p - 1)

        def step(x: int, a: int, b: int) -> tuple[int, int, int]:
            s = x % 3
            if s == 0:
                return x * x % p, 2 * a % n, 2 * b % n
            elif s == 1:
                return x * g % p, (a + 1) % n, b
            else:
                return x * h % p, a, (b + 1) % n

        x, a, b = 1, 0, 0
        X, A, B = 1, 0, 0

        for _ in range(min(4 * math.isqrt(n), 10**8)):
            if time.perf_counter() - t0 > timeout:
                return None
            x, a, b = step(x, a, b)
            X, A, B = step(*step(X, A, B))

            if x == X:
                r = (a - A) % n
                s = (B - b) % n
                if s == 0:
                    break
                # x = r / s mod n
                g_inv = DLogSolver._modinv(s, n)
                if g_inv is not None:
                    return (r * g_inv) % n
                # Try with GCD
                d = math.gcd(s, n)
                if d > 1:
                    nn = n // d
                    rr = (r // d) % nn
                    ss_inv = DLogSolver._modinv(s // d, nn)
                    if ss_inv is not None:
                        base = (rr * ss_inv) % nn
                        for i in range(d):
                            candidate = base + i * nn
                            if pow(g, candidate, p) == h:
                                return candidate
                break

        return None

    # ------------------------------------------------------------------ #
    #  4. Index Calculus                                                  #
    # ------------------------------------------------------------------ #

    def _index_calculus(self, g: int, h: int, p: int, timeout: float = 180.0) -> int | None:
        """Simplified index calculus for DLP in Z_p*."""
        t0 = time.perf_counter()
        n = p - 1

        # Only feasible for moderate p
        if p.bit_length() > 80:
            return None

        # Factor base: small primes up to bound B
        B = max(20, int(math.exp(0.5 * math.sqrt(math.log(p) * math.log(math.log(max(3, p)))))))
        B = min(B, 5000)
        primes = self._sieve_primes(B)
        k = len(primes)

        if k < 3:
            return None

        log.debug("[DLog/IC] Factor base size: %d, bound B=%d", k, B)

        # Phase 1: collect relations g^r ≡ prod(p_i^e_i) mod p
        relations: list[tuple[int, list[int]]] = []  # (r, exponents)
        attempts = 0
        max_attempts = k * 20

        while len(relations) < k + 5 and attempts < max_attempts:
            if time.perf_counter() - t0 > timeout:
                return None
            r = random.randint(1, n - 1)
            val = pow(g, r, p)
            exponents = self._try_factor_over_base(val, primes)
            if exponents is not None:
                relations.append((r, exponents))
            attempts += 1

        if len(relations) < k:
            return None

        # Phase 2: solve linear system mod n for log_g(p_i)
        logs = self._solve_log_system(relations, primes, n)
        if logs is None:
            return None

        # Phase 3: express h in terms of the factor base
        for _ in range(max_attempts):
            if time.perf_counter() - t0 > timeout:
                return None
            s = random.randint(1, n - 1)
            val = (h * pow(g, s, p)) % p
            exponents = self._try_factor_over_base(val, primes)
            if exponents is not None:
                # log_g(h) = sum(e_i * log_g(p_i)) - s mod n
                result = (-s) % n
                for i, e in enumerate(exponents):
                    if e != 0 and i < len(logs) and logs[i] is not None:
                        result = (result + e * logs[i]) % n
                if pow(g, result, p) == h:
                    return result

        return None

    @staticmethod
    def _try_factor_over_base(val: int, primes: list[int]) -> list[int] | None:
        """Try to factor val over the given prime base. Return exponent vector or None."""
        exponents = [0] * len(primes)
        remaining = val
        for i, p in enumerate(primes):
            while remaining % p == 0:
                exponents[i] += 1
                remaining //= p
        if remaining == 1:
            return exponents
        return None

    def _solve_log_system(
        self, relations: list[tuple[int, list[int]]], primes: list[int], n: int
    ) -> list[int | None] | None:
        """Solve the system of linear congruences for discrete logs of the factor base."""
        k = len(primes)
        # Build augmented matrix
        rows = len(relations)
        matrix = []
        for r, exps in relations:
            row = [e % n for e in exps[:k]] + [r % n]
            matrix.append(row)

        # Gaussian elimination mod n
        solved = [None] * k
        for col in range(k):
            # Find pivot
            pivot = -1
            for row in range(len(matrix)):
                if matrix[row][col] % n != 0:
                    inv = self._modinv(matrix[row][col], n)
                    if inv is not None:
                        pivot = row
                        break
            if pivot == -1:
                continue

            # Scale and eliminate
            inv = self._modinv(matrix[pivot][col], n)
            matrix[pivot] = [(v * inv) % n for v in matrix[pivot]]
            for row in range(len(matrix)):
                if row != pivot and matrix[row][col] % n != 0:
                    factor = matrix[row][col]
                    matrix[row] = [(matrix[row][j] - factor * matrix[pivot][j]) % n for j in range(k + 1)]
            solved[col] = matrix[pivot][k]

        if any(s is not None for s in solved):
            return solved
        return None

    # ------------------------------------------------------------------ #
    #  Parameter inference                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_dlp_params(
        pub: list[int], pt: list[int], ct: list[int], p: int
    ) -> tuple[int | None, int | None]:
        """Try to identify (g, h=g^x) from the available data.

        Common patterns:
          - pub = [g, h]  (ElGamal-like)
          - pub = [g], ct = [h] (simple DLP)
          - E(m) = g^m * h^r or similar
        """
        if len(pub) >= 2:
            return pub[0] % p, pub[1] % p
        if len(pub) == 1 and len(ct) >= 1:
            return pub[0] % p, ct[0] % p
        if len(pt) >= 1 and len(ct) >= 1:
            return pt[0] % p, ct[0] % p
        return None, None

    @staticmethod
    def _decrypt_with_dlog(
        x: int, g: int, ct_target: list[int], p: int, instance: CryptoInstance
    ) -> list[int] | None:
        """Attempt decryption given the discrete log x = log_g(h).

        Tries common DLP-based decryption patterns:
          - ElGamal: (c1, c2) → m = c2 * c1^{-x} mod p
          - Simple power: c = g^m → m = log_g(c)
        """
        result = []
        # Try ElGamal pattern: pairs (c1, c2)
        if len(ct_target) % 2 == 0 and len(ct_target) >= 2:
            for i in range(0, len(ct_target), 2):
                c1 = ct_target[i] % p
                c2 = ct_target[i + 1] % p
                s = pow(c1, x, p)
                s_inv = pow(s, p - 2, p)
                m = (c2 * s_inv) % p
                result.append(m)
            return result

        # Try direct: c = m^e mod p (power cipher)
        for c in ct_target:
            result.append(pow(c % p, x, p))
        return result

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        if m <= 0:
            return None
        g, x, _ = DLogSolver._egcd(a % m, m)
        if g != 1:
            return None
        return x % m

    @staticmethod
    def _egcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = DLogSolver._egcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    @staticmethod
    def _crt(residues: list[tuple[int, int]]) -> int | None:
        """Chinese Remainder Theorem: solve x ≡ a_i (mod m_i)."""
        if not residues:
            return None
        x, m = residues[0]
        for a_i, m_i in residues[1:]:
            g = math.gcd(m, m_i)
            if (a_i - x) % g != 0:
                return None
            lcm = m * m_i // g
            inv = DLogSolver._modinv(m // g, m_i // g)
            if inv is None:
                return None
            x = (x + m * ((a_i - x) // g * inv % (m_i // g))) % lcm
            m = lcm
        return x % m

    @staticmethod
    def _factor_small(n: int, bound: int = 2**28) -> dict[int, int]:
        """Factor n using trial division up to bound. Returns {prime: exponent}."""
        factors: dict[int, int] = {}
        d = 2
        while d * d <= n and d <= bound:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = 1
        return factors

    @staticmethod
    def _sieve_primes(bound: int) -> list[int]:
        """Sieve of Eratosthenes up to bound."""
        if bound < 2:
            return []
        is_prime = [True] * (bound + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(bound**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, bound + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, bound + 1) if is_prime[i]]

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[DLog] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
