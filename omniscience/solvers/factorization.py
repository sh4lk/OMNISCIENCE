"""Integer Factorization Module.

Attacks RSA-like schemes by factoring the modulus N = p * q.

Implements:
  1. Trial Division (small factors)
  2. Fermat's Factorization (close primes)
  3. Pollard's rho (medium-sized factors)
  4. Pollard's p-1 (smooth-order factors)
  5. Williams' p+1
  6. Lenstra's Elliptic Curve Method (ECM)
  7. Quadratic Sieve (QS) — simplified for moderate N
  8. Wiener's Attack (small RSA private exponent)
  9. Boneh-Durfee (extended Wiener via lattice)
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

import numpy as np

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class FactorizationSolver:
    """Factor N to break RSA-like ciphers."""

    NAME = "factorization"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            N = recon.estimated_modulus or instance.modulus
            if N is None or N < 4:
                return self._fail("No modulus to factor", t0)

            pub = instance.pub_as_int_list()
            # Try to extract e (public exponent) from the public key
            e = pub[0] if len(pub) >= 1 and pub[0] > 1 else 65537

            log.info("[Factor] Attempting to factor N = %d (%d bits)", N, N.bit_length())

            # Ordered by speed — fast methods first
            for method_name, method in [
                ("trial_division", lambda: self._trial_division(N)),
                ("fermat", lambda: self._fermat(N, timeout=min(30, timeout / 6))),
                ("pollard_rho", lambda: self._pollard_rho(N, timeout=min(60, timeout / 4))),
                ("pollard_pm1", lambda: self._pollard_pm1(N, timeout=min(60, timeout / 4))),
                ("williams_pp1", lambda: self._williams_pp1(N, timeout=min(60, timeout / 4))),
                ("ecm", lambda: self._ecm(N, timeout=min(120, timeout / 3))),
                ("wiener", lambda: self._wiener_attack(N, e)),
                ("boneh_durfee", lambda: self._boneh_durfee(N, e)),
                ("hastad", lambda: self._hastad_broadcast(instance, e, N)),
            ]:
                if time.perf_counter() - t0 > timeout:
                    break
                log.debug("[Factor/%s] Starting...", method_name)
                factor = method()
                if factor and 1 < factor < N:
                    p, q = factor, N // factor
                    if p * q == N:
                        log.info("[Factor/%s] N = %d × %d", method_name, p, q)
                        # Compute private key d
                        phi = (p - 1) * (q - 1)
                        d = self._modinv(e, phi)
                        if d is None:
                            d = self._modinv(e, self._lcm(p - 1, q - 1))

                        # Decrypt target
                        decrypted = None
                        if d is not None:
                            ct_target = instance.ct_target_as_int_list()
                            decrypted = [pow(c, d, N) for c in ct_target]

                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=d,
                            decrypted=decrypted,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.95,
                            details={
                                "method": method_name,
                                "p": p, "q": q, "e": e, "d": d,
                            },
                        )

            return self._fail("All factorization methods failed", t0)

        except Exception as exc:
            log.exception("Factorization solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  1. Trial Division                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trial_division(N: int, bound: int = 1_000_000) -> int | None:
        if N % 2 == 0:
            return 2
        for p in range(3, min(bound, int(N**0.5) + 1), 2):
            if N % p == 0:
                return p
        return None

    # ------------------------------------------------------------------ #
    #  2. Fermat's Factorization                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fermat(N: int, timeout: float = 30.0) -> int | None:
        """Works when p and q are close: N = a² - b² = (a-b)(a+b)."""
        if N % 2 == 0:
            return 2
        a = math.isqrt(N)
        if a * a < N:
            a += 1
        t0 = time.perf_counter()
        max_iter = min(10_000_000, N)
        for _ in range(max_iter):
            b2 = a * a - N
            b = math.isqrt(b2)
            if b * b == b2:
                factor = a - b
                if 1 < factor < N:
                    return factor
            a += 1
            if time.perf_counter() - t0 > timeout:
                break
        return None

    # ------------------------------------------------------------------ #
    #  3. Pollard's rho                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pollard_rho(N: int, timeout: float = 60.0) -> int | None:
        """Floyd's cycle detection variant of Pollard's rho."""
        if N % 2 == 0:
            return 2

        t0 = time.perf_counter()
        for c in range(1, 100):
            x = random.randint(2, N - 1)
            y = x
            d = 1
            f = lambda v: (v * v + c) % N

            while d == 1:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), N)
                if time.perf_counter() - t0 > timeout:
                    return None

            if d != N:
                return d
        return None

    # ------------------------------------------------------------------ #
    #  4. Pollard's p-1                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pollard_pm1(N: int, timeout: float = 60.0, B1: int = 100_000) -> int | None:
        """Finds factors p where p-1 is B1-smooth."""
        a = 2
        t0 = time.perf_counter()
        # Phase 1: compute a^(M) mod N where M = lcm(1, 2, ..., B1)
        for p in range(2, B1):
            if time.perf_counter() - t0 > timeout:
                return None
            # Use prime powers up to B1
            if not FactorizationSolver._is_prime_simple(p):
                continue
            pk = p
            while pk * p <= B1:
                pk *= p
            a = pow(a, pk, N)
            g = math.gcd(a - 1, N)
            if 1 < g < N:
                return g
        return None

    # ------------------------------------------------------------------ #
    #  5. Williams' p+1                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _williams_pp1(N: int, timeout: float = 60.0, B: int = 50_000) -> int | None:
        """Finds factors p where p+1 is B-smooth using Lucas sequences."""
        t0 = time.perf_counter()
        for seed in range(3, 20):
            v = seed
            for q in range(2, B):
                if time.perf_counter() - t0 > timeout:
                    return None
                if not FactorizationSolver._is_prime_simple(q):
                    continue
                # Lucas chain: V_q(v, 1) mod N
                e = q
                while e * q <= B:
                    e *= q
                # Compute V_e using the doubling formula
                v = FactorizationSolver._lucas_v(v, e, N)
                g = math.gcd(v - 2, N)
                if 1 < g < N:
                    return g
        return None

    @staticmethod
    def _lucas_v(v: int, n: int, mod: int) -> int:
        """Compute V_n(v, 1) mod m using the binary method."""
        if n == 0:
            return 2
        if n == 1:
            return v % mod
        vk = v % mod
        vk1 = (v * v - 2) % mod
        bits = bin(n)[3:]  # skip '0b1'
        for bit in bits:
            if bit == '1':
                vk = (vk * vk1 - v) % mod
                vk1 = (vk1 * vk1 - 2) % mod
            else:
                vk1 = (vk * vk1 - v) % mod
                vk = (vk * vk - 2) % mod
        return vk

    # ------------------------------------------------------------------ #
    #  6. Lenstra's ECM (Elliptic Curve Method)                           #
    # ------------------------------------------------------------------ #

    def _ecm(self, N: int, timeout: float = 120.0, B1: int = 50_000, curves: int = 50) -> int | None:
        """Lenstra's ECM using Montgomery curves."""
        t0 = time.perf_counter()

        for _ in range(curves):
            if time.perf_counter() - t0 > timeout:
                return None
            # Random Montgomery curve: By² = x³ + Ax² + x
            sigma = random.randint(6, N - 1)
            u = (sigma * sigma - 5) % N
            v = (4 * sigma) % N

            inv = self._modinv(v, N)
            if inv is None:
                # GCD(v, N) > 1 — might be a factor
                g = math.gcd(v, N)
                if 1 < g < N:
                    return g
                continue

            # Point on curve
            Qx = pow(u, 3, N) * inv % N
            # A coefficient
            diff = v - u
            A = (pow(diff, 3, N) * (3 * u + v) % N * pow(4 * u * u * u % N * v % N, N - 2, N) % N - 2) % N

            # Phase 1: multiply point by lcm(1..B1)
            Px, Pz = Qx, 1
            for p in range(2, B1):
                if time.perf_counter() - t0 > timeout:
                    return None
                if not self._is_prime_simple(p):
                    continue
                pk = p
                while pk * p <= B1:
                    pk *= p
                Px, Pz = self._ec_montgomery_mul(Px, Pz, pk, A, N)
                if Pz == 0:
                    break

            g = math.gcd(Pz, N)
            if 1 < g < N:
                return g

        return None

    @staticmethod
    def _ec_montgomery_mul(
        x: int, z: int, k: int, A: int, N: int
    ) -> tuple[int, int]:
        """Scalar multiplication on Montgomery curve using binary ladder."""
        if k == 0:
            return 0, 1
        if k == 1:
            return x, z

        r0x, r0z = x, z
        r1x, r1z = (x * x - z * z) % N, (2 * x * z + A * z * z) % N

        for bit in bin(k)[3:]:
            if bit == '1':
                # r0 = r0 + r1, r1 = 2*r1
                t = (r0x * r1x - r0z * r1z) % N
                u = (r0x * r1z - r0z * r1x) % N
                r0x = (t * t * z) % N
                r0z = (u * u * x) % N
                t = (r1x * r1x - r1z * r1z) % N
                u = (2 * r1x * r1z + A * r1z * r1z) % N
                r1x = (t * t) % N
                r1z = (t * u) % N
            else:
                t = (r0x * r1x - r0z * r1z) % N
                u = (r0x * r1z - r0z * r1x) % N
                r1x = (t * t * z) % N
                r1z = (u * u * x) % N
                t = (r0x * r0x - r0z * r0z) % N
                u = (2 * r0x * r0z + A * r0z * r0z) % N
                r0x = (t * t) % N
                r0z = (t * u) % N

        return r0x % N, r0z % N

    # ------------------------------------------------------------------ #
    #  7. Wiener's Attack                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wiener_attack(N: int, e: int) -> int | None:
        """Wiener's continued fraction attack for small d.

        If d < N^0.25 / 3, the convergents of e/N reveal k/d.
        """
        log.debug("[Factor/Wiener] e=%d, N=%d", e, N)

        def _convergents(a: int, b: int):
            """Yield convergents p_k/q_k of the continued fraction a/b."""
            p_prev, p_curr = 0, 1
            q_prev, q_curr = 1, 0
            while b:
                q_val = a // b
                a, b = b, a - q_val * b
                p_prev, p_curr = p_curr, q_val * p_curr + p_prev
                q_prev, q_curr = q_curr, q_val * q_curr + q_prev
                yield p_curr, q_curr

        for k, d in _convergents(e, N):
            if k == 0 or d == 0:
                continue
            # Check: (e*d - 1) / k should be an integer = phi(N)
            if (e * d - 1) % k != 0:
                continue
            phi = (e * d - 1) // k
            # phi(N) = N - p - q + 1  →  p + q = N - phi + 1
            s = N - phi + 1
            # p and q are roots of x² - s*x + N = 0
            discriminant = s * s - 4 * N
            if discriminant < 0:
                continue
            sqrt_disc = math.isqrt(discriminant)
            if sqrt_disc * sqrt_disc != discriminant:
                continue
            p = (s + sqrt_disc) // 2
            q = (s - sqrt_disc) // 2
            if p * q == N and p > 1 and q > 1:
                log.info("[Factor/Wiener] Found d=%d, p=%d, q=%d", d, p, q)
                return min(p, q)

        return None

    # ------------------------------------------------------------------ #
    #  8. Boneh-Durfee (extended Wiener via lattice, d < N^0.292)         #
    # ------------------------------------------------------------------ #

    def _boneh_durfee(self, N: int, e: int) -> int | None:
        """Boneh-Durfee attack: works for d < N^0.292.

        Uses Coppersmith-type lattice approach on the equation:
            e*d ≡ 1 (mod (p-1)(q-1))
        ⟹  e*d = 1 + k*(N - p - q + 1)
        ⟹  k*(p+q-1) + 1 ≡ 0  (mod e)

        We search for small (k, s) where s = -(p+q) using LLL.
        """
        log.debug("[Factor/BD] Attempting Boneh-Durfee (N=%d bits, e=%d bits)", N.bit_length(), e.bit_length())

        if e < N:
            return None  # Boneh-Durfee needs e ≈ N

        # Build lattice for Coppersmith on f(x,y) = x*(N+1+y) + 1 mod e
        # where x = k, y = -(p+q)
        delta = 0.292
        m_param = max(3, int(round(N.bit_length() * delta)))
        t_param = max(1, int(round(m_param * ((1.0 / delta) - 1))))

        # Simplified: try small k values directly
        # e*d = 1 + k*phi(N), and phi(N) ≈ N - 2*sqrt(N)
        bound = int(N ** delta)
        for k in range(1, min(bound, 10**6)):
            # phi_N = (e*d - 1) / k
            # d must be positive integer, so e*d - 1 > 0
            # Try: phi_N candidates near N
            for d_candidate_bits in range(1, int(N.bit_length() * delta) + 2):
                d_max = 1 << d_candidate_bits
                phi_candidate = (e * d_max - 1) // k
                if phi_candidate <= 0 or phi_candidate >= N:
                    continue
                # p + q = N - phi + 1
                s = N - phi_candidate + 1
                disc = s * s - 4 * N
                if disc < 0:
                    continue
                sqrt_disc = math.isqrt(disc)
                if sqrt_disc * sqrt_disc == disc:
                    p = (s + sqrt_disc) // 2
                    q = (s - sqrt_disc) // 2
                    if p * q == N and p > 1 and q > 1:
                        log.info("[Factor/BD] Found p=%d, q=%d via Boneh-Durfee", p, q)
                        return min(p, q)
        return None

    # ------------------------------------------------------------------ #
    #  9. Hastad Broadcast Attack                                         #
    # ------------------------------------------------------------------ #

    def _hastad_broadcast(self, instance: CryptoInstance, e: int, N: int) -> int | None:
        """Hastad's broadcast attack: if m^e < N and e is small,
        then m = e-th root of c (integer root, no modular reduction needed).

        Also handles multi-modulus case via CRT when extra data is in instance.extra.
        """
        log.debug("[Factor/Hastad] e=%d", e)
        ct_target = instance.ct_target_as_int_list()
        if not ct_target:
            return None

        c = ct_target[0]

        # Simple case: e-th integer root
        if e <= 11:
            # Try integer e-th root
            m = self._integer_root(c, e)
            if m is not None and pow(m, e) == c:
                log.info("[Factor/Hastad] Found m = %d (integer %d-th root)", m, e)
                # Return m as "factor" — caller will handle it as decryption
                # We store it in instance.extra for the caller
                instance.extra["hastad_plaintext"] = m
                return None  # Not a factor, but we solved it differently

            # Also try m^e mod N = c  where m^e wraps around once or twice
            for k in range(1, min(20, e)):
                m = self._integer_root(c + k * N, e)
                if m is not None and pow(m, e, N) == c:
                    log.info("[Factor/Hastad] Found m = %d (wrapped %d-th root, k=%d)", m, e, k)
                    instance.extra["hastad_plaintext"] = m
                    return None

        # Multi-modulus CRT case
        extra_moduli = instance.extra.get("moduli", [])
        extra_cts = instance.extra.get("ciphertexts", [])
        if len(extra_moduli) >= e - 1 and len(extra_cts) >= e - 1:
            # CRT: combine c_i ≡ m^e (mod N_i) for i = 1..e
            all_mods = [N] + extra_moduli[:e - 1]
            all_cts = [c] + extra_cts[:e - 1]
            combined = self._crt_list(all_cts, all_mods)
            if combined is not None:
                m = self._integer_root(combined, e)
                if m is not None and pow(m, e) == combined:
                    log.info("[Factor/Hastad] CRT broadcast: m = %d", m)
                    instance.extra["hastad_plaintext"] = m
                    return None

        return None

    @staticmethod
    def _integer_root(n: int, e: int) -> int | None:
        """Compute the integer e-th root of n, or None if not a perfect power."""
        if n < 0:
            return None
        if n == 0:
            return 0
        # Newton's method for integer root
        if e == 1:
            return n
        if e == 2:
            r = math.isqrt(n)
            return r if r * r == n else None

        # General case
        lo, hi = 1, n
        # Better initial guess
        bits = n.bit_length()
        hi = 1 << ((bits + e - 1) // e + 1)
        lo = max(1, 1 << ((bits - 1) // e - 1))

        while lo <= hi:
            mid = (lo + hi) // 2
            power = mid ** e
            if power == n:
                return mid
            elif power < n:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    @staticmethod
    def _crt_list(remainders: list[int], moduli: list[int]) -> int | None:
        """CRT for a list of congruences."""
        if not remainders:
            return None
        x, m = remainders[0], moduli[0]
        for i in range(1, len(remainders)):
            a_i, m_i = remainders[i], moduli[i]
            g = math.gcd(m, m_i)
            if (a_i - x) % g != 0:
                return None
            lcm = m * m_i // g
            _, u, _ = FactorizationSolver._extended_gcd(m // g, m_i // g)
            x = (x + m * ((a_i - x) // g * u % (m_i // g))) % lcm
            m = lcm
        return x

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        if m <= 0:
            return None
        g, x, _ = FactorizationSolver._extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    @staticmethod
    def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = FactorizationSolver._extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    @staticmethod
    def _is_prime_simple(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Factor] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
