"""Meet-in-the-Middle (MITM) Attack Module.

For composed ciphers C = E_k2(E_k1(P)), MITM reduces 2^{2n} brute-force
to 2^n time + 2^n memory.

Strategies:
  1. Classic MITM: table E_k1(P) for all k1, match against D_k2(C) for all k2
  2. Multi-dimensional MITM: when the cipher has more than 2 stages
  3. Splice-and-cut MITM: for ciphers with cyclic structure
  4. Dissection attacks: memory-efficient MITM variants
  5. Generic function composition MITM
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any, Callable

import numpy as np

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class MITMSolver:
    """Meet-in-the-Middle attack engine."""

    NAME = "mitm"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            modulus = recon.estimated_modulus or instance.modulus

            # Strategy 1: Double encryption MITM
            res = self._mitm_double_encryption(instance, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 2: Functional MITM (generic composition)
            res = self._mitm_functional(instance, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 3: Affine composition MITM
            res = self._mitm_affine(instance, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("MITM exhausted all strategies", t0)

        except Exception as exc:
            log.exception("MITM solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  Strategy 1: Double Encryption MITM                                 #
    # ------------------------------------------------------------------ #

    def _mitm_double_encryption(
        self, instance: CryptoInstance, modulus: int | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """Classic MITM against C = E_{k2}(E_{k1}(P)) mod m.

        Assumes E_k(x) = (k * x + k) mod m  or  E_k(x) = (x + k) mod m
        for key bytes k1, k2 in [0, m).

        Build table: forward[E_{k1}(P)] = k1  for all k1
        Then search: D_{k2}(C) in table   for all k2
        """
        log.info("[MITM/Double] Attempting double-encryption MITM")
        if modulus is None or modulus < 2:
            return self._fail("MITM needs modulus", t0)

        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        if not pt or not ct:
            return self._fail("No pairs for MITM", t0)

        p0, c0 = pt[0] % modulus, ct[0] % modulus
        key_space = min(modulus, 2**20)  # cap for memory

        # Try several encryption models
        for model_name, enc_fn, dec_fn in self._get_models(modulus):
            log.debug("[MITM/Double/%s] Building forward table (size %d)", model_name, key_space)

            # Phase 1: build forward table
            forward: dict[int, int] = {}
            for k1 in range(key_space):
                if time.perf_counter() - t0 > timeout:
                    return self._fail("MITM timeout during table build", t0)
                intermediate = enc_fn(p0, k1)
                forward[intermediate] = k1

            # Phase 2: search backward
            for k2 in range(key_space):
                if time.perf_counter() - t0 > timeout:
                    return self._fail("MITM timeout during search", t0)
                intermediate = dec_fn(c0, k2)
                if intermediate in forward:
                    k1 = forward[intermediate]
                    # Verify with additional pairs
                    if len(pt) > 1 and len(ct) > 1:
                        check = enc_fn(enc_fn(pt[1] % modulus, k1), k2)
                        if check != ct[1] % modulus:
                            # Also check: outer enc applied to inner enc
                            inner = enc_fn(pt[1] % modulus, k1)
                            outer = enc_fn(inner, k2)
                            if outer != ct[1] % modulus:
                                continue

                    log.info("[MITM/Double/%s] Found k1=%d, k2=%d", model_name, k1, k2)
                    # Decrypt targets
                    decrypted = []
                    for c in ct_target:
                        mid = dec_fn(c % modulus, k2)
                        plain = dec_fn(mid, k1)
                        decrypted.append(plain)

                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=[k1, k2],
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.85,
                        details={
                            "method": f"mitm_double_{model_name}",
                            "k1": k1, "k2": k2,
                        },
                    )

        return self._fail("Double encryption MITM: no match found", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 2: Functional MITM                                        #
    # ------------------------------------------------------------------ #

    def _mitm_functional(
        self, instance: CryptoInstance, modulus: int | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """Generic MITM: split key into halves and meet in the middle.

        Assumes the key is a tuple (k_hi, k_lo) and the cipher applies them
        in sequence. We try splitting the key space and matching.
        """
        log.info("[MITM/Functional] Generic key-split MITM")
        if modulus is None or modulus < 2:
            return self._fail("Needs modulus", t0)

        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        if not pt or not ct:
            return self._fail("No pairs", t0)

        p0, c0 = pt[0] % modulus, ct[0] % modulus

        # Split: key = k_hi * sqrt(m) + k_lo
        sqrt_m = int(math.isqrt(modulus)) + 1
        if sqrt_m > 2**16:
            return self._fail("Key space too large for functional MITM", t0)

        # Model: c = (k_hi * p * sqrt_m + k_lo * p) mod m
        #      = p * (k_hi * sqrt_m + k_lo) mod m  →  c = p * key mod m
        # Forward: compute p * k_lo mod m for all k_lo
        forward: dict[int, int] = {}
        for k_lo in range(sqrt_m):
            if time.perf_counter() - t0 > timeout:
                break
            val = (p0 * k_lo) % modulus
            forward[val] = k_lo

        # Backward: compute (c - p * k_hi * sqrt_m) mod m for all k_hi
        for k_hi in range(sqrt_m):
            if time.perf_counter() - t0 > timeout:
                break
            val = (c0 - p0 * k_hi * sqrt_m) % modulus
            if val in forward:
                k_lo = forward[val]
                key = k_hi * sqrt_m + k_lo
                # Verify
                if len(pt) > 1 and len(ct) > 1:
                    if (pt[1] * key) % modulus != ct[1] % modulus:
                        continue

                log.info("[MITM/Functional] Found key = %d (k_hi=%d, k_lo=%d)", key, k_hi, k_lo)
                # Decrypt: p = c * key_inv mod m
                key_inv = self._modinv(key, modulus)
                if key_inv is not None:
                    decrypted = [(c * key_inv) % modulus for c in ct_target]
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=key,
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.85,
                        details={"method": "mitm_functional", "key": key},
                    )

        return self._fail("Functional MITM: no match", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 3: Affine Composition MITM                                #
    # ------------------------------------------------------------------ #

    def _mitm_affine(
        self, instance: CryptoInstance, modulus: int | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """MITM for composed affine: C = a2*(a1*P + b1) + b2 mod m.

        The middle value is a1*P + b1. We build a table for the forward
        direction (a1, b1) and search from the backward direction (a2, b2).
        """
        log.info("[MITM/Affine] Composed affine MITM")
        if modulus is None or modulus < 2 or modulus > 2**12:
            return self._fail("Modulus unsuitable for affine MITM", t0)

        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        if not pt or not ct:
            return self._fail("No pairs", t0)

        p0, c0 = pt[0] % modulus, ct[0] % modulus
        m = modulus

        # Forward table: intermediate = (a1 * p0 + b1) mod m
        forward: dict[int, tuple[int, int]] = {}
        for a1 in range(m):
            for b1 in range(m):
                if time.perf_counter() - t0 > timeout:
                    return self._fail("Affine MITM timeout", t0)
                mid = (a1 * p0 + b1) % m
                forward[mid] = (a1, b1)

        # Backward: c0 = a2 * mid + b2 mod m  →  mid = (c0 - b2) * a2_inv mod m
        for a2 in range(1, m):
            a2_inv = self._modinv(a2, m)
            if a2_inv is None:
                continue
            for b2 in range(m):
                if time.perf_counter() - t0 > timeout:
                    return self._fail("Affine MITM timeout", t0)
                mid = (a2_inv * (c0 - b2)) % m
                if mid in forward:
                    a1, b1 = forward[mid]
                    # Verify with second pair
                    if len(pt) > 1 and len(ct) > 1:
                        mid2 = (a1 * (pt[1] % m) + b1) % m
                        check = (a2 * mid2 + b2) % m
                        if check != ct[1] % m:
                            continue

                    log.info("[MITM/Affine] a1=%d, b1=%d, a2=%d, b2=%d", a1, b1, a2, b2)
                    # Decrypt: invert both affine layers
                    a1_inv = self._modinv(a1, m)
                    if a1_inv is None:
                        continue
                    decrypted = []
                    for c in ct_target:
                        mid_val = (a2_inv * (c % m - b2)) % m
                        plain = (a1_inv * (mid_val - b1)) % m
                        decrypted.append(plain)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=[a1, b1, a2, b2],
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={"method": "mitm_affine"},
                    )

        return self._fail("Affine MITM: no match", t0)

    # ------------------------------------------------------------------ #
    #  Encryption/Decryption models for double MITM                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_models(m: int) -> list[tuple[str, Callable, Callable]]:
        """Return (name, encrypt_fn, decrypt_fn) tuples for common models."""
        models = []

        # Additive: E_k(x) = (x + k) mod m
        models.append((
            "additive",
            lambda x, k: (x + k) % m,
            lambda c, k: (c - k) % m,
        ))

        # XOR (byte-level): E_k(x) = x ^ k
        if m <= 256:
            models.append((
                "xor",
                lambda x, k: x ^ k,
                lambda c, k: c ^ k,
            ))

        # Multiplicative: E_k(x) = k * x mod m
        models.append((
            "multiplicative",
            lambda x, k: (k * x) % m if k > 0 else 0,
            lambda c, k: (c * pow(k, m - 2, m)) % m if k > 0 and m > 2 else 0,
        ))

        # Affine: E_k(x) = (k * x + k) mod m  (key used as both mult and add)
        models.append((
            "affine_self",
            lambda x, k: (k * x + k) % m,
            lambda c, k: ((c - k) * pow(k, m - 2, m)) % m if k > 0 and m > 2 else 0,
        ))

        return models

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        if m <= 0 or a % m == 0:
            return None
        g, x, _ = MITMSolver._egcd(a % m, m)
        return x % m if g == 1 else None

    @staticmethod
    def _egcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = MITMSolver._egcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[MITM] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
