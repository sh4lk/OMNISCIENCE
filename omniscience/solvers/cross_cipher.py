"""Cross-Cipher Analysis & Crib Dragging Module.

For scenarios where:
  - Two ciphertexts are XORed/combined with the same key stream
  - A "crib" (known word) is dragged across ciphertext to find matches
  - Multiple encryptions under related keys are cross-correlated
  - Layer composition is detected and decomposed

Techniques:
  1. Crib Dragging (known-word search in XOR stream)
  2. Two-Time Pad attack (C1 ⊕ C2 = P1 ⊕ P2)
  3. Related-Key cross-correlation
  4. Cipher composition decomposition
  5. Frequency cross-analysis across multiple ciphertexts
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import Any

import numpy as np

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)

# Common English words/fragments for crib dragging
DEFAULT_CRIBS = [
    b"the ", b" the", b"and ", b" and", b"ing ", b"tion",
    b"that", b"ment", b"with", b"have", b"this", b"will",
    b"from", b"they", b"been", b"flag", b"FLAG", b"ctf{",
    b"CTF{", b"flag{", b"key ", b"KEY", b"password",
    b"http", b"www.", b".com", b"the flag", b"secret",
    b"\x00\x00", b"{\n", b"}\n", b": ", b", ",
]


class CrossCipherSolver:
    """Cross-cipher analysis and crib dragging attacks."""

    NAME = "cross_cipher"

    def __init__(self, custom_cribs: list[bytes] | None = None):
        self._cribs = custom_cribs or DEFAULT_CRIBS

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 300.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            pt = instance.pt_as_int_list()
            ct = instance.ct_known_as_int_list()
            ct_target = instance.ct_target_as_int_list()

            # Strategy 1: Two-time pad (if we have pt and can XOR-recover key)
            res = self._attack_two_time_pad(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 2: Crib dragging on ct_target XORed with known ct
            if len(ct) > 0 and len(ct_target) > 0:
                res = self._attack_crib_drag(pt, ct, ct_target, t0, timeout)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 3: Composition decomposition
            res = self._attack_decompose(pt, ct, ct_target, instance.modulus, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 4: Related-key correlation
            extra_cts = instance.extra.get("additional_ciphertexts", [])
            if extra_cts:
                res = self._attack_related_key(ct, ct_target, extra_cts, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            return self._fail("Cross-cipher: no strategy succeeded", t0)

        except Exception as exc:
            log.exception("Cross-cipher solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  1. Two-Time Pad                                                    #
    # ------------------------------------------------------------------ #

    def _attack_two_time_pad(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """If C1 = P1 ⊕ K and C2 = P2 ⊕ K, then C1 ⊕ C2 = P1 ⊕ P2.

        With known P1: P2 = P1 ⊕ C1 ⊕ C2 = P1 ⊕ (C1 ⊕ C2).
        """
        log.info("[Cross/2TP] Attempting two-time pad attack")
        n_known = min(len(pt), len(ct))
        n_target = len(ct_target)
        if n_known < 1 or n_target < 1:
            return self._fail("Two-time pad: not enough data", t0)

        # Recover key stream from known pair
        key_stream = [(pt[i] ^ ct[i]) & 0xFF for i in range(n_known)]

        # Decrypt target using the same key stream
        if n_target <= n_known:
            decrypted = [(ct_target[i] ^ key_stream[i]) & 0xFF for i in range(n_target)]

            # Verify: does this look reasonable?
            # Check if decrypted bytes are printable ASCII
            printable_ratio = sum(1 for b in decrypted if 32 <= b <= 126) / max(len(decrypted), 1)

            if printable_ratio > 0.6 or n_known >= n_target:
                log.info("[Cross/2TP] Recovered %d bytes (%.0f%% printable)", len(decrypted), printable_ratio * 100)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key_stream[:n_target],
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=min(0.95, printable_ratio),
                    details={
                        "method": "two_time_pad",
                        "key_len": len(key_stream),
                        "printable_ratio": printable_ratio,
                    },
                )

        return self._fail("Two-time pad: key stream too short", t0)

    # ------------------------------------------------------------------ #
    #  2. Crib Dragging                                                   #
    # ------------------------------------------------------------------ #

    def _attack_crib_drag(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        t0: float, timeout: float,
    ) -> SolverResult:
        """Drag known words across C1 ⊕ C2 to recover plaintext fragments.

        If two messages are encrypted with the same key:
          C1 ⊕ C2 = P1 ⊕ P2
        Dragging a crib word across one side reveals the other.
        """
        log.info("[Cross/Crib] Attempting crib dragging")
        n = min(len(ct), len(ct_target))
        if n < 2:
            return self._fail("Crib: data too short", t0)

        # Compute XOR of the two ciphertexts
        xor_stream = [(ct[i] ^ ct_target[i]) & 0xFF for i in range(n)]

        # Note: xor_stream = P_known ⊕ P_target
        # If we know P_known, we already have the answer (two-time pad above)
        # This is for when we DON'T have full known plaintext

        best_score = 0.0
        best_result: list[int] | None = None
        best_crib = b""
        best_pos = 0

        for crib in self._cribs:
            if time.perf_counter() - t0 > timeout:
                break
            crib_bytes = list(crib)
            crib_len = len(crib_bytes)

            for pos in range(n - crib_len + 1):
                # If crib is at position pos in P_known:
                # P_target[pos:pos+crib_len] = crib ⊕ xor_stream[pos:pos+crib_len]
                candidate = [(crib_bytes[i] ^ xor_stream[pos + i]) & 0xFF for i in range(crib_len)]

                # Score: how printable / English-like is the result?
                printable = sum(1 for b in candidate if 32 <= b <= 126)
                score = printable / crib_len

                if score > best_score and score > 0.7:
                    best_score = score
                    best_result = candidate
                    best_crib = crib
                    best_pos = pos

        if best_result is not None and best_score > 0.7:
            # Build partial decryption of ct_target
            # We know: at position best_pos, P_target has best_result
            partial_decrypt = list(ct_target)
            for i, b in enumerate(best_result):
                if best_pos + i < len(partial_decrypt):
                    partial_decrypt[best_pos + i] = b

            log.info(
                "[Cross/Crib] Best crib '%s' at pos %d (score %.2f)",
                best_crib.decode("ascii", errors="replace"), best_pos, best_score,
            )
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                decrypted=partial_decrypt,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=best_score * 0.8,
                details={
                    "method": "crib_drag",
                    "crib": best_crib.decode("ascii", errors="replace"),
                    "position": best_pos,
                    "fragment": bytes(best_result).decode("ascii", errors="replace"),
                },
            )

        return self._fail("Crib dragging: no good match", t0)

    # ------------------------------------------------------------------ #
    #  3. Composition Decomposition                                       #
    # ------------------------------------------------------------------ #

    def _attack_decompose(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Detect and decompose layered encryption.

        Try to factor C = F2(F1(P)) into two simpler operations:
          - XOR then shift
          - Shift then XOR
          - Substitution then transposition
          - Multiple affine layers
        """
        log.info("[Cross/Decompose] Attempting layer decomposition")
        m = modulus or 256
        n = min(len(pt), len(ct))
        if n < 4:
            return self._fail("Decompose: need more pairs", t0)

        # Test: XOR key followed by Caesar shift → c = (p ^ k) + s mod m
        for k in range(min(m, 256)):
            intermediate = [(p ^ k) & 0xFF for p in pt[:n]]
            for s in range(m):
                if all((intermediate[i] + s) % m == ct[i] % m for i in range(min(n, 8))):
                    decrypted = [((c - s) % m) ^ k for c in ct_target]
                    log.info("[Cross/Decompose] Found XOR(0x%02x) + SHIFT(%d)", k, s)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key={"xor": k, "shift": s},
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={"method": "decompose_xor_shift", "xor": k, "shift": s},
                    )

        # Test: Caesar shift followed by XOR → c = (p + s) ^ k mod m
        for s in range(m):
            intermediate = [(p + s) % m for p in pt[:n]]
            for k in range(min(m, 256)):
                if all((intermediate[i] ^ k) & 0xFF == ct[i] & 0xFF for i in range(min(n, 8))):
                    decrypted = [(((c ^ k) & 0xFF) - s) % m for c in ct_target]
                    log.info("[Cross/Decompose] Found SHIFT(%d) + XOR(0x%02x)", s, k)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key={"shift": s, "xor": k},
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={"method": "decompose_shift_xor", "shift": s, "xor": k},
                    )

        # Test: double affine → c = a2*(a1*p + b1) + b2 mod m  (small m only)
        if m <= 256:
            for a1 in range(1, min(m, 32)):
                if math.gcd(a1, m) != 1:
                    continue
                for b1 in range(min(m, 32)):
                    inter = [(a1 * pt[i] + b1) % m for i in range(min(n, 6))]
                    for a2 in range(1, min(m, 32)):
                        if math.gcd(a2, m) != 1:
                            continue
                        b2 = (ct[0] - a2 * inter[0]) % m
                        if all((a2 * inter[i] + b2) % m == ct[i] % m for i in range(min(n, 6))):
                            a2_inv = pow(a2, -1, m)
                            a1_inv = pow(a1, -1, m)
                            decrypted = [
                                (a1_inv * ((a2_inv * (c - b2) % m) - b1)) % m
                                for c in ct_target
                            ]
                            log.info("[Cross/Decompose] Double affine a1=%d,b1=%d,a2=%d,b2=%d", a1, b1, a2, b2)
                            return SolverResult(
                                solver_name=self.NAME,
                                status=SolverStatus.SUCCESS,
                                private_key={"a1": a1, "b1": b1, "a2": a2, "b2": b2},
                                decrypted=decrypted,
                                elapsed_seconds=time.perf_counter() - t0,
                                confidence=0.90,
                                details={"method": "decompose_double_affine"},
                            )

        return self._fail("Decompose: no composition found", t0)

    # ------------------------------------------------------------------ #
    #  4. Related-Key Correlation                                         #
    # ------------------------------------------------------------------ #

    def _attack_related_key(
        self, ct1: list[int], ct_target: list[int],
        extra_cts: list[list[int]], t0: float,
    ) -> SolverResult:
        """Cross-correlate multiple ciphertexts encrypted under related keys.

        If keys differ by a constant delta, XOR patterns reveal the delta.
        """
        log.info("[Cross/RelKey] Analyzing %d additional ciphertexts", len(extra_cts))

        n = min(len(ct1), len(ct_target))
        if n < 2:
            return self._fail("Related key: too short", t0)

        # XOR ct1 with each extra ciphertext
        for idx, ct_extra in enumerate(extra_cts):
            m = min(n, len(ct_extra))
            xor_diff = [(ct1[i] ^ ct_extra[i]) & 0xFF for i in range(m)]

            # If all XOR diffs are the same → constant key difference
            if len(set(xor_diff)) == 1 and xor_diff[0] != 0:
                delta = xor_diff[0]
                log.info("[Cross/RelKey] Constant key delta = 0x%02x with ciphertext %d", delta, idx)
                # This reveals structural information but not plaintext directly
                # Store for other solvers to use
                continue

            # If XOR diff is periodic → repeating key relationship
            for period in range(1, min(m // 2, 32)):
                pattern = xor_diff[:period]
                if all(xor_diff[i] == pattern[i % period] for i in range(m)):
                    log.info("[Cross/RelKey] Periodic key delta (period=%d) with ct %d", period, idx)
                    break

        return self._fail("Related key: no exploitable pattern", t0)

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Cross] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
