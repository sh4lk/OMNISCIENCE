"""Classical Cipher Solvers.

Covers historical and CTF-common ciphers that are NOT asymmetric per se
but frequently appear in "custom crypto" challenges:

  1.  Caesar / ROT-N  (shift cipher)
  2.  Affine cipher: c = a*p + b mod 26
  3.  Vigenère / repeating-key XOR
  4.  Monoalphabetic substitution (frequency analysis)
  5.  Beaufort cipher
  6.  Autokey cipher
  7.  Hill cipher (matrix cipher mod 26)
  8.  Rail-fence / columnar transposition
  9.  Playfair (bigram substitution)
  10. Generic XOR multi-byte key

Detection is automatic via statistical analysis of the known pairs.
"""

from __future__ import annotations

import logging
import math
import string
import time
from collections import Counter
from itertools import product
from typing import Any

import numpy as np

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)

# English letter frequencies (A-Z), used for scoring plaintext candidates
ENGLISH_FREQ = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202,  # A-G
    0.0609, 0.0697, 0.0015, 0.0077, 0.0403, 0.0241, 0.0675,  # H-N
    0.0751, 0.0193, 0.0010, 0.0599, 0.0633, 0.0906, 0.0276,  # O-U
    0.0098, 0.0236, 0.0015, 0.0197, 0.0007,                    # V-Z
]


class ClassicalCipherSolver:
    """Solver for classical / historical ciphers."""

    NAME = "classical"

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

            if not pt or not ct:
                return self._fail("No known pairs", t0)

            # Try each classical cipher in order of speed
            for name, method in [
                ("caesar", self._attack_caesar),
                ("affine", self._attack_affine),
                ("xor_single", self._attack_xor_single),
                ("xor_multi", self._attack_xor_multi),
                ("vigenere", self._attack_vigenere),
                ("hill_2x2", self._attack_hill),
                ("substitution", self._attack_substitution),
                ("beaufort", self._attack_beaufort),
                ("autokey", self._attack_autokey),
                ("rail_fence", self._attack_rail_fence),
            ]:
                if time.perf_counter() - t0 > timeout:
                    break
                res = method(pt, ct, ct_target, instance.modulus, t0)
                if res.status == SolverStatus.SUCCESS:
                    log.info("[Classical/%s] Success", name)
                    return res

            return self._fail("No classical cipher matched", t0)

        except Exception as exc:
            log.exception("Classical solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  1. Caesar / ROT-N                                                  #
    # ------------------------------------------------------------------ #

    def _attack_caesar(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Try all 256 shifts (or mod m shifts)."""
        m = modulus or 256
        n = min(len(pt), len(ct))

        for shift in range(m):
            if all((pt[i] + shift) % m == ct[i] % m for i in range(min(n, 10))):
                decrypted = [(c - shift) % m for c in ct_target]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=shift,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={"method": "caesar", "shift": shift, "modulus": m},
                )

        return self._fail("Caesar: no shift matched", t0)

    # ------------------------------------------------------------------ #
    #  2. Affine cipher: c = a*p + b mod m                                #
    # ------------------------------------------------------------------ #

    def _attack_affine(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Brute-force affine key (a, b) with a coprime to m."""
        m = modulus or 256
        n = min(len(pt), len(ct))
        if n < 2:
            return self._fail("Affine needs ≥2 pairs", t0)

        for a in range(1, m):
            if math.gcd(a, m) != 1:
                continue
            b = (ct[0] - a * pt[0]) % m
            # Verify
            if all((a * pt[i] + b) % m == ct[i] % m for i in range(min(n, 10))):
                a_inv = pow(a, -1, m)
                decrypted = [(a_inv * (c - b)) % m for c in ct_target]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=[a, b],
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={"method": "affine", "a": a, "b": b, "modulus": m},
                )

        return self._fail("Affine: no (a, b) matched", t0)

    # ------------------------------------------------------------------ #
    #  3. Single-byte XOR                                                 #
    # ------------------------------------------------------------------ #

    def _attack_xor_single(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        n = min(len(pt), len(ct))
        for key in range(256):
            if all((pt[i] ^ key) & 0xFF == ct[i] & 0xFF for i in range(min(n, 10))):
                decrypted = [(c ^ key) & 0xFF for c in ct_target]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={"method": "xor_single", "key": hex(key)},
                )
        return self._fail("XOR single: no match", t0)

    # ------------------------------------------------------------------ #
    #  4. Multi-byte XOR / repeating key                                  #
    # ------------------------------------------------------------------ #

    def _attack_xor_multi(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Recover repeating XOR key from known plaintext."""
        n = min(len(pt), len(ct))
        if n < 2:
            return self._fail("XOR multi needs ≥2 bytes", t0)

        # Extract key stream from known pairs
        key_stream = [(pt[i] ^ ct[i]) & 0xFF for i in range(n)]

        # Find the key period via autocorrelation
        for period in range(1, min(n // 2 + 1, 64)):
            key = key_stream[:period]
            # Verify: does repeating this key reproduce all known ct?
            match = True
            for i in range(n):
                if (pt[i] ^ key[i % period]) & 0xFF != ct[i] & 0xFF:
                    match = False
                    break
            if match:
                decrypted = [(ct_target[i] ^ key[i % period]) & 0xFF for i in range(len(ct_target))]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={
                        "method": "xor_multi",
                        "key": [hex(k) for k in key],
                        "period": period,
                    },
                )

        return self._fail("XOR multi: no repeating key found", t0)

    # ------------------------------------------------------------------ #
    #  5. Vigenère cipher                                                 #
    # ------------------------------------------------------------------ #

    def _attack_vigenere(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Vigenère: c_i = (p_i + k_{i mod L}) mod m.

        With known plaintext, key recovery is trivial.
        Without, use Kasiski + frequency analysis.
        """
        m = modulus or 26
        n = min(len(pt), len(ct))

        if n >= 2:
            # Known-plaintext: extract key
            key_stream = [(ct[i] - pt[i]) % m for i in range(n)]

            # Find period
            for period in range(1, min(n // 2 + 1, 64)):
                key = key_stream[:period]
                if all((pt[i] + key[i % period]) % m == ct[i] % m for i in range(n)):
                    decrypted = [(ct_target[i] - key[i % period]) % m for i in range(len(ct_target))]
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=key,
                        decrypted=decrypted,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.95,
                        details={
                            "method": "vigenere_kpa",
                            "key": key,
                            "period": period,
                        },
                    )

        # Ciphertext-only: Kasiski examination + frequency analysis
        if len(ct_target) >= 20 and m <= 26:
            result = self._vigenere_ciphertext_only(ct_target, m, t0)
            if result is not None:
                return result

        return self._fail("Vigenère: no key found", t0)

    def _vigenere_ciphertext_only(
        self, ct: list[int], m: int, t0: float,
    ) -> SolverResult | None:
        """Kasiski + Index of Coincidence + frequency analysis."""
        n = len(ct)

        # Estimate key length via Index of Coincidence
        best_period = 1
        best_ic = 0.0
        for period in range(1, min(20, n // 3)):
            ic_sum = 0.0
            count = 0
            for offset in range(period):
                group = ct[offset::period]
                ic_sum += self._index_of_coincidence(group, m)
                count += 1
            avg_ic = ic_sum / max(count, 1)
            # English IC ≈ 0.065, random ≈ 0.038 (for mod 26)
            if avg_ic > best_ic:
                best_ic = avg_ic
                best_period = period

        if best_ic < 0.05:
            return None  # doesn't look like Vigenère

        # Recover each key byte by frequency analysis
        key = []
        for offset in range(best_period):
            group = ct[offset::best_period]
            best_shift = 0
            best_score = -float("inf")
            for shift in range(m):
                shifted = [(c - shift) % m for c in group]
                score = self._frequency_score(shifted, m)
                if score > best_score:
                    best_score = score
                    best_shift = shift
            key.append(best_shift)

        decrypted = [(ct[i] - key[i % best_period]) % m for i in range(n)]
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.SUCCESS,
            private_key=key,
            decrypted=decrypted,
            elapsed_seconds=time.perf_counter() - t0,
            confidence=0.70,
            details={
                "method": "vigenere_ciphertext_only",
                "key": key,
                "period": best_period,
                "ic": best_ic,
            },
        )

    # ------------------------------------------------------------------ #
    #  6. Beaufort cipher: c = (k - p) mod m                              #
    # ------------------------------------------------------------------ #

    def _attack_beaufort(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        m = modulus or 26
        n = min(len(pt), len(ct))
        if n < 2:
            return self._fail("Beaufort needs pairs", t0)

        key_stream = [(ct[i] + pt[i]) % m for i in range(n)]  # k = c + p mod m
        # Note: Beaufort c = k - p mod m  →  k = c + p mod m

        for period in range(1, min(n // 2 + 1, 64)):
            key = key_stream[:period]
            if all((key[i % period] - pt[i]) % m == ct[i] % m for i in range(n)):
                decrypted = [(key[i % period] - ct_target[i]) % m for i in range(len(ct_target))]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.90,
                    details={"method": "beaufort", "key": key, "period": period},
                )

        return self._fail("Beaufort: no match", t0)

    # ------------------------------------------------------------------ #
    #  7. Autokey cipher                                                  #
    # ------------------------------------------------------------------ #

    def _attack_autokey(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Autokey: c_0 = (p_0 + key) mod m, c_i = (p_i + p_{i-1}) mod m.

        With known plaintext the key seed is trivially recovered.
        """
        m = modulus or 26
        n = min(len(pt), len(ct))
        if n < 2:
            return self._fail("Autokey needs pairs", t0)

        # Try: c_i = (p_i + key_stream_i) mod m where key_stream = [seed, p_0, p_1, ...]
        seed = (ct[0] - pt[0]) % m
        # Verify autokey with plaintext feedback
        ok = True
        for i in range(1, n):
            expected = (pt[i] + pt[i - 1]) % m
            if expected != ct[i] % m:
                ok = False
                break
        if ok:
            # Decrypt target
            decrypted = []
            prev = seed  # The seed acts as "previous plaintext" for first byte
            # Actually: c_0 = (p_0 + seed) mod m → p_0 = (c_0 - seed) mod m
            # c_i = (p_i + p_{i-1}) mod m → p_i = (c_i - p_{i-1}) mod m
            for i, c in enumerate(ct_target):
                if i == 0:
                    p = (c - seed) % m
                else:
                    p = (c - decrypted[i - 1]) % m
                decrypted.append(p)
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=seed,
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={"method": "autokey", "seed": seed},
            )

        return self._fail("Autokey: pattern mismatch", t0)

    # ------------------------------------------------------------------ #
    #  8. Hill cipher (2×2 matrix mod m)                                  #
    # ------------------------------------------------------------------ #

    def _attack_hill(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Recover 2×2 Hill cipher key matrix from known plaintext.

        [c0, c1] = K × [p0, p1]^T mod m
        """
        m = modulus or 26
        n = min(len(pt), len(ct))
        if n < 4:
            return self._fail("Hill 2x2 needs ≥4 values (2 pairs)", t0)

        # Build system: [P0 | P1] × K^T = [C0 | C1]
        # P = [[p0, p1], [p2, p3]], C = [[c0, c1], [c2, c3]]
        P = [[pt[0] % m, pt[1] % m], [pt[2] % m, pt[3] % m]]
        C = [[ct[0] % m, ct[1] % m], [ct[2] % m, ct[3] % m]]

        # K = C × P^{-1} mod m
        det = (P[0][0] * P[1][1] - P[0][1] * P[1][0]) % m
        det_inv = self._modinv(det, m)
        if det_inv is None:
            return self._fail("Hill: plaintext matrix not invertible", t0)

        P_inv = [
            [(P[1][1] * det_inv) % m, ((-P[0][1]) * det_inv) % m],
            [((-P[1][0]) * det_inv) % m, (P[0][0] * det_inv) % m],
        ]

        # K = C × P_inv
        K = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                K[i][j] = (C[i][0] * P_inv[0][j] + C[i][1] * P_inv[1][j]) % m

        # Verify with remaining pairs if available
        if n >= 6:
            v_ct = [
                (K[0][0] * pt[4] + K[0][1] * pt[5]) % m,
                (K[1][0] * pt[4] + K[1][1] * pt[5]) % m,
            ]
            if v_ct[0] != ct[4] % m or v_ct[1] != ct[5] % m:
                return self._fail("Hill: verification failed", t0)

        # Compute K_inv for decryption
        det_K = (K[0][0] * K[1][1] - K[0][1] * K[1][0]) % m
        det_K_inv = self._modinv(det_K, m)
        if det_K_inv is None:
            return self._fail("Hill: key matrix not invertible", t0)

        K_inv = [
            [(K[1][1] * det_K_inv) % m, ((-K[0][1]) * det_K_inv) % m],
            [((-K[1][0]) * det_K_inv) % m, (K[0][0] * det_K_inv) % m],
        ]

        # Decrypt target in blocks of 2
        decrypted = []
        for i in range(0, len(ct_target) - 1, 2):
            c0, c1 = ct_target[i] % m, ct_target[i + 1] % m
            p0 = (K_inv[0][0] * c0 + K_inv[0][1] * c1) % m
            p1 = (K_inv[1][0] * c0 + K_inv[1][1] * c1) % m
            decrypted.extend([p0, p1])
        # Handle odd trailing byte
        if len(ct_target) % 2 == 1:
            decrypted.append(ct_target[-1])

        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.SUCCESS,
            private_key=K,
            decrypted=decrypted,
            elapsed_seconds=time.perf_counter() - t0,
            confidence=0.90,
            details={"method": "hill_2x2", "key_matrix": K, "key_inv": K_inv},
        )

    # ------------------------------------------------------------------ #
    #  9. Monoalphabetic substitution (frequency analysis)                #
    # ------------------------------------------------------------------ #

    def _attack_substitution(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Known-plaintext recovery of the substitution table.

        With enough known pairs, build the lookup directly.
        Without, use frequency analysis on the ciphertext.
        """
        n = min(len(pt), len(ct))
        m = modulus or 256

        # Build forward map from known pairs
        fwd: dict[int, int] = {}
        for i in range(n):
            p, c = pt[i] % m, ct[i] % m
            if p in fwd and fwd[p] != c:
                return self._fail("Substitution: inconsistent mapping", t0)
            fwd[p] = c

        # Check bijectivity of known part
        if len(set(fwd.values())) != len(fwd):
            return self._fail("Substitution: not injective", t0)

        # Build inverse map
        inv: dict[int, int] = {c: p for p, c in fwd.items()}

        # Decrypt target
        decrypted = []
        missing = False
        for c in ct_target:
            c_mod = c % m
            if c_mod in inv:
                decrypted.append(inv[c_mod])
            else:
                missing = True
                decrypted.append(c_mod)  # pass through unknown

        if not missing:
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=list(fwd.items()),
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.95,
                details={"method": "substitution_kpa", "known_mappings": len(fwd)},
            )

        # If coverage > 80%, still report with reduced confidence
        coverage = len(inv) / max(len(set(c % m for c in ct_target)), 1)
        if coverage > 0.8:
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=list(fwd.items()),
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=coverage * 0.9,
                details={"method": "substitution_partial", "coverage": coverage},
            )

        return self._fail("Substitution: insufficient known mappings", t0)

    # ------------------------------------------------------------------ #
    #  10. Rail-fence transposition                                       #
    # ------------------------------------------------------------------ #

    def _attack_rail_fence(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        modulus: int | None, t0: float,
    ) -> SolverResult:
        """Brute-force rail-fence with 2..20 rails."""
        n_known = min(len(pt), len(ct))
        if n_known < 4:
            return self._fail("Rail-fence needs longer text", t0)

        for rails in range(2, min(21, n_known)):
            encrypted = self._rail_fence_encrypt(pt[:n_known], rails)
            if encrypted == ct[:n_known]:
                decrypted = self._rail_fence_decrypt(ct_target, rails)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=rails,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.90,
                    details={"method": "rail_fence", "rails": rails},
                )

        return self._fail("Rail-fence: no rail count matched", t0)

    @staticmethod
    def _rail_fence_encrypt(data: list[int], rails: int) -> list[int]:
        if rails <= 1:
            return list(data)
        fence = [[] for _ in range(rails)]
        rail, direction = 0, 1
        for val in data:
            fence[rail].append(val)
            if rail == 0:
                direction = 1
            elif rail == rails - 1:
                direction = -1
            rail += direction
        return [v for row in fence for v in row]

    @staticmethod
    def _rail_fence_decrypt(data: list[int], rails: int) -> list[int]:
        if rails <= 1:
            return list(data)
        n = len(data)
        # Compute row lengths
        pattern = list(range(rails)) + list(range(rails - 2, 0, -1))
        cycle = len(pattern)
        row_lens = [0] * rails
        for i in range(n):
            row_lens[pattern[i % cycle]] += 1

        # Fill fence
        fence = []
        idx = 0
        for r in range(rails):
            fence.append(data[idx:idx + row_lens[r]])
            idx += row_lens[r]

        # Read off in zigzag order
        row_idx = [0] * rails
        result = []
        rail, direction = 0, 1
        for _ in range(n):
            result.append(fence[rail][row_idx[rail]])
            row_idx[rail] += 1
            if rail == 0:
                direction = 1
            elif rail == rails - 1:
                direction = -1
            rail += direction
        return result

    # ------------------------------------------------------------------ #
    #  Frequency analysis helpers                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _index_of_coincidence(data: list[int], m: int) -> float:
        """Compute the Index of Coincidence."""
        n = len(data)
        if n < 2:
            return 0.0
        freq = Counter(v % m for v in data)
        ic = sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))
        return ic

    @staticmethod
    def _frequency_score(data: list[int], m: int) -> float:
        """Score how well data's frequency distribution matches English."""
        if m > 26 or not data:
            return 0.0
        n = len(data)
        freq = Counter(v % m for v in data)
        score = 0.0
        for i in range(min(m, 26)):
            observed = freq.get(i, 0) / n
            score += observed * ENGLISH_FREQ[i]
        return score

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        a = a % m
        if a == 0:
            return None
        g = math.gcd(a, m)
        if g != 1:
            return None
        return pow(a, -1, m)

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Classical] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
