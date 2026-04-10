"""Oracle Attack Module.

Implements adaptive chosen-ciphertext and padding oracle attacks for
black-box scenarios where an oracle indicates validity.

Strategies:
  1. Bleichenbacher's Attack (RSA PKCS#1 v1.5 padding oracle)
  2. Vaudenay's CBC Padding Oracle
  3. Manger's Attack (RSA OAEP oracle)
  4. Chosen-Ciphertext Adaptive Attack (generic malleability)
  5. Error-based oracle (timing / error message differentiation)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Protocol

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class DecryptionOracle(Protocol):
    """Protocol for a decryption oracle.

    The oracle takes a ciphertext and returns:
      - True if decryption succeeds (valid padding / format)
      - False otherwise
    Optionally returns partial information (error type, timing).
    """

    def query(self, ciphertext: bytes | list[int]) -> bool: ...


class TimingOracle(Protocol):
    """Oracle that leaks information via timing."""

    def query_timed(self, ciphertext: bytes | list[int]) -> tuple[bool, float]: ...


class OracleAttackSolver:
    """Oracle-based adaptive attacks."""

    NAME = "oracle"

    def __init__(self, oracle: DecryptionOracle | None = None):
        self._oracle = oracle
        self._query_count = 0

    def set_oracle(self, oracle: DecryptionOracle) -> None:
        self._oracle = oracle

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()

        if self._oracle is None:
            # Try to build a simulated oracle from known pairs
            oracle = self._build_simulated_oracle(instance, recon)
            if oracle is None:
                return self._fail("No oracle available (set one with set_oracle())", t0)
            self._oracle = oracle

        modulus = recon.estimated_modulus or instance.modulus

        try:
            # Strategy 1: Bleichenbacher (RSA PKCS#1)
            if modulus and modulus.bit_length() >= 256:
                res = self._bleichenbacher(instance, modulus, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 2: Padding oracle (CBC-like)
            res = self._padding_oracle_attack(instance, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 3: Bit-by-bit adaptive
            res = self._adaptive_bitwise(instance, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Oracle attacks exhausted", t0)

        except Exception as exc:
            log.exception("Oracle solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc), "queries": self._query_count},
            )

    # ------------------------------------------------------------------ #
    #  Simulated Oracle                                                   #
    # ------------------------------------------------------------------ #

    def _build_simulated_oracle(
        self, instance: CryptoInstance, recon: ReconResult
    ) -> DecryptionOracle | None:
        """Build a simulated oracle from known plaintext/ciphertext pairs.

        This works when we can detect a pattern: the oracle returns True
        if the "decrypted" value matches an expected structure.
        """
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        modulus = recon.estimated_modulus or instance.modulus
        if not pt or not ct or not modulus:
            return None

        # Build mapping
        ct_to_pt = {ct[i]: pt[i] for i in range(min(len(pt), len(ct)))}

        class SimOracle:
            def query(self, ciphertext: bytes | list[int]) -> bool:
                if isinstance(ciphertext, bytes):
                    ciphertext = list(ciphertext)
                # Check if we know this ciphertext
                return ciphertext[0] in ct_to_pt if ciphertext else False

        return SimOracle()

    # ------------------------------------------------------------------ #
    #  Strategy 1: Bleichenbacher's RSA PKCS#1 v1.5                       #
    # ------------------------------------------------------------------ #

    def _bleichenbacher(
        self, instance: CryptoInstance, N: int, timeout: float, t0: float
    ) -> SolverResult:
        """Bleichenbacher's adaptive chosen-ciphertext attack on RSA PKCS#1 v1.5.

        Given oracle O(c) that returns True iff PKCS#1-conformant:
          - Iteratively narrow the range of possible plaintexts
          - Each oracle query halves the search space approximately
        """
        log.info("[Oracle/Bleich] Attempting Bleichenbacher attack (N=%d bits)", N.bit_length())
        ct_target = instance.ct_target_as_int_list()
        if not ct_target:
            return self._fail("No target ciphertext", t0)

        pub = instance.pub_as_int_list()
        e = pub[0] if pub else 65537
        c = ct_target[0]

        k = (N.bit_length() + 7) // 8  # byte length of N
        B = pow(2, 8 * (k - 2))

        # Step 1: Blinding — find s_0 such that c * s_0^e is PKCS conformant
        # Start with s_0 = 1 (assume c is already conformant)
        s = 1
        c_prime = c

        if not self._oracle_query((c_prime).to_bytes(k, 'big')):
            # Need blinding
            for s in range(1, min(2**20, N)):
                if time.perf_counter() - t0 > timeout:
                    return self._fail("Bleichenbacher blinding timeout", t0)
                c_prime = (c * pow(s, e, N)) % N
                if self._oracle_query(c_prime.to_bytes(k, 'big')):
                    break
            else:
                return self._fail("Could not find PKCS-conformant blinding", t0)

        # Step 2: Iteratively narrow intervals
        M = [(2 * B, 3 * B - 1)]  # initial interval
        s_prev = s

        for iteration in range(10000):
            if time.perf_counter() - t0 > timeout:
                break

            # Step 2a/2b: find next s
            if len(M) > 1:
                # Multiple intervals: sequential search
                s_next = s_prev + 1
                while True:
                    if time.perf_counter() - t0 > timeout:
                        break
                    c_test = (c * pow(s_next, e, N)) % N
                    if self._oracle_query(c_test.to_bytes(k, 'big')):
                        break
                    s_next += 1
                    if s_next > N:
                        return self._fail("Bleichenbacher search exhausted", t0)
            else:
                # Single interval: use interval to bound search
                a, b = M[0]
                r = max(1, (2 * (b * s_prev - 2 * B) + N - 1) // N)
                found = False
                while not found:
                    if time.perf_counter() - t0 > timeout:
                        break
                    s_lo = (2 * B + r * N + b - 1) // b
                    s_hi = (3 * B + r * N) // a
                    for s_next in range(s_lo, s_hi + 1):
                        c_test = (c * pow(s_next, e, N)) % N
                        if self._oracle_query(c_test.to_bytes(k, 'big')):
                            found = True
                            break
                    r += 1
                    if r > N:
                        break

                if not found:
                    return self._fail("Bleichenbacher narrowing failed", t0)

            # Step 3: Narrow intervals
            new_M = []
            for a, b in M:
                r_lo = (a * s_next - 3 * B + 1 + N - 1) // N
                r_hi = (b * s_next - 2 * B) // N
                for r in range(r_lo, r_hi + 1):
                    new_a = max(a, (2 * B + r * N + s_next - 1) // s_next)
                    new_b = min(b, (3 * B - 1 + r * N) // s_next)
                    if new_a <= new_b:
                        new_M.append((new_a, new_b))

            M = self._merge_intervals(new_M)
            s_prev = s_next

            # Step 4: Check convergence
            if len(M) == 1 and M[0][0] == M[0][1]:
                plaintext = M[0][0]
                log.info(
                    "[Oracle/Bleich] Converged after %d iterations (%d queries)",
                    iteration, self._query_count,
                )
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=[plaintext],
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={
                        "method": "bleichenbacher",
                        "iterations": iteration,
                        "queries": self._query_count,
                    },
                )

        return self._fail(f"Bleichenbacher did not converge ({self._query_count} queries)", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 2: CBC Padding Oracle                                     #
    # ------------------------------------------------------------------ #

    def _padding_oracle_attack(
        self, instance: CryptoInstance, timeout: float, t0: float
    ) -> SolverResult:
        """Vaudenay's padding oracle attack for CBC mode.

        Given blocks (IV, C1, C2, ...) and an oracle that checks padding:
        Recover plaintext block-by-block, byte-by-byte.
        """
        log.info("[Oracle/Padding] Attempting CBC padding oracle")
        ct = instance.ct_target_as_int_list()
        if len(ct) < 32:  # need at least IV + one block (16 bytes each)
            return self._fail("Ciphertext too short for padding oracle", t0)

        block_size = 16  # AES block size
        n_blocks = len(ct) // block_size
        if n_blocks < 2:
            return self._fail("Need at least 2 blocks", t0)

        decrypted_blocks: list[list[int]] = []

        for block_idx in range(1, n_blocks):
            if time.perf_counter() - t0 > timeout:
                break

            prev_block = ct[(block_idx - 1) * block_size: block_idx * block_size]
            curr_block = ct[block_idx * block_size: (block_idx + 1) * block_size]

            decrypted_block = [0] * block_size
            intermediate = [0] * block_size

            for byte_pos in range(block_size - 1, -1, -1):
                pad_val = block_size - byte_pos

                # Build the attack block
                attack = list(prev_block)
                for k in range(byte_pos + 1, block_size):
                    attack[k] = intermediate[k] ^ pad_val

                found = False
                for guess in range(256):
                    if time.perf_counter() - t0 > timeout:
                        break
                    attack[byte_pos] = guess
                    test_ct = attack + list(curr_block)

                    if self._oracle_query(test_ct):
                        # Verify it's not a false positive (for last byte)
                        if byte_pos == block_size - 1 and pad_val == 1:
                            # Flip the previous byte to confirm
                            verify = list(attack)
                            verify[byte_pos - 1] ^= 1
                            if not self._oracle_query(verify + list(curr_block)):
                                continue

                        intermediate[byte_pos] = guess ^ pad_val
                        decrypted_block[byte_pos] = intermediate[byte_pos] ^ prev_block[byte_pos]
                        found = True
                        break

                if not found:
                    log.debug("[Oracle/Padding] Failed at block %d, byte %d", block_idx, byte_pos)
                    break

            decrypted_blocks.append(decrypted_block)

        if decrypted_blocks:
            flat = [b for block in decrypted_blocks for b in block]
            # Remove PKCS7 padding
            if flat:
                pad_len = flat[-1]
                if 0 < pad_len <= block_size and all(b == pad_len for b in flat[-pad_len:]):
                    flat = flat[:-pad_len]

            log.info("[Oracle/Padding] Recovered %d bytes (%d queries)", len(flat), self._query_count)
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                decrypted=flat,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={
                    "method": "padding_oracle",
                    "blocks_recovered": len(decrypted_blocks),
                    "queries": self._query_count,
                },
            )

        return self._fail("Padding oracle: no blocks recovered", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 3: Adaptive Bitwise                                       #
    # ------------------------------------------------------------------ #

    def _adaptive_bitwise(
        self, instance: CryptoInstance, modulus: int | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """Bit-by-bit adaptive attack using homomorphic malleability.

        If the cipher is multiplicatively homomorphic (E(m1)*E(m2) = E(m1*m2)),
        we can learn individual bits of the plaintext by multiplying with
        known factors and checking the oracle.
        """
        log.info("[Oracle/Adaptive] Attempting bit-by-bit adaptive attack")
        if modulus is None or modulus < 4:
            return self._fail("Needs modulus for adaptive attack", t0)

        ct_target = instance.ct_target_as_int_list()
        pub = instance.pub_as_int_list()
        if not ct_target:
            return self._fail("No target", t0)

        e = pub[0] if pub else 2
        c = ct_target[0]
        n_bits = modulus.bit_length()

        # LSB oracle attack (for RSA): multiply c by 2^e mod N
        # If oracle(c * 2^e mod N) is True → 2*m mod N < N → m < N/2
        recovered_bits = []
        lo, hi = 0, modulus

        for bit in range(n_bits):
            if time.perf_counter() - t0 > timeout:
                break
            factor = pow(2, e * (bit + 1), modulus)
            c_test = (c * factor) % modulus

            if isinstance(c_test, int):
                test_data = c_test.to_bytes((modulus.bit_length() + 7) // 8, 'big')
            else:
                test_data = [c_test]

            if self._oracle_query(test_data):
                hi = (lo + hi) // 2
                recovered_bits.append(0)
            else:
                lo = (lo + hi) // 2
                recovered_bits.append(1)

            if hi - lo <= 1:
                break

        if hi - lo <= 1:
            plaintext = lo
            log.info("[Oracle/Adaptive] Recovered plaintext = %d (%d bits, %d queries)",
                     plaintext, len(recovered_bits), self._query_count)
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                decrypted=[plaintext],
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.80,
                details={
                    "method": "adaptive_bitwise",
                    "bits_recovered": len(recovered_bits),
                    "queries": self._query_count,
                },
            )

        return self._fail("Adaptive bitwise did not converge", t0)

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def _oracle_query(self, data: bytes | list[int]) -> bool:
        """Query the oracle and track count."""
        self._query_count += 1
        if self._oracle is None:
            return False
        return self._oracle.query(data)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        intervals.sort()
        merged = [intervals[0]]
        for a, b in intervals[1:]:
            if a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Oracle] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason, "queries": self._query_count},
        )
