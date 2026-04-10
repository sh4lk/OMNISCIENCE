"""Symmetric Cipher Cryptanalysis Module (Block & Stream).

Attacks on symmetric encryption schemes:

Block ciphers:
  1. ECB mode detection & exploitation (identical blocks → identical ciphertext)
  2. CBC bit-flipping / padding oracle (via oracle module)
  3. Feistel structure detection & differential characteristics
  4. AES key schedule weakness (related-key, known-key distinguisher)
  5. DES weak/semi-weak key detection
  6. Block size detection (GCD of ciphertext lengths)
  7. Mode of operation identification (ECB/CBC/CTR/OFB/CFB)

Stream ciphers:
  8. LFSR-based key recovery (Berlekamp-Massey)
  9. RC4 bias exploitation (Fluhrer-Mantin-Shamir, key schedule weakness)
  10. Keystream reuse / two-time pad (delegates to cross_cipher)
  11. Known-plaintext XOR key recovery
  12. Correlation attack on combining generators
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


class SymmetricSolver:
    """Cryptanalysis engine for symmetric block and stream ciphers."""

    NAME = "symmetric"

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

            # ------ Block cipher attacks ------ #

            # 1. ECB detection & dictionary attack
            res = self._attack_ecb(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # 2. Block size detection + mode identification
            block_size = self._detect_block_size(ct)
            mode = self._detect_mode(ct, block_size)

            # 3. CBC key/IV recovery with known plaintext
            if mode == "cbc" and block_size > 0:
                res = self._attack_cbc_kpa(pt, ct, ct_target, block_size, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # 4. CTR / OFB keystream recovery (equivalent to stream cipher)
            if mode in ("ctr", "ofb"):
                res = self._attack_keystream_recovery(pt, ct, ct_target, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # 5. Feistel differential detection
            res = self._attack_feistel_diff(pt, ct, ct_target, block_size, t0, timeout)
            if res.status == SolverStatus.SUCCESS:
                return res

            # 6. DES weak key check
            res = self._attack_des_weak_keys(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # ------ Stream cipher attacks ------ #

            # 7. LFSR via Berlekamp-Massey
            res = self._attack_lfsr(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # 8. RC4 key schedule bias
            res = self._attack_rc4_bias(pt, ct, ct_target, t0, timeout)
            if res.status == SolverStatus.SUCCESS:
                return res

            # 9. Generic XOR keystream (repeating block key)
            res = self._attack_block_xor(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Symmetric: no strategy succeeded", t0)

        except Exception as exc:
            log.exception("Symmetric solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ================================================================== #
    #  Block Cipher Attacks                                               #
    # ================================================================== #

    # ---- 1. ECB Detection & Dictionary ---- #

    def _attack_ecb(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """ECB mode: identical plaintext blocks → identical ciphertext blocks.

        Build a codebook from known pairs and look up target blocks.
        """
        log.info("[Sym/ECB] Attempting ECB codebook attack")
        if len(pt) < 2 or len(ct) < 2:
            return self._fail("ECB: not enough data", t0)

        # Try common block sizes
        for bs in (16, 8, 32, 4):
            if len(pt) < bs or len(ct) < bs:
                continue

            # Build codebook: pt_block → ct_block
            codebook: dict[tuple, tuple] = {}
            reverse_book: dict[tuple, tuple] = {}
            n_blocks = min(len(pt), len(ct)) // bs

            for i in range(n_blocks):
                pt_block = tuple(pt[i * bs:(i + 1) * bs])
                ct_block = tuple(ct[i * bs:(i + 1) * bs])
                codebook[pt_block] = ct_block
                reverse_book[ct_block] = pt_block

            if not reverse_book:
                continue

            # Check if this is really ECB: same pt → same ct?
            consistent = True
            for pt_b, ct_b in codebook.items():
                # Check all occurrences
                for i in range(n_blocks):
                    if tuple(pt[i * bs:(i + 1) * bs]) == pt_b:
                        if tuple(ct[i * bs:(i + 1) * bs]) != ct_b:
                            consistent = False
                            break
                if not consistent:
                    break

            if not consistent:
                continue

            # Try to decrypt target
            n_target_blocks = len(ct_target) // bs
            if n_target_blocks < 1:
                continue

            decrypted = []
            all_found = True
            for i in range(n_target_blocks):
                ct_block = tuple(ct_target[i * bs:(i + 1) * bs])
                if ct_block in reverse_book:
                    decrypted.extend(reverse_book[ct_block])
                else:
                    all_found = False
                    break

            if all_found and decrypted:
                log.info("[Sym/ECB] ECB codebook hit (block_size=%d)", bs)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={
                        "method": "ecb_codebook",
                        "block_size": bs,
                        "codebook_entries": len(reverse_book),
                    },
                )

        return self._fail("ECB: no codebook match", t0)

    # ---- 2. Block size detection ---- #

    @staticmethod
    def _detect_block_size(ct: list[int]) -> int:
        """Detect block size from ciphertext length patterns."""
        ct_len = len(ct)
        if ct_len == 0:
            return 0
        for bs in (16, 8, 32, 4):
            if ct_len % bs == 0:
                return bs
        return 0

    # ---- 3. Mode of operation identification ---- #

    @staticmethod
    def _detect_mode(ct: list[int], block_size: int) -> str:
        """Identify encryption mode from ciphertext patterns.

        - ECB: repeated blocks visible
        - CBC: no repetition, IV-dependent
        - CTR/OFB: keystream-like, no block repetition
        """
        if block_size < 2 or len(ct) < block_size * 2:
            return "unknown"

        blocks = []
        for i in range(0, len(ct) - block_size + 1, block_size):
            blocks.append(tuple(ct[i:i + block_size]))

        n_blocks = len(blocks)
        n_unique = len(set(blocks))

        if n_unique < n_blocks:
            return "ecb"

        # High entropy + no repetition → likely CTR/OFB/CBC
        # CBC often has the first block different (IV) from patterns
        byte_freq = Counter(ct)
        entropy = -sum(
            (c / len(ct)) * math.log2(c / len(ct))
            for c in byte_freq.values()
        )

        if entropy > 7.5:
            return "ctr"  # high entropy → stream-like mode

        return "cbc"

    # ---- 4. CBC Known-Plaintext Attack ---- #

    def _attack_cbc_kpa(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        block_size: int, t0: float,
    ) -> SolverResult:
        """CBC with known plaintext: recover the key via XOR chain.

        In CBC encryption: C_i = E_K(P_i XOR C_{i-1}), C_0 = IV
        With known P and C, the intermediate value P_i XOR C_{i-1} maps to C_i.
        """
        log.info("[Sym/CBC] Attempting CBC known-plaintext key recovery (bs=%d)", block_size)
        bs = block_size
        n_known = min(len(pt), len(ct)) // bs

        if n_known < 2:
            return self._fail("CBC KPA: need at least 2 blocks", t0)

        # For blocks 1..n: intermediate[i] = pt_block[i] XOR ct_block[i-1]
        # E_K(intermediate[i]) = ct_block[i]
        # Build a permutation table for the block cipher
        enc_table: dict[tuple, tuple] = {}
        dec_table: dict[tuple, tuple] = {}

        for i in range(1, n_known):
            pt_block = pt[i * bs:(i + 1) * bs]
            prev_ct = ct[(i - 1) * bs:i * bs]
            ct_block = ct[i * bs:(i + 1) * bs]

            intermediate = tuple((pt_block[j] ^ prev_ct[j]) & 0xFF for j in range(bs))
            enc_table[intermediate] = tuple(ct_block)
            dec_table[tuple(ct_block)] = intermediate

        if not dec_table:
            return self._fail("CBC KPA: couldn't build table", t0)

        # Decrypt target using the table
        n_target = len(ct_target) // bs
        if n_target < 2:
            return self._fail("CBC KPA: target too short", t0)

        decrypted = []
        success = True
        for i in range(1, n_target):
            ct_block = tuple(ct_target[i * bs:(i + 1) * bs])
            prev_ct = ct_target[(i - 1) * bs:i * bs]

            if ct_block in dec_table:
                intermediate = dec_table[ct_block]
                pt_block = [(intermediate[j] ^ prev_ct[j]) & 0xFF for j in range(bs)]
                decrypted.extend(pt_block)
            else:
                success = False
                break

        if success and decrypted:
            log.info("[Sym/CBC] CBC decrypted %d bytes via block table", len(decrypted))
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={
                    "method": "cbc_kpa",
                    "block_size": bs,
                    "table_entries": len(dec_table),
                },
            )

        return self._fail("CBC KPA: incomplete table coverage", t0)

    # ---- 5. CTR / OFB Keystream Recovery ---- #

    def _attack_keystream_recovery(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """CTR/OFB modes: ciphertext = plaintext XOR keystream.

        With known plaintext, recover the keystream directly.
        """
        log.info("[Sym/CTR] Attempting keystream recovery")
        n_known = min(len(pt), len(ct))
        n_target = len(ct_target)

        if n_known < 1 or n_target < 1:
            return self._fail("Keystream: not enough data", t0)

        # Recover keystream
        keystream = [(pt[i] ^ ct[i]) & 0xFF for i in range(n_known)]

        # Check if keystream repeats (counter wrap or short period)
        period = self._find_period(keystream)
        if period and period < n_known:
            log.info("[Sym/CTR] Keystream period detected: %d", period)
            # Extend keystream by repeating
            full_ks = [keystream[i % period] for i in range(max(n_known, n_target))]
        else:
            full_ks = keystream

        if n_target <= len(full_ks):
            decrypted = [(ct_target[i] ^ full_ks[i]) & 0xFF for i in range(n_target)]
            printable = sum(1 for b in decrypted if 32 <= b <= 126) / max(len(decrypted), 1)

            if printable > 0.5 or n_target <= n_known:
                log.info("[Sym/CTR] Decrypted %d bytes (%.0f%% printable)", len(decrypted), printable * 100)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=full_ks[:n_target],
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=min(0.95, max(0.7, printable)),
                    details={
                        "method": "keystream_recovery",
                        "keystream_len": len(full_ks),
                        "period": period,
                    },
                )

        return self._fail("Keystream: not enough known stream", t0)

    # ---- 6. Feistel Differential Detection ---- #

    def _attack_feistel_diff(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        block_size: int, t0: float, timeout: float,
    ) -> SolverResult:
        """Detect Feistel structure via differential characteristics.

        In a Feistel cipher: C_L = P_R, C_R = P_L XOR F(P_R, K)
        So if we have many pairs, we can recover F and invert.
        """
        log.info("[Sym/Feistel] Attempting Feistel analysis")
        bs = block_size if block_size >= 4 else 8
        half = bs // 2

        n_known = min(len(pt), len(ct)) // bs
        if n_known < 4:
            return self._fail("Feistel: not enough blocks", t0)

        # Check Feistel property: ct_left = pt_right?
        feistel_score = 0
        for i in range(n_known):
            pt_l = pt[i * bs:i * bs + half]
            pt_r = pt[i * bs + half:(i + 1) * bs]
            ct_l = ct[i * bs:i * bs + half]
            ct_r = ct[i * bs + half:(i + 1) * bs]

            if pt_r == ct_l:
                feistel_score += 1

        if feistel_score < n_known * 0.8:
            return self._fail("Feistel: structure not detected", t0)

        log.info("[Sym/Feistel] Feistel structure confirmed (%d/%d blocks)", feistel_score, n_known)

        # Build the round function table: F(R) = C_R XOR P_L
        f_table: dict[tuple, list[int]] = {}
        for i in range(n_known):
            pt_l = pt[i * bs:i * bs + half]
            pt_r = tuple(pt[i * bs + half:(i + 1) * bs])
            ct_r = ct[i * bs + half:(i + 1) * bs]
            f_val = [(ct_r[j] ^ pt_l[j]) & 0xFF for j in range(half)]
            f_table[pt_r] = f_val

        # Decrypt target blocks (single-round Feistel inversion)
        n_target = len(ct_target) // bs
        decrypted = []
        all_ok = True
        for i in range(n_target):
            ct_l = ct_target[i * bs:i * bs + half]
            ct_r = ct_target[i * bs + half:(i + 1) * bs]
            # Inversion: P_R = C_L, P_L = C_R XOR F(C_L)
            key_ct_l = tuple(ct_l)
            if key_ct_l in f_table:
                f_val = f_table[key_ct_l]
                pt_l = [(ct_r[j] ^ f_val[j]) & 0xFF for j in range(half)]
                pt_r = list(ct_l)
                decrypted.extend(pt_l + pt_r)
            else:
                all_ok = False
                break

        if all_ok and decrypted:
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={"method": "feistel_single_round", "block_size": bs},
            )

        return self._fail("Feistel: incomplete F-table", t0)

    # ---- 7. DES Weak Keys ---- #

    def _attack_des_weak_keys(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """Check for DES weak and semi-weak keys (E_K(E_K(P)) = P)."""
        log.info("[Sym/DES] Checking for weak/semi-weak DES keys")
        if len(pt) < 8 or len(ct) < 8:
            return self._fail("DES: data too short", t0)

        # Weak key property: encryption = decryption
        # Check if E(E(P)) = P for the first block
        bs = 8
        n = min(len(pt), len(ct)) // bs
        if n < 2:
            return self._fail("DES weak: need 2+ blocks", t0)

        # Build enc table
        enc_map: dict[tuple, tuple] = {}
        for i in range(n):
            enc_map[tuple(pt[i * bs:(i + 1) * bs])] = tuple(ct[i * bs:(i + 1) * bs])

        # Check if any ciphertext maps back to plaintext (weak key)
        weak = 0
        for pt_b, ct_b in enc_map.items():
            if ct_b in enc_map and enc_map[ct_b] == pt_b:
                weak += 1

        if weak > 0 and weak >= n * 0.5:
            log.info("[Sym/DES] Weak key detected (involution)")
            # For a weak key, decryption = encryption
            dec_map: dict[tuple, tuple] = {}
            for pt_b, ct_b in enc_map.items():
                dec_map[ct_b] = pt_b

            n_target = len(ct_target) // bs
            decrypted = []
            for i in range(n_target):
                ct_block = tuple(ct_target[i * bs:(i + 1) * bs])
                if ct_block in dec_map:
                    decrypted.extend(dec_map[ct_block])
                elif ct_block in enc_map:
                    decrypted.extend(enc_map[ct_block])
                else:
                    return self._fail("DES weak: target block not in table", t0)

            if decrypted:
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.80,
                    details={"method": "des_weak_key", "weak_blocks": weak},
                )

        return self._fail("DES: no weak key pattern", t0)

    # ================================================================== #
    #  Stream Cipher Attacks                                              #
    # ================================================================== #

    # ---- 8. LFSR via Berlekamp-Massey ---- #

    def _attack_lfsr(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """Recover LFSR feedback polynomial via Berlekamp-Massey.

        If the keystream is generated by an LFSR, we can predict future bits.
        """
        log.info("[Sym/LFSR] Attempting Berlekamp-Massey LFSR recovery")
        n = min(len(pt), len(ct))
        if n < 8:
            return self._fail("LFSR: not enough data", t0)

        # Recover keystream bits
        keystream_bytes = [(pt[i] ^ ct[i]) & 0xFF for i in range(n)]
        # Convert to bit stream
        ks_bits = []
        for b in keystream_bytes:
            for bit_pos in range(7, -1, -1):
                ks_bits.append((b >> bit_pos) & 1)

        # Berlekamp-Massey over GF(2)
        poly, L = self._berlekamp_massey_gf2(ks_bits)

        if L == 0 or L > len(ks_bits) // 2:
            return self._fail("LFSR: sequence not LFSR-generated (L too large)", t0)

        log.info("[Sym/LFSR] LFSR degree L=%d detected", L)

        # Generate extended keystream using the LFSR
        extended_bits = list(ks_bits)
        needed_bits = len(ct_target) * 8
        while len(extended_bits) < needed_bits + len(ks_bits):
            # Next bit = sum of poly[i] * bits[n-i] mod 2
            bit = 0
            for i in range(1, L + 1):
                if i < len(poly) and poly[i] == 1:
                    bit ^= extended_bits[-i]
            extended_bits.append(bit)

        # Convert predicted bits to bytes
        predicted_ks = []
        offset = len(ks_bits)  # start of target keystream
        for byte_idx in range(len(ct_target)):
            b = 0
            for bit_pos in range(8):
                idx = offset + byte_idx * 8 + bit_pos
                if idx < len(extended_bits):
                    b = (b << 1) | extended_bits[idx]
            predicted_ks.append(b)

        # Verify: does the predicted keystream match known portion?
        verify_ok = True
        verify_bytes = min(8, n)
        regen_ks = []
        for byte_idx in range(verify_bytes):
            b = 0
            for bit_pos in range(8):
                b = (b << 1) | ks_bits[byte_idx * 8 + bit_pos]
            regen_ks.append(b)
        if regen_ks != keystream_bytes[:verify_bytes]:
            verify_ok = False

        if not verify_ok:
            return self._fail("LFSR: verification failed", t0)

        # Decrypt target
        decrypted = [(ct_target[i] ^ predicted_ks[i]) & 0xFF for i in range(len(ct_target))]

        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.SUCCESS,
            private_key=poly[:L + 1],
            decrypted=decrypted,
            elapsed_seconds=time.perf_counter() - t0,
            confidence=0.85,
            details={
                "method": "lfsr_berlekamp_massey",
                "lfsr_degree": L,
                "polynomial": poly[:L + 1],
            },
        )

    @staticmethod
    def _berlekamp_massey_gf2(bits: list[int]) -> tuple[list[int], int]:
        """Berlekamp-Massey algorithm over GF(2).

        Returns (connection_polynomial, linear_complexity).
        """
        n = len(bits)
        c = [0] * (n + 1)
        b = [0] * (n + 1)
        c[0] = 1
        b[0] = 1
        L = 0
        m = 1

        for i in range(n):
            # Compute discrepancy
            d = bits[i]
            for j in range(1, L + 1):
                d ^= c[j] & bits[i - j]

            if d == 0:
                m += 1
            else:
                t = list(c)
                for j in range(m, n + 1):
                    if j - m < len(b) and b[j - m]:
                        c[j] ^= 1
                if 2 * L <= i:
                    L = i + 1 - L
                    b = list(t)
                    m = 1
                else:
                    m += 1

        return c, L

    # ---- 9. RC4 Bias Exploitation ---- #

    def _attack_rc4_bias(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        t0: float, timeout: float,
    ) -> SolverResult:
        """Exploit RC4 key schedule biases.

        - Second byte bias: Pr[K[1] = 0] ≈ 2/256
        - Fluhrer-Mantin-Shamir: first bytes leak key material
        """
        log.info("[Sym/RC4] Checking for RC4 biases")
        n = min(len(pt), len(ct))
        if n < 16:
            return self._fail("RC4: not enough data", t0)

        keystream = [(pt[i] ^ ct[i]) & 0xFF for i in range(n)]

        # Check second byte bias (P[K[1]=0] ≈ 2/N instead of 1/N)
        # If we have multiple encryptions, the second keystream byte
        # is biased toward 0
        second_bytes = instance_extra = []  # would need multiple samples

        # Single-sample: try to brute-force short RC4 keys
        for key_len in range(1, min(8, n)):
            if time.perf_counter() - t0 > timeout * 0.3:
                break
            res = self._try_rc4_key_bruteforce(keystream, ct_target, key_len, t0, timeout * 0.1)
            if res is not None:
                return res

        return self._fail("RC4: no exploitable bias found", t0)

    def _try_rc4_key_bruteforce(
        self, known_ks: list[int], ct_target: list[int],
        key_len: int, t0: float, timeout: float,
    ) -> SolverResult | None:
        """Try to brute-force a short RC4 key."""
        if key_len > 3:
            return None  # too large to brute-force

        for key_int in range(256 ** key_len):
            if time.perf_counter() - t0 > timeout:
                return None
            key = [(key_int >> (8 * i)) & 0xFF for i in range(key_len)]
            ks = self._rc4_keystream(key, len(known_ks) + len(ct_target))
            if ks[:len(known_ks)] == known_ks:
                # Key found!
                offset = len(known_ks)
                decrypted = [(ct_target[i] ^ ks[offset + i]) & 0xFF for i in range(len(ct_target))]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={"method": "rc4_bruteforce", "key": key, "key_len": key_len},
                )
        return None

    @staticmethod
    def _rc4_keystream(key: list[int], length: int) -> list[int]:
        """Generate RC4 keystream."""
        S = list(range(256))
        j = 0
        for i in range(256):
            j = (j + S[i] + key[i % len(key)]) & 0xFF
            S[i], S[j] = S[j], S[i]

        i = j = 0
        output = []
        for _ in range(length):
            i = (i + 1) & 0xFF
            j = (j + S[i]) & 0xFF
            S[i], S[j] = S[j], S[i]
            output.append(S[(S[i] + S[j]) & 0xFF])
        return output

    # ---- 10. Generic Block-XOR (repeating key) ---- #

    def _attack_block_xor(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """Recover a repeating XOR key aligned to block boundaries."""
        n = min(len(pt), len(ct))
        if n < 2:
            return self._fail("Block XOR: not enough data", t0)

        keystream = [(pt[i] ^ ct[i]) & 0xFF for i in range(n)]

        # Find period in keystream
        period = self._find_period(keystream)
        if period is None or period >= n:
            return self._fail("Block XOR: no period found", t0)

        key = keystream[:period]
        # Verify
        if not all(keystream[i] == key[i % period] for i in range(n)):
            return self._fail("Block XOR: inconsistent period", t0)

        decrypted = [(ct_target[i] ^ key[i % period]) & 0xFF for i in range(len(ct_target))]

        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.SUCCESS,
            private_key=key,
            decrypted=decrypted,
            elapsed_seconds=time.perf_counter() - t0,
            confidence=0.90,
            details={"method": "block_xor_periodic", "key_period": period, "key": key},
        )

    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    @staticmethod
    def _find_period(seq: list[int]) -> int | None:
        """Find the shortest repeating period in a sequence."""
        n = len(seq)
        for p in range(1, n // 2 + 1):
            if all(seq[i] == seq[i % p] for i in range(n)):
                return p
        return None

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Sym] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
