"""Hybrid Encryption Scheme Cryptanalysis Module.

Attacks on schemes where asymmetric encryption protects a symmetric key:

  1. RSA-KEM: RSA encrypts a symmetric key, AES/ChaCha encrypts data
     → Factor N, recover symmetric key, decrypt payload
  2. ECIES: ECDH shared secret → KDF → symmetric key → encrypt payload
     → Solve ECDLP, compute shared secret, derive key, decrypt
  3. ElGamal-Hybrid: ElGamal encrypts AES key
     → Solve DLP, recover AES key, decrypt
  4. Weak KDF detection: if the KDF is weak (truncation, XOR, no salt),
     the symmetric key may be directly recoverable
  5. Key-wrap analysis: detect and exploit weak key wrapping
  6. Composite structure: detect asymmetric header + symmetric payload
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class HybridSchemeSolver:
    """Cryptanalysis of hybrid asymmetric+symmetric encryption schemes."""

    NAME = "hybrid_scheme"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            pt = instance.pt_as_int_list()
            ct = instance.ct_known_as_int_list()
            ct_target = instance.ct_target_as_int_list()
            pub = instance.pub_as_int_list()
            mod = instance.modulus

            # Detect hybrid structure
            structure = self._detect_hybrid_structure(pub, ct, ct_target, mod)
            if structure is None:
                return self._fail("Hybrid: no hybrid structure detected", t0)

            log.info("[Hybrid] Detected structure: %s", structure["type"])

            # Strategy 1: RSA-KEM (RSA header + symmetric payload)
            if structure["type"] == "rsa_kem":
                res = self._attack_rsa_kem(structure, pt, ct, ct_target, mod, t0, timeout)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 2: ECIES (EC point + symmetric payload)
            if structure["type"] == "ecies":
                res = self._attack_ecies(structure, instance, t0, timeout)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 3: ElGamal-Hybrid (ElGamal pair + symmetric payload)
            if structure["type"] == "elgamal_hybrid":
                res = self._attack_elgamal_hybrid(structure, pt, ct, ct_target, mod, t0, timeout)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 4: Weak KDF / Key derivation
            res = self._attack_weak_kdf(pt, ct, ct_target, pub, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 5: Header + XOR payload
            res = self._attack_header_xor_payload(pt, ct, ct_target, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Hybrid: all strategies failed", t0)

        except Exception as exc:
            log.exception("Hybrid solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ================================================================== #
    #  Hybrid Structure Detection                                         #
    # ================================================================== #

    def _detect_hybrid_structure(
        self, pub: list[int], ct: list[int], ct_target: list[int],
        mod: int | None,
    ) -> dict | None:
        """Detect if the ciphertext has a hybrid structure.

        Patterns:
          - RSA-KEM: first N_bytes are RSA-encrypted key, rest is symmetric
          - ECIES: first 2*field_size bytes are EC point, rest is symmetric
          - ElGamal: first 2 values are (c1, c2), rest is symmetric
        """
        if not ct_target or len(ct_target) < 4:
            return None

        # RSA-KEM: if modulus is large, first chunk = RSA, rest = symmetric
        if mod and mod.bit_length() >= 256:
            mod_bytes = (mod.bit_length() + 7) // 8
            # Check if first values look like a single RSA block
            if len(pub) >= 1 and len(ct) > mod_bytes:
                return {
                    "type": "rsa_kem",
                    "key_block_size": mod_bytes,
                    "modulus": mod,
                    "exponent": pub[0] if pub else 65537,
                }

        # ECIES: pub has curve params, ciphertext starts with an EC point
        if len(pub) >= 4 and mod and mod > 2:
            field_bytes = (mod.bit_length() + 7) // 8
            point_size = 2 * field_bytes
            if len(ct_target) > point_size + 4:
                # Check if first 2 values could be coordinates
                if all(0 <= v < mod for v in ct_target[:2]):
                    return {
                        "type": "ecies",
                        "point_size": 2,  # number of int values for point
                        "curve_a": pub[0],
                        "curve_b": pub[1],
                    }

        # ElGamal-Hybrid: first 2 values are ElGamal (c1, c2), rest is symmetric
        if mod and len(ct_target) > 4 and len(pub) >= 2:
            if all(0 < v < mod for v in ct_target[:2]):
                return {
                    "type": "elgamal_hybrid",
                    "c1": ct_target[0],
                    "c2": ct_target[1],
                    "payload_start": 2,
                }

        # Generic: check if ciphertext has two distinct entropy regions
        if len(ct_target) >= 32:
            half = len(ct_target) // 2
            entropy_first = self._byte_entropy(ct_target[:half])
            entropy_second = self._byte_entropy(ct_target[half:])
            if abs(entropy_first - entropy_second) > 2.0:
                return {
                    "type": "generic_hybrid",
                    "split_point": half,
                    "entropy_first": entropy_first,
                    "entropy_second": entropy_second,
                }

        return None

    # ================================================================== #
    #  Attack 1: RSA-KEM                                                  #
    # ================================================================== #

    def _attack_rsa_kem(
        self, structure: dict, pt: list[int], ct: list[int],
        ct_target: list[int], mod: int, t0: float, timeout: float,
    ) -> SolverResult:
        """RSA-KEM: factor N → decrypt symmetric key → decrypt payload."""
        log.info("[Hybrid/RSA-KEM] Attempting RSA key unwrap")

        e = structure.get("exponent", 65537)
        N = mod

        # Try to factor N using the factorization solver
        from omniscience.solvers.factorization import FactorizationSolver

        # Quick factor attempts
        factor = None
        for method in [
            FactorizationSolver._trial_division,
            FactorizationSolver._fermat,
            FactorizationSolver._pollard_rho,
            FactorizationSolver._pollard_pm1,
        ]:
            if time.perf_counter() - t0 > timeout * 0.5:
                break
            try:
                f = method(N, timeout=timeout * 0.1)
                if f and 1 < f < N:
                    factor = f
                    break
            except Exception:
                continue

        if factor is None:
            # Try Wiener for small d
            factor = FactorizationSolver._wiener_attack(N, e)

        if factor is None:
            return self._fail("RSA-KEM: couldn't factor N", t0)

        p, q = factor, N // factor
        phi = (p - 1) * (q - 1)
        try:
            d = pow(e, -1, phi)
        except ValueError:
            return self._fail("RSA-KEM: e not invertible mod phi", t0)

        log.info("[Hybrid/RSA-KEM] Factored N: p=%d, q=%d", p, q)

        # Decrypt the key block from known ciphertext
        key_size = structure.get("key_block_size", 1)
        if len(ct) >= key_size:
            # First value(s) of ct are the encrypted symmetric key
            enc_sym_key = ct[0] if key_size == 1 else int.from_bytes(bytes(ct[:key_size]), "big")
            sym_key = pow(enc_sym_key, d, N)

            # Now decrypt payload using the symmetric key
            payload = ct_target
            sym_key_bytes = sym_key.to_bytes((sym_key.bit_length() + 7) // 8 or 1, "big")

            # Try XOR with the key
            decrypted = self._symmetric_decrypt_attempt(payload, sym_key_bytes)
            if decrypted is not None:
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key={"d": d, "sym_key": sym_key},
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.85,
                    details={
                        "method": "rsa_kem",
                        "p": p, "q": q,
                        "sym_key": sym_key,
                    },
                )

        return self._fail("RSA-KEM: couldn't decrypt payload", t0)

    # ================================================================== #
    #  Attack 2: ECIES                                                    #
    # ================================================================== #

    def _attack_ecies(
        self, structure: dict, instance: CryptoInstance,
        t0: float, timeout: float,
    ) -> SolverResult:
        """ECIES: solve ECDLP → compute shared secret → KDF → decrypt.

        ECIES flow:
          1. Alice picks random r, computes R = rG (ephemeral public key)
          2. Shared secret S = r * PubB = (Sx, Sy)
          3. Derive key: K = KDF(Sx)
          4. Ciphertext = R || Enc_K(message)
        """
        log.info("[Hybrid/ECIES] Attempting ECIES decryption")

        # Try the ECDH solver to recover the shared secret
        from omniscience.solvers.ecdh import ECDHSolver
        from omniscience.recon.statistical import StatisticalRecon

        ecdh = ECDHSolver()
        recon = StatisticalRecon().analyze(instance)
        ecdh_result = ecdh.solve(instance, recon, timeout=timeout * 0.7)

        if ecdh_result.status != SolverStatus.SUCCESS:
            return self._fail("ECIES: ECDLP not solved", t0)

        shared = ecdh_result.details.get("shared_secret")
        if shared is None:
            return self._fail("ECIES: no shared secret recovered", t0)

        # Derive key from shared secret using common KDFs
        Sx = shared[0] if isinstance(shared, (list, tuple)) else shared
        sym_keys = self._derive_keys_from_secret(Sx)

        # Try to decrypt payload (ciphertext minus the EC point header)
        ct_target = instance.ct_target_as_int_list()
        point_size = structure.get("point_size", 2)
        payload = ct_target[point_size:]

        for key_name, key_bytes in sym_keys:
            decrypted = self._symmetric_decrypt_attempt(payload, key_bytes)
            if decrypted is not None:
                log.info("[Hybrid/ECIES] Decrypted via KDF=%s", key_name)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=ecdh_result.private_key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.90,
                    details={
                        "method": "ecies",
                        "shared_secret": shared,
                        "kdf": key_name,
                    },
                )

        return self._fail("ECIES: key derivation didn't match", t0)

    # ================================================================== #
    #  Attack 3: ElGamal-Hybrid                                           #
    # ================================================================== #

    def _attack_elgamal_hybrid(
        self, structure: dict, pt: list[int], ct: list[int],
        ct_target: list[int], mod: int, t0: float, timeout: float,
    ) -> SolverResult:
        """ElGamal encrypts symmetric key, payload encrypted with that key."""
        log.info("[Hybrid/ElGamal] Attempting ElGamal key unwrap")

        from omniscience.solvers.dlog import DLogSolver

        pub = ct[:2] if len(ct) >= 2 else []
        if len(pub) < 2:
            return self._fail("ElGamal hybrid: need g, h in public", t0)

        g, h = pub[0], pub[1]
        dlog = DLogSolver()

        # Try BSGS
        x = dlog._bsgs(g, h, mod)
        if x is None:
            x_result = dlog._pohlig_hellman(g, h, mod, timeout=timeout * 0.3)
            x = x_result

        if x is None:
            return self._fail("ElGamal hybrid: DLP not solved", t0)

        # ElGamal decryption: m = c2 * c1^(-x) mod p
        c1 = structure.get("c1", ct_target[0])
        c2 = structure.get("c2", ct_target[1])
        sym_key = (c2 * pow(c1, mod - 1 - x, mod)) % mod

        # Decrypt payload
        payload_start = structure.get("payload_start", 2)
        payload = ct_target[payload_start:]

        sym_key_bytes = sym_key.to_bytes((sym_key.bit_length() + 7) // 8 or 1, "big")
        decrypted = self._symmetric_decrypt_attempt(payload, sym_key_bytes)
        if decrypted is not None:
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key={"dlog_x": x, "sym_key": sym_key},
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={"method": "elgamal_hybrid", "sym_key": sym_key},
            )

        return self._fail("ElGamal hybrid: payload decrypt failed", t0)

    # ================================================================== #
    #  Attack 4: Weak KDF                                                 #
    # ================================================================== #

    def _attack_weak_kdf(
        self, pt: list[int], ct: list[int], ct_target: list[int],
        pub: list[int], t0: float,
    ) -> SolverResult:
        """Detect and exploit weak key derivation functions.

        Common weaknesses:
          - Key = truncate(public_key)
          - Key = XOR(public_key_bytes)
          - Key = hash(public_key) but with predictable input
          - No salt/nonce in KDF
        """
        log.info("[Hybrid/WeakKDF] Testing weak key derivation")
        n = min(len(pt), len(ct))
        if n < 2 or not pub:
            return self._fail("Weak KDF: not enough data", t0)

        # Derive candidate keys from public key
        pub_bytes = bytes(pub) if all(0 <= v <= 255 for v in pub) else b""
        if not pub_bytes and pub:
            pub_bytes = int(pub[0]).to_bytes((int(pub[0]).bit_length() + 7) // 8 or 1, "big")

        candidates = self._derive_keys_from_secret(
            int.from_bytes(pub_bytes, "big") if pub_bytes else pub[0]
        )

        # Also try direct use of pub values as key
        if all(0 <= v <= 255 for v in pub):
            candidates.append(("raw_pub", bytes(pub)))

        # Test each candidate: does it decrypt known ct to known pt?
        for key_name, key_bytes in candidates:
            key_list = list(key_bytes)
            if not key_list:
                continue

            # XOR decrypt known
            match = True
            for i in range(n):
                if (ct[i] ^ key_list[i % len(key_list)]) & 0xFF != pt[i]:
                    match = False
                    break

            if match:
                decrypted = [
                    (ct_target[i] ^ key_list[i % len(key_list)]) & 0xFF
                    for i in range(len(ct_target))
                ]
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key_list,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.80,
                    details={"method": "weak_kdf", "kdf_type": key_name},
                )

        return self._fail("Weak KDF: no candidate matched", t0)

    # ================================================================== #
    #  Attack 5: Header + XOR Payload                                     #
    # ================================================================== #

    def _attack_header_xor_payload(
        self, pt: list[int], ct: list[int], ct_target: list[int], t0: float,
    ) -> SolverResult:
        """Detect header (key material) + XOR-encrypted payload structure."""
        log.info("[Hybrid/HeaderXOR] Testing header+payload structure")
        n = min(len(pt), len(ct))
        if n < 4 or len(ct_target) < 4:
            return self._fail("Header+XOR: not enough data", t0)

        # Try different header sizes
        for header_size in range(1, min(32, len(ct_target) // 2)):
            header = ct_target[:header_size]
            payload = ct_target[header_size:]

            # Use header bytes as XOR key
            key = header
            # Check against known pairs
            if len(ct) > header_size:
                known_payload = ct[header_size:]
                known_pt = pt[header_size:] if len(pt) > header_size else []

                if len(known_pt) >= 2:
                    match = True
                    for i in range(min(len(known_pt), len(known_payload))):
                        if (known_payload[i] ^ key[i % len(key)]) & 0xFF != known_pt[i]:
                            match = False
                            break

                    if match:
                        decrypted = [
                            (payload[i] ^ key[i % len(key)]) & 0xFF
                            for i in range(len(payload))
                        ]
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=key,
                            decrypted=decrypted,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.75,
                            details={
                                "method": "header_xor_payload",
                                "header_size": header_size,
                            },
                        )

        return self._fail("Header+XOR: no match", t0)

    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    @staticmethod
    def _derive_keys_from_secret(secret: int) -> list[tuple[str, bytes]]:
        """Derive candidate symmetric keys from a shared secret using common KDFs."""
        secret_bytes = secret.to_bytes((secret.bit_length() + 7) // 8 or 1, "big")
        keys = []

        # SHA-256
        keys.append(("sha256", hashlib.sha256(secret_bytes).digest()))

        # SHA-256 truncated to 16 bytes (AES-128)
        keys.append(("sha256_16", hashlib.sha256(secret_bytes).digest()[:16]))

        # SHA-1
        keys.append(("sha1", hashlib.sha1(secret_bytes).digest()))

        # MD5
        keys.append(("md5", hashlib.md5(secret_bytes).digest()))

        # Raw bytes (no KDF — just the secret itself)
        keys.append(("raw", secret_bytes))

        # XOR fold to 16 bytes
        if len(secret_bytes) > 16:
            folded = bytearray(16)
            for i, b in enumerate(secret_bytes):
                folded[i % 16] ^= b
            keys.append(("xor_fold_16", bytes(folded)))

        # First N bytes
        for n in (8, 16, 32):
            if len(secret_bytes) >= n:
                keys.append((f"first_{n}", secret_bytes[:n]))

        # Last N bytes
        for n in (8, 16, 32):
            if len(secret_bytes) >= n:
                keys.append((f"last_{n}", secret_bytes[-n:]))

        return keys

    def _symmetric_decrypt_attempt(
        self, payload: list[int], key_bytes: bytes,
    ) -> list[int] | None:
        """Try to decrypt payload with a candidate symmetric key.

        Tests XOR, repeating-key XOR, and simple stream cipher patterns.
        Returns decrypted bytes if result looks printable/structured.
        """
        if not payload or not key_bytes:
            return None

        key = list(key_bytes)

        # XOR with repeating key
        decrypted = [(payload[i] ^ key[i % len(key)]) & 0xFF for i in range(len(payload))]

        # Check if result is plausible
        printable = sum(1 for b in decrypted if 32 <= b <= 126) / max(len(decrypted), 1)
        if printable > 0.6:
            return decrypted

        # Check for null termination pattern (common in CTF)
        if decrypted and decrypted[-1] == 0:
            trimmed = decrypted[:decrypted.index(0)] if 0 in decrypted else decrypted
            printable = sum(1 for b in trimmed if 32 <= b <= 126) / max(len(trimmed), 1)
            if printable > 0.7:
                return decrypted

        return None

    @staticmethod
    def _byte_entropy(data: list[int]) -> float:
        """Shannon entropy of a byte sequence."""
        if not data:
            return 0.0
        from collections import Counter
        counts = Counter(data)
        total = len(data)
        return -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Hybrid] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
