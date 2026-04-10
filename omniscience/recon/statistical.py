"""Statistical Reconnaissance Module.

Performs entropy analysis, bit-level correlation, and dependency heatmap
generation to classify the unknown algorithm's family before dispatching
to specialized solvers.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from omniscience.core.types import AlgoFamily, CryptoInstance, ReconResult

log = logging.getLogger(__name__)


class StatisticalRecon:
    """Black-box statistical analysis of an unknown asymmetric cipher."""

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def analyze(self, instance: CryptoInstance) -> ReconResult:
        """Run full reconnaissance pipeline and return classification."""
        pt = np.array(instance.pt_as_int_list(), dtype=np.uint8)
        ct = np.array(instance.ct_known_as_int_list(), dtype=np.uint8)
        pub = np.array(instance.pub_as_int_list(), dtype=np.int64)

        result = ReconResult()

        # 1. Entropy
        result.entropy_plaintext = self._shannon_entropy(pt)
        result.entropy_ciphertext = self._shannon_entropy(ct)
        log.info("Entropy  PT=%.4f  CT=%.4f", result.entropy_plaintext, result.entropy_ciphertext)

        # 2. Bit-level correlation matrix  (PT bits ↔ CT bits)
        pt_bits = self._to_bit_matrix(pt)
        ct_bits = self._to_bit_matrix(ct)
        min_rows = min(pt_bits.shape[0], ct_bits.shape[0])
        pt_bits = pt_bits[:min_rows]
        ct_bits = ct_bits[:min_rows]
        result.bit_correlation_matrix = self._bit_correlation(pt_bits, ct_bits)
        result.heatmap_data = result.bit_correlation_matrix

        # 3. Linearity score (Walsh-Hadamard approximation)
        result.linearity_score = self._linearity_score(pt_bits, ct_bits)
        log.info("Linearity score: %.4f", result.linearity_score)

        # 4. Polynomial degree estimation via higher-order differentials
        result.polynomial_degree_estimate = self._estimate_poly_degree(pt, ct, instance.modulus)

        # 5. Substitution detection (byte frequency analysis)
        result.substitution_detected = self._detect_substitution(pt, ct)

        # 6. Lattice structure heuristic
        result.lattice_structure_detected = self._detect_lattice_structure(pub, ct, instance.modulus)

        # 7. Modulus estimation
        result.estimated_modulus = self._estimate_modulus(pub, ct, instance.modulus)

        # 8. RSA / factorization indicators
        result.details["rsa_like"] = self._detect_rsa_like(pub, ct, instance.modulus)

        # 9. Discrete log indicators
        result.details["dlog_like"] = self._detect_dlog_like(pub, pt, ct, instance.modulus)

        # 10. Elliptic curve indicators
        result.details["ec_like"] = self._detect_ec_like(pub, ct, instance.modulus)

        # 11. AGCD indicators
        result.details["agcd_like"] = self._detect_agcd_like(pub, ct)

        # 12. NTRU / ring-lattice indicators
        result.details["ntru_like"] = self._detect_ntru_like(pub, instance.modulus)

        # 13. Symmetric cipher indicators (block / stream)
        result.details["symmetric_block"] = self._detect_symmetric_block(pt, ct)
        result.details["symmetric_stream"] = self._detect_symmetric_stream(pt, ct)

        # 14. ECDH indicators
        result.details["ecdh_like"] = self._detect_ecdh_like(pub, ct, instance.modulus)

        # 15. Hybrid scheme indicators
        result.details["hybrid_scheme"] = self._detect_hybrid_scheme(pub, ct, instance.ct_target_as_int_list(), instance.modulus)

        # 16. Classify
        result.algo_family, result.confidence = self._classify(result)
        log.info("Classification: %s (confidence %.2f%%)", result.algo_family.value, result.confidence * 100)

        return result

    # ------------------------------------------------------------------ #
    #  Entropy                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _shannon_entropy(data: NDArray[np.uint8]) -> float:
        """Shannon entropy in bits per byte."""
        if len(data) == 0:
            return 0.0
        _, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        return -float(np.sum(probs * np.log2(probs + 1e-30)))

    # ------------------------------------------------------------------ #
    #  Bit manipulation helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_bit_matrix(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert byte array to (n_bytes, 8) bit matrix."""
        return np.unpackbits(data).reshape(-1, 8)

    @staticmethod
    def _bit_correlation(a: NDArray, b: NDArray) -> NDArray[np.float64]:
        """Pearson correlation between each pair of bit columns in a and b.

        Returns an (a.cols × b.cols) correlation matrix.
        """
        n = a.shape[0]
        if n < 2:
            return np.zeros((a.shape[1], b.shape[1]))
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        a_centered = a_f - a_f.mean(axis=0, keepdims=True)
        b_centered = b_f - b_f.mean(axis=0, keepdims=True)
        a_std = a_centered.std(axis=0, keepdims=True) + 1e-15
        b_std = b_centered.std(axis=0, keepdims=True) + 1e-15
        corr = (a_centered / a_std).T @ (b_centered / b_std) / n
        return np.clip(corr, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Linearity analysis (Walsh-Hadamard inspired)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _linearity_score(pt_bits: NDArray, ct_bits: NDArray) -> float:
        """Measure how close the mapping is to an affine/linear function over GF(2).

        Returns 0.0 (completely non-linear) to 1.0 (perfectly linear).
        We test all single-bit input masks against each output bit via the
        Walsh-Hadamard transform approximation.
        """
        n_samples, in_bits = pt_bits.shape
        _, out_bits = ct_bits.shape
        if n_samples < 8:
            return 0.0

        max_biases: list[float] = []
        for j in range(out_bits):
            y = ct_bits[:, j].astype(np.int8)
            best_bias = 0.0
            for i in range(in_bits):
                x = pt_bits[:, i].astype(np.int8)
                # Walsh coefficient for mask = e_i
                correlation = np.abs(np.mean((-1) ** (x ^ y)))
                if correlation > best_bias:
                    best_bias = correlation
            max_biases.append(best_bias)

        return float(np.mean(max_biases))

    # ------------------------------------------------------------------ #
    #  Polynomial degree estimation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_poly_degree(
        pt: NDArray, ct: NDArray, modulus: int | None
    ) -> int | None:
        """Estimate the algebraic degree via iterated finite differences.

        If E(x) is a polynomial of degree d modulo some modulus, then the
        (d+1)-th order finite difference is zero.
        """
        if modulus is None or modulus < 2:
            return None
        pt_int = pt.astype(np.int64)
        ct_int = ct.astype(np.int64)
        # Sort by plaintext value for finite differences
        idx = np.argsort(pt_int)
        vals = ct_int[idx] % modulus
        max_degree = min(32, len(vals) - 1)
        diff = vals.copy()
        for d in range(1, max_degree + 1):
            diff = np.diff(diff) % modulus
            if np.all(diff == 0):
                return d
            if len(diff) < 2:
                break
        return None

    # ------------------------------------------------------------------ #
    #  Substitution detection                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_substitution(pt: NDArray, ct: NDArray) -> bool:
        """Detect byte-level substitution by checking if the mapping is a permutation."""
        if len(pt) != len(ct):
            return False
        if len(pt) < 16:
            return False
        mapping: dict[int, set[int]] = {}
        for p, c in zip(pt.tolist(), ct.tolist()):
            mapping.setdefault(p, set()).add(c)
        # A perfect substitution maps each input byte to exactly one output byte
        injective = all(len(v) == 1 for v in mapping.values())
        coverage = len(mapping) / 256.0 if len(mapping) <= 256 else 0.0
        return injective and coverage > 0.3

    # ------------------------------------------------------------------ #
    #  Lattice structure detection                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_lattice_structure(
        pub: NDArray, ct: NDArray, modulus: int | None
    ) -> bool:
        """Heuristic: detect if ciphertexts look like lattice inner products mod q."""
        if modulus is None or modulus < 2:
            return False
        if len(pub) < 4:
            return False
        # Check if ciphertext values cluster near multiples of small factors of modulus
        ct_int = ct.astype(np.int64) % modulus
        # LWE-like schemes often have small error terms
        residues = ct_int % max(2, modulus // 256)
        unique_ratio = len(np.unique(residues)) / max(1, len(residues))
        return unique_ratio < 0.3  # low diversity ⇒ structured

    # ------------------------------------------------------------------ #
    #  Modulus estimation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_modulus(
        pub: NDArray, ct: NDArray, provided: int | None
    ) -> int | None:
        """Return the provided modulus or try to infer one from the data."""
        if provided is not None:
            return provided
        # Heuristic: modulus is likely slightly above the max observed value
        all_vals = np.concatenate([pub.flatten(), ct.flatten()])
        if len(all_vals) == 0:
            return None
        max_val = int(np.max(np.abs(all_vals)))
        if max_val < 2:
            return None
        # Check common forms: next prime, next power of 2
        candidate = max_val + 1
        # Simple primality-ish check for small candidates
        if candidate < 2**31:
            from sympy import nextprime
            return int(nextprime(max_val))
        return candidate

    # ------------------------------------------------------------------ #
    #  RSA / Factorization detection                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_rsa_like(pub: NDArray, ct: NDArray, modulus: int | None) -> float:
        """Score [0,1] for RSA-like scheme: c = m^e mod N.

        Indicators:
          - Public key contains a small public exponent (e.g. 3, 17, 65537)
          - Modulus is a product of two large primes (semi-prime)
          - Ciphertext values are spread across [0, N)
        """
        score = 0.0
        pub_list = pub.flatten().tolist()

        # Check for typical RSA public exponents
        rsa_exponents = {3, 5, 7, 17, 257, 65537}
        if any(int(v) in rsa_exponents for v in pub_list):
            score += 0.5

        # If modulus provided and large, likely RSA
        if modulus and modulus.bit_length() >= 256:
            score += 0.3
            # Check if modulus is composite (not prime)
            from sympy import isprime
            if not isprime(modulus):
                score += 0.2

        return min(score, 1.0)

    @staticmethod
    def _detect_dlog_like(
        pub: NDArray, pt: NDArray, ct: NDArray, modulus: int | None
    ) -> float:
        """Score [0,1] for discrete-log based scheme (ElGamal, DSA-like).

        Indicators:
          - Public key has 2 elements (g, h=g^x)
          - Ciphertext has paired structure (c1, c2) for ElGamal
          - Modulus is prime
        """
        score = 0.0
        pub_list = pub.flatten().tolist()

        # pub = [g, h] pattern
        if len(pub_list) == 2 and all(v > 1 for v in pub_list):
            score += 0.4

        # Paired ciphertext (ElGamal: (c1, c2) per message)
        if len(ct) % 2 == 0 and len(ct) >= 2:
            score += 0.2

        # Prime modulus
        if modulus and modulus > 2:
            from sympy import isprime
            if isprime(modulus):
                score += 0.3

        # Check if h = g^x mod p is plausible (h < p)
        if modulus and len(pub_list) >= 2:
            if all(0 < v < modulus for v in pub_list):
                score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _detect_ec_like(pub: NDArray, ct: NDArray, modulus: int | None) -> float:
        """Score [0,1] for elliptic curve scheme.

        Indicators:
          - Public key has 4+ elements (a, b, Px, Py) or 6 (a, b, Px, Py, Qx, Qy)
          - Ciphertext is paired (point coordinates)
          - Values satisfy Weierstrass equation
        """
        score = 0.0
        pub_list = pub.flatten().tolist()

        if len(pub_list) >= 4:
            score += 0.2

        if len(pub_list) >= 6:
            score += 0.2
            # Check if (pub[2], pub[3]) lies on y² = x³ + pub[0]*x + pub[1] mod p
            if modulus and modulus > 2:
                a, b = int(pub_list[0]), int(pub_list[1])
                x, y = int(pub_list[2]) % modulus, int(pub_list[3]) % modulus
                lhs = (y * y) % modulus
                rhs = (x * x * x + a * x + b) % modulus
                if lhs == rhs:
                    score += 0.5  # strong indicator

        # Paired output (points)
        if len(ct) % 2 == 0 and len(ct) >= 2:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _detect_agcd_like(pub: NDArray, ct: NDArray) -> float:
        """Score [0,1] for Approximate GCD scheme.

        Indicators:
          - Multiple large values with a hidden common factor
          - Pairwise GCDs are non-trivial but not exact
          - Values cluster near multiples of an unknown integer
        """
        from math import gcd
        score = 0.0
        vals = [abs(int(v)) for v in pub.flatten().tolist() if v != 0]

        if len(vals) < 3:
            return 0.0

        # Check pairwise GCDs
        gcds = []
        for i in range(min(len(vals), 20)):
            for j in range(i + 1, min(len(vals), 20)):
                g = gcd(vals[i], vals[j])
                if g > 1:
                    gcds.append(g)

        if gcds:
            # If GCDs are large but not equal to the values → approximate structure
            avg_gcd = sum(gcds) / len(gcds)
            avg_val = sum(vals) / len(vals)
            if avg_gcd > 1 and avg_gcd < avg_val * 0.9:
                score += 0.5
            # If GCDs are consistent (low variance) → likely AGCD
            if len(gcds) > 2:
                gcd_set = set(gcds)
                if len(gcd_set) < len(gcds) * 0.5:
                    score += 0.3

        return min(score, 1.0)

    @staticmethod
    def _detect_ntru_like(pub: NDArray, modulus: int | None) -> float:
        """Score [0,1] for NTRU-like ring/module lattice scheme.

        Indicators:
          - Public key is a polynomial vector (length = n, moderate coefficients)
          - Coefficients are bounded by q (modulus)
          - Circulant / Toeplitz structure in the data
        """
        score = 0.0
        pub_list = pub.flatten().tolist()
        n = len(pub_list)

        if n < 4:
            return 0.0

        # Power-of-2 or common NTRU dimensions
        ntru_dims = {107, 127, 128, 167, 197, 211, 233, 239, 251, 256, 263,
                     337, 401, 443, 509, 512, 541, 587, 613, 677, 701, 743, 1024}
        if n in ntru_dims:
            score += 0.3

        if modulus and modulus > 1:
            # Coefficients should be in [0, q)
            if all(0 <= v < modulus for v in pub_list):
                score += 0.2

            # Check for circulant structure: h[i] ≈ rotation-related
            # Autocorrelation test
            arr = np.array(pub_list, dtype=np.float64)
            if np.std(arr) > 0:
                normalized = (arr - np.mean(arr)) / np.std(arr)
                autocorr = np.correlate(normalized, normalized, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                # Strong side peaks indicate circulant structure
                if len(autocorr) > 1:
                    peak_ratio = np.max(np.abs(autocorr[1:])) / (autocorr[0] + 1e-10)
                    if peak_ratio > 0.3:
                        score += 0.3

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    #  Symmetric block cipher detection                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_symmetric_block(pt: NDArray, ct: NDArray) -> float:
        """Score [0,1] for symmetric block cipher.

        Indicators:
          - Ciphertext length is a multiple of common block sizes (8, 16, 32)
          - Repeated plaintext blocks → repeated ciphertext blocks (ECB)
          - High entropy ciphertext with block-aligned structure
          - Avalanche: small PT change → large CT change within blocks
        """
        score = 0.0
        n = len(ct)
        if n < 8:
            return 0.0

        # Block-aligned length
        for bs in (16, 8, 32):
            if n % bs == 0 and n >= bs * 2:
                score += 0.15
                break

        # ECB detection: check for repeated blocks
        for bs in (16, 8):
            if n < bs * 2:
                continue
            blocks = [tuple(ct[i:i + bs]) for i in range(0, n - bs + 1, bs)]
            if len(blocks) != len(set(blocks)):
                score += 0.4  # repeated blocks → ECB mode
                break

        # Avalanche: adjacent plaintexts should give very different ciphertexts
        if len(pt) >= 4 and len(ct) >= 4:
            n_check = min(len(pt), len(ct)) - 1
            diff_bits = 0
            total_bits = 0
            for i in range(n_check):
                xor_val = int(pt[i]) ^ int(pt[i + 1])
                ct_xor = int(ct[i]) ^ int(ct[i + 1])
                if xor_val <= 1:  # adjacent or near plaintexts
                    diff_bits += bin(ct_xor).count('1')
                    total_bits += 8
            if total_bits > 0:
                avalanche = diff_bits / total_bits
                if avalanche > 0.3:
                    score += 0.25

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    #  Symmetric stream cipher detection                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_symmetric_stream(pt: NDArray, ct: NDArray) -> float:
        """Score [0,1] for symmetric stream cipher.

        Indicators:
          - XOR relationship: ct = pt XOR keystream
          - Keystream has high entropy
          - No block alignment required
          - Position-dependent: same pt at different positions → different ct
        """
        score = 0.0
        n = min(len(pt), len(ct))
        if n < 4:
            return 0.0

        # Check XOR keystream
        ks = [(int(pt[i]) ^ int(ct[i])) & 0xFF for i in range(n)]

        # Keystream entropy should be high
        ks_arr = np.array(ks, dtype=np.uint8)
        _, counts = np.unique(ks_arr, return_counts=True)
        if len(counts) > 0:
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log2(probs + 1e-30)))
            if entropy > 5.0:
                score += 0.3

        # Position-dependent: same pt value at different positions → different ct
        pt_positions: dict[int, list[int]] = {}
        for i in range(n):
            pt_positions.setdefault(int(pt[i]), []).append(i)

        pos_dependent = 0
        total_checks = 0
        for val, positions in pt_positions.items():
            if len(positions) >= 2:
                ct_vals = [int(ct[pos]) for pos in positions]
                if len(set(ct_vals)) > 1:
                    pos_dependent += 1
                total_checks += 1

        if total_checks > 0 and pos_dependent / total_checks > 0.5:
            score += 0.35

        # Non-block-aligned: length not divisible by common block sizes
        if len(ct) % 16 != 0 and len(ct) % 8 != 0:
            score += 0.15

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    #  ECDH detection                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_ecdh_like(pub: NDArray, ct: NDArray, modulus: int | None) -> float:
        """Score [0,1] for ECDH key exchange.

        Indicators:
          - Public key has 8+ values (a, b, Gx, Gy, Ax, Ay, Bx, By)
          - Or 6 values + ciphertext has 2 more coordinates
          - Points satisfy curve equation
        """
        score = 0.0
        pub_list = pub.flatten().tolist()

        if len(pub_list) >= 8:
            score += 0.4
            # Check if points are on the curve
            if modulus and modulus > 2:
                a, b = int(pub_list[0]), int(pub_list[1])
                for i in range(2, min(len(pub_list) - 1, 8), 2):
                    x, y = int(pub_list[i]) % modulus, int(pub_list[i + 1]) % modulus
                    lhs = (y * y) % modulus
                    rhs = (x * x * x + a * x + b) % modulus
                    if lhs == rhs:
                        score += 0.15

        elif len(pub_list) >= 6 and len(ct) >= 2:
            score += 0.3

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    #  Hybrid scheme detection                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_hybrid_scheme(
        pub: NDArray, ct: NDArray, ct_target: list[int], modulus: int | None
    ) -> float:
        """Score [0,1] for hybrid (asymmetric + symmetric) scheme.

        Indicators:
          - Ciphertext has two distinct regions (different entropy)
          - First part looks like asymmetric output (large values, mod N)
          - Second part looks like symmetric output (uniform high entropy bytes)
        """
        score = 0.0
        if len(ct_target) < 8:
            return 0.0

        # Check for entropy split
        for split_ratio in (0.1, 0.2, 0.3, 0.5):
            split = max(2, int(len(ct_target) * split_ratio))
            first = ct_target[:split]
            second = ct_target[split:]

            if len(second) < 4:
                continue

            # First part: fewer unique values or structured
            unique_first = len(set(first)) / max(len(first), 1)
            unique_second = len(set(second)) / max(len(second), 1)

            if unique_second > unique_first * 1.5:
                score += 0.2
                break

        # Large modulus + extra payload → RSA-KEM or similar
        if modulus and modulus.bit_length() >= 256:
            mod_bytes = (modulus.bit_length() + 7) // 8
            if len(ct_target) > mod_bytes + 8:
                score += 0.3

        # Public key with RSA-like exponent + long ciphertext
        pub_list = pub.flatten().tolist()
        rsa_exponents = {3, 5, 7, 17, 257, 65537}
        if any(int(v) in rsa_exponents for v in pub_list) and len(ct_target) > 32:
            score += 0.2

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    #  Classification engine                                              #
    # ------------------------------------------------------------------ #

    def _classify(self, r: ReconResult) -> tuple[AlgoFamily, float]:
        """Classify the algorithm family based on all gathered evidence."""
        scores: dict[AlgoFamily, float] = {f: 0.0 for f in AlgoFamily}

        # --- Structural indicators from detailed detectors ---
        rsa_score = r.details.get("rsa_like", 0.0)
        dlog_score = r.details.get("dlog_like", 0.0)
        ec_score = r.details.get("ec_like", 0.0)
        agcd_score = r.details.get("agcd_like", 0.0)
        ntru_score = r.details.get("ntru_like", 0.0)
        sym_block_score = r.details.get("symmetric_block", 0.0)
        sym_stream_score = r.details.get("symmetric_stream", 0.0)
        ecdh_score = r.details.get("ecdh_like", 0.0)
        hybrid_score = r.details.get("hybrid_scheme", 0.0)

        scores[AlgoFamily.RSA_LIKE] += rsa_score
        scores[AlgoFamily.DLOG] += dlog_score
        scores[AlgoFamily.EC_LIKE] += ec_score
        scores[AlgoFamily.AGCD] += agcd_score
        scores[AlgoFamily.NTRU_LIKE] += ntru_score
        scores[AlgoFamily.SYMMETRIC_BLOCK] += sym_block_score
        scores[AlgoFamily.SYMMETRIC_STREAM] += sym_stream_score
        scores[AlgoFamily.ECDH] += ecdh_score
        scores[AlgoFamily.HYBRID] += hybrid_score * 0.8  # boost hybrid detection

        # --- Generic indicators ---

        # High linearity → linear or affine cipher
        if r.linearity_score > 0.85:
            scores[AlgoFamily.LINEAR] += 0.9
        elif r.linearity_score > 0.5:
            scores[AlgoFamily.LINEAR] += 0.4

        # Known polynomial degree
        if r.polynomial_degree_estimate is not None:
            deg = r.polynomial_degree_estimate
            if deg == 1:
                scores[AlgoFamily.LINEAR] += 0.5
            elif deg <= 4:
                scores[AlgoFamily.POLYNOMIAL] += 0.8
            else:
                scores[AlgoFamily.POLYNOMIAL] += 0.4

        # Substitution
        if r.substitution_detected:
            scores[AlgoFamily.SUBSTITUTION] += 0.85

        # Lattice structure
        if r.lattice_structure_detected:
            scores[AlgoFamily.LATTICE_BASED] += 0.7
            scores[AlgoFamily.LWE_BASED] += 0.5
            scores[AlgoFamily.KNAPSACK] += 0.4
            scores[AlgoFamily.NTRU_LIKE] += 0.3

        # High entropy ciphertext with low linearity and no other strong signal
        if r.entropy_ciphertext > 7.5 and r.linearity_score < 0.2:
            scores[AlgoFamily.RSA_LIKE] += 0.2
            scores[AlgoFamily.EC_LIKE] += 0.2
            scores[AlgoFamily.DLOG] += 0.15

        best_family = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_family]

        if best_score < 0.2:
            return AlgoFamily.UNKNOWN, 0.0

        # Normalize confidence to [0, 1]
        confidence = min(best_score, 1.0)
        return best_family, confidence
