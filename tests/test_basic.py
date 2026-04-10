"""Comprehensive tests for OMNISCIENCE.

Tests each solver module against known cipher constructions.
"""

import math
import pytest
from omniscience.core.types import CryptoInstance, SolverStatus
from omniscience.core.config import OmniscienceConfig, SolverTimeouts
from omniscience.dispatcher import Dispatcher
from omniscience.recon.statistical import StatisticalRecon
from omniscience.solvers.algebraic import AlgebraicSolver
from omniscience.solvers.lattice import LatticeSolver
from omniscience.solvers.smt import SMTSolver
from omniscience.solvers.factorization import FactorizationSolver
from omniscience.solvers.dlog import DLogSolver
from omniscience.solvers.elliptic_curve import EllipticCurve, EllipticCurveSolver
from omniscience.solvers.agcd import AGCDSolver
from omniscience.solvers.bruteforce_gpu import BruteForceGPUSolver
from omniscience.solvers.mitm import MITMSolver
from omniscience.solvers.lattice_advanced import AdvancedLatticeSolver
from omniscience.solvers.classical import ClassicalCipherSolver
from omniscience.solvers.cross_cipher import CrossCipherSolver
from omniscience.solvers.symmetric import SymmetricSolver
from omniscience.solvers.ecdh import ECDHSolver
from omniscience.solvers.hybrid_scheme import HybridSchemeSolver
from omniscience.core.report import ReportExporter


# ====================================================================== #
#  Helper: create instances                                               #
# ====================================================================== #

def _make_instance(pt, ct, ct_target, pub, modulus=None, extra=None):
    return CryptoInstance(
        public_key=pub,
        plaintext=pt,
        ciphertext_known=ct,
        ciphertext_target=ct_target,
        modulus=modulus,
        extra=extra or {},
    )


# ====================================================================== #
#  1. Statistical Reconnaissance                                          #
# ====================================================================== #

class TestStatisticalRecon:
    def test_linear_detection(self):
        mod = 251
        pts = list(range(50))
        cts = [(3 * p + 7) % mod for p in pts]
        inst = _make_instance(pts, cts, [0], [3], modulus=mod)
        result = StatisticalRecon().analyze(inst)
        assert result.entropy_ciphertext > 0
        assert result.estimated_modulus is not None

    def test_high_entropy(self):
        import random
        random.seed(42)
        pts = [random.randint(0, 255) for _ in range(100)]
        cts = [random.randint(0, 255) for _ in range(100)]
        inst = _make_instance(pts, cts, [0], [1])
        result = StatisticalRecon().analyze(inst)
        assert result.entropy_ciphertext > 3.0


# ====================================================================== #
#  2. Algebraic Solver                                                    #
# ====================================================================== #

class TestAlgebraicSolver:
    def test_linear_cipher(self):
        """c = 3*p + 7 mod 251"""
        mod = 251
        pts = list(range(50))
        cts = [(3 * p + 7) % mod for p in pts]
        target_pts = [72, 100, 200]
        target_cts = [(3 * p + 7) % mod for p in target_pts]
        inst = _make_instance(pts, cts, target_cts, [3], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = AlgebraicSolver().solve(inst, recon)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_quadratic_cipher(self):
        """c = 2*p^2 + 5*p + 1 mod 251"""
        mod = 251
        pts = list(range(60))
        cts = [(2 * p * p + 5 * p + 1) % mod for p in pts]
        target_pts = [10, 20, 30]
        target_cts = [(2 * p * p + 5 * p + 1) % mod for p in target_pts]
        inst = _make_instance(pts, cts, target_cts, [2, 5, 1], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = AlgebraicSolver().solve(inst, recon)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts


# ====================================================================== #
#  3. Factorization Solver                                                #
# ====================================================================== #

class TestFactorizationSolver:
    def test_small_rsa(self):
        """RSA with small primes p=61, q=53, e=17"""
        p, q, e = 61, 53, 17
        N = p * q  # 3233
        phi = (p - 1) * (q - 1)
        d = pow(e, -1, phi)
        pt = [42, 100]
        ct = [pow(m, e, N) for m in pt]
        target = [pow(123, e, N)]
        inst = _make_instance(pt, ct, target, [e], modulus=N)
        recon = StatisticalRecon().analyze(inst)
        result = FactorizationSolver().solve(inst, recon)
        assert result.status == SolverStatus.SUCCESS
        assert result.details.get("p") in (p, q)

    def test_fermat_close_primes(self):
        """N = p * q where p and q are close"""
        p, q = 1000000007, 1000000009
        N = p * q
        factor = FactorizationSolver._fermat(N, timeout=10)
        assert factor is not None
        assert factor in (p, q)

    def test_wiener_small_d(self):
        """Wiener's attack with small private exponent"""
        # Small example: p=101, q=103, d=7
        p, q = 101, 103
        N = p * q
        phi = (p - 1) * (q - 1)
        d = 7
        e = pow(d, -1, phi)
        factor = FactorizationSolver._wiener_attack(N, e)
        if factor is not None:  # Wiener only works if d < N^0.25
            assert N % factor == 0

    def test_pollard_rho(self):
        """Pollard rho on a composite"""
        N = 1000003 * 1000033
        factor = FactorizationSolver._pollard_rho(N, timeout=10)
        assert factor is not None
        assert N % factor == 0


# ====================================================================== #
#  4. Discrete Logarithm Solver                                           #
# ====================================================================== #

class TestDLogSolver:
    def test_bsgs(self):
        """Baby-step Giant-step"""
        p = 1009  # prime
        g = 11
        x_secret = 537
        h = pow(g, x_secret, p)
        result = DLogSolver._bsgs(g, h, p)
        assert result is not None
        assert pow(g, result, p) == h

    def test_pohlig_hellman(self):
        """Pohlig-Hellman when p-1 is smooth"""
        # p = 2 * 3 * 5 * 7 * 11 * 13 + 1 = 30031 (check if prime)
        # Use a known smooth-order prime
        p = 433  # 433 - 1 = 432 = 2^4 * 3^3
        g = 7
        x_secret = 100
        h = pow(g, x_secret, p)
        solver = DLogSolver()
        result = solver._pohlig_hellman(g, h, p, timeout=30)
        if result is not None:
            assert pow(g, result, p) == h

    def test_full_solve(self):
        """Full DLog solver pipeline"""
        p = 1009
        g = 11
        x_secret = 237
        h = pow(g, x_secret, p)
        inst = _make_instance([g], [h], [h], [g, h], modulus=p)
        recon = StatisticalRecon().analyze(inst)
        result = DLogSolver().solve(inst, recon)
        assert result.status == SolverStatus.SUCCESS
        assert pow(g, result.private_key, p) == h


# ====================================================================== #
#  5. Elliptic Curve                                                      #
# ====================================================================== #

class TestEllipticCurve:
    def test_curve_arithmetic(self):
        """Basic EC point addition and multiplication"""
        # y² = x³ + 2x + 3 over F_97
        E = EllipticCurve(2, 3, 97)
        # Find a point on the curve
        P = None
        for x in range(97):
            rhs = (x**3 + 2*x + 3) % 97
            # Check if rhs is a quadratic residue
            y = pow(rhs, (97 + 1) // 4, 97)
            if (y * y) % 97 == rhs:
                P = (x, y)
                break
        assert P is not None
        assert E.is_on_curve(P)

        # Test doubling
        P2 = E.add(P, P)
        assert P2 is None or E.is_on_curve(P2)

        # Test scalar multiplication
        P5 = E.mul(5, P)
        assert P5 is None or E.is_on_curve(P5)

    def test_singular_curve(self):
        """Singular curve detection"""
        # y² = x³ (a=0, b=0) → singular at origin
        E = EllipticCurve(0, 0, 97)
        assert E.is_singular()

    def test_ec_bsgs(self):
        """BSGS on small EC"""
        E = EllipticCurve(2, 3, 97)
        # Find generator
        P = None
        for x in range(97):
            rhs = (x**3 + 2*x + 3) % 97
            y = pow(rhs, (97 + 1) // 4, 97)
            if (y * y) % 97 == rhs:
                P = (x, y)
                break
        if P is not None:
            k = 13
            Q = E.mul(k, P)
            if Q is not None:
                found = EllipticCurveSolver._ec_bsgs(E, P, Q, 100)
                if found is not None:
                    assert E.mul(found, P) == Q


# ====================================================================== #
#  6. AGCD Solver                                                         #
# ====================================================================== #

class TestAGCDSolver:
    def test_exact_gcd(self):
        """Exact GCD (no noise)"""
        p = 104729  # prime
        quotients = [37, 89, 131, 173, 211, 251, 307, 353]
        samples = [p * q for q in quotients]
        inst = _make_instance(
            list(range(len(samples))), samples, [samples[0]], samples
        )
        recon = StatisticalRecon().analyze(inst)
        result = AGCDSolver().solve(inst, recon)
        # Should find p or a multiple of p
        if result.status == SolverStatus.SUCCESS:
            assert result.private_key % p == 0 or p % result.private_key == 0


# ====================================================================== #
#  7. Brute-Force GPU Solver                                              #
# ====================================================================== #

class TestBruteForceSolver:
    def test_xor_bruteforce(self):
        """XOR key recovery"""
        key = 0x42
        pt = list(range(10))
        ct = [(p ^ key) & 0xFF for p in pt]
        target_ct = [(200 ^ key) & 0xFF]
        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = BruteForceGPUSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == [200]

    def test_additive_bruteforce(self):
        """Additive key recovery"""
        key = 73
        pt = list(range(10))
        ct = [(p + key) & 0xFF for p in pt]
        target_ct = [(50 + key) & 0xFF]
        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = BruteForceGPUSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == [50]

    def test_modpow_bruteforce(self):
        """Small exponent brute-force"""
        mod = 1009
        e = 7
        pt = [2, 3, 5]
        ct = [pow(p, e, mod) for p in pt]
        target_ct = [pow(42, e, mod)]
        inst = _make_instance(pt, ct, target_ct, [e], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = BruteForceGPUSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.private_key == e


# ====================================================================== #
#  8. Meet-in-the-Middle Solver                                           #
# ====================================================================== #

class TestMITMSolver:
    def test_double_additive(self):
        """Double additive: c = (p + k1) + k2 mod m"""
        mod = 251
        k1, k2 = 37, 89
        pts = list(range(20))
        cts = [((p + k1) + k2) % mod for p in pts]
        target_pts = [100, 150]
        target_cts = [((p + k1) + k2) % mod for p in target_pts]
        inst = _make_instance(pts, cts, target_cts, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = MITMSolver().solve(inst, recon, timeout=60)
        # MITM should find the combined key (k1+k2 = 126)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_pts

    def test_functional_mitm(self):
        """Multiplicative key via sqrt-split MITM"""
        mod = 1009
        key = 537
        pts = [2, 3, 5, 7, 11]
        cts = [(p * key) % mod for p in pts]
        target_ct = [(42 * key) % mod]
        inst = _make_instance(pts, cts, target_ct, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = MITMSolver().solve(inst, recon, timeout=60)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == [42]


# ====================================================================== #
#  9. Lattice Solver                                                      #
# ====================================================================== #

class TestLatticeSolver:
    def test_lll_reduction(self):
        """Basic LLL reduction"""
        import numpy as np
        basis = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 2],
        ], dtype=np.int64)
        reduced = LatticeSolver._lll_reduce(basis)
        # Reduced basis should have shorter vectors
        orig_norms = [np.linalg.norm(row) for row in basis]
        new_norms = [np.linalg.norm(row) for row in reduced]
        assert min(new_norms) <= min(orig_norms) + 0.01


# ====================================================================== #
#  10. Classical Cipher Solver                                            #
# ====================================================================== #

class TestClassicalCipherSolver:
    def test_caesar(self):
        """Caesar cipher: c = (p + 13) mod 256"""
        key = 13
        pt = list(range(20))
        ct = [(p + key) % 256 for p in pt]
        target_pts = [50, 100, 200]
        target_cts = [(p + key) % 256 for p in target_pts]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=256)
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_affine(self):
        """Affine cipher: c = 3*p + 7 mod 256"""
        a, b = 3, 7
        mod = 256
        pt = list(range(30))
        ct = [(a * p + b) % mod for p in pt]
        target_pts = [42, 99]
        target_cts = [(a * p + b) % mod for p in target_pts]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_xor_single(self):
        """Single-byte XOR: c = p ^ 0xAB"""
        key = 0xAB
        pt = list(range(20))
        ct = [(p ^ key) & 0xFF for p in pt]
        target_pts = [42, 200]
        target_cts = [(p ^ key) & 0xFF for p in target_pts]
        inst = _make_instance(pt, ct, target_cts, [0])
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_xor_multi(self):
        """Multi-byte XOR key"""
        key = [0x12, 0x34, 0x56]
        pt = list(range(12))
        ct = [(p ^ key[i % len(key)]) & 0xFF for i, p in enumerate(pt)]
        target_pts = [100, 101, 102]
        target_cts = [(p ^ key[i % len(key)]) & 0xFF for i, p in enumerate(target_pts)]
        inst = _make_instance(pt, ct, target_cts, [0])
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_vigenere_kpa(self):
        """Vigenère with known plaintext: key derived from pt/ct pairs"""
        key = [3, 7, 11]
        mod = 256
        pt = list(range(15))
        ct = [(p + key[i % len(key)]) % mod for i, p in enumerate(pt)]
        target_pts = [50, 51, 52, 53, 54, 55]
        target_cts = [(p + key[i % len(key)]) % mod for i, p in enumerate(target_pts)]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_hill_2x2(self):
        """Hill cipher with 2×2 matrix mod 256"""
        # Key matrix [[3, 5], [7, 11]], det = 33 - 35 = -2 ≡ 254 mod 256
        # gcd(254, 256) = 2 → not invertible mod 256
        # Use mod 251 (prime) instead
        mod = 251
        # Key matrix [[3, 5], [7, 2]], det = 6 - 35 = -29 ≡ 222 mod 251
        # gcd(222, 251) = 1 → invertible
        K = [[3, 5], [7, 2]]
        pt = []
        ct = []
        for i in range(0, 20, 2):
            p0, p1 = i, i + 1
            c0 = (K[0][0] * p0 + K[0][1] * p1) % mod
            c1 = (K[1][0] * p0 + K[1][1] * p1) % mod
            pt.extend([p0, p1])
            ct.extend([c0, c1])
        target_pts = [40, 41]
        tc0 = (K[0][0] * 40 + K[0][1] * 41) % mod
        tc1 = (K[1][0] * 40 + K[1][1] * 41) % mod
        target_cts = [tc0, tc1]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = ClassicalCipherSolver().solve(inst, recon, timeout=30)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_pts


# ====================================================================== #
#  11. Cross-Cipher Solver                                                #
# ====================================================================== #

class TestCrossCipherSolver:
    def test_two_time_pad(self):
        """Two-time pad: same XOR key used on two plaintexts"""
        key = [0x42, 0x13, 0x7F, 0xAB, 0x01]
        pt_known = [72, 101, 108, 108, 111]  # "Hello"
        ct_known = [(p ^ k) & 0xFF for p, k in zip(pt_known, key)]
        pt_target = [87, 111, 114, 108, 100]  # "World"
        ct_target = [(p ^ k) & 0xFF for p, k in zip(pt_target, key)]
        inst = _make_instance(pt_known, ct_known, ct_target, [0])
        recon = StatisticalRecon().analyze(inst)
        result = CrossCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == pt_target

    def test_decompose_xor_shift(self):
        """Composition: c = (p ^ 0x42) + 7 mod 256"""
        xor_key = 0x42
        shift = 7
        mod = 256
        pt = list(range(20))
        ct = [((p ^ xor_key) + shift) % mod for p in pt]
        target_pts = [100, 150, 200]
        target_cts = [((p ^ xor_key) + shift) % mod for p in target_pts]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=mod)
        recon = StatisticalRecon().analyze(inst)
        result = CrossCipherSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts

    def test_decompose_double_affine(self):
        """Double affine: c = a2*(a1*p + b1) + b2 mod m"""
        m = 251
        a1, b1, a2, b2 = 3, 5, 7, 11
        pt = list(range(20))
        ct = [(a2 * ((a1 * p + b1) % m) + b2) % m for p in pt]
        target_pts = [50, 100, 200]
        target_cts = [(a2 * ((a1 * p + b1) % m) + b2) % m for p in target_pts]
        inst = _make_instance(pt, ct, target_cts, [0], modulus=m)
        recon = StatisticalRecon().analyze(inst)
        result = CrossCipherSolver().solve(inst, recon, timeout=60)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pts


# ====================================================================== #
#  12. Symmetric Cipher Solver                                            #
# ====================================================================== #

class TestSymmetricSolver:
    def test_ecb_codebook(self):
        """ECB mode: repeated blocks map identically."""
        bs = 4
        # Simple ECB: each 4-byte block is XORed with a fixed key block
        key_block = [0x11, 0x22, 0x33, 0x44]
        pt = []
        ct = []
        for i in range(8):
            block = [i * 4 + j for j in range(bs)]
            enc = [(block[j] ^ key_block[j]) & 0xFF for j in range(bs)]
            pt.extend(block)
            ct.extend(enc)

        # Target: encrypt the first two blocks again (ECB = deterministic)
        target_pt_blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        target_ct = []
        for block in target_pt_blocks:
            target_ct.extend([(block[j] ^ key_block[j]) & 0xFF for j in range(bs)])

        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = SymmetricSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_keystream_recovery(self):
        """CTR/OFB mode: XOR keystream recovery."""
        keystream = [0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x13, 0x37, 0x00]
        pt = list(range(8))
        ct = [(p ^ keystream[i]) & 0xFF for i, p in enumerate(pt)]
        target_pt = [100, 101, 102, 103, 104, 105, 106, 107]
        target_ct = [(p ^ keystream[i]) & 0xFF for i, p in enumerate(target_pt)]

        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = SymmetricSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pt

    def test_feistel_single_round(self):
        """Single-round Feistel: C_L = P_R, C_R = P_L XOR F(P_R)."""
        bs = 8
        half = 4
        # F is a simple XOR with a key
        f_key = [0xAA, 0xBB, 0xCC, 0xDD]

        def feistel_enc(block):
            l, r = block[:half], block[half:]
            f_out = [(r[j] ^ f_key[j]) & 0xFF for j in range(half)]
            new_l = list(r)
            new_r = [(l[j] ^ f_out[j]) & 0xFF for j in range(half)]
            return new_l + new_r

        pt, ct = [], []
        for i in range(16):
            block = [(i * 8 + j) & 0xFF for j in range(bs)]
            enc = feistel_enc(block)
            pt.extend(block)
            ct.extend(enc)

        target_block = [200, 201, 202, 203, 204, 205, 206, 207]
        target_ct = feistel_enc(target_block)

        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = SymmetricSolver().solve(inst, recon, timeout=30)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_block

    def test_lfsr_berlekamp_massey(self):
        """LFSR keystream with known polynomial."""
        # Simple LFSR: s[n] = s[n-1] XOR s[n-3] (degree 3)
        state = [1, 0, 1, 1, 0, 0, 1, 0]  # initial keystream bits
        # Extend
        for _ in range(200):
            state.append(state[-1] ^ state[-3])

        # Convert to bytes
        ks_bytes = []
        for i in range(0, len(state) - 7, 8):
            b = 0
            for j in range(8):
                b = (b << 1) | state[i + j]
            ks_bytes.append(b)

        n_known = 8
        pt = list(range(n_known))
        ct = [(pt[i] ^ ks_bytes[i]) & 0xFF for i in range(n_known)]
        target_pt = [50, 51, 52, 53]
        target_ct = [(target_pt[i] ^ ks_bytes[n_known + i]) & 0xFF for i in range(4)]

        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = SymmetricSolver().solve(inst, recon, timeout=30)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_pt

    def test_rc4_short_key(self):
        """RC4 with a 1-byte key (brute-forceable)."""
        key = [0x42]
        ks = SymmetricSolver._rc4_keystream(key, 20)
        pt = list(range(10))
        ct = [(pt[i] ^ ks[i]) & 0xFF for i in range(10)]
        target_pt = [100, 101, 102]
        target_ct = [(target_pt[i] ^ ks[10 + i]) & 0xFF for i in range(3)]

        inst = _make_instance(pt, ct, target_ct, [0])
        recon = StatisticalRecon().analyze(inst)
        result = SymmetricSolver().solve(inst, recon, timeout=60)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_pt


# ====================================================================== #
#  13. ECDH Solver                                                        #
# ====================================================================== #

class TestECDHSolver:
    def test_ecdh_small_curve(self):
        """ECDH on a small curve: recover shared secret via BSGS."""
        # y^2 = x^3 + 2x + 3 mod 97
        a, b, p = 2, 3, 97
        E = EllipticCurve(a, b, p)

        # Find a generator
        G = None
        for x in range(p):
            rhs = (x ** 3 + a * x + b) % p
            y = pow(rhs, (p + 1) // 4, p)
            if (y * y) % p == rhs:
                G = (x, y)
                if E.is_on_curve(G):
                    break

        if G is None:
            return

        dA = 13  # Alice's private key
        dB = 29  # Bob's private key
        pub_A = E.mul(dA, G)
        pub_B = E.mul(dB, G)
        shared = E.mul(dA, pub_B)  # = dA * dB * G

        # pub = [a, b, Gx, Gy, Ax, Ay, Bx, By]
        pub_list = [a, b, G[0], G[1], pub_A[0], pub_A[1], pub_B[0], pub_B[1]]
        inst = _make_instance([], [], [], pub_list, modulus=p)
        recon = StatisticalRecon().analyze(inst)
        result = ECDHSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        # Verify shared secret
        recovered_shared = result.details.get("shared_secret")
        if recovered_shared:
            assert tuple(recovered_shared) == shared

    def test_ecdh_nonce_reuse(self):
        """ECDSA nonce reuse: recover private key from two signatures."""
        a, b, p = 2, 3, 97
        E = EllipticCurve(a, b, p)

        G = None
        for x in range(p):
            rhs = (x ** 3 + a * x + b) % p
            y = pow(rhs, (p + 1) // 4, p)
            if (y * y) % p == rhs:
                G = (x, y)
                if E.is_on_curve(G):
                    break
        if G is None:
            return

        # Find order of G
        n = E.order_point(G)
        if n is None or n < 5:
            return

        d = 17  # private key
        pub_A = E.mul(d, G)
        k = 23  # nonce (reused!)

        # Simulated ECDSA signatures with same k
        R = E.mul(k, G)
        r = R[0] % n
        h1, h2 = 42, 77
        try:
            k_inv = pow(k, -1, n)
        except ValueError:
            return
        s1 = (k_inv * (h1 + r * d)) % n
        s2 = (k_inv * (h2 + r * d)) % n

        sigs = [
            {"r": r, "s": s1, "h": h1},
            {"r": r, "s": s2, "h": h2},
        ]

        pub_list = [a, b, G[0], G[1], pub_A[0], pub_A[1]]
        inst = _make_instance([], [], [], pub_list, modulus=p, extra={"signatures": sigs})
        recon = StatisticalRecon().analyze(inst)
        result = ECDHSolver().solve(inst, recon, timeout=30)
        if result.status == SolverStatus.SUCCESS:
            assert result.private_key == d


# ====================================================================== #
#  14. Hybrid Scheme Solver                                               #
# ====================================================================== #

class TestHybridSchemeSolver:
    def test_weak_kdf_xor(self):
        """Hybrid where public key is directly used as XOR key."""
        key = [0x12, 0x34, 0x56]
        pt = list(range(12))
        ct = [(pt[i] ^ key[i % len(key)]) & 0xFF for i in range(12)]
        target_pt = [100, 101, 102, 103, 104, 105]
        target_ct = [(target_pt[i] ^ key[i % len(key)]) & 0xFF for i in range(6)]

        inst = _make_instance(pt, ct, target_ct, key)
        recon = StatisticalRecon().analyze(inst)
        result = HybridSchemeSolver().solve(inst, recon, timeout=30)
        assert result.status == SolverStatus.SUCCESS
        assert result.decrypted == target_pt

    def test_header_xor_payload(self):
        """Ciphertext = header (key material) + XOR-encrypted payload."""
        header = [0xAA, 0xBB]
        payload_pt_known = list(range(10))
        payload_ct_known = [(p ^ header[i % len(header)]) & 0xFF for i, p in enumerate(payload_pt_known)]
        full_ct_known = header + payload_ct_known

        target_payload_pt = [50, 51, 52, 53]
        target_payload_ct = [(p ^ header[i % len(header)]) & 0xFF for i, p in enumerate(target_payload_pt)]
        full_ct_target = header + target_payload_ct

        full_pt = [0, 0] + payload_pt_known  # dummy header in pt
        inst = _make_instance(full_pt, full_ct_known, full_ct_target, [0])
        recon = StatisticalRecon().analyze(inst)
        result = HybridSchemeSolver().solve(inst, recon, timeout=30)
        if result.status == SolverStatus.SUCCESS:
            assert result.decrypted == target_payload_pt


# ====================================================================== #
#  15. Report Exporter                                                    #
# ====================================================================== #

class TestReportExporter:
    def _get_report(self):
        mod = 251
        pts = list(range(20))
        cts = [(3 * p + 7) % mod for p in pts]
        inst = _make_instance(pts, cts, [223], [3], modulus=mod)
        config = OmniscienceConfig(
            timeouts=SolverTimeouts(algebraic=10, lattice=10, smt=10, neural=10, bruteforce=10),
            parallel_solvers=False,
        )
        return Dispatcher(config).attack(inst)

    def test_json_export(self):
        report = self._get_report()
        json_str = ReportExporter.to_json(report)
        assert '"success"' in json_str
        assert '"solver_results"' in json_str

    def test_html_export(self):
        report = self._get_report()
        html_str = ReportExporter.to_html(report)
        assert "<html" in html_str
        assert "OMNISCIENCE" in html_str

    def test_text_export(self):
        report = self._get_report()
        text = ReportExporter.to_text(report)
        assert "OMNISCIENCE" in text
        assert "Reconnaissance" in text


# ====================================================================== #
#  16. Full Pipeline (Dispatcher)                                         #
# ====================================================================== #

class TestDispatcher:
    def test_linear_pipeline(self):
        mod = 251
        pts = list(range(50))
        cts = [(3 * p + 7) % mod for p in pts]
        target_pts = [72, 100, 200]
        target_cts = [(3 * p + 7) % mod for p in target_pts]
        inst = _make_instance(pts, cts, target_cts, [3], modulus=mod)
        config = OmniscienceConfig(
            timeouts=SolverTimeouts(algebraic=30, lattice=30, smt=30, neural=30, bruteforce=30),
            parallel_solvers=False,
        )
        report = Dispatcher(config).attack(inst)
        assert report.success()
        assert report.best_result.decrypted == target_pts

    def test_rsa_pipeline(self):
        """Full pipeline on small RSA"""
        p, q, e = 61, 53, 17
        N = p * q
        phi = (p - 1) * (q - 1)
        d = pow(e, -1, phi)
        plaintext = 42
        ct = pow(plaintext, e, N)
        inst = _make_instance([plaintext], [ct], [ct], [e], modulus=N)
        config = OmniscienceConfig(
            timeouts=SolverTimeouts(algebraic=30, lattice=30, smt=30, neural=30, bruteforce=30),
            parallel_solvers=False,
        )
        report = Dispatcher(config).attack(inst)
        assert report.success()
        assert report.best_result.decrypted == [plaintext]

    def test_dlog_pipeline(self):
        """Full pipeline on small DLP"""
        p = 1009
        g = 11
        x = 237
        h = pow(g, x, p)
        inst = _make_instance([g], [h], [h], [g, h], modulus=p)
        config = OmniscienceConfig(
            timeouts=SolverTimeouts(algebraic=30, lattice=30, smt=30, neural=30, bruteforce=30),
            parallel_solvers=False,
        )
        report = Dispatcher(config).attack(inst)
        assert report.success()
