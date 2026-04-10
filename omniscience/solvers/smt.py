"""SMT Solver Module (Z3).

Translates the cryptographic problem into a satisfiability-modulo-theories
problem and uses Z3 to search for the private key by symbolic propagation.

Strategies:
  1. Bit-vector model: model the cipher as a bitvector function and assert
     known (P, C) pairs as constraints.
  2. Modular arithmetic model: model as integer arithmetic modulo N.
  3. Lookup-table model: for substitution-based ciphers, enumerate S-box
     entries as uninterpreted functions with known-pair constraints.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from omniscience.core.types import (
    AlgoFamily,
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)


class SMTSolver:
    """Z3-based symbolic cryptanalysis."""

    NAME = "smt_z3"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            import z3
        except ImportError:
            return self._fail("z3-solver not installed", t0)

        try:
            modulus = recon.estimated_modulus or instance.modulus

            # Strategy 1: Bitvector linear/polynomial model
            res = self._bv_poly_attack(instance, recon, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 2: Substitution-box recovery
            if recon.substitution_detected or recon.algo_family == AlgoFamily.SUBSTITUTION:
                res = self._sbox_attack(instance, modulus, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 3: Modular arithmetic model
            if modulus and modulus < 2**64:
                res = self._modular_attack(instance, modulus, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            return self._fail("Z3 exhausted all strategies", t0)
        except Exception as exc:
            log.exception("SMT solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  Strategy 1: Bitvector polynomial model                             #
    # ------------------------------------------------------------------ #

    def _bv_poly_attack(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        modulus: int | None,
        timeout: float,
        t0: float,
    ) -> SolverResult:
        """Model cipher as c = f(p, key) where f is a polynomial over bitvectors."""
        import z3

        log.info("[Z3/BV] Attempting bitvector polynomial model")
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        n_pairs = min(len(pt), len(ct), 50)
        if n_pairs < 2:
            return self._fail("Not enough pairs for BV model", t0)

        # Determine bitvector width
        max_val = max(max(pt[:n_pairs]), max(ct[:n_pairs]))
        if modulus:
            max_val = max(max_val, modulus)
        bw = max(8, max_val.bit_length() + 1) if max_val > 0 else 8
        bw = min(bw, 64)  # cap at 64 bits

        # Symbolic key coefficients for polynomial c = a0 + a1*p + a2*p^2 + ...
        max_deg = min(recon.polynomial_degree_estimate or 3, 5)
        key_vars = [z3.BitVec(f"a{i}", bw) for i in range(max_deg + 1)]

        solver = z3.Solver()
        solver.set("timeout", int(timeout * 1000))

        for i in range(n_pairs):
            p_val = z3.BitVecVal(pt[i], bw)
            c_val = z3.BitVecVal(ct[i], bw)
            # Build polynomial expression
            expr = key_vars[0]
            p_pow = p_val
            for d in range(1, max_deg + 1):
                expr = expr + key_vars[d] * p_pow
                p_pow = p_pow * p_val
            solver.add(expr == c_val)

        log.debug("[Z3/BV] %d constraints, %d-bit vectors, degree ≤ %d", n_pairs, bw, max_deg)
        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            coeffs = [model.eval(k, model_completion=True).as_long() for k in key_vars]
            log.info("[Z3/BV] SAT — coefficients: %s", coeffs)

            # Decrypt target
            ct_target = instance.ct_target_as_int_list()
            decrypted = self._decrypt_bv_poly(coeffs, ct_target, bw, modulus)

            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=coeffs,
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.90,
                details={"method": "bv_polynomial", "degree": max_deg, "bw": bw},
            )
        elif result == z3.unknown:
            return self._fail("Z3 BV solver timed out", t0)
        else:
            return self._fail("Z3 BV model is UNSAT", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 2: S-Box recovery                                         #
    # ------------------------------------------------------------------ #

    def _sbox_attack(
        self,
        instance: CryptoInstance,
        modulus: int | None,
        timeout: float,
        t0: float,
    ) -> SolverResult:
        """Recover a byte-level substitution box using Z3 array theory."""
        import z3

        log.info("[Z3/SBox] Attempting substitution-box recovery")
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        n = min(len(pt), len(ct))

        if n < 1:
            return self._fail("No pairs for S-Box recovery", t0)

        # S-Box as Z3 Array: Int → Int
        sbox = z3.Array("sbox", z3.BitVecSort(8), z3.BitVecSort(8))
        solver = z3.Solver()
        solver.set("timeout", int(timeout * 1000))

        # Constrain known mappings
        for i in range(n):
            p_val = z3.BitVecVal(pt[i] & 0xFF, 8)
            c_val = z3.BitVecVal(ct[i] & 0xFF, 8)
            solver.add(z3.Select(sbox, p_val) == c_val)

        # Bijectivity constraint (permutation)
        for i in range(256):
            for j in range(i + 1, 256):
                vi = z3.BitVecVal(i, 8)
                vj = z3.BitVecVal(j, 8)
                solver.add(z3.Select(sbox, vi) != z3.Select(sbox, vj))

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            # Extract full S-Box
            sbox_table = []
            for i in range(256):
                val = model.eval(z3.Select(sbox, z3.BitVecVal(i, 8)), model_completion=True)
                sbox_table.append(val.as_long())

            log.info("[Z3/SBox] Recovered S-Box")
            # Build inverse S-Box for decryption
            inv_sbox = [0] * 256
            for i, v in enumerate(sbox_table):
                inv_sbox[v] = i

            ct_target = instance.ct_target_as_int_list()
            decrypted = [inv_sbox[c & 0xFF] for c in ct_target]

            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=sbox_table,
                decrypted=decrypted,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.95,
                details={"method": "sbox_recovery", "sbox_size": 256},
            )

        return self._fail("S-Box recovery UNSAT or timeout", t0)

    # ------------------------------------------------------------------ #
    #  Strategy 3: Modular arithmetic model                               #
    # ------------------------------------------------------------------ #

    def _modular_attack(
        self,
        instance: CryptoInstance,
        modulus: int,
        timeout: float,
        t0: float,
    ) -> SolverResult:
        """Model as c ≡ f(p, key) (mod N) using Z3 integer arithmetic."""
        import z3

        log.info("[Z3/Mod] Attempting modular arithmetic model (mod %d)", modulus)
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        n_pairs = min(len(pt), len(ct), 30)

        # Symbolic unknowns: exponent e (RSA-like) or multiplier/additive key
        e = z3.Int("e")
        k = z3.Int("k")
        N = z3.IntVal(modulus)

        solver = z3.Solver()
        solver.set("timeout", int(timeout * 1000))

        # Model 1: c ≡ p^e mod N  (RSA-like)
        solver.push()
        solver.add(e > 1)
        solver.add(e < modulus)
        for i in range(min(n_pairs, 10)):
            p_val = z3.IntVal(pt[i])
            c_val = z3.IntVal(ct[i])
            # Z3 doesn't have native modpow, so we use modular constraint
            # p^e mod N == c  ⟺  ∃q: p^e = q*N + c
            q = z3.Int(f"q_{i}")
            # For small exponents, expand manually
            solver.add(p_val ** e == q * N + c_val)
            solver.add(q >= 0)

        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            e_val = model.eval(e).as_long()
            log.info("[Z3/Mod] Found RSA-like exponent e=%d", e_val)
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=e_val,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.70,
                details={"method": "modular_rsa_like", "e": e_val, "modulus": modulus},
            )
        solver.pop()

        # Model 2: c ≡ k * p mod N  (multiplicative)
        solver.push()
        solver.add(k > 0)
        solver.add(k < N)
        for i in range(n_pairs):
            p_val = z3.IntVal(pt[i])
            c_val = z3.IntVal(ct[i])
            solver.add((k * p_val) % N == c_val)

        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            k_val = model.eval(k).as_long()
            log.info("[Z3/Mod] Found multiplicative key k=%d", k_val)
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=k_val,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.85,
                details={"method": "modular_multiplicative", "k": k_val},
            )
        solver.pop()

        return self._fail("Z3 modular model did not find solution", t0)

    # ------------------------------------------------------------------ #
    #  Decryption helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decrypt_bv_poly(
        coeffs: list[int],
        ct_target: list[int],
        bw: int,
        modulus: int | None,
    ) -> list[int] | None:
        """Invert bitvector polynomial by exhaustive search (for small search spaces)."""
        mask = (1 << bw) - 1
        search_space = modulus if modulus and modulus < 2**20 else (1 << min(bw, 20))

        # Build forward lookup
        lookup: dict[int, int] = {}
        for p in range(search_space):
            val = coeffs[0]
            p_pow = p
            for d in range(1, len(coeffs)):
                val = (val + coeffs[d] * p_pow) & mask
                p_pow = (p_pow * p) & mask
            lookup[val & mask] = p

        result = []
        for c in ct_target:
            c_masked = c & mask
            if c_masked in lookup:
                result.append(lookup[c_masked])
            else:
                return None
        return result

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[Z3] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
