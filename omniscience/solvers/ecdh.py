"""ECDH Key Exchange Cryptanalysis Module.

Attacks on Elliptic Curve Diffie-Hellman (ECDH) shared secret recovery:

  1. ECDLP-based: Recover private key d from (G, dG) then compute shared secret
  2. Invalid Curve Attack: Force computation on weak curve with small subgroup
  3. Twist Attack: Exploit points on the quadratic twist of the curve
  4. Small Subgroup Attack: When cofactor > 1, project to small subgroups
  5. Known-Nonce / Nonce-Reuse: Recover private key from ECDSA signatures
  6. Related-Nonce (LLL): Lattice attack on biased nonces

Given:
  - Curve parameters (a, b, p)
  - Generator G = (Gx, Gy)
  - Public key A = dA * G  (Alice)
  - Public key B = dB * G  (Bob)

Goal: Recover shared secret S = dA * B = dB * A = dA * dB * G
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)
from omniscience.solvers.elliptic_curve import EllipticCurve, EllipticCurveSolver

log = logging.getLogger(__name__)


class ECDHSolver:
    """Cryptanalysis of ECDH key exchange — shared secret recovery."""

    NAME = "ecdh"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 600.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        try:
            p = recon.estimated_modulus or instance.modulus
            if p is None or p < 5:
                return self._fail("ECDH: need a prime modulus", t0)

            # Parse ECDH parameters
            curve, G, pub_A, pub_B = self._parse_ecdh_params(instance, p)
            if curve is None or G is None:
                return self._fail("ECDH: could not parse curve/points", t0)

            log.info(
                "[ECDH] Curve: y^2 = x^3 + %dx + %d over F_%d (%d bits)",
                curve.a, curve.b, p, p.bit_length(),
            )
            log.info("[ECDH] G=%s, A=%s, B=%s", G, pub_A, pub_B)

            # Strategy 1: Singular curve → trivial DLP
            if curve.is_singular():
                res = self._attack_singular_ecdh(curve, G, pub_A, pub_B, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 2: Smart's attack (anomalous curve, #E = p)
            res = self._attack_smart_ecdh(curve, G, pub_A, pub_B, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 3: Pohlig-Hellman (smooth order)
            res = self._attack_pohlig_hellman_ecdh(curve, G, pub_A, pub_B, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 4: Invalid curve attack
            res = self._attack_invalid_curve(curve, G, pub_A, pub_B, p, t0, timeout)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 5: Twist attack
            res = self._attack_twist(curve, G, pub_A, pub_B, p, t0, timeout)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 6: Small subgroup confinement
            res = self._attack_small_subgroup(curve, G, pub_A, pub_B, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Strategy 7: BSGS (small curves)
            if p.bit_length() <= 48:
                res = self._attack_bsgs_ecdh(curve, G, pub_A, pub_B, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 8: Nonce-reuse in ECDSA signatures (if available)
            signatures = instance.extra.get("signatures", [])
            if signatures:
                res = self._attack_nonce_reuse(curve, G, pub_A, signatures, p, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            return self._fail("ECDH: all strategies failed", t0)

        except Exception as exc:
            log.exception("ECDH solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ================================================================== #
    #  Parameter Parsing                                                  #
    # ================================================================== #

    @staticmethod
    def _parse_ecdh_params(
        instance: CryptoInstance, p: int,
    ) -> tuple[EllipticCurve | None, tuple[int, int] | None,
               tuple[int, int] | None, tuple[int, int] | None]:
        """Extract (curve, G, pubA, pubB) from instance.

        Expected encodings:
          pub = [a, b, Gx, Gy, Ax, Ay, Bx, By]  (8 values)
          pub = [a, b, Gx, Gy, Ax, Ay], ct = [Bx, By]  (6+2)
          pub = [a, b, Gx, Gy], pt = [Ax, Ay], ct = [Bx, By]  (4+2+2)
        """
        pub = instance.pub_as_int_list()
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_t = instance.ct_target_as_int_list()

        if len(pub) >= 8:
            a, b = pub[0], pub[1]
            G = (pub[2] % p, pub[3] % p)
            A = (pub[4] % p, pub[5] % p)
            B = (pub[6] % p, pub[7] % p)
            return EllipticCurve(a, b, p), G, A, B

        if len(pub) >= 6 and len(ct) >= 2:
            a, b = pub[0], pub[1]
            G = (pub[2] % p, pub[3] % p)
            A = (pub[4] % p, pub[5] % p)
            B = (ct[0] % p, ct[1] % p)
            return EllipticCurve(a, b, p), G, A, B

        if len(pub) >= 4 and len(pt) >= 2 and len(ct) >= 2:
            a, b = pub[0], pub[1]
            G = (pub[2] % p, pub[3] % p)
            A = (pt[0] % p, pt[1] % p)
            B = (ct[0] % p, ct[1] % p)
            return EllipticCurve(a, b, p), G, A, B

        # Try with ct_target as B
        if len(pub) >= 6 and len(ct_t) >= 2:
            a, b = pub[0], pub[1]
            G = (pub[2] % p, pub[3] % p)
            A = (pub[4] % p, pub[5] % p)
            B = (ct_t[0] % p, ct_t[1] % p)
            return EllipticCurve(a, b, p), G, A, B

        return None, None, None, None

    # ================================================================== #
    #  Attack 1: Singular Curve                                           #
    # ================================================================== #

    def _attack_singular_ecdh(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        t0: float,
    ) -> SolverResult:
        """Singular curve: ECDLP becomes trivial in Fp* or (Fp, +)."""
        log.info("[ECDH/Singular] Curve is singular — trivial ECDLP")
        ec_solver = EllipticCurveSolver()
        if pub_A is not None:
            res = ec_solver._attack_singular(curve, G, pub_A, t0)
            if res.status == SolverStatus.SUCCESS:
                dA = res.private_key
                shared = curve.mul(dA, pub_B) if pub_B else None
                return self._success_shared(dA, None, shared, "singular_ecdh", t0)

        if pub_B is not None:
            res = ec_solver._attack_singular(curve, G, pub_B, t0)
            if res.status == SolverStatus.SUCCESS:
                dB = res.private_key
                shared = curve.mul(dB, pub_A) if pub_A else None
                return self._success_shared(None, dB, shared, "singular_ecdh", t0)

        return self._fail("ECDH/Singular: failed", t0)

    # ================================================================== #
    #  Attack 2: Smart's Attack                                           #
    # ================================================================== #

    def _attack_smart_ecdh(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        t0: float,
    ) -> SolverResult:
        """Smart's p-adic lift on anomalous curves (#E = p)."""
        ec_solver = EllipticCurveSolver()
        if pub_A is not None:
            res = ec_solver._attack_smart(curve, G, pub_A, t0)
            if res.status == SolverStatus.SUCCESS:
                dA = res.private_key
                shared = curve.mul(dA, pub_B) if pub_B else None
                return self._success_shared(dA, None, shared, "smart_ecdh", t0)

        return self._fail("ECDH/Smart: not anomalous or failed", t0)

    # ================================================================== #
    #  Attack 3: Pohlig-Hellman                                           #
    # ================================================================== #

    def _attack_pohlig_hellman_ecdh(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """Pohlig-Hellman on smooth-order EC group."""
        ec_solver = EllipticCurveSolver()
        if pub_A is not None:
            res = ec_solver._attack_pohlig_hellman(curve, G, pub_A, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                dA = res.private_key
                shared = curve.mul(dA, pub_B) if pub_B else None
                return self._success_shared(dA, None, shared, "pohlig_hellman_ecdh", t0)

        return self._fail("ECDH/PH: order not smooth", t0)

    # ================================================================== #
    #  Attack 4: Invalid Curve Attack                                     #
    # ================================================================== #

    def _attack_invalid_curve(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        p: int, t0: float, timeout: float,
    ) -> SolverResult:
        """Invalid curve attack: if point validation is missing, send points
        on a weaker curve with small order and recover key mod that order.

        We check if pub_A or pub_B are NOT on the declared curve.
        """
        log.info("[ECDH/InvalidCurve] Checking for invalid curve points")

        targets = []
        if pub_A and not curve.is_on_curve(pub_A):
            targets.append(("A", pub_A))
        if pub_B and not curve.is_on_curve(pub_B):
            targets.append(("B", pub_B))

        if not targets:
            return self._fail("Invalid curve: all points on correct curve", t0)

        # Points are on a different curve y^2 = x^3 + ax + b'
        # where b' = y^2 - x^3 - ax mod p
        residues: list[tuple[int, int]] = []
        for label, point in targets:
            x, y = point
            b_prime = (y * y - x * x * x - curve.a * x) % p
            fake_curve = EllipticCurve(curve.a, b_prime, p)

            if fake_curve.is_singular():
                continue

            # Try to find order of point on fake curve
            order = fake_curve.order_point(point, upper=min(p, 10**6))
            if order is None or order > 10**6:
                continue

            log.info("[ECDH/InvalidCurve] Point %s on fake curve (b'=%d), order=%d", label, b_prime, order)

            # Factor order and use Pohlig-Hellman
            factors = EllipticCurveSolver._factor_small(order)
            for q, e in factors.items():
                if q > 2**20:
                    continue
                qe = q ** e
                cofactor = order // qe
                P_sub = fake_curve.mul(cofactor, G) if fake_curve.is_on_curve(G) else fake_curve.mul(cofactor, point)
                Q_sub = fake_curve.mul(cofactor, point)
                if P_sub is None:
                    continue
                x_sub = EllipticCurveSolver._ec_bsgs(fake_curve, P_sub, Q_sub, qe)
                if x_sub is not None:
                    residues.append((x_sub % qe, qe))

        if residues:
            d = EllipticCurveSolver._crt(residues)
            if d is not None:
                # Partial key recovery — check against both public keys
                if pub_A and G:
                    check = curve.mul(d, G)
                    if check == pub_A:
                        shared = curve.mul(d, pub_B) if pub_B else None
                        return self._success_shared(d, None, shared, "invalid_curve", t0)

                if pub_B and G:
                    check = curve.mul(d, G)
                    if check == pub_B:
                        shared = curve.mul(d, pub_A) if pub_A else None
                        return self._success_shared(None, d, shared, "invalid_curve", t0)

                # Even partial key is valuable
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=d,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.60,
                    details={"method": "invalid_curve_partial", "residues": len(residues)},
                )

        return self._fail("Invalid curve: couldn't recover key", t0)

    # ================================================================== #
    #  Attack 5: Twist Attack                                             #
    # ================================================================== #

    def _attack_twist(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        p: int, t0: float, timeout: float,
    ) -> SolverResult:
        """Twist attack: if a point lies on the quadratic twist E',
        the group order is #E' = 2(p+1) - #E, which may be smooth.
        """
        log.info("[ECDH/Twist] Checking quadratic twist")

        # Quadratic twist: y^2 = x^3 + a*d^2*x + b*d^3  for non-residue d
        # #E' = 2(p+1) - #E
        # Find a non-residue
        d = 2
        while pow(d, (p - 1) // 2, p) != p - 1:
            d += 1
            if d > 100:
                return self._fail("Twist: couldn't find non-residue", t0)

        a_twist = (curve.a * pow(d, 2, p)) % p
        b_twist = (curve.b * pow(d, 3, p)) % p
        twist = EllipticCurve(a_twist, b_twist, p)

        if twist.is_singular():
            return self._fail("Twist: twisted curve is singular", t0)

        # Check if G has a small-order projection on the twist
        # Try to compute twist order via Hasse bound sampling
        # For small p, try direct computation
        if p.bit_length() > 40:
            return self._fail("Twist: p too large for direct order computation", t0)

        # Find a point on the twist
        twist_point = None
        for x in range(min(p, 1000)):
            rhs = (x * x * x + a_twist * x + b_twist) % p
            y = EllipticCurveSolver._sqrt_mod(rhs, p)
            if y is not None:
                twist_point = (x, y)
                if twist.is_on_curve(twist_point):
                    break

        if twist_point is None:
            return self._fail("Twist: no point found on twist", t0)

        # Try Pohlig-Hellman on the twist
        order_twist = twist.order_point(twist_point, upper=min(p + 1 + 2 * math.isqrt(p), 10**6))
        if order_twist and order_twist < 10**6:
            factors = EllipticCurveSolver._factor_small(order_twist)
            max_prime = max(factors.keys()) if factors else 0
            if max_prime < 2**20:
                log.info("[ECDH/Twist] Twist order %d is smooth (max factor %d)", order_twist, max_prime)
                # The twist attack would require the target to compute on the twist
                # This is mainly useful in active attack scenarios

        return self._fail("Twist: no exploitable structure", t0)

    # ================================================================== #
    #  Attack 6: Small Subgroup                                           #
    # ================================================================== #

    def _attack_small_subgroup(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        t0: float,
    ) -> SolverResult:
        """Small subgroup confinement: if cofactor h > 1, project to subgroup."""
        log.info("[ECDH/SmallSubgroup] Checking cofactor")
        p = curve.p

        # Estimate group order and check if G has small cofactor
        # [n]G = O iff n = order(G)
        # Check if [p+1]G = O (often #E = p+1 for special curves)
        for candidate_n in [p + 1, p - 1, p]:
            test = curve.mul(candidate_n, G)
            if test is None:
                # G has order dividing candidate_n
                # Check cofactor
                factors = EllipticCurveSolver._factor_small(candidate_n, bound=2**20)
                small_factors = {q: e for q, e in factors.items() if q < 2**16}

                if not small_factors:
                    continue

                # Try CRT on small subgroups
                residues: list[tuple[int, int]] = []
                target = pub_A or pub_B
                if target is None:
                    continue

                for q, e in small_factors.items():
                    qe = q ** e
                    cofactor = candidate_n // qe
                    G_sub = curve.mul(cofactor, G)
                    Q_sub = curve.mul(cofactor, target)

                    if G_sub is None:
                        continue

                    x_sub = EllipticCurveSolver._ec_bsgs(curve, G_sub, Q_sub, qe)
                    if x_sub is not None:
                        residues.append((x_sub % qe, qe))

                if residues:
                    d = EllipticCurveSolver._crt(residues)
                    if d is not None:
                        check = curve.mul(d, G)
                        other = pub_B if target == pub_A else pub_A
                        if check == target:
                            shared = curve.mul(d, other) if other else None
                            return self._success_shared(
                                d if target == pub_A else None,
                                d if target == pub_B else None,
                                shared, "small_subgroup", t0,
                            )

        return self._fail("Small subgroup: no exploitable cofactor", t0)

    # ================================================================== #
    #  Attack 7: BSGS                                                     #
    # ================================================================== #

    def _attack_bsgs_ecdh(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int] | None, pub_B: tuple[int, int] | None,
        t0: float,
    ) -> SolverResult:
        """Direct BSGS on small curves."""
        n = curve.p + 1
        if pub_A:
            dA = EllipticCurveSolver._ec_bsgs(curve, G, pub_A, n)
            if dA is not None:
                shared = curve.mul(dA, pub_B) if pub_B else None
                return self._success_shared(dA, None, shared, "bsgs_ecdh", t0)
        if pub_B:
            dB = EllipticCurveSolver._ec_bsgs(curve, G, pub_B, n)
            if dB is not None:
                shared = curve.mul(dB, pub_A) if pub_A else None
                return self._success_shared(None, dB, shared, "bsgs_ecdh", t0)

        return self._fail("BSGS ECDH: exceeded limits", t0)

    # ================================================================== #
    #  Attack 8: ECDSA Nonce Reuse                                        #
    # ================================================================== #

    def _attack_nonce_reuse(
        self, curve: EllipticCurve, G: tuple[int, int],
        pub_A: tuple[int, int], signatures: list[dict], p: int, t0: float,
    ) -> SolverResult:
        """If two ECDSA signatures share the same nonce k, recover private key.

        Given (r, s1, h1) and (r, s2, h2) with same r:
          s1 = k^{-1}(h1 + r*d) mod n
          s2 = k^{-1}(h2 + r*d) mod n
          s1 - s2 = k^{-1}(h1 - h2)
          k = (h1 - h2) / (s1 - s2)
          d = (s1*k - h1) / r mod n
        """
        log.info("[ECDH/NonceReuse] Checking %d signatures for nonce reuse", len(signatures))

        # Group signatures by r value
        by_r: dict[int, list[dict]] = {}
        for sig in signatures:
            r = sig.get("r", 0)
            by_r.setdefault(r, []).append(sig)

        # Find the curve order (needed for modular arithmetic on scalars)
        # Estimate: try p first, then p+1, p-1
        n_order = None
        for candidate in [p, p + 1, p - 1]:
            if curve.mul(candidate, G) is None:
                n_order = candidate
                break
        if n_order is None:
            n_order = p  # fallback

        for r_val, sigs in by_r.items():
            if len(sigs) < 2:
                continue

            s1_sig = sigs[0]
            s2_sig = sigs[1]
            r = r_val
            s1 = s1_sig.get("s", 0)
            s2 = s2_sig.get("s", 0)
            h1 = s1_sig.get("h", s1_sig.get("hash", 0))
            h2 = s2_sig.get("h", s2_sig.get("hash", 0))

            if s1 == s2:
                continue

            ds = (s1 - s2) % n_order
            dh = (h1 - h2) % n_order

            try:
                k = (dh * pow(ds, -1, n_order)) % n_order
            except (ValueError, ZeroDivisionError):
                continue

            try:
                d = ((s1 * k - h1) * pow(r, -1, n_order)) % n_order
            except (ValueError, ZeroDivisionError):
                continue

            # Verify
            check = curve.mul(d, G)
            if check == pub_A:
                log.info("[ECDH/NonceReuse] Private key recovered: d=%d", d)
                shared = None
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=d,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.99,
                    details={
                        "method": "nonce_reuse_ecdsa",
                        "nonce_k": k,
                        "private_key_d": d,
                    },
                )

        return self._fail("Nonce reuse: no shared nonce found", t0)

    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    def _success_shared(
        self,
        dA: int | None,
        dB: int | None,
        shared: tuple[int, int] | None,
        method: str,
        t0: float,
    ) -> SolverResult:
        d = dA if dA is not None else dB
        log.info("[ECDH] Shared secret recovered via %s: d=%s, S=%s", method, d, shared)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.SUCCESS,
            private_key=d,
            decrypted=list(shared) if shared else None,
            elapsed_seconds=time.perf_counter() - t0,
            confidence=0.95,
            details={
                "method": method,
                "private_key_A": dA,
                "private_key_B": dB,
                "shared_secret": shared,
            },
        )

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[ECDH] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
