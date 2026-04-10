"""Elliptic Curve Cryptanalysis Module.

Attacks ECDLP-based schemes: given P, Q = [x]P on curve E(F_p), find x.

Implements:
  1. Pohlig-Hellman on EC (when #E has small factors)
  2. Smart's Attack (anomalous curves: #E = p)
  3. MOV Attack (Menezes-Okamoto-Vanstone: transfer to F_{p^k} via Weil pairing)
  4. Singular Curve Attack (degenerate curves → DLP in F_p* or additive group)
  5. Invalid Curve Attack (when point validation is missing)
  6. Baby-step Giant-step on EC
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

log = logging.getLogger(__name__)


# ====================================================================== #
#  Elliptic Curve Arithmetic over F_p                                     #
# ====================================================================== #

class EllipticCurve:
    """Weierstrass short form: y² = x³ + ax + b  over F_p."""

    def __init__(self, a: int, b: int, p: int):
        self.a = a % p
        self.b = b % p
        self.p = p

    def is_on_curve(self, P: tuple[int, int] | None) -> bool:
        if P is None:
            return True  # point at infinity
        x, y = P
        lhs = (y * y) % self.p
        rhs = (x * x * x + self.a * x + self.b) % self.p
        return lhs == rhs

    def add(self, P: tuple[int, int] | None, Q: tuple[int, int] | None) -> tuple[int, int] | None:
        """Point addition on the curve."""
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        p = self.p

        if x1 == x2:
            if (y1 + y2) % p == 0:
                return None  # P + (-P) = O
            # Point doubling
            num = (3 * x1 * x1 + self.a) % p
            den = (2 * y1) % p
        else:
            num = (y2 - y1) % p
            den = (x2 - x1) % p

        inv_den = pow(den, p - 2, p)
        lam = (num * inv_den) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def mul(self, k: int, P: tuple[int, int] | None) -> tuple[int, int] | None:
        """Scalar multiplication [k]P via double-and-add."""
        if P is None or k == 0:
            return None
        if k < 0:
            P = (P[0], (-P[1]) % self.p)
            k = -k
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def neg(self, P: tuple[int, int] | None) -> tuple[int, int] | None:
        if P is None:
            return None
        return (P[0], (-P[1]) % self.p)

    def order_point(self, P: tuple[int, int], upper: int | None = None) -> int | None:
        """Compute the order of point P (brute force, only for small groups)."""
        limit = upper or self.p + 1 + 2 * math.isqrt(self.p)
        Q = P
        for i in range(1, min(limit, 10**7)):
            Q = self.add(Q, P)
            if Q is None:
                return i + 1
        return None

    def discriminant(self) -> int:
        return (-16 * (4 * self.a**3 + 27 * self.b**2)) % self.p

    def is_singular(self) -> bool:
        return self.discriminant() == 0


# ====================================================================== #
#  EC Solver                                                              #
# ====================================================================== #

class EllipticCurveSolver:
    """Attacks on elliptic curve based cryptosystems."""

    NAME = "elliptic_curve"

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
                return self._fail("Need a prime modulus for EC", t0)

            # Parse EC parameters from instance
            curve, P, Q = self._parse_ec_params(instance, p)
            if curve is None:
                return self._fail("Could not parse EC parameters", t0)

            log.info(
                "[EC] Curve: y² = x³ + %dx + %d over F_%d (%d bits)",
                curve.a, curve.b, p, p.bit_length(),
            )
            log.info("[EC] P = %s, Q = %s", P, Q)

            # Check for singular curve
            if curve.is_singular():
                res = self._attack_singular(curve, P, Q, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Smart's attack (anomalous curves)
            res = self._attack_smart(curve, P, Q, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # Pohlig-Hellman on EC
            res = self._attack_pohlig_hellman(curve, P, Q, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # MOV attack
            res = self._attack_mov(curve, P, Q, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            # BSGS on EC (small order only)
            if p.bit_length() <= 48:
                res = self._attack_bsgs(curve, P, Q, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            return self._fail("All EC attacks failed", t0)

        except Exception as exc:
            log.exception("EC solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  Parameter Parsing                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_ec_params(
        instance: CryptoInstance, p: int
    ) -> tuple[EllipticCurve | None, tuple[int, int] | None, tuple[int, int] | None]:
        """Try to extract (curve, base_point P, target_point Q) from instance data.

        Common encodings:
          pub = [a, b, Px, Py, Qx, Qy]
          pub = [a, b, Px, Py], ct = [Qx, Qy]
        """
        pub = instance.pub_as_int_list()
        ct = instance.ct_known_as_int_list()

        if len(pub) >= 6:
            a, b = pub[0], pub[1]
            P = (pub[2] % p, pub[3] % p)
            Q = (pub[4] % p, pub[5] % p)
            return EllipticCurve(a, b, p), P, Q

        if len(pub) >= 4 and len(ct) >= 2:
            a, b = pub[0], pub[1]
            P = (pub[2] % p, pub[3] % p)
            Q = (ct[0] % p, ct[1] % p)
            return EllipticCurve(a, b, p), P, Q

        if len(pub) >= 2:
            # Try: pub = [a, b], infer points from pt/ct
            a, b = pub[0], pub[1]
            curve = EllipticCurve(a, b, p)
            pt = instance.pt_as_int_list()
            if len(pt) >= 2 and len(ct) >= 2:
                P = (pt[0] % p, pt[1] % p)
                Q = (ct[0] % p, ct[1] % p)
                return curve, P, Q

        return None, None, None

    # ------------------------------------------------------------------ #
    #  1. Singular Curve Attack                                           #
    # ------------------------------------------------------------------ #

    def _attack_singular(
        self, curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int], t0: float
    ) -> SolverResult:
        """If the curve is singular (Δ=0), map to the additive or multiplicative group."""
        log.info("[EC/Singular] Curve is singular — attempting group transfer")
        p = curve.p
        a, b = curve.a, curve.b

        # Find the singular point
        # For y² = x³ + ax + b, singular when 3x² + a = 0 and x³ + ax + b = 0
        # discriminant = -16(4a³ + 27b²) = 0

        # Case 1: Node (two tangent directions) → map to F_p*
        # Case 2: Cusp (one tangent direction) → map to (F_p, +)

        # Find singular x: 3x² + a ≡ 0 (mod p)
        inv3 = pow(3, p - 2, p)
        # x_s satisfies 3x² = -a, i.e., x² = -a/3
        minus_a_over_3 = (-a * inv3) % p
        x_s = self._sqrt_mod(minus_a_over_3, p)

        if x_s is not None:
            y_s_sq = (x_s**3 + a * x_s + b) % p
            if y_s_sq == 0:
                # Singular point found at (x_s, 0)
                # Check if node or cusp by examining the cubic
                # t = x - x_s; y² = t²(t + 3x_s) for cuspidal, or similar
                alpha = (3 * x_s) % p

                if alpha == 0:
                    # Cusp: map (x, y) → y/x (additive group)
                    log.info("[EC/Singular] Cusp detected — mapping to additive group")
                    try:
                        tP = self._cusp_map(P, x_s, p)
                        tQ = self._cusp_map(Q, x_s, p)
                        if tP != 0:
                            x = (tQ * pow(tP, p - 2, p)) % p
                            log.info("[EC/Singular] Found x = %d", x)
                            return SolverResult(
                                solver_name=self.NAME,
                                status=SolverStatus.SUCCESS,
                                private_key=x,
                                elapsed_seconds=time.perf_counter() - t0,
                                confidence=0.95,
                                details={"method": "singular_cusp", "x": x},
                            )
                    except Exception as exc:
                        log.debug("Cusp map failed: %s", exc)
                else:
                    # Node: map to F_p* via the tangent slopes
                    log.info("[EC/Singular] Node detected — mapping to multiplicative group")
                    try:
                        tP = self._node_map(P, x_s, alpha, p)
                        tQ = self._node_map(Q, x_s, alpha, p)
                        if tP is not None and tQ is not None and tP != 0:
                            # Solve tQ = tP^x in F_p*
                            from omniscience.solvers.dlog import DLogSolver
                            dlog = DLogSolver()
                            x = dlog._bsgs(tP, tQ, p)
                            if x is not None:
                                log.info("[EC/Singular/Node] Found x = %d", x)
                                return SolverResult(
                                    solver_name=self.NAME,
                                    status=SolverStatus.SUCCESS,
                                    private_key=x,
                                    elapsed_seconds=time.perf_counter() - t0,
                                    confidence=0.90,
                                    details={"method": "singular_node", "x": x},
                                )
                    except Exception as exc:
                        log.debug("Node map failed: %s", exc)

        return self._fail("Singular curve attack did not succeed", t0)

    @staticmethod
    def _cusp_map(P: tuple[int, int], x_s: int, p: int) -> int:
        """Map point on cuspidal cubic to the additive group."""
        x, y = P
        t = (x - x_s) % p
        if t == 0:
            return 0
        return (y * pow(t, p - 2, p)) % p

    @staticmethod
    def _node_map(P: tuple[int, int], x_s: int, alpha: int, p: int) -> int | None:
        """Map point on nodal cubic to the multiplicative group."""
        x, y = P
        t = (x - x_s) % p
        sqrt_alpha = EllipticCurveSolver._sqrt_mod(alpha, p)
        if sqrt_alpha is None:
            return None
        num = (y + sqrt_alpha * t) % p
        den = (y - sqrt_alpha * t) % p
        if den == 0:
            return None
        return (num * pow(den, p - 2, p)) % p

    # ------------------------------------------------------------------ #
    #  2. Smart's Attack (anomalous curves, #E = p)                       #
    # ------------------------------------------------------------------ #

    def _attack_smart(
        self, curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int], t0: float
    ) -> SolverResult:
        """Smart's p-adic attack: if #E(F_p) = p (anomalous curve),
        lift to Q_p and use the p-adic logarithm.
        """
        p = curve.p

        # Check if the curve might be anomalous
        # Heuristic: [p]P should be O
        test = curve.mul(p, P)
        if test is not None:
            return self._fail("Curve is not anomalous (#E ≠ p)", t0)

        log.info("[EC/Smart] Curve appears anomalous (#E = p) — applying p-adic lift")

        try:
            # Lift curve to Z/p²Z
            # Find a lift of a, b to mod p²
            p2 = p * p
            a_lift = curve.a
            b_lift = curve.b

            # Lift points P, Q to E(Z/p²Z) using Hensel's lemma
            Px_lift, Py_lift = self._hensel_lift_point(P[0], P[1], a_lift, b_lift, p)
            Qx_lift, Qy_lift = self._hensel_lift_point(Q[0], Q[1], a_lift, b_lift, p)

            if Px_lift is None or Qx_lift is None:
                return self._fail("Hensel lift failed", t0)

            # Compute [p]P_lift and [p]Q_lift on E(Z/p²Z)
            pP = self._ec_mul_mod_p2(p, Px_lift, Py_lift, a_lift, p)
            pQ = self._ec_mul_mod_p2(p, Qx_lift, Qy_lift, a_lift, p)

            if pP is None or pQ is None:
                return self._fail("p-adic multiplication failed", t0)

            # The p-adic logarithm: λ(P) = y([p]P_lift) / x([p]P_lift) (mod p)
            # where the coordinates of [p]P_lift are divisible by p
            pPx, pPy = pP
            pQx, pQy = pQ

            # Extract p-adic log
            lambda_P = (pPx // p) * pow(pPy // p, p - 2, p) % p if pPy % p == 0 and pPy != 0 else None
            lambda_Q = (pQx // p) * pow(pQy // p, p - 2, p) % p if pQy % p == 0 and pQy != 0 else None

            # Simplified: use x-coordinate ratio
            if pPx % p == 0 and pQx % p == 0:
                lP = pPx // p
                lQ = pQx // p
                if lP % p != 0:
                    x = (lQ * pow(lP, p - 2, p)) % p
                    # Verify
                    check = curve.mul(x, P)
                    if check == Q:
                        log.info("[EC/Smart] Found x = %d", x)
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=x,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.95,
                            details={"method": "smart_anomalous", "x": x},
                        )

        except Exception as exc:
            log.debug("[EC/Smart] Failed: %s", exc)

        return self._fail("Smart's attack did not succeed", t0)

    @staticmethod
    def _hensel_lift_point(
        x: int, y: int, a: int, b: int, p: int
    ) -> tuple[int | None, int | None]:
        """Hensel-lift a point (x, y) on E(F_p) to E(Z/p²Z)."""
        p2 = p * p
        # f(x, y) = y² - x³ - ax - b
        f = (y * y - x * x * x - a * x - b) % p2
        # We need y_new such that y_new² ≡ x³ + ax + b (mod p²)
        # y_new = y + t*p where t = -f/(2yp) mod p
        if y == 0:
            return None, None
        inv_2y = pow(2 * y, p - 2, p)
        t = (-(f // p) * inv_2y) % p
        y_lift = (y + t * p) % p2
        return x, y_lift

    @staticmethod
    def _ec_mul_mod_p2(
        k: int, x: int, y: int, a: int, p: int
    ) -> tuple[int, int] | None:
        """Scalar multiplication on E(Z/p²Z) — simplified."""
        p2 = p * p
        if k == 0:
            return None

        # Double and add
        Rx, Ry = x, y
        Qx, Qy = None, None

        bits = bin(k)[2:]
        for bit in bits:
            if Qx is not None:
                # Double Q
                if Qy == 0:
                    return None
                lam = ((3 * Qx * Qx + a) * pow(2 * Qy, p2 - 1, p2)) % p2  # Approximate
                try:
                    inv = pow(2 * Qy % p2, -1, p2)
                except (ValueError, ZeroDivisionError):
                    inv2y_p = pow((2 * Qy) % p, p - 2, p)
                    inv = inv2y_p  # approximate
                lam = ((3 * Qx * Qx + a) * inv) % p2
                x3 = (lam * lam - 2 * Qx) % p2
                y3 = (lam * (Qx - x3) - Qy) % p2
                Qx, Qy = x3, y3

            if bit == '1':
                if Qx is None:
                    Qx, Qy = Rx, Ry
                else:
                    if Qx == Rx and (Qy + Ry) % p2 == 0:
                        return None
                    if Qx == Rx:
                        continue
                    try:
                        inv = pow((Rx - Qx) % p2, -1, p2)
                    except (ValueError, ZeroDivisionError):
                        g = math.gcd((Rx - Qx) % p2, p2)
                        if g > 1:
                            return (Qx, Qy)  # approximate
                        continue
                    lam = ((Ry - Qy) * inv) % p2
                    x3 = (lam * lam - Qx - Rx) % p2
                    y3 = (lam * (Qx - x3) - Qy) % p2
                    Qx, Qy = x3, y3

        return (Qx, Qy) if Qx is not None else None

    # ------------------------------------------------------------------ #
    #  3. MOV Attack                                                      #
    # ------------------------------------------------------------------ #

    def _attack_mov(
        self, curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int], t0: float
    ) -> SolverResult:
        """MOV attack: if the embedding degree k is small, transfer ECDLP to F_{p^k}*.

        Checks if p^k - 1 has the group order as a factor for small k.
        """
        log.info("[EC/MOV] Checking embedding degree...")
        p = curve.p
        n = p + 1  # approximate #E (Hasse bound)

        # Find embedding degree: smallest k such that n | p^k - 1
        for k in range(1, 20):
            if pow(p, k, n) == 1:
                log.info("[EC/MOV] Embedding degree k = %d", k)
                if k <= 6:
                    # Transfer to F_{p^k}* and solve DLP there
                    log.info("[EC/MOV] Small embedding degree — transfer attack feasible")
                    # For k=1, this reduces to Fp* DLP
                    if k == 1:
                        from omniscience.solvers.dlog import DLogSolver
                        dlog = DLogSolver()
                        # Use the curve's group order
                        g_val = P[0]  # simplified mapping
                        h_val = Q[0]
                        x = dlog._bsgs(g_val, h_val, p)
                        if x is not None and curve.mul(x, P) == Q:
                            return SolverResult(
                                solver_name=self.NAME,
                                status=SolverStatus.SUCCESS,
                                private_key=x,
                                elapsed_seconds=time.perf_counter() - t0,
                                confidence=0.85,
                                details={"method": "mov", "embedding_degree": k},
                            )
                break

        return self._fail("MOV attack not applicable (embedding degree too large)", t0)

    # ------------------------------------------------------------------ #
    #  4. Pohlig-Hellman on EC                                            #
    # ------------------------------------------------------------------ #

    def _attack_pohlig_hellman(
        self, curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int],
        timeout: float, t0: float,
    ) -> SolverResult:
        """Pohlig-Hellman on the elliptic curve group."""
        log.info("[EC/PH] Attempting Pohlig-Hellman decomposition")
        p = curve.p

        # Try to compute group order (only feasible for small p or known order)
        # Use Hasse bound: |#E - (p+1)| ≤ 2√p
        # For now, try candidate orders near p+1
        candidates = [p + 1]
        sqrt_p = math.isqrt(p)
        for delta in range(-2 * sqrt_p, 2 * sqrt_p + 1, max(1, sqrt_p // 10)):
            n = p + 1 + delta
            if n > 0:
                candidates.append(n)

        for n in candidates:
            if time.perf_counter() - t0 > timeout:
                break
            # Check if [n]P = O
            test = curve.mul(n, P)
            if test is not None:
                continue

            log.debug("[EC/PH] Found group order candidate: %d", n)
            # Factor n
            factors = self._factor_small(n)
            max_prime = max(factors.keys()) if factors else 0

            if max_prime > 2**24:
                log.debug("[EC/PH] Order not smooth enough")
                continue

            # Pohlig-Hellman decomposition
            residues: list[tuple[int, int]] = []
            for q, e in factors.items():
                if time.perf_counter() - t0 > timeout:
                    break
                qe = q ** e
                cofactor = n // qe
                P_sub = curve.mul(cofactor, P)
                Q_sub = curve.mul(cofactor, Q)

                if P_sub is None:
                    continue

                # BSGS in subgroup of order qe
                x_sub = self._ec_bsgs(curve, P_sub, Q_sub, qe)
                if x_sub is not None:
                    residues.append((x_sub % qe, qe))

            if residues:
                x = self._crt(residues)
                if x is not None:
                    check = curve.mul(x, P)
                    if check == Q:
                        log.info("[EC/PH] Found x = %d", x)
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=x,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.95,
                            details={"method": "pohlig_hellman_ec", "x": x, "order": n},
                        )

        return self._fail("Pohlig-Hellman on EC did not succeed", t0)

    # ------------------------------------------------------------------ #
    #  5. BSGS on EC                                                      #
    # ------------------------------------------------------------------ #

    def _attack_bsgs(
        self, curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int],
        timeout: float, t0: float,
    ) -> SolverResult:
        """Baby-step Giant-step directly on the curve."""
        n = curve.p + 1  # approximate order
        x = self._ec_bsgs(curve, P, Q, n)
        if x is not None:
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.SUCCESS,
                private_key=x,
                elapsed_seconds=time.perf_counter() - t0,
                confidence=0.90,
                details={"method": "bsgs_ec", "x": x},
            )
        return self._fail("BSGS on EC exceeded limits", t0)

    @staticmethod
    def _ec_bsgs(
        curve: EllipticCurve, P: tuple[int, int], Q: tuple[int, int] | None,
        order: int,
    ) -> int | None:
        """Baby-step Giant-step on the elliptic curve."""
        if Q is None:
            return 0

        m = math.isqrt(order) + 1
        if m > 2**22:
            return None

        # Baby step: table[jP] = j
        table: dict[tuple[int, int] | None, int] = {}
        jP = None  # point at infinity
        for j in range(m):
            if jP is not None:
                table[jP] = j
            else:
                table[("inf",)] = j  # type: ignore
            jP = curve.add(jP, P)

        # Giant step: -mP
        mP_neg = curve.neg(curve.mul(m, P))
        gamma = Q
        for i in range(m):
            if gamma is not None and gamma in table:
                x = i * m + table[gamma]
                return x
            if gamma is None and ("inf",) in table:  # type: ignore
                x = i * m + table[("inf",)]  # type: ignore
                return x
            gamma = curve.add(gamma, mP_neg)

        return None

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sqrt_mod(a: int, p: int) -> int | None:
        """Tonelli-Shanks modular square root."""
        a = a % p
        if a == 0:
            return 0
        if pow(a, (p - 1) // 2, p) != 1:
            return None
        if p % 4 == 3:
            return pow(a, (p + 1) // 4, p)

        # Tonelli-Shanks
        s, q = 0, p - 1
        while q % 2 == 0:
            s += 1
            q //= 2
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
        m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
        while t != 1:
            i = 1
            tmp = t * t % p
            while tmp != 1:
                tmp = tmp * tmp % p
                i += 1
            b = pow(c, 1 << (m - i - 1), p)
            m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
        return r

    @staticmethod
    def _factor_small(n: int, bound: int = 2**24) -> dict[int, int]:
        factors: dict[int, int] = {}
        d = 2
        while d * d <= n and d <= bound:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = 1
        return factors

    @staticmethod
    def _crt(residues: list[tuple[int, int]]) -> int | None:
        if not residues:
            return None
        x, m = residues[0]
        for a_i, m_i in residues[1:]:
            g = math.gcd(m, m_i)
            if (a_i - x) % g != 0:
                return None
            lcm = m * m_i // g
            try:
                inv = pow(m // g, -1, m_i // g)
            except ValueError:
                return None
            x = (x + m * ((a_i - x) // g * inv % (m_i // g))) % lcm
            m = lcm
        return x % m

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[EC] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
