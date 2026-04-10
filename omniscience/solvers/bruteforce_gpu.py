"""GPU-Accelerated Brute-Force & Collision Search Module.

Leverages CuPy and PyCUDA for massively parallel key-space exhaustion
and birthday-bound collision attacks.

Strategies:
  1. Exhaustive key search (GPU-parallel)
  2. Birthday / collision search (GPU hash tables)
  3. Rainbow table generation & lookup
  4. Modular exponentiation brute-force (small exponent / small key)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from omniscience.core.types import (
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
#  CUDA Kernels (PyCUDA raw CUDA C)                                       #
# ---------------------------------------------------------------------- #

_CUDA_BRUTEFORCE_KERNEL = r"""
__global__ void bruteforce_linear(
    const long long *ct_target,    // ciphertext to crack
    int ct_len,
    long long modulus,
    long long a_start,             // search range start
    long long a_end,               // search range end
    long long b_start,
    long long b_end,
    long long *result_a,           // output: found a
    long long *result_b,           // output: found b
    int *found                     // flag: 1 if found
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long a_range = a_end - a_start;
    long long b_range = b_end - b_start;
    long long total = a_range * b_range;

    if (idx >= total || *found) return;

    long long a = a_start + idx / b_range;
    long long b = b_start + idx % b_range;

    // Check: does E(pt) = a*pt + b mod m match ct for all known pairs?
    // Simplified: check first ciphertext byte
    long long c0 = (a * 0 + b) % modulus;  // E(0)
    if (c0 == ct_target[0]) {
        atomicExch(found, 1);
        *result_a = a;
        *result_b = b;
    }
}

__global__ void bruteforce_modpow(
    const long long *bases,        // plaintext values
    const long long *targets,      // ciphertext values
    int n_pairs,
    long long modulus,
    long long exp_start,
    long long exp_end,
    long long *result_exp,
    int *found
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long e = exp_start + idx;
    if (e >= exp_end || *found) return;

    // Check: bases[0]^e mod modulus == targets[0]
    long long base = bases[0] % modulus;
    long long result = 1;
    long long exp = e;
    long long b = base;
    while (exp > 0) {
        if (exp & 1) result = result * b % modulus;
        b = b * b % modulus;
        exp >>= 1;
    }

    if (result == targets[0]) {
        // Verify with more pairs
        int match = 1;
        for (int i = 1; i < n_pairs && i < 5; i++) {
            long long r2 = 1;
            long long b2 = bases[i] % modulus;
            long long e2 = e;
            while (e2 > 0) {
                if (e2 & 1) r2 = r2 * b2 % modulus;
                b2 = b2 * b2 % modulus;
                e2 >>= 1;
            }
            if (r2 != targets[i]) { match = 0; break; }
        }
        if (match) {
            atomicExch(found, 1);
            *result_exp = e;
        }
    }
}

__global__ void collision_search(
    const long long *table_keys,   // pre-computed f(x) values
    const long long *table_vals,   // corresponding x values
    int table_size,
    const long long *search_vals,  // g(y) values to match
    const long long *search_idx,   // corresponding y indices
    int search_size,
    long long *result_x,
    long long *result_y,
    int *found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= search_size || *found) return;

    long long val = search_vals[idx];

    // Binary search in sorted table
    int lo = 0, hi = table_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (table_keys[mid] == val) {
            atomicExch(found, 1);
            *result_x = table_vals[mid];
            *result_y = search_idx[idx];
            return;
        }
        if (table_keys[mid] < val) lo = mid + 1;
        else hi = mid - 1;
    }
}
"""


class BruteForceGPUSolver:
    """GPU-accelerated brute-force and collision attacks."""

    NAME = "bruteforce_gpu"

    def solve(
        self,
        instance: CryptoInstance,
        recon: ReconResult,
        timeout: float = 1800.0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        modulus = recon.estimated_modulus or instance.modulus

        # Decide between GPU and CPU
        gpu_available = self._check_gpu()

        try:
            # Strategy 1: GPU modular exponentiation brute-force
            if modulus and modulus > 1:
                res = self._bruteforce_modpow(instance, modulus, gpu_available, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 2: Collision / birthday attack
            if modulus and modulus > 1:
                res = self._collision_attack(instance, modulus, gpu_available, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 3: Linear key-space exhaustion
            if modulus and modulus < 2**24:
                res = self._bruteforce_linear(instance, modulus, gpu_available, timeout, t0)
                if res.status == SolverStatus.SUCCESS:
                    return res

            # Strategy 4: CPU fallback — small keyspace exhaustion
            res = self._bruteforce_cpu_generic(instance, modulus, timeout, t0)
            if res.status == SolverStatus.SUCCESS:
                return res

            return self._fail("Brute-force exhausted all strategies", t0)

        except Exception as exc:
            log.exception("Brute-force solver crashed")
            return SolverResult(
                solver_name=self.NAME,
                status=SolverStatus.FAILED,
                elapsed_seconds=time.perf_counter() - t0,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    #  GPU modular exponentiation brute-force                             #
    # ------------------------------------------------------------------ #

    def _bruteforce_modpow(
        self, instance: CryptoInstance, modulus: int,
        gpu: bool, timeout: float, t0: float,
    ) -> SolverResult:
        """Brute-force search for exponent e: c = p^e mod N."""
        log.info("[BF/ModPow] Searching for exponent (GPU=%s)", gpu)
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        n_pairs = min(len(pt), len(ct), 10)
        if n_pairs < 1:
            return self._fail("No pairs for modpow brute-force", t0)

        max_exp = min(modulus, 2**24)

        if gpu and self._check_gpu():
            return self._modpow_gpu(pt[:n_pairs], ct[:n_pairs], modulus, max_exp, t0)

        # CPU fallback
        return self._modpow_cpu(pt[:n_pairs], ct[:n_pairs], modulus, max_exp, timeout, t0)

    def _modpow_gpu(
        self, pt: list[int], ct: list[int], modulus: int, max_exp: int, t0: float
    ) -> SolverResult:
        """GPU-accelerated modpow search using CuPy."""
        try:
            import cupy as cp

            log.info("[BF/ModPow/GPU] Launching CuPy kernel for %d exponents", max_exp)
            bases = cp.array(pt, dtype=cp.int64)
            targets = cp.array(ct, dtype=cp.int64)

            # Batch processing
            batch_size = 1 << 20  # 1M per batch
            for start in range(2, max_exp, batch_size):
                end = min(start + batch_size, max_exp)
                exponents = cp.arange(start, end, dtype=cp.int64)

                # Vectorized modpow for first pair
                results = self._cupy_modpow(bases[0], exponents, modulus)
                matches = cp.where(results == targets[0])[0]

                for idx in matches.get():
                    e = int(exponents[idx])
                    # Verify with all pairs
                    if all(pow(pt[i], e, modulus) == ct[i] for i in range(len(pt))):
                        log.info("[BF/ModPow/GPU] Found exponent e = %d", e)
                        ct_target = []  # will be decrypted by caller
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=e,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.95,
                            details={"method": "modpow_gpu", "exponent": e},
                        )

        except ImportError:
            log.debug("CuPy not available, falling back to CPU")
        except Exception as exc:
            log.warning("GPU modpow failed: %s", exc)

        return self._fail("GPU modpow did not find exponent", t0)

    @staticmethod
    def _cupy_modpow(base, exponents, modulus):
        """Vectorized modular exponentiation using CuPy."""
        import cupy as cp

        result = cp.ones_like(exponents)
        b = cp.full_like(exponents, int(base) % modulus)
        e = exponents.copy()
        mod = int(modulus)

        while cp.any(e > 0):
            mask = (e & 1).astype(cp.bool_)
            result[mask] = (result[mask] * b[mask]) % mod
            b = (b * b) % mod
            e >>= 1

        return result

    def _modpow_cpu(
        self, pt: list[int], ct: list[int], modulus: int,
        max_exp: int, timeout: float, t0: float,
    ) -> SolverResult:
        """CPU modpow brute-force."""
        log.info("[BF/ModPow/CPU] Searching exponents 2..%d", max_exp)
        for e in range(2, max_exp):
            if time.perf_counter() - t0 > timeout:
                break
            if pow(pt[0], e, modulus) == ct[0]:
                if all(pow(pt[i], e, modulus) == ct[i] for i in range(len(pt))):
                    log.info("[BF/ModPow/CPU] Found exponent e = %d", e)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=e,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.95,
                        details={"method": "modpow_cpu", "exponent": e},
                    )
        return self._fail("CPU modpow exhausted range", t0)

    # ------------------------------------------------------------------ #
    #  Collision / Birthday attack                                        #
    # ------------------------------------------------------------------ #

    def _collision_attack(
        self, instance: CryptoInstance, modulus: int,
        gpu: bool, timeout: float, t0: float,
    ) -> SolverResult:
        """Birthday attack: find x, y such that f(x) = g(y).

        For discrete log: g^x = h * g^{-y} ⟹ collision reveals x+y.
        Implemented as a GPU-accelerated hash table join.
        """
        log.info("[BF/Collision] Birthday attack (mod %d)", modulus)
        pub = instance.pub_as_int_list()
        if len(pub) < 2:
            return self._fail("Need at least (g, h) for collision attack", t0)

        g, h = pub[0] % modulus, pub[1] % modulus
        if g < 2:
            return self._fail("Generator too small", t0)

        # Birthday bound: O(√p) space and time
        bound = min(int(math.isqrt(modulus)) + 1, 2**24)

        if gpu and self._check_gpu():
            result = self._collision_gpu(g, h, modulus, bound, t0)
            if result is not None:
                return result

        # CPU fallback: baby-step giant-step style
        log.info("[BF/Collision/CPU] Building table of size %d", bound)
        table: dict[int, int] = {}
        val = 1
        for x in range(bound):
            if time.perf_counter() - t0 > timeout:
                break
            table[val] = x
            val = val * g % modulus

        # Search: h * g^{-y} for y = 0, 1, ...
        g_inv_bound = pow(g, modulus - 1 - bound, modulus) if modulus > 2 else 1
        gamma = h
        for y in range(bound):
            if time.perf_counter() - t0 > timeout:
                break
            if gamma in table:
                x = table[gamma]
                secret = (x + y * bound) % (modulus - 1)
                if pow(g, secret, modulus) == h:
                    log.info("[BF/Collision] Found secret = %d", secret)
                    return SolverResult(
                        solver_name=self.NAME,
                        status=SolverStatus.SUCCESS,
                        private_key=secret,
                        elapsed_seconds=time.perf_counter() - t0,
                        confidence=0.90,
                        details={"method": "collision_birthday", "secret": secret},
                    )
            gamma = gamma * g_inv_bound % modulus

        return self._fail("Collision attack did not find match", t0)

    def _collision_gpu(
        self, g: int, h: int, modulus: int, bound: int, t0: float
    ) -> SolverResult | None:
        """GPU-accelerated collision search using CuPy."""
        try:
            import cupy as cp

            log.info("[BF/Collision/GPU] Building GPU table of size %d", bound)

            # Baby steps: g^x for x in [0, bound)
            indices = cp.arange(bound, dtype=cp.int64)
            # Compute g^x mod p using sequential scan (prefix product)
            baby_steps = cp.ones(bound, dtype=cp.int64)
            g_val = int(g)
            mod = int(modulus)
            # Use CPU for sequential prefix product, then upload
            baby_np = np.ones(bound, dtype=np.int64)
            val = 1
            for i in range(bound):
                baby_np[i] = val
                val = val * g_val % mod
            baby_gpu = cp.array(baby_np)

            # Giant steps: h * g^{-bound*y} for y in [0, bound)
            g_inv_bound = pow(g_val, mod - 1 - bound, mod)
            giant_np = np.ones(bound, dtype=np.int64)
            val = int(h)
            for i in range(bound):
                giant_np[i] = val
                val = val * g_inv_bound % mod
            giant_gpu = cp.array(giant_np)

            # Sort baby steps and search
            sort_idx = cp.argsort(baby_gpu)
            baby_sorted = baby_gpu[sort_idx]

            for y in range(bound):
                target = int(giant_np[y])
                # Binary search in sorted array
                pos = cp.searchsorted(baby_sorted, target)
                if pos < bound and int(baby_sorted[pos]) == target:
                    x = int(sort_idx[pos])
                    secret = (x + y * bound) % (mod - 1)
                    if pow(g_val, secret, mod) == int(h):
                        log.info("[BF/Collision/GPU] Found secret = %d", secret)
                        return SolverResult(
                            solver_name=self.NAME,
                            status=SolverStatus.SUCCESS,
                            private_key=secret,
                            elapsed_seconds=time.perf_counter() - t0,
                            confidence=0.90,
                            details={"method": "collision_gpu", "secret": secret},
                        )
        except ImportError:
            log.debug("CuPy not available for collision GPU")
        except Exception as exc:
            log.warning("GPU collision failed: %s", exc)
        return None

    # ------------------------------------------------------------------ #
    #  Linear key-space exhaustion                                        #
    # ------------------------------------------------------------------ #

    def _bruteforce_linear(
        self, instance: CryptoInstance, modulus: int,
        gpu: bool, timeout: float, t0: float,
    ) -> SolverResult:
        """Exhaustive search over c = a*p + b mod m."""
        log.info("[BF/Linear] Exhaustive search (mod %d)", modulus)
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        if not pt or not ct:
            return self._fail("No pairs", t0)

        p0, c0 = pt[0] % modulus, ct[0] % modulus

        for a in range(modulus):
            if time.perf_counter() - t0 > timeout:
                break
            b = (c0 - a * p0) % modulus
            # Verify with second pair if available
            if len(pt) > 1 and len(ct) > 1:
                if (a * pt[1] + b) % modulus != ct[1] % modulus:
                    continue
            # Decrypt target
            a_inv = self._modinv(a, modulus)
            if a_inv is None:
                continue
            decrypted = [(a_inv * (c - b)) % modulus for c in ct_target]
            # Verify
            if all((a * d + b) % modulus == c % modulus for d, c in zip(decrypted, ct_target)):
                log.info("[BF/Linear] Found a=%d, b=%d", a, b)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=[a_inv, (-a_inv * b) % modulus],
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.95,
                    details={"method": "bruteforce_linear", "a": a, "b": b},
                )

        return self._fail("Linear brute-force exhausted", t0)

    # ------------------------------------------------------------------ #
    #  CPU generic brute-force                                            #
    # ------------------------------------------------------------------ #

    def _bruteforce_cpu_generic(
        self, instance: CryptoInstance, modulus: int | None,
        timeout: float, t0: float,
    ) -> SolverResult:
        """Generic CPU brute-force: try each possible key byte as XOR/add key."""
        log.info("[BF/Generic] Trying XOR and additive keys")
        pt = instance.pt_as_int_list()
        ct = instance.ct_known_as_int_list()
        ct_target = instance.ct_target_as_int_list()
        n = min(len(pt), len(ct))
        if n < 1:
            return self._fail("No pairs", t0)

        # XOR key search (byte-level)
        for key in range(256):
            if all((pt[i] ^ key) & 0xFF == ct[i] & 0xFF for i in range(min(n, 5))):
                decrypted = [(c ^ key) & 0xFF for c in ct_target]
                log.info("[BF/Generic] Found XOR key = 0x%02x", key)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.90,
                    details={"method": "xor_bruteforce", "key": key},
                )

        # Additive key search (byte-level)
        for key in range(256):
            if all((pt[i] + key) & 0xFF == ct[i] & 0xFF for i in range(min(n, 5))):
                decrypted = [(c - key) & 0xFF for c in ct_target]
                log.info("[BF/Generic] Found additive key = %d", key)
                return SolverResult(
                    solver_name=self.NAME,
                    status=SolverStatus.SUCCESS,
                    private_key=key,
                    decrypted=decrypted,
                    elapsed_seconds=time.perf_counter() - t0,
                    confidence=0.90,
                    details={"method": "additive_bruteforce", "key": key},
                )

        return self._fail("Generic brute-force exhausted", t0)

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    @staticmethod
    def _modinv(a: int, m: int) -> int | None:
        if m <= 0 or a == 0:
            return None
        g, x = a, 0
        g, x, _ = BruteForceGPUSolver._egcd(a % m, m)
        if g != 1:
            return None
        return x % m

    @staticmethod
    def _egcd(a: int, b: int) -> tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = BruteForceGPUSolver._egcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    def _fail(self, reason: str, t0: float) -> SolverResult:
        log.debug("[BF] %s", reason)
        return SolverResult(
            solver_name=self.NAME,
            status=SolverStatus.FAILED,
            elapsed_seconds=time.perf_counter() - t0,
            details={"reason": reason},
        )
