"""Core data types for the OMNISCIENCE framework."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class AlgoFamily(enum.Enum):
    """Detected algorithm family after reconnaissance."""

    UNKNOWN = "unknown"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SUBSTITUTION = "substitution"
    LATTICE_BASED = "lattice_based"
    KNAPSACK = "knapsack"
    RSA_LIKE = "rsa_like"
    EC_LIKE = "elliptic_curve_like"
    LWE_BASED = "lwe_based"
    AGCD = "agcd"
    DLOG = "discrete_log"
    NTRU_LIKE = "ntru_like"
    SYMMETRIC_BLOCK = "symmetric_block"
    SYMMETRIC_STREAM = "symmetric_stream"
    ECDH = "ecdh"
    HYBRID = "hybrid"


class SolverStatus(enum.Enum):
    """Status of a solver execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CryptoInstance:
    """A single cryptanalysis problem instance.

    Attributes:
        public_key: The public key (arbitrary structure, stored as bytes or int list).
        plaintext: Known plaintext sample(s).
        ciphertext_known: Ciphertext corresponding to the known plaintext.
        ciphertext_target: The ciphertext to decrypt.
        modulus: Detected or provided modulus (if any).
        key_size_bits: Estimated key size in bits.
        extra: Arbitrary metadata from the user or recon phase.
    """

    public_key: bytes | list[int] | int | None = None
    plaintext: bytes | list[int] = field(default_factory=list)
    ciphertext_known: bytes | list[int] = field(default_factory=list)
    ciphertext_target: bytes | list[int] = field(default_factory=list)
    modulus: int | None = None
    key_size_bits: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def pub_as_int_list(self) -> list[int]:
        if self.public_key is None:
            return []
        if isinstance(self.public_key, int):
            return [self.public_key]
        if isinstance(self.public_key, bytes):
            return list(self.public_key)
        return list(self.public_key)

    def pt_as_int_list(self) -> list[int]:
        if isinstance(self.plaintext, bytes):
            return list(self.plaintext)
        return list(self.plaintext)

    def ct_known_as_int_list(self) -> list[int]:
        if isinstance(self.ciphertext_known, bytes):
            return list(self.ciphertext_known)
        return list(self.ciphertext_known)

    def ct_target_as_int_list(self) -> list[int]:
        if isinstance(self.ciphertext_target, bytes):
            return list(self.ciphertext_target)
        return list(self.ciphertext_target)


@dataclass
class ReconResult:
    """Results from the statistical reconnaissance module."""

    algo_family: AlgoFamily = AlgoFamily.UNKNOWN
    entropy_plaintext: float = 0.0
    entropy_ciphertext: float = 0.0
    bit_correlation_matrix: Any = None  # numpy array
    linearity_score: float = 0.0  # 0 = nonlinear, 1 = perfectly linear
    polynomial_degree_estimate: int | None = None
    substitution_detected: bool = False
    lattice_structure_detected: bool = False
    estimated_modulus: int | None = None
    confidence: float = 0.0
    heatmap_data: Any = None  # numpy array for dependency heatmap
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result returned by any solver module."""

    solver_name: str
    status: SolverStatus
    private_key: bytes | list[int] | int | None = None
    decrypted: bytes | list[int] | None = None
    elapsed_seconds: float = 0.0
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackReport:
    """Final report aggregating all attack attempts."""

    instance: CryptoInstance | None = None
    recon: ReconResult | None = None
    solver_results: list[SolverResult] = field(default_factory=list)
    best_result: SolverResult | None = None
    total_elapsed: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def success(self) -> bool:
        return self.best_result is not None and self.best_result.status == SolverStatus.SUCCESS
