"""Global configuration for OMNISCIENCE."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardwareConfig:
    use_gpu: bool = True
    gpu_device_id: int = 0
    max_cpu_workers: int = 0  # 0 = auto-detect (all cores)
    max_ram_gb: float = 0.0  # 0 = no limit
    gpu_memory_fraction: float = 0.95


@dataclass
class SolverTimeouts:
    """Per-solver timeout in seconds."""

    recon: float = 60.0
    algebraic: float = 300.0
    lattice: float = 600.0
    smt: float = 600.0
    neural: float = 3600.0
    bruteforce: float = 1800.0


@dataclass
class NeuralConfig:
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8  # for transformer variant
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_epochs: int = 500
    early_stop_patience: int = 20
    use_transformer: bool = True  # False → MLP


@dataclass
class OmniscienceConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    timeouts: SolverTimeouts = field(default_factory=SolverTimeouts)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    verbose: bool = True
    log_intermediates: bool = True
    max_known_pairs: int = 100_000  # max plaintext/ciphertext pairs for neural training
    parallel_solvers: bool = True  # run solvers concurrently
