"""Intelligent Dispatcher / Orchestrator.

The dispatcher is the brain of OMNISCIENCE. It:
  1. Runs statistical reconnaissance on the input data.
  2. Based on the detected algorithm family, prioritizes solvers.
  3. Launches solvers in parallel (or sequentially) respecting timeouts.
  4. Aggregates results and selects the best solution.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

from omniscience.core.config import OmniscienceConfig
from omniscience.core.types import (
    AlgoFamily,
    AttackReport,
    CryptoInstance,
    ReconResult,
    SolverResult,
    SolverStatus,
)
from omniscience.hardware.resource_manager import ResourceManager
from omniscience.recon.statistical import StatisticalRecon
from omniscience.solvers.algebraic import AlgebraicSolver
from omniscience.solvers.lattice import LatticeSolver
from omniscience.solvers.lattice_advanced import AdvancedLatticeSolver
from omniscience.solvers.neural import NeuralCryptanalysisSolver
from omniscience.solvers.smt import SMTSolver
from omniscience.solvers.agcd import AGCDSolver
from omniscience.solvers.factorization import FactorizationSolver
from omniscience.solvers.dlog import DLogSolver
from omniscience.solvers.elliptic_curve import EllipticCurveSolver
from omniscience.solvers.bruteforce_gpu import BruteForceGPUSolver
from omniscience.solvers.mitm import MITMSolver
from omniscience.solvers.oracle import OracleAttackSolver
from omniscience.solvers.classical import ClassicalCipherSolver
from omniscience.solvers.cross_cipher import CrossCipherSolver

log = logging.getLogger(__name__)


# Priority mapping: for each detected family, ordered list of solvers to try
SOLVER_PRIORITY: dict[AlgoFamily, list[str]] = {
    AlgoFamily.LINEAR: ["algebraic", "classical", "smt_z3", "mitm", "cross_cipher", "bruteforce", "lattice", "neural"],
    AlgoFamily.POLYNOMIAL: ["algebraic", "smt_z3", "mitm", "lattice", "bruteforce", "neural"],
    AlgoFamily.SUBSTITUTION: ["classical", "smt_z3", "algebraic", "cross_cipher", "mitm", "bruteforce", "neural", "lattice"],
    AlgoFamily.LATTICE_BASED: ["lattice", "lattice_advanced", "smt_z3", "algebraic", "bruteforce", "neural"],
    AlgoFamily.KNAPSACK: ["lattice", "algebraic", "smt_z3", "bruteforce", "neural"],
    AlgoFamily.RSA_LIKE: ["factorization", "algebraic", "lattice_advanced", "oracle", "smt_z3", "bruteforce", "neural"],
    AlgoFamily.EC_LIKE: ["elliptic_curve", "dlog", "lattice", "algebraic", "smt_z3", "bruteforce", "neural"],
    AlgoFamily.LWE_BASED: ["lattice", "lattice_advanced", "smt_z3", "neural", "algebraic", "bruteforce"],
    AlgoFamily.AGCD: ["agcd", "lattice", "factorization", "algebraic", "bruteforce", "neural"],
    AlgoFamily.DLOG: ["dlog", "factorization", "lattice", "bruteforce", "algebraic", "smt_z3", "neural"],
    AlgoFamily.NTRU_LIKE: ["lattice_advanced", "lattice", "smt_z3", "algebraic", "bruteforce", "neural"],
    AlgoFamily.HYBRID: [
        "algebraic", "classical", "cross_cipher", "factorization", "dlog", "mitm",
        "lattice", "lattice_advanced", "smt_z3", "oracle", "bruteforce", "neural",
    ],
    AlgoFamily.UNKNOWN: [
        "classical", "cross_cipher", "algebraic", "smt_z3", "factorization", "dlog",
        "mitm", "lattice", "lattice_advanced", "elliptic_curve", "agcd", "oracle",
        "bruteforce", "neural",
    ],
}


class Dispatcher:
    """Orchestrates the full attack pipeline."""

    def __init__(self, config: OmniscienceConfig | None = None):
        self.config = config or OmniscienceConfig()
        self.resource_mgr = ResourceManager(self.config.hardware)
        self.recon = StatisticalRecon()

        # Solver registry — all 14 engines
        self._solvers: dict[str, Any] = {
            "algebraic": AlgebraicSolver(),
            "lattice": LatticeSolver(),
            "lattice_advanced": AdvancedLatticeSolver(),
            "smt_z3": SMTSolver(),
            "neural": NeuralCryptanalysisSolver(self.config.neural),
            "agcd": AGCDSolver(),
            "factorization": FactorizationSolver(),
            "dlog": DLogSolver(),
            "elliptic_curve": EllipticCurveSolver(),
            "bruteforce": BruteForceGPUSolver(),
            "mitm": MITMSolver(),
            "oracle": OracleAttackSolver(),
            "classical": ClassicalCipherSolver(),
            "cross_cipher": CrossCipherSolver(),
        }

        self._timeout_map: dict[str, float] = {
            "algebraic": self.config.timeouts.algebraic,
            "lattice": self.config.timeouts.lattice,
            "lattice_advanced": self.config.timeouts.lattice,
            "smt_z3": self.config.timeouts.smt,
            "neural": self.config.timeouts.neural,
            "agcd": self.config.timeouts.algebraic,
            "factorization": self.config.timeouts.algebraic,
            "dlog": self.config.timeouts.algebraic,
            "elliptic_curve": self.config.timeouts.algebraic,
            "bruteforce": self.config.timeouts.bruteforce,
            "mitm": self.config.timeouts.algebraic,
            "oracle": self.config.timeouts.algebraic,
            "classical": self.config.timeouts.algebraic,
            "cross_cipher": self.config.timeouts.algebraic,
        }

        # Callbacks for live status updates
        self._on_recon_done: list[Any] = []
        self._on_solver_done: list[Any] = []
        self._on_complete: list[Any] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def attack(self, instance: CryptoInstance) -> AttackReport:
        """Execute the full attack pipeline on a cryptographic instance."""
        report = AttackReport(instance=instance)
        t_global = time.perf_counter()

        # Configure hardware
        self.resource_mgr.configure_gpu()
        self.resource_mgr.start_monitoring(interval=2.0)

        log.info("=" * 60)
        log.info("OMNISCIENCE — Attack started")
        log.info("=" * 60)
        log.info("Hardware: %s", self.resource_mgr.summary())

        # Phase 1: Reconnaissance
        log.info("-" * 40)
        log.info("Phase 1: Statistical Reconnaissance")
        log.info("-" * 40)
        recon = self.recon.analyze(instance)
        report.recon = recon
        self._notify(self._on_recon_done, recon)

        log.info(
            "Recon result: family=%s, confidence=%.1f%%, linearity=%.3f",
            recon.algo_family.value,
            recon.confidence * 100,
            recon.linearity_score,
        )

        # Phase 2: Solver dispatch
        log.info("-" * 40)
        log.info("Phase 2: Solver Dispatch")
        log.info("-" * 40)
        solver_order = SOLVER_PRIORITY.get(recon.algo_family, SOLVER_PRIORITY[AlgoFamily.UNKNOWN])
        log.info("Solver priority: %s", solver_order)

        if self.config.parallel_solvers:
            results = self._run_parallel_ray_or_threads(instance, recon, solver_order)
        else:
            results = self._run_sequential(instance, recon, solver_order)

        report.solver_results = results

        # Phase 3: Select best result
        successes = [r for r in results if r.status == SolverStatus.SUCCESS]
        if successes:
            report.best_result = max(successes, key=lambda r: r.confidence)
            log.info(
                "*** SUCCESS *** Best solver: %s (confidence %.1f%%, %.2fs)",
                report.best_result.solver_name,
                report.best_result.confidence * 100,
                report.best_result.elapsed_seconds,
            )
        else:
            log.warning("All solvers failed.")

        report.total_elapsed = time.perf_counter() - t_global
        self.resource_mgr.stop_monitoring()
        self._notify(self._on_complete, report)

        log.info("=" * 60)
        log.info("Attack finished in %.2fs — %s", report.total_elapsed, "SUCCESS" if report.success() else "FAILED")
        log.info("=" * 60)

        return report

    # ------------------------------------------------------------------ #
    #  Callback registration                                              #
    # ------------------------------------------------------------------ #

    def on_recon_done(self, cb):
        self._on_recon_done.append(cb)

    def on_solver_done(self, cb):
        self._on_solver_done.append(cb)

    def on_complete(self, cb):
        self._on_complete.append(cb)

    # ------------------------------------------------------------------ #
    #  Execution strategies                                               #
    # ------------------------------------------------------------------ #

    def _run_parallel_ray_or_threads(
        self, instance: CryptoInstance, recon: ReconResult, solver_order: list[str]
    ) -> list[SolverResult]:
        """Try Ray first for true multi-process parallelism, fall back to threads."""
        try:
            return self._run_parallel_ray(instance, recon, solver_order)
        except Exception:
            log.debug("Ray not available, falling back to ThreadPoolExecutor")
            return self._run_parallel_threads(instance, recon, solver_order)

    def _run_parallel_ray(
        self, instance: CryptoInstance, recon: ReconResult, solver_order: list[str]
    ) -> list[SolverResult]:
        """Run solvers as Ray remote tasks across all CPU cores."""
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False,
                     num_cpus=self.resource_mgr.cpu_count)
            log.info("Ray initialized with %d CPUs", self.resource_mgr.cpu_count)

        @ray.remote
        def _ray_solve(solver, name, inst, rec, timeout):
            try:
                return solver.solve(inst, rec, timeout=timeout)
            except Exception as exc:
                return SolverResult(
                    solver_name=name, status=SolverStatus.FAILED,
                    details={"error": str(exc)},
                )

        results: list[SolverResult] = []
        refs = {}
        for name in solver_order:
            solver = self._solvers.get(name)
            if solver is None:
                continue
            timeout = self._timeout_map.get(name, 300.0)
            ref = _ray_solve.remote(solver, name, instance, recon, timeout)
            refs[ref] = name

        pending = list(refs.keys())
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                name = refs[ref]
                result = ray.get(ref)
                results.append(result)
                self._notify(self._on_solver_done, result)
                log.info("Solver [%s] finished: %s (%.2fs)",
                         name, result.status.value, result.elapsed_seconds)
                if result.status == SolverStatus.SUCCESS:
                    for p in pending:
                        ray.cancel(p, force=True)
                    pending = []
                    break

        return results

    def _run_parallel_threads(
        self, instance: CryptoInstance, recon: ReconResult, solver_order: list[str]
    ) -> list[SolverResult]:
        """Run all solvers concurrently via threads, stop-on-first-success."""
        results: list[SolverResult] = []
        max_workers = min(len(solver_order), self.resource_mgr.cpu_count)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="solver") as executor:
            futures = {}
            for name in solver_order:
                solver = self._solvers.get(name)
                if solver is None:
                    continue
                timeout = self._timeout_map.get(name, 300.0)
                future = executor.submit(self._run_single_solver, solver, name, instance, recon, timeout)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._notify(self._on_solver_done, result)
                    log.info(
                        "Solver [%s] finished: %s (%.2fs)",
                        name,
                        result.status.value,
                        result.elapsed_seconds,
                    )
                    if result.status == SolverStatus.SUCCESS:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                except Exception as exc:
                    log.error("Solver [%s] raised exception: %s", name, exc)
                    results.append(
                        SolverResult(solver_name=name, status=SolverStatus.FAILED, details={"error": str(exc)})
                    )

        return results

    def _run_sequential(
        self, instance: CryptoInstance, recon: ReconResult, solver_order: list[str]
    ) -> list[SolverResult]:
        """Run solvers one by one, stop on first success."""
        results: list[SolverResult] = []
        for name in solver_order:
            solver = self._solvers.get(name)
            if solver is None:
                continue
            timeout = self._timeout_map.get(name, 300.0)
            result = self._run_single_solver(solver, name, instance, recon, timeout)
            results.append(result)
            self._notify(self._on_solver_done, result)
            log.info(
                "Solver [%s] finished: %s (%.2fs)",
                name,
                result.status.value,
                result.elapsed_seconds,
            )
            if result.status == SolverStatus.SUCCESS:
                break
        return results

    @staticmethod
    def _run_single_solver(solver, name: str, instance, recon, timeout) -> SolverResult:
        try:
            return solver.solve(instance, recon, timeout=timeout)
        except Exception as exc:
            log.exception("Solver [%s] crashed", name)
            return SolverResult(
                solver_name=name,
                status=SolverStatus.FAILED,
                details={"error": str(exc)},
            )

    @staticmethod
    def _notify(callbacks, *args):
        for cb in callbacks:
            try:
                cb(*args)
            except Exception:
                log.exception("Callback error")
