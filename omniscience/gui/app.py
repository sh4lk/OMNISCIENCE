"""OMNISCIENCE GUI — CustomTkinter-based graphical interface.

Features:
  - Input fields for public key, plaintext, ciphertext, target
  - Real-time CPU/GPU/RAM gauges
  - Live log console
  - Solver probability bars
  - Heatmap visualization (dependency matrix)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

from omniscience.core.config import OmniscienceConfig, HardwareConfig
from omniscience.core.types import AttackReport, CryptoInstance, ReconResult, SolverResult, SolverStatus
from omniscience.dispatcher import Dispatcher
from omniscience.hardware.resource_manager import ResourceManager, ResourceSnapshot

log = logging.getLogger(__name__)


class QueueHandler(logging.Handler):
    """Send log records to a thread-safe queue for the GUI to consume."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass


class OmniscienceGUI:
    """Main GUI application."""

    def __init__(self):
        import customtkinter as ctk

        self.ctk = ctk
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("OMNISCIENCE — Cryptanalysis Framework")
        self.root.geometry("1280x900")
        self.root.minsize(1024, 700)

        self.log_queue: queue.Queue[str] = queue.Queue(maxsize=5000)
        self.config = OmniscienceConfig()
        self.resource_mgr = ResourceManager(self.config.hardware)
        self._attack_thread: threading.Thread | None = None

        self._build_ui()
        self._setup_logging()
        self.resource_mgr.on_update(self._on_resource_update)
        self.resource_mgr.start_monitoring(interval=1.0)
        self._poll_logs()

    # ------------------------------------------------------------------ #
    #  UI Construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        ctk = self.ctk

        # Main layout: left panel (inputs + controls) | right panel (monitors + log)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self.root)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right = ctk.CTkFrame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # --- Left Panel: Inputs ---
        ctk.CTkLabel(left, text="OMNISCIENCE", font=("Consolas", 24, "bold")).pack(pady=10)

        input_frame = ctk.CTkFrame(left)
        input_frame.pack(fill="x", padx=10, pady=5)

        self.entries: dict[str, Any] = {}
        for label, key in [
            ("Public Key", "pub_key"),
            ("Plaintext (known)", "plaintext"),
            ("Ciphertext (known)", "ciphertext"),
            ("Target Ciphertext", "target"),
            ("Modulus (optional)", "modulus"),
        ]:
            row = ctk.CTkFrame(input_frame)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, placeholder_text=f"Enter {label.lower()}...")
            entry.pack(side="left", fill="x", expand=True, padx=5)
            self.entries[key] = entry

        # Format selector
        fmt_row = ctk.CTkFrame(input_frame)
        fmt_row.pack(fill="x", pady=2)
        ctk.CTkLabel(fmt_row, text="Input Format", width=160, anchor="w").pack(side="left", padx=5)
        self.format_var = ctk.StringVar(value="int")
        ctk.CTkOptionMenu(fmt_row, variable=self.format_var, values=["int", "hex", "base64", "json"]).pack(
            side="left", padx=5
        )

        # Options
        opt_frame = ctk.CTkFrame(left)
        opt_frame.pack(fill="x", padx=10, pady=5)
        self.gpu_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opt_frame, text="Use GPU", variable=self.gpu_var).pack(side="left", padx=10)
        self.parallel_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opt_frame, text="Parallel Solvers", variable=self.parallel_var).pack(side="left", padx=10)

        # Attack button
        self.attack_btn = ctk.CTkButton(
            left, text="LAUNCH ATTACK", font=("Consolas", 16, "bold"),
            fg_color="#e74c3c", hover_color="#c0392b", height=50,
            command=self._on_attack,
        )
        self.attack_btn.pack(fill="x", padx=10, pady=10)

        # Solver probability bars
        ctk.CTkLabel(left, text="Solver Confidence", font=("Consolas", 14)).pack(pady=(10, 2))
        self.solver_bars: dict[str, Any] = {}
        self.solver_labels: dict[str, Any] = {}
        for name in [
            "algebraic", "lattice", "smt_z3", "neural", "factorization",
            "dlog", "elliptic_curve", "bruteforce", "mitm", "classical",
            "cross_cipher", "symmetric", "ecdh", "hybrid_scheme",
        ]:
            bar_frame = ctk.CTkFrame(left)
            bar_frame.pack(fill="x", padx=10, pady=1)
            lbl = ctk.CTkLabel(bar_frame, text=f"{name:12s}", width=100, anchor="w", font=("Consolas", 11))
            lbl.pack(side="left", padx=5)
            bar = ctk.CTkProgressBar(bar_frame, width=200)
            bar.pack(side="left", fill="x", expand=True, padx=5)
            bar.set(0)
            status_lbl = ctk.CTkLabel(bar_frame, text="—", width=80, font=("Consolas", 11))
            status_lbl.pack(side="left", padx=5)
            self.solver_bars[name] = bar
            self.solver_labels[name] = status_lbl

        # Result display
        self.result_text = ctk.CTkTextbox(left, height=120, font=("Consolas", 11))
        self.result_text.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Right Panel: Monitors + Log ---
        ctk.CTkLabel(right, text="System Monitor", font=("Consolas", 16)).pack(pady=5)

        # Resource gauges
        gauges = ctk.CTkFrame(right)
        gauges.pack(fill="x", padx=10, pady=5)

        self.gauge_bars: dict[str, Any] = {}
        self.gauge_labels: dict[str, Any] = {}
        for name, color in [("CPU", "#3498db"), ("RAM", "#2ecc71"), ("GPU", "#e67e22")]:
            gf = ctk.CTkFrame(gauges)
            gf.pack(fill="x", pady=2)
            ctk.CTkLabel(gf, text=name, width=50, anchor="w", font=("Consolas", 12)).pack(side="left", padx=5)
            bar = ctk.CTkProgressBar(gf, width=200, progress_color=color)
            bar.pack(side="left", fill="x", expand=True, padx=5)
            bar.set(0)
            lbl = ctk.CTkLabel(gf, text="0%", width=60, font=("Consolas", 11))
            lbl.pack(side="left", padx=5)
            self.gauge_bars[name] = bar
            self.gauge_labels[name] = lbl

        # Log console
        ctk.CTkLabel(right, text="Live Log", font=("Consolas", 14)).pack(pady=(10, 2))
        self.log_text = ctk.CTkTextbox(right, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def _setup_logging(self) -> None:
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger("omniscience")
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)

    def _poll_logs(self) -> None:
        """Drain the log queue into the GUI text widget."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    # ------------------------------------------------------------------ #
    #  Resource monitoring callback                                       #
    # ------------------------------------------------------------------ #

    def _on_resource_update(self, snap: ResourceSnapshot) -> None:
        """Called from the monitor thread — schedule GUI update on main thread."""

        def _update():
            self.gauge_bars["CPU"].set(snap.cpu_percent / 100.0)
            self.gauge_labels["CPU"].configure(text=f"{snap.cpu_percent:.0f}%")
            self.gauge_bars["RAM"].set(snap.ram_percent / 100.0)
            self.gauge_labels["RAM"].configure(text=f"{snap.ram_percent:.0f}%")
            if snap.gpu_available and snap.gpu_memory_total_mb > 0:
                gpu_pct = snap.gpu_memory_used_mb / snap.gpu_memory_total_mb
                self.gauge_bars["GPU"].set(gpu_pct)
                self.gauge_labels["GPU"].configure(text=f"{gpu_pct * 100:.0f}%")
            else:
                self.gauge_bars["GPU"].set(0)
                self.gauge_labels["GPU"].configure(text="N/A")

        try:
            self.root.after(0, _update)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Attack execution                                                   #
    # ------------------------------------------------------------------ #

    def _on_attack(self) -> None:
        if self._attack_thread and self._attack_thread.is_alive():
            return

        self.attack_btn.configure(state="disabled", text="ATTACKING...")
        self.result_text.delete("1.0", "end")
        for bar in self.solver_bars.values():
            bar.set(0)
        for lbl in self.solver_labels.values():
            lbl.configure(text="—")

        self._attack_thread = threading.Thread(target=self._run_attack, daemon=True)
        self._attack_thread.start()

    def _run_attack(self) -> None:
        try:
            instance = self._parse_inputs()
            if instance is None:
                self.root.after(0, lambda: self.attack_btn.configure(state="normal", text="LAUNCH ATTACK"))
                return

            config = OmniscienceConfig(
                hardware=HardwareConfig(use_gpu=self.gpu_var.get()),
                parallel_solvers=self.parallel_var.get(),
            )
            dispatcher = Dispatcher(config)

            def on_solver_done(result: SolverResult):
                def _update():
                    name = result.solver_name
                    if name in self.solver_bars:
                        self.solver_bars[name].set(result.confidence)
                        status = "OK" if result.status == SolverStatus.SUCCESS else "FAIL"
                        color = "#2ecc71" if result.status == SolverStatus.SUCCESS else "#e74c3c"
                        self.solver_labels[name].configure(text=f"{status} {result.confidence * 100:.0f}%")

                self.root.after(0, _update)

            dispatcher.on_solver_done(on_solver_done)
            report = dispatcher.attack(instance)

            def _show_result():
                self.attack_btn.configure(state="normal", text="LAUNCH ATTACK")
                if report.success():
                    self.result_text.insert("end", f"SUCCESS — {report.best_result.solver_name}\n")
                    self.result_text.insert("end", f"Confidence: {report.best_result.confidence * 100:.1f}%\n")
                    self.result_text.insert("end", f"Time: {report.total_elapsed:.2f}s\n")
                    self.result_text.insert("end", f"Decrypted: {report.best_result.decrypted}\n")
                else:
                    self.result_text.insert("end", "ALL SOLVERS FAILED\n")
                    self.result_text.insert("end", f"Time: {report.total_elapsed:.2f}s\n")

            self.root.after(0, _show_result)

        except Exception as exc:
            log.exception("Attack thread error")
            self.root.after(
                0,
                lambda: [
                    self.attack_btn.configure(state="normal", text="LAUNCH ATTACK"),
                    self.result_text.insert("end", f"ERROR: {exc}\n"),
                ],
            )

    def _parse_inputs(self) -> CryptoInstance | None:
        try:
            fmt = self.format_var.get()

            def parse(raw: str):
                raw = raw.strip()
                if not raw:
                    return []
                if fmt == "hex":
                    return list(bytes.fromhex(raw))
                elif fmt == "json":
                    return json.loads(raw)
                elif fmt == "base64":
                    import base64
                    return list(base64.b64decode(raw))
                else:
                    return [int(x.strip()) for x in raw.split(",")]

            pub = parse(self.entries["pub_key"].get())
            pt = parse(self.entries["plaintext"].get())
            ct = parse(self.entries["ciphertext"].get())
            target = parse(self.entries["target"].get())
            mod_str = self.entries["modulus"].get().strip()
            modulus = int(mod_str) if mod_str else None

            return CryptoInstance(
                public_key=pub,
                plaintext=pt,
                ciphertext_known=ct,
                ciphertext_target=target,
                modulus=modulus,
            )
        except Exception as exc:
            self.root.after(0, lambda: self.result_text.insert("end", f"Input error: {exc}\n"))
            return None

    # ------------------------------------------------------------------ #
    #  Run                                                                #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.root.mainloop()


def launch() -> None:
    """Entry point for the GUI."""
    gui = OmniscienceGUI()
    gui.run()


if __name__ == "__main__":
    launch()
