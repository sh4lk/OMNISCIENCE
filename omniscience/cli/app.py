"""OMNISCIENCE CLI — Typer-based command-line interface.

Provides verbose, detail-rich output including intermediate mathematical
steps, resource utilization, and solver progress.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from omniscience.core.config import OmniscienceConfig, HardwareConfig, NeuralConfig, SolverTimeouts
from omniscience.core.types import AttackReport, CryptoInstance, SolverStatus
from omniscience.dispatcher import Dispatcher

app = typer.Typer(
    name="omniscience",
    help="OMNISCIENCE — Black-box asymmetric cryptanalysis framework",
    rich_markup_mode="rich",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _load_instance(
    pub_key: Optional[str],
    plaintext: str,
    ciphertext: str,
    target: str,
    modulus: Optional[int],
    input_format: str,
) -> CryptoInstance:
    """Parse les entrees depuis les arguments CLI."""

    def parse_data(raw: str, fmt: str):
        if fmt == "hex":
            return bytes.fromhex(raw)
        elif fmt == "base64":
            import base64
            return base64.b64decode(raw)
        elif fmt == "json":
            return json.loads(raw)
        elif fmt == "file":
            p = Path(raw)
            if not p.exists():
                console.print(f"[red]Fichier introuvable : {raw}[/red]")
                raise typer.Exit(1)
            content = p.read_text().strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return bytes.fromhex(content)
        else:  # "int" — entiers separes par des virgules
            return [int(x.strip()) for x in raw.split(",")]

    return CryptoInstance(
        public_key=parse_data(pub_key, input_format) if pub_key else None,
        plaintext=parse_data(plaintext, input_format),
        ciphertext_known=parse_data(ciphertext, input_format),
        ciphertext_target=parse_data(target, input_format),
        modulus=modulus,
    )


def _print_report(report: AttackReport) -> None:
    """Render the attack report as rich tables."""
    # Recon summary
    if report.recon:
        r = report.recon
        table = Table(title="Reconnaissance Results", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Algorithm Family", r.algo_family.value)
        table.add_row("Confidence", f"{r.confidence * 100:.1f}%")
        table.add_row("Linearity Score", f"{r.linearity_score:.4f}")
        table.add_row("Entropy (PT)", f"{r.entropy_plaintext:.4f}")
        table.add_row("Entropy (CT)", f"{r.entropy_ciphertext:.4f}")
        table.add_row("Polynomial Degree", str(r.polynomial_degree_estimate or "N/A"))
        table.add_row("Substitution Detected", str(r.substitution_detected))
        table.add_row("Lattice Structure", str(r.lattice_structure_detected))
        table.add_row("Estimated Modulus", str(r.estimated_modulus or "N/A"))
        console.print(table)

    # Solver results
    table = Table(title="Solver Results", show_header=True)
    table.add_column("Solver", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Confidence", style="green")
    table.add_column("Time", style="yellow")
    table.add_column("Method")

    for sr in report.solver_results:
        status_style = "green" if sr.status == SolverStatus.SUCCESS else "red"
        table.add_row(
            sr.solver_name,
            f"[{status_style}]{sr.status.value}[/{status_style}]",
            f"{sr.confidence * 100:.1f}%",
            f"{sr.elapsed_seconds:.2f}s",
            sr.details.get("method", "—"),
        )
    console.print(table)

    # Final result
    if report.success():
        console.print(
            Panel(
                f"[bold green]DECRYPTION SUCCESSFUL[/bold green]\n\n"
                f"Solver: {report.best_result.solver_name}\n"
                f"Confidence: {report.best_result.confidence * 100:.1f}%\n"
                f"Time: {report.total_elapsed:.2f}s\n\n"
                f"Decrypted: {report.best_result.decrypted}",
                title="Result",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]ALL SOLVERS FAILED[/bold red]\n\n"
                "Consider providing more plaintext/ciphertext pairs,\n"
                "or adjusting solver timeouts.",
                title="Result",
                border_style="red",
            )
        )


@app.command()
def attack(
    pub_key: Optional[str] = typer.Option(None, "--pub", "-p", help="Cle publique (optionnel, format selon --format)"),
    plaintext: str = typer.Option(..., "--pt", help="Texte clair connu"),
    ciphertext: str = typer.Option(..., "--ct", help="Texte chiffre connu"),
    target: str = typer.Option(..., "--target", "-t", help="Texte chiffre a dechiffrer"),
    modulus: Optional[int] = typer.Option(None, "--mod", "-m", help="Known modulus (if any)"),
    input_format: str = typer.Option("int", "--format", "-f", help="Input format: int, hex, base64, json, file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable GPU acceleration"),
    sequential: bool = typer.Option(False, "--sequential", "-s", help="Run solvers sequentially"),
    timeout_algebraic: float = typer.Option(300.0, help="Algebraic solver timeout (seconds)"),
    timeout_lattice: float = typer.Option(600.0, help="Lattice solver timeout (seconds)"),
    timeout_smt: float = typer.Option(600.0, help="SMT solver timeout (seconds)"),
    timeout_neural: float = typer.Option(3600.0, help="Neural solver timeout (seconds)"),
    timeout_bruteforce: float = typer.Option(1800.0, help="Brute-force solver timeout (seconds)"),
    export_json: Optional[str] = typer.Option(None, "--export-json", help="Export report to JSON file"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="Export report to HTML file"),
) -> None:
    """Launch a full cryptanalysis attack on a black-box cipher."""
    _setup_logging(verbose)

    console.print(
        Panel(
            "[bold cyan]OMNISCIENCE[/bold cyan] — Black-Box Asymmetric Cryptanalysis\n"
            "[dim]17 solver engines • GPU acceleration • Ray parallelism[/dim]",
            border_style="cyan",
        )
    )

    config = OmniscienceConfig(
        hardware=HardwareConfig(use_gpu=not no_gpu),
        timeouts=SolverTimeouts(
            algebraic=timeout_algebraic,
            lattice=timeout_lattice,
            smt=timeout_smt,
            neural=timeout_neural,
            bruteforce=timeout_bruteforce,
        ),
        verbose=verbose,
        parallel_solvers=not sequential,
    )

    instance = _load_instance(pub_key, plaintext, ciphertext, target, modulus, input_format)
    dispatcher = Dispatcher(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running attack...", total=None)
        report = dispatcher.attack(instance)
        progress.update(task, completed=True, description="Attack complete")

    _print_report(report)

    # Export reports
    from omniscience.core.report import ReportExporter
    if export_json:
        ReportExporter.save_json(report, export_json)
        console.print(f"[green]Report saved to {export_json}[/green]")
    if export_html:
        ReportExporter.save_html(report, export_html)
        console.print(f"[green]HTML report saved to {export_html}[/green]")


@app.command()
def info() -> None:
    """Display system information and available resources."""
    from omniscience.hardware.resource_manager import ResourceManager

    _setup_logging(False)
    rm = ResourceManager()
    console.print(Panel(rm.summary(), title="System Resources", border_style="cyan"))


@app.command()
def recon(
    pub_key: Optional[str] = typer.Option(None, "--pub", "-p"),
    plaintext: str = typer.Option(..., "--pt"),
    ciphertext: str = typer.Option(..., "--ct"),
    target: str = typer.Option("", "--target", "-t"),
    modulus: Optional[int] = typer.Option(None, "--mod", "-m"),
    input_format: str = typer.Option("int", "--format", "-f"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run only the reconnaissance phase (no solving)."""
    _setup_logging(verbose)
    from omniscience.recon.statistical import StatisticalRecon
    instance = _load_instance(pub_key, plaintext, ciphertext, target or "0", modulus, input_format)
    recon_engine = StatisticalRecon()
    result = recon_engine.analyze(instance)

    table = Table(title="Reconnaissance Only", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for field_name in [
        "algo_family", "confidence", "linearity_score", "entropy_plaintext",
        "entropy_ciphertext", "polynomial_degree_estimate", "substitution_detected",
        "lattice_structure_detected", "estimated_modulus",
    ]:
        val = getattr(result, field_name)
        if hasattr(val, "value"):
            val = val.value
        table.add_row(field_name, str(val))
    console.print(table)


if __name__ == "__main__":
    app()
