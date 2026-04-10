"""Report Exporter — JSON, HTML, and plaintext.

Generates structured reports from AttackReport objects for archiving,
sharing, and post-analysis.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from omniscience.core.types import AttackReport, SolverResult, SolverStatus


class ReportExporter:
    """Export attack reports in multiple formats."""

    # ------------------------------------------------------------------ #
    #  JSON                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_json(report: AttackReport, pretty: bool = True) -> str:
        """Serialize report to JSON string."""
        data = ReportExporter._report_to_dict(report)
        return json.dumps(data, indent=2 if pretty else None, default=str)

    @staticmethod
    def save_json(report: AttackReport, path: str | Path) -> None:
        Path(path).write_text(ReportExporter.to_json(report))

    # ------------------------------------------------------------------ #
    #  HTML                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_html(report: AttackReport) -> str:
        """Generate a self-contained HTML report."""
        d = ReportExporter._report_to_dict(report)
        success = report.success()
        status_color = "#2ecc71" if success else "#e74c3c"
        status_text = "DECRYPTION SUCCESSFUL" if success else "ALL SOLVERS FAILED"

        solver_rows = ""
        for sr in d.get("solver_results", []):
            s_color = "#2ecc71" if sr["status"] == "success" else "#e74c3c"
            solver_rows += f"""
            <tr>
                <td>{html.escape(sr['solver_name'])}</td>
                <td style="color:{s_color};font-weight:bold">{sr['status']}</td>
                <td>{sr['confidence']:.1%}</td>
                <td>{sr['elapsed_seconds']:.2f}s</td>
                <td>{html.escape(str(sr.get('details', {}).get('method', '—')))}</td>
            </tr>"""

        recon_rows = ""
        recon = d.get("recon", {})
        for key in [
            "algo_family", "confidence", "linearity_score",
            "entropy_plaintext", "entropy_ciphertext",
            "polynomial_degree_estimate", "substitution_detected",
            "lattice_structure_detected", "estimated_modulus",
        ]:
            val = recon.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            recon_rows += f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(val))}</td></tr>"

        best_info = ""
        if success and d.get("best_result"):
            br = d["best_result"]
            decrypted_str = str(br.get("decrypted", "N/A"))
            if len(decrypted_str) > 500:
                decrypted_str = decrypted_str[:500] + "..."
            best_info = f"""
            <div class="result-box" style="border-color:{status_color}">
                <h2 style="color:{status_color}">{status_text}</h2>
                <p><strong>Solver:</strong> {html.escape(br['solver_name'])}</p>
                <p><strong>Confidence:</strong> {br['confidence']:.1%}</p>
                <p><strong>Method:</strong> {html.escape(str(br.get('details', {}).get('method', '—')))}</p>
                <p><strong>Decrypted:</strong> <code>{html.escape(decrypted_str)}</code></p>
            </div>"""
        else:
            best_info = f"""
            <div class="result-box" style="border-color:{status_color}">
                <h2 style="color:{status_color}">{status_text}</h2>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OMNISCIENCE — Attack Report</title>
<style>
    body {{ font-family: 'Consolas', 'Courier New', monospace; background: #1a1a2e; color: #eee; margin: 2em; }}
    h1 {{ color: #00d4ff; text-align: center; }}
    h2 {{ color: #ccc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
    th {{ background: #16213e; color: #00d4ff; }}
    tr:nth-child(even) {{ background: #0f3460; }}
    .result-box {{ border: 2px solid; border-radius: 8px; padding: 1.5em; margin: 1.5em 0; background: #16213e; }}
    code {{ background: #0f3460; padding: 2px 6px; border-radius: 3px; }}
    .meta {{ color: #888; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>OMNISCIENCE — Attack Report</h1>
<p class="meta">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total time: {d.get('total_elapsed', 0):.2f}s</p>

{best_info}

<h2>Reconnaissance</h2>
<table>
<tr><th>Property</th><th>Value</th></tr>
{recon_rows}
</table>

<h2>Solver Results</h2>
<table>
<tr><th>Solver</th><th>Status</th><th>Confidence</th><th>Time</th><th>Method</th></tr>
{solver_rows}
</table>

<h2>Raw Data</h2>
<details>
<summary>Click to expand JSON</summary>
<pre>{html.escape(json.dumps(d, indent=2, default=str))}</pre>
</details>
</body>
</html>"""

    @staticmethod
    def save_html(report: AttackReport, path: str | Path) -> None:
        Path(path).write_text(ReportExporter.to_html(report))

    # ------------------------------------------------------------------ #
    #  Plaintext summary                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_text(report: AttackReport) -> str:
        lines = [
            "=" * 60,
            "  OMNISCIENCE — Attack Report",
            "=" * 60,
            f"  Time: {report.total_elapsed:.2f}s",
            f"  Result: {'SUCCESS' if report.success() else 'FAILED'}",
            "",
        ]

        if report.recon:
            r = report.recon
            lines += [
                "--- Reconnaissance ---",
                f"  Family:      {r.algo_family.value}",
                f"  Confidence:  {r.confidence:.1%}",
                f"  Linearity:   {r.linearity_score:.4f}",
                f"  Entropy PT:  {r.entropy_plaintext:.4f}",
                f"  Entropy CT:  {r.entropy_ciphertext:.4f}",
                f"  Poly degree: {r.polynomial_degree_estimate or 'N/A'}",
                f"  Modulus:     {r.estimated_modulus or 'N/A'}",
                "",
            ]

        lines.append("--- Solvers ---")
        for sr in report.solver_results:
            status = "OK" if sr.status == SolverStatus.SUCCESS else "FAIL"
            method = sr.details.get("method", "—")
            lines.append(f"  [{status:4s}] {sr.solver_name:20s} conf={sr.confidence:.1%}  time={sr.elapsed_seconds:.2f}s  method={method}")

        if report.success():
            lines += [
                "",
                "--- Result ---",
                f"  Solver:     {report.best_result.solver_name}",
                f"  Confidence: {report.best_result.confidence:.1%}",
                f"  Decrypted:  {report.best_result.decrypted}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _report_to_dict(report: AttackReport) -> dict[str, Any]:
        recon_dict = {}
        if report.recon:
            r = report.recon
            recon_dict = {
                "algo_family": r.algo_family.value,
                "confidence": r.confidence,
                "linearity_score": r.linearity_score,
                "entropy_plaintext": r.entropy_plaintext,
                "entropy_ciphertext": r.entropy_ciphertext,
                "polynomial_degree_estimate": r.polynomial_degree_estimate,
                "substitution_detected": r.substitution_detected,
                "lattice_structure_detected": r.lattice_structure_detected,
                "estimated_modulus": r.estimated_modulus,
            }

        def _sr_dict(sr: SolverResult) -> dict:
            return {
                "solver_name": sr.solver_name,
                "status": sr.status.value,
                "confidence": sr.confidence,
                "elapsed_seconds": sr.elapsed_seconds,
                "private_key": str(sr.private_key) if sr.private_key else None,
                "decrypted": sr.decrypted,
                "details": sr.details,
            }

        return {
            "timestamp": report.timestamp,
            "total_elapsed": report.total_elapsed,
            "success": report.success(),
            "recon": recon_dict,
            "solver_results": [_sr_dict(sr) for sr in report.solver_results],
            "best_result": _sr_dict(report.best_result) if report.best_result else None,
        }
