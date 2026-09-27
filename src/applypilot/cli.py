"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import APP_DIR, DB_PATH, load_env, ensure_dirs
    from applypilot.database import StorageInitError, init_db

    load_env()
    try:
        ensure_dirs()
        init_db()
    except StorageInitError as exc:
        console.print("[red]Failed to initialize ApplyPilot storage.[/red]")
        console.print(f"  APP_DIR: {APP_DIR}")
        console.print(f"  DB_PATH: {DB_PATH}")
        console.print(f"  Reason: {exc}")
        console.print(
            "\nSet [bold]APPLYPILOT_DIR[/bold] to a writable path, "
            "then run the command again."
        )
        raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    engine: str = typer.Option(
        "claude",
        "--engine",
        help="Auto-apply engine: claude (Claude Code CLI) or playwright (built-in).",
    ),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    phone: Optional[str] = typer.Option(None, "--phone", help="Phone override for application forms."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
    worker_id: int = typer.Option(0, "--worker-id", hidden=True),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    if phone and phone.strip():
        os.environ["APPLYPILOT_PHONE"] = phone.strip()

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    engine_norm = (engine or "claude").strip().lower()
    if engine_norm not in ("claude", "playwright"):
        console.print(f"[red]Unknown --engine:[/red] {engine}")
        raise typer.Exit(code=1)

    if gen and engine_norm != "claude":
        console.print("[red]--gen is only supported with --engine claude.[/red]")
        raise typer.Exit(code=1)

    # Check 1: dependencies
    if engine_norm == "claude":
        # Claude Code CLI + Chrome
        check_tier(3, "auto-apply")
    else:
        # Built-in Playwright engine: requires Chrome, but not Claude Code CLI.
        from applypilot.config import get_chrome_path

        try:
            get_chrome_path()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    if engine_norm == "playwright":
        profile_data = {}
        try:
            from applypilot.config import load_profile as _load_profile

            profile_data = _load_profile() or {}
        except Exception:
            profile_data = {}

        profile_phone = (profile_data.get("personal", {}).get("phone") or "").strip()
        env_phone = (os.environ.get("APPLYPILOT_PHONE") or "").strip()
        if not profile_phone and not env_phone:
            console.print(
                "[red]Phone number is required for auto-apply forms.[/red]\n"
                "Set it in profile.json or pass [bold]--phone[/bold]."
            )
            raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Engine:   {engine_norm}")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    if engine_norm == "claude":
        console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    if engine_norm == "claude":
        from applypilot.apply.launcher import main as apply_main

        apply_main(
            limit=effective_limit,
            target_url=url,
            min_score=min_score,
            headless=headless,
            model=model,
            dry_run=dry_run,
            continuous=continuous,
            workers=workers,
        )
        return

    # engine_norm == "playwright"
    from applypilot.apply.playwright_engine import main as pw_apply_main

    if workers <= 1:
        pw_apply_main(
            limit=effective_limit,
            target_url=url,
            min_score=min_score,
            headless=headless,
            dry_run=dry_run,
            continuous=continuous,
            worker_id=int(worker_id),
        )
        return

    if url:
        console.print("[red]--url with --engine playwright only supports --workers 1.[/red]")
        raise typer.Exit(code=1)

    # Distribute fixed limits; in continuous mode, every worker runs unbounded.
    if effective_limit:
        base = effective_limit // workers
        extra = effective_limit % workers
        limits = [base + (1 if i < extra else 0) for i in range(workers)]
    else:
        limits = [0] * workers

    procs: list[tuple[int, list[str], subprocess.Popen]] = []
    child_env = dict(os.environ)
    if phone and phone.strip():
        child_env["APPLYPILOT_PHONE"] = phone.strip()
    for i in range(workers):
        args = [
            sys.executable,
            "-m",
            "applypilot",
            "apply",
            "--engine",
            "playwright",
            "--workers",
            "1",
            "--min-score",
            str(min_score),
            "--worker-id",
            str(i),
        ]
        if headless:
            args.append("--headless")
        if dry_run:
            args.append("--dry-run")
        if continuous:
            args.append("--continuous")
        else:
            args.extend(["--limit", str(int(limits[i]))])

        proc = subprocess.Popen(args, env=child_env)
        procs.append((i, args, proc))

    failures = 0
    try:
        while procs:
            next_round: list[tuple[int, list[str], subprocess.Popen]] = []
            for i, args, proc in procs:
                rc = proc.poll()
                if rc is None:
                    next_round.append((i, args, proc))
                    continue

                if rc != 0:
                    failures += 1
                    console.print(f"[yellow]Playwright worker {i} exited ({rc}).[/yellow]")
                    if continuous:
                        restarted = subprocess.Popen(args, env=child_env)
                        console.print(f"[yellow]Restarted worker {i} (pid {restarted.pid}).[/yellow]")
                        next_round.append((i, args, restarted))

            procs = next_round
            if procs:
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping Playwright workers...[/yellow]")
        for _, _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for _, _, proc in procs:
            try:
                proc.wait(timeout=10)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
        raise

    if failures and not continuous:
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


if __name__ == "__main__":
    app()
