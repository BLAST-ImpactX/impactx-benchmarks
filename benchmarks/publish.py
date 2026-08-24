"""Publish results + plots to the dedicated ``benchmarks`` branch.

Uses a detached git **worktree** so the working tree on the current branch is never
touched. The branch is created as an **orphan** the first time. On the branch:

* ``results/<machine>.yaml``      -- latest results for this machine (overwritten)
* ``plots/<scenario>.{svg,pdf}``  -- latest plots (overwritten; embedded in README)
* ``runs/<machine>/<cell>/``      -- per-cell template-resolved input file(s) + ``run.sh``
                                     (exact launch command + env) so the codes' authors can
                                     review the (LLM-generated) templates. Inputs only -- no run
                                     outputs. Written by the runner; backfill via
                                     ``runner --write-manifests``.
* ``history/<utc>_<machine>/``    -- per-run archive of results + plots + run manifests

The commit message carries the full host/OS/CPU/compiler + version metadata.

Local runs are **opt-in**: nothing is committed or pushed without ``--push``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import metadata as meta_mod
from . import results as results_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
BRANCH = "benchmarks"


def _git(*args, check=True, capture=False, cwd=REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=check, text=True,
        capture_output=capture,
    )


def _detect_remote() -> str:
    out = _git("remote", capture=True).stdout.split()
    if "origin" in out:
        return "origin"
    return out[0] if out else "origin"


def _remote_branch_exists(remote: str) -> bool:
    res = _git("ls-remote", "--exit-code", "--heads", remote, BRANCH,
               check=False, capture=True)
    return res.returncode == 0


def _summary(data: dict) -> str:
    counts: dict[str, int] = {}
    for measurements in data.get("results", {}).values():
        for entry in measurements.values():
            key = entry.get("physics") or entry.get("status") or "unknown"
            counts[key] = counts.get(key, 0) + 1
    return "summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def publish(push: bool, remote: str | None = None) -> int:
    remote = remote or _detect_remote()
    slug = meta_mod.machine_slug()
    res_path = results_mod.results_path(slug)
    if not res_path.exists():
        print(f"No results file at {res_path}; nothing to publish.")
        return 1
    data = results_mod.load(res_path)
    msg = meta_mod.as_commit_message(data.get("metadata", {}), _summary(data))

    # Recurse so nested plot dirs (e.g. plots/gpu/ from `plot --gpu`) are included; files only
    # (a bare glob("*") would yield the gpu/ directory itself and shutil.copy2 would choke on it).
    plots = (sorted(p for p in (REPO_ROOT / "plots").rglob("*") if p.is_file())
             if (REPO_ROOT / "plots").is_dir() else [])
    # per-run manifests for THIS machine (template-resolved inputs + run.sh); reviewable on branch
    runs_dir = REPO_ROOT / "runs" / slug
    run_cells = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    print("Would publish:")
    print(f"  results/{slug}.yaml")
    for p in plots:
        print(f"  plots/{p.relative_to(REPO_ROOT / 'plots')}")
    if run_cells:
        print(f"  runs/{slug}/  ({len(run_cells)} run manifests: resolved input + run.sh)")
    print("\nCommit message:\n" + "\n".join("  " + ln for ln in msg.splitlines()))

    if not push:
        print("\n(dry run; pass --push to commit and push)")
        return 0

    utc = data.get("metadata", {}).get("host", {}).get("timestamp_utc", "run")
    with tempfile.TemporaryDirectory(prefix="bench_wt_") as wt:
        wt_path = Path(wt)
        if _remote_branch_exists(remote):
            _git("fetch", remote, BRANCH)
            _git("worktree", "add", str(wt_path), f"{remote}/{BRANCH}")
            _git("switch", "-C", BRANCH, cwd=wt_path)
        else:
            _git("worktree", "add", "--detach", str(wt_path))
            _git("switch", "--orphan", BRANCH, cwd=wt_path)
            # clean any inherited files on the fresh orphan
            for child in wt_path.iterdir():
                if child.name != ".git":
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

        try:
            _publish_files(wt_path, slug, res_path, plots, utc, runs_dir)
            _git("add", "-A", cwd=wt_path)
            staged = _git("diff", "--cached", "--quiet", check=False, cwd=wt_path)
            if staged.returncode == 0:
                print("No changes to publish.")
                return 0
            _git("-c", "user.name=benchmarks-bot",
                 "-c", "user.email=benchmarks-bot@users.noreply.github.com",
                 "commit", "-m", msg, cwd=wt_path)
            _git("push", remote, f"HEAD:{BRANCH}", cwd=wt_path)
            print(f"Pushed to {remote}/{BRANCH}.")
        finally:
            _git("worktree", "remove", "--force", str(wt_path), check=False)
    return 0


def _publish_files(wt: Path, slug: str, res_path: Path, plots: list[Path], utc: str,
                   runs_dir: Path) -> None:
    (wt / "results").mkdir(parents=True, exist_ok=True)
    (wt / "plots").mkdir(parents=True, exist_ok=True)
    shutil.copy2(res_path, wt / "results" / f"{slug}.yaml")
    plots_root = REPO_ROOT / "plots"
    for p in plots:
        dst = wt / "plots" / p.relative_to(plots_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    # per-run manifests (template-resolved inputs + run.sh) -- overwrite the machine's tree
    dst_runs = wt / "runs" / slug
    if dst_runs.exists():
        shutil.rmtree(dst_runs)
    if runs_dir.is_dir():
        shutil.copytree(runs_dir, dst_runs)
    # per-run archive (results + plots + the run manifests, snapshotted under this UTC stamp)
    archive = wt / "history" / f"{utc.replace(':', '').replace('-', '')}_{slug}"
    (archive / "plots").mkdir(parents=True, exist_ok=True)
    shutil.copy2(res_path, archive / f"{slug}.yaml")
    for p in plots:
        dst = archive / "plots" / p.relative_to(plots_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    if runs_dir.is_dir():
        shutil.copytree(runs_dir, archive / "runs")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish results to the benchmarks branch.")
    parser.add_argument("--push", action="store_true", help="actually commit & push")
    parser.add_argument("--remote", default="", help="git remote (default: auto-detect)")
    args = parser.parse_args(argv)
    return publish(push=args.push, remote=args.remote or None)


if __name__ == "__main__":
    sys.exit(main())
