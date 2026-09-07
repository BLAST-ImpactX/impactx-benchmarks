"""Publish results + plots to the dedicated ``benchmarks`` branch.

Uses a detached git **worktree** so the working tree on the current branch is never
touched. The branch is created as an **orphan** the first time. On the branch:

* ``results/<machine>.yaml``      -- latest results for this machine (overwritten)
* ``plots/<machine>/<scenario>.{svg,pdf}`` -- latest per-machine plots, rendered at publish time
                                     from that machine's results (so machines don't overwrite each
                                     other; the legacy flat plots/ layout is migrated away)
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

    # per-run manifests for THIS machine (template-resolved inputs + run.sh); reviewable on branch
    runs_dir = REPO_ROOT / "runs" / slug
    run_cells = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    print("Would publish:")
    print(f"  results/{slug}.yaml")
    print(f"  plots/{slug}/  (rendered from this machine's results at publish; other machines kept)")
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
            _publish_files(wt_path, slug, res_path, data, utc, runs_dir)
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


def _publish_files(wt: Path, slug: str, res_path: Path, data: dict, utc: str,
                   runs_dir: Path) -> None:
    from . import plotting as plot_mod

    (wt / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy2(res_path, wt / "results" / f"{slug}.yaml")

    # Per-machine plots. Render THIS machine's plots into plots/<slug>/ from its own results, so
    # publishing one machine never clobbers another's charts. Other machines' plots/<other>/ live
    # on the fetched branch and are left untouched. Migrate away the legacy FLAT layout: drop any
    # top-level plots/ entry that isn't a known machine dir (old <scenario>.png files + old gpu/).
    plots_root = wt / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    known = {p.stem for p in (wt / "results").glob("*.yaml")}
    for p in plots_root.iterdir():
        if p.name not in known:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    mplots = plots_root / slug
    if mplots.exists():
        shutil.rmtree(mplots)
    mplots.mkdir(parents=True, exist_ok=True)
    plot_mod.plot_all(data, out_dir=mplots)
    plot_mod.plot_all_gpu(data, out_dir=mplots / "gpu")

    # per-run manifests (template-resolved inputs + run.sh) -- overwrite the machine's tree
    dst_runs = wt / "runs" / slug
    if dst_runs.exists():
        shutil.rmtree(dst_runs)
    if runs_dir.is_dir():
        shutil.copytree(runs_dir, dst_runs)
    # per-run archive (results + this machine's plots + the run manifests, under this UTC stamp)
    archive = wt / "history" / f"{utc.replace(':', '').replace('-', '')}_{slug}"
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res_path, archive / f"{slug}.yaml")
    shutil.copytree(mplots, archive / "plots")
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
