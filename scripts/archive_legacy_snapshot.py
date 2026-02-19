from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _run_git(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _run_git_quiet(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False, text=True, capture_output=True).returncode


def _git_output(cmd: list[str]) -> str:
    return _run_git(cmd).stdout.strip()


def _ref_exists(ref: str) -> bool:
    return _run_git_quiet(["git", "show-ref", "--verify", "--quiet", ref]) == 0


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    ap = argparse.ArgumentParser(
        description="Archive current code snapshot before v2 rewrite (create branch + annotated tag)."
    )
    ap.add_argument(
        "--archive-name",
        type=str,
        default=f"v1_freeze_{today}",
        help="Logical archive name used to build branch/tag names.",
    )
    ap.add_argument("--branch-prefix", type=str, default="archive")
    ap.add_argument("--tag-prefix", type=str, default="archive")
    ap.add_argument("--branch-name", type=str, default=None, help="Full branch name override.")
    ap.add_argument("--tag-name", type=str, default=None, help="Full tag name override.")
    ap.add_argument("--message", type=str, default=None, help="Annotated tag message.")
    ap.add_argument("--allow-dirty", action="store_true", help="Allow archiving with uncommitted changes.")
    ap.add_argument("--push", action="store_true", help="Push created branch/tag to remote.")
    ap.add_argument("--remote", type=str, default="origin")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Execute git commands. Without this flag, only print the planned commands.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(_git_output(["git", "rev-parse", "--show-toplevel"]))
    head = _git_output(["git", "rev-parse", "HEAD"])
    head_short = _git_output(["git", "rev-parse", "--short", "HEAD"])
    branch_current = _git_output(["git", "branch", "--show-current"])
    status = _git_output(["git", "status", "--porcelain"])
    dirty = bool(status.strip())

    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Commit/stash changes first, or rerun with --allow-dirty."
        )

    branch_name = args.branch_name or f"{args.branch_prefix}/{args.archive_name}"
    tag_name = args.tag_name or f"{args.tag_prefix}/{args.archive_name}"
    tag_message = args.message or (
        f"Archive snapshot before v2 rewrite | branch={branch_current} | head={head_short} | "
        f"date={datetime.now().isoformat(timespec='seconds')}"
    )

    branch_exists = _ref_exists(f"refs/heads/{branch_name}")
    tag_exists = _ref_exists(f"refs/tags/{tag_name}")
    planned: list[list[str]] = []

    if not branch_exists:
        planned.append(["git", "branch", branch_name, head])
    if not tag_exists:
        planned.append(["git", "tag", "-a", tag_name, head, "-m", tag_message])
    if args.push:
        if not branch_exists:
            planned.append(["git", "push", args.remote, branch_name])
        if not tag_exists:
            planned.append(["git", "push", args.remote, tag_name])

    summary = {
        "repo_root": str(repo_root),
        "current_branch": branch_current,
        "head": head,
        "head_short": head_short,
        "dirty": dirty,
        "branch_name": branch_name,
        "branch_exists": branch_exists,
        "tag_name": tag_name,
        "tag_exists": tag_exists,
        "execute": bool(args.execute),
        "planned_commands": [" ".join(c) for c in planned],
    }

    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    for cmd in planned:
        subprocess.run(cmd, check=True)

    summary["executed"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

