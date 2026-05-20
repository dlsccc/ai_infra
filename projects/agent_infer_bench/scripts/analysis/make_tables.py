from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.analysis.summarize import make_markdown_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    summary = make_markdown_summary(run_dir)
    report_path = run_dir / "summary.md"
    report_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()

