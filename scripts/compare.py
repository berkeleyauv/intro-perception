"""Print the comparison table from ``intro-perception-evaluate`` JSON outputs.

    python scripts/compare.py output/metrics_classical.json output/metrics_yolo.json

Prints a Markdown table (one column per method) with IoU@0.5 precision and
recall, mAP50, mean normalized center error, FPS, and recall on whole versus
cut-off gates. With ``--report REPORT.md`` the table between the
``<!-- comparison:start -->`` and ``<!-- comparison:end -->`` markers is
replaced in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = (
    ("precision @ IoU 0.5", lambda m: m["precision_iou50"], "{:.3f}"),
    ("recall @ IoU 0.5", lambda m: m["recall_iou50"], "{:.3f}"),
    ("mAP50", lambda m: m["map50"], "{:.3f}"),
    ("mean center error (normalized)", lambda m: m["mean_center_error"], "{:.4f}"),
    ("FPS", lambda m: m["fps"], "{:.1f}"),
    ("recall, whole gates", lambda m: m["by_truncation"]["whole"]["recall_iou50"], "{:.3f}"),
    ("recall, cut-off gates", lambda m: m["by_truncation"]["truncated"]["recall_iou50"], "{:.3f}"),
    ("precision, cut-off gates", lambda m: m["by_truncation"]["truncated"]["precision_iou50"], "{:.3f}"),
)


def method_name(path):
    return path.stem.replace("metrics_", "")


def table(paths):
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    # The CLI prints {"classical": {...}} when run live, or the scores directly.
    metrics = [m[next(iter(m))] if "frames" not in m and len(m) == 1 else m for m in metrics]
    lines = ["| metric | " + " | ".join(method_name(p) for p in paths) + " |",
             "|---|" + "---:|" * len(paths)]
    for label, get, fmt in ROWS:
        cells = []
        for m in metrics:
            try:
                value = get(m)
                cells.append(fmt.format(value) if value is not None else "–")
            except (KeyError, TypeError):
                cells.append("–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| test frames | " + " | ".join(str(m.get("frames", "–")) for m in metrics) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    text = table(args.metrics)
    print(text)
    if args.report:
        report = args.report.read_text(encoding="utf-8")
        start, end = "<!-- comparison:start -->", "<!-- comparison:end -->"
        if start not in report or end not in report:
            raise SystemExit(f"{args.report} has no {start} ... {end} markers")
        before, rest = report.split(start, 1)
        _old, after = rest.split(end, 1)
        args.report.write_text(f"{before}{start}\n{text}\n{end}{after}", encoding="utf-8")
        print(f"updated {args.report}")


if __name__ == "__main__":
    main()
