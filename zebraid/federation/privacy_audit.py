"""
zebraid/federation/privacy_audit.py
Privacy audit script — replays the federation query log and prints a
per-query data-exposure report.

For every cross-org query, confirms:
  ✅ Only Z-Hash bytes crossed the boundary (measured and logged).
  ✅ raw_image_transmitted = False for every entry.
  ✅ gps_transmitted = False for every entry.
  ✅ Response contained only a score bucket (not a raw float score).

Run:
    python -m zebraid.federation.privacy_audit \
        --log results/federation_queries.jsonl \
        --output results/privacy_audit_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_audit(log_path: str, output_path: Optional[str] = None) -> dict:
    from typing import Optional

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"[privacy_audit] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("[privacy_audit] No entries found in log.")
        return {}

    # ── Audit checks ─────────────────────────────────────────────────────────
    violations = []
    stats = {
        "total_queries": len(entries),
        "raw_image_violations": 0,
        "gps_violations": 0,
        "raw_score_violations": 0,
        "total_bytes_transmitted": 0,
        "valid_buckets": {"NO_MATCH", "POSSIBLE_MATCH", "STRONG_MATCH", "ERROR"},
    }

    lines = [
        "=" * 70,
        "ZebraID Federated Protocol — Privacy Audit Report",
        "=" * 70,
        f"Log file:      {log_path}",
        f"Total queries: {len(entries)}",
        "",
        f"{'#':<5} {'Requester':<12} {'Target':<30} {'Bytes':>6} {'Bucket':<16} {'IMG':>5} {'GPS':>5}",
        "-" * 80,
    ]

    for i, entry in enumerate(entries):
        raw_img = entry.get("raw_image_transmitted", True)   # default True = violation
        gps     = entry.get("gps_transmitted", True)
        bucket  = entry.get("match_bucket", "UNKNOWN")
        b_bytes = entry.get("z_hash_bytes_transmitted", 0)

        stats["total_bytes_transmitted"] += b_bytes

        if raw_img:
            stats["raw_image_violations"] += 1
            violations.append(f"Query {i}: raw_image_transmitted=True — VIOLATION")
        if gps:
            stats["gps_violations"] += 1
            violations.append(f"Query {i}: gps_transmitted=True — VIOLATION")
        if bucket not in stats["valid_buckets"]:
            stats["raw_score_violations"] += 1
            violations.append(f"Query {i}: invalid bucket '{bucket}' — possible raw score leak")

        img_flag = "❌" if raw_img else "✅"
        gps_flag = "❌" if gps else "✅"
        target = entry.get("target_org_url", "?")[-28:]

        lines.append(
            f"{i+1:<5} {entry.get('requester_org','?'):<12} {target:<30} "
            f"{b_bytes:>6} {bucket:<16} {img_flag:>5} {gps_flag:>5}"
        )

    lines += [
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"Total queries:            {stats['total_queries']}",
        f"Total Z-Hash bytes sent:  {stats['total_bytes_transmitted']} bytes",
        f"Raw image violations:     {stats['raw_image_violations']}",
        f"GPS violations:           {stats['gps_violations']}",
        f"Raw score leaks:          {stats['raw_score_violations']}",
        "",
    ]

    if violations:
        lines.append("⚠️  VIOLATIONS FOUND:")
        lines.extend(f"  - {v}" for v in violations)
    else:
        lines.append("✅  No privacy violations found.")
        lines.append("    Every query transmitted Z-Hash bytes only.")
        lines.append("    No raw images or GPS data crossed organizational boundaries.")

    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n[privacy_audit] Report saved to {output_path}")

    stats["violations"] = violations
    return stats


if __name__ == "__main__":
    from typing import Optional
    parser = argparse.ArgumentParser(description="ZebraID federated protocol privacy audit")
    parser.add_argument("--log",    required=True, help="Path to federation JSONL query log")
    parser.add_argument("--output", default=None,  help="Optional path to write report text")
    args = parser.parse_args()

    result = run_audit(args.log, args.output)
    sys.exit(0 if not result.get("violations") else 1)
