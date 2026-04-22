#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> int:
    base = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/019151743/special_topics/STTran/outputs"
    )
    rows = []
    for summary in sorted(base.glob("sttran_phase3_rank_epoch_*/rank_summary.json")):
        try:
            payload = json.loads(summary.read_text())
        except Exception:
            continue
        payload["summary_path"] = str(summary)
        rows.append(payload)

    rows.sort(key=lambda x: (x.get("with_r20") is None, -(x.get("with_r20") or -1.0), x.get("rank_tag", "")))

    print("rank_tag\twith_r20\tdetector_ckpt\tsummary_path")
    for row in rows:
        r20 = row.get("with_r20")
        print(
            "{}\t{}\t{}\t{}".format(
                row.get("rank_tag", ""),
                "nan" if r20 is None else f"{r20:.6f}",
                row.get("detector_ckpt", ""),
                row.get("summary_path", ""),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
