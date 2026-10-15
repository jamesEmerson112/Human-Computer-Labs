"""Rebuild the browser section of this run's audited analysis from raw summaries."""

import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
RUNS = {
    "python_reference": "python-browser-02",
    "native_before_tcp_nodelay": "rust-browser",
    "native_default_before_tcp_nodelay": "rust-browser-default",
    "native_matched": "rust-browser-02",
    "native_default": "rust-browser-default-02",
}


def summarize(directory: str) -> dict:
    path = ROOT / directory
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    repeats = manifest["repeats"]
    assert len(summary["phases"]) == 2 * repeats
    assert summary["click_to_status_ms"]["count"] == 6 * repeats
    result = {
        "source": directory,
        "timestamp_utc": manifest["timestamp_utc"],
        "repeats": repeats,
        "webgl": manifest["webgl"],
        "click_to_status_ms": summary["click_to_status_ms"],
        "phases": {},
    }
    for phase in ("idle", "active"):
        rows = [row for row in summary["phases"] if row["phase"] == phase]
        assert sorted(row["repeat"] for row in rows) == list(range(1, repeats + 1))
        metrics = {}
        for name in (
            "server_tick_hz", "simulation_seconds_per_wall_second",
            "server_cpu_percent_one_core", "browser_cpu_percent_one_core",
            "browser_draw_frames_per_s",
        ):
            metrics["median_" + name] = median(row[name] for row in rows)
        for name in ("server_peak_sampled_rss_mib", "browser_peak_sampled_tree_rss_mib"):
            metrics[name] = max(row[name] for row in rows)
        metrics["median_phase_p95_tick_compute_ms"] = median(row["server_tick_compute_ms"]["p95"] for row in rows)
        metrics["median_phase_mean_host_cpu_percent"] = median(row["host_cpu_percent"]["mean"] for row in rows)
        metrics["phase_mean_host_cpu_percent_range"] = [
            min(row["host_cpu_percent"]["mean"] for row in rows),
            max(row["host_cpu_percent"]["mean"] for row in rows),
        ]
        metrics["browser_long_task_count"] = sum(row["browser_long_tasks"]["count"] for row in rows)
        if all("dropped_simulation_time_s" in row for row in rows):
            metrics["total_dropped_simulation_time_s"] = sum(row["dropped_simulation_time_s"] for row in rows)
        result["phases"][phase] = metrics
    return result


if __name__ == "__main__":
    target = ROOT / "analysis.json"
    analysis = json.loads(target.read_text(encoding="utf-8"))
    analysis["browser"] = {
        "method": "Rates, CPU and per-phase p95 tick times are medians across three repeats; memory is the maximum sampled phase peak. Latency pools 18 commands. Draw calls are not GPU completion. Background CPU load is recorded, not controlled.",
        "runs": {name: summarize(directory) for name, directory in RUNS.items()},
    }
    target.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Audited five browser runs into analysis.json")
