#!/usr/bin/env python
r"""
run_mana_baseline.py  --  Run MANA framework on all 43,320 MARSIM pcap files.

Produces a CSV with per-file predictions and per-method trigger flags.
Must be run BEFORE train_classifiers.py.

Usage:
    conda activate marsim_env
    python run_mana_baseline.py

Output:
    D:\IDMAN_Downloads\ZIP\8202936\results\mana\mana_predictions.csv

Notes:
    - MANA must be installed in the environment (python setup.py install).
    - methods.json must exist at D:\IDMAN_Downloads\ZIP\8202936\methods.json
    - Two methods excluded (OrbitPositionsMethod, PhysicalEnvironmentLimitMethod)
      because required data files (gps.tle, water_map.png) were not bundled
      during setup.py install.
    - Uses multiprocessing for parallel file processing.
"""

import json
import os
import time
import multiprocessing
import pandas as pd

from mana.feeder import PcapFeeder
from mana.handler import DetectionHandler
from mana.method import load_methods_json

# ============================================================
# Paths
# ============================================================
METHODS_JSON_PATH = r"D:\IDMAN_Downloads\ZIP\8202936\methods.json"
BASE_PATH = r"D:\IDMAN_Downloads\ZIP\8202936\dataset\dataset"
MANA_RESULTS_DIR = r"D:\IDMAN_Downloads\ZIP\8202936\results\mana"
os.makedirs(MANA_RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(MANA_RESULTS_DIR, "mana_predictions.csv")

# ============================================================
# Worker function (runs in separate process)
# ============================================================
def process_single_file(args):
    """Process one pcap file through MANA and return its result dict."""
    filename, base_path, methods_json_path = args
    filepath = os.path.join(base_path, filename)

    # Each worker must load its own MANA config (not shared across processes)
    device_ids, method_classes, method_options = load_methods_json(methods_json_path)

    triggered_methods = []

    def on_spoofing_attack(device_id, spoofing_indicator, method, state):
        method_name = type(method).__name__
        if method_name not in triggered_methods:
            triggered_methods.append(method_name)

    handler = DetectionHandler(
        device_ids=device_ids,
        method_classes=method_classes,
        method_options=method_options,
        detection_threshold=0.1,
        on_spoofing_attack=on_spoofing_attack,
    )

    feeder = PcapFeeder(handler, filepath)

    try:
        feeder.run()
    except Exception as e:
        return {"filename": filename, "mana_pred": -1, "error": str(e)}

    mana_pred = 1 if len(triggered_methods) > 0 else 0

    return {
        "filename": filename,
        "mana_pred": mana_pred,
        "mana_triggered_methods": ",".join(triggered_methods) if triggered_methods else "",
        "pdm": 1 if "MultipleReceiversMethod" in triggered_methods else 0,
        "pcc_sog": 1 if "PhysicalSpeedLimitMethod" in triggered_methods else 0,
        "pcc_rot": 1 if "PhysicalRateOfTurnLimitMethod" in triggered_methods else 0,
        "pcc_height": 1 if "PhysicalHeightLimitMethod" in triggered_methods else 0,
        "cdm": 1 if "TimeDriftMethod" in triggered_methods else 0,
        "cnm": 1 if "CarrierToNoiseDensityMethod" in triggered_methods else 0,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # Load dataset manifest
    with open(os.path.join(BASE_PATH, "dataset.json")) as f:
        dataset = json.load(f)

    # Load method names for display only
    device_ids, method_classes, method_options = load_methods_json(METHODS_JSON_PATH)

    n_workers = max(1, os.cpu_count() - 2)  # leave 2 cores free

    print("=" * 60)
    print("MANA Baseline Runner (Multiprocessing)")
    print("=" * 60)
    print(f"Total files to process: {len(dataset)}")
    print(f"Workers: {n_workers} (of {os.cpu_count()} CPU cores)")
    print(f"Active MANA methods ({len(method_classes)}):")
    for mc in method_classes:
        print(f"  - {mc.__name__}")
    print()
    print("NOTE: OrbitPositionsMethod (EDV) and PhysicalEnvironmentLimitMethod")
    print("      (PCC_env) are EXCLUDED — required data files (gps.tle,")
    print("      water_map.png) were not copied during setup.py install.")
    print("=" * 60)

    # Prepare worker arguments
    work_items = [
        (entry["filename"], BASE_PATH, METHODS_JSON_PATH)
        for entry in dataset
    ]

    # Process in parallel
    results = []
    start_time = time.time()

    with multiprocessing.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_file, work_items, chunksize=20)):
            results.append(result)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(dataset) - (i + 1)) / rate
                print(
                    f"  [{i+1:>5}/{len(dataset)}]  "
                    f"{rate:.1f} files/s, ~{remaining/60:.0f} min remaining"
                )

    # ============================================================
    # Save results
    # ============================================================
    mana_df = pd.DataFrame(results)
    mana_df.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Done in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {mana_df.shape}")
    print(
        f"Predictions: spoofed={int((mana_df['mana_pred']==1).sum())}, "
        f"unspoofed={int((mana_df['mana_pred']==0).sum())}, "
        f"errors={int((mana_df['mana_pred']==-1).sum())}"
    )
    print()

    # Per-method trigger summary
    print("Per-method trigger counts:")
    for col in ["pdm", "pcc_sog", "pcc_rot", "pcc_height", "cdm", "cnm"]:
        if col in mana_df.columns:
            count = int((mana_df[col] == 1).sum())
            print(f"  {col:>12s}: triggered on {count} files")

    print("=" * 60)
