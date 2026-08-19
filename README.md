# maritime_spoofing v2 (refactor for the journal version)

Refactor of the CIMSS 2026 pipeline (ML vs MANA on MARSIM) into a reusable
package supporting the journal-version experiments.

## Layout
```
config.yaml                  all paths and knobs (no more hardcoded D:\ paths)
src/mspoof/
  geo.py        haversine + exact great-circle bearing (FIXES the old
                atan2(dlon, dlat) heading bias: 7.5 deg error at 40N on a
                45 deg course) + angular utils
  nmea.py       talker-agnostic sentence parsers (GP/GN/GL/GA/GB), checksum
                validation, canonical field names
  epochs.py     epoch assembly, RMC->VTG canonical sog/cog fallback, derived
                features, O(n) inter-receiver matching, per-epoch DataFrame
  features.py   aggregation; profiles: "full" (138 feats) and "deployable"
                (93 feats: GGA+VTG+ZDA only, matches real vessel logs)
  splits.py     repetition split (paper), 5-fold repetition (variance),
                param_holdout presets (generalization to unseen attack params)
  readers/
    pcap_reader.py  MARSIM pcaps
    scs_reader.py   NOAA SCS / R2R / OpenRVDAS text logs + sliding windows
scripts/
  extract_features.py      MARSIM -> features CSV (+ per-epoch parquet for
                           future LSTM/Transformer work), multiprocessing
  train_classifiers.py     tiers: default | adjusted | both; SVM full-set by
                           default; MANA error rows excluded and counted;
                           base-rate (prevalence) analysis; run manifest
  extract_real_windows.py  real logs -> 120 s windows -> deployable features
tests/test_pipeline.py     synthetic pcap + SCS end-to-end tests
```

## Typical runs
```
python scripts/extract_features.py --config config.yaml
python scripts/train_classifiers.py --config config.yaml                     # repetition split
python scripts/train_classifiers.py --config config.yaml --split kfold      # mean +/- std
python scripts/train_classifiers.py --config config.yaml --split param_holdout --preset a3_speed_low
python scripts/extract_features.py --config config.yaml --profile deployable
python scripts/extract_real_windows.py --config config.yaml --real-dir /data/real_nmea --out real_windows.csv
python tests/test_pipeline.py
```

## Real-data input layout
```
real_nmea/<cruise_id>/rx1/*   primary GNSS device logs (SCS .Raw or ISO-prefixed)
real_nmea/<cruise_id>/rx2/*   second device (optional; enables inter-receiver features)
```
Sources verified: NOAA OMAO-SCS accessions at NCEI (GGA/VTG/ZDA/HDT/ROT...),
R2R academic-fleet GNSS filesets (e.g. Falkor FK180528: C-Nav 3050 + Seapath 320).

## Behavior changes vs v1 (all intentional, all affect reruns)
1. Position-derived heading uses the exact bearing formula.
2. SVM trains on the full training set with C=1.0 in the "default" tier; the
   old 10k-subsample C=10 config is preserved as "SVM-adjusted" and flagged.
3. MANA error/missing predictions are excluded (and counted), not coerced to
   unspoofed.
4. Winsorization/imputation are config switches so paper text and code match.
5. Per-epoch series are persisted (parquet) for sequence models.

MANA baseline: unchanged; reuse the existing mana_predictions.csv.

## Journal-version findings and workflow (v2.1)

Internal audit + experiments that reshape the paper (all reproducible from
the committed feature CSV; no pcap re-extraction needed):

1. **MARSIM artifact (critical).** Every SNR aggregate carries a simulator
   fingerprint: the spoofer signal is generated with a ~0.07 dB constant SNR
   offset, making `mean_snr_median` a single-feature PERFECT oracle
   (AUC = 1.0000) on scenarios A1 and A2, and 0.97 on A3 at shift_speed = 0.
   Evidence: `python scripts/audit_dataset.py`. Consequence: all journal
   experiments run artifact-free (`preprocessing.ablate_families`), which
   costs ~nothing in-distribution (F1 0.998 remains) and is the only honest
   basis for real-data transfer. Zero-intensity attacks remain legitimately
   detectable via inter-receiver baseline collapse (capture physics, ~4.6 m
   -> ~3.3 m), which the audit confirms is NOT an artifact.
2. **Two generalization cliffs.** `run_generalization_sweep.py`:
   interpolation (leave-one-speed-out) is near-perfect for ML (recall >= 0.98
   except exactly speed 0), vindicating the CIMSS claim in-distribution. But
   EXTRAPOLATION below the training intensity range collapses boosted trees
   (XGB recall 0.003 at speed 0 when trained on speed > 16) while MANA stays
   at 0.85 and RF degrades gracefully. Upward extrapolation is free.
3. **Threshold repair.** The collapse is an operating-point artifact (test
   AUC stays 0.98+). A negatives-anchored threshold (quantile of validation
   NEGATIVE scores at a target FPR) recovers RF to recall 0.971 / FPR 1.1%
   (F1 0.980, within 0.002 of the test-oracle ceiling) with no test labels.
   The textbook val-max-F1 policy is WORSE than default 0.5 under this shift.
4. **Statistics.** Precision-at-prevalence now carries Clopper-Pearson
   intervals (a zero-FP count no longer yields precision == 1.0 at 1e-4
   prevalence); MANA comparisons are denominator-symmetric; k-fold stds are
   labeled descriptive-only.
5. **Code/text mismatches fixed.** Correlation filter now implements the
   algorithm the paper text describes (`mspoof.preprocess`); SVM avoids the
   hidden Platt refit; run manifests record library versions; the SCS reader
   no longer mixes time bases in archives with bare NMEA lines.

### Journal workflow
```
python scripts/audit_dataset.py                                   # artifact screen
python scripts/run_generalization_sweep.py --stage all            # presets + LOSO + extrap + thresholds + figures
python scripts/train_classifiers.py --config config.yaml          # headline tables (artifact-free by default)
python scripts/inject_attacks.py --selftest                       # capture-semantics injector for real logs
```

### Real-data cautions (encode these in the experiment design)
* Injection must model CAPTURE (rx2 collapses onto rx1), not a naive common
  lat/lon shift, or the inter-receiver features are disabled by construction.
  `scripts/inject_attacks.py` implements this and supports mid-window onsets
  (MARSIM aggregates never contain partial-onset windows; time-to-detection
  experiments need onset-augmented windows).
* Learned inter-receiver statistics encode MARSIM's 4 m antenna baseline;
  center inter_rx features on each vessel's clean-cruise median before
  transfer, and discuss that this baseline is poisonable.
* MARSIM is straight-line constant-velocity; real cruises maneuver. Report
  full-cruise FPR honestly plus a steady-transit-segment control.

### Data note
`data/marsim_features.csv` (72 MB) should move to Zenodo (DOI, cited in the
paper) or git-lfs before submission. Regenerate it once with
`scripts/extract_features.py` on this code version and record the manifest:
the committed CSV predates the bearing fix and lacks the `_error` column, so
provenance must be re-established before journal experiments are final.
