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
