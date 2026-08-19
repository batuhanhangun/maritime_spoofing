"""Feature-family ablation, driven by the MARSIM artifact audit.

Audit finding (scripts/audit_dataset.py): every SNR aggregate is contaminated
by a simulator fingerprint. The spoofer signal is generated with a ~0.07 dB
constant SNR offset (within-class std ~0.03 dB), making e.g. mean_snr_median
a SINGLE-FEATURE perfect oracle (AUC = 1.0000) on scenarios A1 and A2 and
0.97 even on A3 at shift_speed = 0 (zero drift). This is not attack physics:
a real attacker controls transmit power. GSA-family features (PDOP/VDOP/
fix_mode) are constant in simulation (dead weight).

Consequently the journal experiments run on the artifact-free set by default
(config: preprocessing.ablate_families). Removing SNR barely changes
in-distribution F1 (0.998 remains), i.e. the models do not NEED the oracle,
but interpretation and any real-data transfer must exclude it.
"""

FAMILY_PATTERNS = {
    # substring match on feature/column names
    'snr': ('snr',),
    'gsa': ('pdop', 'vdop', 'hdop_gsa', 'fix_mode'),
}


def family_of(col):
    for fam, pats in FAMILY_PATTERNS.items():
        if any(p in col for p in pats):
            return fam
    return None


def drop_families(columns, families=('snr', 'gsa')):
    """Return the subset of `columns` NOT belonging to the given families."""
    families = set(families or ())
    return [c for c in columns if family_of(c) not in families]
