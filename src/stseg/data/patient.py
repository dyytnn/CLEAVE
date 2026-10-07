"""Canonical patient identifier for the Nantes video names (CLEAVE grouping unit).

Nantes video names are ``<code>[-<cycle>]-<embryo>`` with three irregular forms in the released
archive (checked against all 704 released names):

===================  ==========  =======  ==========  ============================================
name                 code        cycle    embryo      note
===================  ==========  =======  ==========  ============================================
``AA83-7``           AA83        --       7           the common form (649 of 704 names)
``PMDPI029-1-10``    PMDPI029    1        10          a second treatment cycle of the same couple
``GS490-_6``         GS490       --       _6          stray underscore before the embryo number
``MRA165-7T``        MRA165      --       7T          letter-suffixed embryo number
``RV146N2-6``        RV146N2     --       6           a letter+digit inside the code itself
===================  ==========  =======  ==========  ============================================

Two grouping levels follow from this, and they are NOT the same:

``cycle``
    strip the embryo token only (``PMDPI029-1-10`` -> ``PMDPI029-1``).  This is what
    ``scripts/make_nantes_split.py`` used to build ``nantes_grouped_v1``.  Two treatment cycles of
    the same couple are treated as two different groups, so sibling embryos across cycles can still
    straddle a partition.

``couple`` (default, and the CLEAVE unit)
    strip the embryo token and then the cycle token (``PMDPI029-1-10`` -> ``PMDPI029``).  This is
    the most conservative reading of the name and the one ``nantes_grouped_v2`` groups by.

The two differ for 43 of 704 released names and change the fold-0 leakage count (38 cycle-level
codes vs 39 couple-level codes over slightly different code sets); see
``scripts/audit_protocol_splits.py``, which is the single producer of every leakage number quoted in
the paper.  Nothing else in the repository may re-implement this parsing.
"""
from __future__ import annotations

import re

#: embryo token: ``-7``, ``-_6``, ``-7T``, ``_10``
_EMBRYO = re.compile(r"[-_]_?\d+[A-Za-z]?$")
#: treatment-cycle token, stripped only at ``couple`` level
_CYCLE = re.compile(r"-\d+$")

LEVELS = ("couple", "cycle")


def patient_of(video: str, level: str = "couple") -> str:
    """Grouping key of a Nantes video name.

    Raises ValueError on an unparseable name rather than silently returning the whole name, so a
    future naming convention cannot quietly disable grouping.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    stem = video.strip()
    trimmed = _EMBRYO.sub("", stem, count=1)
    if trimmed == stem or not trimmed:
        raise ValueError(f"unparseable Nantes video name (no embryo number): {video!r}")
    if level == "cycle":
        return trimmed
    code = _CYCLE.sub("", trimmed, count=1)
    return code or trimmed


def group_videos(videos, level: str = "couple") -> dict[str, list[str]]:
    """``{patient: [video, ...]}`` preserving input order within a group."""
    out: dict[str, list[str]] = {}
    for v in videos:
        out.setdefault(patient_of(v, level), []).append(v)
    return out


def leaking_patients(partition_of: dict[str, str], level: str = "couple") -> dict[str, list[str]]:
    """``{patient: sorted partitions}`` for every patient present in more than one partition."""
    seen: dict[str, set[str]] = {}
    for video, part in partition_of.items():
        seen.setdefault(patient_of(video, level), set()).add(part)
    return {p: sorted(s) for p, s in sorted(seen.items()) if len(s) > 1}


def exposed_test_videos(partition_of: dict[str, str], level: str = "couple") -> list[str]:
    """Test videos that have at least one sibling of the same patient in train.

    This is the quantity that matters for a reported test score; the count of leaking patients is a
    weaker summary of the same fact.
    """
    train = {patient_of(v, level) for v, p in partition_of.items() if p == "train"}
    return sorted(v for v, p in partition_of.items() if p == "test" and patient_of(v, level) in train)
