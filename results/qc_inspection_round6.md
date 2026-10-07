# QC inspection, round 6: answers to the reviewer's questions on the defect log (Nantes public dataset)

Date: 2026-10-05. **This is an AI-assisted inspection of the images, not an embryologist's reading.** Cell counts in fragmented
embryos are approximate (±1–2 cells). Nothing in the repository was edited apart from this file. The contact sheets
are in the session scratchpad and are not committed.

## Method

- Images: the reference plane `embryo_dataset/<video>/..._RUN<k>.jpeg` (500×500). `RUN<k>` is image index k. I checked
  that the run numbers are contiguous; the only gaps are single missing files (CAV074-9 RUN464, RM126-5/-8 RUN283,
  RC545-2-8 RUN78, PMDPI029-1-8 RUN1). Other focal planes (F-45…F45) were looked at only for the empty-well and
  EH315 claims.
- Released pairing: image k ↔ annotation frame k. Fitted pairing: j = round(a·k + b), with a and b taken from
  `results/frame_timing/alignment_oof.csv`, so the image for an annotated onset j is k = round((j − b)/a). For the
  two signature videos (0.2-h time file) I also checked the mechanism pairing j ≈ 1.25·k.
- Times: the `time` column of `<video>_timeElapsed.csv` at frame index k. All the times quoted below come from the
  time file, which covers every index cited. **One exception:** in the signature videos GC340-10 and RC545-2-8 the
  time file is on a 0.2-h grid, so the time at index k is the annotation clock, not the true acquisition time of
  image k.
- Phase onsets: the start frame of each row in `<video>_phases.csv`.
- Empty wells: found by eye on consecutive frames and checked in five focal planes. I also used a simple high-pass
  statistic: mean |I − Gσ=4 ∗ I| over the central 150×150 px of a 250-px resize. It falls 6–10× at removal and stays
  low through the last image in all four cases. Note: the log's own structure score (`scan_featureless_frames.structure`,
  thr 22.4) finds **no** featureless frame in LP284-3 or RM126-6, and finds PMDPI029-1-8 and RM126-8 only from
  RUN483 and RUN471, 14 and 5 frames late. The rim and debris of an empty well keep its Laplacian variance high.

---

## A. Strongest unlogged disagreements

### AMT360-1-9: labels correct under the released pairing; heavy fragmentation; embryo removed at 116.9 h
- **Verdict:** not a timeline mismatch. The fitted pairing (a = 1.135, b = 53) puts tPNf and t2 at negative image
  indices and t4 on a one-cell image (RUN26). The released pairing matches the images.
- **Evidence:**
  - The recording starts at 19.1 h.
  - One cell through RUN35 (27.6 h). Cleavage starts at RUN36 (27.9 h). At RUN38–41 (annotated t2 = 38, 28.4 h) there
    are two large blastomeres in a cloud of fragments.
  - ~3–4 cells from RUN82 (annotated t4, 39.4 h).
  - After that, a fragmented many-cell mass. No compacted morula is distinct.
  - Cavity first visible at ~RUN364–367 (110–111 h), against annotated tSB at RUN351 (107.0 h), which is ~3–4 h early.
  - A full blastocyst at RUN373–389 is still labelled tSB, because tB is never annotated.
  - The well is empty, zona included, from **RUN390 (116.9 h)** to the last image, RUN394. tSB is annotated to frame 395.
- **Why the classifier disagreed:** heavy fragmentation makes cleavage-stage counts unreadable. The last 5 frames are
  an empty well, and the late blastocyst carries a tSB label.
- **Confidence:** high (not a timeline mismatch, removal frame); medium (stage reading).
- **Proposed entry:**
```json
"AMT360-1-9": {"defects": ["embryo_removed", "heavy_fragmentation"], "tier": "B", "cutoff_h": 116.9,
 "source": "inspection (round 6)",
 "note": "One cell to RUN35 (27.6 h); two blastomeres in heavy fragments at annotated t2 (RUN38, 28.4 h), index pairing correct. Cavity from ~111 h, tB never annotated; well empty from RUN390 (116.9 h)."}
```

### CAV074-9: labels correct early; slow, arrested embryo held at t4 to the end (atypical biology)
- **Verdict:** not a timeline mismatch. The fit is at the grid edge (a = 1.60, b = 27) and puts annotated t3 (227) on
  RUN125, a pronuclear zygote, and t4 (300) on RUN171, two cells. Under the released pairing the onsets are exact.
- **Evidence:**
  - One cell to RUN152 (39.2 h). Two cells at RUN154 (= annotated t2, 39.7 h).
  - The larger blastomere divides at RUN224–227 (= annotated t3, 58.1 h).
  - From ~RUN230–300 one large blastomere and 3–4 smaller cells with fragments; annotated t4 is at 300 (76.4 h).
  - From RUN345 to ~460 (88–117 h) one large blastomere and ~4–6 smaller ones. The count is uncertain, but the embryo
    is never beyond a cleavage stage.
  - Degenerating, granular mass from ~RUN500–520 (127–132 h) to RUN556 (140.3 h).
  - The annotation ends with t4 from 300 to 556.
- **Why the classifier disagreed:** the embryo is very slow, keeps one oversized blastomere, and degenerates. Late
  frames look like t5–t6 or a degenerating mass while the label stays t4. That is at most one to two stages late, and
  only in an arrested embryo.
- **Confidence:** medium.
- **Proposed entry:** none required. If it is logged, no existing category fits; a new tier-C category would be needed:
```json
"CAV074-9": {"defects": ["developmental_arrest"], "tier": "C", "source": "inspection (round 6)",
 "note": "t2 (RUN154, 39.7 h) and t3 (RUN227, 58.1 h) match the index pairing exactly. Arrests with one large and ~4-6 small blastomeres, degenerates from ~127 h; t4 label held to 140 h. NEW category."}
```

### OJ319-10: early onsets correct; annotation stops at t6 although the embryo reaches a blastocyst
- **Verdict:** a label–image mismatch of a different kind: the last annotated phase is extended to the end. Not a
  timeline mismatch. The fit (a = 1.43, b = −41) makes the early onsets no better than the identity pairing.
- **Evidence:**
  - PN visible to ~RUN93. Two cells at RUN95 (= annotated t2, 24.3 h). Three cells at RUN103 (= t3, 26.3 h).
  - Three cells until RUN146. At RUN151 (annotated t4, 38.3 h) one blastomere has broken into many small cells and
    fragments.
  - t5 at RUN202 (51.1 h) and t6 at RUN212 (53.6 h) fall on a mass of one large blastomere and many small cells.
  - From RUN290 a many-cell mass. **Cavity first visible at RUN399 (100.8 h).** Early blastocyst at RUN440–455
    (111–115 h); blastocyst at RUN465 (117.2 h).
  - Annotated t6 covers frames 212–466, so morula and blastocyst frames carry t6.
- **Why the classifier disagreed:** it correctly sees morula and blastocyst images that are labelled t6.
- **Confidence:** high (labels after ~101 h are wrong); low (exactly where the t6 label stops being true; the
  fragmentation hides the 8-cell transition).
- **Proposed entry:** cut the labels where they are certainly wrong. No existing category describes "annotation stops
  early, last phase extended". The mechanics are those of tier B (keep, cut). A stricter cut, for example ~75 h, would
  also drop the probably-unannotated t8 and t9+ frames.
```json
"OJ319-10": {"defects": ["annotation_truncated", "heavy_fragmentation"], "tier": "B", "cutoff_h": 100.8,
 "source": "inspection (round 6)",
 "note": "t2/t3 match the index pairing (RUN95/103). Annotation ends with t6 (from RUN212, 53.6 h) held to the last frame, but cavitation is visible from RUN399 (100.8 h). NEW category."}
```

### LA367-4: labels correct; embryo off-centre at the dim edge of the well
- **Verdict:** labels are correct to within a few hours under the released pairing. Not a timeline mismatch: the fit
  (a = 0.88, b = 65) puts annotated t2 on RUN56, a one-cell zygote.
- **Evidence:**
  - One cell to RUN111 (28.3 h). Cleavage into two blastomeres plus fragments at RUN113–114 (annotated t2 = 114,
    29.1 h).
  - ~4 cells at RUN164–171 (annotated t3/t4 = 167/171, 42.4/43.4 h).
  - Many cells through RUN436. Annotated tSB at RUN436 (110.5 h) shows no visible cavity yet. The cavity is visible at
    ~RUN450–460 (114–116.5 h), so tSB is ~4–6 h early.
  - Blastocyst at RUN507 (= tB, 128.6 h). Expanded at RUN511–540 (tEB = 511).
  - The expanding or hatching blastocyst runs off the image border at RUN565.
  - Throughout, the embryo sits in the lower-right, vignetted part of the well.
- **Why the classifier disagreed:** probably the off-centre, dim position and the fragmented first cleavage. tSB is
  also somewhat early.
- **Confidence:** medium.
- **Proposed entry:** none.

### OJ319-7: labels correct; low-contrast, granular blastomeres
- **Verdict:** labels correct under the released pairing. The fit (a = 0.80, b = 80) puts annotated t2 on RUN29, a
  one-cell zygote.
- **Evidence:**
  - One cell to RUN101 (25.8 h). Two cells at RUN103 (= t2, 26.3 h).
  - 3–4 cells at RUN151–157 (= t3/t4, 38.3/39.8 h).
  - ~5–6 cells at RUN165–180. ~8 cells at RUN208–240 (t8 = 210, 53.1 h).
  - Compaction at ~RUN383 (= tM, 96.8 h). Cavity at RUN409 (= tB, 103.3 h). Expanded at RUN431 (= tEB, 108.8 h).
  - Cell boundaries are faint at 26–55 h (granular cytoplasm, slightly soft focus). This would hurt a per-frame
    classifier, not the labels.
- **Confidence:** medium.
- **Proposed entry:** none.

## B. RM126-5: not a timeline mismatch; direct cleavage, then arrest

- **Verdict:** not a timeline mismatch. RM126-5 has a 0.25-h time file with 556 rows for 554 images, so the file
  signature is absent, unlike its siblings RM126-6 (592 rows) and RM126-10 (702 rows). The fit sits at the grid edge
  (a = 1.595, b = −70) and puts annotated t2 (138) on RUN130, a one-cell embryo, and t5 (244) on RUN197, three cells.
  Under the released pairing the onsets match.
- **Evidence:**
  - One cell to RUN135 (35.0 h). Direct cleavage at RUN136–138: three cells plus fragments at RUN138 (= annotated t2,
    35.8 h). That is 1 → 3 within ~0.75 h.
  - 3 cells with fragments to ~RUN218. A fourth cell at ~RUN222 (= t3, 56.8 h). ~5 cells at RUN242–250 (= t4/t5,
    61.8/62.3 h).
  - ~7–8 cells at RUN370–395 (t6/t7 = 383/385, 97.9/98.4 h).
  - No blastocyst. A compacting, degenerating mass from ~RUN460 to 555 (117–141 h), labelled t7 to the end.
- **Confidence:** high (not a timeline mismatch); medium (cell counts).
- **Proposed entry:** keep tier C and add `abrupt_cleavage`.
```json
"RM126-5": {"defects": ["abrupt_cleavage", "heavy_fragmentation"], "tier": "C", "source": "inspection (round 6)",
 "note": "One cell to RUN135 (35.0 h), three cells plus fragments at RUN138 (annotated t2, 35.8 h); index pairing fits, the 1.595 fit puts t2 on a one-cell image. Arrests near 8 cells."}
```

## C. Mismatches decided by inspection with empty notes

### GC340-10: timeline mismatch confirmed
- **Observation:**
  - At annotated tPNf (frame 109) the released image RUN109 already shows ~4 blastomeres with fragments.
  - The pronuclei are visible to RUN94 and the first cleavage is at RUN96. The fitted pairing (RUN96 for tPNf, RUN105
    for t2) and especially the 1.25× pairing (RUN87 for tPNf, pronuclei; RUN95 for t2, furrow; RUN99 for t4) match.
  - Late onsets agree too. Under the 1.25× pairing tM falls on RUN358 (compacted) and tB on RUN454 (first cavity).
    Under the released pairing tB and tEB (567/568) lie beyond the last image (554).
- **Confidence:** high.
- **Proposed note** (tier A, defects unchanged): "Pronuclei to RUN94, first cleavage RUN96; released image at
  annotated tPNf (RUN109) shows ~4 cells. Under 1.25x pairing tPNf/t2/tM/tB fall on RUN87/95/358/454: pronuclei,
  furrow, compaction, cavity."

### LP284-3: the reviewer is right. Not a timeline mismatch; embryo removed on day 3 (tier B)
- **Observation:** the released pairing is correct, and the well empties after RUN280.
  - 1 cell with pronuclei at annotated tPNf (RUN98, 25.3 h). The pronuclei stay visible to ~RUN108, so the annotated
    tPNf is ~10 frames early.
  - First cleavage at RUN109–110 (28.1–28.3 h). Annotated t2 (RUN120, 30.8 h) shows 2 blastomeres plus fragments,
    possibly a third cell; t2 is ~10 frames late.
  - ~3 cells with fragments to RUN161. ~4–5 cells at annotated t4 (RUN164, 41.8 h).
  - ~6–7 cells at t6 (RUN231, 58.6 h). ~8 cells at t8 (RUN247, 62.6 h).
  - The fitted pairing (RUN98/118/158/218/232) is no better: it shows ~3 cells at t4 and ~6 at t8.
  - The earlier session note "index pairing ~6 cells at annotated t4" is **not reproduced**.
  - 8-cell embryo present at RUN280 (70.9 h). **The well, zona included, is empty from RUN281 (71.2 h)**, not RUN283
    as the reviewer said; RUN281 and RUN282 are already empty. It stays empty to RUN460. This holds in all five
    focal planes checked.
  - Annotation (t8) runs to frame 298, so frames 281–298 label an empty well. Images 299–460 have neither time nor
    annotation rows.
- **Confidence:** high.
- **Proposed entry:**
```json
"LP284-3": {"defects": ["embryo_removed"], "tier": "B", "cutoff_h": 71.2, "source": "inspection (round 6)",
 "note": "Index pairing correct: 1 cell at tPNf, 2 cells plus fragments at t2, ~4 at t4, ~8 at t8 (RUN247). 8-cell embryo at RUN280 (70.9 h); well and zona gone from RUN281 (71.2 h) to the end."}
```

### PC758-2: the timeline-mismatch decision is not supported; labels late at t5–t8
- **Observation:**
  - At annotated tPNf (RUN70, 23.8 h) the released image shows a zygote with pronuclei. At annotated t2 (RUN86,
    29.1 h) it shows one cell without pronuclei; the first cleavage is at RUN91 (30.8 h).
  - The fitted pairing (a = 1.39, b = −57) puts tPNf on a **2-cell** image (RUN91) and t2 on a 3–4-cell image
    (RUN103), so early on it is worse than the identity pairing.
  - The earlier note "index pairing 3–4 cells at annotated t2" is not reproduced: RUN86 shows one cell.
  - Later the annotation lags the images under both pairings:
    - ~5 cells by ~RUN135–150 (45–51 h) and ~8 cells by ~RUN172 (58.1 h).
    - Annotated t5/t6/t7/t8 fall at RUN175/178/188/190 (59.1–64.1 h), i.e. t5 ~8–12 h and t8 ~6 h late.
    - The fit only partly compensates: t5 → RUN167, ~6 cells.
  - The time file is on a 0.33-h grid with 210 rows for 208 images, so there is no 0.2-h signature.
- **Verdict:** not a clean clock mismatch. The labels lag the images from t5 onwards.
- **Confidence:** medium on "not a timeline mismatch"; low on the t3 reading (a possible third cell at RUN94–100 may
  be a large fragment).
- **Proposed entry:** drop `timeline_mismatch`. If the late lag is to be logged, it needs a new category, the mirror
  of `annotation_ahead`. Otherwise drop the video from the log.
```json
"PC758-2": {"defects": ["annotation_behind"], "tier": "A", "source": "inspection (round 6)",
 "note": "Index pairing fits tPNf and t2 (one cell at RUN86, cleavage RUN91); the fit puts tPNf on 2 cells. ~8 cells by RUN172 (58 h) but t5-t8 annotated RUN175-190: labels 6-12 h late. NEW category."}
```

### RC545-2-8: timeline mismatch confirmed, plus direct cleavage
- **Observation:**
  - Pronuclei visible to RUN144. At RUN146 the zygote cleaves directly into a fragmented many-cell mass.
  - Under the released pairing, annotated tPNf, t2 and t4 (173/184/189) all fall on that mass.
  - Under the 1.25× pairing they fall on RUN138 (pronuclei), RUN147 (just cleaved) and RUN151, which matches.
  - tM and tSB (615/635) lie beyond the last image (559). Under the 1.25× pairing they map to RUN492/508, a
    compacting mass, with cavitation visible from ~RUN520–540.
  - The fitted grid-edge slope 1.595 overshoots: it puts t2 and t4 on one-cell images (RUN134/137). The 1.25×
    pairing fits better than the fit.
- **Confidence:** high (mismatch); medium (late onsets).
- **Proposed entry:** tier A. Defects `["abrupt_cleavage", "timeline_mismatch"]`. Note: "Pronuclei to RUN144, direct
  cleavage to a fragmented mass at RUN146; released tPNf/t2/t4 (173/184/189) all on that mass, 1.25x pairing gives
  RUN138/147/151. tM/tSB lie beyond the last image."

## D. EH315-3 and EH315-8: annotated divisions and blastocyst not visible; abrupt cleavage, not a timeline mismatch

Both videos have 541 images and 555 time rows on a 0.25-h grid. The signature is set only because the annotations run
to frame 553, 12 frames beyond the last image. Neither fitted pairing (a = 0.625 and 0.61) improves the visible
agreement.

### EH315-8: the reviewer's claim is verified
- **Evidence:**
  - One cell with many cytoplasmic inclusions or vacuoles from RUN60 to RUN171 (43.7 h). This holds in all seven
    focal planes at RUN165. It covers annotated t2 (RUN114, 29.4 h), t3 (RUN160, 40.9 h), t4 (RUN170, 43.5 h) and
    t5 (RUN176) territory.
  - Abrupt cleavage at **RUN172–173 (44.1–44.3 h)** into one large blastomere plus a mass of small cells and fragments.
    By RUN174–176 that pattern is established.
  - The large blastomere and the fragmented mass persist to the last image, RUN541 (136.8 h).
  - No blastocoel at annotated tSB (RUN437, 110.7 h), tB (RUN466, 118.1 h) or tEB (RUN476, 120.6 h), in any focal plane
    at RUN476. At most some irregular clearing appears at RUN530–541.
  - The fitted pairing puts t2 on RUN159 (still one cell) and tSB beyond the last image.
- **Verdict:** annotated divisions not visible / abrupt cleavage; the annotation runs ahead of the images, and the
  blastocyst stages are annotated without a blastocyst. Not a timeline mismatch.
- **Confidence:** high.
- **Proposed entry:**
```json
"EH315-8": {"defects": ["abrupt_cleavage", "annotation_ahead", "heavy_fragmentation"], "tier": "A",
 "source": "inspection (round 6)",
 "note": "One cell with inclusions through annotated t2/t3/t4 (to RUN171, 43.7 h); abrupt cleavage to one large blastomere plus fragments at RUN172-173. No blastocoel at annotated tSB/tB/tEB in any plane."}
```

### EH315-3: the reviewer's claim (abrupt cleavage) is verified
- **Evidence:**
  - One cell to RUN108 (27.8 h). A furrow at RUN109 (28.1 h), then fragmentation into a many-cell mass by RUN110–113.
  - At annotated tPNf (RUN113, 29.1 h) and t2 (RUN121, 31.1 h) the images already show many cells and fragments.
  - A compact many-cell mass from ~RUN150 to the end, with no blastocoel at tSB (RUN429, 108.7 h), tB (RUN472,
    119.6 h), tEB (RUN476) or RUN500, in seven focal planes.
  - The fitted pairing puts tPNf and t2 on RUN56 and RUN69 (a zygote) and t3 on RUN150 (a many-cell mass). The images
    do not support it.
- **Verdict:** abrupt cleavage. The blastocyst stages are annotated on a non-cavitated mass. Not a timeline mismatch.
- **Confidence:** high (abrupt cleavage, not timeline); medium (absence of cavity, since a small cavity in a compact
  mass could be missed).
- **Proposed entry:** stays tier A, `timeline_mismatch` replaced by `annotation_ahead`.
```json
"EH315-3": {"defects": ["abrupt_cleavage", "annotation_ahead", "heavy_fragmentation"], "tier": "A",
 "source": "inspection (round 6)",
 "note": "One cell to RUN108 (27.8 h), direct cleavage to a fragmented many-cell mass by RUN110-113 (annotated tPNf). Compact mass to the end; no blastocoel at annotated tSB/tB/tEB in any plane."}
```
If `CONFIRMED_BY_EYE` is kept, EH315-3 and EH315-8 should be `False` (not a timeline mismatch), as should LP284-3 and
PC758-2.

## E. Late cut-offs

All indices below are covered by the video's time file; the times are read from it.

| video | last frame with embryo | first empty-well image | current cutoff_h | proposed cutoff_h |
|---|---|---|---|---|
| PMDPI029-1-8 | RUN468, 119.6 h (expanded blastocyst) | **RUN469, 119.8 h** | 123.4 | **119.8** |
| RM126-8 | RUN465, 118.7 h (hatching blastocyst) | **RUN466, 119.0 h** | 120.2 | **119.0** |

- **PMDPI029-1-8:** the reviewer said RUN470 (120.1 h). The well is already empty one frame earlier, at RUN469
  (119.8 h), in all five planes, and it stays empty to RUN560. The zona disappears with the embryo.
  - Proposed note: "Expanded blastocyst to RUN468 (119.6 h); well and zona empty from RUN469 (119.8 h) to the end."
  - Confidence: high.
- **RM126-8:** the reviewer's RUN466 (119.0 h) is confirmed in all five planes, and the well stays empty to RUN555.
  The whole embryo, zona included, vanishes between two frames 0.3 h apart. That points to removal, not to hatching out
  of view, because hatching leaves an empty zona behind.
  - Proposed note: "Hatching blastocyst to RUN465 (118.7 h); well and zona empty from RUN466 (119.0 h) to the end;
    removal, not hatching out of view."
  - Confidence: high.
- Both current cut-offs were too late because the structure-score scan detects these wells late (see Method).

---

## Summary

| video | verdict | proposed tier | confidence |
|---|---|---|---|
| AMT360-1-9 | labels correct under the released pairing; heavy fragmentation; removed at 116.9 h (RUN390) | B (cutoff 116.9) | high / medium |
| CAV074-9 | labels correct early; arrested slow embryo, t4 held to the end | none (optional C, new category) | medium |
| OJ319-10 | annotation stops at t6 while the embryo cavitates (RUN399, 100.8 h) | B (cutoff 100.8, new category) | high / low on the cut point |
| LA367-4 | labels correct; off-centre dim embryo | none | medium |
| OJ319-7 | labels correct; low-contrast blastomeres | none | medium |
| RM126-5 | not a timeline mismatch; direct cleavage 1→3, arrest | C (add abrupt_cleavage) | high |
| GC340-10 | timeline mismatch confirmed (1.25× pairing matches) | A | high |
| LP284-3 | not a timeline mismatch; embryo removed at RUN281 (71.2 h) | B (cutoff 71.2) | high |
| PC758-2 | not a timeline mismatch; t5–t8 labels late by 6–12 h | A with a new category, or drop | medium / low |
| RC545-2-8 | timeline mismatch confirmed, plus direct cleavage | A | high |
| EH315-3 | abrupt cleavage; blastocyst annotated without a cavity; not a timeline mismatch | A (annotation_ahead) | high / medium |
| EH315-8 | one cell through annotated t2–t4, abrupt cleavage at RUN172–173, no blastocyst; not a timeline mismatch | A (annotation_ahead) | high |
| PMDPI029-1-8 | first empty image RUN469 (119.8 h) | B (cutoff 119.8) | high |
| RM126-8 | first empty image RUN466 (119.0 h) | B (cutoff 119.0) | high |
