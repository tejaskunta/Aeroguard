# A3 — independent near-duplicate check with ImageNet embeddings

Model: ResNet-50 `IMAGENET1K_V2` (frozen ImageNet weights), 2048-d GAP embedding,
256x256 RGB input, cosine similarity, exact search over the
3907 v2 **train** images. 32x32 RGB MSE uses the same code path as
`src/analysis/make_mse_sensitivity.py`.

## Summary

| Quantity | Value |
|---|---:|
| Eval images checked | 370 |
| Train images searched | 3907 |
| Top-1 pairs with MSE > 900 (frozen filter would not catch) | 258 |
| Top-1 pairs with MSE <= 900 | 112 |
| Highest cosine among MSE-uncaught pairs | 0.984902 |
| MSE-uncaught pairs at cosine >= 0.95 | 21 |

### The 30 eval images removed during split construction

- Recovered from the v1 store: **30/30**.
- pHash-proposed partner still in the v2 train split: **30/30**.
- Minimum cosine to the pHash-proposed partner: **0.534297**.

### The 99 residual pHash candidates (declared texture false positives)

- Embedding cosine to own top-1 v2-train image, min / median / max: [0.722369, 0.931714, 0.986905]
- Flagged at the leak-derived threshold: **99/99**
- Embedding top-1 equals the pHash-proposed train image: **1/99**

Montage of the top 50 MSE-uncaught pairs by cosine: `embedding_top50_montage.jpg`.

## Conclusion

### 1. The embedding does **not** validate the "texture false positive" verdict

The declared rule was: a survivor is *flagged* if its cosine to its pHash-proposed
train partner is >= T, where T is derived only from known leaks. That rule is
**vacuous**, not confirmatory:

| Population | n | cosine to own top-1 v2-train image (min / q25 / median / q75 / max) |
|---|---:|---|
| Known leaks (removed) | 30 | 0.772 / 0.900 / **0.939** / 0.970 / 0.984 |
| Declared survivors | 99 | 0.722 / 0.887 / **0.932** / 0.959 / 0.987 |

The two distributions are essentially identical. At the leak-derived threshold
T = 0.5343 (the minimum cosine from a known leak to its pHash partner — one leak,
`Jan_Jan_Afternoon_0_lwir_25.jpg`, has a weak pHash partner at 0.534) **100% of
both populations are flagged** (30/30 and 99/99). Any T that keeps all leaks also
keeps all survivors:

| Threshold T | Source | Known leaks retained | Survivors flagged |
|---|---|---|---|
| 0.5343 | min leak→pHash-partner | 30/30 | 99/99 |
| 0.7722 | min leak→own-NN | 30/30 | 97/99 |
| 0.7998 | p10 leak→own-NN | 27/30 | 96/99 |
| 0.8996 | q25 leak→own-NN | 23/30 | 67/99 |

There is **no cosine threshold that separates the 30 known leaks from the 99
declared survivors**. The embedding method therefore cannot corroborate the
"terrain texture false positive" claim; the claim still rests on the same
pHash + MSE features that built the filter. (The embedding does agree the
survivors are *not* the pHash-proposed partner: top-1 matches the pHash choice
for only 1/99.)

### 2. The top-50 MSE-uncaught pairs are dominated by adjacent video frames

Of the 258 eval images whose top-1 train neighbour has MSE > 900 (invisible to
the filter), the 50 highest-cosine pairs are:
- adjacent frames of the same source sequence for **40/50** (|source-index delta| <= 5);
- pHash-candidate survivors for 21/50, but the embedding agrees with the pHash
  partner in **0/50**.

So the strong embedding matches track the temporal structure of the source
videos, not missed leaks. Montage: `embedding_top50_montage.jpg`.

### 3. Negative control: the `ds0_/ds1_` source index is **not** a scene identifier

Iranian-mine `ds0_1234`-style indices repeat across splits (83 eval images sit on
an index that also appears in train). A naive "same index = same scene" claim
would imply leakage, so it was tested against a null. It is false:

| Comparison (pHash min-Hamming, 8 dihedral orientations) | ds0 | ds1 |
|---|---:|---:|
| Cross-split **same-index** pairs | mean 13.8, 7.6% <=8 bits | mean 16.4, 0% <=8 bits |
| **Random same-source** train pairs | mean 14.8, 6.4% <=8 bits | mean 17.7, 1.0% <=8 bits |

Same-index cross-split pairs are **not** more similar than random same-source
pairs (they are, if anything, marginally *less* similar). Within-train same-index
pairs are the most similar (mean 14.7, 19 pairs at Hamming<=2), i.e. the index
structure is confined to train where it does not leak. **No index-level leakage
claim is supported.** Review montage of the 12 closest same-index pairs:
`montages/same_index_closest12.jpg`.

### 4. Actionable finding — one probable missed leak that the gate dropped

`ds1_0054_JPG.rf.35bfc4f17ba71b62c6182a17f2aa2cf8.jpg` (**test**) has a
near-duplicate partner `ds0_0054_JPG.rf.81c89cf2bd517a6ce49dd44b8da4c59b.jpg`
(**train**) at **32×32 RGB MSE = 577.2** — comfortably inside the MSE<=900
criterion — but **pHash min-Hamming = 4**, i.e. just outside the Hamming<=2 gate
that the pipeline required before the MSE test was ever applied
(`scripts/phase2_second_filter.py` skips any candidate with `min_hamming > 2`).
The pre-removal candidate file `audit_v2/near_dup_pairs.csv` *does* record it
(`min_hamming = 4`), so the miss is a threshold artifact of the two-gate rule,
not missing data.

A second pair is borderline and needs a human eye:
`ds1_0111_jpg.rf.fc71e30a2fedd64cb231dfc8e3d157ec.jpg` (**test**) vs the three
`ds1_0111_png.rf.*` exports in **train**: MSE 758.2 / 844.9 / 893.4 (all < 900)
but Hamming 18 / 20 / 16 (high). The two features disagree.

Both are listed in `paper_results/audit_v2/embedding_actionable_pairs.csv`;
**no files were modified** — per the standing rule, label/data files are not
edited, only candidates are reported.

### 5. Accounting note: `split_v2/removed_leaks.csv` holds 17 of the 30 rows

`paper_results/split_v2/removed_leaks.csv` contains **17** rows, but **30**
eval images were removed. The 13 missing rows were removed in the *first* pass
(MSE<=100) and that pass did not append to the file that the second pass
overwrote (`RUN_LOG.md` line 18: "Removed 30 leak images in two passes
(13 then 17; 19 valid / 11 test)"). Splits of the recovered 30
(`embedding_removed_recheck.csv`): valid 15 / test 8, plus 7 whose `v1_split`
was train (removed from v1's train side). `audit_v2_final/audit_report.md`
already states the 13+17 split, so the 17-row CSV is an incomplete ledger, not a
discrepancy in the removal count itself.
