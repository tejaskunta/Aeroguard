### T5. Grad-CAM metrics

| experiment | metric | value |
|---|---|---|
| E1 energy-in-box (n=244) | median energy share | 0.939 |
| E1 | mean energy share | 0.690 |
| E1 | median enrichment | 58.2x |
| E1 | peak inside rate | 0.730 |
| E2 deletion (n=182) | mean drop, mine masked | 0.739 |
| E2 | mean drop, background masked | 0.0015 |
| E6 head-to-head | YOLO enrichment median | 86.0x |
| E6 | MNv3 baseline enrichment median | 1.82x |
| E6 | paired diff (Y-M) 95% CI | [125.4, 170.0] |
| E6 | stage-2 crop enrichment median | 2.13x |

_Grad-CAM shows which regions influenced the prediction; it does not prove the prediction is correct (plan wording rule)._
