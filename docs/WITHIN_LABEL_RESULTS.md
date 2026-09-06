# Within-label rationale shuffle control

This follow-up holds each donor rationale's class label constant while breaking its
sample-level correspondence. Both rationale channels move together, with zero fixed points.
The clean checkpoint is frozen for every intervention evaluation.

- Clean mean Macro-F1: **0.9333**
- Within-label shuffled mean Macro-F1: **0.9517**
- Mean change: **+0.0184**
- Individual paired-bootstrap intervals strictly above zero: **2/9**

| Shuffle seed | Model seed | Clean | Within-label shuffle | Delta | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|
| 13 | 13 | 0.9211 | 0.9418 | +0.0206 | [+0.0000, +0.0426] |
| 13 | 42 | 0.9382 | 0.9625 | +0.0243 | [+0.0037, +0.0474] |
| 13 | 97 | 0.9405 | 0.9661 | +0.0257 | [+0.0084, +0.0485] |
| 42 | 13 | 0.9211 | 0.9415 | +0.0204 | [-0.0003, +0.0434] |
| 42 | 42 | 0.9382 | 0.9584 | +0.0202 | [+0.0000, +0.0445] |
| 42 | 97 | 0.9405 | 0.9579 | +0.0174 | [+0.0000, +0.0393] |
| 97 | 13 | 0.9211 | 0.9334 | +0.0123 | [-0.0045, +0.0302] |
| 97 | 42 | 0.9382 | 0.9501 | +0.0119 | [-0.0087, +0.0364] |
| 97 | 97 | 0.9405 | 0.9533 | +0.0129 | [+0.0000, +0.0291] |

## Interpretation boundary

Within-label shuffling did not reduce mean Macro-F1 in this internal run. Therefore,
the earlier unrestricted-shuffle drop cannot be attributed to sample-level semantic
misalignment alone. This result does not show that rationales are generally useless:
the experiment uses internal splits, one released architecture path, and generated
rationales whose label-associated signals remain intact under this control.

These are internal EARAM-style results, not an official-score reproduction.
