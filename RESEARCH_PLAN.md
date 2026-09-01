# Research Plan

## Working title

**Stress-Testing Rationale-Augmented Fake News Detection under Corrupted Rationales**

## Motivation

EARAM uses two LVLM-generated analyses to augment multimodal fake-news detection. This creates a dependency: useful analyses can improve a task-specific detector, but incorrect or irrelevant analyses may also become a new failure channel. The project measures that channel under controlled corruptions before proposing a lightweight mitigation.

## Questions

1. Which rationale failures cause the largest EARAM performance drop?
2. Does damage scale monotonically with the proportion of corrupted rationales?
3. Can a transparent reliability filter recover performance without retraining the LVLM?
4. Does filtering improve classification calibration as well as Macro-F1?

## Hypotheses

- H1: conclusion flips and contradictions will be more harmful than evidence deletion at equal corruption rates.
- H2: replacing rationales with analyses from unrelated samples will expose stronger dependence on rationale relevance.
- H3: filtering will help at moderate corruption rates but may hurt clean performance through false rejection.
- H4: corruption of both analysis channels will be more harmful than corruption of one channel.

## Experimental design

### Independent variables

- Perturbation: evidence deletion, irrelevant rationale, contradiction, unsupported claim, conclusion flip.
- Target rate: 0.25, 0.50, 0.75, 1.00.
- Corrupted channel: analysis 1, analysis 2, or both. The MVP currently samples both independently; channel-specific targeting is the next implementation item.
- Filter: none, peer replacement, strict drop.

### Controls

- Reuse EARAM's original dataset split, preprocessing, model hyperparameters, and random seeds.
- Change only the two rationale files between conditions.
- Run at least three seeds and report mean and standard deviation.
- Tune the reliability threshold on development corruptions only; freeze it before test evaluation.

### Dependent variables

- Accuracy and Macro-F1.
- Per-class precision, recall, and F1.
- Expected Calibration Error if class probabilities are exported.
- Robustness AUC: area under Macro-F1 versus corruption-rate curve.
- Filter precision/recall against synthetically corrupted rationale labels.

## Minimum result table

| Perturbation | Rate | Clean F1 | Corrupted F1 | Filtered F1 | Recovery |
|---|---:|---:|---:|---:|---:|
| evidence deletion | 0.25 |  |  |  |  |
| irrelevant | 0.25 |  |  |  |  |
| contradiction | 0.25 |  |  |  |  |
| unsupported claim | 0.25 |  |  |  |  |
| conclusion flip | 0.25 |  |  |  |  |

Recovery is defined as:

\[
\text{Recovery}=\frac{F1_{filtered}-F1_{corrupted}}{F1_{clean}-F1_{corrupted}}.
\]

Report it only when the denominator is positive.

## Qualitative analysis

For at least three failures, show:

1. news caption and a short description of the image;
2. clean rationale;
3. corrupted rationale;
4. original and changed prediction;
5. reliability components and filter decision;
6. the first rationale segment that introduces the harmful claim.

## Claims boundary

- Synthetic perturbations establish controlled robustness, not prevalence of these failures in real LVLM outputs.
- The lexical filter is a baseline, not a factual verifier.
- Empty or duplicated rationales may create distribution shift; report this as a limitation.
- Full EARAM numbers require its original images, captions, labels, CLIP checkpoint, and training protocol.
- Do not present text-level reliability-score changes as fake-news detection improvements.

## Deliverables

- Reproducible corruption/filtering toolkit.
- EARAM adapter and experiment manifest.
- Robustness curves and result table.
- Three-case error analysis.
- A concise four-page report and a public code repository after results are verified.
