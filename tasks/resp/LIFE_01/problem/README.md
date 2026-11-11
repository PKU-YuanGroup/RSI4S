# LIFE_01: Atezolizumab binding-bin prediction

## 1. Scientific problem

Predict the experimental binding bin of an Atezolizumab heavy-chain variant
against PD-L1. RH01 (class 0) has faster dissociation, RH02 (class 1) is similar
to wild type, and RH03 (class 2) has slower dissociation. These are ordered
experimental categories, not numerical KD measurements.

## 2. Task and research protocol

This task covers the Atezolizumab prediction experiment.

Train and test membership and row order are fixed by the host.
The initial labels are floor(N/10), with floor(N/2) additional unique queries
shared across all Task/Meta generations. One query reveals a sequence's class
and confidence weight together. Task proposes queries; Meta approves them.
There is no official validation set. Task registers exact acquired sample IDs
for internal validation and may revise or clear the selection during research. Always train only on records
with acquired=1 AND role='train'. Test inputs and labels remain private during
research; the selected frozen model performs Test inference after research ends.

## 3. Inputs and submission

Read [data_description.md](data_description.md). Submit a self-contained model.py
with fit(task, records) and predict(model, records), following the host contract.
Return one {"probabilities": [p_RH01, p_RH02, p_RH03]} dictionary per input.
The probabilities must sum to one and lie in [0,1]. The host derives the class
by argmax, choosing the lowest class for a tie.

## 4. Metrics and interpretation

The primary metric is unweighted multiclass Matthews correlation coefficient
(MCC). The secondary metric is ROC-AUC for RH03 versus the other two bins,
using p_RH03. The confidence weight is optional training information and does
not weight either metric.

Terminal Test scoring is a separate organizer operation after frozen inference.
It cannot influence research, label acquisition or final model selection.
