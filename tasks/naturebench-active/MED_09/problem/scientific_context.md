# COVID-19 Hospitalization Trend Forecasting

## 1. Scientific Problem

Accurate short-term forecasting of disease hospitalization trends is a critical public health challenge. During an ongoing pandemic, decision-makers need reliable guidance on whether hospitalizations will increase, stabilize, or decrease in the coming weeks in order to allocate healthcare resources, time policy interventions, and communicate risk to the public. The challenge is compounded by the heterogeneous and rapidly evolving nature of the relevant data: numerical time-series of case counts, hospitalization rates, and vaccination uptake must be interpreted alongside text-based public health policies, genomic surveillance reports of emerging variants, and static demographic and healthcare system characteristics of each region.

Traditional statistical and mechanistic forecasting models handle numerical inputs well but struggle to incorporate text-format or rapidly changing qualitative information, limiting their ability to respond to events such as policy changes or the emergence of new variants. This task asks you to build a forecasting system capable of processing multi-modal pandemic data for all 50 US states and predicting ordinal hospitalization trend categories for both 1-week and 3-week ahead horizons.

## 2. Task Objective

Given multi-modal COVID-19 state-level data for a specific week, predict the Hospitalization Trend Category (HTC) for that state 1 week ahead (t1) and 3 weeks ahead (t3).

The HTC is a 5-class ordinal classification:
- `SUBSTANTIAL INCREASING`
- `MODERATE INCREASING`
- `STABLE`
- `MODERATE DECREASING`
- `SUBSTANTIAL DECREASING`

Additionally, to compute several evaluation metrics the model must output a **probability distribution** over the 5 classes (not just the argmax prediction).

*   **Input**: For each test sample (state-week pair), the following multi-modal features are available:
    - Epidemiological time-series sequences: recent weekly hospitalization rates per 100K, smoothed hospitalization rates, reported case rates, vaccination rates (partial, complete, booster), absolute changes in hospitalization, and recent trend category sequences
    - AI-generated narrative text describing recent hospitalization and case trends
    - Text prompt combining vaccination and epidemiological trend narratives
    - Static state demographics: population, age group fractions, racial/ethnic composition
    - Healthcare system performance metrics: coverage rates, spending rankings, vulnerability indices
    - Political affiliation (2020 election results)
    - Text description of state demographics and healthcare ranking
    - Genomic surveillance: dominant variant name, virological characteristics (transmission, immunity escape, severity), AI-generated variant proportion trend narrative
*   **Output**: For each test sample, a probability distribution over 5 HTC classes for **both** the 1-week-ahead prediction (t1) and the 3-week-ahead prediction (t3).

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Accuracy
*   **Definition**: Fraction of test samples where the predicted class (argmax of the predicted distribution) matches the ground truth HTC label. Higher is better. Reported separately for 1-week (t1) and 3-week (t3) predictions.

### Other Metrics
*   **MSE**: Mean squared error computed on class ordinal values (mapping: SUBSTANTIAL DECREASING=0, MODERATE DECREASING=1, STABLE=2, MODERATE INCREASING=3, SUBSTANTIAL INCREASING=4). Compares the predicted class (argmax) against the ground truth class. Lower is better.
*   **WMSE**: Weighted MSE that incorporates the full predicted probability distribution. For each sample, the weighted squared error is summed over all classes: $\text{WMSE} = \frac{1}{N}\sum_i \sum_k P(\hat{y}_i = k)(k - \tilde{y}_i)^2$ where $k$ is the numeric class value and $\tilde{y}_i$ is the ground truth numeric value. Lower is better.
*   **BrierScore**: Mean squared difference between the predicted probability distribution and the one-hot ground truth: $\text{BS} = \frac{1}{N}\sum_i \sum_k (P(\hat{y}_i = k) - o_k^{(i)})^2$. Lower is better.
*   **RPS**: Ranked Probability Score, computed over cumulative probability distributions: $\text{RPS} = \frac{1}{N}\sum_i \sum_k (q(\hat{y}_i = k) - \bar{q}_k^{(i)})^2$ where $q$ is the predicted cumulative probability and $\bar{q}$ is the ground truth cumulative (step function at the true class). Lower is better.

All five metrics are computed independently for t1 (1-week) and t3 (3-week) predictions, yielding 10 scores per instance.
