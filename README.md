# ✈️ British Airways — Data Science Job Simulation

> **Forage Data Science Virtual Experience Program**  
> Two end-to-end data science tasks completed using real British Airways datasets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Task 1 — Lounge Eligibility Modelling](#task-1--lounge-eligibility-modelling)
  - [Problem Statement](#problem-statement-task-1)
  - [Dataset](#dataset-task-1)
  - [Approach](#approach-task-1)
  - [Lookup Table](#lookup-table)
  - [Key Findings](#key-findings-task-1)
- [Task 2 — Customer Booking Prediction](#task-2--customer-booking-prediction)
  - [Problem Statement](#problem-statement-task-2)
  - [Dataset](#dataset-task-2)
  - [Methodology](#methodology)
  - [Feature Engineering](#feature-engineering)
  - [Model and Results](#model-and-results)
  - [Key Findings](#key-findings-task-2)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Skills Demonstrated](#skills-demonstrated)

---

## Overview

This repository contains both tasks from the **British Airways Data Science Job Simulation** on [Forage](https://www.theforage.com/). Each task tackles a different real-world business problem using a different analytical approach.

| | Task 1 | Task 2 |
|---|---|---|
| **Problem** | How many passengers will use each lounge? | Which customers will complete a booking? |
| **Type** | Data modelling & estimation | Machine learning classification |
| **Tool** | Excel (data analysis + structured model) | Python (Random Forest + cross-validation) |
| **Dataset** | 10,000 flight schedule records | 50,000 booking records |
| **Output** | Lounge eligibility lookup table | Predictive model + feature importance |

---

## Repository Structure

```
british-airways-data-science/
│
├── README.md                                         ← This file
├── requirements.txt                                  ← Python dependencies (Task 2)
├── .gitignore
│
├── task1_lounge_eligibility/
│   ├── BA_Lounge_Eligibility_Submission.xlsx         ← Lookup table + justification (2 sheets)
│   └── British_Airways_Summer_Schedule_Dataset.xlsx  ← Source dataset
│
└── task2_booking_prediction/
    ├── ba_booking_analysis.py                        ← Standalone Python script (full pipeline)
    ├── BA_Booking_Prediction.ipynb                   ← Jupyter notebook (step-by-step)
    ├── customer_booking.csv                          ← Source dataset
    │
    ├── figures/                                      ← Generated visualisations (run script first)
    │   ├── 01_dataset_overview.png
    │   ├── 02_numeric_distributions.png
    │   ├── 03_temporal_patterns.png
    │   ├── 04_addon_booking_rates.png
    │   ├── 05_cv_performance.png
    │   ├── 06_roc_pr_curves.png
    │   ├── 07_feature_importance.png
    │   └── 08_feature_deepdives.png
    │
    └── outputs/
        ├── BA_Booking_Prediction_Slides.pptx         ← Manager summary (7 slides)
        ├── model_metrics.csv
        └── feature_importances.csv
```

---

## Task 1 — Lounge Eligibility Modelling

### Problem Statement (Task 1)

British Airways operates lounges at Terminal 3, Heathrow across three tiers:

| Tier | Lounge | Eligible Passengers |
|---|---|---|
| Tier 1 | Concorde Room *(hypothetical at T3)* | First Class, Gold Guest List |
| Tier 2 | Club World Lounge | Business Class, Gold & Silver Executive Club |
| Tier 3 | Economy Lounge | Eligible economy passengers |

The business question: **"Given a flight schedule, how many passengers will use each lounge, and when?"** This informs staffing, catering, and capacity planning decisions.

> ⚠️ **Note:** No Concorde Room currently exists at Terminal 3. Tier 1 figures are retained as a hypothetical planning signal — for example, to inform whether a Tier 1 facility might be viable in future.

### Dataset (Task 1)

- **Source:** British Airways Summer Schedule (Forage)
- **Size:** 10,000 flight records
- **Key fields:** `HAUL`, `TIME_OF_DAY`, `FIRST_CLASS_SEATS`, `BUSINESS_CLASS_SEATS`, `ECONOMY_SEATS`, `TIER1_ELIGIBLE_PAX`, `TIER2_ELIGIBLE_PAX`, `TIER3_ELIGIBLE_PAX`

### Approach (Task 1)

Rather than modelling individual flights, a **2×4 category matrix** was built using two dimensions that genuinely drive lounge demand:

**Dimension 1 — Haul Type (Short / Long)**
- Long-haul flights carry First Class cabins and attract a higher share of Executive Club premium members
- Short-haul at T3 is predominantly A320-family aircraft with no First Class → materially different tier profile

**Dimension 2 — Time-of-Day Band**

| Band | Clock Range | Passenger Mix |
|---|---|---|
| Morning | 06:00 – 11:59 | Business-heavy; high Executive Club uptake |
| Lunchtime | 12:00 – 13:59 | Mixed leisure; moderate premium |
| Afternoon | 14:00 – 17:59 | Leisure-heavy; lower lounge propensity |
| Evening | 18:00 – 23:59 | Post-work business + evening long-haul; high-yield |

This yields **8 categories** (SH-MO, SH-LN, SH-AF, SH-EV, LH-MO, LH-LN, LH-AF, LH-EV), each backed by hundreds of real flight records.

### Lookup Table

Tier percentages were derived **directly from the data** — `TIER_N_ELIGIBLE_PAX` as a share of total seats, averaged per category. This makes the model data-driven rather than subjective.

| Category | Haul | Time Band | Avg Seats | Tier 1 % | Tier 2 % | Tier 3 % | Est. T2 Pax | Est. T3 Pax |
|---|---|---|---|---|---|---|---|---|
| SH-MO | Short | Morning | 180 | 0.3% | 4.4% | 16.7% | 7.9 | 30.1 |
| SH-LN | Short | Lunchtime | 180 | 0.4% | 4.6% | 17.3% | 8.3 | 31.1 |
| SH-AF | Short | Afternoon | 180 | 0.3% | 4.3% | 16.6% | 7.7 | 29.9 |
| SH-EV | Short | Evening | 180 | 0.3% | 4.4% | 17.0% | 7.9 | 30.6 |
| LH-MO | Long | Morning | 292 | 0.2% | 2.8% | 10.6% | 8.2 | 30.9 |
| LH-LN | Long | Lunchtime | 290 | 0.2% | 2.7% | 10.4% | 7.8 | 30.2 |
| LH-AF | Long | Afternoon | 293 | 0.2% | 2.8% | 10.5% | 8.2 | 30.8 |
| LH-EV | Long | Evening | 292 | 0.2% | 2.7% | 10.3% | 7.9 | 30.1 |

> **How to use:** For any future flight, identify haul type + departure time band → read off the tier percentages → multiply by expected passenger load → sum across all departures in a time window for peak lounge capacity planning.

**The Excel file contains two sheets:**
- **Sheet 1 — Lounge Eligibility Lookup:** Time-of-day band definitions, the 8-row lookup table with live Excel formulas, and 16 sample flights with the model applied
- **Sheet 2 — Justification:** Four structured questions answered — how groups were chosen, why they make sense, assumptions made and their reasoning, and how the model scales to future schedules

### Key Findings (Task 1)

1. **Short-haul has higher tier percentages than long-haul** — not because more premium passengers fly short-haul, but because the aircraft are smaller (~180 seats vs ~292), so the same absolute number of eligible passengers represents a larger percentage of total load

2. **Time of day has a smaller effect than haul type** — tier percentages are stable within each haul category across time bands (~4.3–4.6% Tier 2 for short-haul regardless of whether it is morning or evening)

3. **Tier 1 eligibility is very low (~0.2–0.4%)** — even on long-haul morning flights only ~0.6 passengers per flight qualify for First Class-equivalent service, raising a genuine business question about whether a dedicated Tier 1 space at T3 would be viable without route consolidation

4. **The model is fully reusable** — any new flight can be classified in exactly two steps (haul type + departure time), making it applicable to winter schedules, charter operations, and new route launches without any recalibration

---

## Task 2 — Customer Booking Prediction

### Problem Statement (Task 2)

British Airways customers frequently start the booking process but do not complete it. The business question: **"Can we predict, from information available at booking time, which customers are likely to complete their booking?"**

This enables BA to:
- Focus conversion campaigns on high-intent customers
- Understand which factors actually drive booking completion
- Avoid wasting marketing spend on customers unlikely to convert

### Dataset (Task 2)

- **Source:** British Airways booking data (Forage)
- **Size:** 50,000 records × 14 columns
- **Class balance:** 85% not booked (42,522) / 15% booked (7,478) — imbalanced

| Column | Type | Description |
|---|---|---|
| `num_passengers` | int | Number of passengers in the booking |
| `sales_channel` | str | Internet or Mobile |
| `trip_type` | str | RoundTrip, OneWay, CircleTrip |
| `purchase_lead` | int | Days between booking start and flight date |
| `length_of_stay` | int | Nights at destination |
| `flight_hour` | int | Hour of departure (0–23) |
| `flight_day` | str | Day of week (Mon–Sun) |
| `route` | str | Origin–destination code (799 unique values) |
| `booking_origin` | str | Country where booking was made (104 unique) |
| `wants_extra_baggage` | int | Add-on flag (0/1) |
| `wants_preferred_seat` | int | Add-on flag (0/1) |
| `wants_in_flight_meals` | int | Add-on flag (0/1) |
| `flight_duration` | float | Flight length in hours |
| `booking_complete` | int | **Target** — 1 = booking completed |

### Methodology

```
Raw Data (50,000 rows, 14 columns)
           │
           ▼
   Exploratory Data Analysis
   • Class balance check → 85/15 imbalance identified
   • Distribution analysis split by target variable
   • Categorical breakdown (channel, trip type, day, hour)
   • Add-on behaviour vs booking completion rate
           │
           ▼
   Feature Engineering
   • 8 original features → 19 total features
   • Log transforms, frequency encoding,
     binary flags, aggregation, interaction term
           │
           ▼
   Random Forest Classifier
   • n_estimators=300, max_depth=12
   • class_weight='balanced'  ← critical for imbalanced data
   • random_state=42
           │
           ▼
   5-Fold Stratified Cross-Validation
   • Stratified = each fold preserves 85/15 class ratio
   • Out-of-fold predictions for honest evaluation
   • Metrics: Accuracy, ROC-AUC, F1, Precision, Recall
           │
           ▼
   Feature Importance + Visualisations
   • 8 figures covering EDA, evaluation, and importance
```

### Feature Engineering

11 new features were created on top of the 8 original numeric/binary features:

| Feature | Method | Rationale |
|---|---|---|
| `log_purchase_lead` | `log1p(purchase_lead)` | Compresses right-skewed lead time distribution |
| `log_length_of_stay` | `log1p(length_of_stay)` | Same — stay duration is heavily right-skewed |
| `route_freq` | Frequency encode | Replaces 799 unique route strings with their proportion in the dataset; captures route popularity as a signal |
| `origin_freq` | Frequency encode | Same for 104 booking origins — became the single most important feature at 32.2% |
| `total_addons` | Sum of 3 flags | 0–3 commitment score; a customer who picked seat + baggage + meals is demonstrably more committed |
| `is_weekend` | Binary flag | Sat/Sun departure — leisure vs business proxy |
| `is_mobile` | Binary flag | Mobile vs Internet channel |
| `is_roundtrip` | Binary flag | Dominant trip type (99%); flags minority types |
| `is_peak_hour` | Binary flag | 06–09 or 17–20 departure hour |
| `pax_lead_interaction` | `num_passengers × log_purchase_lead` | Group bookings made far in advance are a qualitatively different signal than either variable alone |
| `flight_day_num` | Ordinal encode | Mon=1 through Sun=7 |

### Model and Results

**Random Forest Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=300,        # 300 trees — stable; diminishing returns after ~200
    max_depth=12,            # caps tree depth to prevent overfitting
    min_samples_split=20,    # a node needs 20+ samples before it can be split
    min_samples_leaf=10,     # each leaf needs 10+ samples — smooths predictions
    max_features='sqrt',     # each tree sees sqrt(n_features) — creates diversity between trees
    class_weight='balanced', # corrects the 85/15 class imbalance
    random_state=42          # reproducibility
)
```

**5-Fold Cross-Validation Results:**

| Metric | Mean | Std Dev |
|---|---|---|
| **ROC-AUC** | **0.761** | ±0.004 |
| Accuracy | 0.731 | ±0.004 |
| F1-Score | 0.412 | ±0.005 |
| Precision | 0.306 | ±0.004 |
| Recall | 0.631 | ±0.016 |
| Avg Precision | 0.357 | — |

**Confusion Matrix (Out-of-Fold Predictions):**

| | Predicted: Not Booked | Predicted: Booked |
|---|---|---|
| **Actual: Not Booked** | 74.8% (n=31,812) | 25.2% (n=10,710) |
| **Actual: Booked** | 36.9% (n=2,756) | **63.1% (n=4,722)** |

> The model correctly identifies **63% of genuine bookers**. At a base rate of only 15%, this represents roughly 2× lift over random selection — meaningful for any conversion targeting campaign.

**Feature Importance (Top 10):**

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | `origin_freq` | 32.2% | Engineered |
| 2 | `route_freq` | 10.3% | Engineered |
| 3 | `flight_duration` | 8.0% | Original |
| 4 | `length_of_stay` | 7.3% | Original |
| 5 | `log_length_of_stay` | 6.9% | Engineered |
| 6 | `pax_lead_interaction` | 5.7% | Engineered |
| 7 | `purchase_lead` | 5.6% | Original |
| 8 | `log_purchase_lead` | 5.5% | Engineered |
| 9 | `flight_hour` | 4.3% | Original |
| 10 | `total_addons` | 3.1% | Engineered |

### Key Findings (Task 2)

1. **Booking origin is the single strongest predictor (32.2%)** — frequency-encoding `booking_origin` unlocked far more signal than any raw feature. Customers from high-frequency booking countries complete bookings at distinctly different rates, pointing to market-level differences in intent and booking behaviour

2. **Route popularity matters (10.3%)** — passengers on popular routes complete bookings at higher rates, possibly because popular routes have less price uncertainty, fewer alternatives, or stronger demand confidence

3. **Longer flights see higher completion (8.0%)** — long-haul passengers are more committed to their trip plan. These are also higher-value customers, making them a priority for retention efforts

4. **Add-ons are a strong intent signal** — customers who request extra baggage, preferred seats, or in-flight meals are significantly more likely to complete. Each add-on represents incremental commitment; triggering a conversion nudge after the first add-on is selected is a concrete actionable recommendation

5. **Class imbalance must be handled explicitly** — without `class_weight='balanced'`, the model learns to predict "not booked" for nearly everyone and achieves 85% accuracy while being completely useless. Stratified cross-validation ensures the evaluation is honest across all folds

6. **Engineered features outperform raw features** — 6 of the top 10 features by importance were engineered rather than original. Frequency-encoded origin and route combined account for over 42% of model importance, demonstrating the outsized value of thoughtful feature construction

---

## How to Run

### Task 1
No code to run. Open `task1_lounge_eligibility/BA_Lounge_Eligibility_Submission.xlsx` in Excel or LibreOffice Calc. All formulas are live and recalculate automatically if you update assumption cells.

### Task 2 — Option A: Jupyter Notebook (recommended for step-by-step exploration)
```bash
cd task2_booking_prediction
pip install -r ../requirements.txt
jupyter notebook BA_Booking_Prediction.ipynb
```
Run all cells top to bottom. Figures are saved to `figures/`. **Run this before pushing to GitHub** so cell outputs (charts, tables) are embedded in the notebook and visible to anyone who opens the repo.

### Task 2 — Option B: Python Script (recommended for full reproducibility)
```bash
cd task2_booking_prediction
pip install -r ../requirements.txt
python ba_booking_analysis.py
```
Runs the entire pipeline in one shot. All 8 figures saved to `figures/`, metrics saved to `outputs/`.

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
notebook>=6.5.0
ipykernel>=6.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Skills Demonstrated

**Task 1 — Lounge Eligibility Modelling**
- Exploratory data analysis on a real airline schedule dataset (10,000 records)
- Structured estimation-based modelling without machine learning
- Business assumption documentation and written justification
- Excel: multi-sheet workbook, dynamic formulas, data-driven segmentation, professional formatting

**Task 2 — Customer Booking Prediction**
- Identifying and handling class imbalance (`class_weight='balanced'`, stratified cross-validation)
- Feature engineering: frequency encoding, log transforms, interaction terms, binary flags, aggregation
- Random Forest classification with fully justified hyperparameter choices
- Model evaluation: ROC-AUC, precision-recall curves, confusion matrix, 5-fold cross-validation
- Data visualisation with matplotlib and seaborn (8 publication-quality figures)
- Translating model outputs into concrete, actionable business recommendations
- Manager communication: 7-slide PowerPoint summary presentation

---

## About

Completed as part of the **British Airways Data Science Job Simulation** on [Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz).  
The simulation mirrors real analytical work done by the BA data science team.
