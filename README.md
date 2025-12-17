# Route-Stratified Separation-Based Diagnose Pass Outcome

**A Triple-Threat Approach to NFL Quarterback Pressure Analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org/)
[![NFL Big Data Bowl 2026](https://img.shields.io/badge/NFL%20Big%20Data%20Bowl-2026-green)](https://www.kaggle.com/c/nfl-big-data-bowl-2026-analytics)

---

## 📊 Project Overview

This project analyzes **pre-throw separation-based pressure metrics** to diagnose how different types of defensive pressure influence pass outcomes across route depths using NFL tracking data.

**Key Finding:** Defensive pressure changes meaning depending on route type:
- **SHORT routes** (≤5 yards): Coverage pressure dominates - 30 percentage point failure rate swing
- **DEEP routes** (>15 yards): Convergence pressure dominates - 18 percentage point failure rate swing  
- **INTERMEDIATE routes** (5-15 yards): Multi-dimensional pressure - both coverage and convergence matter

### Three Pressure Dimensions

1. **Convergence Pressure [0-100]**: How many defenders are closing in on the QB? (spatial threat)
2. **Velocity Pressure [0-100]**: How rapidly is QB-defender separation shrinking? (temporal dynamics)
3. **Coverage Pressure [0-100]**: How tight is the throwing window at the receiver? (execution difficulty)

**Dataset:** 14,105 passing plays from 2023 NFL season (18 weeks)  
**Source:** [NFL Big Data Bowl 2026](https://www.kaggle.com/c/nfl-big-data-bowl-2026-analytics)

---

## 🎯 Project Purpose

This repository serves as:
- **Competition Submission:** NFL Big Data Bowl 2026 Analytics Challenge
- **Learning Portfolio:** Demonstrating ML/Data Science practices in Python
- **Football Analytics Research:** Novel route-stratified pressure analysis framework

**Focus:** Pattern diagnosis and insight generation rather than predictive modeling. The goal is understanding *which* pressure dimensions matter for different play types, not predicting individual pass outcomes.

---

## 📁 Repository Structure

```
Route-Stratified-Seperation-based-Diagnose-Pass-Outcome/
│
├── data/                                      # NFL tracking data (included)
│   ├── input/                                 # Raw weekly tracking files
│   │   ├── input_2023_w01.csv
│   │   ├── input_2023_w02.csv
│   │   └── ... (weeks 1-18)
│   ├── output/                                # Engineered features
│   │   └── final_ml_features_with_component_scores.csv
│   └── supplementary_data.csv                 # Play metadata
│
├── figures/                                   # Generated visualizations
│   ├── outcome_distributions.png
│   ├── coverage_pressure_vs_outcome.png
│   └── ... (various analysis plots)
│
├── 01_FeatureEnginnering_seperationBased.ipynb                      # 01-Feature creation pipeline
├── 02_Features_Evaluation_3Way.ipynb                                 # 02-Model evaluation & analysis
├── requirements.txt                                                 # Python dependencies
└── README.md                                                        # This file
└── Route-Stratified_PreThrow_Pressure_Diagnose_Pass-Outcome.ipynb   # Feature creation pipeline and features Evaluation-Analysis -Step 01 and 02 together

```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Hsotuhsa-S/Route-Stratified-Seperation-based-Diagnose-Pass-Outcome.git
cd Route-Stratified-Seperation-based-Diagnose-Pass-Outcome
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files exist:**
```bash
ls data/input/  # Should show input_2023_w01.csv through input_2023_w18.csv
ls data/supplementary_data.csv
```

### Running the Analysis

**Step 1: Feature Engineering**
```bash
jupyter notebook 01_FeatureEnginnering_seperationBased.ipynb
```
- Run all cells sequentially (Cell → Run All)
- Creates `data/output/final_ml_features_with_component_scores.csv`

**Step 2: Evaluation & Analysis**
```bash
jupyter notebook 02_Features_Evaluation_3Way.ipynb
```
- Run all cells sequentially
- Generates visualizations in `figures/` directory

### Output Verification

After running both notebooks, verify:
```bash
ls data/output/final_ml_features_with_component_scores.csv  # Feature file created
ls figures/*.png  # Should show 10+ visualization files
```

---

## 📓 Notebook Documentation

### Notebook 1: `01_FeatureEnginnering_seperationBased.ipynb`

**Purpose:** Feature engineering pipeline that transforms raw NFL tracking data into ML-ready pressure metrics.

**Input:**
- `data/input/input_2023_w{01-18}.csv` - Frame-level player tracking data (x, y coordinates, velocities)
- `data/supplementary_data.csv` - Play metadata (route types, pass results, yards gained)

**Processing Steps:**
1. **Load & Merge:** Concatenate 18 weeks of tracking data with play metadata
2. **QB Separation Calculation:** Frame-level minimum distance between QB and nearest defender
3. **Pressure Velocity:** Rate of change in QB-defender separation (yards/second)
4. **Defender Convergence:** Count defenders within proximity zones (3/5/7 yards from QB)
5. **Coverage Pressure:** Targeted receiver separation from nearest defender
6. **Normalization:** Scale all metrics to [0-100] scores using MinMaxScaler

**Output:**
- `data/output/final_ml_features_with_component_scores.csv` 
  - Columns: `convergence_pressure_score`, `velocity_pressure_score`, `tr_coverage_pressure_score`
  - Shape: ~14,105 plays × 46 features

**Key Functions:**
- `calculate_frame_level_separation_qb()` - Vectorized QB-defender distance calculation
- `calculate_frame_level_pressure_velocity_qb()` - Frame-to-frame separation velocity
- `calculate_frame_level_defender_convergence()` - Zone-based convergence counting

---

### Notebook 2: `02_Features_Evaluation_3Way.ipynb`

**Purpose:** Evaluate engineered pressure features through descriptive analysis, statistical testing, and baseline modeling.

**Input:**
- `data/output/final_ml_features_with_component_scores.csv` (from Notebook 1)

**Analysis Pipeline:**

#### 1. **Target Variable Creation**
- `y_incomplete` (Binary): Complete (0) vs Incomplete/Intercepted (1)
- `y_outcome` (3-Way Classification):
  - 0 = Negative Play (Incomplete OR ≤0 yards)
  - 1 = Neutral Play (Complete, 1-4 yards)
  - 2 = Positive Play (Complete, 5+ yards)

#### 2. **Route Stratification**
- SHORT: HITCH, SLANT, SCREEN, FLAT, WHEEL (n=6,069, 43%)
- INTERMEDIATE: OUT, IN, CROSS, ANGLE (n=5,359, 38%)
- DEEP: GO, POST, CORNER (n=2,675, 19%)

#### 3. **Descriptive Analysis**
- Feature distributions (histograms with KDE)
- Pressure-outcome relationships (boxplots, violin plots)
- Correlation heatmaps (Spearman coefficients)

#### 4. **Route-Stratified Evaluation**
- Separate logistic regression models for each route group
- Coefficient significance testing (p-values)
- Identifies which pressure dimensions matter per route type

#### 5. **Baseline Model Comparison**
- Individual feature models (single pressure type)
- Pairwise combinations (two pressure types)
- Combined model (all three pressure types)
- Models: Logistic Regression, Random Forest

**Output:**
- **Visualizations:** Saved to `figures/` directory
  - Outcome distributions with percentages
  - Pressure component histograms with summary stats
  - Boxplots comparing pressure across outcomes
  - ROC curves and confusion matrices
  - Feature correlation heatmaps

- **Performance Metrics:**
  - ROC AUC (One-vs-Rest & One-vs-One)
  - Precision, Recall, F1-Score
  - Confusion matrices
  - Feature importances

- **Statistical Results:**
  - Model coefficients with p-values
  - Route-stratified regression tables

**Key Findings:**
- Coverage pressure: Primary signal for SHORT routes (β=+0.296, p<0.001)
- Convergence pressure: Primary signal for DEEP routes (β=-0.208, p<0.001)
- INTERMEDIATE routes: Multi-dimensional (both coverage and convergence significant)

---



---

## 📈 Key Results Summary

### Pressure Signal Strength by Route Type

| Route Type | Coverage Pressure | Convergence Pressure | Velocity Pressure |
|------------|-------------------|----------------------|-------------------|
| **SHORT** | ✓✓✓ Primary (30pp impact) | ✗ Noise | ✓✓ Secondary |
| **INTERMEDIATE** | ✓✓✓ Primary (β=+0.374) | ✓ Secondary | ✗ Noise |
| **DEEP** | ✗ Noise | ✓✓✓ Primary (18pp impact) | ✓ Moderate |

### Failure Rate Patterns

| Route Type | Failure Rate | Average Coverage | Average Convergence |
|------------|--------------|------------------|---------------------|
| SHORT | 25.4% | 48.5 | 9.1 |
| INTERMEDIATE | 31.6% | 64.1 | 10.2 |
| DEEP | **52.8%** | 80.4 | 11.5 |

**Insight:** DEEP routes fail at 2.1× the rate of SHORT routes (52.8% vs 25.4%), driven by convergence pressure accumulation over extended play development time (3.4 seconds vs 2.1 seconds).

For detailed findings and football context, see **[NFLProject_writeup_md.txt](NFLProject_writeup_md.txt)**.

---

## 🛠️ Technical Approach

### Feature Engineering Techniques

**Vectorized Distance Calculations:**
```python
def calculate_euclidean_distance_vectorized(tr_x, tr_y, def_x, def_y):
    return np.sqrt((def_x - tr_x)**2 + (def_y - tr_y)**2)
```
- Processes all defender distances simultaneously (10× speedup vs loops)

**Frame-Level Pressure Velocity:**
```python
df['separation_velocity'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['qb_min_separation'].diff() / 0.1
```
- Calculates rate of change across frames (0.1 second intervals)
- Negative velocity = defenders closing in

**Zone-Based Convergence:**
- Immediate zone (0-3 yards): Must throw NOW
- Closing zone (3-5 yards): Pressure building rapidly
- Potential zone (5-7 yards): Defenders approaching

### Modeling Strategy

**Cross-Validation:** 5-fold Stratified K-Fold (maintains class distribution)

**Baseline Models:**
- Logistic Regression: Interpretable coefficients for feature importance
- Random Forest: Captures non-linear interactions

**Evaluation Focus:**
- ROC AUC for multi-class classification (One-vs-Rest, One-vs-One)
- Statistical significance (p-values) over raw accuracy
- Permutation importance for feature ranking

---


## 🔧 Dependencies

```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical operations
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualizations
scikit-learn>=1.0.0    # Machine learning
scipy>=1.9.0           # Statistical tests
```

Full requirements: [requirements.txt](requirements.txt)

---


## 👤 Author

**Ashutosh Shirsat(Hsotuhsa-S)**

*Project created as part of the NFL Big Data Bowl 2026 Analytics Challenge and for Machine Learning/Data Science practice.*

---
