# Water Content Drainage Predictor

A public decision-support tool for estimating water-content change rate based on field conditions including fine content, cross slope, elapsed time, current water content, and wicking-geotextile presence.

---

## Model Overview

- Algorithm: Random Forest Regressor (200 trees)
- Dataset size: 6,682 observations
- Train/Test split: 80/20 random holdout
- Performance: R² ≈ 0.905 (test set)
- Output unit: percentage points per hour

---

## Input Features

The model uses the following inputs:

- Fine content (%)
- Cross slope (%)
- Elapsed time (hours)
- Water content (%)
- Wicking geotextile (binary: 0/1)

---

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```
