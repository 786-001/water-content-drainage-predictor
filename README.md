# Water Content Drainage Predictor

Public decision-support application for estimating water-content change rate from fine content, cross slope, elapsed time, current water content, and wicking-geotextile conditions.

## Model

- Random forest regressor with 200 trees
- 6,682 observations
- Random 80/20 holdout R2: 0.905
- Output displayed in percentage points per hour

Predictions should be validated with field measurements before operational use.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
