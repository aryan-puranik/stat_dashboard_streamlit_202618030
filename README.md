# Medical Insurance Costs — Statistical Dashboard

An end-to-end data science project (EDA → hypothesis testing → OLS regression →
Gauss-Markov diagnostics) delivered as a 3-tab interactive Streamlit app.

## Files
- `app.py` — the Streamlit application
- `insurance.csv` — the dataset (age, sex, bmi, children, smoker, region, charges; n=1338)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
(If `streamlit` isn't recognized as a command on Windows, use `python -m streamlit run app.py` instead.)

## Dataset Summary

**Source:** Medical Insurance Costs dataset, `insurance.csv` — 1,338 records, 7 raw
features (an 8th, `smoker_flag`, is derived for correlation purposes).

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of primary beneficiary (18–64) |
| `sex` | categorical | `male` / `female` (676 / 662) |
| `bmi` | numeric | Body mass index |
| `children` | numeric | Number of dependents covered (0–5) |
| `smoker` | categorical | `yes` / `no` (274 / 1064) |
| `region` | categorical | `northeast` / `northwest` / `southeast` / `southwest` (324–364 each) |
| `charges` | numeric | Individual medical charges billed — the target variable |

**Numeric feature stats (full dataset):**

| Feature | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|
| age | 39.2 | 39.0 | 14.0 | 18 | 64 |
| bmi | 30.7 | 30.4 | 6.1 | 16.0 | 53.1 |
| children | 1.1 | 1.0 | 1.2 | 0 | 5 |
| charges | $13,270 | $9,382 | $12,110 | $1,122 | $63,770 |

`charges` is strongly right-skewed (a small share of very high-cost cases, largely
smokers, pull the mean well above the median) — this is why several tests below
route to non-parametric alternatives.

## Model

Fit with `statsmodels.api.OLS` (matrix form, not the formula API), per the assignment spec:

```
charges = β0 + β1·age + β2·bmi + β3·children + β4·sex_male + β5·smoker_yes
        + β6·region_NW + β7·region_SE + β8·region_SW + β9·(smoker_yes × bmi) + ε
```

R² ≈ 0.84. The `smoker × bmi` interaction captures how smoking amplifies BMI's cost impact.

## Tabs

**Tab 1 — Data Exploration**
Sidebar filters (age/BMI sliders, region/sex/smoker multi-selects) drive reactive
Plotly charts: distribution histograms split by smoker status, scatter plots vs.
charges with OLS trendlines, a correlation heatmap, and categorical box plots —
plus a live summary-statistics table (mean, median, std, IQR, skewness, kurtosis).

**Tab 2 — Hypothesis Testing Lab**
- *Group comparison:* pick any categorical factor + numeric metric. The app runs
  Shapiro-Wilk per group and Levene's test, then automatically selects a
  Two-Sample t-test / Mann-Whitney U (2 groups) or One-Way ANOVA / Kruskal-Wallis
  (3+ groups, e.g. region), and states Reject/Fail to Reject H₀ at α = 0.05.
- *Association test:* pick any two categorical variables for a chi-square test of
  independence, with contingency table, expected counts, and Cramér's V effect size.

**Tab 3 — Live Prediction & Diagnostics**
- Enter age, BMI, children, sex, smoker status, and region via sliders/inputs to
  get a real-time predicted charge with a 95% **confidence interval** (average
  charge for that profile) and 95% **prediction interval** (likely range for one
  individual), computed via `model.get_prediction(...).summary_frame()`.
- Residual diagnostics: residuals-vs-fitted plot + Breusch-Pagan test
  (homoscedasticity), Q-Q plot + Jarque-Bera and Omnibus normality tests, VIF for
  the continuous predictors (age, bmi, children), and the Durbin-Watson statistic.

## Synthesis of Statistical Findings (full dataset)

Smoking status is, by a wide margin, the dominant driver of medical charges in
this data — it shows up as significant in the group-comparison test, as the
largest-magnitude coefficient in the regression, and as the reason BMI's effect
on cost is so much steeper for smokers. Region and sex, by contrast, show little
to no significant relationship with cost once smoking and BMI are accounted for.

- **Smoker vs. non-smoker charges:** neither group passes Shapiro-Wilk, so the app
  runs Mann-Whitney U → p ≈ 5×10⁻¹³⁰ → **reject H₀**.
- **Region as a 4-level factor on charges:** non-normal residuals → Kruskal-Wallis;
  check the live result in-app, it updates with your dropdown selections.
- **Smoking × region (chi-square):** χ² ≈ 7.34, dof = 3, p ≈ 0.062 → **fail to
  reject H₀** at α = 0.05.
- **Residual diagnostics:** Jarque-Bera and Omnibus both strongly reject normality
  (p ≈ 0), and Breusch-Pagan flags heteroscedasticity — expected for right-skewed
  cost data. VIFs for age/bmi/children are all ≈ 1, so multicollinearity is not a
  concern among the continuous predictors.

## Deploying (optional)

Push this folder to GitHub and deploy free on
[Streamlit Community Cloud](https://streamlit.io/cloud), pointing it at `app.py`.
