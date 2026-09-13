

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ----------------------------------------------------------------------------
# 1. Page Config & CSS Styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
<style>
    /* Global font & background enhancements */
    .main {
        background-color: #0e1117;
    }
    
    /* Custom metric card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d2ff;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aab2;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Tab navigation tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

ALPHA = 0.05
NUMERIC_COLS = ["age", "bmi", "children", "charges"]
CONTINUOUS_PREDICTORS = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]

# Helper function to generate standardized metric cards
def make_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2. Data Loading & Model Setup
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    df["smoker_flag"] = (df["smoker"] == "yes").astype(int)
    return df

df_full = load_data()

@st.cache_resource
def fit_model(data: pd.DataFrame):
    X = pd.DataFrame(index=data.index)
    X["age"] = data["age"]
    X["bmi"] = data["bmi"]
    X["children"] = data["children"]
    X["sex_male"] = (data["sex"] == "male").astype(int)
    X["smoker_yes"] = (data["smoker"] == "yes").astype(int)
    X["region_northwest"] = (data["region"] == "northwest").astype(int)
    X["region_southeast"] = (data["region"] == "southeast").astype(int)
    X["region_southwest"] = (data["region"] == "southwest").astype(int)
    X["smoker_bmi"] = X["smoker_yes"] * X["bmi"]
    X = sm.add_constant(X)
    y = data["charges"]
    model = sm.OLS(y, X).fit()
    return model, X

model, X_design = fit_model(df_full)

def build_design_row(age, bmi, children, sex, smoker, region):
    row = {
        "const": 1.0,
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    row["smoker_bmi"] = row["smoker_yes"] * row["bmi"]
    return pd.DataFrame([row])[X_design.columns]

# ----------------------------------------------------------------------------
# 3. Header Section
# ----------------------------------------------------------------------------
st.title("💊 Medical Insurance Cost Analytics")
st.caption(
    "Interactive Statistical Dashboard & Gauss-Markov Regression Engine • "
    "**Model:** `charges = f(age, bmi, children, sex, smoker, region, smoker × bmi)`"
)

tab1, tab2, tab3 = st.tabs(["📊 Interactive EDA", "🔬 Hypothesis Testing Lab", "🔮 Predictive Model & Diagnostics"])

# ============================================================================
# TAB 1 — INTERACTIVE DATA EXPLORATION
# ============================================================================
with tab1:
    st.sidebar.header("🔎 Dynamic Filters")
    st.sidebar.caption("Applies exclusively to Tab 1 visualizations.")
    
    age_range = st.sidebar.slider("Age Range", int(df_full.age.min()), int(df_full.age.max()),
                                  (int(df_full.age.min()), int(df_full.age.max())), key="t1_age")
    bmi_range = st.sidebar.slider("BMI Range", float(df_full.bmi.min()), float(df_full.bmi.max()),
                                  (float(df_full.bmi.min()), float(df_full.bmi.max())), key="t1_bmi")
    regions_sel = st.sidebar.multiselect("Region", sorted(df_full.region.unique()),
                                         default=sorted(df_full.region.unique()), key="t1_region")
    sex_sel = st.sidebar.multiselect("Sex", sorted(df_full.sex.unique()),
                                     default=sorted(df_full.sex.unique()), key="t1_sex")
    smoker_sel = st.sidebar.multiselect("Smoker Status", sorted(df_full.smoker.unique()),
                                        default=sorted(df_full.smoker.unique()), key="t1_smoker")

    dff = df_full[
        df_full.age.between(*age_range) & df_full.bmi.between(*bmi_range) &
        df_full.region.isin(regions_sel) & df_full.sex.isin(sex_sel) & df_full.smoker.isin(smoker_sel)
    ]

    if dff.empty:
        st.error("⚠️ No data matches the selected filter criteria. Adjust your filters in the sidebar.")
    else:
        # Custom KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1: make_card("Filtered Records", f"{len(dff):,}")
        with c2: make_card("Avg. Charges", f"${dff.charges.mean():,.2f}")
        with c3: make_card("Smoker Proportion", f"{(dff.smoker=='yes').mean()*100:.1f}%")
        with c4: make_card("Avg. BMI", f"{dff.bmi.mean():.1f}")

        st.markdown("---")

        # Summary Metrics Table
        st.subheader("📋 Descriptive Statistics Table")
        rows = []
        for col in NUMERIC_COLS:
            s = dff[col]
            q1, q3 = s.quantile(.25), s.quantile(.75)
            rows.append({
                "Feature": col, "Mean": s.mean(), "Std Dev": s.std(),
                "Median": s.median(), "IQR": q3 - q1, 
                "Skewness": s.skew(), "Kurtosis": s.kurt()
            })
        st.dataframe(pd.DataFrame(rows).set_index("Feature").style.format("{:.2f}"), use_container_width=True)

        # Plot Grid Layout
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📈 Distribution Analysis")
            dist_col = st.selectbox("Select Feature for Distribution", NUMERIC_COLS, index=3, key="t1_dist")
            fig_hist = px.histogram(
                dff, x=dist_col, color="smoker", marginal="violin", nbins=40,
                opacity=0.7, color_discrete_sequence=["#00d2ff", "#ff4b4b"],
                template="plotly_dark", title=f"Distribution of {dist_col.title()} by Smoker Status"
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("🔥 Dynamic Correlation Heatmap")
            corr = dff[NUMERIC_COLS + ["smoker_flag"]].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="Viridis", zmin=-1, zmax=1,
                template="plotly_dark", title="Feature Correlation Matrix"
            )
            fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_right:
            st.subheader("🎯 Charges vs Predictors")
            xcol = st.selectbox("Select X-Axis Feature", CONTINUOUS_PREDICTORS, key="t1_scatter_x")
            fig_scatter = px.scatter(
                dff, x=xcol, y="charges", color="smoker", trendline="ols",
                size="bmi" if xcol != "bmi" else "age",
                hover_data=["sex", "region", "children"],
                color_discrete_sequence=["#ff4b4b", "#00d2ff"],
                template="plotly_dark", title=f"Charges vs {xcol.title()} (Trendline OLS)"
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("📦 Category Cost Breakdowns")
            cat_col = st.selectbox("Select Categorical Split", CATEGORICAL_COLS, key="t1_cat")
            fig_box = px.box(
                dff, x=cat_col, y="charges", color=cat_col, points="all",
                template="plotly_dark", title=f"Cost Distribution across {cat_col.title()}"
            )
            fig_box.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

        # Advanced Interactive Matrix/Pair Plot
        with st.expander("🔍 View Pairwise Matrix Plot"):
            fig_pair = px.scatter_matrix(
                dff, dimensions=NUMERIC_COLS, color="smoker",
                color_discrete_sequence=["#ff4b4b", "#00d2ff"],
                template="plotly_dark", title="Multivariate Pairwise Relationships"
            )
            st.plotly_chart(fig_pair, use_container_width=True)

# ============================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ============================================================================
with tab2:
    st.header("🔬 Hypothesis Testing Lab")
    st.caption("Statistical significance checks executed on full dataset (n = 1,338).")

    st.subheader("A. Numerical Differences Across Categorical Groups")
    colA, colB = st.columns(2)
    factor = colA.selectbox("Categorical Factor (Group)", CATEGORICAL_COLS, key="t2_factor")
    metric = colB.selectbox("Numerical Metric (Target)", NUMERIC_COLS, index=3, key="t2_metric")

    levels = sorted(df_full[factor].unique())
    groups = {lvl: df_full.loc[df_full[factor] == lvl, metric] for lvl in levels}

    # Step 1: Normality Test
    norm_rows = []
    for lvl, g in groups.items():
        sw = stats.shapiro(g) if len(g) >= 3 else None
        norm_rows.append({
            "Group": lvl, "Sample Size (N)": len(g),
            "Shapiro Statistic": sw.statistic if sw else np.nan,
            "p-value": sw.pvalue if sw else np.nan,
            "Is Normal? (α=0.05)": (sw.pvalue > ALPHA) if sw else False
        })
    norm_df = pd.DataFrame(norm_rows)
    all_normal = norm_df["Is Normal? (α=0.05)"].all()

    # Step 2: Variance Homogeneity Test
    levene_stat, levene_p = stats.levene(*groups.values())
    equal_var = levene_p > ALPHA

    # Step 3: Compute Correct Statistical Test
    if len(levels) == 2:
        g1, g2 = groups[levels[0]], groups[levels[1]]
        if all_normal:
            stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
            test_name = f"Two-Sample t-test ({'Equal' if equal_var else 'Welch\'s Unequal'} Var)"
            stat_label = "t-statistic"
        else:
            stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U Test (Non-parametric)"
            stat_label = "U-statistic"
    else:
        if all_normal:
            stat, p_val = stats.f_oneway(*groups.values())
            test_name = "One-Way ANOVA"
            stat_label = "F-statistic"
        else:
            stat, p_val = stats.kruskal(*groups.values())
            test_name = "Kruskal-Wallis Test (Non-parametric)"
            stat_label = "H-statistic"

    # Stat Card Display
    c1, c2, c3, c4 = st.columns(4)
    with c1: make_card("Selected Test", test_name)
    with c2: make_card("Levene p-val (Equal Var)", f"{levene_p:.4g}")
    with c3: make_card(stat_label, f"{stat:.3f}")
    with c4: make_card("Hypothesis Test p-val", f"{p_val:.4g}")

    st.write("")
    if p_val < ALPHA:
        st.success(f"✅ **Reject H₀** (p = {p_val:.4g} < {ALPHA}). Statistically significant difference observed in **{metric}** across **{factor}** groups.")
    else:
        st.warning(f"⚖️ **Fail to Reject H₀** (p = {p_val:.4g} ≥ {ALPHA}). Insufficient evidence to show a difference in **{metric}** across **{factor}** groups.")

    # Enhanced Split Visualizations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.dataframe(norm_df.style.format({"Shapiro Statistic": "{:.3f}", "p-value": "{:.4g}"}), use_container_width=True)
    with col_v2:
        fig_violin = px.violin(
            df_full, x=factor, y=metric, color=factor, box=True, points="all",
            template="plotly_dark", title=f"Distribution Split: {metric.title()} by {factor.title()}"
        )
        fig_violin.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")
    st.subheader("B. Independence Test (Categorical × Categorical)")
    colC, colD = st.columns(2)
    catA = colC.selectbox("Categorical Factor A", CATEGORICAL_COLS, index=1, key="t2_chiA")
    catB = colD.selectbox("Categorical Factor B", CATEGORICAL_COLS, index=2, key="t2_chiB")

    if catA == catB:
        st.info("ℹ️ Select two distinct categorical features to calculate Chi-Square independence.")
    else:
        contingency = pd.crosstab(df_full[catA], df_full[catB])
        chi2, p_val2, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.values.sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1: make_card("Chi-Square (χ²)", f"{chi2:.3f}")
        with c2: make_card("Degrees of Freedom", f"{dof}")
        with c3: make_card("p-value", f"{p_val2:.4g}")
        with c4: make_card("Cramér's V Effect", f"{cramers_v:.3f}")

        st.write("")
        fig_chi = px.imshow(
            contingency, text_auto=True, color_continuous_scale="Teal",
            template="plotly_dark", title=f"Contingency Cross-Tabulation: {catA.title()} vs {catB.title()}"
        )
        st.plotly_chart(fig_chi, use_container_width=True)

# ============================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================================
with tab3:
    st.header("🔮 Model Inference & Diagnostic Lab")

    # Interactive Inputs Section
    st.subheader("🎛️ Live Prediction Simulator")
    with st.container():
        c1, c2, c3 = st.columns(3)
        in_age = c1.slider("Age", int(df_full.age.min()), int(df_full.age.max()), 35)
        in_bmi = c2.slider("BMI", float(df_full.bmi.min()), float(df_full.bmi.max()), 28.0, step=0.1)
        in_children = c3.slider("Children", 0, int(df_full.children.max()), 1)

        c4, c5, c6 = st.columns(3)
        in_sex = c4.selectbox("Sex", sorted(df_full.sex.unique()))
        in_smoker = c5.selectbox("Smoker", sorted(df_full.smoker.unique()))
        in_region = c6.selectbox("Region", sorted(df_full.region.unique()))

    # Inference Calculation
    new_X = build_design_row(in_age, in_bmi, in_children, in_sex, in_smoker, in_region)
    pred = model.get_prediction(new_X)
    summary_frame = pred.summary_frame(alpha=0.05)

    point = summary_frame["mean"].iloc[0]
    ci_lo, ci_hi = summary_frame["mean_ci_lower"].iloc[0], summary_frame["mean_ci_upper"].iloc[0]
    pi_lo, pi_hi = summary_frame["obs_ci_lower"].iloc[0], summary_frame["obs_ci_upper"].iloc[0]

    p1, p2, p3 = st.columns(3)
    with p1: make_card("Point Estimate (Predicted Charge)", f"${point:,.2f}")
    with p2: make_card("95% Mean Confidence Interval", f"${ci_lo:,.0f} – ${ci_hi:,.0f}")
    with p3: make_card("95% Individual Prediction Interval", f"${pi_lo:,.0f} – ${pi_hi:,.0f}")

    # Dynamic Waterfall / Feature Contribution Breakdown Chart
    st.write("")
    st.subheader("🧩 Linear Feature Contribution Breakdown")
    contributions = new_X.iloc[0] * model.params
    
    waterfall_fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(contributions),
        x=contributions.index,
        textposition="outside",
        text=[f"${v:+,.0f}" for v in contributions.values],
        y=contributions.values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#00d2ff"}},
        increasing={"marker": {"color": "#ff4b4b"}}
    ))
    waterfall_fig.update_layout(
        template="plotly_dark", title="Additive Linear Components to Final Prediction Target ($)",
        showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(waterfall_fig, use_container_width=True)

    st.markdown("---")

    # Diagnostics Layout
    st.subheader("📉 Gauss-Markov Residual Diagnostics")
    fitted = model.fittedvalues
    resid = model.resid

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**1. Residuals vs. Fitted Values (Homoscedasticity Check)**")
        fig_res = px.scatter(
            x=fitted, y=resid, opacity=0.5,
            labels={"x": "Fitted Values", "y": "Residuals"},
            template="plotly_dark", title="Residuals vs Fitted"
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

        bp_stat, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
        st.info(f"**Breusch-Pagan Test:** LM Stat = `{bp_stat:.3f}`, p-value = `{bp_p:.4g}` → "
                f"{'Heteroscedasticity Detected (Reject H₀)' if bp_p < ALPHA else 'Homoscedasticity Holds (Fail to Reject H₀)'}")

    with d2:
        st.markdown("**2. Normal Q-Q Plot (Residual Normality Check)**")
        qq = stats.probplot(resid, dist="norm")
        qq_x = qq[0][0]
        qq_y = qq[0][1]

        fig_qq = px.scatter(
            x=qq_x, y=qq_y, labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
            template="plotly_dark", title="Normal Q-Q Plot"
        )
        # Add reference line
        line_x = np.array([qq_x.min(), qq_x.max()])
        line_y = qq[1][1] + qq[1][0] * line_x
        fig_qq.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color="red", dash="dash"), name="Reference Line"))
        st.plotly_chart(fig_qq, use_container_width=True)

        jb_stat, jb_p, _, _ = jarque_bera(resid)
        st.info(f"**Jarque-Bera Test:** Stat = `{jb_stat:.3f}`, p-value = `{jb_p:.4g}` → "
                f"{'Non-Normal Residuals (Reject H₀)' if jb_p < ALPHA else 'Normal Residuals (Fail to Reject H₀)'}")

    # Multicollinearity Check
    st.markdown("**3. Multicollinearity Assessment (VIF)**")
    cont_X = sm.add_constant(df_full[CONTINUOUS_PREDICTORS])
    vif_df = pd.DataFrame({
        "Continuous Variable": cont_X.columns,
        "VIF Score": [variance_inflation_factor(cont_X.values, i) for i in range(cont_X.shape[1])],
    })
    
    col_vif, col_dw = st.columns([2, 1])
    with col_vif:
        st.dataframe(vif_df.round(3).style.format({"VIF Score": "{:.3f}"}), use_container_width=True)
    with col_dw:
        dw = durbin_watson(resid)
        make_card("Durbin-Watson Stat", f"{dw:.3f}")
        st.caption("Values near 2 indicate no autocorrelation in residuals.")

    with st.expander("📄 Full OLS Regression Table"):
        st.text(model.summary().as_text())

st.markdown("---")
st.caption("Enhanced Dashboard • Built with Streamlit & Plotly")