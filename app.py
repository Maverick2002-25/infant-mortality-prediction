"""
Tanzania Infant Mortality Prediction — Streamlit App
Models: Linear Regression & Decision Tree Regressor
Self-contained: trains from dataset on startup, no pkl files needed.

Deploy to Streamlit Cloud:
  Repo must contain: app.py, requirements.txt, tanzania_infant_mortality_dataset.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇹🇿 Tanzania Infant Mortality",
    page_icon="🇹🇿",
    layout="wide",
)

st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg,#1B5E20,#388E3C,#FFC107);
    padding:24px 32px; border-radius:14px; color:white; text-align:center; margin-bottom:20px;
}
.header-box h1 { margin:0; font-size:2rem; }
.header-box p  { margin:6px 0 0; opacity:.9; }
.result-high { background:#FFF3E0; border-left:6px solid #FF9800; border-radius:10px; padding:18px; line-height:2; }
.result-low  { background:#E8F5E9; border-left:6px solid #4CAF50; border-radius:10px; padding:18px; line-height:2; }
.sec { font-size:1.15rem; font-weight:700; color:#1B5E20;
       border-bottom:2px solid #4CAF50; padding-bottom:4px; margin:16px 0 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
  <h1>🇹🇿 Tanzania Infant Mortality Prediction</h1>
  <p>Linear Regression &amp; Decision Tree Regressor — deaths per 1,000 live births</p>
</div>
""", unsafe_allow_html=True)

def render_best_banner(best_model, best_r2, best_rmse, best_mae):
    st.markdown(f"""
    <div style="background:#E8F5E9;border:2px solid #2E7D32;border-radius:12px;
                padding:16px 24px;margin-bottom:18px;">
      <span style="font-size:1.3rem;font-weight:800;color:#1B5E20">
        🏆 Best Performing Model: {best_model}
      </span><br>
      <span style="color:#333;font-size:0.97rem">
        R² = <b>{best_r2:.4f}</b> &nbsp;|&nbsp;
        RMSE = <b>{best_rmse:.4f}</b> &nbsp;|&nbsp;
        MAE = <b>{best_mae:.4f}</b>
        &nbsp;&nbsp;— auto-selected by highest R² on the 20% test set
      </span>
    </div>
    """, unsafe_allow_html=True)

# ── Load data & train ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and training models…")
def train():
    df = pd.read_excel("tanzania_infant_mortality_dataset.xlsx")
    df.columns = [
        "ID", "Region", "Area",
        "Skilled_Birth_Pct", "Clinic_Vaccination_Pct",
        "Clean_Water_Pct", "Infant_Mortality"
    ]

    # Encode categoricals
    le_area   = LabelEncoder().fit(df["Area"])
    le_region = LabelEncoder().fit(df["Region"])
    df["Area_Enc"]   = le_area.transform(df["Area"])
    df["Region_Enc"] = le_region.transform(df["Region"])

    FEATS = ["Skilled_Birth_Pct", "Clinic_Vaccination_Pct",
             "Clean_Water_Pct", "Area_Enc", "Region_Enc"]

    X = df[FEATS].values
    y = df["Infant_Mortality"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc  = StandardScaler().fit(Xtr)
    lr  = LinearRegression().fit(sc.transform(Xtr), ytr)
    dt  = DecisionTreeRegressor(
            max_depth=5, min_samples_split=10,
            min_samples_leaf=5, random_state=42
          ).fit(Xtr, ytr)

    lp = lr.predict(sc.transform(Xte))
    dp = dt.predict(Xte)

    metrics = {
        "lr_r2":   round(float(r2_score(yte, lp)), 4),
        "lr_rmse": round(float(np.sqrt(mean_squared_error(yte, lp))), 4),
        "lr_mae":  round(float(mean_absolute_error(yte, lp)), 4),
        "dt_r2":   round(float(r2_score(yte, dp)), 4),
        "dt_rmse": round(float(np.sqrt(mean_squared_error(yte, dp))), 4),
        "dt_mae":  round(float(mean_absolute_error(yte, dp)), 4),
        "y_min":   float(y.min()),
        "y_max":   float(y.max()),
        "y_mean":  float(y.mean()),
    }

    # store test set for charts
    test_df = pd.DataFrame(Xte, columns=FEATS)
    test_df["Actual"]   = yte
    test_df["LR_Pred"]  = lp
    test_df["DT_Pred"]  = dp
    test_df["LR_Resid"] = yte - lp
    test_df["DT_Resid"] = yte - dp

    return (lr, dt, sc, le_area, le_region,
            FEATS, metrics, test_df,
            sorted(le_area.classes_.tolist()),
            sorted(le_region.classes_.tolist()),
            df)

(lr_model, dt_model, scaler,
 le_area, le_region,
 FEATS, M, test_df,
 AREAS, REGIONS, full_df) = train()

# ── Determine best model automatically ───────────────────────────────────────
BEST_MODEL = (
    "Linear Regression"
    if M["lr_r2"] >= M["dt_r2"]
    else "Decision Tree Regressor"
)
BEST_R2   = max(M["lr_r2"], M["dt_r2"])
BEST_RMSE = M["lr_rmse"] if BEST_MODEL == "Linear Regression" else M["dt_rmse"]
BEST_MAE  = M["lr_mae"]  if BEST_MODEL == "Linear Regression" else M["dt_mae"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Make a Prediction")
    st.success(f"🏆 Best model: **{BEST_MODEL}**")
    model_choice = st.radio(
        "Model",
        ["Linear Regression", "Decision Tree Regressor"],
        index=0 if BEST_MODEL == "Linear Regression" else 1
    )
    st.markdown("---")
    skilled = st.slider("Skilled Birth Attendance (%)", 0.0, 100.0, 65.0, 0.5)
    vacc    = st.slider("Clinic Vaccination Rate (%)",  0.0, 100.0, 70.0, 0.5)
    water   = st.slider("Access to Clean Water (%)",    0.0, 100.0, 60.0, 0.5)
    area    = st.selectbox("Area Type",  AREAS)
    region  = st.selectbox("Region",     REGIONS)

def make_pred(s, v, w, a, r, model_name):
    xi = np.array([[s, v, w,
                    le_area.transform([a])[0],
                    le_region.transform([r])[0]]], dtype=float)
    if model_name == "Linear Regression":
        return max(0.0, round(float(lr_model.predict(scaler.transform(xi))[0]), 2))
    return max(0.0, round(float(dt_model.predict(xi)[0]), 2))

pred = make_pred(skilled, vacc, water, area, region, model_choice)
lr_p = make_pred(skilled, vacc, water, area, region, "Linear Regression")
dt_p = make_pred(skilled, vacc, water, area, region, "Decision Tree Regressor")

# ── Best model banner ─────────────────────────────────────────────────────────
render_best_banner(BEST_MODEL, BEST_R2, BEST_RMSE, BEST_MAE)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🔮 Prediction", "📊 Model Performance",
    "🌳 Feature Importance", "📈 Feature Analysis"
])

# ════════════════════════════════════════════════════
# TAB 1 – Prediction
# ════════════════════════════════════════════════════
with t1:
    left, right = st.columns(2)

    with left:
        st.markdown('<p class="sec">Input Summary</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Indicator": ["Skilled Birth (%)","Vaccination (%)","Clean Water (%)","Area","Region","Model"],
            "Value":     [f"{skilled:.1f}%", f"{vacc:.1f}%", f"{water:.1f}%",
                          area, region, model_choice]
        }), hide_index=True, use_container_width=True)

        fig_bar = go.Figure(go.Bar(
            x=[skilled, vacc, water],
            y=["Skilled Birth", "Vaccination", "Clean Water"],
            orientation="h",
            marker_color=["#2196F3", "#4CAF50", "#FF9800"],
            text=[f"{v:.0f}%" for v in [skilled, vacc, water]],
            textposition="outside"
        ))
        fig_bar.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.4)
        fig_bar.update_layout(
            title="Health Indicator Profile",
            xaxis=dict(range=[0, 118], title="Percentage (%)"),
            yaxis_title="",
            height=260, showlegend=False,
            margin=dict(l=0, r=40, t=40, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown('<p class="sec">Prediction Result</p>', unsafe_allow_html=True)

        mean_v   = M["y_mean"]
        risk     = "HIGH" if pred >= 30 else "LOW"
        box_cls  = "result-high" if pred >= 30 else "result-low"
        icon     = "🔴" if pred >= 30 else "🟢"
        diff     = pred - mean_v
        diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
        best_badge = " &nbsp;<span style='background:#2E7D32;color:white;border-radius:6px;padding:2px 8px;font-size:0.8rem'>🏆 Best Model</span>" if model_choice == BEST_MODEL else ""

        st.markdown(f"""
        <div class="{box_cls}">
          <span style="font-size:1.7rem;font-weight:800">{icon} {pred:.1f} per 1,000 live births</span><br>
          <b>Risk Level:</b> {risk} &emsp;(threshold ≥ 30)<br>
          <b>vs. National Mean ({mean_v:.1f}):</b> {diff_str}<br>
          <b>Model used:</b> {model_choice}{best_badge}
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        gcol = "#F44336" if pred >= 30 else "#4CAF50"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Mortality (per 1,000)"},
            gauge={
                "axis":  {"range": [M["y_min"], M["y_max"]]},
                "bar":   {"color": gcol, "thickness": 0.28},
                "steps": [
                    {"range": [M["y_min"], 30],    "color": "#C8E6C9"},
                    {"range": [30, M["y_max"]],    "color": "#FFCCBC"},
                ],
                "threshold": {
                    "line": {"color": "#1565C0", "width": 3},
                    "thickness": 0.8, "value": mean_v
                }
            }
        ))
        fig_g.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("**Both models side-by-side:**")
        st.dataframe(pd.DataFrame({
            "Model":     [f"Linear Regression {'🏆' if BEST_MODEL=='Linear Regression' else ''}",
                          f"Decision Tree Regressor {'🏆' if BEST_MODEL=='Decision Tree Regressor' else ''}"],
            "Predicted": [f"{lr_p:.2f}", f"{dt_p:.2f}"],
            "Risk":      ["HIGH 🔴" if lr_p >= 30 else "LOW 🟢",
                          "HIGH 🔴" if dt_p >= 30 else "LOW 🟢"]
        }), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 2 – Model Performance
# ════════════════════════════════════════════════════
with t2:
    st.markdown('<p class="sec">Model Comparison — Test Set (80/20 split)</p>',
                unsafe_allow_html=True)

    # ── comparison table ──────────────────────────────────────────────────────
    lr_badge = "🏆 WINNER" if BEST_MODEL == "Linear Regression"      else ""
    dt_badge = "🏆 WINNER" if BEST_MODEL == "Decision Tree Regressor" else ""

    comp_df = pd.DataFrame({
        "Model":  [f"Linear Regression {lr_badge}",
                   f"Decision Tree Regressor {dt_badge}"],
        "R²  ↑ (higher=better)":   [M["lr_r2"],   M["dt_r2"]],
        "RMSE ↓ (lower=better)":   [M["lr_rmse"], M["dt_rmse"]],
        "MAE  ↓ (lower=better)":   [M["lr_mae"],  M["dt_mae"]],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ── bar chart comparison ──────────────────────────────────────────────────
    bar_df = pd.DataFrame({
        "Metric": ["R²", "R²", "RMSE", "RMSE", "MAE", "MAE"],
        "Model":  ["Linear Regression", "Decision Tree Regressor"] * 3,
        "Value":  [M["lr_r2"], M["dt_r2"],
                   M["lr_rmse"], M["dt_rmse"],
                   M["lr_mae"], M["dt_mae"]],
    })
    fig_cmp = px.bar(
        bar_df, x="Metric", y="Value", color="Model", barmode="group",
        color_discrete_map={
            "Linear Regression":      "#2196F3",
            "Decision Tree Regressor":"#FF9800"
        },
        text="Value",
        title="Model Comparison — R², RMSE, MAE"
    )
    fig_cmp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_cmp.update_layout(height=420, yaxis_title="Score",
                          margin=dict(t=50, b=0))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── individual metric cards ───────────────────────────────────────────────
    st.markdown('<p class="sec">Detailed Metrics</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("LR — R²",   f"{M['lr_r2']:.4f}",
              delta="✅ BEST" if BEST_MODEL == "Linear Regression" else None)
    c2.metric("LR — RMSE", f"{M['lr_rmse']:.4f}")
    c3.metric("LR — MAE",  f"{M['lr_mae']:.4f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("DT — R²",   f"{M['dt_r2']:.4f}",
              delta="✅ BEST" if BEST_MODEL == "Decision Tree Regressor" else
                    f"{M['dt_r2'] - M['lr_r2']:+.4f} vs LR")
    c5.metric("DT — RMSE", f"{M['dt_rmse']:.4f}")
    c6.metric("DT — MAE",  f"{M['dt_mae']:.4f}")

    st.markdown('<p class="sec">Actual vs Predicted</p>', unsafe_allow_html=True)
    lims = [float(test_df["Actual"].min()) - 1,
            float(test_df["Actual"].max()) + 1]

    fig_avp = make_subplots(rows=1, cols=2,
        subplot_titles=["Linear Regression", "Decision Tree Regressor"])
    for col_n, col_pred, color, ci in [
        ("LR_Pred", "LR_Pred", "#2196F3", 1),
        ("DT_Pred", "DT_Pred", "#FF9800", 2)
    ]:
        fig_avp.add_trace(go.Scatter(
            x=test_df["Actual"].tolist(), y=test_df[col_pred].tolist(),
            mode="markers", marker=dict(color=color, size=6, opacity=0.55),
            name=col_n), row=1, col=ci)
        fig_avp.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="red", dash="dash", width=2), showlegend=False),
            row=1, col=ci)
    fig_avp.update_xaxes(title_text="Actual")
    fig_avp.update_yaxes(title_text="Predicted")
    fig_avp.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown('<p class="sec">Residual Plots</p>', unsafe_allow_html=True)
    fig_res = make_subplots(rows=1, cols=2,
        subplot_titles=["LR Residuals", "DT Residuals"])
    fig_res.add_trace(go.Scatter(
        x=test_df["LR_Pred"].tolist(), y=test_df["LR_Resid"].tolist(),
        mode="markers", marker=dict(color="#2196F3", size=6, opacity=0.55),
        name="LR"), row=1, col=1)
    fig_res.add_trace(go.Scatter(
        x=test_df["DT_Pred"].tolist(), y=test_df["DT_Resid"].tolist(),
        mode="markers", marker=dict(color="#FF9800", size=6, opacity=0.55),
        name="DT"), row=1, col=2)
    for ci in [1, 2]:
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=ci)
    fig_res.update_xaxes(title_text="Predicted")
    fig_res.update_yaxes(title_text="Residual")
    fig_res.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown('<p class="sec">Residual Distributions</p>', unsafe_allow_html=True)
    fig_hist = make_subplots(rows=1, cols=2,
        subplot_titles=["LR Residuals", "DT Residuals"])
    fig_hist.add_trace(go.Histogram(
        x=test_df["LR_Resid"].tolist(), nbinsx=30,
        marker_color="#2196F3", opacity=0.85, name="LR"), row=1, col=1)
    fig_hist.add_trace(go.Histogram(
        x=test_df["DT_Resid"].tolist(), nbinsx=30,
        marker_color="#FF9800", opacity=0.85, name="DT"), row=1, col=2)
    fig_hist.update_xaxes(title_text="Residual")
    fig_hist.update_yaxes(title_text="Count")
    fig_hist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 3 – Feature Importance (Decision Tree)
# ════════════════════════════════════════════════════
with t3:
    st.markdown('<p class="sec">Decision Tree — Feature Importances</p>', unsafe_allow_html=True)

    feat_labels = [
        "Skilled Birth (%)", "Vaccination (%)",
        "Clean Water (%)", "Area Type", "Region"
    ]
    imp_df = pd.DataFrame({
        "Feature":    feat_labels,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Greens",
        text=imp_df["Importance"].round(4),
        title="Feature Importances — Decision Tree Regressor"
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        height=380, coloraxis_showscale=False,
        margin=dict(l=0, r=60, t=50, b=0)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        imp_df.sort_values("Importance", ascending=False).round(4),
        hide_index=True, use_container_width=True
    )

    st.markdown('<p class="sec">EDA — Feature vs Infant Mortality</p>', unsafe_allow_html=True)
    num_feats = ["Skilled_Birth_Pct", "Clinic_Vaccination_Pct", "Clean_Water_Pct"]
    feat_display = ["Skilled Birth (%)", "Vaccination (%)", "Clean Water (%)"]
    colors_eda   = ["#2196F3", "#4CAF50", "#FF9800"]

    fig_eda = make_subplots(rows=1, cols=3,
        subplot_titles=feat_display)
    for i, (feat, color) in enumerate(zip(num_feats, colors_eda), 1):
        fig_eda.add_trace(go.Scatter(
            x=full_df[feat].tolist(),
            y=full_df["Infant_Mortality"].tolist(),
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.45),
            name=feat), row=1, col=i)
        # trendline
        z = np.polyfit(full_df[feat], full_df["Infant_Mortality"], 1)
        xl = np.linspace(full_df[feat].min(), full_df[feat].max(), 80)
        fig_eda.add_trace(go.Scatter(
            x=xl.tolist(), y=np.poly1d(z)(xl).tolist(),
            mode="lines", line=dict(color="red", width=2), showlegend=False),
            row=1, col=i)
    fig_eda.update_xaxes(title_text="Value (%)")
    fig_eda.update_yaxes(title_text="Infant Mortality", col=1)
    fig_eda.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_eda, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 4 – Feature Analysis (LR Coefficients)
# ════════════════════════════════════════════════════
with t4:
    st.markdown('<p class="sec">Linear Regression — Coefficients</p>', unsafe_allow_html=True)

    feat_labels = [
        "Skilled Birth (%)", "Vaccination (%)",
        "Clean Water (%)", "Area Type", "Region"
    ]
    coef_df = pd.DataFrame({
        "Feature":     feat_labels,
        "Coefficient": lr_model.coef_
    }).sort_values("Coefficient")
    coef_df["Effect"] = coef_df["Coefficient"].apply(
        lambda x: "Increases mortality" if x > 0 else "Decreases mortality"
    )

    fig_coef = px.bar(
        coef_df, x="Coefficient", y="Feature", orientation="h",
        color="Effect",
        color_discrete_map={
            "Increases mortality": "#F44336",
            "Decreases mortality": "#2196F3"
        },
        text=coef_df["Coefficient"].round(3),
        title="LR Coefficients (standardised features) — Effect on Infant Mortality"
    )
    fig_coef.update_traces(textposition="outside")
    fig_coef.add_vline(x=0, line_color="black", line_width=1.5)
    fig_coef.update_layout(
        height=400, margin=dict(l=0, r=70, t=50, b=0)
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.dataframe(
        coef_df[["Feature","Coefficient","Effect"]].sort_values(
            "Coefficient", key=abs, ascending=False).round(4),
        hide_index=True, use_container_width=True
    )

    st.markdown(f"""
**Intercept:** `{lr_model.intercept_:.4f}`

**How to read:**
- 🔴 **Positive** → raising this feature *increases* predicted infant mortality  
- 🔵 **Negative** → raising this feature *decreases* predicted infant mortality  
- Features are **standardised** before fitting, so coefficient sizes are directly comparable
""")

    st.markdown('<p class="sec">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = full_df[["Skilled_Birth_Pct","Clinic_Vaccination_Pct",
                    "Clean_Water_Pct","Infant_Mortality"]].corr().round(3)
    fig_heat = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdYlGn_r",
        title="Correlation Matrix", aspect="auto"
    )
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:0.9rem'>"
    "🇹🇿 Tanzania Infant Mortality | Linear Regression &amp; Decision Tree | "
    "scikit-learn + Streamlit + Plotly</p>",
    unsafe_allow_html=True
)
