"""
Tanzania Infant Mortality Prediction — Streamlit App
Regression: Linear Regression & Decision Tree Regressor

Run:  streamlit run app.py
Needs in same folder:
  lr_model.pkl, dt_model.pkl, scaler.pkl,
  le_area.pkl, le_region.pkl, model_meta.json,
  tanzania_infant_mortality_dataset.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tanzania Infant Mortality Predictor",
    page_icon="🇹🇿",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #FFC107 100%);
    padding: 22px 30px; border-radius: 14px; margin-bottom: 24px;
    color: white; text-align: center;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 6px 0 0; opacity: 0.9; font-size: 1.05rem; }
.result-high { background:#FFF3E0; border-left:6px solid #FF9800; border-radius:10px; padding:18px; }
.result-low  { background:#E8F5E9; border-left:6px solid #4CAF50; border-radius:10px; padding:18px; }
.section-hdr { font-size:1.2rem; font-weight:700; color:#1B5E20;
               border-bottom:2px solid #4CAF50; padding-bottom:5px; margin:18px 0 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🇹🇿 Tanzania Infant Mortality Prediction</h1>
    <p>Linear Regression &amp; Decision Tree Regressor — Predicting deaths per 1,000 live births</p>
</div>
""", unsafe_allow_html=True)

# ── Load / train models ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pkl_files = ['lr_model.pkl','dt_model.pkl','scaler.pkl',
                 'le_area.pkl','le_region.pkl','model_meta.json']
    if all(os.path.exists(f) for f in pkl_files):
        lr   = joblib.load('lr_model.pkl')
        dt   = joblib.load('dt_model.pkl')
        sc   = joblib.load('scaler.pkl')
        le_a = joblib.load('le_area.pkl')
        le_r = joblib.load('le_region.pkl')
        with open('model_meta.json') as f:
            meta = json.load(f)
        return lr, dt, sc, le_a, le_r, meta, None

    ds = 'tanzania_infant_mortality_dataset.xlsx'
    if not os.path.exists(ds):
        return None,None,None,None,None,None, \
            "Dataset not found. Place tanzania_infant_mortality_dataset.xlsx in the same folder."

    df = pd.read_excel(ds)
    df.columns = ['ID','Region','Area','Skilled_Birth_Pct',
                  'Clinic_Vaccination_Pct','Clean_Water_Pct','Infant_Mortality']

    le_a = LabelEncoder().fit(df['Area'])
    le_r = LabelEncoder().fit(df['Region'])
    df['Area_Encoded']   = le_a.transform(df['Area'])
    df['Region_Encoded'] = le_r.transform(df['Region'])

    FEATS = ['Skilled_Birth_Pct','Clinic_Vaccination_Pct','Clean_Water_Pct',
             'Area_Encoded','Region_Encoded']
    X = df[FEATS]; y = df['Infant_Mortality']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc  = StandardScaler().fit(Xtr)
    lr  = LinearRegression().fit(sc.transform(Xtr), ytr)
    dt  = DecisionTreeRegressor(random_state=42, max_depth=5,
                                min_samples_split=10, min_samples_leaf=5).fit(Xtr, ytr)

    lp = lr.predict(sc.transform(Xte))
    dp = dt.predict(Xte)

    meta = {
        'features': FEATS,
        'area_classes':   le_a.classes_.tolist(),
        'region_classes': le_r.classes_.tolist(),
        'y_min': float(y.min()), 'y_max': float(y.max()), 'y_mean': float(y.mean()),
        'lr_r2':   round(r2_score(yte,lp),4),
        'dt_r2':   round(r2_score(yte,dp),4),
        'lr_rmse': round(float(np.sqrt(mean_squared_error(yte,lp))),4),
        'dt_rmse': round(float(np.sqrt(mean_squared_error(yte,dp))),4),
        'lr_mae':  round(mean_absolute_error(yte,lp),4),
        'dt_mae':  round(mean_absolute_error(yte,dp),4),
    }
    joblib.dump(lr,  'lr_model.pkl');  joblib.dump(dt,  'dt_model.pkl')
    joblib.dump(sc,  'scaler.pkl');    joblib.dump(le_a,'le_area.pkl')
    joblib.dump(le_r,'le_region.pkl')
    with open('model_meta.json','w') as f: json.dump(meta, f)
    return lr, dt, sc, le_a, le_r, meta, None

lr_model, dt_model, scaler, le_area, le_region, meta, err = load_models()
if err:
    st.error(err); st.stop()

FEATURES       = meta['features']
AREA_CLASSES   = meta['area_classes']
REGION_CLASSES = meta['region_classes']

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔮 Predict Infant Mortality")
st.sidebar.markdown("Adjust the community health indicators:")

model_choice  = st.sidebar.radio("Select Model",
    ["Linear Regression", "Decision Tree Regressor"])
skilled_birth = st.sidebar.slider("Skilled Birth Attendance (%)", 0.0, 100.0, 65.0, 0.5)
vaccination   = st.sidebar.slider("Clinic Vaccination Rate (%)",  0.0, 100.0, 70.0, 0.5)
clean_water   = st.sidebar.slider("Access to Clean Water (%)",    0.0, 100.0, 60.0, 0.5)
area          = st.sidebar.selectbox("Area Type",  AREA_CLASSES)
region        = st.sidebar.selectbox("Region",     REGION_CLASSES)

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(skilled, vacc, water, area_v, region_v, model_name):
    ae  = le_area.transform([area_v])[0]
    re  = le_region.transform([region_v])[0]
    Xi  = pd.DataFrame([[skilled, vacc, water, ae, re]], columns=FEATURES)
    if model_name == "Linear Regression":
        val = lr_model.predict(scaler.transform(Xi))[0]
    else:
        val = dt_model.predict(Xi)[0]
    return max(0.0, round(float(val), 2))

pred_val = predict(skilled_birth, vaccination, clean_water, area, region, model_choice)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "📊 Model Performance", "🌳 Decision Tree", "📈 Feature Analysis"])

# ════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-hdr">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Indicator": ["Skilled Birth (%)","Vaccination (%)","Clean Water (%)","Area","Region","Model"],
            "Value":     [f"{skilled_birth:.1f}%", f"{vaccination:.1f}%",
                          f"{clean_water:.1f}%", area, region, model_choice]
        }), hide_index=True, use_container_width=True)

        fig_bar = px.bar(
            x=[skilled_birth, vaccination, clean_water],
            y=["Skilled Birth","Vaccination","Clean Water"],
            orientation='h',
            color=["Skilled Birth","Vaccination","Clean Water"],
            color_discrete_sequence=["#2196F3","#4CAF50","#FF9800"],
            labels={"x":"Percentage (%)","y":""},
            title="Health Indicator Profile",
            text=[f"{v:.0f}%" for v in [skilled_birth, vaccination, clean_water]]
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, xaxis_range=[0,115],
                              height=280, margin=dict(l=0,r=10,t=40,b=0))
        fig_bar.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-hdr">Prediction Result</div>', unsafe_allow_html=True)

        risk     = "HIGH" if pred_val >= 30 else "LOW"
        box_cls  = "result-high" if pred_val >= 30 else "result-low"
        icon     = "🔴" if pred_val >= 30 else "🟢"
        mean_v   = meta['y_mean']
        diff_str = (f"+{pred_val-mean_v:.1f}" if pred_val >= mean_v
                    else f"{pred_val-mean_v:.1f}")

        st.markdown(f"""
        <div class="{box_cls}">
            <span style="font-size:1.6rem;font-weight:700">{icon} {pred_val:.1f} per 1,000 live births</span><br>
            <b>Risk Level:</b> {risk} &nbsp;(threshold: 30)<br>
            <b>vs. National Mean ({mean_v:.1f}):</b> {diff_str}<br>
            <b>Model:</b> {model_choice}
        </div>
        """, unsafe_allow_html=True)

        y_min, y_max = meta['y_min'], meta['y_max']
        gauge_color  = "#F44336" if pred_val >= 30 else "#4CAF50"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_val,
            title={"text": "Predicted Mortality (per 1,000)"},
            gauge={
                "axis": {"range": [y_min, y_max]},
                "bar":  {"color": gauge_color, "thickness": 0.3},
                "steps": [
                    {"range": [y_min, 30],   "color": "#C8E6C9"},
                    {"range": [30,   y_max], "color": "#FFCCBC"},
                ],
                "threshold": {
                    "line":      {"color": "blue", "width": 3},
                    "thickness": 0.8,
                    "value":     mean_v
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        lr_v = predict(skilled_birth, vaccination, clean_water, area, region, "Linear Regression")
        dt_v = predict(skilled_birth, vaccination, clean_water, area, region, "Decision Tree Regressor")
        st.markdown("**Both Model Predictions**")
        st.dataframe(pd.DataFrame({
            "Model":      ["Linear Regression","Decision Tree Regressor"],
            "Predicted":  [f"{lr_v:.2f}", f"{dt_v:.2f}"],
            "Risk Level": ["HIGH 🔴" if lr_v>=30 else "LOW 🟢",
                           "HIGH 🔴" if dt_v>=30 else "LOW 🟢"]
        }), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Test Set Metrics</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("LR — R²",   f"{meta['lr_r2']:.4f}")
    c2.metric("LR — RMSE", f"{meta['lr_rmse']:.4f}")
    c3.metric("LR — MAE",  f"{meta['lr_mae']:.4f}")
    c4,c5,c6 = st.columns(3)
    c4.metric("DT — R²",   f"{meta['dt_r2']:.4f}",
              delta=f"{meta['dt_r2']-meta['lr_r2']:+.4f} vs LR")
    c5.metric("DT — RMSE", f"{meta['dt_rmse']:.4f}")
    c6.metric("DT — MAE",  f"{meta['dt_mae']:.4f}")

    @st.cache_data
    def get_test_results():
        ds = 'tanzania_infant_mortality_dataset.xlsx'
        if not os.path.exists(ds): return None
        df = pd.read_excel(ds)
        df.columns = ['ID','Region','Area','Skilled_Birth_Pct',
                      'Clinic_Vaccination_Pct','Clean_Water_Pct','Infant_Mortality']
        df['Area_Encoded']   = le_area.transform(df['Area'])
        df['Region_Encoded'] = le_region.transform(df['Region'])
        X = df[FEATURES]; y = df['Infant_Mortality']
        _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        lp = lr_model.predict(scaler.transform(Xte))
        dp = dt_model.predict(Xte)
        return yte.values, lp, dp

    res = get_test_results()
    if res:
        y_test, lr_pred, dt_pred = res
        lims = [float(min(y_test.min(), lr_pred.min(), dt_pred.min()))-1,
                float(max(y_test.max(), lr_pred.max(), dt_pred.max()))+1]

        fig_avp = make_subplots(rows=1, cols=2,
            subplot_titles=["Linear Regression — Actual vs Predicted",
                            "Decision Tree — Actual vs Predicted"])
        fig_avp.add_trace(go.Scatter(x=y_test.tolist(), y=lr_pred.tolist(), mode='markers',
            marker=dict(color='#2196F3', opacity=0.5, size=6), name='LR'), row=1, col=1)
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
            line=dict(color='red', dash='dash'), showlegend=False), row=1, col=1)
        fig_avp.add_trace(go.Scatter(x=y_test.tolist(), y=dt_pred.tolist(), mode='markers',
            marker=dict(color='#FF9800', opacity=0.5, size=6), name='DT'), row=1, col=2)
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
            line=dict(color='red', dash='dash'), showlegend=False), row=1, col=2)
        fig_avp.update_xaxes(title_text="Actual")
        fig_avp.update_yaxes(title_text="Predicted")
        fig_avp.update_layout(height=420)
        st.plotly_chart(fig_avp, use_container_width=True)

        lr_res = (y_test - lr_pred).tolist()
        dt_res = (y_test - dt_pred).tolist()
        fig_res = make_subplots(rows=1, cols=2,
            subplot_titles=["LR Residual Plot","DT Residual Plot"])
        fig_res.add_trace(go.Scatter(x=lr_pred.tolist(), y=lr_res, mode='markers',
            marker=dict(color='#2196F3', opacity=0.5, size=6), name='LR'), row=1, col=1)
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.add_trace(go.Scatter(x=dt_pred.tolist(), y=dt_res, mode='markers',
            marker=dict(color='#FF9800', opacity=0.5, size=6), name='DT'), row=1, col=2)
        fig_res.update_xaxes(title_text="Predicted")
        fig_res.update_yaxes(title_text="Residual")
        fig_res.update_layout(height=380)
        st.plotly_chart(fig_res, use_container_width=True)

        fig_hist = make_subplots(rows=1, cols=2,
            subplot_titles=["LR Residual Distribution","DT Residual Distribution"])
        fig_hist.add_trace(go.Histogram(x=lr_res, nbinsx=30,
            marker_color='#2196F3', opacity=0.85, name='LR'), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=dt_res, nbinsx=30,
            marker_color='#FF9800', opacity=0.85, name='DT'), row=1, col=2)
        fig_hist.update_xaxes(title_text="Residual")
        fig_hist.update_yaxes(title_text="Count")
        fig_hist.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — Decision Tree
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Feature Importances — Decision Tree</div>',
                unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        'Feature':    FEATURES,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Greens',
        title='Feature Importances (Decision Tree Regressor)',
        text=imp_df['Importance'].round(4))
    fig_imp.update_traces(textposition='outside')
    fig_imp.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(l=0,r=60,t=50,b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        imp_df.sort_values('Importance', ascending=False).round(4),
        hide_index=True, use_container_width=True)

    st.info("💡 To view the full tree diagram, run the Jupyter notebook "
            "`tanzania_infant_mortality_model.ipynb` — the tree renders best there.")

# ════════════════════════════════════════════
# TAB 4 — Feature Analysis
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">Linear Regression Coefficients</div>',
                unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        'Feature':     FEATURES,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient')
    coef_df['Direction'] = coef_df['Coefficient'].apply(
        lambda x: 'Increases mortality' if x > 0 else 'Decreases mortality')

    fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
        color='Direction',
        color_discrete_map={
            'Increases mortality': '#F44336',
            'Decreases mortality': '#2196F3'
        },
        title='LR Coefficients — Effect on Infant Mortality Rate (standardised features)',
        text=coef_df['Coefficient'].round(3))
    fig_coef.update_traces(textposition='outside')
    fig_coef.add_vline(x=0, line_color='black', line_width=1.5)
    fig_coef.update_layout(height=400, margin=dict(l=0,r=60,t=50,b=0))
    st.plotly_chart(fig_coef, use_container_width=True)

    st.dataframe(
        coef_df[['Feature','Coefficient','Direction']].sort_values(
            'Coefficient', key=abs, ascending=False).round(4),
        hide_index=True, use_container_width=True)

    st.markdown(f"""
**Intercept:** `{lr_model.intercept_:.4f}`

**Interpretation:**
- 🔴 **Positive coefficient** → increasing this feature *raises* predicted infant mortality  
- 🔵 **Negative coefficient** → increasing this feature *lowers* predicted infant mortality  
- Coefficients are on **standardised** features, so magnitudes are directly comparable
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>🇹🇿 Tanzania Infant Mortality Regression | "
    "Linear Regression &amp; Decision Tree | scikit-learn + Streamlit</p>",
    unsafe_allow_html=True
)
