"""
Tanzania Infant Mortality Prediction — Streamlit App
Regression: Linear Regression & Decision Tree Regressor

Run:  streamlit run app.py
Needs: lr_model.pkl, dt_model.pkl, scaler.pkl, le_area.pkl, le_region.pkl, model_meta.json
       tanzania_infant_mortality_dataset.xlsx (same folder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
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
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #FFC107 100%);
    padding: 22px 30px; border-radius: 14px; margin-bottom: 24px;
    color: white; text-align: center;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 6px 0 0; opacity: 0.9; }
.result-box {
    border-radius: 12px; padding: 20px; margin-top: 12px;
    font-size: 1.1rem; line-height: 1.8;
}
.high-result { background:#FFF3E0; border-left: 6px solid #FF9800; }
.low-result  { background:#E8F5E9; border-left: 6px solid #4CAF50; }
.section-hdr { font-size:1.25rem; font-weight:700; color:#1B5E20;
               border-bottom:2px solid #4CAF50; padding-bottom:5px; margin:18px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🇹🇿 Tanzania Infant Mortality Prediction</h1>
    <p>Linear Regression & Decision Tree Regressor — Predicting deaths per 1,000 live births</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = ['lr_model.pkl','dt_model.pkl','scaler.pkl','le_area.pkl','le_region.pkl','model_meta.json']
    if all(os.path.exists(f) for f in required):
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
        return None,None,None,None,None,None,"⚠️ Dataset not found. Place the .xlsx file in the same folder as app.py"

    df = pd.read_excel(ds)
    df.columns = ['ID','Region','Area','Skilled_Birth_Pct','Clinic_Vaccination_Pct','Clean_Water_Pct','Infant_Mortality']

    le_a = LabelEncoder().fit(df['Area'])
    le_r = LabelEncoder().fit(df['Region'])
    df['Area_Encoded']   = le_a.transform(df['Area'])
    df['Region_Encoded'] = le_r.transform(df['Region'])

    FEATS = ['Skilled_Birth_Pct','Clinic_Vaccination_Pct','Clean_Water_Pct','Area_Encoded','Region_Encoded']
    X = df[FEATS]; y = df['Infant_Mortality']
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    sc  = StandardScaler().fit(Xtr)
    lr  = LinearRegression().fit(sc.transform(Xtr), ytr)
    dt  = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5).fit(Xtr, ytr)

    Xt, yt = train_test_split(X, test_size=0.2, random_state=42)[1], train_test_split(y, test_size=0.2, random_state=42)[1]
    lp = lr.predict(sc.transform(Xt)); dp = dt.predict(Xt)

    meta = {
        'features': FEATS,
        'area_classes': le_a.classes_.tolist(),
        'region_classes': le_r.classes_.tolist(),
        'y_min': float(y.min()), 'y_max': float(y.max()), 'y_mean': float(y.mean()),
        'lr_r2': round(r2_score(yt,lp),4), 'dt_r2': round(r2_score(yt,dp),4),
        'lr_rmse': round(float(np.sqrt(mean_squared_error(yt,lp))),4),
        'dt_rmse': round(float(np.sqrt(mean_squared_error(yt,dp))),4),
        'lr_mae': round(mean_absolute_error(yt,lp),4),
        'dt_mae': round(mean_absolute_error(yt,dp),4),
    }
    return lr, dt, sc, le_a, le_r, meta, None

lr_model, dt_model, scaler, le_area, le_region, meta, err = load_models()
if err:
    st.error(err); st.stop()

FEATURES       = meta['features']
AREA_CLASSES   = meta['area_classes']
REGION_CLASSES = meta['region_classes']

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔮 Predict Infant Mortality")
st.sidebar.markdown("Adjust the community health indicators below:")

model_choice  = st.sidebar.radio("Select Model", ["Linear Regression", "Decision Tree Regressor"], horizontal=False)
skilled_birth = st.sidebar.slider("Skilled Birth Attendance (%)", 0.0, 100.0, 65.0, 0.5)
vaccination   = st.sidebar.slider("Clinic Vaccination Rate (%)",  0.0, 100.0, 70.0, 0.5)
clean_water   = st.sidebar.slider("Access to Clean Water (%)",    0.0, 100.0, 60.0, 0.5)
area          = st.sidebar.selectbox("Area Type",  AREA_CLASSES)
region        = st.sidebar.selectbox("Region",     REGION_CLASSES)

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(skilled, vacc, water, area_v, region_v, model_name):
    ae  = le_area.transform([area_v])[0]
    re  = le_region.transform([region_v])[0]
    X_i = pd.DataFrame([[skilled, vacc, water, ae, re]], columns=FEATURES)
    if model_name == "Linear Regression":
        pred = lr_model.predict(scaler.transform(X_i))[0]
    else:
        pred = dt_model.predict(X_i)[0]
    return max(0.0, round(float(pred), 2))

predicted = predict(skilled_birth, vaccination, clean_water, area, region, model_choice)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Performance", "🌳 Decision Tree", "📈 Feature Analysis"])

# ════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-hdr">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Indicator": ["Skilled Birth (%)", "Vaccination (%)", "Clean Water (%)", "Area", "Region", "Model"],
            "Value":     [f"{skilled_birth:.1f}%", f"{vaccination:.1f}%", f"{clean_water:.1f}%", area, region, model_choice]
        }), hide_index=True, use_container_width=True)

        # Indicator bar chart
        fig, ax = plt.subplots(figsize=(5.5, 3))
        cats   = ['Skilled Birth', 'Vaccination', 'Clean Water']
        vals   = [skilled_birth, vaccination, clean_water]
        colors = ['#2196F3','#4CAF50','#FF9800']
        bars   = ax.barh(cats, vals, color=colors, height=0.5, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(v+1, bar.get_y()+bar.get_height()/2, f'{v:.0f}%', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 115)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Health Indicator Profile', fontweight='bold')
        ax.set_xlabel('Percentage (%)')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown('<div class="section-hdr">Prediction Result</div>', unsafe_allow_html=True)

        risk_level = "HIGH" if predicted >= 30 else "LOW"
        box_class  = "high-result" if predicted >= 30 else "low-result"
        icon       = "🔴" if predicted >= 30 else "🟢"
        mean_val   = meta['y_mean']
        diff       = predicted - mean_val
        diff_str   = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"

        st.markdown(f"""
        <div class="result-box {box_class}">
            <b style="font-size:1.5rem">{icon} {predicted:.1f} deaths per 1,000 live births</b><br>
            <b>Risk Level:</b> {risk_level} (threshold: 30)<br>
            <b>vs. National Mean:</b> {mean_val:.1f}  ({diff_str})<br>
            <b>Model:</b> {model_choice}
        </div>
        """, unsafe_allow_html=True)

        # Gauge: where prediction sits in range
        y_min, y_max = meta['y_min'], meta['y_max']
        fig2, ax2 = plt.subplots(figsize=(5.5, 2.2))
        ax2.barh([''], [y_max - y_min], left=y_min, color='#E0E0E0', height=0.5)
        ax2.barh([''], [predicted - y_min], left=y_min,
                 color='#F44336' if predicted >= 30 else '#4CAF50', height=0.5)
        ax2.axvline(predicted, color='black', linewidth=2.5, label=f'Prediction: {predicted:.1f}')
        ax2.axvline(mean_val,  color='blue',  linewidth=1.5, linestyle='--', label=f'Mean: {mean_val:.1f}')
        ax2.axvline(30,        color='orange',linewidth=1.5, linestyle=':',  label='Threshold: 30')
        ax2.set_xlim(y_min - 2, y_max + 2)
        ax2.set_xlabel('Infant Mortality (per 1,000)')
        ax2.set_title('Prediction on National Range', fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

        # Compare both models
        st.markdown("**Both Model Predictions**")
        lr_val = predict(skilled_birth, vaccination, clean_water, area, region, "Linear Regression")
        dt_val = predict(skilled_birth, vaccination, clean_water, area, region, "Decision Tree Regressor")
        compare_df = pd.DataFrame({
            "Model": ["Linear Regression", "Decision Tree Regressor"],
            "Predicted Mortality": [f"{lr_val:.2f}", f"{dt_val:.2f}"],
            "Risk": ["HIGH 🔴" if lr_val >= 30 else "LOW 🟢",
                     "HIGH 🔴" if dt_val >= 30 else "LOW 🟢"]
        })
        st.dataframe(compare_df, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Test Set Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("LR — R²",   f"{meta['lr_r2']:.4f}")
    c2.metric("LR — RMSE", f"{meta['lr_rmse']:.4f}")
    c3.metric("LR — MAE",  f"{meta['lr_mae']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("DT — R²",   f"{meta['dt_r2']:.4f}", delta=f"{meta['dt_r2']-meta['lr_r2']:+.4f} vs LR")
    c5.metric("DT — RMSE", f"{meta['dt_rmse']:.4f}")
    c6.metric("DT — MAE",  f"{meta['dt_mae']:.4f}")

    @st.cache_data
    def get_test_data():
        ds = 'tanzania_infant_mortality_dataset.xlsx'
        if not os.path.exists(ds): return None
        df = pd.read_excel(ds)
        df.columns = ['ID','Region','Area','Skilled_Birth_Pct','Clinic_Vaccination_Pct','Clean_Water_Pct','Infant_Mortality']
        df['Area_Encoded']   = le_area.transform(df['Area'])
        df['Region_Encoded'] = le_region.transform(df['Region'])
        X = df[FEATURES]; y = df['Infant_Mortality']
        _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        lp = lr_model.predict(scaler.transform(Xt))
        dp = dt_model.predict(Xt)
        return yt.values, lp, dp

    res = get_test_data()
    if res:
        y_test, lr_pred, dt_pred = res

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('Regression Evaluation — Test Set', fontsize=15, fontweight='bold')

        lims = [min(y_test.min(), lr_pred.min(), dt_pred.min())-1,
                max(y_test.max(), lr_pred.max(), dt_pred.max())+1]

        # Actual vs Predicted - LR
        axes[0,0].scatter(y_test, lr_pred, alpha=0.5, color='#2196F3', s=25)
        axes[0,0].plot(lims, lims, 'r--', lw=2, label='Perfect fit')
        axes[0,0].set_xlabel('Actual'); axes[0,0].set_ylabel('Predicted')
        axes[0,0].set_title(f'Linear Regression — Actual vs Predicted\nR²={meta["lr_r2"]:.4f}')
        axes[0,0].legend()

        # Actual vs Predicted - DT
        axes[0,1].scatter(y_test, dt_pred, alpha=0.5, color='#FF9800', s=25)
        axes[0,1].plot(lims, lims, 'r--', lw=2, label='Perfect fit')
        axes[0,1].set_xlabel('Actual'); axes[0,1].set_ylabel('Predicted')
        axes[0,1].set_title(f'Decision Tree — Actual vs Predicted\nR²={meta["dt_r2"]:.4f}')
        axes[0,1].legend()

        # Residuals - LR
        lr_res = y_test - lr_pred
        axes[1,0].scatter(lr_pred, lr_res, alpha=0.5, color='#2196F3', s=25)
        axes[1,0].axhline(0, color='red', lw=2, linestyle='--')
        axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('LR Residual Plot')

        # Residuals - DT
        dt_res = y_test - dt_pred
        axes[1,1].scatter(dt_pred, dt_res, alpha=0.5, color='#FF9800', s=25)
        axes[1,1].axhline(0, color='red', lw=2, linestyle='--')
        axes[1,1].set_xlabel('Predicted'); axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('DT Residual Plot')

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Residual distribution
        st.markdown('<div class="section-hdr">Residual Distributions</div>', unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
        axes2[0].hist(lr_res, bins=30, color='#2196F3', alpha=0.8, edgecolor='white')
        axes2[0].axvline(0, color='red', lw=2, linestyle='--')
        axes2[0].set_title('LR Residual Distribution'); axes2[0].set_xlabel('Residual')
        axes2[1].hist(dt_res, bins=30, color='#FF9800', alpha=0.8, edgecolor='white')
        axes2[1].axvline(0, color='red', lw=2, linestyle='--')
        axes2[1].set_title('DT Residual Distribution'); axes2[1].set_xlabel('Residual')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

# ════════════════════════════════════════════
# TAB 3 — Decision Tree
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Decision Tree Structure (max_depth=5)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(dt_model, feature_names=FEATURES, filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Decision Tree Regressor — Tanzania Infant Mortality", fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Feature Importances</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': dt_model.feature_importances_}).sort_values('Importance', ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(imp_df['Feature'], imp_df['Importance'], color='#4CAF50', edgecolor='white')
    ax2.set_title('Feature Importances — Decision Tree', fontweight='bold')
    ax2.set_ylabel('Importance'); ax2.set_xlabel('Feature')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    st.pyplot(fig2); plt.close()
    st.dataframe(imp_df.round(4), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — Feature Analysis
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">Linear Regression Coefficients</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        'Feature':     FEATURES,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_c = ['#F44336' if c > 0 else '#2196F3' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1.2)
    red_p  = mpatches.Patch(color='#F44336', label='Increases mortality')
    blue_p = mpatches.Patch(color='#2196F3', label='Decreases mortality')
    ax.legend(handles=[red_p, blue_p])
    ax.set_title('LR Coefficients — Effect on Infant Mortality Rate', fontweight='bold')
    ax.set_xlabel('Coefficient (on standardised features)')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.dataframe(coef_df.round(4), hide_index=True, use_container_width=True)

    st.markdown(f"""
    **Intercept:** `{lr_model.intercept_:.4f}`

    **How to read this:**
    - A **negative** coefficient means increasing that feature reduces predicted infant mortality.
    - A **positive** coefficient means increasing that feature raises predicted infant mortality.
    - Coefficients are based on **standardised** (scaled) features, so magnitudes are directly comparable.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>🇹🇿 Tanzania Infant Mortality Regression | "
    "Linear Regression & Decision Tree | Built with scikit-learn & Streamlit</p>",
    unsafe_allow_html=True
)
