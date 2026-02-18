"""
Tanzania Infant Mortality Prediction — Streamlit App
DATA IS EMBEDDED — no external files needed.
Deploy: just push app.py + requirements.txt to GitHub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
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
.result-high {
    background: #E65100 !important;
    border-left: 6px solid #FF6D00;
    border-radius: 10px;
    padding: 20px 24px;
    line-height: 2;
    color: #FFFFFF !important;
}
.result-high * { color: #FFFFFF !important; }
.result-low {
    background: #1B5E20 !important;
    border-left: 6px solid #00C853;
    border-radius: 10px;
    padding: 20px 24px;
    line-height: 2;
    color: #FFFFFF !important;
}
.result-low * { color: #FFFFFF !important; }
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

# ── Embedded dataset ──────────────────────────────────────────────────────────
_EMBEDDED_DATA = '{"Region": ["Njombe", "Lindi", "Kigoma", "Dar es Salaam", "Simiyu", "Geita", "Iringa", "Pwani", "Iringa", "Mara", "Arusha", "Morogoro", "Mtwara", "Singida", "Mbeya", "Rukwa", "Mwanza", "Shinyanga", "Songwe", "Mtwara", "Iringa", "Mbeya", "Katavi", "Songwe", "Dar es Salaam", "Njombe", "Dar es Salaam", "Mtwara", "Rukwa", "Lindi", "Tabora", "Simiyu", "Njombe", "Lindi", "Iringa", "Dar es Salaam", "Mwanza", "Kigoma", "Arusha", "Dar es Salaam", "Simiyu", "Pwani", "Katavi", "Morogoro", "Lindi", "Mbeya", "Mbeya", "Morogoro", "Arusha", "Simiyu", "Katavi", "Mbeya", "Simiyu", "Mbeya", "Shinyanga", "Shinyanga", "Singida", "Lindi", "Simiyu", "Mara", "Kigoma", "Mara", "Dar es Salaam", "Singida", "Iringa", "Iringa", "Dodoma", "Dar es Salaam", "Lindi", "Kagera", "Mwanza", "Kagera", "Tanga", "Mwanza", "Njombe", "Njombe", "Arusha", "Mtwara", "Ruvuma", "Dar es Salaam", "Njombe", "Ruvuma", "Mbeya", "Lindi", "Pwani", "Rukwa", "Katavi", "Rukwa", "Mbeya", "Simiyu", "Mara", "Mwanza", "Tabora", "Singida", "Tanga", "Morogoro", "Mwanza", "Morogoro", "Mtwara", "Mwanza", "Mtwara", "Rukwa", "Singida", "Mwanza", "Geita", "Morogoro", "Dar es Salaam", "Songwe", "Rukwa", "Katavi", "Shinyanga", "Singida", "Lindi", "Mbeya", "Dodoma", "Shinyanga", "Songwe", "Lindi", "Shinyanga", "Lindi", "Geita", "Songwe", "Mbeya", "Mwanza", "Morogoro", "Simiyu", "Mwanza", "Iringa", "Tabora", "Tabora", "Pwani", "Tabora", "Singida", "Rukwa", "Katavi", "Katavi", "Pwani", "Tabora", "Njombe", "Pwani", "Kagera", "Dodoma", "Kagera", "Dodoma", "Songwe", "Arusha", "Arusha", "Mtwara", "Singida", "Singida", "Simiyu", "Mbeya", "Mtwara", "Shinyanga", "Njombe", "Simiyu", "Morogoro", "Dar es Salaam", "Mbeya", "Kigoma", "Ruvuma", "Mara", "Mwanza", "Lindi", "Mara", "Iringa", "Kigoma", "Rukwa", "Njombe", "Rukwa", "Songwe", "Tabora", "Pwani", "Lindi", "Dodoma", "Dar es Salaam", "Kagera", "Lindi", "Arusha", "Geita", "Rukwa", "Rukwa", "Songwe", "Njombe", "Mara", "Katavi", "Mara", "Songwe", "Mwanza", "Iringa", "Mara", "Songwe", "Singida", "Tanga", "Arusha", "Rukwa", "Katavi", "Mara", "Tabora", "Rukwa", "Iringa", "Mtwara", "Morogoro", "Arusha", "Singida", "Rukwa", "Katavi", "Mbeya", "Shinyanga", "Tanga", "Lindi", "Tanga", "Njombe", "Ruvuma", "Lindi", "Iringa", "Geita", "Dodoma", "Shinyanga", "Arusha", "Kigoma", "Iringa", "Mtwara", "Rukwa", "Dodoma", "Rukwa", "Njombe", "Mwanza", "Rukwa", "Kagera", "Kagera", "Lindi", "Ruvuma", "Singida", "Shinyanga", "Kigoma", "Mbeya", "Morogoro", "Geita", "Mwanza", "Songwe", "Iringa", "Morogoro", "Mbeya", "Dar es Salaam", "Simiyu", "Njombe", "Simiyu", "Morogoro", "Morogoro", "Katavi", "Pwani", "Dodoma", "Tabora", "Songwe", "Katavi", "Kigoma", "Katavi", "Arusha", "Singida", "Kagera", "Mbeya", "Dodoma", "Songwe", "Morogoro", "Mwanza", "Geita", "Tabora", "Dodoma", "Rukwa", "Morogoro", "Njombe", "Lindi", "Songwe", "Morogoro", "Singida", "Mwanza", "Rukwa", "Tabora", "Kagera", "Singida", "Iringa", "Ruvuma", "Mbeya", "Morogoro", "Njombe", "Dodoma", "Singida", "Geita", "Dar es Salaam", "Geita", "Lindi", "Ruvuma", "Rukwa", "Dar es Salaam", "Mara", "Morogoro", "Mbeya", "Tabora", "Dodoma", "Iringa", "Tabora", "Arusha", "Dar es Salaam", "Morogoro", "Mtwara", "Songwe", "Shinyanga", "Mbeya", "Katavi", "Shinyanga", "Ruvuma", "Mwanza", "Iringa", "Kagera", "Mwanza", "Geita", "Dar es Salaam", "Mbeya", "Kigoma", "Lindi", "Kigoma", "Tabora", "Rukwa", "Songwe", "Morogoro", "Katavi", "Arusha", "Lindi", "Mbeya", "Ruvuma", "Mtwara", "Mara", "Simiyu", "Songwe", "Tabora", "Kigoma", "Mwanza", "Ruvuma", "Arusha", "Arusha", "Dar es Salaam", "Kagera", "Njombe", "Kagera", "Tabora", "Kigoma", "Mbeya", "Mara", "Arusha", "Simiyu", "Mbeya", "Mbeya", "Singida", "Mara", "Dodoma", "Iringa", "Shinyanga", "Mtwara", "Kagera", "Kagera", "Tanga", "Kagera", "Mara", "Singida", "Morogoro", "Dar es Salaam", "Katavi", "Rukwa", "Dar es Salaam", "Lindi", "Dar es Salaam", "Rukwa", "Katavi", "Mwanza", "Mara", "Simiyu", "Kagera", "Kagera", "Singida", "Tanga", "Singida", "Morogoro", "Morogoro", "Arusha", "Tanga", "Singida", "Kagera", "Shinyanga", "Geita", "Rukwa", "Rukwa", "Dar es Salaam", "Geita", "Tanga", "Dodoma", "Morogoro", "Dar es Salaam", "Mbeya", "Katavi", "Dar es Salaam", "Dodoma", "Njombe", "Mwanza", "Shinyanga", "Mbeya", "Shinyanga", "Dar es Salaam", "Njombe", "Morogoro", "Iringa", "Rukwa", "Dodoma", "Iringa", "Songwe", "Geita", "Songwe", "Rukwa", "Kagera", "Mbeya", "Rukwa", "Dodoma", "Ruvuma", "Iringa", "Mwanza", "Singida", "Singida", "Dodoma", "Dodoma", "Tabora", "Mbeya", "Geita", "Tabora", "Shinyanga", "Kagera", "Morogoro", "Morogoro", "Kagera", "Katavi", "Rukwa", "Mara", "Tabora", "Ruvuma", "Iringa", "Kagera", "Mtwara", "Dar es Salaam", "Rukwa", "Kagera", "Tabora", "Ruvuma", "Ruvuma", "Dodoma", "Pwani", "Dodoma", "Mbeya", "Songwe", "Mbeya", "Kagera", "Shinyanga", "Lindi", "Kigoma", "Njombe", "Singida", "Mara", "Kagera", "Kigoma", "Njombe", "Tabora", "Katavi", "Dar es Salaam", "Mara", "Mbeya", "Simiyu", "Mara", "Dodoma", "Njombe", "Katavi", "Dar es Salaam", "Njombe", "Katavi", "Singida", "Kagera", "Rukwa", "Simiyu", "Rukwa", "Kigoma", "Simiyu", "Kigoma", "Shinyanga", "Geita", "Simiyu", "Kigoma", "Arusha", "Geita", "Tabora", "Morogoro", "Mwanza", "Tanga", "Mtwara"], "Area": ["Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Mjini", "Mjini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Mjini", "Vijijini", "Vijijini", "Vijijini", "Mjini", "Vijijini", "Vijijini"], "Skilled_Birth_Pct": [70.0, 52.0, 54.1, 50.4, 86.5, 90.5, 78.4, 55.9, 52.5, 69.3, 82.6, 72.2, 72.7, 88.1, 100.0, 75.0, 88.9, 88.4, 71.1, 80.7, 74.0, 75.1, 98.8, 57.5, 40.2, 62.7, 76.5, 68.5, 95.0, 97.3, 33.5, 80.0, 100.0, 50.0, 82.4, 62.9, 85.1, 89.9, 53.9, 76.9, 81.4, 63.7, 88.5, 50.8, 97.8, 56.6, 81.8, 41.3, 88.7, 89.2, 58.9, 69.5, 45.6, 60.5, 47.2, 81.4, 88.1, 88.5, 74.5, 80.8, 49.3, 82.3, 49.5, 65.9, 98.9, 69.9, 89.8, 23.8, 96.0, 91.8, 77.9, 87.2, 84.2, 65.2, 69.1, 54.2, 62.8, 91.8, 45.1, 80.2, 68.4, 64.9, 87.5, 77.3, 100.0, 88.7, 40.9, 54.2, 65.5, 52.6, 79.6, 29.6, 49.8, 43.9, 47.5, 48.4, 38.1, 69.3, 81.9, 88.0, 87.4, 47.6, 83.5, 71.8, 30.0, 82.8, 73.6, 50.2, 93.7, 77.3, 100.0, 86.3, 70.1, 38.4, 33.5, 81.7, 36.4, 87.1, 42.7, 89.2, 65.4, 72.5, 92.3, 89.1, 92.2, 34.7, 41.6, 69.0, 90.4, 55.2, 55.8, 85.8, 91.0, 79.6, 45.6, 85.2, 59.7, 63.7, 47.9, 65.8, 34.7, 87.4, 23.0, 82.7, 93.3, 96.1, 48.3, 62.1, 98.3, 93.5, 62.5, 98.9, 98.8, 58.4, 47.8, 84.9, 35.0, 40.1, 72.4, 33.8, 49.7, 70.2, 64.7, 50.0, 98.9, 86.4, 98.9, 66.3, 98.8, 23.6, 80.1, 71.9, 84.6, 87.6, 52.6, 98.4, 49.8, 72.7, 68.7, 99.1, 97.8, 94.7, 71.7, 34.0, 86.3, 72.0, 46.0, 63.3, 72.3, 86.6, 87.8, 57.0, 51.8, 62.5, 81.6, 86.4, 52.4, 67.5, 58.7, 54.5, 38.3, 94.0, 75.3, 62.2, 56.8, 80.9, 82.8, 78.6, 90.2, 88.4, 47.4, 70.8, 66.2, 88.4, 80.8, 85.3, 92.9, 46.9, 49.1, 50.4, 72.3, 64.4, 88.8, 68.1, 93.1, 100.0, 63.7, 25.4, 38.2, 56.9, 64.3, 56.5, 49.5, 52.2, 86.9, 52.2, 76.0, 82.3, 64.9, 87.8, 72.0, 50.6, 40.1, 65.7, 86.1, 46.6, 57.8, 91.3, 47.7, 51.1, 89.2, 39.4, 64.6, 100.0, 91.3, 73.4, 62.4, 47.0, 55.0, 84.2, 56.7, 60.3, 90.5, 65.6, 21.5, 56.5, 35.4, 53.9, 56.1, 93.1, 57.5, 61.9, 69.9, 76.4, 56.2, 59.4, 58.2, 49.9, 46.5, 23.9, 41.9, 86.1, 76.8, 84.8, 56.1, 93.4, 58.4, 45.3, 94.0, 61.7, 82.4, 87.7, 72.4, 57.2, 74.4, 52.4, 42.8, 84.3, 70.5, 78.2, 88.7, 64.3, 61.4, 44.4, 54.4, 46.3, 25.5, 85.2, 50.3, 83.5, 61.2, 80.6, 81.8, 58.0, 69.2, 52.6, 62.5, 100.0, 78.3, 89.2, 38.6, 61.5, 55.8, 88.2, 45.6, 71.6, 83.9, 62.5, 62.8, 92.8, 40.1, 57.4, 91.6, 80.0, 95.1, 74.3, 45.1, 61.7, 73.0, 78.6, 40.1, 54.5, 97.7, 61.7, 66.1, 63.1, 45.2, 88.4, 63.1, 70.3, 74.5, 45.1, 64.5, 88.7, 76.9, 80.3, 95.4, 95.0, 61.5, 61.5, 70.4, 70.0, 74.9, 69.8, 56.4, 76.3, 41.9, 58.1, 92.1, 98.5, 100.0, 86.4, 100.0, 44.9, 87.2, 100.0, 72.7, 84.3, 84.9, 96.6, 100.0, 53.2, 54.3, 59.7, 95.0, 52.9, 83.5, 87.3, 74.4, 36.7, 64.8, 45.7, 66.3, 50.6, 84.7, 63.5, 63.4, 49.3, 48.4, 38.3, 99.1, 71.2, 44.8, 49.9, 95.6, 92.1, 66.2, 88.6, 62.6, 75.7, 98.5, 83.1, 83.2, 69.0, 49.2, 38.5, 84.1, 93.0, 68.4, 56.5, 71.6, 79.4, 89.8, 60.2, 82.4, 52.8, 68.9, 55.4, 53.2, 74.8, 100.0, 47.6, 57.0, 82.3, 81.0, 54.4, 100.0, 50.7, 73.8, 83.6, 91.5, 41.6, 55.5, 59.7, 69.8, 44.6, 42.1, 60.1, 54.1, 88.9, 65.3, 66.3, 56.4, 93.3, 92.0, 67.2, 93.7, 41.9, 96.8, 86.7, 100.0, 49.1, 63.6, 58.5, 50.6, 37.1, 83.0, 61.1, 100.0, 72.1, 84.2, 51.5, 75.6, 54.7, 41.1, 67.8, 79.5, 54.1, 72.2, 47.7, 53.5, 45.8, 26.0, 57.5, 59.9, 74.0, 87.6, 89.6, 43.4, 64.9, 81.7, 63.7, 92.9, 89.1, 54.0, 47.5, 46.6, 95.5, 50.2, 76.1], "Clinic_Vaccination_Pct": [82.2, 71.3, 59.3, 69.5, 79.7, 79.9, 82.4, 55.1, 70.1, 65.6, 95.5, 61.7, 60.2, 95.6, 72.7, 87.1, 69.3, 59.3, 63.9, 86.8, 38.5, 81.8, 92.8, 34.0, 61.0, 82.2, 88.4, 74.4, 20.0, 84.6, 48.3, 96.7, 76.4, 95.0, 71.8, 60.0, 79.8, 72.4, 76.8, 60.6, 87.9, 51.7, 71.7, 75.7, 92.4, 44.1, 85.7, 59.8, 60.3, 82.0, 69.3, 71.5, 43.0, 48.5, 88.7, 58.0, 73.6, 100.0, 100.0, 90.1, 78.1, 88.8, 53.2, 70.4, 75.8, 63.2, 93.7, 47.3, 79.0, 58.1, 61.9, 79.8, 83.0, 95.0, 48.8, 66.4, 31.3, 73.6, 84.4, 79.2, 82.3, 57.1, 98.1, 81.2, 81.2, 84.8, 55.4, 35.7, 70.9, 92.6, 95.0, 48.3, 58.7, 65.1, 77.8, 59.3, 56.3, 81.9, 83.8, 82.8, 85.9, 79.7, 93.3, 47.7, 75.4, 100.0, 63.7, 56.3, 90.6, 100.0, 80.0, 93.0, 50.2, 83.8, 63.1, 85.6, 51.1, 92.9, 68.6, 58.8, 79.2, 22.3, 73.8, 83.9, 75.1, 52.1, 46.4, 73.6, 99.0, 59.1, 81.9, 80.0, 76.0, 75.7, 72.6, 84.8, 46.6, 45.4, 55.5, 47.2, 74.7, 88.0, 40.9, 69.7, 100.0, 79.9, 64.4, 30.5, 100.0, 78.0, 21.7, 86.2, 80.2, 70.8, 65.1, 70.6, 74.8, 79.3, 79.2, 58.7, 57.1, 66.0, 58.3, 60.8, 84.4, 42.0, 99.8, 77.9, 88.4, 23.4, 58.7, 73.3, 20.0, 75.2, 54.8, 90.4, 70.8, 84.5, 63.0, 95.9, 86.9, 76.8, 20.0, 86.3, 68.3, 26.9, 69.6, 56.5, 72.4, 98.3, 87.3, 80.3, 62.1, 86.4, 89.5, 74.3, 62.5, 63.8, 50.1, 53.1, 39.9, 100.0, 83.1, 59.5, 80.2, 70.1, 95.5, 72.8, 77.6, 83.3, 84.3, 41.1, 65.0, 89.6, 74.6, 63.4, 100.0, 55.5, 70.5, 86.0, 61.5, 57.2, 89.5, 62.7, 83.5, 89.3, 37.8, 68.7, 47.7, 62.7, 69.0, 95.0, 75.4, 86.6, 85.3, 70.8, 67.3, 70.5, 59.9, 80.6, 54.8, 95.0, 64.1, 46.2, 80.3, 79.2, 80.1, 25.9, 95.0, 51.9, 74.2, 95.0, 95.0, 66.9, 79.8, 87.7, 64.6, 53.0, 73.2, 93.5, 74.6, 63.2, 73.9, 45.9, 87.6, 74.6, 73.6, 41.9, 92.6, 90.6, 75.1, 56.2, 66.7, 67.3, 20.0, 72.5, 95.0, 71.7, 44.8, 73.7, 81.1, 100.0, 60.7, 64.0, 57.6, 85.8, 36.7, 55.3, 85.2, 91.4, 87.2, 81.5, 57.1, 78.1, 100.0, 61.0, 53.5, 100.0, 31.8, 86.9, 98.6, 88.5, 72.8, 51.5, 77.4, 55.0, 80.0, 88.8, 81.0, 84.9, 95.0, 94.6, 77.2, 53.4, 62.0, 20.0, 70.4, 95.8, 99.3, 98.1, 47.6, 20.0, 79.6, 100.0, 61.5, 59.2, 88.8, 63.5, 76.6, 89.9, 86.5, 57.7, 82.8, 76.3, 75.4, 94.8, 53.1, 72.2, 72.1, 100.0, 37.0, 88.1, 80.3, 48.2, 73.5, 56.8, 67.5, 97.4, 67.1, 65.9, 52.8, 24.7, 58.2, 78.0, 85.6, 28.9, 86.7, 50.5, 58.0, 20.0, 90.7, 53.6, 63.8, 82.2, 49.8, 79.3, 24.9, 71.9, 59.8, 100.0, 77.3, 74.7, 90.2, 55.9, 78.0, 100.0, 47.3, 93.1, 83.6, 69.6, 92.1, 62.7, 89.0, 52.9, 68.9, 76.2, 89.8, 83.3, 50.9, 42.9, 95.0, 86.9, 61.2, 95.0, 83.9, 75.0, 62.2, 69.2, 67.3, 54.4, 85.8, 74.9, 80.5, 67.8, 90.5, 92.8, 57.3, 76.7, 73.4, 77.0, 100.0, 81.3, 58.0, 91.3, 57.2, 68.1, 87.9, 76.7, 80.8, 95.0, 55.5, 83.9, 76.7, 61.8, 88.9, 42.7, 37.9, 60.4, 73.9, 83.4, 88.7, 62.6, 76.2, 100.0, 76.9, 60.3, 86.1, 88.8, 46.1, 90.9, 75.7, 59.7, 72.5, 58.3, 82.0, 54.7, 55.0, 80.9, 48.3, 75.5, 32.1, 94.8, 83.7, 95.0, 80.6, 41.7, 87.3, 45.5, 83.1, 78.1, 73.7, 33.0, 53.2, 49.6, 39.3, 95.0, 91.3, 69.4, 100.0, 66.6, 90.9, 31.4, 88.6, 89.1, 39.4, 68.4, 69.5, 67.1, 69.2, 73.8, 68.8, 91.0, 43.3, 66.3, 29.3, 24.7, 93.5, 100.0, 91.4, 95.0, 85.8, 75.1, 78.3, 91.3, 59.4, 85.2, 78.2, 75.6, 73.1, 71.5], "Clean_Water_Pct": [65.0, 91.9, 42.0, 42.3, 85.5, 93.2, 88.7, 61.8, 34.1, 20.3, 96.9, 53.2, 61.3, 99.9, 90.5, 80.0, 86.9, 30.3, 40.3, 91.9, 61.1, 94.1, 100.0, 62.2, 57.9, 32.4, 87.9, 92.0, 59.9, 97.8, 79.8, 82.2, 77.7, 57.3, 21.3, 15.0, 99.4, 95.6, 92.0, 55.2, 100.0, 47.3, 87.8, 47.2, 99.7, 38.1, 86.4, 58.2, 37.7, 69.5, 66.1, 32.8, 68.2, 67.8, 60.7, 92.0, 86.4, 91.1, 91.2, 90.4, 78.2, 84.8, 30.0, 70.3, 80.6, 67.2, 90.7, 41.1, 94.5, 30.3, 35.0, 87.2, 76.0, 48.9, 63.7, 62.8, 85.6, 90.9, 78.9, 96.8, 92.0, 52.5, 91.2, 87.9, 91.4, 84.1, 69.3, 15.0, 62.5, 61.2, 80.3, 52.4, 23.2, 72.5, 62.5, 77.0, 24.2, 65.1, 91.7, 91.4, 88.2, 65.9, 94.8, 34.0, 56.7, 99.9, 53.3, 15.0, 75.9, 81.2, 95.1, 92.2, 67.7, 48.5, 32.0, 58.6, 57.5, 81.1, 46.1, 78.9, 44.2, 28.1, 93.8, 81.5, 86.9, 61.6, 53.2, 74.5, 91.4, 15.1, 52.6, 88.1, 93.7, 95.9, 35.3, 92.6, 71.9, 79.1, 58.8, 35.1, 70.6, 97.3, 57.4, 91.3, 83.0, 96.9, 53.4, 16.0, 94.4, 75.7, 60.6, 82.7, 90.1, 58.9, 33.9, 87.2, 68.5, 79.0, 84.4, 54.1, 53.5, 51.7, 55.8, 70.1, 96.5, 24.5, 100.0, 90.5, 97.7, 57.7, 75.2, 89.4, 64.2, 91.2, 89.5, 87.2, 44.7, 72.4, 56.9, 98.2, 95.9, 88.6, 35.4, 26.0, 75.7, 60.0, 70.4, 19.8, 76.3, 83.9, 77.0, 53.7, 35.4, 55.0, 100.0, 94.4, 85.3, 84.6, 15.0, 36.3, 15.0, 85.7, 92.7, 63.2, 44.6, 83.5, 91.0, 20.8, 95.2, 97.0, 51.0, 56.5, 85.0, 90.8, 39.6, 94.3, 88.5, 44.7, 60.5, 17.8, 72.6, 88.9, 94.9, 49.2, 100.0, 88.8, 48.5, 73.7, 64.8, 92.0, 48.4, 15.5, 26.3, 55.2, 100.0, 41.9, 91.6, 93.4, 35.7, 93.2, 31.6, 78.6, 85.8, 23.4, 97.9, 83.1, 63.0, 45.2, 24.1, 78.1, 88.3, 25.5, 78.8, 85.4, 81.4, 49.7, 87.4, 49.8, 49.7, 97.5, 42.2, 38.7, 99.7, 55.1, 15.0, 24.8, 52.8, 57.4, 50.3, 86.6, 46.8, 54.6, 63.7, 63.5, 41.2, 73.3, 58.1, 79.6, 83.7, 26.1, 37.8, 94.9, 21.7, 87.0, 48.1, 83.1, 52.7, 27.2, 95.9, 92.0, 80.8, 96.0, 15.0, 82.0, 88.6, 20.3, 15.0, 94.0, 67.9, 93.1, 98.9, 65.0, 58.2, 47.7, 74.4, 30.9, 78.0, 90.5, 90.6, 92.0, 48.0, 92.9, 82.8, 64.8, 29.3, 54.3, 57.8, 85.5, 98.1, 100.0, 48.5, 39.9, 48.5, 88.2, 84.3, 80.4, 45.8, 73.2, 57.2, 85.0, 53.5, 90.2, 80.0, 93.1, 91.3, 91.9, 65.6, 44.6, 73.8, 93.3, 62.8, 15.0, 92.7, 92.0, 75.7, 85.7, 41.7, 95.3, 80.6, 67.7, 65.2, 87.6, 34.2, 92.1, 88.2, 33.1, 91.0, 36.7, 69.7, 88.2, 51.4, 65.4, 69.9, 30.6, 42.7, 34.9, 44.3, 69.3, 85.0, 96.3, 86.3, 91.6, 100.0, 85.5, 88.8, 92.9, 51.6, 81.3, 66.8, 97.6, 85.0, 47.6, 43.9, 35.5, 71.7, 51.5, 81.4, 94.7, 41.4, 48.8, 59.1, 34.3, 37.2, 82.6, 89.6, 44.0, 28.1, 72.0, 33.9, 54.7, 85.1, 71.4, 78.5, 64.7, 90.7, 71.8, 38.5, 92.0, 47.3, 82.2, 74.2, 80.5, 92.0, 27.6, 41.2, 57.9, 81.8, 75.2, 73.3, 70.0, 67.2, 88.2, 90.0, 84.6, 79.0, 70.7, 34.4, 36.8, 68.2, 91.4, 91.1, 80.2, 58.6, 95.8, 82.5, 68.5, 84.9, 36.6, 47.2, 92.1, 82.5, 27.7, 69.0, 50.1, 55.3, 43.4, 64.5, 59.0, 79.3, 85.6, 50.9, 40.2, 72.3, 49.3, 99.7, 51.6, 96.6, 51.9, 97.6, 86.0, 85.9, 82.7, 47.5, 46.0, 75.7, 26.6, 83.7, 55.9, 88.9, 63.9, 93.0, 76.0, 82.3, 63.2, 79.8, 82.6, 89.2, 64.3, 47.8, 92.0, 70.6, 56.7, 61.3, 63.1, 63.7, 50.2, 85.1, 82.5, 32.2, 69.4, 39.0, 33.8, 82.9, 83.2, 60.0, 43.6, 47.1, 93.6, 67.6, 66.8], "Infant_Mortality": [31.6, 36.8, 49.3, 43.7, 15.7, 11.9, 14.3, 42.4, 50.3, 49.7, 5.0, 46.5, 27.9, 5.0, 11.0, 11.7, 9.0, 40.1, 49.8, 6.2, 37.7, 8.6, 5.0, 53.6, 47.5, 39.8, 14.2, 26.7, 41.8, 5.0, 56.3, 9.1, 5.0, 38.4, 31.4, 50.4, 5.4, 5.0, 30.5, 35.4, 5.0, 33.1, 14.5, 43.6, 5.0, 62.4, 14.8, 48.1, 30.7, 11.9, 39.9, 43.4, 46.2, 48.6, 37.4, 30.9, 5.1, 14.6, 10.4, 21.5, 42.3, 5.1, 58.1, 39.5, 7.2, 38.9, 6.3, 58.5, 5.0, 31.7, 40.0, 6.6, 5.5, 37.3, 41.8, 43.0, 41.7, 5.0, 33.9, 13.4, 23.2, 43.3, 10.6, 9.5, 6.2, 9.3, 50.5, 61.5, 36.1, 36.7, 5.6, 58.8, 50.8, 49.7, 33.6, 45.8, 60.5, 33.0, 13.1, 5.0, 7.2, 32.3, 5.0, 49.8, 44.1, 5.0, 33.6, 55.3, 5.0, 5.0, 5.0, 5.0, 41.9, 51.0, 63.5, 27.9, 60.5, 5.0, 54.1, 29.1, 39.4, 55.0, 5.0, 5.0, 15.5, 53.0, 53.2, 26.6, 5.0, 52.7, 38.0, 8.1, 6.3, 22.5, 47.8, 14.1, 51.3, 44.1, 47.0, 41.0, 48.7, 8.7, 70.9, 6.5, 5.0, 5.0, 43.7, 60.7, 5.0, 5.0, 54.5, 5.0, 5.0, 36.9, 57.3, 11.0, 58.5, 42.1, 5.0, 42.1, 51.3, 44.6, 37.4, 54.3, 8.4, 44.1, 5.0, 20.4, 8.3, 69.6, 32.2, 18.4, 44.0, 12.5, 32.5, 5.0, 48.8, 27.9, 42.3, 6.4, 5.0, 8.7, 60.3, 53.4, 8.9, 48.6, 32.1, 38.1, 32.4, 5.0, 5.0, 37.5, 46.0, 34.7, 10.6, 9.2, 35.2, 39.9, 46.8, 51.8, 57.0, 5.0, 9.0, 46.0, 44.9, 10.5, 5.0, 37.6, 14.9, 12.5, 38.0, 47.1, 24.6, 5.0, 29.5, 16.5, 5.0, 53.6, 43.6, 46.9, 42.6, 44.0, 5.0, 38.5, 5.0, 5.0, 58.1, 50.3, 56.5, 29.2, 38.2, 45.9, 49.6, 43.6, 5.0, 37.3, 20.0, 6.1, 47.5, 8.4, 35.8, 38.5, 41.0, 50.8, 5.0, 46.1, 43.9, 31.9, 56.7, 40.6, 18.6, 47.1, 25.9, 5.0, 5.3, 44.4, 34.1, 49.5, 43.1, 5.0, 51.2, 38.0, 5.0, 48.9, 59.7, 49.5, 57.6, 54.5, 38.0, 13.2, 38.7, 42.8, 35.5, 32.4, 51.9, 28.5, 36.4, 36.0, 56.0, 59.4, 47.2, 7.0, 44.8, 11.8, 43.3, 22.9, 51.7, 61.1, 5.0, 24.2, 19.6, 12.3, 49.0, 32.3, 6.4, 49.3, 53.7, 7.1, 46.0, 12.1, 5.0, 33.4, 36.4, 49.2, 45.3, 55.2, 49.7, 9.8, 35.2, 8.2, 29.8, 5.0, 6.7, 46.5, 44.3, 51.6, 41.7, 5.0, 5.0, 5.0, 58.7, 51.1, 43.9, 5.0, 37.5, 34.6, 29.9, 34.4, 34.5, 5.0, 42.4, 37.5, 5.0, 5.0, 5.0, 6.2, 52.6, 44.3, 35.2, 5.0, 55.8, 34.1, 5.0, 40.9, 31.3, 37.9, 42.1, 5.0, 38.7, 37.4, 34.9, 55.3, 46.5, 5.0, 9.3, 44.7, 5.0, 41.3, 37.1, 41.6, 24.2, 25.4, 36.6, 42.4, 55.4, 34.9, 59.5, 35.4, 9.1, 5.0, 5.0, 10.4, 5.0, 50.6, 5.0, 5.0, 43.7, 15.5, 23.7, 5.0, 5.0, 48.1, 46.7, 42.3, 29.5, 34.7, 6.4, 7.4, 43.2, 63.2, 22.4, 46.8, 49.7, 32.8, 5.0, 45.1, 43.2, 37.5, 45.3, 52.1, 5.0, 25.2, 33.9, 39.7, 5.0, 18.8, 46.3, 8.1, 41.3, 22.4, 5.0, 27.2, 24.8, 41.8, 48.4, 51.1, 5.0, 9.4, 25.9, 37.3, 41.4, 20.3, 17.1, 35.1, 5.0, 43.2, 43.8, 50.3, 33.4, 8.2, 5.0, 43.0, 39.4, 5.0, 10.0, 34.4, 5.0, 45.5, 41.8, 7.4, 5.0, 60.9, 26.1, 45.2, 40.3, 60.3, 48.4, 30.3, 50.3, 8.2, 47.9, 30.3, 28.8, 24.9, 6.4, 46.4, 5.0, 49.2, 5.0, 5.8, 5.0, 40.0, 48.0, 41.4, 52.7, 51.1, 8.7, 48.3, 5.0, 39.1, 5.0, 62.2, 9.2, 33.5, 52.9, 31.2, 9.2, 36.9, 44.6, 31.3, 41.4, 45.9, 65.1, 37.8, 47.9, 45.2, 5.0, 5.4, 46.6, 36.6, 27.0, 47.4, 12.5, 5.1, 33.7, 41.4, 54.3, 5.6, 44.6, 30.9]}'

# ── Load data & train ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models…")
def train():
    raw = json.loads(_EMBEDDED_DATA)
    df = pd.DataFrame(raw)

    le_area   = LabelEncoder().fit(df["Area"])
    le_region = LabelEncoder().fit(df["Region"])
    df["Area_Enc"]   = le_area.transform(df["Area"])
    df["Region_Enc"] = le_region.transform(df["Region"])

    FEATS = ["Skilled_Birth_Pct", "Clinic_Vaccination_Pct",
             "Clean_Water_Pct", "Area_Enc", "Region_Enc"]

    X = df[FEATS].values
    y = df["Infant_Mortality"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler().fit(Xtr)
    lr = LinearRegression().fit(sc.transform(Xtr), ytr)
    dt = DecisionTreeRegressor(
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

# ── Best model ────────────────────────────────────────────────────────────────
BEST_MODEL = "Linear Regression" if M["lr_r2"] >= M["dt_r2"] else "Decision Tree Regressor"
BEST_R2    = max(M["lr_r2"],   M["dt_r2"])
BEST_RMSE  = M["lr_rmse"] if BEST_MODEL == "Linear Regression" else M["dt_rmse"]
BEST_MAE   = M["lr_mae"]  if BEST_MODEL == "Linear Regression" else M["dt_mae"]

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

# ── Banner ────────────────────────────────────────────────────────────────────
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
            yaxis_title="", height=260, showlegend=False,
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
        best_badge = " &nbsp;<span style='background:rgba(255,255,255,0.25);color:#FFFFFF;border-radius:6px;padding:3px 10px;font-size:0.85rem;font-weight:700;border:1px solid rgba(255,255,255,0.5)'>🏆 Best Model</span>" if model_choice == BEST_MODEL else ""

        bg_color = "#B71C1C" if pred >= 30 else "#1B5E20"
        border_color = "#FF5252" if pred >= 30 else "#69F0AE"
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 7px solid {border_color};
            border-radius: 12px;
            padding: 22px 26px;
            line-height: 2.2;
            color: #FFFFFF;
            margin-bottom: 12px;
        ">
          <div style="font-size:2rem; font-weight:900; color:#FFFFFF; margin-bottom:6px;">
            {icon} {pred:.1f} <span style="font-size:1.1rem; font-weight:600;">per 1,000 live births</span>
          </div>
          <div style="font-size:1rem; color:#FFFFFF;">
            <span style="opacity:0.85;">Risk Level:</span>
            <b style="color:#FFFFFF; font-size:1.1rem;">&nbsp;{risk}</b>
            <span style="opacity:0.75; font-size:0.9rem;">&nbsp;(threshold ≥ 30)</span>
          </div>
          <div style="font-size:1rem; color:#FFFFFF;">
            <span style="opacity:0.85;">vs. National Mean ({mean_v:.1f}):</span>
            <b style="color:#FFFFFF;">&nbsp;{diff_str} per 1,000</b>
          </div>
          <div style="font-size:0.95rem; color:#FFFFFF; opacity:0.9;">
            Model: {model_choice}{best_badge}
          </div>
        </div>
        """, unsafe_allow_html=True)

        gcol = "#F44336" if pred >= 30 else "#4CAF50"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Mortality (per 1,000)"},
            gauge={
                "axis":  {"range": [M["y_min"], M["y_max"]]},
                "bar":   {"color": gcol, "thickness": 0.28},
                "steps": [
                    {"range": [M["y_min"], 30],   "color": "#C8E6C9"},
                    {"range": [30, M["y_max"]],   "color": "#FFCCBC"},
                ],
                "threshold": {
                    "line": {"color": "#1565C0", "width": 3},
                    "thickness": 0.8, "value": mean_v
                }
            }
        ))
        fig_g.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 2 – Model Performance
# ════════════════════════════════════════════════════
with t2:
    st.markdown('<p class="sec">Model Comparison — Test Set (80/20 split)</p>', unsafe_allow_html=True)

    lr_badge = "🏆 WINNER" if BEST_MODEL == "Linear Regression"      else ""
    dt_badge = "🏆 WINNER" if BEST_MODEL == "Decision Tree Regressor" else ""

    comp_df = pd.DataFrame({
        "Model":                    [f"Linear Regression {lr_badge}", f"Decision Tree Regressor {dt_badge}"],
        "R²  ↑ (higher=better)":   [M["lr_r2"],   M["dt_r2"]],
        "RMSE ↓ (lower=better)":   [M["lr_rmse"], M["dt_rmse"]],
        "MAE  ↓ (lower=better)":   [M["lr_mae"],  M["dt_mae"]],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    bar_df = pd.DataFrame({
        "Metric": ["R²","R²","RMSE","RMSE","MAE","MAE"],
        "Model":  ["Linear Regression","Decision Tree Regressor"] * 3,
        "Value":  [M["lr_r2"], M["dt_r2"],
                   M["lr_rmse"], M["dt_rmse"],
                   M["lr_mae"],  M["dt_mae"]],
    })
    fig_cmp = px.bar(
        bar_df, x="Metric", y="Value", color="Model", barmode="group",
        color_discrete_map={"Linear Regression":"#2196F3","Decision Tree Regressor":"#FF9800"},
        text="Value", title="Model Comparison — R², RMSE, MAE"
    )
    fig_cmp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_cmp.update_layout(height=420, yaxis_title="Score", margin=dict(t=50,b=0))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown('<p class="sec">Detailed Metrics</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("LR — R²",   f"{M['lr_r2']:.4f}", delta="✅ BEST" if BEST_MODEL=="Linear Regression" else None)
    c2.metric("LR — RMSE", f"{M['lr_rmse']:.4f}")
    c3.metric("LR — MAE",  f"{M['lr_mae']:.4f}")
    c4,c5,c6 = st.columns(3)
    c4.metric("DT — R²",   f"{M['dt_r2']:.4f}",
              delta="✅ BEST" if BEST_MODEL=="Decision Tree Regressor" else f"{M['dt_r2']-M['lr_r2']:+.4f} vs LR")
    c5.metric("DT — RMSE", f"{M['dt_rmse']:.4f}")
    c6.metric("DT — MAE",  f"{M['dt_mae']:.4f}")

    st.markdown('<p class="sec">Actual vs Predicted</p>', unsafe_allow_html=True)
    lims = [float(test_df["Actual"].min())-1, float(test_df["Actual"].max())+1]
    fig_avp = make_subplots(rows=1, cols=2,
        subplot_titles=["Linear Regression","Decision Tree Regressor"])
    for col_pred, color, ci in [("LR_Pred","#2196F3",1),("DT_Pred","#FF9800",2)]:
        fig_avp.add_trace(go.Scatter(x=test_df["Actual"].tolist(), y=test_df[col_pred].tolist(),
            mode="markers", marker=dict(color=color, size=6, opacity=0.55), name=col_pred), row=1, col=ci)
        fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
            line=dict(color="red", dash="dash", width=2), showlegend=False), row=1, col=ci)
    fig_avp.update_xaxes(title_text="Actual")
    fig_avp.update_yaxes(title_text="Predicted")
    fig_avp.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown('<p class="sec">Residual Plots</p>', unsafe_allow_html=True)
    fig_res = make_subplots(rows=1, cols=2, subplot_titles=["LR Residuals","DT Residuals"])
    fig_res.add_trace(go.Scatter(x=test_df["LR_Pred"].tolist(), y=test_df["LR_Resid"].tolist(),
        mode="markers", marker=dict(color="#2196F3", size=6, opacity=0.55), name="LR"), row=1, col=1)
    fig_res.add_trace(go.Scatter(x=test_df["DT_Pred"].tolist(), y=test_df["DT_Resid"].tolist(),
        mode="markers", marker=dict(color="#FF9800", size=6, opacity=0.55), name="DT"), row=1, col=2)
    for ci in [1,2]:
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=ci)
    fig_res.update_xaxes(title_text="Predicted")
    fig_res.update_yaxes(title_text="Residual")
    fig_res.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 3 – Feature Importance
# ════════════════════════════════════════════════════
with t3:
    st.markdown('<p class="sec">Decision Tree — Feature Importances</p>', unsafe_allow_html=True)
    feat_labels = ["Skilled Birth (%)","Vaccination (%)","Clean Water (%)","Area Type","Region"]
    imp_df = pd.DataFrame({"Feature": feat_labels, "Importance": dt_model.feature_importances_}
                         ).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Greens",
        text=imp_df["Importance"].round(4),
        title="Feature Importances — Decision Tree Regressor")
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0,r=60,t=50,b=0))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.dataframe(imp_df.sort_values("Importance", ascending=False).round(4),
                 hide_index=True, use_container_width=True)

    st.markdown('<p class="sec">EDA — Feature vs Infant Mortality</p>', unsafe_allow_html=True)
    num_feats    = ["Skilled_Birth_Pct","Clinic_Vaccination_Pct","Clean_Water_Pct"]
    feat_display = ["Skilled Birth (%)","Vaccination (%)","Clean Water (%)"]
    colors_eda   = ["#2196F3","#4CAF50","#FF9800"]
    fig_eda = make_subplots(rows=1, cols=3, subplot_titles=feat_display)
    for i,(feat,color) in enumerate(zip(num_feats,colors_eda),1):
        fig_eda.add_trace(go.Scatter(x=full_df[feat].tolist(), y=full_df["Infant_Mortality"].tolist(),
            mode="markers", marker=dict(color=color, size=5, opacity=0.45), name=feat), row=1, col=i)
        z  = np.polyfit(full_df[feat], full_df["Infant_Mortality"], 1)
        xl = np.linspace(full_df[feat].min(), full_df[feat].max(), 80)
        fig_eda.add_trace(go.Scatter(x=xl.tolist(), y=np.poly1d(z)(xl).tolist(),
            mode="lines", line=dict(color="red",width=2), showlegend=False), row=1, col=i)
    fig_eda.update_xaxes(title_text="Value (%)")
    fig_eda.update_yaxes(title_text="Infant Mortality", col=1)
    fig_eda.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_eda, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 4 – Feature Analysis
# ════════════════════════════════════════════════════
with t4:
    st.markdown('<p class="sec">Linear Regression — Coefficients</p>', unsafe_allow_html=True)
    feat_labels = ["Skilled Birth (%)","Vaccination (%)","Clean Water (%)","Area Type","Region"]
    coef_df = pd.DataFrame({"Feature": feat_labels, "Coefficient": lr_model.coef_}
                           ).sort_values("Coefficient")
    coef_df["Effect"] = coef_df["Coefficient"].apply(
        lambda x: "Increases mortality" if x > 0 else "Decreases mortality")
    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
        color="Effect",
        color_discrete_map={"Increases mortality":"#F44336","Decreases mortality":"#2196F3"},
        text=coef_df["Coefficient"].round(3),
        title="LR Coefficients (standardised) — Effect on Infant Mortality")
    fig_coef.update_traces(textposition="outside")
    fig_coef.add_vline(x=0, line_color="black", line_width=1.5)
    fig_coef.update_layout(height=400, margin=dict(l=0,r=70,t=50,b=0))
    st.plotly_chart(fig_coef, use_container_width=True)
    st.dataframe(coef_df[["Feature","Coefficient","Effect"]].sort_values(
        "Coefficient", key=abs, ascending=False).round(4),
        hide_index=True, use_container_width=True)
    st.markdown(f"""
**Intercept:** `{lr_model.intercept_:.4f}`

**How to read:**
- 🔴 **Positive** → raising this feature *increases* predicted infant mortality
- 🔵 **Negative** → raising this feature *decreases* predicted infant mortality
- Features are **standardised** so coefficient sizes are directly comparable
""")

    st.markdown('<p class="sec">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = full_df[["Skilled_Birth_Pct","Clinic_Vaccination_Pct",
                    "Clean_Water_Pct","Infant_Mortality"]].corr().round(3)
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn_r",
        title="Correlation Matrix", aspect="auto")
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
