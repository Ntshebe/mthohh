import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
full_data = pd.read_csv("data/full_data.csv", parse_dates=['visit_date'])
forecast = pd.read_csv("data/forecast.csv", parse_dates=['ds'])

# --- Page Setup ---
st.set_page_config(page_title="Hospital Dashboard", layout="wide")
st.title("📊 Hospital Data Insights")

# --- Tabs for Each Graph ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Department Distribution", "Symptom vs Outcome", "Diagnosis Frequency",
    "Age vs Outcome", "Monthly Visits", "Prophet Forecast"
])

# --- 1. Patient Count by Department ---
with tab1:
    dept_counts = full_data['department'].value_counts().reset_index()
    fig1 = px.bar(dept_counts, x='index', y='department', labels={'index':'Department', 'department':'Count'})
    st.plotly_chart(fig1, use_container_width=True)

# --- 2. Symptom Score vs Outcome ---
with tab2:
    fig2 = px.box(full_data, x='outcome', y='symptom_score', color='outcome')
    st.plotly_chart(fig2, use_container_width=True)

# --- 3. Diagnosis Frequency ---
with tab3:
    diag_counts = full_data['diagnosis'].value_counts().reset_index()
    fig3 = px.bar(diag_counts, x='index', y='diagnosis', labels={'index':'Diagnosis', 'diagnosis':'Count'})
    st.plotly_chart(fig3, use_container_width=True)

# --- 4. Age Distribution by Outcome ---
with tab4:
    fig4 = px.histogram(full_data, x='age', color='outcome', nbins=15, barmode='stack')
    st.plotly_chart(fig4, use_container_width=True)

# --- 5. Monthly Visit Trends ---
with tab5:
    full_data['month'] = full_data['visit_date'].dt.to_period('M').astype(str)
    monthly = full_data['month'].value_counts().sort_index().reset_index()
    fig5 = px.bar(monthly, x='index', y='month', labels={'index':'Month', 'month':'Visits'})
    st.plotly_chart(fig5, use_container_width=True)

# --- 6. Prophet Forecast ---
with tab6:
    fig6 = px.line(forecast, x='ds', y='yhat', title='Forecasted Patient Visits')
    fig6.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
    fig6.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', fill='tonexty', fillcolor='rgba(0,255,0,0.2)')
    st.plotly_chart(fig6, use_container_width=True)
    st.download_button("Download Forecast CSV", data=forecast.to_csv(index=False), file_name="forecast.csv")