# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:21:45 2025

@author: Sukhdeep.Sangha
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO, StringIO
from matplotlib.backends.backend_pdf import PdfPages
import requests

# --- Load Data ---
@st.cache_data
def load_data():
    st.write("🔄 Starting download...")
    url = "https://drive.google.com/uc?export=download&id=1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk"

    try:
        response = requests.get(url)
        response.raise_for_status()
        st.write("✅ Download complete")
    except Exception as e:
        st.error(f"❌ Failed to download file: {e}")
        return pd.DataFrame()

    try:
        st.write("📦 Reading Parquet file...")
        df = pd.read_parquet(BytesIO(response.content))
        st.write("✅ Parquet loaded")
    except Exception as e:
        st.error(f"❌ Failed to read Parquet: {e}")
        return pd.DataFrame()

    try:
        st.write("🗓️ Parsing datetime...")
        df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce', format='%Y-%m-%d %H:%M:%S.%f')
    except Exception as e:
        st.error(f"❌ Failed to parse datetime: {e}")
        return pd.DataFrame()

    st.write(f"✅ Data ready! {len(df):,} rows loaded")
    return df.dropna(subset=['EVENT_START_TIMESTAMP'])

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

exp_types = st.sidebar.multiselect("Select Expectancy Types (up to 6)", [
    'Favourite Goals', 'Underdog Goals', 'Total Goals',
    'Favourite Corners', 'Underdog Corners', 'Total Corners',
    'Favourite Yellow', 'Underdog Yellow', 'Total Yellow'
], default=['Favourite Goals'])

min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
df = df[(df['EVENT_START_TIMESTAMP'].dt.date >= start_date) & (df['EVENT_START_TIMESTAMP'].dt.date <= end_date)]

fav_options = ['Strong Favourite', 'Medium Favourite', 'Slight Favourite']
scoreline_options = ['Favourite Winning', 'Scores Level', 'Underdog Winning']

fav_filter = st.sidebar.multiselect("Goal Favouritism Level", fav_options, default=fav_options)
scoreline_filter = st.sidebar.multiselect("Goal Scoreline Filter", scoreline_options, default=scoreline_options)

st.markdown("*Favourites are determined using Goal Expectancy at the earliest available minute in each match")

# --- Helper Functions ---
def classify_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return 'Strong Favourite'
    elif diff > 0.5:
        return 'Medium Favourite'
    else:
        return 'Slight Favourite'

def classify_scoreline_simple(row):
    fav = 'Home' if row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY'] else 'Away'
    score_diff = row['GOALS_HOME'] - row['GOALS_AWAY']
    fav_diff = score_diff if fav == 'Home' else -score_diff
    if fav_diff > 0:
        return "Favourite Winning"
    elif fav_diff == 0:
        return "Scores Level"
    else:
        return "Underdog Winning"

def compute_exp_by_role(df, exp_type):
    incident_map = {
        'Goals': ('GOAL_EXP_HOME', 'GOAL_EXP_AWAY'),
        'Corners': ('CORNERS_EXP_HOME', 'CORNERS_EXP_AWAY'),
        'Yellow': ('YELLOW_CARDS_EXP_HOME', 'YELLOW_CARDS_EXP_AWAY')
    }

    if 'Goals' in exp_type:
        incident = 'Goals'
    elif 'Corners' in exp_type:
        incident = 'Corners'
    else:
        incident = 'Yellow'

    home_col, away_col = incident_map[incident]

    role = 'Favourite' if 'Favourite' in exp_type else 'Underdog' if 'Underdog' in exp_type else 'Total'
    change_data = []

    for event_id, group in df.groupby('SRC_EVENT_ID'):
        group = group.sort_values('MINUTES')
        if group.empty:
            continue
        earliest_minute = group['MINUTES'].min()
        base_row = group[group['MINUTES'] == earliest_minute].iloc[0]

        home_exp_0 = base_row['GOAL_EXP_HOME']
        away_exp_0 = base_row['GOAL_EXP_AWAY']
        if home_exp_0 == away_exp_0:
            continue
        home_is_fav = home_exp_0 > away_exp_0

        base_home = base_row[home_col]
        base_away = base_row[away_col]
        prev_home = base_home
        prev_away = base_away

        for _, row in group.iterrows():
            minute = row['MINUTES']
            h_val = row[home_col]
            a_val = row[away_col]
            change = None

            if role == 'Total':
                if h_val != prev_home or a_val != prev_away:
                    change = (h_val + a_val) - (base_home + base_away)
                    prev_home, prev_away = h_val, a_val
            else:
                relevant_col = home_col if (role == 'Favourite' and home_is_fav) or (role == 'Underdog' and not home_is_fav) else away_col
                prev_val = prev_home if relevant_col == home_col else prev_away
                curr_val = h_val if relevant_col == home_col else a_val
                base_val = base_home if relevant_col == home_col else base_away

                if curr_val != prev_val:
                    change = curr_val - base_val
                    if relevant_col == home_col:
                        prev_home = curr_val
                    else:
                        prev_away = curr_val

            if change is not None:
                change_data.append({
                    'MINUTES': minute,
                    'Change': change,
                    'GOAL_EXP_HOME': row['GOAL_EXP_HOME'],
                    'GOAL_EXP_AWAY': row['GOAL_EXP_AWAY'],
                    'GOALS_HOME': row['GOALS_HOME'],
                    'GOALS_AWAY': row['GOALS_AWAY'],
                    'EVENT_START_TIMESTAMP': row['EVENT_START_TIMESTAMP'],
                    'SRC_EVENT_ID': event_id
                })

    return pd.DataFrame(change_data)

# --- Generate Graphs ---
plots = []

if exp_types:
    # Dynamic layout
    if len(exp_types) == 1:
        layout_cols = [st.container()]
        n_cols = 1
        fig_size = (14, 6)
    
    elif len(exp_types) <= 3:
        layout_cols = st.columns(2)
        n_cols = 2
        fig_size = (12, 6)
    else:
        layout_cols = st.columns(3)
        n_cols = 3
        fig_size = (10, 5.5)

    for i, exp_type in enumerate(exp_types[:6]):
        df_changes = compute_exp_by_role(df, exp_type)

        df_changes['Favouritism'] = df_changes.apply(classify_favouritism, axis=1)
        df_changes['Scoreline'] = df_changes.apply(classify_scoreline_simple, axis=1)
        df_changes = df_changes[df_changes['Favouritism'].isin(fav_filter) & df_changes['Scoreline'].isin(scoreline_filter)]

        df_changes['Time Band'] = pd.cut(
            df_changes['MINUTES'],
            bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 1000],
            right=False,
            labels=[f"{i}-{i+5}" for i in range(0, 90, 5)]
        )

        avg_change = df_changes.groupby('Time Band')['Change'].mean()

        title_line = f"{exp_type} Expectancy Change"
        filter_context = f"Date: {start_date.strftime('%d/%m/%Y')}–{end_date.strftime('%d/%m/%Y')} | Fav: {', '.join(fav_filter)} | Scoreline: {', '.join(scoreline_filter)}"

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
        ax.set_title(f"{title_line}\n{filter_context}", fontsize=12)
        ax.set_xlabel("Time Band (Minutes)")
        ax.set_ylabel("Avg Change")
        ax.grid(True)
        fig.tight_layout()

        plots.append(fig)

        with layout_cols[i % n_cols]:
            st.pyplot(fig, use_container_width=True)
else:
    st.warning("Please select at least one expectancy type to display charts.")

# --- Export to PDF Button ---
def export_all_to_pdf(figures):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    buffer.seek(0)
    return buffer

if plots:
    st.download_button(
        label="Download All Charts as PDF",
        data=export_all_to_pdf(plots),
        file_name="expectancy_graphs.pdf",
        mime="application/pdf"
    )
