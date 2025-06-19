import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.stroke_analysis_utils import get_stroke_performance, plot_combined_stroke_bar_chart, generate_stroke_performance_summary

def display_service_analysis():
    if "db" not in st.session_state:
        st.error("Database connection is not available. Please go back to the main page to initialize it.")
        return

    db = st.session_state.db

    st.set_page_config(page_title="Service Analysis", page_icon="🎾")
    st.title("🎾 Stroke Analysis")
    st.markdown("Analyze stroke performance based on different factors.")

    player_name = st.text_input("Enter player's name:")

    stroke_type = st.selectbox("Select stroke type:", ["Forehand", "Backhand"])
    stroke_prefix = "F" if stroke_type == "Forehand" else "B"

    if player_name:
        data = get_stroke_performance(db, player_name, stroke_prefix)
        analysis_summary = generate_stroke_performance_summary(data, stroke_prefix, player_name)
        #analysis_summary = "BLA BLA BLA"
        st.subheader("📊 Stroke Analysis Summary")
        st.markdown(analysis_summary)
        
        if data:
            stroke_data = {
                row[0]: {
                    "winner_percentage": row[1],
                    "induced_forced_percentage": row[2],
                    "unforced_error_percentage": row[3],
                }
                for row in data
            }

            st.subheader(f"Stroke Performance: {stroke_type}")

            selected_strokes = [s for s in stroke_data.keys() if s.startswith(stroke_prefix)]

            selected_strokes = [s for s in selected_strokes if s in ['F-XC', 'F-DTM', 'F-DTL']] if stroke_prefix == "F" else \
                [s for s in selected_strokes if s in ['B-XC', 'B-DTM', 'B-DTL']]

            bar_fig = plot_combined_stroke_bar_chart(stroke_data, stroke_prefix=stroke_prefix)
            st.plotly_chart(bar_fig, use_container_width=True)

        else:
            st.warning("No stroke performance data found.")

# Run the service analysis page
display_service_analysis()
