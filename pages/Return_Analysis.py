import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.utils.return_analysis_utils import plot_return_quality_comparison_fh_bh, plot_return_quality_comparison_1st_2nd, generate_return_analysis_summary

def display_return_analysis():
    """Streamlit UI for displaying FH vs BH return quality comparison."""
    if "db" not in st.session_state:
        st.error("Database connection is not available. Please go back to the main page to initialize it.")
        return
    
    db = st.session_state.db
    st.set_page_config(page_title="Return Quality Analysis", page_icon="🎾")
    st.title("🎾 Return Quality Analysis")
    st.markdown("Compare forehand and backhand return qualities.")
    
    player_name = st.text_input("Enter player's name:")
    
    if player_name:
        fig_1, first_second_data = plot_return_quality_comparison_1st_2nd(db, player_name)
        fig_2, fh_bh_data = plot_return_quality_comparison_fh_bh(db, player_name)

        analysis_summary = generate_return_analysis_summary(fh_bh_data, first_second_data)
        #analysis_summary = "BLA BLA BLA"
        st.subheader("📊 Return Analysis Summary")
        st.markdown(analysis_summary)
        
        if fig_1:
            st.pyplot(fig_1)
        else:
            st.warning("No return quality data found for this player.")


        if fig_2:
            st.pyplot(fig_2)
        else:
            st.warning("No return quality data found for this player.")
    else:
        st.warning("Please enter a player's name to analyze return quality.")


# Run the service analysis page
display_return_analysis()