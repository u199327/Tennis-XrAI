
import matplotlib.pyplot as plt
import numpy as np
from src.sql_react_agent import ReActSQLAgent
import streamlit as st


def get_return_quality(db, player_name, point_context):
    """Fetch return quality data for the given player and context."""
    query = f"""
        SELECT 
            AVG(returnable) AS avg_returnable,
            AVG(shallow) AS avg_shallow,
            AVG(deep) AS avg_deep,
            AVG(very_deep) AS avg_very_deep
        FROM charting_m_stats_ReturnDepth_top25_2024
        WHERE player = '{player_name}'
        GROUP BY point_context
        HAVING point_context = '{point_context}';
    """
    return db._execute_query(query)

def plot_return_quality_comparison_fh_bh(db, player_name):
    """Generate and return a bar plot comparing FH and BH returns along with the data."""
    fh_data = get_return_quality(db, player_name, "fh")
    bh_data = get_return_quality(db, player_name, "bh")
    
    if not fh_data or not bh_data:
        return None, None
    
    categories = ["Returnable", "Shallow", "Deep", "Very Deep"]
    fh_values = list(fh_data[0])
    bh_values = list(bh_data[0])
    
    x = np.arange(len(categories))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, fh_values, width, label="Forehand Return", color="#A7C7E7")
    ax.bar(x + width/2, bh_values, width, label="Backhand Return", color="#FFB7B2")
    
    ax.set_xlabel("Return Type")
    ax.set_ylabel("Average Values")
    ax.set_title(f"Return Quality Forehand vs Backhand Comparison for {player_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    return fig, {"categories": categories, "fh_values": fh_values, "bh_values": bh_values}

def plot_return_quality_comparison_1st_2nd(db, player_name):
    """Generate and return a bar plot comparing 1st and 2nd serve returns along with the data."""
    first_serve_data = get_return_quality(db, player_name, "v1st")
    second_serve_data = get_return_quality(db, player_name, "v2nd")
    
    if not first_serve_data or not second_serve_data:
        return None, None
    
    categories = ["Returnable", "Shallow", "Deep", "Very Deep"]
    first_values = list(first_serve_data[0])
    second_values = list(second_serve_data[0])
    
    x = np.arange(len(categories))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, first_values, width, label="1st Serve Return", color="#A7C7E7")
    ax.bar(x + width/2, second_values, width, label="2nd Serve Return", color="#FFB7B2")
    
    ax.set_xlabel("Return Type")
    ax.set_ylabel("Average Values")
    ax.set_title(f"Return Quality 1st Serve vs 2nd Serve Comparison for {player_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    return fig, {"categories": categories, "first_values": first_values, "second_values": second_values}


def generate_return_analysis_summary(fh_bh_data, first_second_data):
    agent = ReActSQLAgent(
        llm_model=None, 
        db_manager=st.session_state.db, 
        table_provider=st.session_state.table_provider,
        sql_dialect='SQLite'
    )
    
    prompt = """
    You are an expert tennis analyst. Given the following return quality data, generate a concise textual summary highlighting key insights and providing some advice.
    
    **Forehand vs Backhand Return Quality:**
    - Returnable: {fh_returnable}% FH / {bh_returnable}% BH
    - Shallow: {fh_shallow}% FH / {bh_shallow}% BH
    - Deep: {fh_deep}% FH / {bh_deep}% BH
    - Very Deep: {fh_very_deep}% FH / {bh_very_deep}% BH
    
    **1st vs 2nd Serve Return Quality:**
    - Returnable: {first_returnable}% 1st / {second_returnable}% 2nd
    - Shallow: {first_shallow}% 1st / {second_shallow}% 2nd
    - Deep: {first_deep}% 1st / {second_deep}% 2nd
    - Very Deep: {first_very_deep}% 1st / {second_very_deep}% 2nd
    
    Provide a short summary in a few bullet points, focusing on text insights rather than numbers. Identify:
    - The strongest and weakest return types.
    - Differences between forehand and backhand returns.
    - Differences between returning first and second serves.
    - Any interesting patterns or trends.
    - Advice where we can take an advantage agains that player
    
    Do not exceed 150 words.
    """
    
    if fh_bh_data:
        fh_bh_dict = {
            "fh_returnable": round(fh_bh_data["fh_values"][0], 1),
            "fh_shallow": round(fh_bh_data["fh_values"][1], 1),
            "fh_deep": round(fh_bh_data["fh_values"][2], 1),
            "fh_very_deep": round(fh_bh_data["fh_values"][3], 1),
            "bh_returnable": round(fh_bh_data["bh_values"][0], 1),
            "bh_shallow": round(fh_bh_data["bh_values"][1], 1),
            "bh_deep": round(fh_bh_data["bh_values"][2], 1),
            "bh_very_deep": round(fh_bh_data["bh_values"][3], 1),
        }
    else:
        fh_bh_dict = {key: "N/A" for key in ["fh_returnable", "fh_shallow", "fh_deep", "fh_very_deep", "bh_returnable", "bh_shallow", "bh_deep", "bh_very_deep"]}
    
    if first_second_data:
        first_second_dict = {
            "first_returnable": round(first_second_data["first_values"][0], 1),
            "first_shallow": round(first_second_data["first_values"][1], 1),
            "first_deep": round(first_second_data["first_values"][2], 1),
            "first_very_deep": round(first_second_data["first_values"][3], 1),
            "second_returnable": round(first_second_data["second_values"][0], 1),
            "second_shallow": round(first_second_data["second_values"][1], 1),
            "second_deep": round(first_second_data["second_values"][2], 1),
            "second_very_deep": round(first_second_data["second_values"][3], 1),
        }
    else:
        first_second_dict = {key: "N/A" for key in ["first_returnable", "first_shallow", "first_deep", "first_very_deep", "second_returnable", "second_shallow", "second_deep", "second_very_deep"]}
    
    prompt = prompt.format(**fh_bh_dict, **first_second_dict)
    
    response_messages = agent.invoke(prompt)
    
    return response_messages["messages"][-1].content