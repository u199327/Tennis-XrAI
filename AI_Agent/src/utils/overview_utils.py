import matplotlib.pyplot as plt
import numpy as np
from src.sql_react_agent import ReActSQLAgent
import streamlit as st

def get_serve_performance(db, player_name):
    """Fetch serve-related statistics for the given player."""
    query = f"""
        SELECT 
            ROUND(AVG(aces), 2) AS avg_aces,
            ROUND(AVG(double_faults), 2) AS avg_double_faults,
            ROUND(AVG(first_in * 100.0 / serve_pts), 2) AS avg_first_serve_in_percentage,
            ROUND(AVG(first_won * 100.0 / first_in), 2) AS avg_first_serve_won_percentage,
            ROUND(AVG(second_won * 100.0 / second_in), 2) AS avg_second_serve_won_percentage
        FROM charting_m_stats_Overview_top25_2024
        WHERE player = '{player_name}';
    """
    return db._execute_query(query)

def get_break_point_efficiency(db, player_name):
    """Fetch break point saving efficiency for the given player."""
    query = f"""
    WITH player_stats AS (
        SELECT 
            ROUND(SUM(break_points_saved) * 100.0 / NULLIF(SUM(break_pts), 0), 2) AS avg_break_points_saved_percentage
        FROM charting_m_stats_Overview_top25_2024
        WHERE break_pts > 0
        AND player = '{player_name}'
        GROUP BY player
    ),
    global_avg AS (
        SELECT 
            ROUND(SUM(break_points_saved) * 100.0 / NULLIF(SUM(break_pts), 0), 2) AS global_avg_break_points_saved_percentage
        FROM charting_m_stats_Overview_top25_2024
        WHERE break_pts > 0
    )
    SELECT 
        player_stats.avg_break_points_saved_percentage,
        global_avg.global_avg_break_points_saved_percentage
    FROM player_stats, global_avg;
    """
    return db._execute_query(query)
    
def get_return_performance(db, player_name):
    """Fetch return performance statistics and compare with other players."""
    query = f"""
        WITH player_stats AS (
            SELECT 
                player,
                ROUND(AVG(return_pts_won * 100.0 / return_pts), 2) AS avg_return_points_won_percentage
            FROM charting_m_stats_Overview_top25_2024
            WHERE return_pts > 0
            GROUP BY player
        )
        SELECT 
            (SELECT avg_return_points_won_percentage FROM player_stats WHERE player = '{player_name}') AS player_avg,
            ROUND(AVG(avg_return_points_won_percentage), 2) AS overall_avg
        FROM player_stats;
    """
    return db._execute_query(query)

def get_winners_and_errors(db, player_name):
    """Fetch winners and unforced errors statistics for the given player."""
    query = f"""
        SELECT 
            ROUND(AVG(winners), 2) AS avg_winners,
            ROUND(AVG(unforced), 2) AS avg_unforced_errors,
            ROUND(AVG(winners_forehand), 2) AS avg_forehand_winners,
            ROUND(AVG(winners_backhand), 2) AS avg_backhand_winners,
            ROUND(AVG(unforced_forehand), 2) AS avg_forehand_unforced_errors,
            ROUND(AVG(unforced_backhand), 2) AS avg_backhand_unforced_errors
        FROM charting_m_stats_Overview_top25_2024
        WHERE player = '{player_name}';
    """
    return db._execute_query(query)

def generate_full_performance_summary(serve_data, break_point_data, return_data, winners_errors_data):
    agent = ReActSQLAgent(
        llm_model=None, 
        db_manager=st.session_state.db, 
        table_provider=st.session_state.table_provider,
        sql_dialect='SQLite'
    )

    prompt_template = """
    You are an expert tennis analyst. Based on the following performance statistics, write a concise summary (max 150 words) with insightful analysis and improvement advice. Focus on trends and interpretation rather than quoting numbers.

    **Serve Performance:**
    - Average Aces: {avg_aces}
    - Average Double Faults: {avg_double_faults}
    - First Serve In %: {avg_first_serve_in_percentage}%
    - First Serve Points Won %: {avg_first_serve_won_percentage}%
    - Second Serve Points Won %: {avg_second_serve_won_percentage}%

    **Break Point Efficiency:**
    - Player Break Points Saved %: {player_bps_saved}%
    - Tour Average Break Points Saved %: {global_bps_saved}%

    **Return Performance:**
    - Player Return Points Won %: {player_return}%
    - Tour Average Return Points Won %: {global_return}%

    **Winners and Unforced Errors:**
    - Avg Winners: {avg_winners}
    - Avg Unforced Errors: {avg_unforced_errors}
    - Forehand Winners: {avg_forehand_winners}, Backhand Winners: {avg_backhand_winners}
    - Forehand Unforced Errors: {avg_forehand_unforced_errors}, Backhand Unforced Errors: {avg_backhand_unforced_errors}

    Highlight:
    - Serve reliability and effectiveness.
    - Break point and return performance compared to tour averages.
    - Shotmaking tendencies (winners/errors balance).
    - Areas for improvement and tactical advice.
    
    Use bullet points for better user interpretation
    """

    def safe_value(data, index):
        return round(data[0][index], 2) if data and not isinstance(data[0], str) else "N/A"

    serve_dict = {
        "avg_aces": safe_value(serve_data, 0),
        "avg_double_faults": safe_value(serve_data, 1),
        "avg_first_serve_in_percentage": safe_value(serve_data, 2),
        "avg_first_serve_won_percentage": safe_value(serve_data, 3),
        "avg_second_serve_won_percentage": safe_value(serve_data, 4),
    }

    break_point_dict = {
        "player_bps_saved": safe_value(break_point_data, 0),
        "global_bps_saved": safe_value(break_point_data, 1),
    }

    return_dict = {
        "player_return": safe_value(return_data, 0),
        "global_return": safe_value(return_data, 1),
    }

    winners_dict = {
        "avg_winners": safe_value(winners_errors_data, 0),
        "avg_unforced_errors": safe_value(winners_errors_data, 1),
        "avg_forehand_winners": safe_value(winners_errors_data, 2),
        "avg_backhand_winners": safe_value(winners_errors_data, 3),
        "avg_forehand_unforced_errors": safe_value(winners_errors_data, 4),
        "avg_backhand_unforced_errors": safe_value(winners_errors_data, 5),
    }

    all_data = {**serve_dict, **break_point_dict, **return_dict, **winners_dict}
    formatted_prompt = prompt_template.format(**all_data)

    response = agent.invoke(formatted_prompt)

    return response["messages"][-1].content