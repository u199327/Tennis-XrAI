from src.sql_react_agent import ReActSQLAgent
import streamlit as st

def generate_match_tendency_analysis():
    agent = ReActSQLAgent(
        llm_model=None,  # Set your model here
        db_manager=st.session_state.db,
        table_provider=st.session_state.table_provider,
        sql_dialect='SQLite'
    )

    match_data = {
        "(2-1, 0-0)": ["Struff", "Tsitsipas", "Deuce"],
        "(2-1, 0-15)": ["Struff", "Tsitsipas", "Advantage"],
        "(2-1, 0-30)": ["Struff", "Tsitsipas", "Deuce"],
        "(2-1, 0-40)": ["Struff", "Tsitsipas", "Advantage"],
        "(3-1, 15-0)": ["Struff", "Struff", "Advantage"],
        "(3-1, 30-0)": ["Struff", "Struff", "Deuce"],
        "(3-1, 30-15)": ["Tsitsipas", "Struff", "Advantage"],
        "(3-1, 30-30)": ["Tsitsipas", "Struff", "Deuce"],
        "(3-1, 30-40)": ["Tsitsipas", "Struff", "Advantage"],
        "(3-1, 40-40)": ["Struff", "Struff", "Deuce"],
        "(3-1, 40-Ad)": ["Tsitsipas", "Struff", "Advantage"]
    }

    prompt = f"""
    You are a tennis tendency analyst. Below is point-by-point data gathered from video analysis of the match between Tsitsipas and Struff.

    Each entry contains:
    - Result: the current point score
    - Winner: the player who won the point
    - Server: the player serving the point
    - Serve Side: either Deuce or Advantage side

    Example data format:
    - "(0-0, 0-0)": ["Sinner", "Sinner", "Deuce"],
    - "(0-0, 15-0)": ["Sinner", "Sinner", "Advantage"],
    ...

    Your task is to analyze this match and extract useful tendencies and insights, such as:
    - Number of points won consecutively
    - Out of X points, how many each player has won
    - Out of X points on the Advantage/Deuce side, how many each player has won
    - Out of X points when player 1/player 2 is serving, how many they have won
    - Out of X points when player 1/player 2 is returning, how many they have won

    After identifying these tendencies (or others that you find useful), provide brief strategic recommendations for each player — for example, what patterns to reinforce or adjust.

    Output instructions:
    - Use bullet points
    - Keep the analysis concise (maximum 100 words)
    - Make insights and advice easy to understand
    - Do not invent any data

    Here is the match data:
    {match_data}
    """


    #response_messages = agent.invoke(prompt)
    #return response_messages["messages"][-1].content
    return """Let's analyze the match data provided for the tennis game between Tsitsipas and Struff. Below are the extracted tendencies:
**Number of Consecutive Points Won**
- Struff:
  - Won two consecutive points in two instances: 
    - From 15-0 to 30-0
    - From 0-0 to 0-15 to 0-30
- Tsitsipas:
  - Won three consecutive points from 30-15 to 30-40

**Points Won by Each Player**
- Tsitsipas: 5 points won
- Struff: 6 points won

**Points on Serve Side**
- **Deuce side**: 
  - Tsitsipas: 3 points
  - Struff: 1 point
- **Advantage side**:
  - Tsitsipas: 2 points
  - Struff: 5 points

**Points Won When Serving**
- Tsitsipas serving: 1 out of 4 points won
- Struff serving: 5 out of 7 points won

**Points Won When Returning**
- Tsitsipas returning: 4 out of 7 points won
- Struff returning: 1 out of 4 points won

---

### 🎯 Strategic Recommendations

**Tsitsipas**
- Strengthen return play, where he performs better — especially focus on Advantage side returns.
- Improve serve effectiveness; consider changes in serve placement or strategy.

**Struff**
- Keep leveraging strong serve performance, particularly on the Advantage side.
- Focus on improving return effectiveness, especially on the Deuce side.
"""
