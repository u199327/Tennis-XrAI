from src.utils.utils import init_azure_llm
import streamlit as st
import time
from src.sql_react_agent import ReActSQLAgent, WikipediaTool, ToolDispatcher
from langchain_core.messages.ai import AIMessage

def chat_assistant(db, table_provider):
    # Initialize the SQL agent and Wikipedia tool
    sql_agent = ReActSQLAgent(
        llm_model=None, 
        db_manager=db, 
        table_provider=table_provider,
        sql_dialect='SQLite'
    )
    wiki_tool = WikipediaTool()

    # Initialize the dispatcher
    dispatcher = ToolDispatcher(sql_agent, wiki_tool, llm_model=init_azure_llm)

    st.title("💬 Chat with Tennis Assistant")
    st.markdown("Ask about player stats, match analysis, or strategies!")
    
    st.session_state.show_steps = st.checkbox("Show assistant steps", value=st.session_state.get("show_steps", False))
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about player stats, match analysis, or strategy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_container = st.container()
            loading_message = st.empty()
            
            for i in range(3):
                loading_message.markdown(f"Analyzing data{'.' * (i % 3 + 1)}")
                time.sleep(1)
            
            # Use the dispatcher to handle the query
            final_response = dispatcher.invoke(prompt)
            loading_message.empty()
            
            #final_response = response_message.content
            
            if st.session_state.show_steps:
                for message in final_response.tool_calls:
                    if message['name'] == 'execute_query':
                        with message_container:
                            st.markdown(final_response.content)
                            st.divider()
            
            st.markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})
