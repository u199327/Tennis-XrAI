import logging
import wikipediaapi
from typing import Annotated

from typing import List, Dict, Union
from typing_extensions import TypedDict
from functools import wraps
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.utils.utils import init_azure_llm
from langchain_core.tools import tool, Tool
from src.semantic_model import TableProvider
from src.sql_agents import BaseSQLAgent
from src.db_manager import DatabaseManager
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessage
from src.db_manager import DatabaseManager, DBConfig
from src.semantic_model import TableProvider

logger = logging.getLogger(__name__)

class WikipediaTool:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='TFG_Tennis_Project/1.0 alejandropastorrubio@gmail.com'
        )

    def fetch_wikipedia_page(self, page_title: str) -> str:
        page = self.wiki_wiki.page(page_title)
        if page.exists():
            return page.text
        else:
            return "Page not found."

def create_react_sql_agent_for_adalytics_db():
    config = DBConfig.from_env()
    db_manager = DatabaseManager(config)
    table_provider = TableProvider(tables_dict)
    agent = ReActSQLAgent(llm_model=None, db_manager=db_manager, table_provider=table_provider, sql_dialect='ORACLE')
    return agent

def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        logger.info(f"Calling {method_name} with args {args}, {kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    call_count: int
    

class ToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> dict:
        """
        Execute tools found "tool calls" in the last message in the state
        """
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
            except Exception as e:
                logger.error(f"Error executing {tool_call['name']}: {str(e)}")
                tool_result = f"Error executing tool: {str(e)}"
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
class ReActSQLAgent(BaseSQLAgent):

    def __init__(self, llm_model, db_manager: DatabaseManager, table_provider: TableProvider, sql_dialect: str = "standard SQL"):
        super().__init__(llm_model, db_manager, table_provider, sql_dialect)
        
        self.graph = None
        self.tools = None
        
        self._init_tools()
        self._init_graph()
        self.llm_with_tools = self.model.bind_tools(self.tools)
    
    @log_method_call
    def _call_llm(self, state: State):
        """
        Call the LLM with the messages in the state
        """
        
        messages = state['messages']
        llm_response = self.llm_with_tools.invoke(messages)
        return {"messages": [llm_response], "call_count": state.get("call_count", 0) + 1}
    
    def _route_tools(
        self,
        state: State
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        
        if state.get("call_count", 0) > 10:
            return END
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        
        return END
    
    def _init_tools(self):
        """
        Initialize the tools for the agent
        """
        @tool
        def get_db_info():
            """ Returns a list of tables with a short description and PK columns and their relationships """
            return self._get_db_info()
        @tool
        def get_tables_info(table_names: list[str]):
            """ Returns detailed information about the tables and its columns """
            return self._get_tables_info(table_names)

        @tool
        def execute_query(query: str) -> str:
            """ Executes a SQL query and returns the result

            Args:
                query (str): SQL query to execute

            Returns:
                Union[List[Tuple], str]: Query results as list of dictionaries, or error message
            """
            return self._execute_query(query)

        self.tools = [get_db_info, get_tables_info, execute_query]

    @log_method_call
    def _get_db_info(self):
        """
        Returns a list of tables with a short description and PK columns and their relationships
        """

        all_table_names = self.table_provider.available_tables
        all_tables = self.table_provider.get_tables(all_table_names)
        table_info = "\n\n".join([t.get_table_short_description() for t in all_tables])


        return f"The following tables are available in the data base: \n\n{table_info}"

    @log_method_call
    def _get_tables_info(self, table_names: list[str]):
        """
        Returns detailed information about the tables and its columns
        """
        try:
            tables = self.table_provider.get_tables(table_names)
            table_info = "\n\n".join([t.get_table_context_for_llm() for t in tables])
        except Exception as e:
            return "Error getting table information: " + str(e)

        return f"Information about the tables: \n\n{table_info}"

    @log_method_call
    def _execute_query(self, query: str)-> str:
        """
        Executes a SQL query and returns the result
        """
        #If query ends with ";" remove it, it raises error through python interfaces
        if query.endswith(";"):
            query = query[:-1]

        try:
            # Execute the query and return results
            result = self.db_manager.safe_execute_query(query)
            return f"Query executed succesfully returning:\n{result}"
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _init_graph(self):
        """ Initialize the graph for the agent"""
        
        tool_node = ToolNode(self.tools)

        graph_builder = StateGraph(State)
        graph_builder.add_node("call_llm", self._call_llm)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge(START, "call_llm")
        graph_builder.add_conditional_edges("call_llm", self._route_tools, {"tools": "tools", END: END})
        graph_builder.add_edge("tools", "call_llm")
        self.graph = graph_builder.compile()

    def _get_system_message(self) -> dict:
        """
        Get the system message for the agent
        """
        
        system_prompt = f"""You are an expert Data Engineer specializing in {self.sql_dialect} databases, SQL optimization, and data modeling.

        For each user question, follow this exact process:
        1. Analyze the question to determine what data is needed
        2. Use available tools to inspect database schema if needed
        3. Develop a step-by-step query plan specifying:
        - Tables required (prioritize smallest sufficient set)
        - Join conditions (verify against foreign keys)
        - Columns needed (only those directly answering the question)
        - Filter conditions (WHERE clauses)
        - Any aggregations (GROUP BY) or sorting (ORDER BY)
        4. Validate all identifiers against schema to prevent typos
        5. Apply CAST/CASE WHEN only when explicitly needed
        6. Execute the query only after full validation

        Adhere strictly to these SQL quality standards:
        - SELECT only columns referenced in the question
        - Verify every JOIN condition matches documented foreign keys
        - Confirm all WHERE/GROUP BY/ORDER BY columns exist
        - Use the most efficient query structure possible for {self.sql_dialect}
        - Never include redundant operations or columns

        Work iteratively using the Reason-Act-Observe loop:
        Plan → Action (tool use) → Observation → Repeat until solution is complete

        Special attention to:
        1. Correct join columns (validate against schema)
        2. Appropriate filtering conditions
        3. Proper aggregation when needed
        4. Efficient sorting when requested
        5. Type conversions only when necessary

        When ready with a fully validated query, execute it using the execute_query tool.

        Begin by analyzing the current question and planning your first steps:"""

        return {"role": "system", "content": system_prompt}
    
    
    def invoke(self, user_query: str) -> State:
        """Invoke the agent with a user query."""
        state = {"messages": [self._get_system_message(), {"role": "user", "content": user_query}]}
        return self.graph.invoke(state)
    
    def extract_sql_query(self, messages) -> str:
        inverted_messages = messages[::-1]
        for message in inverted_messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    if tool_call["name"] == "execute_query":
                        return tool_call["args"]["query"]
        
    def invoke_for_sql_query(self, user_query: str) -> str:
        """ Invoke the agent with a user query and return the last SQL query in messages """
        messages = self.invoke(user_query)["messages"]
        sql_query = self.extract_sql_query(messages)
        if sql_query.endswith(";"):
            sql_query = sql_query[:-1]
        return sql_query

    

class ToolDispatcher:
    def __init__(self, sql_agent: ReActSQLAgent, wiki_tool: WikipediaTool, llm_model):
        self.sql_agent = sql_agent
        self.wiki_tool = wiki_tool
        self.llm_model = llm_model

    def decide_tool(self, user_query: str) -> str:
        if "Wikipedia" in user_query or "wikipedia" in user_query or "internet" in user_query or "Internet" in user_query:
            return "wikipedia"
        else:
            return "sql"

    def invoke(self, user_query: str) -> AIMessage:
        tool_choice = self.decide_tool(user_query)
        if tool_choice == "sql":
            return self.sql_agent.invoke(user_query)['messages'][-1].content
        elif tool_choice == "wikipedia":
            page_title = self.extract_wikipedia_title(user_query)
            page_content = self.wiki_tool.fetch_wikipedia_page(page_title)
            summary = self.summarize_content(page_content)
            return summary
        else:
            return "I'm not sure which tool to use for this query."

    def extract_wikipedia_title(self, user_query: str) -> str:
        prompt = f"Extract the main topic or entity from the following query that corresponds to a Wikipedia page: '{user_query}'. Make sure to answer only the main topic or entity"
        response = self.sql_agent.invoke(prompt)['messages'][-1]
        
        page_title = response.content
        return page_title

    def summarize_content(self, content: str) -> str:
        prompt = f"Please provide a concise summary of the following text:\n\n{content}"
        response = self.sql_agent.invoke(prompt)['messages'][-1]
        
        summary = response.content
        return summary