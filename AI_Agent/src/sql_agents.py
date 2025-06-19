import logging
from typing import Annotated, TypedDict, Union

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages

from src.db_manager import DatabaseManager
from src.semantic_model import TableProvider
from src.utils.utils import init_azure_llm

logger = logging.getLogger(__name__)

#set info level
logger.setLevel(logging.INFO)



load_dotenv("../.env")

class AgentState(TypedDict):
    plan: str
    user_query: str
    sql_query: Annotated[str, None]

class BaseSQLAgent(ABC):

    def __init__(self, llm_model: Union[BaseChatModel, None], db_manager: DatabaseManager, table_provider: TableProvider, sql_dialect: str = "standard SQL") -> None:
        
        if llm_model:
            self.model = llm_model
        else:
            self.model = init_azure_llm()
        
        self.sql_dialect = sql_dialect
        self.db_manager = db_manager
        self.table_provider = table_provider
    
    @abstractmethod
    def invoke(self, user_query: str) -> AIMessage:
        """Invoke the agent with a user query."""
        pass

    @abstractmethod
    def invoke_for_sql_query(self, user_query: str) -> str:
        """Invoke the agent with a user query and return the SQL query."""
        pass

class CoTSQLAgent(BaseSQLAgent):
    
    def invoke(self, user_query: str) -> AIMessage:
        """Invoke the agent with a user query."""
        state = {"plan": "", "user_query": user_query, "sql_query": None}
        state = self.cot_query_to_sql(state)
        sql_query = state["sql_query"]
        try:
            result = self.db_manager.execute_query(sql_query)
        except Exception as e:
            result = str(e)
        
        return AIMessage(content=state["plan"] + "\n\n Executing this query gives the result: \n\n" + result)
        
    def get_semantic_model_context(self, table_names: list[str]) -> str:
        """
        Get the context of the semantic model to be used in the LLM.
        """
        tables =  self.table_provider.get_tables(table_names)
        context = "\n".join([table.get_table_context_for_llm() for table in tables])

        return context

    def cot_query_to_sql(self, state: AgentState) -> AgentState:
        """Plan how to answer the user query with SQL using a structured output chain."""

        all_tables= self.table_provider.available_tables
        SEMANTIC_MODEL_CONTEXT = self.get_semantic_model_context(all_tables)
        
        # Initialize the LLM
        llm = init_azure_llm()

        # Define the prompt template
        prompt = PromptTemplate(
            template="""You are an expert Data Engineer with experitse in SQL, databases and data modeling.

            Given a user question, your job is to create a step-by-step plan for writing an SQL query in {sql_dialect} to answer the question.
            In your plan you should mention, what tables to use, how to join them, what columns to select, and any other relevant steps.
            Finally, write the SQL query that answers the user question within a markdown code block.

            Here are the available tables and their respective columns:
            {semantic_model}

            User question: {query}

            Lets think step by step on what operations we need to perform to answer the question:
            """,
            input_variables=["query"],
            partial_variables={"semantic_model" : SEMANTIC_MODEL_CONTEXT, "sql_dialect": self.sql_dialect}
        )

        # Create the chain
        plan_chain = prompt | llm

        # Execute the chain
        plan_output = plan_chain.invoke({"query": state["user_query"]}).content

        try:
            sql_query = plan_output.split("```sql")[1].split("```")[0].strip()	
        except:
            sql_query = "ParsingError: No SQL query found in the output."
        # Return the updated state
        state.update({"plan": plan_output, "sql_query": sql_query})
        return state

    def invoke_for_sql_query(self, user_query: str) -> str:
        """Run the COT and return sql query"""
        state = {"plan": "", "user_query": user_query, "sql_query": None}
        state = self.cot_query_to_sql(state)
        return state["sql_query"]


        

