
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("../.env")

def init_azure_llm():
    model_params = {
        "model_name": "gpt-4o",
        "deployment_name": "az-gpt-4o",
        "max_output_tokens": 4096,
        "context_window_tokens": 128000,
        "tokenizer": "o200k_base",
        "tokens_limit": 20000,
    }

    llm = AzureChatOpenAI(
        model_name = model_params["model_name"], 
        api_key = os.getenv("AZURE_KEY"),
        deployment_name = model_params["deployment_name"],
        azure_endpoint= os.getenv("AZURE_OPENAI_API_BASE"),
        api_version = "2024-12-01-preview"
    )
    return llm