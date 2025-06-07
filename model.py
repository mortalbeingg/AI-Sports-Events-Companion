import os
from dotenv import load_dotenv
from pydantic_ai.openai import OpenAIModel

load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
BASE_URL = os.getenv("BASE_URL")

if not OPENAI_API_KEY:
    raise EnvironmentError("LLM_API_KEY is missing in the .env file")

def get_openai_model():
    
    return OpenAIModel(
        model=MODEL_NAME,
        base_url = BASE_URL,
        api_key=OPENAI_API_KEY
    )
