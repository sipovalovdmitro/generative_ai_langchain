import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

