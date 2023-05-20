import os
from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationKGMemory)
from langchain.callbacks import get_openai_callback
import tiktoken

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm=OpenAI(
    temperature=0,
    model_name='text-davinci-003'
    )