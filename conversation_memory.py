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


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    
    return result

# conversation = ConversationChain(
#     llm=llm
# )

# print(conversation.prompt.template)

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

count_tokens(conversation_buf, "Good morning AI!")
count_tokens(conversation_buf, "My interest here is to explore the potential of integrating LLM.")
count_tokens(conversation_buf, "What is my aim?")

print(conversation_buf.memory.buffer)