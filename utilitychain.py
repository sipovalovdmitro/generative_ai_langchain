import os
import inspect
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain, LLMMathChain
from langchain.callbacks import get_openai_callback
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    
    return result

llm_math = LLMMathChain(llm=llm, verbose=True)

# print(llm_math.prompt.template)

print(inspect.getsource(llm_math._call))

# count_tokens(llm_math, "What is 13 raised to the .3432 power?")