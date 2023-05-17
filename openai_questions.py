import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

davinci = OpenAI(model_name='text-davinci-003')

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=['question'])

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)
question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.run(question))

qs=[
    {'question':"Which NFT team won the Super Bowl in the 2010 season?"},
    {'question':"If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question':"Who was the 12th person on the moon?"},
    {'question':"How many eyes does a blade of grass have?"}
]

res = llm_chain.generate(qs)
print(res)