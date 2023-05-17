import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

davinci = OpenAI(model_name='text-davinci-003')

# template = """Question: {question}

# Answer: """

# prompt = PromptTemplate(template=template, input_variables=['question'])
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    },
    {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

llm_chain = LLMChain(
    prompt=example_prompt,
    llm=davinci
)
# question = "Which NFL team won the Super Bowl in the 2010 season?"

# print(llm_chain.run(question))

# qs=[
#     {'question':"Which NFT team won the Super Bowl in the 2010 season?"},
#     {'question':"If I am 6 ft 4 inches, how tall am I in centimeters?"},
#     {'question':"Who was the 12th person on the moon?"},
#     {'question':"How many eyes does a blade of grass have?"}
# ]

# res = llm_chain.generate(qs)
# print(res)
