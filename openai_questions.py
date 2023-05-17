import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

openai = OpenAI(model_name='text-davinci-003')

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

prefix = """The following are exerpts from conversations with an AI assistant. 
The assistant is always sarcastic and witty. Here are a few examples:"""

suffix = """
User: {query}
AI:"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

query="What is the meaning of life?"

print(few_shot_prompt_template.format(query=query))

print(openai(few_shot_prompt_template.format(query=query)))
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
