import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature":1e-10}
)

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=['question'])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.run(question))

# qs=[
#     {'question':"Which NFT team won the Super Bowl in the 2010 season?"},
#     {'question':"If I am 6 ft 4 inches, how tall am I in centimeters?"},
#     {'question':"Who was the 12th person on the moon?"},
#     {'question':"How many eyes does a blade of grass have?"}
# ]

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])
qs_str =(
    "Which NFT team won the Super Bowl in the 2010 season?\n"+
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n"+
    "Who was the 12th person on the moon?\n"+
    "How many eyes does a blade of grass have?"
)

res = llm_chain.run(qs_str)
print(res)