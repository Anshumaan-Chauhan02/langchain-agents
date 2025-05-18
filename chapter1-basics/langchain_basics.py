from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Defining the system prompt (how AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles"
)

# User input provided by the user to LLM
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for an article.
    The article is here for you the examine:

    ---

    {article}

    ---

    The name should be based of the context of the article.
    Be creative, but make sure the names are clear, catchy, and relevant to the theme of the article.

    Only output the article name, no other explanation or text can be provided.
    """,
    input_variables=["article"]
)

# Replaces the {article} with the provided string
# print(user_prompt.format(article="TEST STRING"))

# ChatPromptTemplate is used to combine both user and system prompts
# It also prefixes each individual message with it's role, i.e., System:, Human:
first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
# print(first_prompt.format(article="TEST STRING"))

# Define the LLM Model
creative_llm = OllamaLLM(model="deepseek-r1:1.5b")

# We can chain together our first_prompt template and the ollama chat object to create a simple chain
# We use LangChain Expression Language (LCEL) to construct our chain.
# We use pipe (|) operator to say the output from the left of the pipe will be fed into the input to the right of the pipe
chain_one = (
    {"article": lambda x: x["article"]}
    | first_prompt
    | creative_llm
    | {"article_title": lambda x: x}
)

# Let's invoke our first chain
article = open("article.txt", "r").read()
article_title = chain_one.invoke({"article": article})
# print(article_title)

# Let's prompt LLM to generate article descriptions for us
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate articles."
)
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a description for the article.
    The article is here for you the examine:

    ---

    {article}

    ---

    Here is the article title '{article_title}'.
    Output the SEO friendly article description. 
    Do not output anything over than the description.
    """,
    input_variables=["article", "article_title"]
)
second_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
chain_two = (
    {
        "article": lambda x: x["article"],
        "article_title": lambda x: x["article_title"]
    }
    | second_prompt
    | creative_llm
    | {"summary": lambda x: x}
)
article = open("article.txt", "r").read()
article_description = chain_two.invoke(
    {
        "article": article,
        "article_title": article_title['article_title']
    }
)
# print(article_description['summary'])


# Now let's look at the Structured output feature
third_user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a new paragraph for the
    article. The article is here for you to examine:

    ---

    {article}

    ---

    Choose one paragraph to review and edit. During your edit
    ensure you provide constructive feedback to the user so they
    can learn where to improve their own writing.
    """,
    input_variables=["article"]
)
# prompt template 3: creating a new paragraph for the article
third_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    third_user_prompt
])
# We create a pydantic object describing the output format we need.
# This format description is then passed to our model using the with_structured_output method:
from pydantic import BaseModel, Field
class Paragraph(BaseModel):
    original_paragraph: str = Field(description="The original paragraph")
    edited_paragraph: str = Field(description="The improced edited paragraph")
    feedback: str = Field(description="Constructive feedback on the original paragraph")


# Forces LLM that it has to output a dictionary with the specified Field(s)
# with_structured_output gives issue with OllamaLLM, therefore we need to redefine our model as follows
from langchain_ollama import ChatOllama
creative_llm = ChatOllama(model='deepseek-r1:1.5b', temperature=0.9)
structured_llm = creative_llm.with_structured_output(Paragraph)
chain_three = (
    {"article": lambda x: x["article"]}
    | third_prompt
    | structured_llm
    | {
        "model_output": lambda x: x,  # Returns a Paragraph object
        "original_paragraph": lambda x: x.original_paragraph,
        "edited_paragraph": lambda x: x.edited_paragraph,
        "feedback": lambda x: x.feedback
    }
)
out = chain_three.invoke({"article": article})
print(out)
