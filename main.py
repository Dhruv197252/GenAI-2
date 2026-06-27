from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

data = PyPDFLoader("document loaders/Java_Backend_Notes.pdf")

docs = data.load()

template = ChatPromptTemplate(
    [("system", "You are a great summarizer agent"),
     "human", "{data}"]
)

final_prompt = template.format_messages(data = docs[4].page_content)


model = ChatMistralAI(model = "mistral-small-2506")
result = model.invoke(final_prompt)
print(result.content)


