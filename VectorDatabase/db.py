from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="You are amazing", metadata={"source": "AI Book"}),
    Document(page_content="Python is a versatile programming language.", metadata={"source": "Python Handbook"}),
    Document(page_content="Machine learning enables computers to learn from data.", metadata={"source": "ML Notes"}),
    Document(page_content="Large Language Models are trained on massive text datasets.", metadata={"source": "LLM Guide"}),
    Document(page_content="Retrieval-Augmented Generation combines search with language models.", metadata={"source": "RAG Documentation"}),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db",
)