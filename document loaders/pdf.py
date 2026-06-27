from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/notes.txt")

docs = data.load()

print(docs[0].page_content)



