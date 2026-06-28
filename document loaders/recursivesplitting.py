from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data = TextLoader("document loaders/notes.txt")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

print(len(chunks))
# print(chunks[0].page_content)
print(chunks[1].page_content)