from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# data = TextLoader("document loaders/notes.txt")
splitter= CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

docs = TextLoader("document loaders/notes.txt")
docs = docs.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()





# print(docs[0].page_content)



