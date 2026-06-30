import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
    max-width:1100px;
}

.title{
    font-size:48px;
    font-weight:700;
    text-align:center;
    background: linear-gradient(90deg,#4F8BF9,#8E44AD);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#BBBBBB;
    margin-bottom:35px;
}

.user-box{

    background:#1f2937;
    padding:15px;
    border-radius:15px;
    margin-bottom:12px;
}

.ai-box{

    background:#111827;
    padding:15px;
    border-radius:15px;
    margin-bottom:12px;
    border-left:5px solid #4F8BF9;
}

.stButton>button{

    width:100%;
    border-radius:12px;
    background:#4F8BF9;
    color:white;
    height:45px;
    font-size:16px;
    border:none;
}

.stTextInput input{

    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------

st.markdown(
    '<div class="title">📚 Retrieval Augmented Generation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by LangChain • ChromaDB • Mistral AI</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Load Models
# ---------------------------

@st.cache_resource
def load_rag():

    embedding_model = MistralAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
'I could not find the answer in the document.'
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ])

    return retriever, llm, prompt

retriever, llm, prompt = load_rag()

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:

    st.title("⚙️ Configuration")

    st.success("Vector DB Loaded")

    st.info("""
Search Type: MMR

Top Chunks: 4

Fetch: 10

Embedding:
Mistral Embeddings

LLM:
Mistral Small
""")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# Chat History
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f'<div class="user-box">👤 <b>You</b><br><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="ai-box">🤖 <b>Assistant</b><br><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ---------------------------
# User Input
# ---------------------------

query = st.chat_input("Ask anything from your documents...")

if query:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

    st.markdown(
        f'<div class="user-box">👤 <b>You</b><br><br>{query}</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Searching documents..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        final_prompt = prompt.invoke({
            "context":context,
            "question":query
        })

        response = llm.invoke(final_prompt)

    answer = response.content

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    st.markdown(
        f'<div class="ai-box">🤖 <b>Assistant</b><br><br>{answer}</div>',
        unsafe_allow_html=True
    )

    with st.expander("📚 Retrieved Context"):

        for i, doc in enumerate(docs,1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            if doc.metadata:
                st.caption(doc.metadata)
