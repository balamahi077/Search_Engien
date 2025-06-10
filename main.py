import streamlit as st
import openai
import google.generativeai as genai
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pickle

st.set_page_config(page_title="🔍 GenAI Search Engine", layout="wide")

st.title("🌐 GenAI Search Engine")
st.subheader(f"Any API_KEY can give and ask a quesion...")


openai_api_key = st.text_input("🔑 Enter your OpenAI API Key (optional)", type="password")
gemini_api_key = st.text_input("🔑 Enter your Gemini API Key (optional)", type="password")

# --- Session State for Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Functions ---
def load_docs():
    raw_text = """
    OpenAI is a leading AI research lab.
    Gemini (by Google DeepMind) is a state-of-the-art AI model for reasoning.
    Vector search enables semantic retrieval using embeddings.
    """
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in chunks]

def build_vector_index(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = FAISS.from_documents(docs, embeddings)
    return index

def search_index(query, index):
    return index.similarity_search(query, k=3)

def ask_openai(query, context=""):
    client = openai(api_key=openai_api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ] + st.session_state.chat_history + [
        {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model=" gpt-3.5-turbo",
        messages=messages
    )
    reply = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    return reply

def ask_gemini(query):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use the appropriate model name

    # Prepare chat history in the expected format
    history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history.append({"role": "user", "parts": [msg["content"]]})
        else:
            history.append({"role": "model", "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(query)

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": response.text})

    return response.text



def choose_engine(query):
    keywords_live = ["today", "now", "latest", "current", "weather", "news", "stock", "price", "match", "headline", "event"]
    if any(k in query.lower() for k in keywords_live):
        return "gemini" if gemini_api_key else "openai"
    return "openai" if openai_api_key else "gemini"

# --- Main Logic ---
query = st.text_input("💬 Ask your question (like a Google search)...")

if query:
    engine = choose_engine(query)
    st.subheader(f"⚙️ Selected Engine: {'Gemini 🔮' if engine == 'gemini' else 'OpenAI 🧠'}")

    try:
        if engine == "gemini" and gemini_api_key:
            gemini_answer = ask_gemini(query)
            st.write(gemini_answer)
        elif engine == "openai" and openai_api_key:
            if "vector_index" not in st.session_state:
                docs = load_docs()
                index = build_vector_index(docs)
                st.session_state.vector_index = index
            results = search_index(query, st.session_state.vector_index)
            context = "\n\n".join([doc.page_content for doc in results])
            openai_answer = ask_openai(query, context)
            st.write(openai_answer)
            st.subheader("📎 Sources")
            for i, doc in enumerate(results):
                st.code(doc.page_content, language="text")
        else:
            st.warning("⚠️ Please provide a valid API key for the selected engine.")

        with st.expander("🧠 Chat History"):
            for msg in st.session_state.chat_history:
                role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg['content']}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("🔎 Enter a search-style question above to get started!")
