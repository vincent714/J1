import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

if st.secrets["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


st.title("HyDE Chatbot")
st.write(
    "You can only ask questions related to child education; for other questions, the AI will respond by saying it doesn't know."
)

# Get an OpenAI API key before starting
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Initialize chat history
if "messages_4" not in st.session_state:
    st.session_state.messages_4 = []

# Display chat messages from history on app rerun
for message in st.session_state.messages_4:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

vectorstore = Chroma(
    persist_directory="chroma.db",
    collection_name="child_edu",
    embedding_function=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

# HyDE document genrdocsgetget_retrieval_docs
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = prompt_hyde | llm | StrOutputParser()

retrieval_chain = generate_docs_for_retrieval | retriever

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = prompt | llm | StrOutputParser()

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    st.session_state.messages_4.append({"role": "user", "content": prompt})

    retireved_docs = retrieval_chain.invoke({"question": prompt})
    results = final_rag_chain.invoke({"context": retireved_docs, "question": prompt})
    st.chat_message("bot").write(results)
    st.session_state.messages_4.append({"role": "ai", "content": results})
