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


st.title("RAG-fusion Chatbot")
st.write(
    "You can only ask questions related to child education; for other questions, the AI will respond by saying it doesn't know."
)

# Get an OpenAI API key before starting
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Initialize chat history
if "messages_2" not in st.session_state:
    st.session_state.messages_2 = []

# Display chat messages from history on app rerun
for message in st.session_state.messages_2:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


vectorstore = Chroma(
    persist_directory="chroma.db",
    collection_name="child_edu",
    embedding_function=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

# RAG-Fusion: generate related multiple search queries based on a single input query
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}

The language of the question must match the input question. For example, if the question is asked in Chinese, 
the answer should also be in Chinese. \n
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)


# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    st.session_state.messages_2.append({"role": "user", "content": prompt})

    response = final_rag_chain.invoke({"question": prompt})
    st.chat_message("ai").write(response)
    st.session_state.messages_2.append({"role": "ai", "content": response})
