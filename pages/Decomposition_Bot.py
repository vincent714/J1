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


st.title("Decomposition Chatbot")
st.write(
    "You can only ask questions related to child education; for other questions, the AI will respond by saying it doesn't know."
)

# Get an OpenAI API key before starting
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


vectorstore = Chroma(
    persist_directory="chroma.db",
    collection_name="child_edu",
    embedding_function=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)


def decompose(question: str):
    """Decompose a question into multiple sub-questions"""
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    The language of the sub-questions must match the input question. For example, if the question is asked in Chinese, the sub-questions should also be in Chinese. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""

    prompt_decomposition = ChatPromptTemplate.from_template(template)

    generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    return generate_queries_decomposition.invoke({"question": question})


# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def get_final_answer(questions: list[str]):
    """Get the final answer from the decomposed questions"""

    answer = ""
    q_a_pairs = ""
    for q in questions:
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs"),
            }
            | decomposition_prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

    return answer


# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    questions = decompose(prompt)
    st.chat_message("ai").write(questions)
    answer = get_final_answer(questions)
    st.chat_message("ai").write(answer)
    # st.chat_message("ai").write(response)
