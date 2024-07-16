import streamlit as st
import pyktok as pyk
import requests
import whisper
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


st.set_page_config(
    page_title=f"Tiktok Video Indexer",
    page_icon=":memo:",
    initial_sidebar_state="expanded",
    layout="wide",
)


"st.session_state object", st.session_state


if "text" not in st.session_state:
    st.session_state.text = ""

if "show_modify_form" not in st.session_state:
    st.session_state.show_modify_form = False

if "tiktok_id" not in st.session_state:
    st.session_state.tiktok_id = None


###### Get the tiktok video url ########
tiktok_form = st.form(key="tiktok_form", clear_on_submit=True)
tiktok_url = tiktok_form.text_input(
    "TikTok Video URL",
    "https://www.tiktok.com/@meettalent/video/7360299225353096453",
    key="tiktok_url",
)
whisper_model_size = tiktok_form.selectbox(
    "Whisper Model Size (Large means Multilingual model)",
    ("tiny", "base", "small", "medium", "large"),
    index=1,
    key="whisper_model_size",
)
del_downloaded_mp3 = tiktok_form.checkbox(
    "Delete downloaded mp3", value=True, key="del_downloaded_mp3"
)
submit = tiktok_form.form_submit_button("Get the Video Text")


###### Get the text from the tiktok video ########
if submit:
    tiktok_url = st.session_state.tiktok_url
    whisper_model_size = st.session_state.whisper_model_size
    del_downloaded_mp3 = st.session_state.del_downloaded_mp3

    if not "tiktok.com" in tiktok_url:
        st.error("Please enter a valid TikTok video URL")
        st.stop()

    st.write("url: ", tiktok_url)
    st.write("whisper_model_size: ", whisper_model_size)
    st.write("delete downloaded mp3: ", del_downloaded_mp3)

    # get the tiktok json, then get the mp3 url and download it
    # use whisper api to transcribe the mp3 to text, and save text to the file
    with st.status("Convert Audio to Text ...", expanded=True):
        st.write("Get the video metadata...")
        tt_json = pyk.alt_get_tiktok_json(tiktok_url)
        mp3_url = tt_json["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"][
            "itemStruct"
        ]["music"]["playUrl"]

        st.write("Download the mp3 file...")
        mp3 = requests.get(mp3_url)

        # tiktok_id as a temp file name
        tiktok_id = tt_json["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"][
            "itemStruct"
        ]["id"]
        st.session_state.tiktok_id = tiktok_id

        # save the mp3 file (using the tiktok_id as the file name)
        with open(f"{tiktok_id}.mp3", "wb") as f:
            f.write(mp3.content)

        st.write("Transcribe the mp3 file to text...")
        model = whisper.load_model(whisper_model_size)
        result = model.transcribe(f"{tiktok_id}.mp3")
        st.session_state.text = result["text"]

        # if the mp3 file exists, remove it
        st.write("Remove the mp3 file...")
        if os.path.exists(f"{tiktok_id}.mp3") and del_downloaded_mp3:
            os.remove(f"{tiktok_id}.mp3")

        st.session_state.show_modify_form = True

if st.session_state.show_modify_form:
    modify_text_form = st.form(key="modify_text_form")
    modified_text = modify_text_form.text_area(
        "Modified Text", st.session_state.text, height=30
    )
    submit_modified_text = modify_text_form.form_submit_button("Submit Modified Text")

    if submit_modified_text:
        tiktok_id = st.session_state.tiktok_id

        with st.status("Saving modified text to the vectorstore...", expanded=True):
            st.write(f"Saved modified text to {tiktok_id}.txt...")
            # save text to file (using utf-8 encoding)
            with open(f"{tiktok_id}.txt", "w", encoding="utf-8") as file:
                file.write(modified_text)

            st.write("Load the text...")
            loader = TextLoader(f"{tiktok_id}.txt")
            documents = loader.load()

            st.write("Split the text...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)

            st.write("Embed the text...")
            st.write("Save embeddings to the Chroma database...")
            # save to chroma database
            Chroma.from_documents(
                docs,
                collection_name="child_edu",
                embedding=OpenAIEmbeddings(),
                persist_directory="chroma.db",
            )

            # if modified_text.txt exists, remove it (from the root directory)
            st.write("Remove the modified text file...")
            if os.path.exists(f"{tiktok_id}.txt") and submit_modified_text:
                os.remove(f"{tiktok_id}.txt")

            # clear the session
            st.write("Clear the session state...")
            for key in st.session_state.keys():
                del st.session_state[key]

            st.session_state
