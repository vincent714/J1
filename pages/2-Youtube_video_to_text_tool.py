import streamlit as st
import pyktok as pyk
import requests
import whisper
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

# TODO: https://github.com/JuanBindez/pytubefix  改用這個
from pytubefix import YouTube
from pytubefix.cli import on_progress

st.set_page_config(
    page_title=f"Youtube Video Indexer",
    page_icon=":memo:",
    initial_sidebar_state="expanded",
    layout="wide",
)

# "st.session_state object", st.session_state

if "text" not in st.session_state:
    st.session_state.text = ""

if "show_modify_form" not in st.session_state:
    st.session_state.show_modify_form = False

if "youtube_id" not in st.session_state:
    st.session_state.youtube_id = None


###### Get the Youtube video url ########
youtube_form = st.form(key="youtube_form", clear_on_submit=True)
youtube_url = youtube_form.text_input(
    "Youtube Video URL",
    "https://www.youtube.com/watch?v=XdTgb7r_8v0",
    key="youtube_url",
)
whisper_model_size = youtube_form.selectbox(
    "Whisper Model Size (Large means Multilingual model)",
    ("tiny", "base", "small", "medium", "large"),
    index=4,
    key="whisper_model_size",
)
del_downloaded_mp3 = youtube_form.checkbox(
    "Delete downloaded mp3", value=True, key="del_downloaded_mp3"
)
submit = youtube_form.form_submit_button("Get the Video Text")


###### Get the text from the youtube video ########
if submit:
    youtube_url = st.session_state.youtube_url
    whisper_model_size = st.session_state.whisper_model_size
    del_downloaded_mp3 = st.session_state.del_downloaded_mp3

    if not "youtube.com" in youtube_url:
        st.error("Please enter a valid Youtube video URL")
        st.stop()

    st.write("url: ", youtube_url)
    st.write("whisper_model_size: ", whisper_model_size)
    st.write("delete downloaded mp3: ", del_downloaded_mp3)

    # get the youtube json, then get the mp3 url and download it
    # use whisper api to transcribe the mp3 to text, and save text to the file
    with st.status("Convert Audio to Text ...", expanded=True):

        st.write("Download the mp3 file...")
        yt = YouTube(youtube_url, on_progress_callback=on_progress)

        st.write(yt.video_id)
        st.session_state.youtube_id = yt.video_id

        yt.streams.get_audio_only().download(
            output_path=".", filename=yt.video_id, mp3=True
        )

        st.write("Transcribe the mp3 file to text...")
        model = whisper.load_model(whisper_model_size)
        result = model.transcribe(f"{yt.video_id}.mp3")
        st.session_state.text = result["text"]

        # if the mp3 file exists, remove it
        st.write("Remove the mp3 file...")
        if os.path.exists(f"{yt.video_id}.mp3") and del_downloaded_mp3:
            os.remove(f"{yt.video_id}.mp3")

        st.session_state.show_modify_form = True

if st.session_state.show_modify_form:
    # st.write(st.session_state.text)
    modify_text_form = st.form(key="modify_text_form")
    modified_text = modify_text_form.text_area(
        "Modified Text", st.session_state.text, height=30
    )
    submit_modified_text = modify_text_form.form_submit_button("Submit Modified Text")

    if submit_modified_text:
        youtube_id = st.session_state.youtube_id

        persistent_client = chromadb.PersistentClient(path="chroma.db")

        langchain_chroma = Chroma(
            client=persistent_client,
            collection_name="child_edu",
            embedding_function=OpenAIEmbeddings(),
        )

        with st.status("Saving modified text to the vectorstore...", expanded=True):
            st.write("Split the text...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            texts = text_splitter.split_text(modified_text)

            # create metadatas
            st.write("Create metadata...")
            metadata = {
                "source": "youtube",
                "video_id": youtube_id,
            }
            # create a metadata array with the same length as the text.
            metadatas = [metadata] * len(texts)

            st.write("Embed the text...")
            st.write("Save embeddings to the Chroma database...")
            # save to chroma database
            langchain_chroma.add_texts(texts, metadatas=metadatas)

            # clear the session
            st.write("Clear the session state...")
            for key in st.session_state.keys():
                del st.session_state[key]
