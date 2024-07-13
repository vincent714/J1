import streamlit as st
import pyktok as pyk
import requests
import whisper
import os


if "text" not in st.session_state:
    st.session_state.text = ""

tiktok_form = st.form(key="tiktok_form", clear_on_submit=True)
url = tiktok_form.text_input(
    "TikTok Video URL", "https://www.tiktok.com/@meettalent/video/7360299225353096453"
)
whisper_model_size = tiktok_form.selectbox(
    "Whisper Model Size (Large means Multilingual model)",
    ("tiny", "base", "small", "medium", "large"),
    index=1,
)
del_downloaded_mp3 = tiktok_form.checkbox("Delete downloaded mp3", value=True)
submit = tiktok_form.form_submit_button("Get the Video Text")


def is_tiktok_url(url):
    return "tiktok.com" in url


if submit and not is_tiktok_url(url):
    st.error("Please enter a TikTok URL.")
    st.stop()

if submit:
    st.write(f"URL: {url}")
    st.write(f"Whisper Model Size: {whisper_model_size}")
    st.write(f"Delete downloaded mp3: {del_downloaded_mp3}")

    with st.spinner("Processing..."):
        tt_json = pyk.alt_get_tiktok_json(url)
        # st.write(tt_json)
        mp3_url = tt_json["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"][
            "itemStruct"
        ]["music"]["playUrl"]
        mp3 = requests.get(mp3_url)

        with open("test.mp3", "wb") as f:
            f.write(mp3.content)
        model = whisper.load_model(whisper_model_size)
        result = model.transcribe("test.mp3")
        st.session_state.text = result["text"]

if st.session_state.text != "":
    modify_text_form = st.form(key="modify_text_form")
    modified_text = modify_text_form.text_area(
        "Modified Text", st.session_state.text, height=30
    )
    submit_modified_text = modify_text_form.form_submit_button("Submit Modified Text")

    if submit_modified_text:
        st.write(modified_text)

# if test.mp3 exists, remove it (from the root directory)
if os.path.exists("test.mp3") and del_downloaded_mp3:
    os.remove("test.mp3")
