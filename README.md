# J1

The purpose of this project is to provide child-rearing education solutions for new parents, specifically targeting preschool children who already exhibit autonomous behavior. Suitable advice will be offered through a chatbot.

The advice provided is sourced from streaming media and various collectable documents. Through RAG optimization, the recommendations are made more precise.

Initially starting with a chatbot, in addition to RAG accuracy optimization, we plan to incorporate features such as facial detection, image and voice recognition. Beyond providing advice, it can offer more direct solutions, such as playing content trained on the parents' voices or connecting to external IoT devices, achieving a more comprehensive solution.

## Features

Provide various tool pages for testing purposes first:

- [x] Chatbot by using OpenAI APIs
- [x] Chatbot by using Langchain
- [x] A tool page for saving TikTok video-to-text data to a vector database
- [x] Chatbot with a basic RAG strategy
- [x] Chatbot based on RAG-fusion strategy
- [x] Chatbot based on Decomposition strategy
- [x] Chatbot based on HyDE strategy

## Development

Make sure you've install pipenv first:

```console
$ pip install --user pipenv
```

Then:

```console
$ git clone https://github.com/vincent714/J1.git

$ cd J1

$ pipenv install

$ pipenv shell

$ streamlit run ./OpenAI_Bot.py
```
