# J1

I am a father of two preschool-aged daughters, and I love them dearly. I firmly believe that raising children should not rely on scolding or punishment, but rather on teaching them to manage their emotions. I believe it's okay for children to feel sad, cry, or get angry, but they should learn to express their feelings verbally once they calm down. With repeated practice, they will improve.

However, this is not easy. Sometimes they throw tantrums and cry for 30 minutes or even longer. There are times when I truly feel overwhelmed and end up getting angry and scolding them, but after scolding, I regret getting angry at them. In these moments, I understand that some parents, under greater pressure, might eventually lose control and resort to hitting their children. In severe cases, this can turn into child abuse.

I believe every parent loves their children deeply, but during their child's tantrums, their own anxiety makes them feel helpless. This system provides parents with a chance to catch their breath. It is an assistant for parents, using AI to offer specific methods to help parents soothe their children and manage their own emotions.

The advice provided is sourced from streaming media and various collectible documents. Through RAG optimization, the recommendations are made more precise.

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
