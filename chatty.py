# Let's Python

##### Import necessary libraries and frameworks

import numpy as np
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import string
import nltk
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Download the DialoGPT model and tokenizer
checkpoint = "microsoft/DialoGPT-medium"
# Initialize the tokenizer with the padding_side argument set to 'left'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
# Download and cache the pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Download necessary libraries and frameworks
nltk.download('stopwords')
nltk.download('wordnet')

# Create a lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()


# Remove punctuation and lowercase all text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Tokenize and lemmatize text
def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


# Remove stop words from text
def remove_stop_words(text):
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


# Custom tokenizer that preprocesses and lemmatizes the text
def custom_tokenize(text):
    text = preprocess(text)
    lemmatized_tokens = lemmatize(text)
    filtered_tokens = remove_stop_words(text)
    return filtered_tokens


# Create a list of stop words
stop_words = nltk.corpus.stopwords.words('english')

# Create a dictionary to remove punctuation
remove_punct_dict = {ord(punct): None for punct in string.punctuation}


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(remove_punct_dict)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens into a single string
    text = ' '.join(tokens)
    return text


# Download Microsoft's DialoGPT model and tokenizer

##### The Hugging Face checkpoint for the model and its tokenizer is `"microsoft/DialoGPT-medium"`

# A ChatBot class

# Build a ChatBot class with all necessary modules to make a complete conversation
class ChatBot():
    # initialize
    def __init__(self):
        # once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # a flag to check whether to end the conversation
        self.end_chat = False
        # greet while starting
        self.welcome()

    def welcome(self):
        print("Initializing ChatBot ...")
        # some time to get user ready
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("ChatBot >>  " + greeting)

    def user_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print('ChatBot >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ChatBot ...')
        else:
            # continue chat, preprocess input text
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, \
                                                       return_tensors='pt')

    def bot_response(self):
        # append the new user input tokens to the chat history
        # if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
        else:
            # if first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids

        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, \
                                               pad_token_id=tokenizer.eos_token_id)

        # last ouput tokens from bot
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], \
                                    skip_special_tokens=True)
        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        print('ChatBot >>  ' + response)

    # in case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                                    skip_special_tokens=True)
        # iterate over history backwards to find the last token
        while response == '':
            i = i - 1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                                        skip_special_tokens=True)
        # if it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know",
                                      "I am not sure"])
        # not a question? answer suitably
        else:
            reply = np.random.choice(["Great",
                                      "Fine. What's up?",
                                      "Okay"
                                      ])
        return reply


def bot_response(self):
    # append the new user input tokens to the chat history
    # if chat has already begun
    if self.chat_history_ids is not None:
        self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
    else:
        # if first entry, initialize bot_input_ids
        self.bot_input_ids = self.new_user_input_ids

    # generate a response while limiting the total chat history to 1000 tokens,
    response_ids = model.generate(self.bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # last output tokens from bot
    response = tokenizer.decode(response_ids[:, self.bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    # in case, bot fails to answer
    if response == "":
        response = self.random_response()
    # print bot response
    print('ChatBot >>  ' + response)

    # update chat history to include the latest response
    self.chat_history_ids = response_ids

    # check if the latest response is the same as the previous one
    if self.previous_response == response:
        # if it is, generate a new response
        self.bot_response()
    else:
        # otherwise, update the previous response to the current one
        self.previous_response = response


# Happy Chatting!

# **Response Generation**

def response(user_response):
    sent_tokens.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer=custom_tokenize)
    tfidf = tfidfvec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.remove(user_response)
    if req_tfidf == 0:
        return "I'm sorry, I don't understand you."
    else:
        return sent_tokens[idx]


# **Testing**

f = open("dataset.txt", "r", errors='ignore')
raw_doc = f.read()
raw_doc = preprocess(raw_doc)

# build a ChatBot object
bot = ChatBot()
# start chatting
while True:
    # receive user input
    bot.user_input()
    # check whether to end chat
    if bot.end_chat:
        break
    # output bot response
    bot.bot_response()