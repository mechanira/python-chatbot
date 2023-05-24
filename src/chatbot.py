import random
import json
import datetime
import requests
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('src/intents.json').read())

words = pickle.load(open('src/models/words.pkl', 'rb'))
classes = pickle.load(open('src/models/classes.pkl', 'rb'))
model = load_model('src/models/chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.0
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    if not return_list:
        return_list.append({'intent': 'fallback', 'probability': '1.0'})

    return return_list


def time(response):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    response = response.replace("{time}", current_time)
    return response

def joke(response):
    resp = requests.get('https://official-joke-api.appspot.com/random_joke')
    if resp.status_code == 200:
        joke_data = resp.json()
        return f"{joke_data['setup']} {joke_data['punchline']}"


intent_mapping = {
    "time": time,
    "joke": joke
}


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            handler = intent_mapping.get(tag)  # Get the corresponding function handler from the mapping
            if handler is not None:
                result = handler(result)  # Pass the response to the function for further processing
            break
    return result

print("Bot is running")
DEBUG_MODE = True


while True:
    message = input("\nYou: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot: " + res)

    if DEBUG_MODE:
        probability = ints[0]['probability']
        print(f"DEBUG: From intent '{ints[0]['intent']}' with a probablity of {float(probability) * 100} %")