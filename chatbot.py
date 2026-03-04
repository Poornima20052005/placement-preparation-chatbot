import json
import pickle
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

with open(os.path.join(BASE_DIR, "intents.json")) as file:
    intents = json.load(file)

def get_response(user_input):
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]

    print("Predicted Tag:", prediction)

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."