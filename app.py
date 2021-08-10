import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
import joblib
import torch
from model import NeuralNet

# nltk.download('punkt')

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array for each sentence given a vocabulary set
    """
    # stem each word
    stemmer = PorterStemmer()   
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
            
    return bag

bot_name = 'Vatie'

# load chat data and trained model
with open('data/msds_chat_data.json', 'r') as json_data:
    chatbot_json = json.load(json_data)
    
FILE = "data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data['all_words']
classes = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    #checks is a user has given a name, in order to give a personalized feedback
    return predict_class(msg, model)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # predict category with multi class classification model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = classes[predicted.item()]    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If highest probability > .75, randomly select message from inputted options
    if prob.item() > 0.75:
        for categ in chatbot_json['chatbot_data']:
            if tag == categ["tag"]:
                return(f"{bot_name}: {random.choice(categ['responses'])}")
            
    # If none of the probs > .75, say standard "I do not understand" response
    else:
        return f"{bot_name}: I do not understand... Maybe try the MSDS <a href=\"https://www.usfca.edu/arts-sciences/graduate-programs/data-science\">website</a>?"


if __name__ == "__main__":
    app.run()
