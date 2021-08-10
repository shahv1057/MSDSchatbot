import random
import json
import joblib
import torch
import numpy as np
from model import NeuralNet
import nltk
from nltk.stem.porter import PorterStemmer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

with open('data/msds_chat_data.json', 'r') as json_data:
    chatbot_json = json.load(json_data)

FILE = "data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Vatie"
print("Welcome to the MSDS chatbot system, I'm Vatie, how can I help? (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    #preprocess user response
    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # predict category with multi class classification model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If highest probability > .75, randomly select message from inputted options
    if prob.item() > 0.75:
        for categ in chatbot_json['chatbot_data']:
            if tag == categ["tag"]:
                print(f"{bot_name}: {random.choice(categ['responses'])}")
    # If none of the probs > .75, say "I do not understand"
    else:
        print(f"{bot_name}: I do not understand...")
        
