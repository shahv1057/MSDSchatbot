import numpy as np
import random
import json
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

import nltk
from nltk.stem.porter import PorterStemmer

with open('data/msds_chat_data.json', 'r') as json_data:
    chatbot_json = json.load(json_data)
    
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

all_words = []
tags = []
xypairs = []
# loop through each category 
for category in chatbot_json['chatbot_data']:
    tag = category['tag']
    # add to tag list
    tags.append(tag)
    for pattern in category['patterns']:
        # tokenize each word in the sentence and then add to words list
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        # add to xy pair
        xypairs.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [PorterStemmer().stem(w.lower()) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xypairs), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xypairs:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

chat_ds = ChatDataset()
train_loader = DataLoader(dataset=chat_ds,
                          batch_size=batch_size,
                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = lossFunc(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data/data.pth"
torch.save(data, FILE)
joblib.dump(model, 'code/model')

print(f'Completed training chatbot model. File saved to {FILE}')


