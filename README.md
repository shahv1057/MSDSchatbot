# PyTorch Chatbot for the USF Master's in Data Science program

We built a chatbot that we tuned to apply to MSDS based questions for current students, prospective students, and just about anyone interested in the details about the 12-month accelerated Data Science program. This bot is trained with a [3-Layer PyTorch Neural Network with ReLU activation](model.py). The bot is then surfaced on a Flask web application for interactive usage. Ultimately, we hope to integrate our bot as an automated tool on the USF MSDS website (particularly the FAQ page) so visitors can get a chatty alternative to answering questions on topics like courses, professors, tuition, practicum, and more about the MSDS program!

Keep reading to train, develop, and surface our chatbot locally!

## Installation

### Clone repository, create an env, and install reqs

To develop the MSDS Chatbot, please clone our repository, create a virtual environment, and download the necessary packages listed in our [requirements.txt](requirements.txt) file. The code necessary to do so is provided below:

```
$ git clone https://github.com/shahv1057/MSDSchatbot.git
$ python -m venv chatbot-env
$ source chatbot-env/bin/activate
$ python -m pip install -r requirements.txt
```

For installation of PyTorch see [official website](https://pytorch.org/).

You also need to download the `nltk` punkt package:
```
$ pip install nltk
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage

### 1) Train the model
  - Run `python train.py` in your terminal

### 2a) Flask Web Application Chatbot
  - Run `python app.py` in your terminal
  - Open http://127.0.0.1:5000 in your favorite browser!

### 2b) Terminal Chatbot
  - Run `python chat.py` in your terminal
  - Enter your messages and recieve answers directly in your terminal



## The Data 

The data used to train this model was hand curated by Veeral and Catie. This inlcuded coming up with potential question categories, questions for the categories and responses that the chatbot would return. The dataset is included in the repository.

<img width="650" alt="data" src="https://user-images.githubusercontent.com/67168388/128942045-bc00d0a1-1585-420e-9d09-5ffca62872ab.png">

## How it Works

Here is a quick summary of the organization of this repository, what different scripts and files do, and how it comes together to create the MSDS chatbot:

- [train.py](train.py) creates a stemmed and lowercased Bag-of-Words matrix representation using our [chatbot data](data/msds_chat_data.json). It then uses that data matrix and corresponding category labels from the data to train the 3-layer PyTorch Neural Network model from [model.py](model.py). The model parameters and details are then saved [here](data/data.pth).
- [app.py](app.py) creates a Flask app that inputs user questions, preproccesses them the same way as [train.py](train.py), and then predicts class of response using the [saved model](data/data.pth). The app chatbot then responds to the user with one of the responses indicated for that category in the data.
- [chat.py](chat.py) opens a Terminal chatbot that works the same way as the Flask app, just in the Terminal


