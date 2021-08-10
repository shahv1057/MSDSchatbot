# PyTorch Chatbot for the USF Master's in Data Science program

We built a chatbot that we tuned to apply to MSDS based questions for current students, prospective students, and just about anyone interested in the details about the 12-month accelerated Data Science program. This bot is trained with a 3-Layer PyTorch Neural Network with ReLU activation. The bot is then surfaced on a Flask web application for interactive usage. Ultimately, we hope to integrate our bot as an automated tool on the USF MSDS website (particularly the FAQ page) so visitors can get a chatty alternative to answering questions on topics like courses, professors, tuition, practicum, and more about the MSDS program!

Keep reading to train, develop, and surface our chatbot locally!

## Installation

### Clone repository, create an env, and install reqs

To develop the MSDS Chatbot, please clone our repository, create a virtual environment, and download the necessary packages listed in our [requirements.txt](requirements.txt) file. The code necessary to do so is provided below:

```
$ git clone https://github.com/shahv1057/MSDSchatbot.git
$ python -m venv chatbot-env
$ source tutorial-env/bin/activate
$ python -m pip install -r requirements.txt
```

For installation of PyTorch see [official website](https://pytorch.org/).

You also need to download the `nltk` punkt package:
```
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage

### 1) Train the model
  - Run `python code/train.py` in your terminal

### 2a) Flask Web Application Chatbot
  - Run `python app.py` in your terminal
  - Open http://127.0.0.1:5000 in your favorite browser!

### 2b) Terminal Chatbot
  - Run `python chat.py` in your terminal
  - Enter your messages and recieve answers directly in your terminal


