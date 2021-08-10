# PyTorch MSDS Program Chatbox
  
## Installation

### Clone repository, create an env, and install reqs
i.e.
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


