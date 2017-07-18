from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk, re

news= fetch_20newsgroups(subset='all')
X,y=news.data,news.target
#nltk.download()
def news_to_sentences(news):
    news_text=BeautifulSoup(news).get_text()
    token=nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentence=token.tokenize(news_text)
    print(news_text)

for x in X:
    news_to_sentences(x)
