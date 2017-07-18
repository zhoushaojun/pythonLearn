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
    sentences= []
    for sen in raw_sentence:
        sentences.append(re.sub('[^a-zA-Z]]','',sen.lower().strip()).split())
    return  sentences

sentences=[]
for x in X:
    sentences +=news_to_sentences(x)

from gensim.models import word2vec

num_feature=300

min_word_count=20

num_worker=2

context=5
downsampling=1e-3

model=word2vec.Word2Vec(sentences,sg=1,workers=num_worker, size=num_feature, min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)

print(model['morning'])
