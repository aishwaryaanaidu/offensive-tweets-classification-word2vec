from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
import string
import pandas as pd


def cleaning_stopwords(text, stop_words):
    return " ".join([word for word in str(text).split() if word not in stop_words])


def cleaning_punctuations(text):
    punctuations_list = string.punctuation
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


def tokenization(data):
    return data.split()


def lemmatizer_on_text(data, lemmatizer):
    text = [lemmatizer.lemmatize(word) for word in data]
    return text


def preprocess(data):
    # Remove mentions (@USER)
    data.loc[:, 'tweet'] = data.tweet.str.replace('@USER', '')
    # Converting the text to lowercase
    data['tweet'] = data['tweet'].str.lower()

    # Remove stop words from each row
    stop_words = set(stopwords.words('english'))
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_stopwords(text, stop_words))

    # Removing all the punctuations
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_punctuations(text))

    # Cleaning numbers
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_numbers(text))

    # Removing emojis
    data.loc[:, 'tweet'] = data.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )

    # Tokenization
    data['tweet'] = data['tweet'].apply(lambda text: tokenization(text))

    return data


train_data = pd.read_csv("data/olid-training-v1.0.tsv", sep='\t')
# Pre-process the data
sentences_train = preprocess(train_data)
# Training a word2vec model
model = Word2Vec(sentences_train, min_count=1)
print(model)
