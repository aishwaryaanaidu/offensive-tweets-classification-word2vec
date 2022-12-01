import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import string

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

info = api.info()
model = api.load("glove-twitter-25")


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


def make_feature_vectors(words, model, num_features):
    feature_vectors = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vectors = np.add(feature_vectors, model[word])
    if nwords == 0:
        nwords = 1
    feature_vectors = np.divide(feature_vectors, nwords)
    return feature_vectors


def get_average_feature_vectors(reviews, model, num_features):
    review_feature_vectors = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0
    for review in reviews:
        review_feature_vectors[counter] = make_feature_vectors(review, model, num_features)
        counter = counter + 1
    return review_feature_vectors


def train_MLP_model(path_to_train_file, num_layers=2):

    train_data = pd.read_csv(path_to_train_file, sep='\t')
    sentences_train = preprocess(train_data)
    train_matrix = get_average_feature_vectors(sentences_train, model, 25)
    classes = [0, 1]
    y_train = train_data['subtask_a']

    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(num_layers, num_layers,), random_state=1)

    return classifier, train_matrix, y_train, classes


def test_MLP_model(path_to_test_file, mlp_model):
    test_data = pd.read_csv(path_to_test_file, sep='\t')
    sentences_test = preprocess(test_data)
    test_matrix = get_average_feature_vectors(sentences_test, model, 25)
    return test_matrix


actual_labels = pd.read_csv("data/labels-levela.csv", header=None)
actual_labels = actual_labels.iloc[:, 1]
actual_labels = actual_labels.factorize()[0]

# MLP model with 2 layers
print("MLP model with 2 layers")
mlp_model, train_matrix, y_train, classes = train_MLP_model("data/olid-training-v1.0.tsv", 2)

test_matrix = test_MLP_model("data/testset-levela.tsv", mlp_model)
mlp_model.partial_fit(train_matrix, y_train, classes)
predictions = mlp_model.predict(test_matrix)
score = accuracy_score(actual_labels, predictions)
print("Accuracy: {}".format(score))

print("-------------------------------------------------------------------")

print("MLP model with 1 layers")
mlp_model, train_matrix, y_train, classes = train_MLP_model("data/olid-training-v1.0.tsv", 1)

test_matrix = test_MLP_model("data/testset-levela.tsv", mlp_model)
mlp_model.partial_fit(train_matrix, y_train, classes)
predictions = mlp_model.predict(test_matrix)
score = accuracy_score(actual_labels, predictions)
print("Accuracy: {}".format(score))

print("-------------------------------------------------------------------")

print("MLP model with 3 layers")
mlp_model, train_matrix, y_train, classes = train_MLP_model("data/olid-training-v1.0.tsv", 3)

test_matrix = test_MLP_model("data/testset-levela.tsv", mlp_model)
mlp_model.partial_fit(train_matrix, y_train, classes)
predictions = mlp_model.predict(test_matrix)
score = accuracy_score(actual_labels, predictions)
print("Accuracy: {}".format(score))

