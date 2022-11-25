import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import string

from sklearn.neural_network import MLPClassifier

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


def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index_to_key)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0
    # Loop through the reviews
    for review in reviews:
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def train_MLP_model(path_to_train_file, num_layers=2):

    train_data = pd.read_csv(path_to_train_file, sep='\t')
    sentences_train = preprocess(train_data)
    f_matrix_train = getAvgFeatureVecs(sentences_train, model, 25)
    print(f_matrix_train)
    classes = [0, 1]
    y_train = train_data['subtask_a']

    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(30, 30,), random_state=1)

    return classifier, f_matrix_train, y_train, classes


def test_MLP_model(path_to_test_file, mlp_model):
    test_data = pd.read_csv(path_to_test_file, sep='\t')
    sentences_test = preprocess(test_data)
    f_matrix_test = getAvgFeatureVecs(sentences_test, model, 25)
    return f_matrix_test


mlp_model, f_matrix_train, y_train, classes = train_MLP_model("data/olid-training-v1.0.tsv", 2)

f_matrix_test = test_MLP_model("data/testset-levela.tsv", mlp_model)
mlp_model.partial_fit(f_matrix_train, y_train, classes)
mlp_model.predict(f_matrix_test)