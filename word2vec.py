from nltk.probability import FreqDist
import gensim.downloader as api
import pandas as pd
from nltk.corpus import stopwords
import string
import re
from collections import Counter


info = api.info()
model = api.load("glove-twitter-100")


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
    return data


def compare_texts_word2vec(file_one, file_two, k=10):
    print("k value = {}".format(k))

    # Fetching the most common words
    file_one_string = " ".join(file_one)
    file_two_string = " ".join(file_two)
    file_one_words = re.findall('\w+', file_one_string)
    most_common_words_one = []
    temp_most_common_words_one = Counter(file_one_words).most_common(k)
    for word, count in temp_most_common_words_one:
        most_common_words_one.append(word)

    print("Most common words in file one: {}".format(most_common_words_one))

    file_two_words = re.findall('\w+', file_two_string)
    most_common_words_two = []
    temp_most_common_words_two = Counter(file_two_words).most_common(k)
    for word, count in temp_most_common_words_two:
        most_common_words_two.append(word)

    print("Most common words in file two: {}".format(most_common_words_two))

    similarity_sum = 0
    for i in range(k):
        similarity_sum += model.similarity(most_common_words_one[i], most_common_words_two[i])
    average_similarity = similarity_sum/k
    print("Similarity between the two text files: {}".format(average_similarity))

    file_one_similar_words = []
    for word in most_common_words_one:
        similar_words = model.most_similar(positive=[word], topn=10)
        for similar, prob in similar_words:
            file_one_similar_words.append(similar)

    file_two_similar_words = []
    for word in most_common_words_two:
        similar_words = model.most_similar(positive=[word], topn=10)
        for similar, prob in similar_words:
            file_two_similar_words.append(similar)

    file_one_unique_words = list(set(file_one_similar_words))
    file_two_unique_words = list(set(file_two_similar_words))

    print("Unique words in file one {}".format(file_one_unique_words))
    print("Unique words in file two {}".format(file_two_unique_words))

    overlapping_words = list(set(file_one_unique_words).intersection(file_two_unique_words))
    print("Overlapping words: {}".format(overlapping_words))
    print("--------------------------------------------------------------------------------")


train_data = pd.read_csv("data/olid-training-v1.0.tsv", sep='\t')
train_data = preprocess(train_data)

non_offensive = train_data.loc[train_data['subtask_a'] == "NOT"]
non_offensive_tweets = non_offensive['tweet']

offensive = train_data.loc[train_data['subtask_a'] == "OFF"]
offensive_tweets = offensive['tweet']

compare_texts_word2vec(non_offensive_tweets, offensive_tweets, 5)
compare_texts_word2vec(non_offensive_tweets, offensive_tweets, 10)
compare_texts_word2vec(non_offensive_tweets, offensive_tweets, 20)

# find 10 common words, use pretrained word2vec to find 10 similar words to each of the top k words
# form two lists for file 1 and file 2 ->
# find overlapping etc for these lists


