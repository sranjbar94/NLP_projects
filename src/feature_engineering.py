# feature extraction utilities

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf(train_text, test_text, max_features=5000):

    # initialize tfidf vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )

    # fit and transform training data
    X_train = vectorizer.fit_transform(train_text)

    # transform test data
    X_test = vectorizer.transform(test_text)

    return X_train, X_test, vectorizer