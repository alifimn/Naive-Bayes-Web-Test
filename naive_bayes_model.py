from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

class NaiveBayesClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        predictions = self.model.predict(X_vec)
        return predictions

    def save_model(self, filename="naive_bayes_model.joblib"):
        joblib.dump((self.vectorizer, self.model), filename)

    def load_model(self, filename="naive_bayes_model.joblib"):
        self.vectorizer, self.model = joblib.load(filename)
