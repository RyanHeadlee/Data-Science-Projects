import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Load Dataset
df = pd.read_csv(".\\spam.csv", encoding="ISO-8859-1")

# Split into Training and Test data
x = df.v2  # Email Text
y = df.v1  # Whether spam or not
x_train, y_train = x[:4457], y[:4457]  # Train on about 80% of data
x_test, y_test = x[4457:], y[4457:]  # Test on about 20% of data

# Extract Features
count_vector = CountVectorizer()
features = count_vector.fit_transform(x_train)

# Build a model
tuned_params = {
    "kernel": ["rbf"],
    "gamma": [1e-3],
    "C": [100],
}
model = GridSearchCV(svm.SVC(), tuned_params)
model.fit(features, y_train)

# Test accuracy
features_test = count_vector.transform(x_test)
acc_score = model.score(features_test, y_test)

print("Accuracy of the model: {}%".format(round(acc_score * 100, 2)))
