import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read file and clean dataset
df = pd.read_csv(".\\iris.csv")
df = df.drop("adapted from https://en.wikipedia.org/wiki/Iris_flower_data_set", axis=1)
df = df.drop([141, 142], axis=0)

# Encode irisname column of dataset
encode = LabelEncoder()
df["irisname"] = encode.fit_transform(df["irisname"])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    df.drop("irisname", axis=1), df["irisname"], test_size=0.2, random_state=0
)

# Create model
model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

# Print predictions and accuracy
print("The model predictions were: {}".format(encode.inverse_transform(predict)))
print("Accuracy Score: {}%".format(round(accuracy_score(y_test, predict) * 100, 2)))
