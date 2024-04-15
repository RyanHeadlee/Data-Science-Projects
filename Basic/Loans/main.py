import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv(".\\loan_data.csv")

# Encode Loan_Status label
encode = LabelEncoder()
df["Loan_Status"] = encode.fit_transform(df["Loan_Status"])

# Drop NULL values
df.dropna(how="any", inplace=True)

# Split data into x train and test and y train and test
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(["Loan_ID", "Loan_Status"], axis=1),
    df["Loan_Status"],
    test_size=0.2,
    random_state=14,
)

# Encode Data
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Create the model
model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

# Print the results
print("The predicted values were: {}".format(predict))
print("The accuracy was: {}%".format(round(accuracy_score(y_test, predict) * 100, 2)))
