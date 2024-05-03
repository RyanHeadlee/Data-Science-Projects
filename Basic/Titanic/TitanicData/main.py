import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)

# Read the data
df_train = pd.read_csv(".\\train.csv")
df_test = pd.read_csv(".\\test.csv")
combine = [df_train, df_test]

# print(df_train.describe(include=["O"]))  # include=["O"] describes categorical features
# print(df_train.describe())

# # Histogram for every Pclass and whether they survived based on Age
# g = sns.FacetGrid(df_train, col="Survived", row="Pclass", aspect=1.6)
# g.map(plt.hist, "Age", alpha=0.5, bins=20)
# g.add_legend()
# plt.show()

# # Point plot based on Pclass and Survived for both sexes and every embarked
# g = sns.FacetGrid(df_train, row="Embarked", aspect=1.6)
# g.map(
#     sns.pointplot,
#     "Pclass",
#     "Survived",
#     "Sex",
#     palette="deep",
#     order=[1, 2, 3],
#     hue_order=["male", "female"],
# )
# g.add_legend()
# plt.show()

# # Bar plot based on Sex and Fare for every Embarked and whether they survived
# g = sns.FacetGrid(df_train, row="Embarked", col="Survived", aspect=1.6)
# g.map(sns.barplot, "Sex", "Fare", alpha=0.5, errorbar=None, order=["male", "female"])
# g.add_legend()
# plt.show()

# Drop unimportant features (lots of missing values)
df_train = df_train.drop(["Ticket", "Cabin"], axis=1)
df_test = df_test.drop(["Ticket", "Cabin"], axis=1)
combine = [df_train, df_test]

# Create Title feature, replace rare titles, map to ordinal values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Name"].str.extract("([A-Za-z]+)\\.", expand=False)
    dataset["Title"] = dataset["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    dataset["Title"] = dataset["Title"].replace(["Mlle", "Ms"], "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

# Drop names from dataset
df_train = df_train.drop(["PassengerId", "Name"], axis=1)
df_test = df_test.drop("Name", axis=1)
combine = [df_train, df_test]

# Convert sex to numerical values
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1}).astype(int)

# Replace any Null ages with median values based on Sex (i) and Pclass (j)
guess_ages = np.zeros((2, 3))
for dataset in combine:
    # Get median age and pass median age to guess_age for each sex and class
    # ie. [0, 0] is male, first class; [1, 2] is female, third class
    for i in range(0, 2):
        for j in range(0, 3):
            df_guess = dataset[(dataset["Sex"] == i) & (dataset["Pclass"] == j + 1)][
                "Age"
            ].dropna()
            age_guess = df_guess.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    # For every sex and class that has null age pass in the guess_age
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[
                (dataset["Age"].isnull())
                & (dataset["Sex"] == i)
                & (dataset["Pclass"] == j + 1),
                ["Age"],
            ] = guess_ages[i, j]

    dataset["Age"] = dataset["Age"].astype(int)

# Convert age into ordinals based on age ranges
for dataset in combine:
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[(dataset["Age"] > 64), "Age"] = 4

# Create temporary feature for FamilySize that combines SibSp and Parch
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

# Create new feature IsAlone that determines if they had family aboard or not
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

# Drop Parch, SibSp, and FamilySize
df_train = df_train.drop(["Parch", "SibSp", "FamilySize"], axis=1)
df_test = df_test.drop(["Parch", "SibSp", "FamilySize"], axis=1)
combine = [df_train, df_test]

# Create new feature Age*Class that multiplies Age and Pclass
for dataset in combine:
    dataset["Age*Class"] = dataset["Age"] * dataset["Pclass"]

# Fill missing values in Embarked with most common value and map to numerical values
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(
        df_train["Embarked"].dropna().mode()[0]
    )
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

# Fill missing value with median fare value
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].dropna().median())

# Convert fare ranges into ordinal values, the more money the higher the value
for dataset in combine:
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

# Split the data into train and test
df_answer = pd.read_csv(".\\gender_submission.csv")
x_train = df_train.drop("Survived", axis=1)
y_train = df_train["Survived"]
x_test = df_test.drop("PassengerId", axis=1)
y_test = df_answer.drop("PassengerId", axis=1)

# # Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test).reshape(-1, 1)
acc_log = round(logreg.score(x_test, y_test) * 100, 2)

# # Coefficient of features
# # For example, Sex is 2.201033 which as sex increases 0 (male) -> 1 (female) chance of
# # surviving increases. As Pclass increases, probability of survival decreases.
# df_coeff = pd.DataFrame(df_train.columns.delete(0))
# df_coeff.columns = ["Feature"]
# df_coeff["Correlation"] = pd.Series(logreg.coef_[0])
# print(df_coeff.sort_values(by="Correlation", ascending=False))

# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100, 2)

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_test, y_test) * 100, 2)

# Gaussian Naive Bayes
gauss = GaussianNB()
gauss.fit(x_train, y_train)
y_pred = gauss.predict(x_test)
acc_gauss = round(gauss.score(x_test, y_test) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_test, y_test) * 100, 2)

# Linear SVC
linear_svc = LinearSVC(dual="auto")
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear = round(linear_svc.score(x_test, y_test) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_test, y_test) * 100, 2)

# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
y_pred = dec_tree.predict(x_test)
acc_dec_tree = round(dec_tree.score(x_test, y_test) * 100, 2)

# Random Forest
rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(x_train, y_train)
y_pred = rand_forest.predict(x_test)
acc_rand_forest = round(rand_forest.score(x_test, y_test) * 100, 2)

# Print accuracy scores
models = pd.DataFrame(
    {
        "Model": [
            "Support Vector Machines",
            "KNN",
            "Logistic Regression",
            "Random Forest",
            "Naive Bayes",
            "Perceptron",
            "Stochastic Gradient Descent",
            "Linear SVC",
            "Decision Tree",
        ],
        "Score": [
            acc_svc,
            acc_knn,
            acc_log,
            acc_rand_forest,
            acc_gauss,
            acc_perceptron,
            acc_sgd,
            acc_linear,
            acc_dec_tree,
        ],
    }
)
print(models.sort_values(by="Score", ascending=False))
