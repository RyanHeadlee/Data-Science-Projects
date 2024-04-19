import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import warnings

warnings.filterwarnings("ignore")

# Read the dataset
df_train = pd.read_csv(".\\train.csv")

# # Skew and Kurtosis for sale prices
# print("Skewness: {}".format(df_train["SalePrice"].skew()))
# print("Kurtosis: {}".format(df_train["SalePrice"].kurt()))

# # Correlation Matrix
# corrmat = df_train.corr(numeric_only=True)
# fig, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()

# # SalePrice correlation matrix
# corrmat = df_train.corr(numeric_only=True)
# fig, ax = plt.subplots(figsize=(12, 9))
# k = 10  # Number of variables for heatmap
# cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set_theme(font_scale=1.25)
# hm = sns.heatmap(
#     cm,
#     cbar=True,
#     annot=True,
#     square=True,
#     fmt=".2f",
#     annot_kws={"size": 10},
#     yticklabels=cols.values,
#     xticklabels=cols.values,
# )
# plt.show()

# Missing Data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(
    ascending=False
)
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

# Dealing with missing data
df_train = df_train.drop((missing_data[missing_data["Total"] > 1]).index, axis=1)
df_train = df_train.drop(df_train.loc[df_train["Electrical"].isnull()].index)

# Standardizing Data
saleprice_scaled = StandardScaler().fit_transform(
    np.array(df_train["SalePrice"]).reshape(-1, 1)
)
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()[:10]]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()[-10:]]
# print("Outer range (low) of the distribution: \n{}".format(low_range))
# print("Outer range (high) of the distribution: \n{}".format(high_range))

# Deleting Points (outliers)
df_train = df_train.drop(
    df_train.sort_values(by="GrLivArea", ascending=False)[:2].index
)

# Applying log transformations (due to the positive skew)
df_train["SalePrice"] = np.log(df_train["SalePrice"])
df_train["GrLivArea"] = np.log(df_train["GrLivArea"])
df_train.loc[df_train["TotalBsmtSF"] > 0, "TotalBsmtSF"] = np.log(
    df_train["TotalBsmtSF"]
)  # Only apply log if TotalBsmtSF is greater than zero

# # Histogram and normal probability plot
# sns.distplot(df_train["SalePrice"], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train["SalePrice"], plot=plt)
# plt.show()

# New data frame with most relevant features
df_new_train = pd.concat(
    [
        df_train["GrLivArea"],
        df_train["TotalBsmtSF"],
        df_train["SalePrice"],
        df_train["OverallQual"],
    ],
    axis=1,
)

# Split the data via [GrLivArea, TotalBsmtSF, OverallQual] and SalePrice
x_train, x_test, y_train, y_test = train_test_split(
    df_new_train.drop("SalePrice", axis=1),
    df_new_train["SalePrice"],
    test_size=0.2,
    random_state=42,
)

# Convert to array and reshape
x_train = np.array(x_train).reshape(-1, len(df_new_train.columns) - 1)
x_test = np.array(x_test).reshape(-1, len(df_new_train.columns) - 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Create model
model = LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

# Print explained variance (closer to 1 is more accurate)
print(
    "Explained Variance: {}".format(round(explained_variance_score(y_test, predict), 5))
)  # Using more variables (features) proved to result in far more accurate results (.57 -> .79)
