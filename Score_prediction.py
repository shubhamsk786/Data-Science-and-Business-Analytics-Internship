#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading data from remote url
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

#Exploring Data
print(data.shape)
data.head()

data.describe()
data.info()

data.plot(kind="scatter", x="Hours", y="Scores");
plt.show()

data.corr(method="pearson")
data.corr(method="spearman")

hours = data['Hours']
scores = data['Scores']

sns.distplot(hours)
sns.distplot(scores)

#Linear Regression
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

m = reg.coef_
c = reg.intercept_
line = m*X+c
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

y_pred = reg.predict(X_test)

actual_predicted = pd.DataFrame({'Target':y_test, "Predicted":y_pred})
actual_predicted

sns.set_style("whitegrid")
sns.distplot(np.array(y_test-y_pred))
plt.show()

#What would be the predicted score if a student studies for 9.25 hours/day?
h = 9.25
s = reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam.".format(h,s))

#Model Evaluation
from sklearn import metrics
from sklearn.metrics import r2_score
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
