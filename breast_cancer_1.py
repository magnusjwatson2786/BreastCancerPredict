# -*- coding: utf-8 -*-
"""breast-cancer-1.ipynb

#Load Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

"""#Loading the dataset"""

df = pd.read_csv("data(1).csv")

df.head()

"""#Preprocessing"""

df.loc[:, df.columns.str.contains('Unnamed')]

df = df.loc[:, ~df.columns.str.contains('Unnamed')]

df.head()

df['diagnosis'].unique()

df['diagnosis']= (df['diagnosis'] == 'M').astype(int)

df

df['diagnosis'].unique()

df = df.drop('id', axis= 1)
df

"""#Exploratory Analysis of dataset"""

df.describe()

for label in df.columns[1:11]:
  plt.hist(df[df['diagnosis']==1][label], color = 'red', label ='M', alpha = 0.7, density = True)
  plt.hist(df[df['diagnosis']==0][label], color = 'green', label ='B', alpha = 0.7, density = True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

df.columns[1:]

"""#Training, Validation, Testing Datasets"""

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

train

def scale_dataset(dataframe, oversample=False):
  x = dataframe[dataframe.columns[1:]].values
  y = dataframe[dataframe.columns[0]].values

  sc= MinMaxScaler()
  x =sc.fit_transform(x)

  if oversample:
    ros= RandomOverSampler()
    x, y = ros.fit_resample(x, y)

  data = np.hstack((x,np.reshape(y, (len(y), 1))))

  return data, x, y

train, x_tr, y_tr = scale_dataset(train, True)
valid, x_vd, y_vd = scale_dataset(valid)
test, x_ts, y_ts = scale_dataset(test)

"""#Feature Selection"""

select_feature = SelectKBest(chi2, k=10).fit(x_tr, y_tr)
select_feature.scores_

X_tr=select_feature.transform(x_tr)
X_vd=select_feature.transform(x_vd)
X_ts=select_feature.transform(x_ts)

"""#Model classification

##kNN
"""

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_tr, y_tr)

y_pred_knn = knn_model.predict(X_ts)

print(classification_report(y_ts, y_pred_knn))

"""##Naive Bayes"""

nb_model = GaussianNB()
nb_model = nb_model.fit(X_tr, y_tr)

y_pred_nb = nb_model.predict(X_ts)

print(classification_report(y_ts, y_pred_nb))

"""##Support VM"""

svm_model = SVC()
svm_model = svm_model.fit(X_tr, y_tr)

y_pred_svm = svm_model.predict(X_ts)

print(classification_report(y_ts, y_pred_svm))

"""##Logistic Regression"""

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_tr, y_tr)

y_pred_lg = lg_model.predict(X_ts)

print(classification_report(y_ts, y_pred_lg))

"""##Decision Tree"""

dt_model = DecisionTreeClassifier()
dt_model = dt_model.fit(X_tr, y_tr)

y_pred_dt = dt_model.predict(X_ts)

print(classification_report(y_ts, y_pred_dt))

"""##Random Forest"""

rf_model = RandomForestClassifier(n_estimators= 10, criterion="entropy")
rf_model = rf_model.fit(X_tr, y_tr)

y_pred_rf = rf_model.predict(X_ts)

print(classification_report(y_ts, y_pred_rf))