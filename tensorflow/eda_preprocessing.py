#exploratory data analysis
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('DATA/cancer_classification.csv')
print(df.info()) #data info
print(df.describe()) #describe

#sns.countplot(x='benign_0__mal_1', data=df)

# df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

sns.heatmap(df.corr())
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.show()

