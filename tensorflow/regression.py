import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

df = pd.read_csv('DATA/kc_house_data.csv')
# df.head()
# df.isnull().sum()
# df.describe().transpose()
# print(df.transpose())

# plt.figure(figsize=(10,6))
# sns.distplot(df['price'])
# plt.show()

# sns.countplot(df['bedrooms'])

df.corr()['price'].sort_values() #correlation

# plt.figure(figsize=(10,5))
# sns.scatterplot(x='price', y='sqft_living', data=df)
# plt.figure(figsize=(10,6))
# sns.boxplot(x='bedrooms', y='price', data=df) #prices between bedrooms
# plt.figure(figsize=(12,8))
# sns.scatterplot(x='price',y='long',data=df)
# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long', y='lat', data=df, hue='price') #heatmap of scatterplot
# plt.show()
df.sort_values('price', ascending=False).head(20)

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

#plt.figure(figsize=(12,8))
# sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None,alpha=0.2, palette='RdYlGn',hue='price') #shows top 1% geogrpahically
#sns.boxplot(x='waterfront',y='price',data=df)
df = df.drop('id', axis=1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
#plt.figure(figsize=(10,6))
#sns.boxplot(x='month', y='price', data=df)
# print(df.groupby(['month', 'year']).mean()['price'].unstack())
df.groupby(['month', 'year']).mean()['price'].unstack().plot()
df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)
# df['zipcode'].value_counts()
# df['yr_renovated'].value_count()
# plt.show()
# https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
X = df.drop('price',axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)

losses = pd.DataFrame(model.history.history)

predictions = model.predict(X_test)

# np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
# print(mae)

explained_variance_score(y_test,predictions)

plt.figure(figsize=(12,6))
# plt.scatter(y_test, predictions)
# df['price'].describe()
# losses.plot()
plt.plot(y_test,y_test,'r')
plt.show()

single_house = df.drop('price', axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))

print(model.predict(single_house))
print(df.head(1))
# print(history)



