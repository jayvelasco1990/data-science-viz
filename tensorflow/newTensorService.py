import numpy as py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('DATA/fake_reg.csv')
graph = df.head()

# sns.pairplot(df)
#plt.show()

X = df[['feature1', 'feature2']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train) #do not include test data, would be cheating
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=250)

loss_df = pd.DataFrame(model.history.history)
#loss_df.plot()

model.evaluate(X_test, y_test, verbose=0)

model.evaluate(X_train,y_train,verbose=0)

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']
sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
mae = mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])
rmse = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])**.5
print(pred_df)
plt.show()
# https://www.google.com/search?client=safari&rls=en&q=what+is+a+pairplot&ie=UTF-8&oe=UTF-8