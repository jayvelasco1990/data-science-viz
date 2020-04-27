#exploratory data analysis
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('DATA/cancer_classification.csv')
# print(df.info()) #data info
# print(df.describe()) #describe

#sns.countplot(x='benign_0__mal_1', data=df)

# df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# sns.heatmap(df.corr())
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# plt.show()

model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5)) #half neurons will be turned off randomly
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5)) #half neurons will be turned off randomly
#Binary classification
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test,y_test))

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test,y_test),
callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict_classes(X_test)


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))