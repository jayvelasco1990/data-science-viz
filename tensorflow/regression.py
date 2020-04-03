import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None,
alpha=0.2, palette='RdYlGn',hue='price') #shows top 1% geogrpahically
sns.boxplot(x='waterfront',y='price',data=df)
plt.show()