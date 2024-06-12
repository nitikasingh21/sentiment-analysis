# -*- coding: utf-8 -*-
import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from google.colab import files
upload= files.upload()

df=pd.read_csv('sentimentdataset.csv')
df.head()

df['Text'].values[1]

print(df.shape)

df=df.head(500)
print(df.shape)

ax=df['Retweets'].value_counts().sort_index().plot(kind="bar", title="Sentiment analysis")
ax.set_xlabel('Likes')
plt.show()

Example=df['Text'].values[50]
print(Example)

nltk.download('punkt')
tokens= nltk.word_tokenize(Example)
tokens[:10]

from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
tagged=nltk.pos_tag(tokens)
tagged [:10]

nltk.download('all')

from nltk import ne_chunk
nltk.download('maxent_ne_chunker')

entities= nltk.chunk.ne_chunk(tagged)
entities.pprint

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia=SentimentIntensityAnalyzer()

from tqdm import tqdm

res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['Text'])
    idname = row['User']
    res[idname] = sia.polarity_scores(text)

res

vaders=pd.DataFrame(res).T
vaders

vaders=vaders.reset_index().rename(columns= {'index': 'User'})
vaders

if vaders['User'].dtypes != df['User'].dtypes:
  vaders = pd.concat([vaders, df], axis=1)
else:

  vaders = vaders.merge(df, how='left', left_on='User', right_on='User')

  vaders = vaders.loc[:,~vaders.columns.duplicated()]

vaders

vaders.head()

#Plotting vaders result
sns.barplot(data=vaders, x='Likes', y='compound')
ax.set_title('Social media review')
plt.show()

fig, axs= plt.subplots(1,3, figsize=(15,5))
sns.barplot(data=vaders, x='Text', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Text', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Text', y='neg', ax=axs[2])
axs[0].set_title='Positive'
axs[1].set_title='Neutral'
axs[2].set_title='Negative'
plt.tight_layout()
plt.show()