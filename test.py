import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('tweets.csv')
# Get tf-idf matrix using fit_transform function
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['text']) # Store tf-idf representations of all docs

print(X.shape)

query = "France"


query_vec = vectorizer.transform([query]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc

# Print Top 10 results
for i in results.argsort()[-20:][::-1]:
    print('Les 20 premiers tweets correspondats sont \n:' + df.iloc[i,3],"--",df.iloc[i,5])