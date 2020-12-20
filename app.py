# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, redirect,url_for
import json

app = Flask(__name__)

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 

def search(query):
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
	query_vec = vectorizer.transform([query]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
	results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc

# Print Top 10 results
	for i in results.argsort()[-20:][::-1]:
	 return (df.iloc[i,3],"--",df.iloc[i,5])


@app.route('/')
def home():
	query = request.args.get('validation')
	return render_template('twitter.html',query=query)

@app.route('/', methods=['POST'])
def test():
    query = request.form['validation']
    query = search(listToString([query]))
    print(query)
    return render_template('twitter.html', query=query)


if __name__ == '__main__':
	app.run(host='0.0.0.0')