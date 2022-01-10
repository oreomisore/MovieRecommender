# Description: A movie recommendation engine using python.

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('/Users/oreomisore/Downloads/IMDB-Movie-Data.csv')
df['Movie_id'] = range(0, 1000)

# List of important columns for the recommendation engine.
columns = ['Actors', 'Director', 'Genre', 'Title']

#  Function to combine the values of the important columns into a single string


def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['Actors'][i]+' '+ data['Director'][i]+' '+data['Genre'][i]+' '+data['Title'][i])
    return important_features


# Column to hold the combineD strings
df['important_features'] = get_important_features(df)

# Text to a matrix of token counts
cm = CountVectorizer().fit_transform(df['important_features'])

# Cosine similarity matrix from the count matrix
cs = cosine_similarity(cm)

# Title of the movie that the user likes
title = 'Interstellar'

# Find the movies id
movie_id = df[df.Title == title]['Movie_id'].values[0]

# List enumerations for the similarity score. Will create list of tuples (movie_id, similarity score), (...)
scores = list(enumerate(cs[movie_id]))

# Sort the list
# Lambda function takes input x and will return the element at position 1 of x)
# x is the movie id & the element at position x is the similarity score to the given movie title.
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]

# Loop to print the first 7 similar movies.
j = 0
print('The 7 most recommended movies to', title, 'are:\n')
for item in sorted_scores:
    movie_title = df[df.Movie_id == item[0]]['Title'].values[0]
    print(j+1, movie_title)
    j = j+1
    if j > 6:
        break
