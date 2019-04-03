
# coding: utf-8

# # MOVIE RECOMMENDATIONS: 

# # Imports:
#     1. sklearn: for cosine_similarity and CountVectorizer
#     2. rake_nltk: to analyze key phrases in text
#     3. pandas: to store data from CSV file of around 5000 movies

# In[1]:

import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# # Reads data from CSV file and stores it into pandas df:
#     -- Includes keywords of plot, the movie titles, genre, and director's name -- 

# In[2]:

df = pd.read_csv("movie_metadata.csv", encoding='utf-8')
df = df[['movie_title','genres','director_name','plot_keywords']]
df.head()


# # Cleans data:
#     - Gets rid of excess characters (eg.: '|' and replaces any null entries with spaces)
#     - Lowercases any capital titles

# In[3]:

df = df.replace(np.nan, '', regex = True)
df['plot_keywords']= [review.replace("|"," ") for review in df['plot_keywords'].values]
df['genres']= [review.replace("|"," ") for review in df['genres'].values]


# In[4]:

df['Key_words'] = ""
df['Key_words'] = df["plot_keywords"].map(str) + ' ' + df['genres'].map(str) + ' ' + df['director_name']

c = df.columns[df.dtypes == object]
df[c] = df[c].apply(lambda x: x.str.replace(r'[^\x00-\x7F]+', ''))
df.head()


# # Vectorizing Data:
#     - Generates similarity matrix with similarity indices based on how similar keywords are to one another 

# In[5]:

count = CountVectorizer()
count_matrix = count.fit_transform(df['Key_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)


# # Get Movie Recommendations:
#     - Finds the movies that are quite similar to user's request (a.k.a: ones with highest similarity scores)

# In[6]:

def recommendations(title,num):
    recommended_movies = []
    idx = df[df['movie_title'] == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:num+1].index)
    
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    recs = set(df.iloc[recommended_movies]['movie_title'])
    if title in recs:
        recs.remove(title)
    
    return recs


# # Implements Process:
#     - Gets movie title (that is within database) from user and print a definite user-defined number of movierecommendations within database

# In[7]:

def main():
    print('Enter a movie: ',end='')
    movie = input()
    print('How many movie recommendations would you like?: ',end='')
    number = int(input())
    print()
    print('Here are some movie recommendations: ')
    for i,item in enumerate(recommendations(movie,number)):
        print(str(i+1) + ". " + item)


# In[ ]:

main()


# In[ ]:



