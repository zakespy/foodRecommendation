import pandas as pd
import numpy as np
import streamlit as st
# Data Preprocessing
from sklearn import preprocessing

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Recommender System Imps
# Content Based Filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Collaborative Based Filtering
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# To work with text data
import re
import string
df = pd.read_csv('1662574418893344.csv')
# df.head()
len(list(df['Name'].unique()))
df['C_Type'].unique() # Categorical Data
df['Veg_Non'].unique()
def text_cleaning(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    return text
df['Describe'] = df['Describe'].apply(text_cleaning)
# df.head()
df.duplicated().sum()
# Are there any null values?
df.isnull().sum()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Describe'])
# tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# cosine_sim
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
# indices
def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar food
    sim_scores = sim_scores[1:6]

    food_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[food_indices]
features = ['C_Type','Veg_Non', 'Describe']
def create_soup(x):
    return x['C_Type'] + " " + x['Veg_Non'] + " " + x['Describe']
def create_soup(x):
    return x['C_Type'] + " " + x['Veg_Non'] + " " + x['Describe']
df['soup'] = df.apply(create_soup, axis=1)
# df.head()
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index()
indices = pd.Series(df.index, index=df['Name'])
# display(indices)
# print(get_recommendations('tricolour salad'))
print("content")
print(get_recommendations('tricolour salad', cosine_sim2))


# Collabrative based filtering ********************************************************************************** 
rating = pd.read_csv('ratings.csv')
# rating.head()
# rating.shape
# Checking for null values
rating.isnull().sum()
rating.tail()
rating = rating[:511]
rating.tail()
rating.isnull().sum()
food_rating = rating.groupby(by = 'Food_ID').count()
food_rating = food_rating['Rating'].reset_index().rename(columns={'Rating':'Rating_count'})
# food_rating
food_rating['Rating_count'].describe()
user_rating = rating.groupby(by='User_ID').count()
user_rating = user_rating['Rating'].reset_index().rename(columns={'Rating':'Rating_count'})
# user_rating
user_rating["Rating_count"].describe()
rating_matrix = rating.pivot_table(index='Food_ID',columns='User_ID',values='Rating').fillna(0)
# rating_matrix.head()
# rating_matrix.shape
csr_rating_matrix =  csr_matrix(rating_matrix.values)
# print(csr_rating_matrix)
recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)
# The main recommender code!
def Get_Recommendations(title):
    user= df[df['Name']==title]
    user_index = np.where(rating_matrix.index==int(user['Food_ID']))[0][0]
    user_ratings = rating_matrix.iloc[user_index]

    reshaped = user_ratings.values.reshape(1,-1)
    distances, indices = recommender.kneighbors(reshaped,n_neighbors=16)

    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})

    result = pd.merge(nearest_neighbors,df,on='Food_ID',how='left')

    return result.head()
# print(Get_Recommendations('tricolour salad'))


# UI
st.title('Food Recommendation')
food = st.selectbox('Select a Food:', df.Name)

content = st.toggle('Content Based filtering')
collabrative = st.toggle('Collabrative Based Filtering')

# Display the selected color
if content and not collabrative:
    if st.button('Get Recommendation'):
        # get_recommendations('tricolour salad', cosine_sim2)
        foodData = get_recommendations(food, cosine_sim2)
        for i in foodData:
            st.write(i)
            print(i)
elif collabrative and not content:
    if st.button('Get Recommendation'):
        foodData = Get_Recommendations(food).Name
        for food in foodData:
            st.write(food)
elif content and collabrative:
    st.write('Select anyone') 
else:
    st.write('Select one filtering option')


    

