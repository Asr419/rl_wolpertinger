import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import os
import plotly.express as px
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


spotify_data = pd.read_csv('data/data.csv')
genre_data = pd.read_csv('data/data_by_genres.csv')
data_by_year = pd.read_csv('data/data_by_year.csv')


sound_features = ['acousticness', 'danceability',
                  'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(data_by_year, x='year', y=sound_features)
fig.show()

loudness = spotify_data[['loudness']].values
min_max_scaler = preprocessing.MinMaxScaler()
loudness_scaled = min_max_scaler.fit_transform(loudness)

songs_features = spotify_data[[
    "danceability", "loudness", "speechiness", "acousticness", "liveness"]]

kmeans = KMeans(n_clusters=4)
kmeans.fit(songs_features)

y_kmeans = kmeans.predict(songs_features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(songs_features)
