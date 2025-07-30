import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

st.title("Collaborative Filtering Recommendation System")

st.write("""
This app demonstrates a simple recommendation system using collaborative filtering and matrix factorization (SVD).
""")

# Sample user-item ratings matrix (users as rows, items as columns)
ratings_dict = {
    'User': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
    'Item1': [5, 4, np.nan, 2, 1],
    'Item2': [3, np.nan, 4, 1, 1],
    'Item3': [np.nan, 2, 5, 4, 2],
    'Item4': [1, 2, 2, 4, 5],
    'Item5': [np.nan, 5, 4, 3, 3]
}
df = pd.DataFrame(ratings_dict).set_index('User')
st.write("### User-Item Ratings Matrix")
st.dataframe(df)

# Fill missing values with the mean rating for each item
filled_df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# Matrix factorization using TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
matrix = svd.fit_transform(filled_df)

# Reconstruct the ratings
reconstructed = np.dot(matrix, svd.components_)
reconstructed_df = pd.DataFrame(reconstructed, index=filled_df.index, columns=filled_df.columns)

st.write("### Predicted Ratings Matrix")
st.dataframe(np.round(reconstructed_df, 2))

# Recommend top N items for a selected user
user = st.selectbox("Select a user for recommendations:", df.index)
top_n = st.slider("Number of recommendations:", 1, 3, 2)

user_ratings = df.loc[user]
pred_ratings = reconstructed_df.loc[user]

# Recommend items not already rated by the user
unrated_items = user_ratings[user_ratings.isna()].index
recommendations = pred_ratings[unrated_items].sort_values(ascending=False).head(top_n)

st.write(f"### Top {top_n} recommendations for {user}:")
for item, score in recommendations.items():
    st.write(f"- {item} (predicted rating: {score:.2f})")