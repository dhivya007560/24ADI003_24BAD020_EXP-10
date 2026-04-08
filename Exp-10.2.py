print("DHIVYA A : 24BAD020")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

data = pd.read_csv("E:/ASSIGNMENTS/ML/DATASETS/ml-latest-small/ratings.csv")

user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

matrix = user_item_matrix.values / 5.0

model = NMF(n_components=20, init='random', random_state=42, max_iter=200)

W = model.fit_transform(matrix)
H = model.components_

reconstructed = np.dot(W, H)

mask = matrix > 0
actual = matrix[mask]
pred = reconstructed[mask]

rmse = np.sqrt(mean_squared_error(actual, pred))
print("RMSE:", rmse)

user_id = user_item_matrix.index[0]
user_index = list(user_item_matrix.index).index(user_id)

user_ratings = reconstructed[user_index]
top_movies = np.argsort(user_ratings)[-10:][::-1]

plt.figure()
plt.imshow(W, aspect='auto')
plt.title("User-Feature Matrix (W)")
plt.xlabel("Latent Features")
plt.ylabel("Users")
plt.colorbar()
plt.show()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(user_item_matrix.values, aspect='auto')
plt.title("Original Matrix")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(reconstructed, aspect='auto')
plt.title("Reconstructed Matrix (NMF)")
plt.colorbar()

plt.tight_layout()
plt.show()

plt.figure()
plt.bar(range(len(top_movies)), user_ratings[top_movies])
plt.xticks(range(len(top_movies)), top_movies, rotation=45)
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.title("Top Recommended Movies (NMF)")
plt.show()
