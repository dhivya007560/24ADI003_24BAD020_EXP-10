print("DHIVYA A : 24BAD020")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv("E:/ASSIGNMENTS/ML/DATASETS/ml-latest-small/ratings.csv")
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item_matrix.values / 5.0
user_mean = np.mean(matrix, axis=1).reshape(-1,1)
matrix_centered = matrix - user_mean
U, sigma, Vt = np.linalg.svd(matrix_centered, full_matrices=False)
k = 20
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]
reconstructed = np.dot(np.dot(U_k, sigma_k), Vt_k) + user_mean
predicted = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)
mask = matrix > 0
actual = matrix[mask]
pred = reconstructed[mask]
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)
print("RMSE:", rmse)
print("MAE:", mae)
user_id = user_item_matrix.index[0]
user_ratings = predicted.loc[user_id]
top_movies = user_ratings.sort_values(ascending=False).head(10)
errors = []
k_values = [5, 10, 20, 30, 40]
for k in k_values:
    sigma_k = np.diag(sigma[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    recon = np.dot(np.dot(U_k, sigma_k), Vt_k) + user_mean
    pred_k = recon[mask]
    errors.append(np.sqrt(mean_squared_error(actual, pred_k)))
plt.figure()
plt.imshow(user_item_matrix.values, aspect='auto')
plt.colorbar()
plt.title("Original User-Item Matrix")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure()
plt.imshow(reconstructed, aspect='auto')
plt.colorbar()
plt.title("Reconstructed Matrix using SVD")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure()
plt.plot(k_values, errors, marker='o')
plt.xlabel("Number of Latent Factors (k)")
plt.ylabel("RMSE")
plt.title("Error vs Number of Latent Factors")
plt.show()

plt.figure()
plt.bar(range(len(top_movies)), top_movies.values)
plt.xticks(range(len(top_movies)), top_movies.index, rotation=45)
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.title("Top Recommended Movies")
plt.show()
