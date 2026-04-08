# 24ADI003_24BAD020_EXP-10


SCENARIO 1 – MATRIX FACTORIZATION USING SVD


Problem Statement:
Use Singular Value Decomposition (SVD) to recommend movies based on latent user-item interactions.


Dataset (Kaggle – Public):
MovieLens Dataset Dataset Link: https://www.kaggle.com/datasets/abhikjha/movielens-100k


Description of the code:

Singular Value Decomposition (SVD) is applied to a movie recommendation system using the MovieLens dataset. The dataset containing user IDs, movie IDs, and ratings is first preprocessed by converting it into a user-item interaction matrix, where rows represent users and columns represent movies. Missing values in the matrix are filled with zeros, and the ratings are normalized to ensure that evaluation metrics such as RMSE and MAE fall within a standard range. Mean centering is performed to remove user bias before applying SVD. The matrix is then decomposed into three components representing latent user and item features. By selecting a reduced number of latent factors, the matrix is reconstructed to approximate the original data while capturing essential patterns. The reconstructed matrix is used to predict missing ratings and generate top-N movie recommendations for users. The performance of the model is evaluated using RMSE and MAE, where lower values indicate better prediction accuracy.

The visualizations play a crucial role in understanding the model performance. The heatmap of the original user-item matrix highlights the sparsity of the dataset, while the reconstructed matrix heatmap shows how SVD fills in missing values with predicted ratings. The error versus number of latent factors graph helps analyze the impact of dimensionality reduction, showing how the prediction error decreases initially and may increase due to overfitting. The top recommended movies bar chart visualizes the highest predicted ratings for a user, helping interpret recommendation results effectively. Overall, SVD demonstrates how dimensionality reduction can improve recommendation accuracy by capturing hidden relationships between users and items.







SCENARIO 2 – MATRIX FACTORIZATION USING NMF


Problem Statement:
Use Non-negative Matrix Factorization (NMF) to generate recommendations based on latent features.



Dataset (Kaggle – Public):
MovieLens Dataset Dataset Link: https://www.kaggle.com/datasets/abhikjha/movielens-100k


Description of the code:

Non-negative Matrix Factorization (NMF) is used to generate recommendations based on latent features extracted from the MovieLens dataset. The dataset is first transformed into a user-item matrix, where missing ratings are replaced with zeros. To ensure proper scaling, the ratings are normalized before applying the NMF model. NMF decomposes the matrix into two non-negative matrices: a user-feature matrix and an item-feature matrix. These matrices represent latent characteristics of users and items in an interpretable manner. The original matrix is reconstructed by multiplying these two matrices, allowing the prediction of missing ratings. Based on these predicted values, top-N recommendations are generated for each user. The model performance is evaluated using RMSE, ensuring the prediction error remains within an acceptable range.

The visualizations help in analyzing the effectiveness of the NMF model. The latent feature visualization (user-feature matrix) illustrates how users are represented in terms of hidden features. The reconstruction comparison heatmap displays the original and reconstructed matrices side by side, highlighting how missing values are approximated. The recommendation ranking chart shows the top recommended movies based on predicted ratings, making it easy to interpret the results. Compared to SVD, NMF provides more interpretable results due to its non-negative constraints, making it suitable for applications where understanding latent features is important. Overall, NMF effectively handles sparse data and provides meaningful recommendations by capturing additive relationships between users and items.
