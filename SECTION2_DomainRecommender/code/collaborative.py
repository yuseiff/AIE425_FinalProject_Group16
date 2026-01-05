# **Group 16**

# Adham Mohmed Elsaied Elwakel 222100195,
# Samaa Khaled Eltaky 222100761,
# Habiba Ahmed Abdelnapy 222100471, 
# Youssef Hussieny 222101943

# collaborative.py

import pandas as pd
import numpy as np


def get_user_item_matrix(df):
    """
    Creates the User-Item Interaction Matrix.
    """
    matrix = df.pivot_table(
        index='user_id', columns='item_id', values='rating', aggfunc='mean'
    ).fillna(0)
    return matrix

def compute_cosine_similarity_scratch(matrix):
    """
    Computes Cosine Similarity manually using NumPy algebra.
    matrix: (Users x Items) DataFrame
    """
    print("[Collaborative] Computing Cosine Similarity from scratch...")
    
    # Transpose so Items are rows (We want Item-Item similarity)
    item_matrix_np = matrix.T.values
    
    # Dot Product
    numerator = np.dot(item_matrix_np, item_matrix_np.T)
    
    # Magnitudes (L2 Norm)
    magnitudes = np.sqrt(np.sum(item_matrix_np**2, axis=1))
    denominator = np.outer(magnitudes, magnitudes)
    
    # Division (Safe division to avoid NaN)
    sim = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)
    return sim_df

def compute_svd_scratch(matrix, k=20):
    """
    Performs Matrix Factorization (SVD) from scratch using Eigendecomposition.
    
    Mathematical Logic:
    1. Center the data (M)
    2. Compute Covariance Matrix (C = M^T * M)
    3. Compute Eigenvalues/Vectors of C to get V and Sigma
    4. Compute U using projection (U = M * V * Sigma^-1)
    5. Reconstruct = U * Sigma * V^T + Mean
    """
    print(f"[Collaborative] Training SVD from scratch (k={k})...")
    
    # 1. Convert to Numpy and Center Data
    X = matrix.values
    user_means = np.mean(X, axis=1)
    # Subtract mean (broadcasting)
    X_centered = X - user_means[:, None]
    
    # 2. Compute Item-Item Covariance Matrix (X^T * X)
    # We use X^T * X because it is smaller (Items x Items) than X * X^T (Users x Users)
    covariance_matrix = np.dot(X_centered.T, X_centered)
    
    # 3. Eigendecomposition
    # eigh is optimized for symmetric/hermitian matrices (Covariance is symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 4. Sort Eigenvalues and Vectors in Descending Order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 5. Truncate to top k latent factors
    top_k_indices = sorted_indices[:k]
    top_eigenvalues = eigenvalues[top_k_indices]
    
    # Vt (V Transpose) are the top k eigenvectors
    # eigenvectors from eigh are columns, so we transpose to get V^T rows
    Vt = eigenvectors[:, top_k_indices].T 
    V = Vt.T
    
    # 6. Compute Sigma (Singular Values)
    # Eigenvalues of X^T*X are the square of singular values
    # Clip to 0 to handle potential negative machine precision errors
    singular_values = np.sqrt(np.maximum(top_eigenvalues, 0))
    Sigma = np.diag(singular_values)
    
    # 7. Compute U (User Matrix)
    # Projection formula: U = X_centered * V * Sigma_inverse
    # We add a small epsilon to avoid division by zero
    Sigma_inv = np.diag(1 / (singular_values + 1e-10))
    U = np.dot(np.dot(X_centered, V), Sigma_inv)
    
    print(f"[Collaborative] SVD Components shape: U={U.shape}, Sigma={Sigma.shape}, Vt={Vt.shape}")
    
    # 8. Reconstruct Matrix
    # Prediction = U * Sigma * Vt + Mean
    reconstructed_data = np.dot(np.dot(U, Sigma), Vt) + user_means[:, None]
    
    # Convert to DataFrame
    preds_df = pd.DataFrame(
        reconstructed_data, 
        columns=matrix.columns, 
        index=matrix.index
    )
    
    return preds_df

if __name__ == "__main__":
    # Load Data
    try:
        df = pd.read_csv('../data/preprocessed_data.csv')
    except FileNotFoundError:
        print("Error: '../data/preprocessed_data.csv' not found.")
        exit()

    df['item_id'] = df['item_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    
    matrix = get_user_item_matrix(df)
    
    # 1. Item-Based CF (Cosine)
    cf_sim = compute_cosine_similarity_scratch(matrix)
    cf_sim.to_csv('../results/item_similarity_cf.csv')
    print("[Collaborative] CF Matrix saved to '../results/item_similarity_cf.csv'")
    
    # 2. Matrix Factorization (SVD)
    # We run this to verify it works, though predictions are typically generated on the fly
    svd_preds = compute_svd_scratch(matrix, k=20)
    
    # Optional: Save SVD predictions if needed, or just keep the logic
    svd_preds.to_csv('../results/svd_predictions.csv')
    print(f"[Collaborative] SVD Reconstruction Complete. Shape: {svd_preds.shape}")