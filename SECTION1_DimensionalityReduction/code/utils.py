"""
Utility functions for dimensionality reduction analysis.

This module contains helper functions used across different notebooks
for PCA and SVD analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath)


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ratings data.
    
    Args:
        df: DataFrame containing ratings
        
    Returns:
        Preprocessed DataFrame
    """
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Sort by user and item
    df = df.sort_values(['userId', 'movieId'])
    
    return df


def create_user_item_matrix(df: pd.DataFrame, 
                            user_col: str = 'userId',
                            item_col: str = 'movieId', 
                            rating_col: str = 'rating') -> pd.DataFrame:
    """
    Create user-item matrix from ratings dataframe.
    
    Args:
        df: DataFrame containing ratings
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column
        
    Returns:
        User-item matrix with users as rows and items as columns
    """
    return df.pivot_table(index=user_col, columns=item_col, values=rating_col)


def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> float:
    """
    Calculate Root Mean Square Error between original and reconstructed matrices.
    
    Args:
        original: Original matrix
        reconstructed: Reconstructed matrix
        mask: Optional boolean mask for known values
        
    Returns:
        RMSE value
    """
    if mask is not None:
        diff = original[mask] - reconstructed[mask]
    else:
        diff = original - reconstructed
        
    return np.sqrt(np.mean(diff ** 2))


def plot_variance_explained(variance_ratios: np.ndarray, 
                           n_components: int = 50,
                           save_path: Optional[str] = None):
    """
    Plot explained variance ratio for PCA components.
    
    Args:
        variance_ratios: Array of explained variance ratios
        n_components: Number of components to plot
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual variance
    ax1.plot(range(1, n_components + 1), variance_ratios[:n_components], 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    cumulative_variance = np.cumsum(variance_ratios[:n_components])
    ax2.plot(range(1, n_components + 1), cumulative_variance, 'ro-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_singular_values(singular_values: np.ndarray,
                        n_values: int = 50,
                        save_path: Optional[str] = None):
    """
    Plot singular values from SVD.
    
    Args:
        singular_values: Array of singular values
        n_values: Number of values to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_values + 1), singular_values[:n_values], 'bo-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(range(1, n_values + 1), singular_values[:n_values], 'ro-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_results_table(data: dict, filepath: str, index_name: str = 'Metric'):
    """
    Save results as a formatted table.
    
    Args:
        data: Dictionary containing results
        filepath: Path to save the CSV file
        index_name: Name for the index column
    """
    df = pd.DataFrame(data)
    df.to_csv(filepath, index_label=index_name)
    print(f"Results saved to: {filepath}")


def print_summary_statistics(matrix: np.ndarray, name: str = "Matrix"):
    """
    Print summary statistics for a matrix.
    
    Args:
        matrix: Input matrix
        name: Name to display
    """
    print(f"\n{name} Summary Statistics:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Min: {np.min(matrix):.4f}")
    print(f"  Max: {np.max(matrix):.4f}")
    print(f"  Mean: {np.mean(matrix):.4f}")
    print(f"  Std: {np.std(matrix):.4f}")
    print(f"  Missing values: {np.isnan(matrix).sum()}")
