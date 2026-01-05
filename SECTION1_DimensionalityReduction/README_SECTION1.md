# Section 1: Dimensionality Reduction

## Overview
This section explores dimensionality reduction techniques applied to collaborative filtering for recommendation systems. We implement and compare different approaches including PCA with various imputation strategies and SVD-based collaborative filtering.

## Contents

### Notebooks

#### 1. `pca_mean_filling.ipynb`
- **Purpose**: Implement PCA with mean value imputation for missing ratings
- **Key Topics**:
  - Data loading and preprocessing
  - Mean-based missing value imputation
  - PCA dimensionality reduction
  - Variance analysis
  - Reconstruction and evaluation

#### 2. `pca_mle.ipynb`
- **Purpose**: Implement PCA with MLE-based imputation
- **Key Topics**:
  - MLE approach for missing data
  - Statistical imputation methods
  - PCA applied to imputed data
  - Comparative analysis with mean filling

#### 3. `svd_analysis.ipynb`
- **Purpose**: SVD-based collaborative filtering
- **Key Topics**:
  - Singular Value Decomposition
  - Latent factor models
  - Collaborative filtering implementation
  - Performance evaluation and comparison
  - Cold-start problem analysis

### Utility Module
`utils.py` contains reusable functions:
- Data loading and preprocessing
- Matrix creation and manipulation
- Evaluation metrics (RMSE, MAE)
- Visualization functions
- Result saving utilities

## Data
The `data/` directory contains the MovieLens dataset used for analysis. Ensure the following files are present:
- `ratings.csv` - User-item ratings data (userId, movieId, rating, timestamp)
- `movies.csv` - Movie metadata (movieId, title, genres)

## Running the Analysis

### Prerequisites
Make sure you have installed all requirements:
```bash
pip install -r ../../requirements.txt
```

### Execution Order
Run the notebooks in sequence:

1. **Start with PCA Mean Filling**:
   ```bash
   jupyter notebook code/pca_mean_filling.ipynb
   ```
   This establishes baseline performance using simple mean imputation.

2. **Run PCA MLE**:
   ```bash
   jupyter notebook code/pca_mle.ipynb
   ```
   Compare MLE-based imputation with the mean filling approach.

3. **Execute SVD Analysis**:
   ```bash
   jupyter notebook code/svd_analysis.ipynb
   ```
   Perform comprehensive SVD analysis and compare with PCA methods.

### Expected Outputs
Each notebook generates:
- **Plots**: Saved to `results/plots/`
  - Variance explained plots
  - Singular value distributions
  - Error curves
  - Comparative visualizations

- **Tables**: Saved to `results/tables/`
  - Performance metrics (RMSE, MAE)
  - Computational statistics
  - Comparative results

## Key Findings
*(To be updated after running the analysis)*

### PCA with Mean Filling
- Variance explained by top components
- Reconstruction error metrics
- Optimal number of components

### PCA with MLE
- Comparison with mean filling
- Imputation quality assessment
- Performance differences

### SVD Analysis
- Latent factor interpretation
- Collaborative filtering performance
- Cold-start analysis results
- Comparison with PCA approaches

## Methodology

### Data Preprocessing
1. Load ratings data
2. Create user-item matrix
3. Handle missing values (different strategies per notebook)
4. Normalize data if required

### Dimensionality Reduction
1. Apply PCA or SVD
2. Analyze component/singular value distributions
3. Select optimal dimensionality
4. Reconstruct ratings matrix

### Evaluation
1. Calculate RMSE on test set
2. Compare reconstruction quality
3. Analyze computational efficiency
4. Assess scalability

## Dependencies
All dependencies are listed in the main `requirements.txt`:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` when loading data
- **Solution**: Ensure data files are in the `data/` directory
- Check file paths in the notebooks

**Issue**: Memory errors with large datasets
- **Solution**: Reduce the dataset size or increase available RAM
- Consider using sparse matrix representations

**Issue**: Import errors for `utils`
- **Solution**: Ensure `utils.py` is in the same directory as notebooks
- Restart the Jupyter kernel

## References
- PCA documentation: scikit-learn PCA
- SVD theory and applications
- Collaborative filtering literature
- Course materials: AIE425

## Contact
For questions about this section, please refer to the main README or contact the group members.
