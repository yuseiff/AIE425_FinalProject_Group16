# Section 2: Domain-Specific Recommender System

## Overview
This section implements a comprehensive recommendation system for a fashion rental platform (Rent the Runway dataset). The system combines multiple recommendation approaches to provide personalized fashion item suggestions based on user preferences, item features, and collaborative patterns.

## Domain: Fashion Rental Platform
The recommendation system is designed specifically for fashion rental services, where users rent clothing items for special occasions. The system considers:
- User preferences and rental history
- Item attributes (category, size, color, brand, etc.)
- User reviews and ratings
- Collaborative patterns among similar users

## Contents

### Modules

#### 1. `data_preprocessing.py`
- **Purpose**: Load and preprocess the Rent the Runway dataset
- **Key Functions**:
  - Data loading from JSON format
  - Data cleaning and validation
  - Feature extraction and encoding
  - Train-test split preparation
  - Export preprocessed data to CSV

#### 2. `content_based.py`
- **Purpose**: Content-based recommendation using item features
- **Key Features**:
  - TF-IDF vectorization for text features
  - Feature engineering for categorical and numerical attributes
  - User profile construction based on rental history
  - Cosine similarity computation
  - Top-N recommendation generation
  - k-Nearest Neighbors (k-NN) implementation

#### 3. `collaborative.py`
- **Purpose**: Collaborative filtering using user-item interactions
- **Key Features**:
  - User-item rating matrix construction
  - Matrix factorization techniques
  - User-based collaborative filtering
  - Item-based collaborative filtering
  - Similarity computation (cosine, Pearson)
  - Prediction and recommendation generation

#### 4. `hybrid.py`
- **Purpose**: Hybrid recommendation combining multiple approaches
- **Key Features**:
  - Weighted combination of content-based and collaborative methods
  - Ensemble recommendation strategies
  - Score normalization and aggregation
  - Handling cold-start problems

#### 5. `main.py`
- **Purpose**: Main entry point and orchestration
- **Key Features**:
  - Pipeline execution
  - Configuration management
  - Model training and evaluation
  - Results generation and export

## Data

### Dataset
The `data/` directory contains:
- `renttherunway_final_data.json` - Raw fashion rental data with user reviews and item details
- `preprocessed_data.csv` - Cleaned and preprocessed data ready for modeling

### Data Schema
Key fields include:
- User information (user_id, demographics)
- Item attributes (item_id, category, size, color, brand)
- Rental history (rental_date, return_date)
- Reviews and ratings (rating, review_text, fit_feedback)

## Running the System

### Prerequisites
Make sure you have installed all requirements:
```bash
pip install -r ../requirements.txt
```

### Execution

#### Option 1: Run Complete Pipeline
```bash
cd SECTION2_DomainRecommender/code
python main.py
```

This will execute the full recommendation pipeline:
1. Data preprocessing
2. Content-based model training
3. Collaborative filtering model training
4. Hybrid model creation
5. Evaluation and results generation

#### Option 2: Run Individual Modules
```python
# Data preprocessing
python data_preprocessing.py

# Content-based recommendations
python content_based.py

# Collaborative filtering
python collaborative.py

# Hybrid approach
python hybrid.py
```

### Expected Outputs
The system generates:
- **Recommendations**: Top-N item recommendations for each user
- **Evaluation Metrics**: Precision, Recall, NDCG, Coverage
- **Results Files**: Saved to `results/` directory
  - `content_based_recommendations.csv`
  - `collaborative_recommendations.csv`
  - `hybrid_recommendations.csv`
  - `evaluation_metrics.json`

## Methodology

### 1. Data Preprocessing
- Load raw JSON data
- Clean missing values and outliers
- Extract and encode features
- Create user-item interaction matrix
- Split data for training and testing

### 2. Content-Based Filtering
- Extract item features (category, brand, color, size)
- Apply TF-IDF to textual descriptions
- Build user profiles from rental history
- Compute item-item similarity
- Generate recommendations based on profile matching

### 3. Collaborative Filtering
- Construct user-item rating matrix
- Apply matrix factorization (SVD, NMF)
- Compute user-user and item-item similarities
- Predict missing ratings
- Generate top-N recommendations

### 4. Hybrid Approach
- Combine content-based and collaborative scores
- Apply weighted averaging or ensemble methods
- Handle cold-start scenarios
- Optimize weights based on validation performance

### 5. Evaluation
- Split data into train/test sets
- Calculate precision@k, recall@k
- Compute NDCG (Normalized Discounted Cumulative Gain)
- Measure catalog coverage
- Analyze diversity of recommendations

## Key Features

### Handling Cold-Start Problem
- **New Users**: Use content-based recommendations based on initial preferences
- **New Items**: Leverage item features for content-based matching
- **Hybrid Fallback**: Switch between methods based on data availability

### Personalization
- User-specific feature weighting
- Temporal dynamics (seasonal trends)
- Contextual recommendations (occasion, season)

### Scalability
- Efficient matrix operations using sparse representations
- Batch processing for large user bases
- Incremental model updates

## Dependencies
All dependencies are listed in the main `requirements.txt`:
- numpy - Numerical computations
- pandas - Data manipulation
- scikit-learn - Machine learning algorithms
- scipy - Scientific computing

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` when loading data
- **Solution**: Ensure `renttherunway_final_data.json` is in the `data/` directory
- Check file paths in the scripts

**Issue**: Memory errors with large dataset
- **Solution**: Use sparse matrix representations
- Process data in batches
- Reduce feature dimensionality

**Issue**: Poor recommendation quality
- **Solution**: Tune hyperparameters (k for k-NN, number of factors for matrix factorization)
- Adjust hybrid weights
- Increase training data size

**Issue**: Import errors
- **Solution**: Ensure all modules are in the `code/` directory
- Check Python path configuration
- Reinstall requirements

## Performance Metrics
*(To be updated after running the analysis)*

### Content-Based
- Precision@10
- Recall@10
- Coverage

### Collaborative Filtering
- RMSE on test ratings
- Precision@10
- Recall@10

### Hybrid
- Combined metrics
- Improvement over individual methods
- Cold-start performance

## References
- Rent the Runway Dataset
- Collaborative Filtering Techniques
- Content-Based Recommendation Systems
- Hybrid Recommendation Approaches
- Course materials: AIE425

