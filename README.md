# AIE425 Final Project - Group 16

## Project Overview
This project implements advanced recommendation systems using dimensionality reduction techniques and domain-specific approaches. The project is divided into two main sections:
- **Section 1**: Dimensionality reduction techniques (PCA, SVD) for collaborative filtering
- **Section 2**: Domain-specific recommendation system for fashion rental platform

## Project Structure
```
AIE425_FinalProject_Group16/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── SECTION1_DimensionalityReduction/  # Section 1: Dimensionality Reduction
│   ├── data/                          # MovieLens dataset
│   │   ├── ratings.csv               # User-movie ratings
│   │   └── movies.csv                # Movie metadata
│   ├── code/                          # Implementation notebooks
│   │   ├── pca_mean_filling.ipynb    # PCA with mean filling
│   │   ├── pca_mle.ipynb             # PCA with MLE imputation
│   │   ├── svd_analysis.ipynb        # SVD collaborative filtering
│   │   └── utils.py                  # Utility functions
│   ├── results/                       # Output directory
│   │   ├── plots/                    # Generated plots
│   │   └── tables/                   # Result tables
│   └── README_SECTION1.md            # Section 1 documentation
│
└── SECTION2_DomainRecommender/        # Section 2: Fashion Recommender
    ├── data/                          # Fashion rental dataset
    │   ├── renttherunway_final_data.json  # Raw data
    │   └── preprocessed_data.csv      # Preprocessed data
    ├── code/                          # Implementation modules
    │   ├── data_preprocessing.py      # Data preprocessing
    │   ├── content_based.py           # Content-based filtering
    │   ├── collaborative.py           # Collaborative filtering
    │   ├── hybrid.py                  # Hybrid approach
    │   └── main.py                    # Main entry point
    ├── results/                       # Output results
    └── README_SECTION2.md            # Section 2 documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup
1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Section 1: Dimensionality Reduction
Navigate to `SECTION1_DimensionalityReduction/` and refer to `README_SECTION1.md` for detailed instructions.

Quick start:
```bash
cd SECTION1_DimensionalityReduction/code
jupyter notebook
```

Then open and run the notebooks in the following order:
1. `pca_mean_filling.ipynb` - PCA with mean value imputation
2. `pca_mle.ipynb` - PCA with MLE-based imputation
3. `svd_analysis.ipynb` - SVD-based collaborative filtering

### Section 2: Domain Recommender
Navigate to `SECTION2_DomainRecommender/` and refer to `README_SECTION2.md` for detailed instructions.

Quick start:
```bash
cd SECTION2_DomainRecommender/code
python main.py
```

## Results
- **Section 1**: Plots and tables are saved in `SECTION1_DimensionalityReduction/results/`
- **Section 2**: Recommendation results are saved in `SECTION2_DomainRecommender/results/`

## Contributors
**Group 16**

Adham Mohmed Elsaied Elwakel 222100195,
Samaa Khaled Eltaky 222100761,
Habiba Ahmed Abdelnapy 222100471, 
Youssef Hussieny 222101943

## License
This project is submitted as part of AIE425 course requirements.


