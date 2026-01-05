# **Group 16**

# Adham Mohmed Elsaied Elwakel 222100195,
# Samaa Khaled Eltaky 222100761,
# Habiba Ahmed Abdelnapy 222100471, 
# Youssef Husseiny 222101943

# data_preprocessing.py
import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """
    Loads raw JSON/CSV data and performs initial cleaning.
    """
    print(f"[Preprocessing] Loading data from {file_path}...")
    # Assuming standard RentTheRunway format
    try:
        df = pd.read_json(file_path, lines=True)
    except ValueError:
        df = pd.read_csv(file_path)
    
    # Basic Cleaning
    df = df.dropna(subset=['rating', 'rented for', 'category'])
    
    # Feature Engineering (Example: Weight/Height processing would go here)
    if 'weight' in df.columns:
        df['weight'] = df['weight'].astype(str).str.replace('lbs', '').astype(float)
        
    # Standardize IDs
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    
    print(f"[Preprocessing] Cleaned Data Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage for testing
    df = load_and_clean_data('../data/renttherunway_final_data.json')
    df.to_csv('../data/preprocessed_data.csv', index=False)
    print("[Preprocessing] Saved to ../data/preprocessed_data.csv")