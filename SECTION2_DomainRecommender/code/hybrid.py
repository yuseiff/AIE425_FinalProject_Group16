# **Group 16**

# Adham Mohmed Elsaied Elwakel 222100195,
# Samaa Khaled Eltaky 222100761,
# Habiba Ahmed Abdelnapy 222100471, 
# Youssef Hussieny 222101943


# hybrid.py
import pandas as pd
import numpy as np

def build_hybrid_matrix(sim_cb_df, sim_cf_df, alpha=0.5):
    """
    Weighted Combination: Score = alpha*CB + (1-alpha)*CF
    """
    print(f"[Hybrid] Aligning matrices and combining (Alpha={alpha})...")
    
    # Find common items
    common = sim_cb_df.index.intersection(sim_cf_df.index)
    print(f"[Hybrid] Common Items: {len(common)}")
    
    # Align
    cb_aligned = sim_cb_df.loc[common, common]
    cf_aligned = sim_cf_df.loc[common, common]
    
    # Combine
    hybrid_sim = (alpha * cb_aligned) + ((1 - alpha) * cf_aligned)
    
    return hybrid_sim

def recommend_hybrid(user_id, user_item_matrix, hybrid_matrix, k=10, exclude=[]):
    if user_id not in user_item_matrix.index: return []
    
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.intersection(hybrid_matrix.index)
    
    if len(rated_items) == 0: return []
    
    scores = pd.Series(0.0, index=hybrid_matrix.index)
    for item in rated_items:
        rating = user_ratings[item]
        scores += hybrid_matrix[item] * rating
        
    scores = scores.drop(exclude + rated_items.tolist(), errors='ignore')
    return scores.nlargest(k).index.tolist()

if __name__ == "__main__":
    # Load Matrices
    sim_cb = pd.read_csv('../results/item_similarity.csv', index_col=0)
    sim_cf = pd.read_csv('../results/item_similarity_cf.csv', index_col=0)
    
    # Ensure String IDs
    sim_cb.index = sim_cb.index.astype(str)
    sim_cb.columns = sim_cb.columns.astype(str)
    sim_cf.index = sim_cf.index.astype(str)
    sim_cf.columns = sim_cf.columns.astype(str)
    
    hybrid_matrix = build_hybrid_matrix(sim_cb, sim_cf, alpha=0.5)
    hybrid_matrix.to_csv('../results/final_hybrid_matrix.csv')
    print("[Hybrid] Final Matrix saved.")