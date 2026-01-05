# **Group 16**

# Adham Mohmed Elsaied Elwakel 222100195,
# Samaa Khaled Eltaky 222100761,
# Habiba Ahmed Abdelnapy 222100471, 
# Youssef Hussieny 222101943

# collaborative.py

# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import modules
from collaborative import get_user_item_matrix
from hybrid import recommend_hybrid

print("="*60)
print("SECTION 2: INTELLIGENT RECOMMENDER SYSTEM (EXECUTION)")
print("="*60)

# 1. LOAD DATA & ARTIFACTS
print("\n[Main] Loading Data & Models...")
df = pd.read_csv('../data/preprocessed_data.csv')
df['item_id'] = df['item_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)

user_matrix = get_user_item_matrix(df)

try:
    hybrid_matrix = pd.read_csv('../results/final_hybrid_matrix.csv', index_col=0)
    hybrid_matrix.index = hybrid_matrix.index.astype(str)
    hybrid_matrix.columns = hybrid_matrix.columns.astype(str)
except FileNotFoundError:
    print("Error: Hybrid matrix not found. Run hybrid.py first.")
    exit()

# 2. COLD START SIMULATION (Step 10)
print("\n[Main] Running Cold-Start Simulation...")

def get_popular_recs(k=10, exclude=[]):
    pop_items = df['item_id'].value_counts().index.tolist()
    return [x for x in pop_items if x not in exclude][:k]

valid_users = user_matrix.index[user_matrix.gt(0).sum(axis=1) >= 15].tolist()
test_users = np.random.choice(valid_users, 50, replace=False)

logs = []
for n in [3, 5, 10]:
    h_hits, p_hits, total = 0, 0, 0
    for user in test_users:
        full = user_matrix.loc[user]
        items = full[full>0].index.tolist()
        
        known = items[:n]
        hidden = items[n:]
        if not hidden: continue
        
        # Simulate partial history
        known_dict = {i: full[i] for i in known}
        
        # We simulate recommend_hybrid logic here for partial history
        scores = pd.Series(0.0, index=hybrid_matrix.index)
        for item, rating in known_dict.items():
            if item in hybrid_matrix.index:
                scores += hybrid_matrix[item] * rating
        h_recs = scores.nlargest(10).index.tolist()
        
        p_recs = get_popular_recs(k=10, exclude=known)
        
        h_hits += len(set(h_recs).intersection(hidden))
        p_hits += len(set(p_recs).intersection(hidden))
        total += 10
        
    logs.append({
        'History': n,
        'Hybrid Prec': h_hits/total,
        'Pop Prec': p_hits/total
    })
    
print(pd.DataFrame(logs).round(4))

# 3. BASELINE COMPARISON (Step 11)
print("\n[Main] Running Baseline Comparison (Leave-One-Out)...")
model_metrics = {'Hybrid': 0, 'Popularity': 0}
eval_count = 0

for user in test_users:
    full = user_matrix.loc[user]
    items = full[full>0].index.tolist()
    target = items[-1]
    train = items[:-1]
    
    if target not in hybrid_matrix.index: continue
    
    # Hybrid
    h_recs = recommend_hybrid(user, user_matrix, hybrid_matrix, k=10, exclude=[target])
    if target in h_recs: model_metrics['Hybrid'] += 1
    
    # Pop
    p_recs = get_popular_recs(k=10, exclude=train)
    if target in p_recs: model_metrics['Popularity'] += 1
    
    eval_count += 1

print(f"Results (N={eval_count}):")
for m, hits in model_metrics.items():
    print(f"{m} Hit Rate: {hits/eval_count:.4f}")

print("\n[Main] Execution Complete.")