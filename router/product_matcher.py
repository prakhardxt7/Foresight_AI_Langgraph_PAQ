# File: router/product_matcher.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# === Parameters ===
TOP_N = 3
NYKAA_DATA_PATH = "data/Nykaa_Enriched_Dataset.csv"
COMPETITOR_DATA_PATH = "data/Competitor_Dataset.csv"
OUTPUT_PATH = "data/matched_products.csv"

# === Load datasets ===
print("📥 Loading datasets...")
nykaa_df = pd.read_csv(NYKAA_DATA_PATH)
competitor_df = pd.read_csv(COMPETITOR_DATA_PATH)

# === Generate descriptions for semantic comparison ===
nykaa_df['description'] = nykaa_df['Product_Name'] + ' - ' + nykaa_df['Category'] + ' - ' + nykaa_df['Sub_Category']
competitor_df['description'] = competitor_df['Product_Name'] + ' - ' + competitor_df['Category'] + ' - ' + competitor_df['Sub_Category']

# === Deduplicate by Product_ID ===
nykaa_unique = nykaa_df[['Product_ID', 'Product_Name', 'description']].drop_duplicates(subset="Product_ID").reset_index(drop=True)
competitor_unique = competitor_df[['Product_ID', 'Product_Name', 'description']].drop_duplicates(subset="Product_ID").reset_index(drop=True)

# === Load transformer ===
print("🔍 Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Encode descriptions ===
print("🔗 Encoding product descriptions...")
nykaa_embeddings = model.encode(nykaa_unique['description'].tolist(), convert_to_tensor=True)
competitor_embeddings = model.encode(competitor_unique['description'].tolist(), convert_to_tensor=True)

# === Compute similarity matrix ===
print("⚖️ Computing cosine similarity...")
similarity_matrix = util.pytorch_cos_sim(nykaa_embeddings, competitor_embeddings)

# === Match top-N competitors for each Nykaa product ===
print(f"🧠 Finding Top-{TOP_N} matches for each Nykaa product...")
matches = []
for idx, (_, row) in enumerate(nykaa_unique.iterrows()):
    top_indices = torch.topk(similarity_matrix[idx], k=TOP_N).indices.tolist()
    top_scores = torch.topk(similarity_matrix[idx], k=TOP_N).values.tolist()
    
    for i, comp_idx in enumerate(top_indices):
        comp_row = competitor_unique.iloc[comp_idx]  # ✅ FIXED: comp_idx is already int
        matches.append({
            "Nykaa_Product_ID": row['Product_ID'],
            "Nykaa_Product_Name": row['Product_Name'],
            "Competitor_Rank": i + 1,
            "Competitor_Product_ID": comp_row['Product_ID'],
            "Competitor_Product_Name": comp_row['Product_Name'],
            "Similarity_Score": round(top_scores[i], 4)
        })

# === Save output ===
matched_df = pd.DataFrame(matches)
os.makedirs("data", exist_ok=True)
matched_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Matching complete! Saved to {OUTPUT_PATH}")
