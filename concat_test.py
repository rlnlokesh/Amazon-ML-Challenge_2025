import pandas as pd
import ast
import numpy as np

# === CONFIG ===
input_csv = "D:/DA/AAA/embed_test_16.csv"     # path to your input file
output_csv = "test_embed_16.csv"   # path for saving output

# === READ CSV ===
df = pd.read_csv(input_csv)

# Ensure embeddings are parsed correctly (from string to list)
def parse_embedding(emb):
    if isinstance(emb, str):
        try:
            return np.array(ast.literal_eval(emb), dtype=float)
        except:
            return np.array([], dtype=float)
    elif isinstance(emb, list) or isinstance(emb, np.ndarray):
        return np.array(emb, dtype=float)
    else:
        return np.array([], dtype=float)

# Apply parsing
df["text_embedding"] = df["text_embedding"].apply(parse_embedding)
df["image_embedding"] = df["image_embedding"].apply(parse_embedding)

# === CONCATENATE ===
def concat_embeddings(row):
    if len(row["text_embedding"]) == 512 and len(row["image_embedding"]) == 512:
        return np.concatenate([row["text_embedding"], row["image_embedding"]]).tolist()
    else:
        return []

df["concatenate_embedding"] = df.apply(concat_embeddings, axis=1)

# === SAVE OUTPUT ===
df.to_csv(output_csv, index=False)

print(f"Saved output with concatenated embeddings to {output_csv}")
