# import pandas as pd
# import os
#
# # Paths
# csv_path = "D:/DA/68e8d1d70b66d_student_resource/student_resource/dataset/test.csv"  # your original CSV
# image_folder = "images_test_all"  # folder where images are saved
# output_csv = "updated_test.csv"  # new CSV to save
#
# # Read CSV
# df = pd.read_csv(csv_path)
#
# # Create image path column
# df['image'] = df['sample_id'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))
#
# # Save new CSV
# df.to_csv(output_csv, index=False)




from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # progress bar

# Load model and data
model = SentenceTransformer('sentence-transformers/clip-ViT-B-16')
df = pd.read_csv('updated_test.csv')

print("Generating embeddings...")

# Generate text embeddings
text_embeddings = model.encode(df['catalog_content'].fillna(''), show_progress_bar=True)

# Generate image embeddings with progress
image_embeddings = []
print("Processing images...")
for i, img_path in enumerate(tqdm(df['image'], total=len(df))):
    try:
        if os.path.exists(img_path):
            image = Image.open(img_path)
            emb = model.encode([image])[0]
            image_embeddings.append(emb)
        else:
            image_embeddings.append(None)
    except Exception as e:
        image_embeddings.append(None)

print(f"Image embeddings generated for {sum(e is not None for e in image_embeddings)} / {len(df)} images")

# Create compact ML dataset
ml_data = []
for idx, row in df.iterrows():
    data = {
        'sample_id': row['sample_id'],
        'price': row['price'] if 'price' in df.columns else None,
        'text_embedding': text_embeddings[idx].tolist() if text_embeddings[idx] is not None else None,
        'image_embedding': image_embeddings[idx].tolist() if image_embeddings[idx] is not None else None,
        'catalog_content': row['catalog_content']
    }
    ml_data.append(data)

ml_df = pd.DataFrame(ml_data)
ml_df.to_csv('embed_test_16.csv', index=False)

print("Compact ML dataset saved successfully!")
