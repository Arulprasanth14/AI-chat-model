from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

embedding = model.encode(["What is the leave policy?"])

print("Shape:", embedding.shape)
print("Embedding:", embedding[0][:5])