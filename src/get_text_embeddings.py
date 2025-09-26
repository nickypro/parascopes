from sentence_transformers import SentenceTransformer

# No need to add instruction for retrieval documents
documents = [

]
input_texts = queries + documents

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# [[91.92853546142578, 67.5802993774414], [70.38143157958984, 92.13307189941406]]

