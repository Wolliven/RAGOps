from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Users can upload documents",
    "The platform supports file uploads",
    "I like cooking pasta"
]

embeddings = model.encode(sentences, convert_to_tensor=True)

similarity_0_1 = cos_sim(embeddings[0], embeddings[1])
similarity_0_2 = cos_sim(embeddings[0], embeddings[2])

print("upload vs file upload:", similarity_0_1.item())
print("upload vs pasta:", similarity_0_2.item())