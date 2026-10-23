from utils.embedding_utils import get_bge_m3_ef

model = get_bge_m3_ef()

docs_embedding = model.encode_documents(["hello world", "hello world"])

print(docs_embedding)