import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregar JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print("CARREGADO")

# Extrair e segmentar documentos
# documents = []
# for page in data["pages"]:
#     for item in page["items"]:
#         if item["type"] in ["heading", "text"]:
#             documents.append({
#                 "content": item["md"],
#                 "metadata": {
#                     "page": page["page"],
#                     "type": item["type"],
#                     "value": item["value"]
#                 }
#             })
documents = data    
print("EXTRAIDO E SEGMENTADO")

# Gerar embeddings
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
contents = [doc["content"] for doc in documents]
embeddings = model.encode(contents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")  # Conversão necessária para o FAISS
print("EMBEDDINGS GERADOS")

# Criar índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "index.faiss")
print("FAISS INDEX CRIADO")

with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)
print("DOCUMENTOS SALVOS")

# Carregar modelo LLaMA
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Substitua pelo modelo desejado
tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForCausalLM.from_pretrained(model_name)
print("MODELO CARREGADO")

# Verifica se há GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_model = llama_model.to(device)
print(f"USANDO DISPOSITIVO: {device}")

# Função de recuperação
def retrieve_documents(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    print("DOCUMENTOS RECUPERADOS")
    return [documents[i] for i in indices[0]]

# Função de geração
def generate_response(query, retrieved_docs):
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    prompt = f"Contexto extraído do regulamento da UFRN (incluindo artigos e definições): \n{context}\n\n Com base exclusivamente nesse conteúdo, responda claramente à pergunta abaixo, citando o artigo se possível. \n\nPergunta: {query}\n\nResposta:"
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llama_model.generate(**inputs, max_new_tokens=300)
    print("RESPOSTA GERADA")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Testar
query = "O que é matriz curricular?"
retrieved_docs = retrieve_documents(query)
response = generate_response(query, retrieved_docs)
print(response)
