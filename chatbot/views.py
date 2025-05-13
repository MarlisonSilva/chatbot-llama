from django.shortcuts import render
from django.http import JsonResponse
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# === Carregamentos únicos ao iniciar o servidor ===

# Carrega JSON com documentos (chunks)
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Carrega índice FAISS
index = faiss.read_index("index.faiss")

# Modelo de embeddings
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Modelo LLM (LLaMA 3)
llm_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = llm_model.to(device)

# === Views ===

def home(request):
    return render(request, "chat/index.html")

def ask(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return JsonResponse({"error": "Pergunta vazia."})

    # Embedding da pergunta
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Busca FAISS
    _, indices = index.search(query_embedding, 5)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Gera contexto para o modelo
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    prompt = f"""
Contexto extraído do regulamento da UFRN (incluindo artigos e definições): <br>
{context}

<br><br>Com base exclusivamente nesse conteúdo, responda claramente à pergunta abaixo, citando o artigo se possível.
Pergunta: {query}

<br><br>Resposta:
"""

    # Geração com LLaMA
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return JsonResponse({"response": response})
