from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import re

# 1. Carrega o PDF
loader = PyPDFLoader("regulamento.pdf")
pages = loader.load()
full_text = "\n".join([page.page_content for page in pages])

# 2. Expressão regular para pegar cada artigo completo
pattern = r"(Art\. ?\d+º?.*?)(?=\nArt\. ?\d+º|\Z)"
artigos = re.findall(pattern, full_text, re.DOTALL)

# 3. Configura o splitter caso algum artigo seja muito grande
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
)

# 4. Divide e salva os chunks
chunks = []
for artigo in artigos:
    sub_chunks = splitter.split_text(artigo.strip())
    for c in sub_chunks:
        chunks.append({
            "content": c,
            "metadata": {
                "source_text": c
            }
        })

# 5. Exporta como JSON
with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# 6. Gera e salva os embeddings no FAISS
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
embeddings = model.encode([d["content"] for d in chunks], convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "index.faiss")

print(f"✅ Processado: {len(chunks)} chunks salvos.")
