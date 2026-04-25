import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import faiss

# ================================
# SETUP
# ================================
load_dotenv()

# Add config path (same as your original)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)
import config

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# TODO: Extract the TEXT_EMBEDDING_MODEL environment variable value and store it in a variable EMBED_MODEL. 


# TODO: Extract the MODEL_NAME environment variable value and store it in a variable CHAT_MODEL. 



# ================================
# GLOBAL VECTOR STORE (FAISS)
# ================================
class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.metadatas = []

    def add(self, embeddings, docs, metadatas):
        vectors = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(vectors)
        self.documents.extend(docs)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding, k=3, where=None):
        q = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(q, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                if where:
                    # metadata filter
                    key, value = list(where.items())[0]
                    if self.metadatas[idx].get(key) != value:
                        continue
                results.append(self.documents[idx])

        return results

vector_store = VectorStore()

# TODO: Create a method "get_all_documents" that returns all the documents from the vector store.


# TODO: Create a method "get_all_documents_with_metadata" that returns all the documents from the vector store
# along with the document metadata.


# ================================
# EMBEDDING
# ================================
def embed_texts(texts):
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return [d.embedding for d in response.data]

# ================================
# PDF LOADER
# ================================
def load_pdf(file_path):
    # TODO: Read the file at the given path and return the contents as text.


    return text

# ================================
# CHUNKING
# ================================
def chunk_text(text, chunk_size=200, overlap=50):
    print("\n chunk_text()...")
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ================================
# INDEX PDFs
# ================================
def index_pdfs():
    print("\n index_pdfs()...")

    files = [
        ("documents/leave_policy.pdf", "LeavePolicy"),
        ("documents/travel_policy.pdf", "TravelPolicy")
    ]

    all_chunks = []
    all_metadata = []

    for file_path, category in files:
        # TODO: Read the file using the load_pdf() method into a variable "text".


        # TODO: Pass the text read from the file above and use chunk_test() to create chunks.
        # Store the result in a variable "chunks".


        for chunk in chunks:
            # TODO: Append the chunks to the all_chunks collection.


            # TODO: Append the category of the file to the all_metadata collection.
            # Store it as a dictionary with the key as "source" and value as the categeory.


    print(f"Total chunks: {len(all_chunks)}")

    # Embed
    embeddings = embed_texts(all_chunks)

    # TODO: Add the embeddings, all the chunks and all the metadata to the vector store. 


    print("Indexing complete.")

# ================================
# RETRIEVAL
# ================================
def retrieve(query, top_k=3, where=None):
    print(f"\n retrieve(query={query}, top_k={top_k}, where={where})...")
    query_emb = embed_texts([query])[0]
    return vector_store.search(query_emb, k=top_k, where=where)

def retrieve_with_scores(query, top_k=3, where=None):
    query_emb = embed_texts([query])[0]

    q = np.array([query_emb]).astype("float32")
    distances, indices = vector_store.index.search(q, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(vector_store.documents):
            if where:
                key, value = list(where.items())[0]
                if vector_store.metadatas[idx].get(key) != value:
                    continue

            similarity = 1 / (1 + dist)
            results.append((similarity, vector_store.documents[idx]))

    return results

def print_header(title):
    print('=' * 50)
    print(f"{title}")
    print('=' * 50)

# ================================
# DEMOS (UNCHANGED LOGIC)
# ================================
QUERY = "What are the rules for employee travel reimbursement and approvals?"

# ================================
# DEMO 1: BASELINE
# ================================
def demo1():
    print_header("demo1() - BASELINE")
    # TODO: Retrieve the documents using the retrieve() function.
    # Retrieve only the top 2 documents.
    # Store the results in a variable "docs"


    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question: {QUERY}
"""

    res = client.chat.completions.create(
        # TODO: Call the OpenAI client with the CHAT_MODEL and the user message.
        # Pass the prompt in the user message.
        # Set the temperature to 0. 
    )

    print("\n--- DEMO 1 ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 2: RERANK
# ================================
def demo2():
    print_header("demo2() - RERANK")
    docs = retrieve(QUERY, top_k=5)

    query_emb = embed_texts([QUERY])[0]

    scored = []
    for doc in docs:
        doc_emb = embed_texts([doc])[0]
        score = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        scored.append((score, doc))

    scored.sort(reverse=True)
    docs = [doc for _, doc in scored[:2]]

    context = "\n".join(docs)

    res = client.chat.completions.create(
        # TODO: Call the OpenAI client with the CHAT_MODEL and the user message.
        # Pass the prompt in the user message.
        # Also pass the context in the user message.
        # Set the temperature to 0. 
    )

    print("\n--- DEMO 2 (RERANK) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 3: QUERY REWRITE
# ================================
def demo3():
    print_header("demo3() - QUERY REWRITE")

    rewrite = client.chat.completions.create(
        # TODO: Call the OpenAI client with the CHAT_MODEL and the user message.
        # Ask the model to rewrite the given QUERY.
        # Set the temperature to 0. 
    ).choices[0].message.content

    docs = retrieve(rewrite, top_k=3)

    context = "\n".join(docs)

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 3 (QUERY REWRITE) ---")
    print("Original:", QUERY)
    print("Rewrite:", rewrite)
    print(f"\n Result:\n {res.choices[0].message.content}")


# ================================
# DEMO 4: METADATA FILTER
# ================================
def demo4(source="TravelPolicy"):
    print_header(f"demo4(source={source}) - METADATA FILTER")
    print(f"\n demo4(source={source})...")

    # TODO: Retrieve the top 3 documents for the given QUERY and filter them by the given source.
    # Use the retrieve function.
    # Store the results in a variable "docs" 


    context = "\n".join(docs)
    print(f"\n Context: {context}")

    # For "LeavePolicy", it still returns "travel" info because the context is empty.
    # And we are not enforcing "answer only from context".
    # Hallucination?
    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 4 (FILTER) ---")
    print(res.choices[0].message.content)

def demo4_improved(source="TravelPolicy"):
    print_header(f"demo4_improved(source={source}) - IMPROVED METADATA FILTER")
    # TODO: Retrieve the top 3 documents with their scores for the given QUERY and 
    # filter them by the given source.
    # Use the appropriate function to retrieve the documents with their respective scores.
    # Store the results in a variable "scored_docs" 


    print(f"\n Scored Docs:: {scored_docs}")

    # KEY FIX: Filter by relevance threshold
    # TODOL Filter the documents scored documents and extract only those that have a score of 
    # more than 0.60.
    # Store results in a variable "filtered_docs".


    print(f"\n Filtered Docs: {filtered_docs}")

    if not filtered_docs:
        context = ""
    else:
        context = "\n".join(filtered_docs)

    prompt = f"""
Answer ONLY from the context.
If the context is empty or not relevant, say "I don't know".

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 4 (FILTER FIXED) ---")
    print("Answer:", response.choices[0].message.content)

# ================================
# DEMO 5: STRUCTURED CONTEXT
# ================================
def demo5():
    print_header(f"demo5() - STRUCTURED CONTENT")
    docs = retrieve(QUERY, top_k=3)

    context = ""
    for i, d in enumerate(docs, 1):
        context += f"[Source {i}]: {d}\n"
    print(f"\n Context: {context}")

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 5 (STRUCTURED) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 6: PROMPT IMPROVEMENT
# ================================
def demo6(query: str):
    print_header(f"demo6() - PROMPT IMPROVEMENT")
    print(f"\n demo6()...")
    docs = retrieve(query, top_k=3)
    context = "\n".join(docs)

    prompt = f"""
# TODO: Create a prompt that tells the LLM to answer only from the given context.
# Also give the following instructions:
#   - ask it to be structured.
#   - If it is not sure, it should say "I don't know".
# Pass then context and the query as placeholders.
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 6 (PROMPT) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 7: HALLUCINATION
# ================================
def demo7():
    print_header(f"demo7() - HALLUCINATION")
    docs = retrieve(QUERY, top_k=1)
    context = "\n".join(docs)

    prompt = f"""
Answer using your knowledge also.

Context:
{context}

Question: {QUERY}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    print("\n--- DEMO 7 (HALLUCINATION) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 8: NO DATA
# ================================
def demo8():
    print_header(f"demo8() - NO DATA")
    query = "What is Walmart's stock trading strategy?"
    print(f"\n Query = {query}")

    docs = retrieve(query, top_k=2)
    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from context.
If not found say "I don't know"

Context:
{context}

Q: {query}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 8 (NO DATA) ---")
    print(res.choices[0].message.content)


# ================================
# RUN
# ================================
if __name__ == "__main__":
    index_pdfs()

    all_docs = get_all_documents()
    print(f"\n All docs:\n {all_docs}")
    all_docs = get_all_documents_with_metadata()
    print(f"\n All docs with metadata:\n {all_docs}")

    demo1()
    demo2()
    demo3()
    demo4()
    demo4("LeavePolicy")
    demo4_improved("TravelPolicy")
    demo4_improved("LeavePolicy")
    demo5()
    demo6(QUERY)
    demo6("Who won the last FIFA World Cup?")
    demo7()
    demo8()