import time
from langchain_community.vectorstores import FAISS


# Load documents
# TODO: Load the data using the load_data() method in common_setup_faiss.py



def print_divider(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


# --------------------------------------------------
# Demo 1 — Baseline RAG
# --------------------------------------------------
def demo_1_baseline():
    print_divider("Demo 1 — Baseline RAG (Top-k = 4)")

    query = "What is the leave policy?"

    retriever = common_setup_faiss.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below:

    {context}

    Question: {query}
    """

    response = common_setup_faiss.llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs:")
    for d in docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 2 — Top-k Tuning
# --------------------------------------------------
def demo_2_topk():
    print_divider("Demo 2 — Top-k Tuning (k = 2)")

    query = "What is the leave policy?"

    retriever = common_setup_faiss.vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = common_setup_faiss.llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs:")
    for d in docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 3 — Relevance Scores
# --------------------------------------------------
def demo_3_scores():
    print_divider("Demo 3 — Relevance Scores")

    query = "What is the leave policy?"

    results = common_setup_faiss.vectorstore.similarity_search_with_score(query, k=4)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs with Scores:")
    for doc, score in results:
        print(f"Score: {round(score, 4)} | {doc.page_content}")


# --------------------------------------------------
# Demo 4 — Threshold Filtering
# --------------------------------------------------
def demo_4_threshold():
    print_divider("Demo 4 — Threshold Filtering")

    query = "What is the leave policy?"

    # TODO: Search the vector store for the given query and retrieve the top 5 documents.
    # Perform a similarity search with score on the vector store.
    # The method can be found in common_setup_faiss.
    # Store the results in a variable "results".



    original_context = ""
    for doc, score in results:
        original_context += doc.page_content + f" (Score: {score})\n"

    # TODO: Define a threshold to filter the documents even further.'
    # Start with a threshold value of 0.5.
    # Keep fine tuning this value to see the results.
    # Filter the doucments in the "results" list and fetch only those docs that have
    # a score less than the threshold value.
    # Store the results in a variable named "filtered_docs".



    context = "\n".join([doc.page_content for doc in filtered_docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = common_setup_faiss.llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\n--- Original Context ---")
    print(original_context)

    print("\nFiltered Docs:")
    for d in filtered_docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 5 — Context Optimization
# --------------------------------------------------
def demo_5_context_limit():
    print_divider("Demo 5 — Context Window Optimization")

    query = "Explain leave rules"

    results = common_setup_faiss.vectorstore.similarity_search_with_score(query, k=5)

    original_docs = ""
    for doc, score in results:
        original_docs += doc.page_content + f" (Score: {score})\n"

    # Sort by relevance (higher score = better in FAISS)
    sorted_docs = sorted(results, key=lambda x: x[1], reverse=True)

    max_chars = 50
    context = ""

    for doc, score in sorted_docs:
        if len(context) + len(doc.page_content) < max_chars:
            context += doc.page_content + "\n"

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = common_setup_faiss.llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\n--- Original Docs ---")
    print(original_docs)

    print("\nOptimized Context:")
    print(context)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 6 — Metadata Filtering (Manual)
# --------------------------------------------------
def demo_6_metadata_filter():
    print_divider("Demo 6 — Metadata Filtering")

    query = "What is the leave policy?"

    docs = common_setup_faiss.vectorstore.similarity_search(query, k=4)

    # FAISS does NOT support native metadata filtering → manual filter
    # TODO: Filter the documents only for those documents that have the source "LeavePolicy".
    # Store the results in a variable "filtered_docs".


    context = "\n".join([doc.page_content for doc in filtered_docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = common_setup_faiss.llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nFiltered Docs:")
    for d in filtered_docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Run all demos
# --------------------------------------------------
if __name__ == "__main__":
    demo_1_baseline()
    time.sleep(2)

    demo_2_topk()
    time.sleep(2)

    demo_3_scores()
    time.sleep(2)

    demo_4_threshold()
    time.sleep(2)

    demo_5_context_limit()
    time.sleep(2)

    demo_6_metadata_filter()
