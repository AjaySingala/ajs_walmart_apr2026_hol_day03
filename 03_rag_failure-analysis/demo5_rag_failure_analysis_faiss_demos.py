import uuid
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# COMMON SETUP
# -------------------------------

import sys
import os

folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config


def get_llm():
    print(f"\n get_llm()...")
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0
    )


def get_embeddings():
    print(f"\n get_embedding()...")
    return OpenAIEmbeddings(
        model=os.getenv("TEXT_EMBEDDING_MODEL")
    )


def format_docs(docs):
    print(f"\n format_docs()...")
    context = ""
    for d in docs:
        print(f"\nDocument: {d.page_content.strip()} (source: {d.metadata})")
        context += d.page_content + "\n\n"
    return context


# -------------------------------
# BASE DOCUMENTS
# -------------------------------

documents = [
    Document(
        page_content="Employees are entitled to 20 days of paid leave annually.",
        metadata={"source": "LeavePolicy"}
    ),
    Document(
        page_content="Travel reimbursement is allowed only for approved business trips.",
        metadata={"source": "TravelPolicy"}
    ),
    Document(
        page_content="""
The company provides employees with a leave policy.
Employees are eligible for annual leave and other types of leave as per company guidelines.
""",
        metadata={"source": "LeavePolicy"}
    ),
    Document(
        page_content="""
The organization supports employee well-being through various benefits and HR policies.
These include leave programs and employee support initiatives.
""",
        metadata={"source": "HRPolicy"}
    ),
    Document(
        page_content="""
Employees are eligible for structured leave programs.
These programs are designed to support employee needs across different situations.
""",
        metadata={"source": "LeavePolicy"}
    )
]

# =========================================================
# DEMO 1: BASELINE RAG
# =========================================================

def demo1_baseline():
    print("\n===== DEMO 1: BASELINE RAG =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")

    question = "How many leave days do employees get?"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(prompt.format(context=context, question=question))

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# DEMO 2: HALLUCINATION
# Always hallucinates.
# =========================================================

def demo2_hallucination():
    print("\n===== DEMO 2: HALLUCINATION =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""
# TODO: Create a prompt telling the LLM to be an HR Expert.
# Ask it to give the answer confidently.
# This is to see if it hallucinates "confidently".
# Pass the context and the question as placeholders to the prompt.

""")

    question = "What benefits are included in the leave policy?"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(prompt.format(context=context, question=question))

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# DEMO 3: LOW RELEVANCE
# Shows empty retrieval
# =========================================================
# TODO: Define a method named "demo3_low_relevance()".
# Implement the following:
#   - Create a vector store ofthe documents with embeddings.
#   - Retrieve the top 2 documents from the vector store.
#   - Create a prompt telling the LLM to answer only from the context provided.
#   - Also, tell it to sy "I don't know" if the answer is not present in the context.
#   - Pass the context and question as placeholders to the prompt.
#   - The question to ask is ""Explain company culture".
#   - Invoke the retriever with the question.
#   - Format the documents and create the context.
#   - Invoke the LLM passing it the formatted prompt with the context and question.
#   - print the Question and the Answer received from the LLM.



# ---------------- FIX: QUERY REWRITE ----------------

def rewrite_query(question):
    print(f"\n rewrite_query()...")
    rewrite_prompt = f"Rewrite this query for HR policy search:\n{question}"
    return get_llm().invoke(rewrite_prompt).content


def demo3_fix():
    print("\n===== DEMO 3 FIX: QUERY REWRITE =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    question = "Tell me about company culture"

    rewritten = rewrite_query(question)

    docs = retriever.invoke(rewritten)
    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nRewritten Query:", rewritten)
    print("Answer:", response.content)


# =========================================================
# DEMO 4: NOISE
# Clearly shows noisy context failure.
# =========================================================

demo4_documents = [
    # TODO: Define a list of around 7-8 documents related to leaves and leae policies.
    # Also include some non-leave related documents to create "noise".
    # Assign metadata to each document by setting a "source" property.
    # For e.g.; source="TravelPolicy" OR source="OperationsPolicy" etc.
]


def demo4_noise():
    print("\n===== DEMO 4: NOISY CONTEXT =====")

    vectorstore = FAISS.from_documents(
        demo4_documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = "Explain the leave policy"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# ---------------- FIX: MANUAL FILTER ----------------

def demo4_fix():
    print("\n===== DEMO 4 FIX: FILTERED RETRIEVAL =====")

    vectorstore = FAISS.from_documents(
        demo4_documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = "Explain the leave policy"

    docs = retriever.invoke(question)

    # Manual filtering (FAISS workaround)
    # TODO: Filter the documents to fetch only LeavePolicy documents.
    # Store in a variable "docs".


    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    demo1_baseline()
    demo2_hallucination()
    demo3_low_relevance()
    demo3_fix()

    demo4_noise()
    demo4_fix()
