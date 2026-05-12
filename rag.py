# @Author: Bhanu Prakash Sige.

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import requests
from bs4 import BeautifulSoup

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path("/tmp/vectorstore")
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def load_url(url):
    """
    Fetches and parses a URL using requests + BeautifulSoup.
    Returns a LangChain Document with page content and source metadata.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠️  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        print(f"  ⚠️  No content extracted from {url}")
        return None

    return Document(page_content=text, metadata={"source": url})


def process_urls(urls):
    """
    Scrapes data from URLs and stores it in a vector database.
    :param urls: list of input URLs
    """
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store... ✅"
    vector_store.reset_collection()

    yield "Loading data... ✅"
    data = []
    for url in urls:
        print(f"  Fetching: {url}")
        doc = load_url(url)
        if doc:
            data.append(doc)
            print(f"  Loaded {len(doc.page_content)} chars from {url}")

    if not data:
        yield "❌ No data loaded. Check URLs or network access."
        return

    yield "Splitting text into chunks... ✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(data)
    print(f"  Created {len(docs)} chunks")

    yield "Adding chunks to vector database... ✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    yield "Done adding docs to vector database. ✅"


def format_docs(docs):
    """Concatenate document page content into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(query):
    """
    Generates an answer for the given query using LCEL.
    Returns a tuple of (answer, sources).
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Call process_urls() first.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based strictly on the provided context.
        If the answer is not found in the context, say "I don't have enough information to answer that."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    # Extract unique sources from retrieved documents
    retrieved_docs = retriever.invoke(query)
    sources = ", ".join(
        set(
            doc.metadata.get("source", "")
            for doc in retrieved_docs
            if doc.metadata.get("source")
        )
    )

    return answer, sources


def debug_retrieval(query, k=4):
    """
    Prints retrieved document chunks for a given query.
    Useful for diagnosing empty or irrelevant answers.
    """
    if not vector_store:
        print("Vector store not initialized.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)

    print(f"\n--- Retrieved {len(retrieved_docs)} docs for query: '{query}' ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[Doc {i + 1}] source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content preview:\n{doc.page_content[:400]}")
    print("--- End of retrieved docs ---\n")


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    query = "Tell me what was the 30 year fixed mortgage rate along with the date?"

    # Uncomment to inspect retrieved chunks if answer is wrong/empty:
    # debug_retrieval(query)

    answer, sources = generate_answer(query)
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")
