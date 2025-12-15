import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

DATA_PATH = "data/raw"
VECTORSTORE_PATH = "vectorstore"

TEXT_EXTENSIONS = (
    ".md", ".txt",
    ".cpp", ".h",
    ".cs", ".js",
    ".py", ".glsl"
)

def load_documents():
    documents = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)

            elif file.endswith(TEXT_EXTENSIONS):
                loader = TextLoader(path, encoding="utf-8")

            else:
                continue

            docs = loader.load()
            for d in docs:
                d.metadata["source"] = path
            documents.extend(docs)

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    vectordb.persist()

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    print("Ingestion complete.")
