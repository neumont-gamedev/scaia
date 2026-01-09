import warnings
warnings.filterwarnings("ignore")

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#from langchain.llms import Ollama
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

def load_prompt():
    with open("prompts/system.txt") as f:
        return f.read()

def format_source(doc):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")

    if page is not None:
        return f"{source}, page {page}"
    return source

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    system_prompt = load_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    llm = Ollama(
        model="llama3.1",
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        query = input("Student> ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa.invoke({"query": query})

        answer = result["result"]
        source_docs = result["source_documents"]

        print("\nAssistant>", answer, "\n")

        if source_docs:
            print("Sources:")
            seen = set()

            for doc in source_docs:
                label = format_source(doc)
                if label not in seen:
                    print(f"- {label}")
                    seen.add(label)

            print()

if __name__ == "__main__":
    main()
