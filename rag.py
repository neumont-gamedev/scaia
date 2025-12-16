from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    with open("prompts/system.txt") as f:
        system_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
        "Context:\n{context}\n\n"
        "Question:\n{question}")
    ])

    llm = Ollama(
        model="llama3.1",
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
