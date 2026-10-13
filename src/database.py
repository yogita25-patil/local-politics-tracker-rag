import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_database():
    print("1. Loading ALL PDFs from the data folder...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data")
    
    # Langchain automatically reads every single PDF in the folder!
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} total pages of documents!")
    
    print("2. Chopping text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split everything into {len(chunks)} chunks.")
    
    print("3. Loading AI Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("4. Saving to the Filing Cabinet (ChromaDB)...")
    db_path = os.path.join(current_dir, "../vector_db")
    
    # Overwrite the old database and save the massive new one!
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    print("✅ Database successfully built! You can now search all your files in the web app.")

if __name__ == "__main__":
    build_database()