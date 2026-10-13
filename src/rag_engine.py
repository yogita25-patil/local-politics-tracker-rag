import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load the secret API key from the .env file so Gemini works
load_dotenv()

# Helper function to combine our retrieved chunks into one block of text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(question):
    print(f"User Question: {question}")
    
    # 2. Wake up the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    # 3. Load the Filing Cabinet (ChromaDB)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "../vector_db")
    
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 4. Give the AI its strict instructions (The Prompt)
    system_prompt = (
        "You are a helpful civic assistant tracking local politics. "
        "Use ONLY the following pieces of retrieved context from city meeting notes to answer the question. "
        "If the answer is not in the context, do not make it up; just say that you don't know based on the document. "
        "Explain it simply so a normal citizen can understand.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Connect them using Modern LCEL (The Pipeline)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\nThinking...")
    
    # Run the AI Pipeline!
    answer = rag_chain.invoke(question)
    
    print("\n=== AI ANSWER ===")
    print(answer)
    print("=================\n")
    return answer  # <-- This must be indented inside the ask_question function!
    
if __name__ == "__main__":
    my_question = "What is the budget impact of this meeting?"
    ask_question(my_question)

