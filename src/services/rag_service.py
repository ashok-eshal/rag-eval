"""RAG service for creating chains and generating responses"""

from typing import List, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.config.constants import RAG_PROMPT_TEMPLATE


def create_rag_chain(llm, vectorstore, num_chunks: int = 4, chat_history_str: str = ""):
    """Create RAG chain with conversation memory"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    if chat_history_str and chat_history_str.strip():
        memory.chat_memory.add_user_message("Previous conversation context")
        memory.chat_memory.add_ai_message(chat_history_str)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": num_chunks}
    )
    
    PROMPT = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return chain


def generate_rag_response(chain, question: str) -> Dict[str, Any]:
    """Generate response using RAG chain"""
    try:
        response = chain({"question": question})
        return {
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "source_documents": []
        }
