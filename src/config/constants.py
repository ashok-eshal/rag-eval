"""Constants for the LLM Response Evaluation System"""

# Default collection name
DEFAULT_COLLECTION = "rag_collection_eerrur_8415"

# Common Base URLs for OpenAI-compatible APIs
BASE_URLS = {
    "DeepInfra": "https://api.deepinfra.com/v1/openai"
}

# Common Providers
PROVIDERS = [
    "DeepInfra"
]

# Embedding model configurations
EMBEDDING_MODELS = {
    "OpenAI": {
        "text-embedding-3-large": {
            "name": "text-embedding-3-large",
            "dimensions": 3072,
            "description": "Highest quality OpenAI embeddings",
            "requires_api_key": True
        },
        "text-embedding-3-small": {
            "name": "text-embedding-3-small", 
            "dimensions": 1536,
            "description": "Cost-effective OpenAI embeddings",
            "requires_api_key": True
        }
    },
    "OpenAI-Compatible": {
        "Qwen/Qwen3-Embedding-8B": {
            "name": "Qwen/Qwen3-Embedding-8B",
            "dimensions": 4096,
            "description": "Qwen3 8B embeddings via OpenAI-compatible API (e.g., DeepInfra)",
            "requires_api_key": True
        }
    },
    "HuggingFace": {
        "BAAI/bge-large-en-v1.5": {
            "name": "BAAI/bge-large-en-v1.5",
            "dimensions": 1024,
            "description": "High quality open-source embeddings",
            "requires_api_key": False
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "description": "Fast, lightweight embeddings",
            "requires_api_key": False
        },
        "intfloat/e5-large-v2": {
            "name": "intfloat/e5-large-v2",
            "dimensions": 1024,
            "description": "State-of-the-art open-source embeddings",
            "requires_api_key": False
        }
    },
    "SentenceTransformers": {
        "all-MiniLM-L6-v2": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "description": "Fast local embeddings",
            "requires_api_key": False
        },
        "all-mpnet-base-v2": {
            "name": "all-mpnet-base-v2",
            "dimensions": 768,
            "description": "High quality local embeddings",
            "requires_api_key": False
        }
    },
    "Ollama": {
        "llama2": {
            "name": "llama2",
            "dimensions": 4096,
            "description": "Llama 2 embeddings via Ollama",
            "requires_api_key": False,
            "requires_ollama": True
        },
        "mistral": {
            "name": "mistral",
            "dimensions": 4096,
            "description": "Mistral embeddings via Ollama",
            "requires_api_key": False,
            "requires_ollama": True
        }
    }
}

# Default evaluation metrics
DEFAULT_METRICS = {
    "answer_relevancy": {
        "name": "Answer Relevancy",
        "description": "Determines whether an LLM output is able to address the given input in an informative and concise manner. The response should directly answer the question without unnecessary information."
    },
    "task_completion": {
        "name": "Task Completion",
        "description": "Determines whether an LLM agent is able to complete the task it was set out to do. Evaluate if all aspects of the requested task have been addressed."
    },
    "correctness": {
        "name": "Correctness",
        "description": "Determines whether an LLM output is factually correct based on the provided ground truth. Compare the response against the ground truth for accuracy."
    },
    "hallucination": {
        "name": "Hallucination",
        "description": "Determines whether an LLM output contains fake or made-up information not supported by the context or ground truth. Check for fabricated facts or unsupported claims."
    },
    "contextual_relevancy": {
        "name": "Contextual Relevancy",
        "description": "Determines whether the retriever in a RAG-based LLM system is able to extract the most relevant information for your LLM as context. Evaluate if the retrieved chunks are pertinent to the question."
    },
    "responsible_metrics": {
        "name": "Responsible Metrics",
        "description": "Includes metrics such as bias and toxicity, which determines whether an LLM output contains harmful, offensive, or biased content."
    },
    "task_specific": {
        "name": "Task-Specific Metrics",
        "description": "Includes metrics such as summarization quality, format adherence, or other custom criteria depending on the specific use-case."
    },
    "engagement": {
        "name": "Engagement / User Satisfaction",
        "description": "Evaluates the usefulness, engaging nature, and overall user satisfaction potential of the response. Consider clarity, helpfulness, user-friendliness, and how well it maintains conversation flow with the chat history."
    }
}

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """To generate your answer: 
 - Carefully analyze the question and identify the key information needed to address it 
 - Locate the specific parts of each context that contain this key information 
 - Concisely summarize the relevant information from the  context(s) in your own words 
 - Provide a direct answer to the question 
 - Give detailed and accurate responses for things like 'write a blog' or long-form questions. 
 - For greeting messages, please greet the user appropriately. 
 - Please refrain from inventing responses. 
 - If the `Question` is not related to the provided `Context` and `Chat History`  then kindly RESPOND with 'I'm sorry, but that topic is beyond what I currently know.'. 
Use the following context to answer the question: 
 ------ 

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
