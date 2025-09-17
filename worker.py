import os
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# For GGUF (local file) via CTransformers:
from langchain_community.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings  # Or alternative

# --- For MistralAI API-based inference (uncomment if using web API) ---
# from mistralai.client import MistralClient
# from langchain_community.llms import MistralAIEmbeddings

load_dotenv()


conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None


def init_llm():
    global llm, llm_embeddings

    # LOCAL MISTRAL 7B (GGUF format using CTransformers)
    llm = CTransformers(
        model="models/mistral-7b-v0.1.Q4_K_M.gguf",  # Update with your path
        model_type="llama",
        max_new_tokens=256,
        temperature=0.7
    )
    # Embeddings: HuggingFace or sentence-transformers models (choose one):
    llm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def process_document(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, llm_embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    # Append prompt and response to chat history
    chat_history.append((prompt, result['answer']))
    return result['answer']

init_llm()
