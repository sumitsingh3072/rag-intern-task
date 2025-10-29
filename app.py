import streamlit as st
import os
import requests
import tempfile
from langchain_classic.chains.conversational_retrieval import base
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def get_embeddings_model():
    """Loads the BGE embedding model."""
    print("Loading embedding model...")
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cuda" if "cuda" in "available" else "cpu"} # Simple check
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

@st.cache_resource
def get_llm(api_key):
    """Initializes the ChatGroq LLM."""
    print("Initializing Groq LLM...")
    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    print("Groq LLM initialized.")
    return llm

def load_and_split_pdf(source: str, is_url: bool = False):
    """
    Loads a PDF from a URL or a local file path and splits it into chunks.
    """
    pdf_path = None
    try:
        if is_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                response = requests.get(source)
                response.raise_for_status()
                temp_f.write(response.content)
                pdf_path = temp_f.name
        else:
            # Source is a local file path (from tempfile)
            pdf_path = source

        if not pdf_path or not os.path.exists(pdf_path):
            st.error("Error: PDF file not found.")
            return None

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None
    finally:
        # Clean up the temp file if it was a URL download
        if is_url and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def setup_retriever(splits, embeddings):
    """Initializes the hybrid ("fuzzy") retriever."""
    if not splits:
        return None
    
    try:
        vectorstore = Chroma.from_documents(splits, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        doc_texts = [d.page_content for d in splits]
        bm25_retriever = BM25Retriever.from_texts(doc_texts)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5] # Give them equal weight
        )
        return ensemble_retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        return None


st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your PDF 💬")
st.markdown("Powered by Groq, LangChain, and Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please process a PDF in the sidebar to start chatting."}]
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


with st.sidebar:
    st.header("Configuration")
    st.subheader("PDF Source")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    pdf_url = st.text_input("Or enter a PDF URL")
    
    if st.button("Process PDF"):
        groq_api_key = GROQ_API_KEY

        # Validation
        if not groq_api_key:
            st.error("Groq API key not found. Please add it to your Streamlit secrets.")
        elif not uploaded_file and not pdf_url:
            st.error("Please upload a PDF or provide a URL.")
        elif uploaded_file and pdf_url:
            st.error("Please provide only one source: either an upload or a URL.")
        else:
            with st.spinner("Processing PDF... This may take a moment."):
                try:
                    # Load and split the PDF
                    splits = None
                    if uploaded_file:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                            temp_f.write(uploaded_file.getvalue())
                            temp_pdf_path = temp_f.name
                        
                        splits = load_and_split_pdf(temp_pdf_path, is_url=False)
                        os.remove(temp_pdf_path) # Clean up temp file
                    
                    elif pdf_url:
                        splits = load_and_split_pdf(pdf_url, is_url=True)

                    if not splits:
                        st.error("Failed to load and split the PDF. Please try again.")
                    else:
                        # Get cached embedding model
                        embeddings = get_embeddings_model()
                        
                        # Set up the retriever
                        retriever = setup_retriever(splits, embeddings)
                        
                        if retriever:
                            # Get cached LLM
                            llm = get_llm(groq_api_key)
                            
                            # Create the conversational chain
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", 
                                return_messages=True
                            )

                            st.session_state.qa_chain = base.ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                memory=memory,
                                verbose=False # Set to False for a clean UI
                            )
                            
                            st.session_state.messages = [{"role": "assistant", "content": "PDF processed successfully! How can I help you?"}]
                            st.success("PDF processed! You can now ask questions.")
                        else:
                            st.error("Failed to set up the document retriever.")
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if the RAG chain is ready
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the chain
                    # The chain automatically uses its internal memory
                    result = st.session_state.qa_chain.invoke({"question": prompt})
                    response = result["answer"]
                    st.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"An error occurred while generating a response: {e}")
    else:
        st.error("The PDF has not been processed. Please process a PDF in the sidebar.")

