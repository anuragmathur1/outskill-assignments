# Before running the Python code below, you need to start Qdrant.
# Open your terminal or command prompt and run the following Docker command:
# docker run -p 6333:6333 -p 6334:6334 \
#     -v $(pwd)/qdrant_storage:/qdrant/storage \
#     qdrant/qdrant

import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import google.generativeai as genai
import os
import dotenv
import uuid # For unique IDs for Qdrant points

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="RAG Document Q&A App", layout="wide")


# --- Configuration and Setup ---

# Load environment variables (for API keys)
dotenv.load_dotenv()

# Configure Google Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Qdrant Collection Name
COLLECTION_NAME = "my_rag_documents"

# --- Cache expensive resources ---
# Use st.cache_resource to initialize models and clients only once
@st.cache_resource
def get_embedding_model():
    """Loads and caches the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_qdrant_client():
    """Initializes and caches the Qdrant client."""
    client = QdrantClient(host="localhost", port=6333)
    return client

@st.cache_resource
def get_gemini_model(model_name='models/gemini-1.5-pro'):
    """Initializes and caches the Google Gemini GenerativeModel."""
    try:
        # --- DEBUGGING: Print all available models ---
        print("--- Fetching available Gemini models from Google AI ---")
        all_models = list(genai.list_models())
        for m in all_models:
            # Check if the model supports the 'generateContent' method
            is_supported = 'generateContent' in m.supported_generation_methods
            print(f"Found model: {m.name}, Supports 'generateContent': {is_supported}")
        print("-----------------------------------------------------")
        # --- END DEBUGGING ---

        model_info = next(
            (m for m in all_models if m.name == model_name and 'generateContent' in m.supported_generation_methods),
            None
        )
        if model_info:
            print(f"Successfully found and selected model: {model_name}")
            return genai.GenerativeModel(model_name)
        else:
            st.error(f"Selected Gemini model '{model_name}' not found or does not support generateContent.")
            return None
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}. Check API key and internet connection.")
        return None

# Get cached instances
embedding_model = get_embedding_model()
qdrant_client = get_qdrant_client()
gemini_model = get_gemini_model() # Default to gemini-pro

# --- RAG Core Functions (Adapted for Streamlit) ---

def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text from an uploaded PDF file.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def upload_chunks_to_qdrant(chunks, source_name):
    """
    Generates embeddings and uploads chunks to Qdrant.
    Creates collection if it doesn't exist or if forced refresh.
    """
    if not chunks:
        st.warning("No chunks to upload.")
        return

    vector_size = embedding_model.get_sentence_embedding_dimension()

    # Ensure collection exists
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        # Recreate if not found
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        st.info(f"Collection '{COLLECTION_NAME}' created.")

    points = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue # Skip empty chunks

        embedding = embedding_model.encode(chunk).tolist()
        point_id = str(uuid.uuid4())

        payload = {
            "text": chunk,
            "source": source_name,
            "chunk_index": i
        }
        points.append(models.PointStruct(id=point_id, vector=embedding, payload=payload))

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        st.success(f"Uploaded {len(points)} chunks from {source_name} to Qdrant.")

def retrieve_context(query, top_k=3):
    """
    Retrieves the most relevant text chunks from Qdrant.
    """
    query_embedding = embedding_model.encode(query).tolist()
    
    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        context_chunks = [hit.payload for hit in search_result]
        return context_chunks
    except Exception as e:
        st.error(f"Error during Qdrant search: {e}. Is Qdrant running?")
        return []

def generate_answer(query, context_chunks, selected_llm_model_name):
    """
    Generates an answer using the LLM with the provided context.
    """
    if not gemini_model: # Check if model initialized successfully
        return "LLM model not available. Please check configuration."

    if not context_chunks:
        return "I couldn't find relevant information in the documents to answer your question."

    combined_context = "\n\n".join([chunk['text'] for chunk in context_chunks])

    prompt = f"""
    You are a helpful assistant that answers questions only based on the provided context.
    If the answer cannot be found in the context, clearly state that you cannot answer from the provided information.
    Do not make up information.

    Question: {query}

    Context:
    {combined_context}

    Answer:
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024
            ),
        )

        if response.parts:
            return response.text
        else:
            st.warning("Gemini response was empty or blocked. Check prompt feedback/safety settings.")
            # st.json(response.prompt_feedback) # For debugging
            return "I'm sorry, I couldn't generate an answer. The model might have blocked the response due to safety concerns or found no suitable content."

    except genai.types.BlockedPromptException as e:
        st.error(f"Gemini API Error: Prompt was blocked by safety settings. Details: {e}")
        return "I'm sorry, your request could not be processed due to safety guidelines."
    except Exception as e:
        st.error(f"An unexpected error occurred while generating the answer: {e}")
        return "An error occurred while trying to generate an answer."

# --- Streamlit UI ---

st.title("📚 RAG Document Q&A App")
st.markdown("Ask questions about your uploaded PDFs using a Retrieval Augmented Generation (RAG) system.")

# --- Document Upload Section ---
st.header("1. Attach Documents")
uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to upload and use for RAG."
)

if st.button("Process Documents"):
    if uploaded_files:
        total_chunks_processed = 0
        for uploaded_file in uploaded_files:
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if text:
                with st.spinner(f"Chunking text from {uploaded_file.name}..."):
                    chunks = chunk_text(text)
                
                with st.spinner(f"Generating embeddings and uploading chunks for {uploaded_file.name} to Qdrant..."):
                    upload_chunks_to_qdrant(chunks, uploaded_file.name)
                    total_chunks_processed += len(chunks)
            else:
                st.warning(f"Could not extract text from {uploaded_file.name}.")
        st.success(f"Finished processing. Total chunks uploaded: {total_chunks_processed}")
        st.session_state.processed_docs = True # Set a flag that docs have been processed
    else:
        st.warning("Please upload at least one PDF file to process.")

# --- Model Choice Section ---
st.header("2. Model Configuration")

# Embedding Model Choice (simplified, as we only set up one for now)
st.markdown("Using `all-MiniLM-L6-v2` for embeddings.")
# In a more advanced app, you could offer choices here.

# LLM Model Choice
# Get available Gemini models that support generateContent
available_gemini_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_gemini_models.append(m.name)
except Exception as e:
    st.warning(f"Could not list Gemini models. Check API key/connection: {e}")

if gemini_model and available_gemini_models:
    # Set default to the currently initialized model name
    default_model_index = 0
    if gemini_model.model_name in available_gemini_models:
        default_model_index = available_gemini_models.index(gemini_model.model_name)

    selected_llm_model_name = st.selectbox(
        "Choose Gemini LLM Model:",
        options=available_gemini_models,
        index=default_model_index
    )
    # Re-initialize Gemini model if selection changes (though get_gemini_model uses cache)
    # The @st.cache_resource decorator handles this.
    st.session_state.current_llm_model_name = selected_llm_model_name
elif not gemini_model:
    st.warning("Gemini model initialization failed. Cannot select LLM.")
else:
    st.warning("No Gemini models found that support generateContent. Please check API key.")

# --- Query Section ---
st.header("3. Query Your Documents")
user_query = st.text_input("Enter your question here:", placeholder="e.g., What is Amazon EC2?")

if st.button("Get Answer"):
    if 'processed_docs' not in st.session_state or not st.session_state.processed_docs:
        st.warning("Please process your documents first before querying.")
    elif not user_query:
        st.warning("Please enter a question.")
    elif not gemini_model:
        st.error("LLM model not initialized. Cannot answer query.")
    else:
        with st.spinner("Searching and generating answer..."):
            # Ensure we use the correct LLM model name from selection
            final_answer = generate_answer(user_query, retrieve_context(user_query), st.session_state.current_llm_model_name)
            st.session_state.last_answer = final_answer # Store for display

# --- Response Display ---
st.header("4. App Response")
if 'last_answer' in st.session_state:
    st.markdown("### Answer:")
    st.write(st.session_state.last_answer)

st.markdown("---")
st.markdown("App by Coding Partner")