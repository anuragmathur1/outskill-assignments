"""
Fashion RAG Pipeline - Assignment
Week 9: Multimodal RAG Pipeline with H&M Fashion Dataset

OBJECTIVE: Build a complete multimodal RAG (Retrieval-Augmented Generation) pipeline
that can search through fashion items using both text and image queries, then generate
helpful responses using an LLM.

LEARNING GOALS:
- Understand the three phases of RAG: Retrieval, Augmentation, Generation
- Work with multimodal data (images + text)
- Use vector databases for similarity search
- Integrate LLM for response generation
- Build an end-to-end AI pipeline

DATASET: H&M Fashion Caption Dataset
- 20K+ fashion items with images and text descriptions
- URL: https://huggingface.co/datasets/tomytjandra/h-and-m-fashion-caption

PIPELINE OVERVIEW:
1. RETRIEVAL: Find similar fashion items using vector search
2. AUGMENTATION: Create enhanced prompts with retrieved context
3. GENERATION: Generate helpful responses using LLM

Commands to run:
python assignment_fashion_rag.py --query "black dress for evening"
python assignment_fashion_rag.py --app
"""

import argparse
import os
import re

# Suppress warnings
import warnings
from typing import Any, Dict, List, Optional, Tuple
from random import sample
from pathlib import Path
from dotenv import load_dotenv

# Gradio for web interface
import gradio as gr

# Core dependencies
import lancedb
import pandas as pd
import torch
from datasets import load_dataset
from datasets.inspect import get_dataset_split_names
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

# LLM dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import google.generativeai as genai

warnings.filterwarnings("ignore")


def is_huggingface_space():
    """
    Checks if the code is running within a Hugging Face Spaces environment.

    Returns:
        bool: True if running in HF Spaces, False otherwise.
    """
    if os.environ.get("SYSTEM") == "spaces":
        return True
    else:
        return False


# =============================================================================
# SECTION 1: DATABASE SETUP AND SCHEMA
# =============================================================================


def register_embedding_model(model_name: str = "open-clip") -> Any:
    """
    Register embedding model for vector search

    Complete this function
    HINT: Use EmbeddingFunctionRegistry to get and create the model

    Args:
        model_name: Name of the embedding model
    Returns:
        Embedding model instance
    """
    # Get the registry instance
    # registry = ?
    registry = EmbeddingFunctionRegistry.get_instance()

    # Check if model is already registered
    # If not registered, get and create the model
    try:
        # The EAFP (Easier to Ask for Forgiveness than Permission) pattern is
        # best here. We try to get the model, and handle the error if it fails.
        model = registry.get(model_name).create()
        print(f"✅ Model '{model_name}' is available.")
        return model
    except Exception:
        raise ValueError(f"Embedding model '{model_name}' not found in the registry or failed to create.")

# Global embedding model
clip_model = register_embedding_model()


class FashionItem(LanceModel):
    """
    Schema for fashion items in vector database

    Complete the schema definition
    HINT: This defines the structure of data stored in the vector database

    REQUIRED FIELDS:
    1. vector: Vector field for CLIP embeddings (use clip_model.ndims())
    2. image_uri: String field for image file paths
    3. description: Optional string field for text descriptions
    """

    # Add image field (source)
    image_uri: str = clip_model.SourceField()
    
    # Add vector field for embeddings
    vector: Vector(clip_model.ndims()) = clip_model.VectorField()

    # Add text description field
    # description = ?
    description: Optional[str] = None
    # DUMMY IMPLEMENTATION - Replace with actual schema
    # pass


# =============================================================================
# SECTION 2: RETRIEVAL - Vector Database Operations
# =============================================================================


def setup_fashion_database(
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    dataset_name: str = "tomytjandra/h-and-m-fashion-caption",
    sample_size: int = 1000,
    images_dir: str = "fashion_images",
) -> None:
    """
    Set up vector database with H&M fashion dataset

    Complete this function to:
    1. Connect to LanceDB database
    2. Check if table already exists (skip if it does)
    3. Load H&M dataset from HuggingFace
    4. Process and save images locally
    5. Create vector database table
    """

    # Connect to LanceDB
    # db = ?
    db = lancedb.connect(database_path)
    
    # Check if table already exists
    if table_name in db:
        # open table
        existing_table = db.open_table(table_name)
        print(f"✅ Table '{table_name}' already exists with {len(existing_table)} items")
        return
    else:
        print(f"🏗️ Table '{table_name}' does not exist, creating new fashion database...")

    # Load dataset from HuggingFace
    print("📥 Loading H&M fashion dataset...")
    # Get splits for the 'rotten_tomatoes' dataset
    split_names = get_dataset_split_names(dataset_name)

    print(f"Available splits: {split_names}")
    # Expected Output: Available splits: ['train', 'validation', 'test']

    train_data = load_dataset(dataset_name, split="train")

    # Sample data to specified size in the sample_size parameter
    # train_data = ?
    # print(f"Processing {len(train_data)} fashion items...")
    if len(train_data) > sample_size:
        indices = sample(range(len(train_data)), sample_size)
        train_data = train_data.select(indices)

    print(f"Processing {len(train_data)} samples...")


    # Create images directory
    os.makedirs(images_dir, exist_ok=True)

    # Process each item
    table_data = []
    for i, item in enumerate(train_data):
        # Get image and text
        image = item["image"]
        text = item["text"]

        # Save image
        image_path = os.path.join(images_dir, f"fashion_{i:04d}.jpg")
        image.save(image_path)

        # Create record
        record = {
            "image_uri": image_path,
            "description": text
        }
        table_data.append(record)

        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(train_data)} items...")

    # Create vector database table
    # print("🗄️ Creating vector database table...")
    if table_data:
        if table_name in db:
            db.drop_table(table_name)

        table = db.create_table(table_name, schema=FashionItem, mode="create")
        table.add(pd.DataFrame(table_data))
        print(f"Added {len(table_data)} shoes to table")
    else:
        print("No data to add")
    # print(f"✅ Created table '{table_name}' with {len(table_data)} items")

    # # DUMMY IMPLEMENTATION
    # print("⚠️ Implement database setup")
    # print(f"Database path: {database_path}")
    # print(f"Dataset: {dataset_name}")
    # print(f"Sample size: {sample_size}")


def search_fashion_items(
    database_path: str,
    table_name: str,
    query: Any,
    search_type: str = "auto",
    limit: int = 3,
) -> Tuple[List[Dict], str]:
    """
    Search for fashion items using text or image query

    Complete this function to:
    1. Determine if query is text or image (auto-detection)
    2. Connect to the vector database
    3. Perform similarity search using CLIP embeddings
    4. Return search results and detected search type

    STEPS TO IMPLEMENT:
    1. Auto-detect search type: check if query is a file path
    2. Connect to database
    3. Open table
    4. Search based on type:
       - Image: load with PIL and search
       - Text: search directly with string
    5. Return results and search type

    Args:
        database_path: Path to LanceDB database
        table_name: Name of the table to search
        query: Search query (text or image path)
        search_type: "auto", "text", or "image"
        limit: Number of results to return

    Returns:
        Tuple of (results_list, actual_search_type)
    """

    print(f"🔍 Searching for: {query}")

    # Determine search type automatically
    # HINT: Use os.path.exists(query) to check if query is a file path
    # HINT: If file exists, it's an image search; otherwise, it's text search
    # actual_search_type = ?

    actual_search_type = search_type
    processed_query = query

    if search_type == "auto":
        if isinstance(query, str):
            # Using pathlib.Path is a robust way to check for file existence
            if Path(query).is_file():
                actual_search_type = "image"
            else:
                actual_search_type = "text"
        elif hasattr(query, "save"):  # It's a PIL Image object
            actual_search_type = "image"
        else:
            actual_search_type = "text"

    # Connect to database
    # db = ?
    db = lancedb.connect(database_path)
    try:
        # Open the table
        # table = ?
        table = db.open_table(table_name)
    except FileNotFoundError:
        print(f"❌ Table '{table_name}' not found in database '{database_path}'. Please run with --setup-db first.")
        return [], "error"




    # Perform search based on detected type
    # if actual_search_type == "image":
    #     # Load image and search
    #     image = ?
    #     results = ?
    # else:
    #     # Text search
    #     results = ?
    if actual_search_type == "image":
        if isinstance(query, str):
            try:
                processed_query = Image.open(query)
            except FileNotFoundError:
                print(f"Image file not found at: {query}")
                return [], "error"
        results_pydantic = table.search(processed_query).limit(limit).to_pydantic(FashionItem)
    else:
        results_pydantic = table.search(query).limit(limit).to_pydantic(FashionItem)


    # Convert pydantic models to a list of dictionaries
    results = [item.dict() for item in results_pydantic]

    # Print results found
    # print(f"   Found {len(results)} results using {actual_search_type} search")
    print(f"   Found {len(results)} results using {actual_search_type} search")

    # Return results and search type
    # return results, actual_search_type
    return results, actual_search_type

    # # DUMMY IMPLEMENTATION
    # print("⚠️ Implement fashion search")
    # dummy_results = [
    #     {
    #         "description": "solid black jersey top with narrow shoulder straps",
    #         "image_uri": "fashion_images/fashion_0001.jpg",
    #     },
    #     {
    #         "description": "blue denim jacket with button closure",
    #         "image_uri": "fashion_images/fashion_0002.jpg",
    #     },
    # ]

    # return dummy_results, "text"


# =============================================================================
# SECTION 3: AUGMENTATION - Prompt Engineering
# =============================================================================


def create_fashion_prompt(
    query: str, retrieved_items: List[Dict], search_type: str
) -> str:
    """
    Create enhanced prompt for LLM using retrieved fashion items

    Complete this function to create a well-structured prompt that:
    1. Creates a system prompt defining the AI assistant's role
    2. Formats retrieved items as context for the LLM
    3. Includes the user's query appropriately
    4. Combines everything into a coherent prompt

    PROMPT STRUCTURE:
    1. System prompt: Define the AI as a fashion assistant
    2. Context section: List retrieved fashion items with descriptions
    3. Query section: Include user's original query
    4. Instruction: Ask for fashion recommendations

    Args:
        query: Original user query
        retrieved_items: List of retrieved fashion items
        search_type: Type of search performed

    Returns:
        Enhanced prompt string for LLM
    """

    # Create system prompt
    # HINT: Define the AI as a fashion assistant with expertise
    # system_prompt = "You are a ..."
    system_prompt = "You are a friendly and knowledgeable fashion assistant. Your goal is to help users find the perfect clothing items based on their needs. Use the provided context to give helpful and concise recommendations. Do not make up information."


    # Format retrieved items context
    # context = "Here are some relevant fashion items from our catalog:\n\n"
    # for i, item in enumerate(retrieved_items, 1):
    #     context += f"{i}. {item['description']}\n\n"
    context = "Here are some fashion items that might match the user's request:\n\n"
    if not retrieved_items:
        context += "No specific items were found, but you can still offer general advice."
    else:
        for i, item in enumerate(retrieved_items, 1):
            description = item.get('description', 'No description available.')
            context += f"{i}. Description: {description}\n"

    # Create user query section
    # HINT: Handle different search types (image vs text)
    # if search_type == "image":
    #     query_section = ?
    # else:
    #     query_section = ?
    if search_type == "image":
        user_prompt = f"""Context:
{context}

The user provided an image as a query. Based on the items above, which are visually similar to the user's image? Please describe the best matches and why they are a good fit. Keep your response to 2-3 sentences."""
    else:  # text search
        user_prompt = f"""Context:
{context}

User's request: "{query}"

Based on the user's request and the provided context, which item(s) would you recommend? 
Explain your choice in 2-3 sentences, referencing the item descriptions."""


    # Combine into final prompt
    # HINT: Combine system prompt, context, query section, and response instruction
    # prompt = f"{system_prompt}\n\n{context}\n{query_section}\n\nResponse:"
    # return prompt
    final_prompt = f"{system_prompt}\n\n{user_prompt}"
    return final_prompt

    # # DUMMY IMPLEMENTATION
    # print("⚠️ Create enhanced prompt")
    # return f"Fashion query: {query}\nRetrieved {len(retrieved_items)} items."


# =============================================================================
# SECTION 4: GENERATION - LLM Response Generation
# =============================================================================


def setup_qwen_llm_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Tuple[Any, Any]:
    """
    Set up LLM model and tokenizer

    Complete this function to load the LLM model and tokenizer

    STEPS TO IMPLEMENT:
    1. Load tokenizer
    2. Load model
    3. Configure model settings for GPU/CPU
    5. Return tokenizer and model

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (tokenizer, model)
    """

    print(f"🤖 Loading LLM model: {model_name}")

    # Load tokenizer
    # tokenizer = ?
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    # model = ?
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print success message and return
    print("✅ LLM model loaded successfully")
    return tokenizer, model

    # DUMMY IMPLEMENTATION
    # print("⚠️ Load LLM model and tokenizer")
    # return None, None


# def setup_openai_llm_model(api_key: str) -> OpenAI:
def setup_openai_llm_model(api_key: Optional[str] = None) -> OpenAI:
    """
    Set up OpenAI client for LLM requests.

    Uses the provided API key if available, otherwise loads from .env.

    Args:
        api_key: Optional OpenAI API key.

    Returns:
        OpenAI client instance.
    Raises:
        ValueError: If no API key is provided and it's not in the environment.
    """
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError(
            "OpenAI API key not found. Provide it in the UI or set OPENAI_API_KEY in a .env file."
        )

    try:
        client = OpenAI(api_key=api_key)
        client.models.list()  # Test the key
        print("✅ OpenAI client setup successful")
        return client
    except Exception as e:
        raise ValueError(f"Failed to set up OpenAI client: {e}") from e
    
def setup_gemini_llm_model(
    model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None
) -> Any:
    """
    Set up Google Gemini client for LLM requests.

    Uses the provided API key if available, otherwise loads from .env.

    Args:
        model_name: The name of the Gemini model to use.
        api_key: Optional Google API key.

    Returns:
        A configured Gemini GenerativeModel instance.

    Raises:
        ValueError: If no API key is provided and it's not in the environment, or if setup fails.
    """
    # Load environment variables from .env file
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError(
            "Google API key not found. Provide it in the UI or set GOOGLE_API_KEY in a .env file."
        )

    try:
        genai.configure(api_key=api_key)
        # Create a model instance to test the setup and return it
        model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini client setup successful with model '{model_name}'")
        return model
    except Exception as e:
        raise ValueError(f"Failed to set up Gemini client: {e}") from e


def get_available_models() -> Dict[str, List[str]]:
    """Get available models for each provider."""
    models = {
        "qwen": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
    }
    return models

def generate_fashion_response(
    # prompt: str, tokenizer: Any, model: Any, max_tokens: int = 200
    prompt: str,
    llm_provider: str,
    llm_components: Any,
    model_name: Optional[str] = None,
    max_tokens: int = 200,
) -> str:
    """
    Generate response using a specified LLM provider.

    This function is designed to be flexible and work with different LLM providers
    by checking the `llm_provider` string.

    Complete this function to generate text using the LLM

    STEPS TO IMPLEMENT:
    1. Check if tokenizer and model are loaded
    2. Encode the prompt with attention mask
    3. Generate response using model.generate()
    4. Decode the response and clean it up
    5. Return the generated text

    Args:
        prompt: Input prompt for the model.
        llm_provider: The provider of the LLM ('qwen', 'openai', 'gemini').
        llm_components: The necessary components for the LLM.
                        - For 'qwen': a tuple (tokenizer, model).
                        - For 'openai': an OpenAI client instance.
                        - For 'gemini': a Gemini model instance.
        model_name: The specific model name, required for OpenAI.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated response text
    """

    # if not tokenizer or not model:
    if not llm_components:
        return "⚠️ LLM not loaded - showing search results only"

    # Encode prompt with attention mask
    # HINT: Use tokenizer() with return_tensors="pt", truncation=True, max_length=1024, padding=True
    # inputs = ?

    # Generate response
    # with torch.no_grad():
    #     outputs = model.generate(
    #         inputs.input_ids,
    #         attention_mask=inputs.attention_mask,
    #         max_new_tokens=max_tokens,
    #         temperature=0.7,
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id,
    #         eos_token_id=tokenizer.eos_token_id
    #     )

    # Decode response and clean it up
    # full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = full_response.replace(prompt, "").strip()
    # return response

    # DUMMY IMPLEMENTATION
    # print("⚠️ Generate LLM response")
    # return "This is a dummy response. Please implement the LLM generation logic."
    print(f"🤖 Generating response using {llm_provider.upper()}...")

    try:
        if llm_provider == "qwen":
            tokenizer, model = llm_components
            if not tokenizer or not model:
                raise ValueError("Tokenizer and model must be provided for Qwen.")

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True
            )
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.replace(prompt, "").strip()
            return response

        elif llm_provider == "openai":
            client = llm_components
            if not model_name:
                raise ValueError("model_name is required for OpenAI.")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        elif llm_provider == "gemini":
            model = llm_components
            response = model.generate_content(prompt)
            return response.text.strip()
            
        else:
            return f"⚠️ Unknown LLM provider: {llm_provider}"

    except Exception as e:
        print(f"❌ Error during LLM generation with {llm_provider}: {e}")
        return f"An error occurred while generating the response with {llm_provider}."


# =============================================================================
# SECTION 5: IMAGE STORAGE
# =============================================================================


def save_retrieved_images(
    results: Dict[str, Any], output_dir: str = "retrieved_fashion_images"
) -> List[str]:
    """Save retrieved fashion images to output directory"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    query_safe = re.sub(r"[^\w\s-]", "", str(results["query"]))[:30]
    query_safe = re.sub(r"[-\s]+", "_", query_safe)

    saved_paths = []

    print(f"💾 Saving {len(results['results'])} retrieved images...")

    for i, item in enumerate(results["results"], 1):
        original_path = item["image_uri"]
        image = Image.open(original_path)

        # Generate new filename
        filename = f"{query_safe}_result_{i:02d}.jpg"
        save_path = os.path.join(output_dir, filename)

        # Save image
        image.save(save_path, "JPEG", quality=95)
        saved_paths.append(save_path)

        print(f"   ✅ Saved image {i}: {filename}")
        print(f"      Description: {item.get('description', 'No description')[:60]}...")

    print(f"💾 Saved {len(saved_paths)} images to: {output_dir}")
    return saved_paths


# =============================================================================
# SECTION 6: COMPLETE RAG PIPELINE
# =============================================================================


def run_fashion_rag_pipeline(
    query: str,
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    search_type: str = "auto",
    limit: int = 3,
    save_images: bool = True,
    llm_provider: str = "gemini",
    model_name: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete fashion RAG pipeline

    Complete this function to orchestrate the entire pipeline:
    1. RETRIEVAL: Search for relevant fashion items using vector database
    2. AUGMENTATION: Create enhanced prompt with retrieved context
    3. GENERATION: Generate LLM response using the enhanced prompt
    4. IMAGE STORAGE: Save retrieved images if requested

    This is the main function that ties everything together!

    PIPELINE PHASES:
    Phase 1 - RETRIEVAL: Find similar fashion items
    Phase 2 - AUGMENTATION: Create context-rich prompt
    Phase 3 - GENERATION: Generate helpful response
    Phase 4 - STORAGE: Save retrieved images
    """

    print("🚀 Starting Fashion RAG Pipeline")
    print("=" * 50)

    # PHASE 1: RETRIEVAL
    print("🔍 PHASE 1: RETRIEVAL")
    # Search for fashion items using the search function
    # HINT: Call search_fashion_items() with the provided parameters
    # results, actual_search_type = ?
    # print(f"   Found {len(results)} relevant items")
    results, actual_search_type = search_fashion_items(
        database_path=database_path,
        table_name=table_name,
        query=query,
        search_type=search_type,
        limit=limit,
    )
    print(f"   Found {len(results)} relevant items")


    # PHASE 2: AUGMENTATION
    print("📝 PHASE 2: AUGMENTATION")
    # Create enhanced prompt using retrieved items
    # HINT: Call create_fashion_prompt() with parameters
    # enhanced_prompt = ?
    # print(f"   Created enhanced prompt ({len(enhanced_prompt)} chars)")
    enhanced_prompt = create_fashion_prompt(
        query=query, retrieved_items=results, search_type=actual_search_type
    )
    print(f"   Created enhanced prompt ({len(enhanced_prompt)} chars)")


    # PHASE 3: GENERATION
    print("🤖 PHASE 3: GENERATION")
    # Set up LLM and generate response
    # tokenizer, model = ?
    # response = ?
    # print(f"   Generated response ({len(response)} chars)")

    # Set default model name if not provided
    if not model_name:
        available_models = get_available_models()
        if llm_provider in available_models:
            model_name = available_models[llm_provider][0]
        else:
            # This case should ideally not be hit due to argparse choices
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")


    if llm_provider == "qwen":
        tokenizer, model = setup_qwen_llm_model(model_name=model_name)
        llm_components = (tokenizer, model)
    elif llm_provider == "openai":
        client = setup_openai_llm_model(api_key=openai_api_key)
        llm_components = client
    elif llm_provider == "gemini":
        model = setup_gemini_llm_model(model_name=model_name, api_key=gemini_api_key)
        llm_components = model
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Prepare final results dictionary
    # final_results = {
    #     "query": query,
    #     "results": results,
    #     "response": response,
    #     "search_type": actual_search_type
    # }
    response = generate_fashion_response(
        prompt=enhanced_prompt,
        llm_provider=llm_provider,
        llm_components=llm_components,
        model_name=model_name,
        max_tokens=200,
    )
    print(f"   Generated response ({len(response)} chars)")

    # Prepare final results dictionary
    final_results = {
        "query": query,
        "results": results,
        "response": response,
        "search_type": actual_search_type,
    }

    # Save retrieved images if requested
    # if save_images:
    #     saved_image_paths = save_retrieved_images(final_results)
    #     final_results["saved_image_paths"] = saved_image_paths
    if save_images:
        saved_image_paths = save_retrieved_images(final_results)
        final_results["saved_image_paths"] = saved_image_paths


    # Return final results
    # return final_results
    return final_results

    # # DUMMY IMPLEMENTATION
    # print("⚠️ Implement complete RAG pipeline")

    # return {
    #     "query": query,
    #     "results": [],
    #     "response": "Pipeline not implemented yet",
    #     "search_type": "unknown",
    # }


# =============================================================================
# GRADIO WEB APP
# =============================================================================


def fashion_search_app(
    text_query,
    image_query,
    llm_provider,
    model_name,
    openai_api_key,
    gemini_api_key,
):
    """
    Process fashion query and return response with images for Gradio

    Complete this function to handle web app queries

    STEPS TO IMPLEMENT:
    1. Check if query is provided
    2. Setup database if needed
    3. Run RAG pipeline
    4. Extract LLM response and images
    5. Return formatted results for Gradio
    """

    # if not query.strip():
    #     return "Please enter a search query", []
    query = image_query if image_query is not None else text_query

    if query is None or (isinstance(query, str) and not query.strip()):
        return "Please enter a search query or upload an image", [], "", None
    

    # Setup database if needed (will skip if exists)
    print("🔧 Checking/setting up fashion database for Gradio app...")
    setup_fashion_database()

    # Run the RAG pipeline
    # result = ?
    print(f"Running RAG pipeline for query: {query}")
    result = run_fashion_rag_pipeline(
        query=query,
        database_path="fashion_db",
        table_name="fashion_items",
        search_type="auto",
        limit=3,
        save_images=False,  # Don't need to save images for Gradio display
        llm_provider=llm_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
    )

    # Get LLM response
    llm_response = result.get('response', "Sorry, I couldn't generate a response.")


    # Get retrieved images for display

    retrieved_images = []
    retrieved_items = result.get('results', [])
    for item in retrieved_items:
        if 'image_uri' in item and os.path.exists(item['image_uri']):
            try:
                img = Image.open(item['image_uri'])
                retrieved_images.append(img)
            except Exception as e:
                print(f"Error loading image for Gradio app: {item['image_uri']} - {e}")



    # Return response and images and clear inputs
    return llm_response, retrieved_images, "", None

def launch_gradio_app():
    """Launch the Gradio web interface"""

    # Create Gradio interface
    with gr.Blocks(title="Fashion RAG Assistant") as app:

        gr.Markdown("# 👗 Fashion RAG Assistant")
        gr.Markdown("Search for fashion items and get AI-powered recommendations!")

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your fashion query (e.g., 'black dress for evening')",
                    lines=2,
                )

                image_input = gr.Image(type="pil", label="Or Upload an Image")
                search_btn = gr.Button("Search", variant="primary")

                # Examples
                gr.Examples(
                    examples=[
                        "black dress for evening",
                        "casual summer outfit",
                        "blue jeans",
                        "white shirt",
                        "winter jacket",
                    ],
                    inputs=query_input,
                )
            with gr.Column(scale=1):
                # LLM selection
                available_models = get_available_models()
                llm_provider_radio = gr.Radio(
                    choices=list(available_models.keys()),
                    label="Select LLM Provider",
                    value="gemini",  # Default provider
                )

                model_dropdown = gr.Dropdown(
                    choices=available_models["gemini"],  # Default to Gemini models
                    label="Select Model",
                    value="gemini-1.5-flash",  # Default Gemini model
                )
                openai_api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key...",
                    visible=False,
                )

                gemini_api_key_input = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="Enter your Google API key...",
                    visible=True,  # Default provider is Gemini
                )

                def update_models_and_keys(provider):
                    """Updates model dropdown and API key visibility based on provider."""
                    return (
                        gr.update(
                            choices=available_models[provider],
                            value=available_models[provider][0],
                        ),
                        gr.update(visible=(provider == "openai")),
                        gr.update(visible=(provider == "gemini")),
                    )

            with gr.Column(scale=2):
                # Output
                response_output = gr.Textbox(
                    label="Fashion Recommendation", lines=8, interactive=False
                )

        # Retrieved Images
        images_output = gr.Gallery(
            label="Retrieved Fashion Items", columns=3, height=400
        )

        # Connect provider radio to model dropdown
        llm_provider_radio.change(
            fn=update_models_and_keys,
            inputs=llm_provider_radio,
            outputs=[model_dropdown, openai_api_key_input, gemini_api_key_input],
        )

        # Connect the search function
        search_btn.click(
            fn=fashion_search_app,
            inputs=[
                query_input,
                image_input,
                llm_provider_radio,
                model_dropdown,
                openai_api_key_input,
                gemini_api_key_input,
            ],
            outputs=[response_output, images_output, query_input, image_input],
        )

        # Also trigger on Enter key
        query_input.submit(
            fn=fashion_search_app,
            inputs=[
                query_input,
                image_input,
                llm_provider_radio,
                model_dropdown,
                openai_api_key_input,
                gemini_api_key_input,
            ],
            outputs=[response_output, images_output, query_input, image_input],
        )

    print("🚀 Starting Fashion RAG Gradio App...")
    print("📝 Note: First run will download dataset and setup database")
    app.launch(share=False)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function to handle command line arguments and run the pipeline"""

    # If running in Hugging Face Spaces, automatically launch the app
    if is_huggingface_space():
        print("🤗 Running in Hugging Face Spaces - launching web app automatically")
        launch_gradio_app()
        return

    parser = argparse.ArgumentParser(
        description="Fashion RAG Pipeline Assignment - SOLUTION"
    )
    parser.add_argument("--query", type=str, help="Search query (text or image path)")
    parser.add_argument("--app", action="store_true", help="Launch Gradio web app")
    parser.add_argument(
        "--llm-provider",
        choices=["qwen", "openai", "gemini"],
        default="gemini",
        help="LLM provider to use (default: gemini)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Specific LLM model to use for the selected provider.",
    )
    
    args = parser.parse_args()

    # Launch web app if requested
    if args.app:
        launch_gradio_app()
        return

    if not args.query:
        print("❌ Please provide a query with --query or use --app for web interface")
        print("Examples:")
        print("  python solution_fashion_rag.py --query 'black dress for evening'")
        print("  python solution_fashion_rag.py --query 'fashion_images/dress.jpg'")
        print("  python solution_fashion_rag.py --app")
        return

    # Setup database first (will skip if already exists)
    print("🔧 Checking/setting up fashion database...")
    setup_fashion_database()

    # Run the complete RAG pipeline with default settings
    result = run_fashion_rag_pipeline(
        query=args.query,
        database_path="fashion_db",
        table_name="fashion_items",
        search_type="auto",
        limit=3,
        save_images=True,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        openai_api_key=None,  # CLI uses .env
        gemini_api_key=None,   # CLI uses .env
    )

    # Display results
    print("\n" + "=" * 50)
    print("🎯 PIPELINE RESULTS")
    print("=" * 50)
    print(f"Query: {result['query']}")
    print(f"Search Type: {result['search_type']}")
    print(f"Results Found: {len(result['results'])}")
    print("\n📋 Retrieved Items:")
    for i, item in enumerate(result["results"], 1):
        print(f"{i}. {item.get('description', 'No description')}")

    print(f"\n🤖 LLM Response:")
    print(result["response"])

    # Show saved images info if any
    if result.get("saved_image_paths"):
        print(f"\n📸 Saved Images:")
        for i, path in enumerate(result["saved_image_paths"], 1):
            print(f"{i}. {path}")


if __name__ == "__main__":
    main()
