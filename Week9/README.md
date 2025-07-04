# Fashion RAG Pipeline Assignment

This project implements a complete multimodal RAG (Retrieval-Augmented Generation) pipeline for searching a fashion dataset. It allows users to query fashion items using both text descriptions and images, and receive AI-generated recommendations.

## 🚀 Features

- **Multimodal Search**: Find fashion items using either text queries (e.g., "blue denim jacket") or by providing an image.
- **Vector Database**: Uses LanceDB to store and efficiently search through vector embeddings of fashion items.
- **Multiple LLM Providers**: Supports response generation from various Large Language Models:
  - Qwen (local, via `transformers`)
  - OpenAI (via API)
  - Google Gemini (via API)
- **Gradio Web Interface**: An easy-to-use web app for interactive searching.
- **End-to-End RAG Pipeline**: Integrates the three core phases of RAG:
  1.  **Retrieval**: Fetches relevant items from the vector database.
  2.  **Augmentation**: Creates a rich, contextual prompt for the LLM.
  3.  **Generation**: Produces a helpful, human-like response.

## ⚙️ Setup

### 1. Install Dependencies

First, ensure you have all the required Python libraries installed. You can install them using pip:

```bash
pip install gradio lancedb pandas torch "datasets[vision]" transformers openai google-generativeai python-dotenv open_clip_torch Pillow
```

### 2. Configure API Keys

To use the OpenAI or Google Gemini models, you need to provide API keys.

1.  Create a file named `.env` in the same directory as the `assignment_fashion_rag.py` script.
2.  Add your API keys to this file. You only need to add keys for the services you intend to use.

    ```env
    OPENAI_API_KEY="sk-..."
    GOOGLE_API_KEY="AIza..."
    ```

## 🏃‍♀️ How to Run

### First-Time Setup

The very first time you run the script, it will automatically download the H&M fashion dataset, process the images, and build the vector database. This might take a few minutes. Subsequent runs will be much faster as they will use the existing database.

You can also trigger this process manually if needed:

```bash
# This command is optional, as the script runs it automatically on first use.
python assignment_fashion_rag.py --setup-db
```

### Option A: Run from the Command Line

This is ideal for quick tests and scripting.

*   **Search with a text query (defaults to OpenAI):**
    ```bash
    python assignment_fashion_rag.py --query "a comfortable pair of blue jeans"
    ```

*   **Specify a different LLM provider (e.g., Gemini or Qwen):**
    ```bash
    python assignment_fashion_rag.py --query "a comfortable pair of blue jeans" --llm-provider gemini
    ```
    ```bash
    python assignment_fashion_rag.py --query "a comfortable pair of blue jeans" --llm-provider qwen
    ```

*   **Search with an image query:**
    The script creates an `fashion_images` directory on the first run. You can use a path to one of those images for your query.
    ```bash
    python assignment_fashion_rag.py --query "fashion_images/fashion_0005.jpg"
    ```

### Option B: Launch the Gradio Web App

This provides a user-friendly interface for interacting with your RAG pipeline.

*   **To start the web app:**
    ```bash
    python assignment_fashion_rag.py --app
    ```

After running this command, you'll see a local URL in your terminal (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to use the application.

## 📁 Project Structure

- `assignment_fashion_rag.py`: The main script containing the entire RAG pipeline, Gradio app, and command-line interface.
- `fashion_db/`: Directory where the LanceDB vector database is stored.
- `fashion_images/`: Directory where images from the dataset are saved locally.
- `retrieved_fashion_images/`: Directory where images from search results are saved.