# IntelliDocs-RAG: Advanced Conversational AI with Hybrid Search and Tool Calling

IntelliDocs-RAG is a comprehensive backend system built with FastAPI that provides a powerful and accurate solution for conversing with your documents. It goes beyond basic RAG by implementing a production-grade pipeline featuring hybrid search, cross-encoder reranking, and intelligent tool-calling for tasks like booking appointments.

## Features

-   **Document Ingestion API**: Upload `.pdf` and `.txt` files through a simple REST endpoint.
-   **Selectable Chunking Strategies**: Choose between `fixed-size` character chunking or advanced `semantic` chunking that groups sentences by their meaning.
-   **Advanced RAG Pipeline**:
    -   **Hybrid Search**: Combines the strengths of keyword-based search (**BM25**) and semantic vector search (**Qdrant**) to retrieve the most relevant documents.
    -   **Cross-Encoder Reranking**: A second-stage reranker refines the search results, significantly boosting context precision before it's sent to the LLM.
-   **Conversational Chat API**:
    -   **Multi-Turn Memory**: Maintains conversation history using **Redis**, allowing for follow-up questions and contextual understanding.
    -   **Dual LLM Support**: Easily switch between a powerful cloud model (**Google Gemini**) and a fast, local model (**Llama.cpp / Qwen**).
-   **Intelligent Tool Calling**:
    -   **Interview Booking**: The system can detect a user's intent to book an interview and use **Gemini's native Function Calling** to reliably extract details (name, email, date, time).
    -   **Data Persistence**: Booking information is saved to a **PostgreSQL** database.
-   **Fully Containerized**: The entire application stack (API, databases, vector store) is managed with **Docker and Docker Compose** for easy setup and deployment.
-   **Modular & Scalable Architecture**: Built using a clean, service-oriented architecture that separates concerns and follows industry best practices.

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **Databases** | **PostgreSQL** (Metadata & Bookings), **Qdrant** (Vector Store), **Redis** (Chat Memory) |
| **AI / RAG Pipeline** | `sentence-transformers` (Embeddings & Reranking), `rank-bm25` (Sparse Search) |
| **Language Models** | `google-genai` (Gemini for Generation & Tool Calling), `llama-cpp-python` (Local LLM Support) |
| **Deployment** | Docker, Docker Compose |

## Project Structure

The project follows a clean, modular structure to ensure maintainability and scalability.

```text
.
├── app/
│   ├── api/          # FastAPI routers (endpoints)
│   ├── core/         # Configuration and settings
│   ├── models/       # SQLAlchemy models and Pydantic schemas
│   ├── services/     # Business logic (RAG, booking, vector DB)
│   ├── utils/        # Helper utilities (chunking, LLM clients)
│   └── main.py       # Main FastAPI application entrypoint
├── local_llm/        # (Git-ignored) Directory for local GGUF models
├── models/           # (Git-ignored) Directory for embedding/reranker models
├── .env              # (Git-ignored) Environment variables
├── docker-compose.yml # Docker service definitions
├── Dockerfile        # Docker build instructions for the API
└── README.md         # This file
```

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   [Git](https://git-scm.com/)
-   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/INTELLIDOCS-RAG.git
cd INTELLIDOCS-RAG
```

### 3. Configure Environment

Create a `.env` file from the example and add your Google API key.

```bash
cp .env.example .env
```

Now, open the `.env` file and paste your key:

```env
# .env
GOOGLE_API_KEY=AIzaSy...your...actual...key...
```

### 4. Download AI Models (Crucial Step)

The local models are not stored in Git. You must download them manually.

First, create the necessary directories:

```bash
mkdir -p local_llm
mkdir -p models
```

Download the following models and place them in the specified paths:

-   **LLM (Qwen-GGUF):**
    -   **Download from:** [Hugging Face](https://huggingface.co/Qwen/Qwen1.5-4B-Chat-GGUF/resolve/main/qwen1_5-4b-chat-q4_k_m.gguf)
    -   **Place at:** `local_llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`

-   **Embedding Model (Gemma):**
    -   **Download from:** [Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
    -   **Place the entire folder at:** `models/embeddinggemma-300m`

-   **Semantic Chunking Model (MiniLM):**
    -   **Download from:** [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    -   **Place the entire folder at:** `models/all-MiniLM-L6-v2`

-   **Reranker Model (ms-marco):**
    -   **Download from:** [Hugging Face](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
    -   **Place the entire folder at:** `models/ms-marco-MiniLM-L6-v2`

### 5. Build and Run with Docker

Once the models are in place, you can manage the entire application stack using Docker Compose.

**1. Build the Docker Images**

This command builds the image for the API service as defined in the `Dockerfile`. You only need to run this the first time or after changing dependencies in `requirements.txt` or the `Dockerfile` itself.

```bash
docker-compose build
```

**2. Start the Services**

This command starts all services (API, PostgreSQL, Redis, Qdrant) in the background.

```bash
docker-compose up -d
```

-   The `-d` flag runs the containers in "detached" mode, freeing up your terminal.
-   The API will now be available at `http://localhost:8000`.
-   For subsequent runs, you can just use `docker-compose up -d` to start everything again.

**3. View Logs (Optional)**

To monitor the application's output or debug issues, you can stream the logs from the API service:

```bash
docker-compose logs -f api
```
Press `Ctrl+C` to stop viewing the logs.

**4. Stop the Application**

When you are finished, use this command to gracefully stop and remove all the running containers and networks.

```bash
docker-compose down
```

## API Usage

You can interact with the API using any HTTP client (e.g., `curl`, Postman, or Python's `requests`).

### 1. Document Ingestion API

Upload one or more documents to be processed and indexed.

-   **Endpoint**: `POST /api/ingest`
-   **Type**: `multipart/form-data`

**Parameters:**
-   `files`: One or more `.pdf` or `.txt` files.
-   `chunking_strategy`: (form data) `fixed` or `semantic`. Defaults to `fixed`.

**Example (`curl`):**

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "files=@/path/to/your/document.pdf" \
  -F "files=@/path/to/another/document.txt" \
  -F "chunking_strategy=semantic"
```

### 2. Conversational RAG API

Send messages to the RAG pipeline to ask questions or book interviews.

-   **Endpoint**: `POST /api/chat`
-   **Type**: `application/json`

**Request Body:**

```json
{
  "session_id": "user-123-abc",
  "message": "What is the main topic of the document?",
  "model": "gemini"
}
```

**Example (`curl`):**

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123-abc",
    "message": "I want to book an interview for Sajeena Malla at sajeena@example.com for tomorrow at 2:30 PM.",
    "model": "gemini"
  }'
```

**Response:**

The API will respond with a generated reply, source chunks used for the answer, and booking confirmation details if applicable.

```json
{
  "reply": "I see you'd like to book an interview. To confirm, could you please provide the exact date in YYYY-MM-DD format?",
  "sources": [],
  "booking_created": false,
  "booking_id": null
}
```
