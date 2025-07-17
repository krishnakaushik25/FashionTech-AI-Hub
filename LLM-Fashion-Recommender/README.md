
# LLM-Based Fashion Recommendation Engine

A modular AI system for delivering intelligent, personalized product recommendations in online fashion platforms. This repository showcases a chatbot that combines large language models, retrieval-augmented search, and real-time user interaction for seamless shopping assistance.

---

## Summary

The application is architected as a hybrid Retrieval-Augmented Generation (RAG) chatbot for e-commerce. Users receive product suggestions, answers to queries, and an enhanced browsing experience through the synergy of language models and vector search. A scalable backend, modern UI, and robust data pipeline support large-scale, responsive recommendation flows.

---

## Main Features

- **Intelligent Product Suggestion:** AI generates custom recommendations tailored to user needs.
- **Multi-Modal Search:** Leverages a combination of FAISS, BM25, and ChromaDB for optimal retrieval across different query types.
- **Conversational Q&A:** Chatbot interface powered by leading LLMs for natural product exploration.
- **Reranking with Cross-Encoders:** Boosts precision in result ordering.
- **Structured Query Processing:** Automatically transforms user questions into actionable product filters.
- **Modern UI:** Built with Streamlit for interactive chat-based engagement.
- **Robust API Layer:** FastAPI powers backend endpoints for secure, scalable communication.
- **Container-Ready:** Full Docker and Docker Compose support for rapid deployment.

---

## Technology Overview

| Purpose                | Frameworks / Libraries                    |
|------------------------|-------------------------------------------|
| Core Language Models   | GPT-4o-mini, Llama 3.2:3B, Ollama         |
| Vector Search          | FAISS, ChromaDB                           |
| Retrieval Algorithms   | BM25, LangChain                           |
| Backend/API            | FastAPI, Pydantic, Loguru                 |
| Frontend/UI            | Streamlit                                 |
| Deployment             | Docker, Docker Compose                    |
| Data/Utilities         | Pandas, Numpy, Kaggle API                 |

---

## Setup & Usage

### Requirements

- Python 3.12 or later
- Docker, Docker Compose
- Ollama (local installation required)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/amine-akrout/llm-based-recommender.git
   cd llm-based-recommender
   ```

2. **Environment Configuration:**

   Copy the sample environment file and update your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` for Kaggle API, OpenAI key, etc.

3. **Launching with Docker (Preferred):**
   ```bash
   docker-compose up --build
   ```
   Or, using the Makefile:
   ```bash
   Make docker-start
   ```

4. **Manual Local Setup:**

   ```bash
   Make install-python
   Make install
   Make indexing
   Make retriever
   Make app
   Make ui
   ```

5. **Interfaces:**

   - **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

---

## Endpoints

| Method | Path         | Function                             |
|--------|--------------|--------------------------------------|
| POST   | /recommend/  | Retrieve recommended fashion products |
| GET    | /health      | System health check                  |

---

## Data & Indexing Pipeline

- Download data via Kaggle API
- Clean and prepare product dataset
- Generate vector representations for search
- Store embeddings in FAISS and BM25 for rapid retrieval

---

## Recommendation Workflow

The system employs a multi-stage pipeline:

1. **Topic Analysis:** Confirms user query is fashion-related
2. **Retrieval:** Collects candidate items using FAISS/BM25/ChromaDB
3. **Reranking:** Applies cross-encoder for precision sorting
4. **LLM Augmentation:** Large language model formulates and explains the final recommendations

---

## System Architecture

```
llm-based-recommender/
├── src/
│   ├── api/           # FastAPI application
│   ├── indexing/      # Data and index operations
│   ├── retriever/     # Retrieval algorithms and logic
│   ├── recommender/   # Core recommendation engine
│   ├── ui/            # Streamlit interface
│   └── config.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .env.example
├── assets/
```

---

## Demo & Visualization

Streamlit-based chatbot demo available at [http://localhost:8501](http://localhost:8501). Sample recommendation flow and UI screenshots included in the `assets/` folder.

---

## Development & Contributions

- Fork the repository and branch for your feature or fix.
- Commit, push, and submit a pull request describing your changes.
- For significant contributions, open an issue for discussion prior to coding.

---

## Future Roadmap

- Fine-tuning LLMs for higher recommendation accuracy
- UI improvements (product images, filtering, etc.)
- Support for multiple languages
- Cloud deployment options (AWS, GCP, etc.)

---