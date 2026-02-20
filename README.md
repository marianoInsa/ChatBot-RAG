# ChatBot-RAG

[![Build and Deploy](https://github.com/marianoInsa/ChatBot-RAG/actions/workflows/main.yml/badge.svg)](https://github.com/marianoInsa/ChatBot-RAG/actions/workflows/main.yml)

## Description

The chatbot leverages **LangChain**, **FastAPI**, and a Multi-LLM architecture (**Ollama**, **Groq**, **Gemini**) to answer questions based on web documents and PDF files using RAG (Retrieval-Augmented Generation).

The solution is fully containerized using **Docker**, allowing for seamless hybrid deployment: a lightweight version for **Azure App Service** and a full version with local LLMs (LLaMa 2) for local development. The architecture focuses on modularity, efficient information retrieval using **FAISS**, and "Lazy Loading" strategies for resource optimization.

- **Live Demo:** [https://chatbot-hermanosjota.azurewebsites.net](https://chatbot-hermanosjota.azurewebsites.net)

## Technologies Used

- **LangChain**: Framework for integrating Large Language Models (LLMs) and building RAG chains.
- **FastAPI**: High-performance framework for exposing the chatbot's REST API and serving the frontend.
- **Ollama**: Local server for hosting the LLaMa 2 model (offline privacy-focused capability).
- **Groq & Gemini**: Cloud-based LLM providers for high-speed inference in deployed environments.
- **FAISS**: VectorStore for efficient similarity search and information retrieval.
- **HuggingFace**: Used for generating embeddings locally.
- **Docker**: Containerization of services (API + Ollama) for seamless deployment.
- **GitHub Actions**: CI/CD automation for building and pushing images to Docker Hub and Azure.
- **Azure App Service**: Cloud platform used for the production deployment (Web App for Containers).

## Key Features

- **RAG Architecture**: Uses RAG to provide accurate answers based strictly on the provided corpus.
- **Multi-Model Support**: Users can switch between Ollama (Local), Groq, and Gemini.
- **Seamless Deployment**: Optimized for both resource-constrained cloud environments and powerful local machines.
- **Data Ingestion Pipeline**: Automatically loads and indexes data from PDF documents and Web URLs.
- **Interactive UI**: Includes a web interface to interact with the bot and select models.
- **Design Patterns**: Implements the **Factory Pattern** for scalable model integration (dynamic selection of LLMs and Embeddings) and the **Service Layer Pattern** to strictly decouple business logic from the API endpoints.

## Component Diagram

![Component Diagram](doc/component_diagram.png)

## Folder Structure

The project follows a clean, modular architecture designed for maintainability and scalability:

```text
├── app
│   ├── chat_models          # Factory implementations for LLMs (Groq, Gemini, Ollama)
│   ├── embedding_models     # Factory implementations for Embeddings (HuggingFace, Gemini)
│   ├── config               # Configuration settings and environment variable management
│   ├── loaders              # Strategies for data ingestion (PDF, Web scraping)
│   ├── models               # Pydantic data models for request/response validation
│   ├── services             # Core business logic (RAG orchestration, Vectorization)
│   ├── static               # Frontend assets (HTML, CSS, JS)
│   └── main.py              # Application entry point (FastAPI)
├── corpus                   # Source documents (PDFs) for the Knowledge Base
├── vector_store             # Persistent storage for FAISS vector indices
├── Dockerfile               # Production image definition (Application)
├── Dockerfile.ollama        # Custom image definition for Ollama pre-loaded with LLaMa2
├── docker-compose.yml       # Orchestration for local development
└── requirements.txt         # Python dependencies
```

**Detailed Breakdown**

- `app/chat_models` & `app/embedding_models`: Contains the **Factory** logic to instantiate the correct model provider based on user configuration or environment variables.
- `app/services`: Implements the **Service Layer**. `ChatService` handles the RAG flow (Retrieve + Generate), while `DataService` manages document loading and vector store creation.
- `app/loaders`: Contains the logic to parse different data sources, normalizing them into a standard format for the vector store.
- `corpus`: Directory where raw documents (PDFs) are placed to be ingested by the system.
- `vector_store`: Generated at runtime/build time; stores the FAISS indices to avoid re-calculating embeddings on every restart.

---

# Deployment with Docker on Local Machines

## Prerequisites

- Ensure **Docker** and **Docker Compose** are installed on your machine.
- Verify you have sufficient memory (at least 8GB RAM recommended) if running the local **Ollama** instance.

## Steps to Deploy

- **Clone the Repository**: Clone the project repository to your local machine:

```sh
  git clone [https://github.com/marianoInsa/ChatBot-RAG.git](https://github.com/marianoInsa/ChatBot-RAG.git)
  cd ChatBot-RAG
```

- **Start the Services**: Run the following command to start the containers using the provided `docker-compose.yml` file. This will pull the pre-loaded Ollama image and build the backend:

```sh
docker compose up
```

- **Access the Chatbot**: Once the containers are running (look for "Uvicorn running" in logs), access the web interface at URL: [http://localhost:8000](http://localhost:8000)

## Test and Verify

Open a browser and navigate to the chatbot URL. You can also access the auto-generated documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

Ask sample questions related to the loaded corpus (e.g., "Hermanos Jota" furniture), such as:

1. _¿Qué productos ofrecen en Hermanos Jota? (What products do you offer at Hermanos Jota?)_

<img src="doc/Q1.png" width="600">

2. _¿De qué material están hechos sus productos? (What material are your products made of?)_

<img src="doc/Q2.png" width="600">

3. _¿Cuáles son sus productos disponibles? (What products are available?)_

<img src="doc/Q3.png" width="600">

4. _¿Hacen envíos a Chaco? (Do you ship to Chaco?)_ _(a province in Argentina)_

<img src="doc/Q4.png" width="600">

5. _¿Cómo puedo contactarlos y ubicarlos? (How can I contact you and find you?)_

<img src="doc/Q5.png" width="600">

**Shut Down the Services**: When you're done testing, stop and remove the containers using:

```sh
docker compose down
```

---

# Deployment on Azure Web App for Containers

- **Container image**: The Azure Web App is configured to run the image built by GitHub Actions from this repository.
- **Image name**: `${DOCKERHUB_USERNAME}/chatbot-rag`, where `DOCKERHUB_USERNAME` is the Docker Hub username stored as a GitHub secret.
- **Tags**:
  - `latest`: always points to the most recent successful build from the `main` branch.
  - Short commit SHA tag (e.g. `abc1234`): allows you to know exactly which commit is deployed and to roll back quickly.
- **CI/CD flow**:
  - On every push to `main`, GitHub Actions builds the image from the `Dockerfile`, pushes it to Docker Hub with both tags (`latest` and the short SHA), and then updates the Azure Web App to use the new SHA tag.
  - Azure Web App for Containers pulls the new tagged image and restarts the container, ensuring that each deployment uses a fresh image version.

---

© 2026 | Made with blood, sweat, and tears by [Mariano Insaurralde](https://www.linkedin.com/in/marianoinsa).
