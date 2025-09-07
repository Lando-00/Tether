# Adreano - MLC-LLM Session Service

Adreano is a FastAPI-based service that provides a session-based API for interacting with machine learning models compiled with [MLC-LLM](https://github.com/mlc-ai/mlc-llm). It persists conversation history in a local SQLite database and is designed to run on devices like those powered by the Snapdragon X Elite.

## Features

-   **Session Management**: Create, list, and delete conversation sessions.
-   **Persistent History**: Chat history is saved and reloaded for each session.
-   **Dynamic Model Loading**: Discovers and loads available MLC-LLM models.
-   **Streaming API**: Provides a token-by-token streaming endpoint.
-   **Model Caching**: Caches recently used models in memory for faster responses.

## Setup

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd Adreano
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Add Models

**(This is where you will add instructions for compiling models)**

Create a `dist` folder in the root of the project. Your compiled models should be placed inside this folder. The service will automatically discover them.

The expected structure is:

```
.
├── dist/
│   ├── libs/
│   │   └── Qwen2.5-7B-q4f16_0-MLC.dll
│   └── Qwen2.5-7B-q4f16_0-MLC/
│       ├── mlc-chat-config.json
│       ├── ndarray-cache.json
│       └── ... (model weights)
└── ... (project files)
```

## Running the Service

To start the API server, run the following command:

```sh
python llm_service/mlc_service_advanced.py
```

The service will be available at `http://127.0.0.1:8090`. You can view the interactive API documentation at `http://127.0.0.1:8090/docs`.
