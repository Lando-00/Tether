# Tether - MLC-LLM Session Service with MCP Architecture

Tether is a FastAPI-based service that provides a session-based API for interacting with machine learning models compiled with [MLC-LLM](https://github.com/mlc-ai/mlc-llm). It persists conversation history in a local SQLite database and is designed to run on devices with specific hardware requirements. The service has been refactored to follow the Model-Context-Protocol (MCP) architecture.

**TODO: Update hardware requirements and target platforms**

## Features

-   **MCP Architecture**: Clean separation of model, context, and protocol components.
-   **Session Management**: Create, list, and delete conversation sessions.
-   **Persistent History**: Chat history is saved and reloaded for each session.
-   **Dynamic Model Loading**: Discovers and loads available MLC-LLM models.
-   **Streaming API**: Provides a token-by-token streaming endpoint with NDJSON format.
-   **Model Caching**: Caches recently used models in memory for faster responses.
-   **Function Calling**: Support for tool use via function calling mechanism.
-   **SOLID Design**: Protocol component follows SOLID principles with modular architecture.
-   **Tool Orchestration**: Sophisticated tool detection, execution, and integration.
-   **Hidden Thoughts**: Support for reflective thinking and hidden blocks in responses.

## Setup

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd Tether
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Add Models

**TODO: Add detailed instructions for compiling models with MLC-LLM**

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

## Architecture Overview

This project implements a Model-Context-Protocol (MCP) architecture for interacting with MLC-LLM models. The MCP architecture separates concerns into three distinct components:

1. **Model**: Handles loading, inference, and management of language models
2. **Context**: Manages conversation history and state persistence
3. **Protocol**: Provides API endpoints and handles client interactions

This separation makes the codebase more maintainable, testable, and extensible. Each component can be developed, tested, and replaced independently.

## Project Structure

```
llm_service/
├── model/                 # Model component - handles LLM operations
│   ├── __init__.py       # Exports ModelComponent
│   └── mlc_engine.py     # Core model operations (loading, inference, cache)
├── context/               # Context component - manages conversation state
│   ├── __init__.py       # Exports ContextComponent
│   └── session_store.py  # Database operations for sessions and messages
├── protocol/              # Protocol component - handles API endpoints
│   ├── __init__.py       # Exports for backward compatibility
│   ├── compatibility.py  # Compatibility layer for legacy code
│   ├── core/             # Core interfaces and implementations
│   │   ├── interfaces.py # Protocol interfaces and types
│   │   ├── types.py      # Data classes and type definitions
│   │   ├── config.py     # Configuration providers
│   │   ├── loggers.py    # Logging implementations
│   │   └── execution.py  # Execution strategy implementations
│   ├── orchestration/    # Tool orchestration components
│   │   ├── detector.py   # Tool call detection
│   │   ├── parsers.py    # Argument parsing
│   │   ├── executor.py   # Tool execution
│   │   ├── emitter.py    # Event emission
│   │   ├── controller.py # Tool boundary control
│   │   ├── orchestrator.py # Main orchestration logic
│   │   ├── filters.py    # Token filtering
│   │   └── adapters/     # Adapters for external components
│   ├── service/          # Service implementations
│   │   ├── protocol_service.py # Main protocol service
│   │   └── generation_service.py # Streaming generation service
│   └── api/              # API layer
│       ├── schemas.py    # API request/response models
│       ├── app_factory.py # FastAPI application factory
│       └── routers/      # API endpoint routers
├── tools/                 # Tools for function calling (re-exports from root)
│   ├── __init__.py       # Tool registration system
│   └── time_tools.py     # Example tool implementations
└── app.py                # Main application entry point
```

## Component Details

### Model Component (`model/`)

The Model component encapsulates all language model operations:

- **Model Loading**: Loads MLC-LLM models and their dependencies
- **DLL Resolution**: Finds the appropriate library files for models
- **Inference**: Handles text generation with various parameters
- **Engine Cache**: Manages model instances efficiently with thread safety

Main Classes:
- `ModelComponent`: Core class handling all model operations

### Context Component (`context/`)

The Context component manages conversation state and persistence:

- **Database Models**: Defines SQLite schemas for sessions and messages
- **CRUD Operations**: Creates, reads, updates, and deletes sessions
- **History Management**: Formats conversation history for model input
- **State Persistence**: Ensures conversations persist across restarts

Main Classes:
- `ContextComponent`: Manages conversation state and database operations
- `SessionDB`: SQLAlchemy model for conversation sessions
- `MessageDB`: SQLAlchemy model for individual messages

### Protocol Component (`protocol/`)

The Protocol component provides the API interface, following SOLID principles with a modular architecture:

#### Core (`protocol/core/`)
- **Interfaces**: Defines protocols for all components
- **Types**: Common data structures and type definitions
- **Configuration**: Environment-based configuration providers
- **Logging**: Standardized logging implementations
- **Execution**: Thread pool and execution strategies

#### Orchestration (`protocol/orchestration/`)
- **Tool Call Detection**: Identifies and extracts tool calls from text
- **Argument Parsing**: Parses arguments from raw tool call text
- **Tool Execution**: Executes tools with proper error handling
- **Event Emission**: Formats and emits events for streaming
- **Filtering**: Processes hidden blocks and reflective thinking
- **Controllers**: Manages tool call boundaries and history
- **Adapters**: Bridges between components and external systems

#### Services (`protocol/service/`)
- **Protocol Service**: Coordinates all protocol operations
- **Generation Service**: Handles streaming text generation with tools

#### API (`protocol/api/`)
- **Schemas**: Request and response Pydantic models
- **Routers**: FastAPI route handlers organized by domain
- **App Factory**: Creates and configures the FastAPI application

Main Classes:
- `ProtocolService`: Main service coordinating model and context operations
- `GenerationService`: Handles streaming generation with tool execution
- `ToolOrchestrator`: Orchestrates token streaming, tool detection, and execution

## Running the Service

To start the API server, run the following command:

```sh
python -m llm_service.app
```

The service will be available at `http://127.0.0.1:8090`. You can view the interactive API documentation at `http://127.0.0.1:8090/docs`.

## Configuration

The service can be configured through environment variables:

- `MLC_DIST_PATH`: Path to the models directory (default: "dist")
- `MLC_SQLITE_URL`: Database connection URL (default: "sqlite:///mlc_sessions.db")
- `MLC_HOST`: Host to run the server on (default: "127.0.0.1")
- `MLC_PORT`: Port to run the server on (default: "8090")

## TODO Items

1. **Testing**: Add unit and integration tests for all components
2. **Performance Optimization**: Review and optimize model loading and inference
3. **CLI Client**: Create a command-line client for interacting with the API
4. **Security**: Add authentication and authorization mechanisms 
5. **Extended Documentation**: Add detailed API documentation
6. **Monitoring**: Add system monitoring and alerting
7. **Remove Legacy Code**: Remove the original monolithic api.py after confirming all functionality works with the new architecture
