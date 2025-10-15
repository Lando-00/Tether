# Protocol Module

## Architecture Overview

The protocol module follows a modular, layered architecture that implements SOLID design principles:

### Core Layer
- **Interfaces**: Abstract protocol definitions for key components
- **Types**: Common data types and structures
- **Config**: Configuration providers
- **Loggers**: Logging utilities
- **Execution**: Thread and async execution strategies

### Orchestration Layer
- **Detector**: Tool call detection
- **Parsers**: Tool argument parsing
- **Executor**: Tool execution
- **Emitter**: Event emission for streaming
- **Filters**: Text filtering (hidden blocks)
- **Controller**: State management for tool boundaries
- **Adapters**: Integration with model and context components
- **Orchestrator**: High-level orchestration of token streaming and tool execution

### API Layer
- **Schemas**: Pydantic models for API interactions
- **Routers**: FastAPI route handlers organized by domain
- **App Factory**: Assembly of FastAPI application
- **Lifespan**: Application lifecycle management

### Service Layer
- **Protocol Service**: Main service coordinating API interactions
- **Generation Service**: Streaming text generation
- **Stream Adapter**: Integration with streaming components

## Key Components

### ProtocolService
The central service that handles API interactions, including:
- Model and session management
- Tool registration and discovery
- Text generation with and without streaming
- System prompt generation

### GenerationService
Specialized service for streaming generation:
- Integration with ToolOrchestrator
- Asynchronous streaming
- Tool execution during generation
- Generation abortion

### ToolOrchestrator
Core component for managing the generation flow:
- Token streaming from models
- Tool call detection and execution
- Event emission
- State management

  
### System Prompt
  
- The system prompt is now neutral and format-agnostic. Tools are provided separately via the `tools` parameter with `tool_choice="auto"`. The streaming parser detects tool calls in the output, so Python-style function call formatting is not required.

## API Endpoints

The API is organized into domain-specific routers:

- `/models`: Model management
- `/sessions`: Session management
- `/generations`: Text generation with streaming support
- `/tools`: Tool discovery and execution

## Usage

```python
from llm_service.protocol.api.app_factory import create_app
from llm_service.model import ModelComponent
from llm_service.context import ContextComponent

# Create components
model_component = ModelComponent()
context_component = ContextComponent()

# Create FastAPI app with all routes
app = create_app(model_component, context_component)
```

## Backward Compatibility

For backward compatibility with the original monolithic implementation, the module provides compatibility layers:

```python
from llm_service.protocol import ProtocolComponent, create_api_app

# Create using compatibility layer
protocol = ProtocolComponent()
app = create_api_app(protocol)
```
