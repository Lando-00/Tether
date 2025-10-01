"""
Stream adapter for text generation.
"""
from typing import Dict, Any, AsyncIterator, Optional, List
import asyncio
import json

from llm_service.protocol.service.protocol_service import ProtocolService
from llm_service.protocol.orchestration.orchestrator import ToolOrchestrator


class StreamAdapter:
    """
    Adapter for streaming text generation using the ToolOrchestrator.
    """
    
    def __init__(self, protocol_service: ProtocolService):
        """
        Initialize the stream adapter with a protocol service.
        
        Args:
            protocol_service: ProtocolService instance
        """
        self.protocol = protocol_service
        self.model = protocol_service.model
        self.context = protocol_service.context
        
        # Create a tool orchestrator
        # This is a simplified version - in a real implementation,
        # we would need to create all the necessary components
        self.orchestrator = None
    
    async def generate_stream(
        self,
        session_id: str,
        model_name: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **parameters
    ) -> AsyncIterator[bytes]:
        """
        Generate text using a model with streaming.
        
        Args:
            session_id: Session ID
            model_name: Model name
            prompt: Text prompt
            tools: Optional list of tools
            parameters: Additional parameters for model generation
            
        Returns:
            Async iterator of event stream bytes
        """
        # Add user message to context
        self.context.add_message(session_id, "user", prompt)
        
        # Get conversation history
        history = self.context.get_conversation_history(session_id, format="chat")
        
        # Placeholder for streaming implementation
        # In a real implementation, we would use the ToolOrchestrator
        yield b"data: {\"type\": \"start\"}\n\n"
        
        # Simulate streaming with a simple delay
        for i in range(5):
            await asyncio.sleep(0.2)
            token = f"Token {i} "
            event = {"type": "token", "content": token}
            yield f"data: {json.dumps(event)}\n\n".encode()
        
        # Generate response without streaming and simulate
        response = self.protocol.generate({
            "session_id": session_id,
            "prompt": prompt,
            "model_name": model_name,
            "tools": tools,
            **parameters
        })
        
        yield b"data: {\"type\": \"done\"}\n\n"