"""
tool_chat_example.py - Example showing how to use tools in a chat context with MLC-LLM

This script demonstrates how to use tool calling functionality in a chat context
with the MLC-LLM Session Service. It shows how to:

1. Fetch available tools
2. Enable tools in chat requests
3. Process and display tool calls and responses
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
import json
import typer
import requests

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8090"
console = Console()

def get_available_models(api_base_url: str) -> list:
    """Fetches the list of available models from the service."""
    try:
        response = requests.get(f"{api_base_url}/models")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the service at {api_base_url}.")
        console.print(f"Please ensure the MLC service is running.")
        console.print(f"Details: {e}")
        raise typer.Exit(1)

def get_available_tools(api_base_url: str) -> list:
    """Fetches the list of available tools from the service."""
    try:
        response = requests.get(f"{api_base_url}/tools")
        response.raise_for_status()
        return response.json().get("tools", [])
    except requests.RequestException as e:
        console.print(f"[yellow]Warning:[/yellow] Could not fetch tools: {e}")
        return []

def create_session(api_base_url: str) -> str:
    """Creates a new session and returns its ID."""
    try:
        response = requests.post(f"{api_base_url}/sessions")
        response.raise_for_status()
        return response.json()["session_id"]
    except requests.RequestException as e:
        console.print(f"[bold red]Error creating session:[/bold red] {e}")
        raise typer.Exit(1)

def display_tool_definition(tool):
    """Display a tool definition in a readable format."""
    if "function" in tool:
        func = tool["function"]
        console.print(f"[bold cyan]Function:[/bold cyan] {func.get('name', 'unnamed')}")
        console.print(f"[cyan]Description:[/cyan] {func.get('description', 'No description')}")
        
        if "parameters" in func:
            params = func["parameters"]
            console.print("[cyan]Parameters:[/cyan]")
            
            if "properties" in params:
                for param_name, param_details in params["properties"].items():
                    required = param_name in params.get("required", [])
                    req_str = "[bold red](required)[/bold red]" if required else "[gray](optional)[/gray]"
                    console.print(f"  • {param_name} {req_str}: {param_details.get('description', 'No description')}")
            
            console.print()

def format_message_content(content):
    """Format message content, handling both string and object formats."""
    if isinstance(content, str):
        try:
            # Try to parse as JSON if it looks like it might be JSON
            if content.strip().startswith("{") and content.strip().endswith("}"):
                content_obj = json.loads(content)
                return content_obj
        except json.JSONDecodeError:
            pass
        return content
    return content

def run_chat_with_tools(model_name: str):
    """Run a chat session with tool calling enabled."""
    console.print(Panel(f"[bold]Tool-Enabled Chat with {model_name}[/bold]", 
                        border_style="green"))
    
    # Get available tools
    tools = get_available_tools(API_BASE_URL)
    
    if not tools:
        console.print("[yellow]Warning:[/yellow] No tools available. The model won't be able to call any functions.")
    else:
        console.print(f"[green]Found {len(tools)} available tools:[/green]\n")
        for tool in tools:
            display_tool_definition(tool)
    
    # Create a new session
    session_id = create_session(API_BASE_URL)
    console.print(f"[green]Created new chat session:[/green] {session_id}\n")
    
    # Start the chat loop
    console.print("[bold]Start chatting! Type 'exit' to end the session.[/bold]")
    
    while True:
        # Get user input
        prompt = console.input("[bold cyan]You:[/bold cyan] ")
        
        if prompt.lower() in ["exit", "quit", "bye"]:
            console.print("[yellow]Ending chat session[/yellow]")
            break
        
        # Send the request with tools enabled
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "session_id": session_id,
                    "prompt": prompt,
                    "model_name": model_name,
                    "tools": tools,
                    "temperature": 0.5,
                    "max_tokens": 1024
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Get the messages to check for tool calls
            messages_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/messages")
            messages_response.raise_for_status()
            messages = messages_response.json()
            
            # Find assistant messages with tool calls
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            tool_messages = [m for m in messages if m["role"] == "tool"]
            
            # Show the last few messages with tool calls highlighted
            last_messages = assistant_messages[-3:] + tool_messages[-3:]
            last_messages.sort(key=lambda x: x["id"])
            
            for msg in last_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "assistant":
                    formatted_content = format_message_content(content)
                    
                    # Check for tool calls
                    if isinstance(formatted_content, dict) and ("tool_calls" in formatted_content or "function_call" in formatted_content):
                        console.print("[bold green]Assistant:[/bold green] I need to use a tool to help with that.")
                        
                        # Display the tool call
                        if "tool_calls" in formatted_content:
                            for tool_call in formatted_content["tool_calls"]:
                                if "function" in tool_call:
                                    func = tool_call["function"]
                                    console.print(f"[bold yellow]Using tool:[/bold yellow] {func.get('name', 'unnamed')}")
                                    
                                    # Display arguments in pretty JSON
                                    if "arguments" in func:
                                        try:
                                            if isinstance(func["arguments"], str):
                                                args = json.loads(func["arguments"])
                                            else:
                                                args = func["arguments"]
                                                
                                            args_json = json.dumps(args, indent=2)
                                            console.print(Syntax(args_json, "json", theme="monokai", 
                                                            line_numbers=False))
                                        except Exception:
                                            console.print(f"  Arguments: {func['arguments']}")
                        
                        # Alternative format for function_call
                        elif "function_call" in formatted_content:
                            func = formatted_content["function_call"]
                            console.print(f"[bold yellow]Using tool:[/bold yellow] {func.get('name', 'unnamed')}")
                            
                            # Display arguments in pretty JSON
                            if "arguments" in func:
                                try:
                                    args_json = json.dumps(json.loads(func["arguments"]), indent=2)
                                    console.print(Syntax(args_json, "json", theme="monokai", 
                                                   line_numbers=False))
                                except Exception:
                                    console.print(f"  Arguments: {func['arguments']}")
                    else:
                        # Regular assistant message
                        if isinstance(formatted_content, dict):
                            # If we have a dict but no tool calls, just show the original content
                            console.print(f"[bold green]Assistant:[/bold green] {content}")
                        else:
                            console.print(f"[bold green]Assistant:[/bold green] {formatted_content}")
                
                elif role == "tool":
                    # Display tool response
                    try:
                        # Try to parse and pretty-print the tool response
                        tool_content = json.loads(content)
                        if "content" in tool_content:
                            console.print(f"[bold blue]Tool result:[/bold blue] {tool_content['content']}")
                        elif "name" in tool_content and "content" in tool_content:
                            console.print(f"[bold blue]Tool {tool_content['name']} result:[/bold blue] {tool_content['content']}")
                        else:
                            # Show the full content if we can't extract specific fields
                            tool_json = json.dumps(tool_content, indent=2)
                            console.print("[bold blue]Tool result:[/bold blue]")
                            console.print(Syntax(tool_json, "json", theme="monokai", line_numbers=False))
                    except json.JSONDecodeError:
                        # If not valid JSON, just show the raw content
                        console.print(f"[bold blue]Tool result:[/bold blue] {content}")
            
            # Show the final response if not already displayed
            if result.get("reply") and result["reply"] not in [m["content"] for m in last_messages]:
                console.print(f"[bold green]Assistant:[/bold green] {result['reply']}")
            
        except requests.RequestException as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
    
    console.print("\n[green]Chat session ended.[/green]")

def main():
    # Check if the server is available
    try:
        response = requests.get(f"{API_BASE_URL}/healthz")
        response.raise_for_status()
    except requests.RequestException:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the server at {API_BASE_URL}. "
                     "Please make sure it's running.")
        return
    
    # Get available models
    available_models = get_available_models(API_BASE_URL)
    
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found.")
        return
    
    # List available models
    console.print("[green]Available models:[/green]")
    for i, model in enumerate(available_models):
        console.print(f"  {i+1}. {model['model_name']}")
    
    # Select a model
    choice = Prompt.ask(
        "[bold]Choose a model by number[/bold]",
        choices=[str(i+1) for i in range(len(available_models))],
        default="1"
    )
    
    model_name = available_models[int(choice)-1]["model_name"]
    
    # Start chat with tools
    run_chat_with_tools(model_name)

if __name__ == "__main__":
    typer.run(main)
