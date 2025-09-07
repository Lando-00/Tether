from rich.prompt import Prompt, IntPrompt
from rich.console import Console
import requests
import json
import typer

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8090"
# In the future, you might have a separate endpoint for tool-enabled generation
GENERATE_ENDPOINT = "/generate_stream"

from typing import Optional

# --- Rich Console Initialization ---
console = Console()

def get_available_models() -> list[str]:
    """Fetches the list of available models from the service."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the service at {API_BASE_URL}.")
        console.print(f"Please ensure the MLC service is running: [bold]python llm_service/mlc_service_advanced.py[/bold]")
        console.print(f"Details: {e}")
        raise typer.Exit(1)

def main(
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The name of the model to use. If not provided, a list will be shown.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="The device to run the model on (e.g., 'auto', 'vulkan', 'opencl').",
    ),
):
    """
    A CLI for chatting with the MLC-LLM Session Service.
    """
    if not model_name:
        console.print("🔍 Searching for available models...")
        available_models = get_available_models()
        if not available_models:
            console.print("[bold red]Error:[/bold red] No models found in the 'dist' directory.")
            console.print("Please make sure your compiled models are correctly placed and the service is running.")
            raise typer.Exit(1)

        # Extract just the names for the user to choose from
        model_name_choices = available_models

        console.print("\nAvailable Models:")
        for i, name in enumerate(model_name_choices):
            console.print(f"  [bold cyan][{i+1}][/bold cyan] {name['model_name']}") # type: ignore

        choice = IntPrompt.ask(
            "\nPlease enter the number of the model you want to use",
            choices=[str(i + 1) for i in range(len(model_name_choices))],
            show_choices=False,
            default=1,
        )
        model_name = model_name_choices[choice - 1]

    console.print(f"🤖 Starting chat session with [bold green]{model_name}[/bold green] on device [bold]{device}[/bold]...")

    # --- 1. Create a new session ---
    try:
        response = requests.post(
            f"{API_BASE_URL}/sessions",
            json={"model_name": model_name, "device": device},
        )
        response.raise_for_status()
        session_data = response.json()
        session_id = session_data["session_id"]
        console.print(f"✅ Session created: [yellow]{session_id}[/yellow]")
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not create session.")
        console.print(f"Details: {e}")
        raise typer.Exit(1)

    console.print("Type 'exit' or 'quit' to end the session.")
    console.rule()

    # --- 2. Main chat loop ---
    while True:
        user_prompt = Prompt.ask("[bold cyan]You[/bold cyan]")

        if user_prompt.lower() in ["exit", "quit"]:
            console.print("👋 Goodbye!")
            # Optional: Add logic to delete the session on the server
            # requests.delete(f"{API_BASE_URL}/sessions/{session_id}")
            break

        # --- 3. Call the generate_stream endpoint ---
        full_response_content = ""
        try:
            with requests.post(
                f"{API_BASE_URL}{GENERATE_ENDPOINT}",
                json={"session_id": session_id, "message": user_prompt},
                stream=True,
            ) as r:
                r.raise_for_status()
                console.print("[bold green]AI[/bold green]: ", end="")
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        try:
                            # Each chunk is a JSON object with a 'text' field
                            data = json.loads(chunk.decode('utf-8'))
                            token = data.get("text", "")
                            full_response_content += token
                            console.print(token, end="")
                        except json.JSONDecodeError:
                            # Handle potential malformed JSON chunks if necessary
                            console.print(f"[red](Error decoding chunk: {chunk})[/red]", end="")
                console.print() # Newline after the full response

            # --- 4. Future MCP/Tool-Use Handling ---
            # When your API supports tool calls, the response here might be a
            # structured JSON object instead of just text.
            #
            # Example of what you might do:
            # if response_is_tool_call(full_response_content):
            #     tool_name, tool_args = parse_tool_call(full_response_content)
            #     console.print(f"🤖 Using tool: {tool_name} with args: {tool_args}")
            #     # The MCP server would execute this and feed it back to the model.
            #     # The final text response would then be streamed back.
            # else:
            #     # For now, we just print the markdown of the final text
            #     console.print(Markdown(full_response_content))

        except requests.RequestException as e:
            console.print(f"\n[bold red]Error:[/bold red] Could not get response from server. {e}")
            continue
        finally:
            console.rule()


if __name__ == "__main__":
    typer.run(main)
