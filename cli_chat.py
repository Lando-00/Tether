from rich.prompt import Prompt, IntPrompt
from rich.console import Console
import requests
import json
import typer

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8090"
# The endpoint for streaming chat interaction (NDJSON events)
GENERATE_ENDPOINT = "/generations"

from typing import Optional
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import FormattedText

# --- Rich Console Initialization ---
console = Console()

def get_available_models(api_base_url: str) -> list:
    """Fetches the list of available models from the service."""
    try:
        response = requests.get(f"{api_base_url}/models")
        response.raise_for_status()
        # The response has a 'models' key containing the list
        return response.json().get("models", [])
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the service at {api_base_url}.")
        console.print(f"Please ensure the MLC service is running: [bold]python -m llm_service.app[/bold]")
        console.print(f"Details: {e}")
        raise typer.Exit(1)

def get_sessions(api_base_url: str) -> list:
    """Fetches the list of active sessions."""
    try:
        response = requests.get(f"{api_base_url}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        # Fail silently if the server isn't up, the main model check will catch it.
        return []

def delete_all_sessions(api_base_url: str):
    """Deletes all sessions on the server."""
    try:
        response = requests.delete(f"{api_base_url}/sessions")
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'All sessions deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting all sessions:[/bold red] {e}")

def unload_all_models(api_base_url: str):
    """Calls the endpoint to unload all models from the cache."""
    console.print("Attempting to unload all models from memory...")
    # The endpoint requires a model name, even if it clears all.
    # We'll use the first available model to make the request.
    available_models = get_available_models(api_base_url)
    if not available_models:
        console.print("[yellow]Warning:[/yellow] No available models found to specify for unload request. The cache might be empty already.")
        return

    model_to_specify = available_models[0]["model_name"]

    try:
        response = requests.post(
            f"{api_base_url}/models/unload",
            json={"model_name": model_to_specify, "device": "auto"} # device is optional
        )
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Unload command sent successfully.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error unloading models:[/bold red] {e}")

def delete_session(api_base_url: str, session_id: str):
    """Deletes a specific session."""
    try:
        response = requests.delete(f"{api_base_url}/sessions/{session_id}")
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Session deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting session {session_id}:[/bold red] {e}")

def get_session_history(api_base_url: str, session_id: str) -> list:
    """Fetches the message history for a given session."""
    try:
        response = requests.get(f"{api_base_url}/sessions/{session_id}/messages")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        console.print(f"[bold red]Error fetching history for session {session_id}:[/bold red] {e}")
        return []

def display_history(messages: list):
    """Renders the chat history."""
    if not messages:
        return
    console.rule("Chat History")
    for msg in messages:
        color = "cyan" if msg.get("role") == "user" else "green"
        console.print(f"[bold {color}]{msg.get('role', 'unknown').capitalize()}[/bold {color}]: {msg.get('content', '')}")
    console.rule()

def select_model(model_name: Optional[str]) -> str:
    """Guides the user to select a model if one isn't provided."""
    if model_name:
        # Find the full model object if only a name is passed
        models = get_available_models(API_BASE_URL)
        for m in models:
            if model_name in m.get("model_name", ""):
                return m.get("model_name") # Return the full name
        console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
        raise typer.Exit(1)

    console.print("🔍 Searching for available models...")
    available_models = get_available_models(API_BASE_URL)
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found in the 'dist' directory.")
        console.print("Please make sure your compiled models are correctly placed and the service is running.")
        raise typer.Exit(1)

    model_name_choices = [m["model_name"] for m in available_models]

    console.print("\nAvailable Models:")
    for i, name in enumerate(model_name_choices):
        console.print(f"  [bold cyan][{i+1}][/bold cyan] {name}")

    while True:
        try:
            prompt_text = "\nPlease enter the number of the model you want to use [default=1]: "
            choice_str = console.input(prompt_text)
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(model_name_choices):
                return model_name_choices[choice - 1]
            else:
                console.print(f"[red]Invalid choice. Please enter a number between 1 and {len(model_name_choices)}.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")

def manage_sessions() -> tuple[Optional[str], str]:
    """Display and manage chat sessions. Returns (session_id, action)."""
    sessions = get_sessions(API_BASE_URL)
    choices = {"c": "Create a new chat"}
    console.rule("Chat Session Management")

    if sessions:
        console.print("\nExisting Sessions:")
        for i, s in enumerate(sessions):
            session_id = s.get('session_id', 'N/A')
            created = s.get('created_at', 'N/A')
            console.print(f"  [bold cyan][{i+1}][/bold cyan] Resume session from {created} ([yellow]{session_id[:8]}...[/yellow])")
            choices[str(i+1)] = f"Resume session {session_id}"
        choices["d"] = "Delete a session"
        choices["da"] = "Delete ALL sessions"
    choices["u"] = "Unload all models from memory"

    prompt_text = "\nChoose an action: (c)reate new"
    if sessions:
        prompt_text += ", (1-N) resume, (d)elete, (da) delete all"
    prompt_text += ", (u)nload models"

    action = Prompt.ask(prompt_text, choices=list(choices.keys()), default="c").lower()

    if action == "c":
        return None, "new"
    elif action == "u":
        unload_all_models(API_BASE_URL)
        return None, "manage"
    elif action == "da":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "quit"
        confirm = Prompt.ask(
            "[bold yellow]Are you sure you want to delete all sessions? (y/n)[/bold yellow]",
            choices=["y", "n"],
            default="n"
        )
        if confirm.lower() == 'y':
            delete_all_sessions(API_BASE_URL)
        return None, "manage"
    elif action == "d":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "quit"
        del_choice = IntPrompt.ask(
            "Enter the number of the session to DELETE",
            choices=[str(i + 1) for i in range(len(sessions))],
            show_choices=False,
        )
        session_to_delete = sessions[del_choice - 1]['session_id']
        delete_session(API_BASE_URL, session_to_delete)
        return None, "manage" # Go back to the management screen
    elif action.isdigit() and sessions and 0 < int(action) <= len(sessions):
        session_id = sessions[int(action) - 1]['session_id']
        return session_id, "resume"
    else:
        console.print("[red]Invalid choice.[/red]")
        return None, "quit"

def main(
    model_name_arg: Optional[str] = typer.Option(
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
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode to show detailed event information.",
    ),
    show_thinking: bool = typer.Option(
        True,
        "--show-thinking",
        help="Enable to show the model's thinking process.",
    ),
):
    """
    A CLI for chatting with the MLC-LLM Session Service.
    """
    # --- 0. Welcome and Model Selection ---
    console.print("[bold blue]Welcome to the MLC Chat CLI![/bold blue]")
    model_name = select_model(model_name_arg)

    # --- 1. Session Management Loop ---
    session_id = None
    while session_id is None:
        selected_session_id, action = manage_sessions()

        if action == "new":
            try:
                response = requests.post(f"{API_BASE_URL}/sessions")
                response.raise_for_status()
                session_id = response.json()["session_id"]
                console.print(f"✅ New session created: [yellow]{session_id}[/yellow]")
            except requests.RequestException as e:
                console.print(f"[bold red]Error:[/bold red] Could not create session: {e}")
                raise typer.Exit(1)
        elif action == "resume":
            if selected_session_id:
                session_id = selected_session_id
                console.print(f"✅ Resuming session: [yellow]{session_id}[/yellow]")
                history = get_session_history(API_BASE_URL, session_id)
                display_history(history)
            else:
                # This case should ideally not be reached if logic is correct
                console.print("[red]Error: Tried to resume a session without an ID.[/red]")
                continue
        elif action == "manage":
            continue # Loop back to the management screen
        elif action == "quit":
            raise typer.Exit()

    console.print(f"🤖 Starting chat with [bold green]{model_name}[/bold green] on device [bold]{device}[/bold]...")
    debug_status = "[bold green]enabled[/bold green]" if debug else "[dim]disabled[/dim]"
    console.print(f"Debug mode: {debug_status}")
    thinking_status = "[bold green]enabled[/bold green]" if show_thinking else "[dim]disabled[/dim]"
    console.print(f"Show thinking: {thinking_status}")
    console.print("Type '\\exit' or '\\quit' to end. Type '\\menu' for session management. Type '\\thinking' to toggle thinking output.")
    console.rule()

    # --- 2. Main chat loop ---
    while True:
        # Use prompt_toolkit for multi-line input with custom keybindings
        prompt_message = [
            ('bold cyan', 'You '),
            ('', '(Alt+Enter to send)\n')
        ]
        user_prompt = ptk_prompt(FormattedText(prompt_message), multiline=True)

        stripped_prompt = user_prompt.strip().lower()

        if stripped_prompt in ["\\exit", "\\quit"]:
            console.print("👋 Goodbye!")
            break
        if stripped_prompt == "\\menu":
            main(model_name_arg=model_name, device=device, debug=debug, show_thinking=show_thinking) # Restart the main function to show the menu
            break
        if stripped_prompt == "\\thinking":
            show_thinking = not show_thinking
            thinking_status = "[bold green]enabled[/bold green]" if show_thinking else "[dim]disabled[/dim]"
            console.print(f"Show thinking is now {thinking_status}.")
            console.rule()
            continue # Go to next prompt

        # --- 3. Call the streaming generate endpoint and process events ---
        try:
            response = requests.post(
                f"{API_BASE_URL}{GENERATE_ENDPOINT}",
                json={
                    "session_id": session_id,
                    "prompt": user_prompt,
                    "model_name": model_name,
                    "device": device,
                    "stream": True,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.95
                },
                stream=True,
            )
            response.raise_for_status()
            console.print()  # space before streaming output
            thinking_started = False  # reset flag for single ‘Thought:’ prefix
            # Process NDJSON event stream
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    console.print(f"[red]Error parsing JSON: {line.decode('utf-8', errors='replace')}[/red]")
                    continue
                
                evt_type = event.get("type")
                if evt_type == "tool_start":
                    console.print(f"[bold yellow]Running tool {event.get('name')[7:]}...[/bold yellow]")
                    # Debug output if debug mode is enabled
                    if debug:
                        console.print(f"[dim]Tool start event: {event}[/dim]")
                elif evt_type == "tool_end":

                    result = event.get('result')
                    name = event.get('name')
                    # Check if the result is an error message
                    if isinstance(result, dict) and 'error' in result:
                        console.print(f"[bold red]Tool {name} error:[/bold red] {result['error']}")
                    else:
                        # Ensure we have a string to slice
                        output_str = result if isinstance(result, str) else json.dumps(result)
                        console.print(f"[dim yellow]Tool {name[7:]} output:[/dim yellow] {output_str[:50]}")
                    
                    # Debug output if debug mode is enabled
                    if debug:
                        console.print(f"[dim]Tool end event: {event}[/dim]")
                elif evt_type == "token" or evt_type == "text":
                    console.print(event.get("content"), end="")
                elif evt_type == "hidden_thought" or evt_type == "think_stream":
                    # Only show hidden thoughts if enabled
                    if debug or show_thinking:
                        content = event.get("content")
                        # print prefix only once, then inline tokens
                        if not thinking_started:
                            console.print(f"[dim]Thought: {content}[/dim]", end="")
                            thinking_started = True
                        else:
                            console.print(f"[dim]{content}[/dim]", end="")
                elif evt_type == "error":
                    console.print(f"[bold red]Error:[/bold red] {event.get('error')}")
                elif evt_type == "done":
                    if debug:
                        console.print("[dim][Generation complete][/dim]", end="")
                elif evt_type == "cancelled":
                    console.print("[bold red]Generation cancelled.[/bold red]")
                else:
                    # Handle unknown event types - useful for debugging
                    if debug:
                        console.print(f"[dim][Unknown event: {evt_type}] {event}[/dim]")
            console.print()  # newline after complete stream
        except requests.RequestException as e:
            console.print(f"\n[bold red]Error:[/bold red] Could not get response from server. {e}")
            continue
        except Exception as e:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
            continue
        finally:
            if debug:
                console.print("[dim]Debug: API stream completed[/dim]")
            console.rule()


if __name__ == "__main__":
    typer.run(main)
