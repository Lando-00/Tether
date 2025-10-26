"""
A modern CLI for interacting with the Tether service.
"""
import json
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import FormattedText
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8080/api/v1"


# --- Rich Console Initialization ---
console = Console()
app = typer.Typer(
    name="tether-cli",
    help="A modern CLI for interacting with the Tether service.",
    add_completion=False,
)


# --- API Interaction Functions ---

def get_available_models() -> list:
    """Fetches the list of available models from the service."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        # The new API returns a list of strings directly
        return response.json()
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the service at {API_BASE_URL}.")
        console.print(f"Please ensure the Tether service is running: [bold]python -m tether_service.app[/bold]")
        console.print(f"Details: {e}")
        raise typer.Exit(1)

def get_sessions() -> list:
    """Fetches the list of active sessions."""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []

def create_session() -> Optional[str]:
    """Creates a new session and returns its ID."""
    try:
        response = requests.post(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        session_id = response.json().get("session_id")
        console.print(f"✅ New session created: [yellow]{session_id}[/yellow]")
        return session_id
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not create session: {e}")
        return None

def delete_session(session_id: str):
    """Deletes a specific session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions/{session_id}")
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Session deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting session {session_id}:[/bold red] {e}")

def delete_all_sessions():
    """Deletes all sessions on the server."""
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'All sessions deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting all sessions:[/bold red] {e}")

def get_session_history(session_id: str) -> list:
    """Fetches the message history for a given session."""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/messages")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        console.print(f"[bold red]Error fetching history for session {session_id}:[/bold red] {e}")
        return []

def unload_all_models():
    """Calls the endpoint to unload all models from the cache."""
    console.print("Attempting to unload all models from memory...")
    available_models = get_available_models()
    if not available_models:
        console.print("[yellow]Warning:[/yellow] No available models found to specify for unload request. The cache might be empty already.")
        return

    # The new API unloads all models, but still requires a model name in the path
    model_to_specify = available_models[0]

    try:
        response = requests.post(f"{API_BASE_URL}/models/{model_to_specify}/unload")
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Unload command sent successfully.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error unloading models:[/bold red] {e}")


def display_history(messages: list):
    """Renders the chat history using a more structured format."""
    if not messages:
        return
    
    console.print(Panel("Chat History", style="bold blue", expand=False))
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            panel_content = Text(content, style="cyan")
            console.print(Panel(panel_content, title="You", title_align="left", border_style="cyan"))
        elif role == "assistant":
            # Assistant messages can be complex (text or tool calls)
            if isinstance(content, list):  # Tool call
                for part in content:
                    if part.get("type") == "tool_use":
                        tool_name = part.get("tool_name")
                        tool_args = part.get("tool_args")
                        panel_content = Text(f"Tool: {tool_name}\nArgs: {tool_args}", style="green")
                        console.print(Panel(panel_content, title="Assistant (Tool Call)", title_align="left", border_style="green"))
            else:
                panel_content = Text(content, style="green")
                console.print(Panel(panel_content, title="Assistant", title_align="left", border_style="green"))
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown_tool")
            tool_content = msg.get("content", "")
            panel_content = Text(str(tool_content), style="yellow")
            console.print(Panel(panel_content, title=f"Tool Output ({tool_name})", title_align="left", border_style="yellow"))
    console.print()


def manage_sessions() -> tuple[Optional[str], str]:
    """Display and manage chat sessions. Returns (session_id, action)."""
    sessions = get_sessions()
    
    table = Table(title="Chat Session Management", border_style="blue", show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Action")

    table.add_row("c", "Create a new chat")

    choices = {"c": "Create a new chat"}
    if sessions:
        table.add_section()
        for i, s in enumerate(sessions):
            session_id = s.get('session_id', 'N/A')
            created_at = s.get('created_at', 'N/A')
            table.add_row(str(i+1), f"Resume session from {created_at} ([yellow]{session_id[:8]}...[/yellow])")
            choices[str(i+1)] = f"Resume session {session_id}"
        table.add_section()
        table.add_row("d", "Delete a session")
        table.add_row("da", "Delete ALL sessions")
        choices["d"] = "Delete a session"
        choices["da"] = "Delete ALL sessions"
    
    table.add_section()
    table.add_row("u", "Unload all models from memory")
    table.add_row("q", "Quit")
    choices["u"] = "Unload all models from memory"
    choices["q"] = "Quit"

    console.print(table)

    prompt_text = "\nChoose an action"
    action = Prompt.ask(prompt_text, choices=list(choices.keys()), default="c").lower()

    if action == "c":
        return None, "new"
    elif action == "u":
        unload_all_models()
        return None, "manage"
    elif action == "da":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "manage"
        confirm = Prompt.ask(
            "[bold yellow]Are you sure you want to delete all sessions? (y/n)[/bold yellow]",
            choices=["y", "n"],
            default="n"
        )
        if confirm.lower() == 'y':
            delete_all_sessions()
        return None, "manage"
    elif action == "d":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "manage"
        del_choice = IntPrompt.ask(
            "Enter the number of the session to DELETE",
            choices=[str(i + 1) for i in range(len(sessions))],
            show_choices=False,
        )
        session_to_delete = sessions[del_choice - 1]['session_id']
        delete_session(session_to_delete)
        return None, "manage"
    elif action.isdigit() and sessions and 0 < int(action) <= len(sessions):
        session_id = sessions[int(action) - 1]['session_id']
        return session_id, "resume"
    elif action == "q":
        return None, "quit"
    else:
        console.print("[red]Invalid choice.[/red]")
        return None, "manage"


def select_model(model_name: Optional[str]) -> str:
    """Guides the user to select a model if one isn't provided."""
    if model_name:
        # The new API just returns a list of strings, so we just check for existence
        models = get_available_models()
        if model_name in models:
            return model_name
        console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
        raise typer.Exit(1)

    console.print("🔍 Searching for available models...")
    available_models = get_available_models()
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found.")
        console.print("Please make sure your compiled models are correctly placed and the service is running.")
        raise typer.Exit(1)

    console.print("\nAvailable Models:")
    for i, name in enumerate(available_models):
        console.print(f"  [bold cyan][{i+1}][/bold cyan] {name}")

    while True:
        try:
            prompt_text = "\nPlease enter the number of the model you want to use [default=1]: "
            choice_str = console.input(prompt_text)
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            else:
                console.print(f"[red]Invalid choice. Please enter a number between 1 and {len(available_models)}.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


@app.command()
def main(
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The name of the model to use. If not provided, a list will be shown.",
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
    Main entry point for the Tether CLI.
    """
    console.print(Panel.fit(
        "[bold blue]Welcome to the Tether CLI![/bold blue]\n"
        "Your modern interface for interacting with language models.",
        style="bold blue"
    ))
    
    model_name_arg = model_name
    model_name = select_model(model_name_arg)

    # --- Session Management Loop ---
    session_id = None
    while session_id is None:
        selected_session_id, action = manage_sessions()

        if action == "new":
            session_id = create_session()
            if not session_id:
                # Creation failed, loop back to menu
                continue
        elif action == "resume":
            if selected_session_id:
                session_id = selected_session_id
                console.print(f"✅ Resuming session: [yellow]{session_id}[/yellow]")
                history = get_session_history(session_id)
                display_history(history)
            else:
                console.print("[red]Error: Tried to resume a session without an ID.[/red]")
                continue
        elif action == "manage":
            continue # Loop back to the management screen
        elif action == "quit":
            raise typer.Exit()

    console.print(f"🤖 Starting chat with [bold green]{model_name}[/bold green]...")
    
    info_table = Table.grid(padding=1, expand=True)
    info_table.add_column()
    info_table.add_column(justify="right")
    info_table.add_row(
        f"Debug mode: {'[bold green]enabled[/bold green]' if debug else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\menu[/bold cyan] for session management"
    )
    info_table.add_row(
        f"Show thinking: {'[bold green]enabled[/bold green]' if show_thinking else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\thinking[/bold cyan] to toggle thinking"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\exit[/bold cyan] or [bold cyan]\\quit[/bold cyan] to end"
    )
    console.print(Panel(info_table, title="Chat Info", border_style="dim"))


    # --- Main chat loop ---
    while True:
        try:
            prompt_message = [
                ('bold cyan', 'You '),
                ('', '(Alt+Enter for newline)\n')
            ]
            user_prompt = ptk_prompt(FormattedText(prompt_message), multiline=True)

            stripped_prompt = user_prompt.strip().lower()
            if stripped_prompt in ["\\exit", "\\quit"]:
                console.print("👋 Goodbye!")
                break
            if stripped_prompt == "\\menu":
                # We need to pass the original arguments to main to restart it correctly
                main(model_name=model_name, debug=debug, show_thinking=show_thinking)
                break # Exit current chat loop to prevent it from continuing after menu
            if stripped_prompt == "\\thinking":
                show_thinking = not show_thinking
                thinking_status = "[bold green]enabled[/bold green]" if show_thinking else "[dim]disabled[/dim]"
                console.print(f"Show thinking is now {thinking_status}.")
                console.rule()
                continue # Go to next prompt

            # --- Call the streaming generate endpoint and process events ---
            with requests.post(
                f"{API_BASE_URL}/chat/stream",
                json={
                    "session_id": session_id,
                    "prompt": user_prompt,
                    "model_name": model_name,
                },
                stream=True,
            ) as response:
                response.raise_for_status()
                
                assistant_response = ""
                thinking_response = ""
                text_started = False
                thinking_panel_active = False
                spinner_active = True
                
                # Start spinner while waiting for first response
                with Live(Spinner("dots", text="[dim]Waiting for response...[/dim]"), console=console, refresh_per_second=10) as live:
                    for line in response.iter_lines():
                        # Stop spinner on first event
                        if spinner_active:
                            live.stop()
                            spinner_active = False
                        
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            if debug:
                                console.print(f"[red]Error parsing JSON: {line.decode('utf-8', errors='replace')}[/red]")
                            continue
                        
                        evt_type = event.get("type")
                        evt_data = event.get("data", {})

                        if debug:
                            console.print(f"[dim]Received event: {event}[/dim]")

                        if evt_type == "text":
                            delta = evt_data.get("delta", "")
                            assistant_response += delta
                            
                            # Print header on first text token
                            if not text_started:
                                console.print("\n[bold green]Assistant:[/bold green]")
                                text_started = True
                            
                            # Stream token immediately
                            console.print(delta, end="", style="green")
                            
                        elif evt_type == "think":
                            if show_thinking:
                                delta = evt_data.get("delta", "")
                                thinking_response += delta
                                
                                # Print thinking header on first thinking token
                                if not thinking_panel_active:
                                    console.print("\n[dim italic]💭 Thinking:[/dim italic]")
                                    thinking_panel_active = True
                                
                                # Stream thinking token immediately
                                console.print(delta, end="", style="dim italic")
                                
                        elif evt_type == "tool_started":
                            # Ensure we're on a new line before printing tool panels
                            if text_started or thinking_panel_active:
                                console.print()  # New line
                            tool_name = evt_data.get('tool_name')
                            console.print(Panel(f"Calling tool: [bold yellow]{tool_name}[/bold yellow]", expand=False, border_style="yellow"))
                            
                        elif evt_type == "tool_completed":
                            tool_name = evt_data.get('tool_name')
                            tool_result = evt_data.get('tool_result')
                            output_str = str(tool_result)
                            console.print(Panel(f"Tool [bold yellow]{tool_name}[/bold yellow] output: {output_str[:150]}...", title="Tool Output", expand=False, border_style="dim yellow"))
                            
                        elif evt_type == "tool_error":
                            if text_started or thinking_panel_active:
                                console.print()  # New line
                            tool_name = evt_data.get('tool_name')
                            error = evt_data.get('error')
                            console.print(Panel(f"Tool [bold red]{tool_name}[/bold red] error: {error}", title="Tool Error", border_style="red"))
                            
                        elif evt_type == "error":
                            if text_started or thinking_panel_active:
                                console.print()  # New line
                            console.print(Panel(f"API Error: {evt_data.get('message')}", title="Error", border_style="bold red"))
                            
                        elif evt_type == "done":
                            # Ensure we end on a new line
                            if text_started or thinking_panel_active:
                                console.print()
                            if debug:
                                console.print("[dim][Generation complete][/dim]")

        except requests.RequestException as e:
            console.print(f"\n[bold red]Error:[/bold red] Could not get response from server. {e}")
            continue
        except Exception as e:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
            continue
        finally:
            console.rule()


if __name__ == "__main__":
    app()
